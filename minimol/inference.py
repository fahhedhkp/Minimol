"""
Inference Module for Minimol
Handles text generation with model routing and optimization.
"""

import asyncio
import time
from typing import Optional, AsyncGenerator, Dict, Any
from minimol.llm_router import LLMRouter, LLMConfig, UseCase
from minimol.ollama_manager import OllamaManager


class InferenceEngine:
    """Main inference engine for text generation"""
    
    def __init__(
        self,
        config_path: str = "config/llm_providers.yaml",
        use_ollama: bool = True,
    ):
        self.llm_config = LLMConfig.from_yaml(config_path)
        self.router = LLMRouter(self.llm_config)
        self.ollama_manager = OllamaManager() if use_ollama else None
        self.stats = {
            "total_requests": 0,
            "total_tokens": 0,
            "total_latency_ms": 0,
            "total_cost": 0.0,
        }
    
    async def initialize(self):
        """Initialize inference engine"""
        if self.ollama_manager:
            # Check if Ollama is running
            if not await self.ollama_manager.check_health():
                print("⚠️  Ollama not running. Attempting to start...")
                if await self.ollama_manager.start_ollama():
                    print("✅ Ollama started")
                else:
                    print("⚠️  Could not start Ollama, will use remote providers")
    
    async def generate(
        self,
        prompt: str,
        use_case: UseCase = UseCase.GENERAL,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_k: Optional[int] = None,
        top_p: float = 0.9,
    ) -> str:
        """Generate text completion"""
        start_time = time.time()
        
        try:
            result = await self.router.complete(
                prompt,
                use_case=use_case,
                provider=provider,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                top_k=top_k,
                top_p=top_p,
            )
            
            # Update stats
            elapsed_ms = (time.time() - start_time) * 1000
            self.stats["total_requests"] += 1
            self.stats["total_latency_ms"] += elapsed_ms
            
            return result
        
        except Exception as e:
            print(f"Generation failed: {e}")
            raise
    
    async def stream(
        self,
        prompt: str,
        use_case: UseCase = UseCase.GENERAL,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> AsyncGenerator[str, None]:
        """Stream text generation"""
        start_time = time.time()
        
        try:
            async for chunk in self.router.stream(
                prompt,
                use_case=use_case,
                provider=provider,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            ):
                yield chunk
            
            # Update stats
            elapsed_ms = (time.time() - start_time) * 1000
            self.stats["total_requests"] += 1
            self.stats["total_latency_ms"] += elapsed_ms
        
        except Exception as e:
            print(f"Streaming failed: {e}")
            raise
    
    async def batch_generate(
        self,
        prompts: list,
        use_case: UseCase = UseCase.GENERAL,
        parallel: bool = False,
    ) -> list:
        """Generate completions for multiple prompts"""
        if parallel:
            tasks = [
                self.generate(prompt, use_case=use_case)
                for prompt in prompts
            ]
            return await asyncio.gather(*tasks)
        else:
            results = []
            for prompt in prompts:
                result = await self.generate(prompt, use_case=use_case)
                results.append(result)
            return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get inference statistics"""
        avg_latency = (
            self.stats["total_latency_ms"] / self.stats["total_requests"]
            if self.stats["total_requests"] > 0
            else 0
        )
        
        return {
            "total_requests": self.stats["total_requests"],
            "total_tokens": self.stats["total_tokens"],
            "average_latency_ms": avg_latency,
            "total_latency_ms": self.stats["total_latency_ms"],
            "total_cost": self.stats["total_cost"],
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.ollama_manager:
            await self.ollama_manager.cleanup()


async def main():
    """Example usage"""
    print("=" * 80)
    print("Minimol Inference Engine")
    print("=" * 80)
    
    engine = InferenceEngine()
    await engine.initialize()
    
    # Test generation
    print("\n🤖 Testing inference...")
    
    prompt = "Explain quantum computing in 2 sentences."
    print(f"\nPrompt: {prompt}\n")
    
    try:
        # Stream generation
        print("Response: ", end="", flush=True)
        async for chunk in engine.stream(
            prompt,
            use_case=UseCase.REASONING,
            provider="ollama",
        ):
            print(chunk, end="", flush=True)
        print()
    
    except Exception as e:
        print(f"Error: {e}")
    
    # Print stats
    print("\n" + "=" * 80)
    print("Statistics")
    print("=" * 80)
    stats = engine.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    await engine.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
