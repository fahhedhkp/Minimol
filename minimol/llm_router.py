"""
LLM Router Module for Minimol
Manages multi-provider LLM support with intelligent routing,
fallback chains, and cost optimization.
"""

import asyncio
from typing import Optional, List, Dict, Any, AsyncGenerator
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
import os
import json
import yaml
from abc import ABC, abstractmethod
from datetime import datetime
import httpx

# Provider imports (optional)
try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import openai
except ImportError:
    openai = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    import cohere
except ImportError:
    cohere = None


class ProviderType(Enum):
    """Supported LLM providers"""
    OLLAMA = "ollama"
    CLAUDE = "claude"
    OPENAI = "openai"
    GEMINI = "gemini"
    COHERE = "cohere"
    MISTRAL = "mistral"
    AZURE_OPENAI = "azure_openai"
    AWS_BEDROCK = "aws_bedrock"


class UseCase(Enum):
    """Task use cases for provider selection"""
    GENERAL = "general"
    REASONING = "complex_reasoning"
    CODING = "code_generation"
    ANALYSIS = "analysis"
    CREATIVE = "creative_writing"
    FAST = "fast_inference"
    LOCAL = "privacy_sensitive"
    BUDGET = "cost_optimization"


@dataclass
class ProviderConfig:
    """Configuration for a single provider"""
    provider_type: ProviderType
    api_key: Optional[str] = None
    endpoint: Optional[str] = None
    models: List[str] = field(default_factory=list)
    temperature: float = 0.7
    max_tokens: int = 2048
    enabled: bool = True
    use_cases: List[UseCase] = field(default_factory=list)
    cost_per_1m_tokens: float = 0.0  # Cost for 1M tokens
    latency_ms: float = 0.0  # Average latency
    supports_streaming: bool = True
    supports_function_calling: bool = False


@dataclass
class LLMConfig:
    """Complete LLM configuration"""
    providers: Dict[str, ProviderConfig] = field(default_factory=dict)
    default_provider: str = "ollama"
    fallback_chain: List[str] = field(default_factory=lambda: ["ollama", "claude", "openai"])
    auto_select: bool = True
    enable_cost_optimization: bool = True
    budget_per_hour: Optional[float] = None
    
    @classmethod
    def from_yaml(cls, filepath: str) -> "LLMConfig":
        """Load configuration from YAML file"""
        if not Path(filepath).exists():
            return cls()
        
        with open(filepath, "r") as f:
            config_data = yaml.safe_load(f) or {}
        
        config = cls()
        
        # Load providers
        providers_data = config_data.get("providers", {})
        for provider_name, provider_data in providers_data.items():
            provider_type = ProviderType(provider_data.get("type", provider_name))
            
            # Load API key from env or config
            api_key = provider_data.get("api_key")
            if isinstance(api_key, str) and api_key.startswith("${"):
                env_var = api_key.strip("${}")
                api_key = os.getenv(env_var)
            
            provider_config = ProviderConfig(
                provider_type=provider_type,
                api_key=api_key,
                endpoint=provider_data.get("endpoint"),
                models=provider_data.get("models", []),
                temperature=provider_data.get("settings", {}).get("temperature", 0.7),
                max_tokens=provider_data.get("settings", {}).get("max_tokens", 2048),
                enabled=provider_data.get("enabled", True),
                use_cases=[UseCase(uc) for uc in provider_data.get("use_cases", [])],
                cost_per_1m_tokens=provider_data.get("cost_per_1m_tokens", 0.0),
                latency_ms=provider_data.get("latency_ms", 0.0),
            )
            
            config.providers[provider_name] = provider_config
        
        # Load routing config
        routing = config_data.get("routing", {})
        config.default_provider = routing.get("default_provider", "ollama")
        config.fallback_chain = routing.get("fallback_chain", ["ollama", "claude", "openai"])
        config.auto_select = routing.get("auto_select", True)
        
        # Load cost settings
        cost_config = config_data.get("cost_optimization", {})
        config.enable_cost_optimization = cost_config.get("enabled", True)
        config.budget_per_hour = cost_config.get("budget_per_hour")
        
        return config


class BaseProviderAdapter(ABC):
    """Base class for provider adapters"""
    
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.name = config.provider_type.value
        self.request_count = 0
        self.total_tokens = 0
        self.total_cost = 0.0
    
    @abstractmethod
    async def complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate text completion"""
        pass
    
    @abstractmethod
    async def stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream text generation"""
        pass
    
    def get_request_count(self) -> int:
        """Get total requests made"""
        return self.request_count
    
    def get_total_cost(self) -> float:
        """Get total cost incurred"""
        return self.total_cost


class OllamaAdapter(BaseProviderAdapter):
    """Adapter for Ollama local models"""
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.endpoint = config.endpoint or "http://localhost:11434"
        self.client = httpx.AsyncClient(base_url=self.endpoint)
    
    async def complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate completion using Ollama"""
        model = model or self.config.models[0] if self.config.models else "mistral"
        
        try:
            response = await self.client.post(
                "/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": self.config.temperature,
                    **kwargs,
                },
                timeout=300.0,
            )
            
            self.request_count += 1
            result = response.json()
            return result.get("response", "")
        
        except Exception as e:
            raise RuntimeError(f"Ollama request failed: {e}")
    
    async def stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream completion from Ollama"""
        model = model or self.config.models[0] if self.config.models else "mistral"
        
        try:
            async with self.client.stream(
                "POST",
                "/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": True,
                    "temperature": self.config.temperature,
                    **kwargs,
                },
                timeout=300.0,
            ) as response:
                self.request_count += 1
                
                async for line in response.aiter_lines():
                    if line:
                        data = json.loads(line)
                        chunk = data.get("response", "")
                        if chunk:
                            yield chunk
        
        except Exception as e:
            raise RuntimeError(f"Ollama streaming failed: {e}")


class ClaudeAdapter(BaseProviderAdapter):
    """Adapter for Anthropic Claude"""
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        if not anthropic:
            raise ImportError("anthropic package not installed")
        
        self.client = anthropic.AsyncAnthropic(api_key=config.api_key)
    
    async def complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate completion using Claude"""
        model = model or self.config.models[0] if self.config.models else "claude-3-sonnet-20240229"
        
        try:
            message = await self.client.messages.create(
                model=model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
            
            self.request_count += 1
            
            # Track usage
            if hasattr(message, "usage"):
                self.total_tokens += message.usage.input_tokens + message.usage.output_tokens
                self.total_cost += (message.usage.input_tokens + message.usage.output_tokens) * self.config.cost_per_1m_tokens / 1_000_000
            
            return message.content[0].text
        
        except Exception as e:
            raise RuntimeError(f"Claude request failed: {e}")
    
    async def stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream completion from Claude"""
        model = model or self.config.models[0] if self.config.models else "claude-3-sonnet-20240229"
        
        try:
            async with self.client.messages.stream(
                model=model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            ) as stream:
                self.request_count += 1
                
                async for text in stream.text_stream:
                    yield text
        
        except Exception as e:
            raise RuntimeError(f"Claude streaming failed: {e}")


class OpenAIAdapter(BaseProviderAdapter):
    """Adapter for OpenAI GPT models"""
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        if not openai:
            raise ImportError("openai package not installed")
        
        self.client = openai.AsyncOpenAI(api_key=config.api_key)
    
    async def complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate completion using GPT"""
        model = model or self.config.models[0] if self.config.models else "gpt-4"
        
        try:
            response = await self.client.chat.completions.create(
                model=model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
            
            self.request_count += 1
            
            # Track usage
            if hasattr(response, "usage"):
                self.total_tokens += response.usage.total_tokens
                self.total_cost += response.usage.total_tokens * self.config.cost_per_1m_tokens / 1_000_000
            
            return response.choices[0].message.content
        
        except Exception as e:
            raise RuntimeError(f"OpenAI request failed: {e}")
    
    async def stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream completion from GPT"""
        model = model or self.config.models[0] if self.config.models else "gpt-4"
        
        try:
            async with await self.client.chat.completions.create(
                model=model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                **kwargs,
            ) as stream:
                self.request_count += 1
                
                async for chunk in stream:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
        
        except Exception as e:
            raise RuntimeError(f"OpenAI streaming failed: {e}")


class GeminiAdapter(BaseProviderAdapter):
    """Adapter for Google Gemini"""
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        if not genai:
            raise ImportError("google-generativeai package not installed")
        
        genai.configure(api_key=config.api_key)
        self.client = genai.GenerativeModel(
            model_name=config.models[0] if config.models else "gemini-pro"
        )
    
    async def complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate completion using Gemini"""
        try:
            response = await asyncio.to_thread(
                self.client.generate_content,
                prompt,
                **kwargs,
            )
            
            self.request_count += 1
            return response.text
        
        except Exception as e:
            raise RuntimeError(f"Gemini request failed: {e}")
    
    async def stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream completion from Gemini"""
        try:
            response = await asyncio.to_thread(
                self.client.generate_content,
                prompt,
                stream=True,
                **kwargs,
            )
            
            self.request_count += 1
            
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        
        except Exception as e:
            raise RuntimeError(f"Gemini streaming failed: {e}")


class CostOptimizer:
    """Optimizes provider selection based on cost and performance"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.hourly_cost = 0.0
        self.budget_per_hour = config.budget_per_hour
    
    def select_best_provider(
        self,
        use_case: UseCase,
        adapters: Dict[str, BaseProviderAdapter],
    ) -> Optional[str]:
        """Select best provider for use case"""
        if not self.config.enable_cost_optimization:
            return self.config.default_provider
        
        # Filter by use case
        candidates = []
        for name, adapter in adapters.items():
            provider_config = self.config.providers.get(name)
            if provider_config and provider_config.enabled:
                if not provider_config.use_cases or use_case in provider_config.use_cases:
                    candidates.append((name, adapter, provider_config))
        
        if not candidates:
            return self.config.default_provider
        
        # Sort by cost
        if self.config.enable_cost_optimization:
            candidates.sort(key=lambda x: x[2].cost_per_1m_tokens)
        
        return candidates[0][0]
    
    def is_within_budget(self) -> bool:
        """Check if hourly budget is exceeded"""
        if self.budget_per_hour is None:
            return True
        return self.hourly_cost <= self.budget_per_hour
    
    def update_cost(self, amount: float):
        """Update hourly cost"""
        self.hourly_cost += amount


class FallbackChain:
    """Manages fallback logic when provider fails"""
    
    def __init__(self, chain: List[str], adapters: Dict[str, BaseProviderAdapter]):
        self.chain = chain
        self.adapters = adapters
        self.failed_providers = set()
    
    def get_next_provider(self, current: Optional[str] = None) -> Optional[str]:
        """Get next provider in fallback chain"""
        start_idx = 0
        if current and current in self.chain:
            start_idx = self.chain.index(current) + 1
        
        for i in range(start_idx, len(self.chain)):
            provider = self.chain[i]
            if provider not in self.failed_providers and provider in self.adapters:
                return provider
        
        return None
    
    def mark_failed(self, provider: str):
        """Mark provider as failed"""
        self.failed_providers.add(provider)
    
    def reset(self):
        """Reset failed providers"""
        self.failed_providers.clear()


class LLMRouter:
    """Main router for managing multiple LLM providers"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.adapters: Dict[str, BaseProviderAdapter] = {}
        self.cost_optimizer = CostOptimizer(config)
        self.fallback_chain = FallbackChain(config.fallback_chain, self.adapters)
        
        # Initialize adapters
        self._initialize_adapters()
    
    def _initialize_adapters(self):
        """Initialize provider adapters"""
        for provider_name, provider_config in self.config.providers.items():
            if not provider_config.enabled:
                continue
            
            try:
                if provider_config.provider_type == ProviderType.OLLAMA:
                    self.adapters[provider_name] = OllamaAdapter(provider_config)
                elif provider_config.provider_type == ProviderType.CLAUDE:
                    self.adapters[provider_name] = ClaudeAdapter(provider_config)
                elif provider_config.provider_type == ProviderType.OPENAI:
                    self.adapters[provider_name] = OpenAIAdapter(provider_config)
                elif provider_config.provider_type == ProviderType.GEMINI:
                    self.adapters[provider_name] = GeminiAdapter(provider_config)
                
                print(f"✅ Initialized {provider_name}")
            
            except Exception as e:
                print(f"⚠️  Failed to initialize {provider_name}: {e}")
    
    async def complete(
        self,
        prompt: str,
        use_case: UseCase = UseCase.GENERAL,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate completion with automatic routing"""
        
        # Select provider
        if provider is None:
            if self.config.auto_select:
                provider = self.cost_optimizer.select_best_provider(use_case, self.adapters)
            else:
                provider = self.config.default_provider
        
        if provider not in self.adapters:
            raise ValueError(f"Provider {provider} not available")
        
        adapter = self.adapters[provider]
        
        try:
            result = await adapter.complete(prompt, model, **kwargs)
            self.fallback_chain.reset()
            return result
        
        except Exception as e:
            print(f"Provider {provider} failed: {e}")
            self.fallback_chain.mark_failed(provider)
            
            # Try fallback
            next_provider = self.fallback_chain.get_next_provider(provider)
            if next_provider:
                print(f"Falling back to {next_provider}")
                return await self.complete(prompt, use_case, next_provider, model, **kwargs)
            
            raise RuntimeError("All providers exhausted")
    
    async def stream(
        self,
        prompt: str,
        use_case: UseCase = UseCase.GENERAL,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream completion with automatic routing"""
        
        # Select provider
        if provider is None:
            if self.config.auto_select:
                provider = self.cost_optimizer.select_best_provider(use_case, self.adapters)
            else:
                provider = self.config.default_provider
        
        if provider not in self.adapters:
            raise ValueError(f"Provider {provider} not available")
        
        adapter = self.adapters[provider]
        
        try:
            async for chunk in adapter.stream(prompt, model, **kwargs):
                yield chunk
            
            self.fallback_chain.reset()
        
        except Exception as e:
            print(f"Provider {provider} failed: {e}")
            self.fallback_chain.mark_failed(provider)
            
            # Try fallback
            next_provider = self.fallback_chain.get_next_provider(provider)
            if next_provider:
                print(f"Falling back to {next_provider}")
                async for chunk in self.stream(prompt, use_case, next_provider, model, **kwargs):
                    yield chunk
            else:
                raise RuntimeError("All providers exhausted")
    
    def get_provider_stats(self, provider: str) -> Dict[str, Any]:
        """Get statistics for a provider"""
        if provider not in self.adapters:
            return {}
        
        adapter = self.adapters[provider]
        provider_config = self.config.providers.get(provider)
        
        return {
            "name": provider,
            "enabled": provider_config.enabled if provider_config else False,
            "models": provider_config.models if provider_config else [],
            "requests": adapter.get_request_count(),
            "total_tokens": adapter.total_tokens,
            "total_cost": adapter.get_total_cost(),
        }
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all providers"""
        return {
            provider: self.get_provider_stats(provider)
            for provider in self.adapters.keys()
        }


async def main():
    """Example usage of LLM Router"""
    print("=" * 80)
    print("Minimol LLM Router - Multi-Provider Support")
    print("=" * 80)
    
    # Load config
    config = LLMConfig.from_yaml("config/llm_providers.yaml")
    
    # Create router
    router = LLMRouter(config)
    
    print("\n✅ LLM Router initialized")
    print(f"Available providers: {list(router.adapters.keys())}")
    
    # Test completion
    print("\n" + "=" * 80)
    print("Testing LLM completion...")
    print("=" * 80)
    
    prompt = "Explain quantum computing in 2 sentences."
    
    try:
        result = await router.complete(
            prompt,
            use_case=UseCase.REASONING,
            provider="ollama",
        )
        print(f"\nResult:\n{result}")
    
    except Exception as e:
        print(f"Error: {e}")
    
    # Test streaming
    print("\n" + "=" * 80)
    print("Testing LLM streaming...")
    print("=" * 80)
    
    print("\nStreaming response:")
    try:
        async for chunk in router.stream(
            "What is machine learning?",
            use_case=UseCase.GENERAL,
            provider="ollama",
        ):
            print(chunk, end="", flush=True)
        print()
    
    except Exception as e:
        print(f"Error: {e}")
    
    # Print stats
    print("\n" + "=" * 80)
    print("Provider Statistics")
    print("=" * 80)
    stats = router.get_all_stats()
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
