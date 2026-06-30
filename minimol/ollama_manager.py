"""
Ollama Manager Module for Minimol
Manages local Ollama model downloads, caching, and resource monitoring.
"""

import asyncio
import subprocess
import json
import psutil
import httpx
from typing import Optional, List, Dict, Any, AsyncGenerator
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import platform


@dataclass
class ModelInfo:
    """Information about an Ollama model"""
    name: str
    digest: str
    size: int
    created_at: str
    modified_at: str
    quantization: str = "unknown"
    parameters: str = "unknown"
    
    def get_size_gb(self) -> float:
        """Get model size in GB"""
        return self.size / (1024 ** 3)


@dataclass
class SystemResources:
    """System resource information"""
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_available_gb: float
    gpu_available: bool
    gpu_memory_gb: float = 0.0


class OllamaManager:
    """Manages Ollama local models and resources"""
    
    def __init__(self, endpoint: str = "http://localhost:11434"):
        self.endpoint = endpoint
        self.client = httpx.AsyncClient(base_url=endpoint, timeout=300.0)
        self.models_cache: Dict[str, ModelInfo] = {}
        self.is_running = False
    
    async def check_health(self) -> bool:
        """Check if Ollama service is running"""
        try:
            response = await self.client.get("/api/tags")
            self.is_running = response.status_code == 200
            return self.is_running
        except Exception as e:
            print(f"Ollama health check failed: {e}")
            self.is_running = False
            return False
    
    async def start_ollama(self) -> bool:
        """Start Ollama service"""
        try:
            system = platform.system()
            
            if system == "Darwin":  # macOS
                subprocess.Popen(["open", "-a", "Ollama"])
            elif system == "Windows":
                subprocess.Popen(["ollama", "serve"])
            elif system == "Linux":
                subprocess.Popen(["ollama", "serve"])
            
            # Wait for service to start
            for _ in range(30):
                if await self.check_health():
                    print("✅ Ollama service started")
                    return True
                await asyncio.sleep(1)
            
            return False
        
        except Exception as e:
            print(f"Failed to start Ollama: {e}")
            return False
    
    async def list_models(self, refresh: bool = False) -> List[ModelInfo]:
        """List available models"""
        if self.models_cache and not refresh:
            return list(self.models_cache.values())
        
        try:
            response = await self.client.get("/api/tags")
            data = response.json()
            
            self.models_cache.clear()
            for model_data in data.get("models", []):
                model = ModelInfo(
                    name=model_data["name"],
                    digest=model_data.get("digest", ""),
                    size=model_data.get("size", 0),
                    created_at=model_data.get("created_at", ""),
                    modified_at=model_data.get("modified_at", ""),
                )
                self.models_cache[model.name] = model
            
            return list(self.models_cache.values())
        
        except Exception as e:
            print(f"Failed to list models: {e}")
            return []
    
    async def pull_model(
        self,
        model_name: str,
        progress_callback=None,
    ) -> bool:
        """Download a model from Ollama registry"""
        try:
            print(f"📥 Pulling model: {model_name}")
            
            # Check available space
            resources = await self.get_system_resources()
            model_size_estimate = 7.0  # Default 7GB estimate
            
            if resources.disk_available_gb < model_size_estimate:
                print(f"❌ Insufficient disk space. Required: {model_size_estimate}GB, Available: {resources.disk_available_gb:.1f}GB")
                return False
            
            async with self.client.stream("POST", "/api/pull", json={"name": model_name}) as response:
                async for line in response.aiter_lines():
                    if line:
                        data = json.loads(line)
                        
                        if "error" in data:
                            print(f"❌ Error: {data['error']}")
                            return False
                        
                        # Call progress callback
                        if progress_callback:
                            completed = data.get("completed", 0)
                            total = data.get("total", 0)
                            status = data.get("status", "")
                            
                            progress_callback({
                                "status": status,
                                "completed": completed,
                                "total": total,
                                "percent": (completed / total * 100) if total > 0 else 0,
                            })
            
            print(f"✅ Model pulled successfully: {model_name}")
            
            # Refresh cache
            await self.list_models(refresh=True)
            return True
        
        except Exception as e:
            print(f"❌ Failed to pull model: {e}")
            return False
    
    async def remove_model(self, model_name: str) -> bool:
        """Remove a model from local storage"""
        try:
            response = await self.client.delete("/api/delete", json={"name": model_name})
            
            if response.status_code == 200:
                print(f"✅ Model removed: {model_name}")
                
                # Remove from cache
                if model_name in self.models_cache:
                    del self.models_cache[model_name]
                
                return True
            else:
                print(f"❌ Failed to remove model: {response.text}")
                return False
        
        except Exception as e:
            print(f"❌ Error removing model: {e}")
            return False
    
    async def generate(
        self,
        model: str,
        prompt: str,
        stream: bool = True,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate text using a model"""
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": stream,
                **kwargs,
            }
            
            if stream:
                async with self.client.stream("POST", "/api/generate", json=payload) as response:
                    async for line in response.aiter_lines():
                        if line:
                            data = json.loads(line)
                            chunk = data.get("response", "")
                            if chunk:
                                yield chunk
            else:
                response = await self.client.post("/api/generate", json=payload)
                data = response.json()
                yield data.get("response", "")
        
        except Exception as e:
            print(f"Generation error: {e}")
            yield ""
    
    async def embed(self, model: str, text: str) -> Optional[List[float]]:
        """Get embeddings for text"""
        try:
            response = await self.client.post(
                "/api/embeddings",
                json={"model": model, "prompt": text}
            )
            data = response.json()
            return data.get("embedding")
        
        except Exception as e:
            print(f"Embedding error: {e}")
            return None
    
    async def get_system_resources(self) -> SystemResources:
        """Get current system resource information"""
        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        
        # GPU detection
        gpu_available = False
        gpu_memory_gb = 0.0
        
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_available = True
                gpu_memory_gb = gpus[0].memoryTotal / 1024  # Convert to GB
        except Exception:
            pass
        
        return SystemResources(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_available_gb=memory.available / (1024 ** 3),
            disk_available_gb=disk.free / (1024 ** 3),
            gpu_available=gpu_available,
            gpu_memory_gb=gpu_memory_gb,
        )
    
    async def optimize_resources(self) -> Dict[str, Any]:
        """Check and optimize system resources"""
        resources = await self.get_system_resources()
        recommendations = []
        
        # CPU check
        if resources.cpu_percent > 80:
            recommendations.append("High CPU usage - consider reducing batch size")
        
        # Memory check
        if resources.memory_percent > 85:
            recommendations.append("High memory usage - consider using smaller models")
        
        # Disk check
        if resources.disk_available_gb < 10:
            recommendations.append("Low disk space - consider removing unused models")
        
        return {
            "resources": {
                "cpu_percent": resources.cpu_percent,
                "memory_percent": resources.memory_percent,
                "memory_available_gb": resources.memory_available_gb,
                "disk_available_gb": resources.disk_available_gb,
                "gpu_available": resources.gpu_available,
                "gpu_memory_gb": resources.gpu_memory_gb,
            },
            "recommendations": recommendations,
            "status": "optimal" if not recommendations else "warning",
        }
    
    async def show_models(self) -> str:
        """Display models in formatted table"""
        models = await self.list_models()
        
        if not models:
            return "No models found"
        
        # Format as table
        header = f"{'Model':<30} {'Size (GB)':<15} {'Modified':<20}"
        lines = [header, "-" * len(header)]
        
        for model in models:
            size_gb = model.get_size_gb()
            modified = model.modified_at.split("T")[0] if model.modified_at else "unknown"
            lines.append(f"{model.name:<30} {size_gb:<15.2f} {modified:<20}")
        
        return "\n".join(lines)
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.client.aclose()


async def main():
    """Example usage of Ollama Manager"""
    print("=" * 80)
    print("Minimol Ollama Manager")
    print("=" * 80)
    
    manager = OllamaManager()
    
    # Check health
    print("\n📋 Checking Ollama service...")
    is_healthy = await manager.check_health()
    
    if not is_healthy:
        print("⚠️  Ollama service not running. Starting...")
        if await manager.start_ollama():
            print("✅ Ollama started successfully")
        else:
            print("❌ Failed to start Ollama")
            return
    else:
        print("✅ Ollama service is running")
    
    # List models
    print("\n📦 Available Models:")
    print(await manager.show_models())
    
    # System resources
    print("\n💻 System Resources:")
    resources_info = await manager.optimize_resources()
    print(json.dumps(resources_info, indent=2))
    
    # Test generation
    print("\n🤖 Testing model generation...")
    try:
        async for chunk in manager.generate(
            "llama2:70b",
            "What is machine learning? (1 sentence)",
            temperature=0.7,
        ):
            print(chunk, end="", flush=True)
        print()
    except Exception as e:
        print(f"Generation test failed: {e}")
    
    # Cleanup
    await manager.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
