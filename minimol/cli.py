"""
CLI Module for Minimol
Command-line interface for the Minimol system.
"""

import asyncio
import click
import json
import yaml
from pathlib import Path
from typing import Optional

from minimol.terminal_ui import TerminalUI, UIConfig, Theme
from minimol.inference import InferenceEngine
from minimol.ollama_manager import OllamaManager
from minimol.llm_router import LLMConfig, UseCase


@click.group()
def cli():
    """Minimol - Advanced 70B Parameter Neural Network CLI"""
    pass


@cli.command()
@click.option("--theme", type=click.Choice(["dark", "light", "nord", "monokai", "dracula", "solarized"]), default="dark")
@click.option("--config", type=click.Path(), default="config/ui.yaml")
def chat(theme: str, config: str):
    """Start interactive chat session"""
    async def _chat():
        click.echo(f"🚀 Starting Minimol Terminal UI (theme: {theme})")
        
        ui_config = UIConfig.from_yaml(config)
        ui_config.theme = Theme(theme)
        
        ui = TerminalUI(ui_config=ui_config)
        await ui.run()
    
    asyncio.run(_chat())


@cli.command()
@click.argument("prompt")
@click.option("--provider", default="ollama", help="LLM provider to use")
@click.option("--model", default=None, help="Specific model to use")
@click.option("--use-case", type=click.Choice(["general", "reasoning", "coding", "analysis", "creative", "fast", "local", "budget"]), default="general")
@click.option("--temperature", type=float, default=0.7)
@click.option("--max-tokens", type=int, default=2048)
@click.option("--stream/--no-stream", default=True, help="Stream output")
def generate(prompt: str, provider: str, model: Optional[str], use_case: str, temperature: float, max_tokens: int, stream: bool):
    """Generate text completion"""
    async def _generate():
        engine = InferenceEngine()
        await engine.initialize()
        
        try:
            use_case_enum = UseCase[use_case.upper()]
            
            if stream:
                click.echo("Response: ", nl=False)
                async for chunk in engine.stream(
                    prompt,
                    use_case=use_case_enum,
                    provider=provider,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ):
                    click.echo(chunk, nl=False)
                click.echo()
            else:
                result = await engine.generate(
                    prompt,
                    use_case=use_case_enum,
                    provider=provider,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                click.echo(f"Response:\n{result}")
        
        except Exception as e:
            click.echo(f"Error: {e}", err=True)
        
        finally:
            await engine.cleanup()
    
    asyncio.run(_generate())


@cli.group(name="ollama")
def ollama_group():
    """Ollama model management commands"""
    pass


@ollama_group.command(name="list")
def ollama_list():
    """List available Ollama models"""
    async def _list():
        manager = OllamaManager()
        
        if not await manager.check_health():
            click.echo("⚠️  Ollama service not running", err=True)
            return
        
        models = await manager.list_models()
        
        if not models:
            click.echo("No models found")
            return
        
        click.echo(await manager.show_models())
        await manager.cleanup()
    
    asyncio.run(_list())


@ollama_group.command(name="pull")
@click.argument("model")
def ollama_pull(model: str):
    """Download an Ollama model"""
    async def _pull():
        manager = OllamaManager()
        
        if not await manager.check_health():
            click.echo("⚠️  Ollama service not running", err=True)
            return
        
        def progress_callback(info):
            percent = info["percent"]
            status = info["status"]
            click.echo(f"{status}: {percent:.0f}%", err=True)
        
        await manager.pull_model(model, progress_callback=progress_callback)
        await manager.cleanup()
    
    asyncio.run(_pull())


@ollama_group.command(name="remove")
@click.argument("model")
def ollama_remove(model: str):
    """Remove an Ollama model"""
    async def _remove():
        manager = OllamaManager()
        
        if not await manager.check_health():
            click.echo("⚠️  Ollama service not running", err=True)
            return
        
        success = await manager.remove_model(model)
        if success:
            click.echo(f"✅ Model removed: {model}")
        else:
            click.echo(f"❌ Failed to remove model: {model}", err=True)
        
        await manager.cleanup()
    
    asyncio.run(_remove())


@ollama_group.command(name="start")
def ollama_start():
    """Start Ollama service"""
    async def _start():
        manager = OllamaManager()
        
        click.echo("🚀 Starting Ollama service...")
        if await manager.start_ollama():
            click.echo("✅ Ollama started successfully")
        else:
            click.echo("❌ Failed to start Ollama", err=True)
        
        await manager.cleanup()
    
    asyncio.run(_start())


@ollama_group.command(name="status")
def ollama_status():
    """Check Ollama service status"""
    async def _status():
        manager = OllamaManager()
        
        if await manager.check_health():
            click.echo("✅ Ollama service is running")
            resources_info = await manager.optimize_resources()
            click.echo(json.dumps(resources_info, indent=2))
        else:
            click.echo("❌ Ollama service is not running", err=True)
        
        await manager.cleanup()
    
    asyncio.run(_status())


@cli.group(name="config")
def config_group():
    """Configuration management commands"""
    pass


@config_group.command(name="show")
@click.option("--section", default=None, help="Specific config section")
def config_show(section: Optional[str]):
    """Show configuration"""
    config_path = Path("config/llm_providers.yaml")
    
    if not config_path.exists():
        click.echo(f"Config file not found: {config_path}", err=True)
        return
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    if section:
        if section in config:
            click.echo(yaml.dump({section: config[section]}))
        else:
            click.echo(f"Section not found: {section}", err=True)
    else:
        click.echo(yaml.dump(config))


@config_group.command(name="init")
def config_init():
    """Initialize default configuration"""
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    click.echo("✅ Configuration initialized")


@cli.command()
def version():
    """Show version information"""
    from minimol import __version__
    click.echo(f"Minimol v{__version__}")


@cli.command()
def info():
    """Show system and package information"""
    import platform
    try:
        import torch
        torch_available = True
        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
    except ImportError:
        torch_available = False
        torch_version = "N/A"
        cuda_available = False
    
    info_dict = {
        "system": platform.system(),
        "python_version": platform.python_version(),
        "pytorch_available": torch_available,
        "pytorch_version": torch_version,
        "cuda_available": cuda_available,
    }
    
    click.echo(json.dumps(info_dict, indent=2))


def main():
    """Main CLI entry point"""
    try:
        cli()
    except KeyboardInterrupt:
        click.echo("\n\nInterrupted by user", err=True)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


if __name__ == "__main__":
    main()
