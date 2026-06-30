"""
Terminal UI Module for Minimol
Claude-style interactive terminal interface with real-time streaming,
syntax highlighting, and agent capabilities.
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any, AsyncGenerator
from datetime import datetime
from enum import Enum
import json

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.live import Live
from rich.text import Text
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.align import Align
from rich.layout import Layout
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style


class Theme(Enum):
    """Available color themes"""
    DARK = "dark"
    LIGHT = "light"
    NORD = "nord"
    MONOKAI = "monokai"
    DRACULA = "dracula"
    SOLARIZED = "solarized"


class ThemeConfig:
    """Theme configuration management"""
    
    THEMES = {
        Theme.DARK: {
            "user_text": "cyan",
            "assistant_text": "green",
            "tool_text": "yellow",
            "error_text": "red",
            "status_text": "dim white",
            "bg": "black",
        },
        Theme.LIGHT: {
            "user_text": "blue",
            "assistant_text": "green",
            "tool_text": "orange1",
            "error_text": "red",
            "status_text": "grey37",
            "bg": "white",
        },
        Theme.NORD: {
            "user_text": "cornflower_blue",
            "assistant_text": "sea_green1",
            "tool_text": "yellow",
            "error_text": "light_red",
            "status_text": "grey74",
            "bg": "grey15",
        },
        Theme.MONOKAI: {
            "user_text": "cyan",
            "assistant_text": "green",
            "tool_text": "yellow",
            "error_text": "red1",
            "status_text": "grey50",
            "bg": "black",
        },
        Theme.DRACULA: {
            "user_text": "cornflower_blue",
            "assistant_text": "light_green",
            "tool_text": "yellow",
            "error_text": "light_red",
            "status_text": "grey69",
            "bg": "grey11",
        },
        Theme.SOLARIZED: {
            "user_text": "blue",
            "assistant_text": "green",
            "tool_text": "yellow",
            "error_text": "red",
            "status_text": "base01",
            "bg": "base03",
        },
    }
    
    def __init__(self, theme: Theme = Theme.DARK):
        self.theme = theme
        self.colors = self.THEMES[theme]
    
    def get_color(self, element: str) -> str:
        """Get color for element type"""
        return self.colors.get(element, "white")


class ConversationMemory:
    """Manages conversation history and context"""
    
    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self.messages: List[Dict[str, Any]] = []
        self.created_at = datetime.now()
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add message to memory"""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }
        self.messages.append(message)
        
        # Maintain max history
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]
    
    def get_context(self, num_messages: int = 10) -> List[Dict[str, Any]]:
        """Get last N messages for context"""
        return self.messages[-num_messages:]
    
    def clear(self):
        """Clear conversation history"""
        self.messages = []
    
    def export_json(self, filepath: str):
        """Export conversation to JSON"""
        data = {
            "created_at": self.created_at.isoformat(),
            "num_messages": len(self.messages),
            "messages": self.messages,
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
    
    def export_markdown(self, filepath: str):
        """Export conversation to Markdown"""
        with open(filepath, "w") as f:
            f.write(f"# Conversation\n\n")
            f.write(f"Created: {self.created_at.isoformat()}\n\n")
            
            for msg in self.messages:
                role = msg["role"].upper()
                content = msg["content"]
                timestamp = msg["timestamp"]
                
                f.write(f"## {role}\n")
                f.write(f"*{timestamp}*\n\n")
                f.write(f"{content}\n\n")
                f.write("---\n\n")


class StatusBar:
    """Displays model status, tokens, and provider info"""
    
    def __init__(self, theme_config: ThemeConfig):
        self.theme_config = theme_config
        self.tokens_used = 0
        self.memory_percent = 0.0
        self.current_provider = "ollama"
        self.current_model = "mistral"
        self.response_time = 0.0
    
    def get_status_text(self) -> str:
        """Generate status bar text"""
        status = f"📊 Tokens: {self.tokens_used} | "
        status += f"Memory: {self.memory_percent:.1f}% | "
        status += f"Provider: {self.current_provider} "
        status += f"({self.response_time:.0f}ms)"
        return status
    
    def render(self) -> Text:
        """Render status bar"""
        status_text = self.get_status_text()
        color = self.theme_config.get_color("status_text")
        return Text(status_text, style=color)


class ResponseRenderer:
    """Renders responses with syntax highlighting and formatting"""
    
    def __init__(self, theme_config: ThemeConfig, console: Optional[Console] = None):
        self.theme_config = theme_config
        self.console = console or Console()
    
    def render_text(self, text: str, role: str = "assistant") -> str:
        """Render plain text response"""
        color = self.theme_config.get_color(f"{role}_text")
        return f"[{color}]{text}[/{color}]"
    
    def render_code(self, code: str, language: str = "python") -> Syntax:
        """Render code block with syntax highlighting"""
        return Syntax(code, language, theme="monokai", line_numbers=True)
    
    def render_markdown(self, text: str) -> Markdown:
        """Render markdown content"""
        return Markdown(text)
    
    def render_table(self, data: List[List[str]], headers: List[str]) -> Table:
        """Render table data"""
        table = Table(title="Results")
        
        for header in headers:
            table.add_column(header, style="cyan")
        
        for row in data:
            table.add_row(*row)
        
        return table


class UIConfig:
    """Configuration for terminal UI"""
    
    def __init__(
        self,
        theme: Theme = Theme.DARK,
        streaming: bool = True,
        token_counter: bool = True,
        memory_indicator: bool = True,
        provider_badge: bool = True,
        syntax_highlight: bool = True,
        max_history_display: int = 10,
        auto_scroll: bool = True,
        line_wrap: bool = True,
        render_fps: int = 30,
    ):
        self.theme = theme
        self.streaming = streaming
        self.token_counter = token_counter
        self.memory_indicator = memory_indicator
        self.provider_badge = provider_badge
        self.syntax_highlight = syntax_highlight
        self.max_history_display = max_history_display
        self.auto_scroll = auto_scroll
        self.line_wrap = line_wrap
        self.render_fps = render_fps
    
    @classmethod
    def from_yaml(cls, filepath: str) -> "UIConfig":
        """Load configuration from YAML file"""
        import yaml
        
        if not Path(filepath).exists():
            return cls()
        
        with open(filepath, "r") as f:
            config_data = yaml.safe_load(f) or {}
        
        ui_config = config_data.get("ui", {})
        return cls(
            theme=Theme(ui_config.get("theme", "dark")),
            streaming=ui_config.get("streaming", True),
            token_counter=ui_config.get("token_counter", True),
            memory_indicator=ui_config.get("memory_indicator", True),
            provider_badge=ui_config.get("provider_badge", True),
            syntax_highlight=ui_config.get("syntax_highlight", True),
            max_history_display=ui_config.get("max_history_display", 10),
            auto_scroll=ui_config.get("auto_scroll", True),
            line_wrap=ui_config.get("line_wrap", True),
            render_fps=ui_config.get("render_fps", 30),
        )


class CommandParser:
    """Parses and handles UI commands"""
    
    COMMANDS = {
        "/help": "Show help menu",
        "/clear": "Clear screen",
        "/reset": "Reset conversation",
        "/switch": "Switch model/provider",
        "/export": "Export conversation",
        "/memory": "Show memory state",
        "/stats": "Show statistics",
        "/settings": "Configure settings",
        "/quit": "Exit application",
        "@tool": "Invoke specific tool",
        "#tag": "Search by tag",
    }
    
    def __init__(self, console: Console):
        self.console = console
    
    def is_command(self, text: str) -> bool:
        """Check if text is a command"""
        return any(text.startswith(cmd) for cmd in self.COMMANDS.keys())
    
    def parse(self, text: str) -> tuple[str, Optional[str]]:
        """Parse command and arguments"""
        parts = text.split(maxsplit=1)
        command = parts[0]
        args = parts[1] if len(parts) > 1 else None
        return command, args
    
    def show_help(self):
        """Display help menu"""
        table = Table(title="Available Commands")
        table.add_column("Command", style="cyan")
        table.add_column("Description", style="green")
        
        for cmd, desc in self.COMMANDS.items():
            table.add_row(cmd, desc)
        
        self.console.print(table)
    
    def handle_command(self, command: str, args: Optional[str]) -> Dict[str, Any]:
        """Handle command execution"""
        return {
            "command": command,
            "args": args,
            "timestamp": datetime.now().isoformat(),
        }


class TerminalUI:
    """Main terminal UI class for interactive agent sessions"""
    
    def __init__(
        self,
        ui_config: Optional[UIConfig] = None,
        theme: Theme = Theme.DARK,
    ):
        self.ui_config = ui_config or UIConfig(theme=theme)
        self.console = Console(force_terminal=True, width=100)
        self.theme_config = ThemeConfig(self.ui_config.theme)
        self.memory = ConversationMemory()
        self.status_bar = StatusBar(self.theme_config)
        self.renderer = ResponseRenderer(self.theme_config, self.console)
        self.command_parser = CommandParser(self.console)
        
        # Session state
        self.is_running = False
        self.current_provider = "ollama"
        self.current_model = "mistral"
        self.system_prompt = "You are a helpful AI assistant."
        
        # Prompt session for multiline input
        history_file = Path.home() / ".minimol_history"
        self.prompt_session = PromptSession(
            history=FileHistory(str(history_file)),
            multiline=True,
        )
    
    def display_header(self):
        """Display application header"""
        title = "🧠 Minimol Terminal UI"
        provider_info = f"[{self.current_provider}] {self.current_model}"
        
        header = Panel(
            f"{title}\n{provider_info}",
            style="bold cyan",
            expand=False,
        )
        self.console.print(header)
    
    def display_welcome(self):
        """Display welcome message"""
        welcome_text = """
[cyan]Welcome to Minimol Terminal UI[/cyan]

A powerful transformer-based neural network with 70B parameters.

[green]Commands:[/green]
  /help    - Show all commands
  /clear   - Clear screen
  /reset   - Reset conversation
  /switch  - Switch provider/model
  /export  - Export conversation
  /quit    - Exit application

Type your message and press Enter to send.
Press Ctrl+D or type '/quit' to exit.
        """
        self.console.print(Markdown(welcome_text))
    
    async def get_user_input(self) -> str:
        """Get user input with multiline support"""
        try:
            user_input = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.prompt_session.prompt("You: "),
            )
            return user_input.strip()
        except (EOFError, KeyboardInterrupt):
            return "/quit"
    
    def display_thinking(self):
        """Display thinking indicator"""
        thinking_text = Text("🤖 Minimol is thinking... ", style="yellow")
        self.console.print(thinking_text)
    
    async def stream_response(self, response: str) -> AsyncGenerator[str, None]:
        """Stream response character by character"""
        if not self.ui_config.streaming:
            yield response
            return
        
        for char in response:
            yield char
            await asyncio.sleep(0.01)  # Control streaming speed
    
    def display_response(self, response: str, role: str = "assistant"):
        """Display assistant response"""
        color = self.theme_config.get_color(f"{role}_text")
        
        # Check if response contains code blocks
        if "```" in response:
            self.console.print(Markdown(response))
        else:
            text = Text(response, style=color)
            panel = Panel(text, title="Response", expand=True)
            self.console.print(panel)
    
    def display_tool_execution(self, tool_name: str, status: str = "executing"):
        """Display tool execution status"""
        color = self.theme_config.get_color("tool_text")
        status_text = f"🔧 {tool_name}: {status}"
        self.console.print(Text(status_text, style=color))
    
    def update_stats(self, tokens: int = 0, memory_percent: float = 0.0, latency_ms: float = 0.0):
        """Update status bar statistics"""
        self.status_bar.tokens_used += tokens
        self.status_bar.memory_percent = memory_percent
        self.status_bar.response_time = latency_ms
    
    def display_status(self):
        """Display status bar"""
        status = self.status_bar.render()
        self.console.print(status)
    
    def handle_command(self, command_text: str):
        """Handle UI command"""
        command, args = self.command_parser.parse(command_text)
        
        if command == "/help":
            self.command_parser.show_help()
        
        elif command == "/clear":
            self.console.clear()
        
        elif command == "/reset":
            self.memory.clear()
            self.console.print("[yellow]Conversation reset.[/yellow]")
        
        elif command == "/switch":
            if args:
                parts = args.split()
                self.current_provider = parts[0] if len(parts) > 0 else self.current_provider
                self.current_model = parts[1] if len(parts) > 1 else self.current_model
                self.status_bar.current_provider = self.current_provider
                self.status_bar.current_model = self.current_model
                msg = f"[green]Switched to {self.current_provider}:{self.current_model}[/green]"
                self.console.print(msg)
        
        elif command == "/export":
            format_type = args or "json"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if format_type == "json":
                filepath = f"conversation_{timestamp}.json"
                self.memory.export_json(filepath)
            elif format_type == "md":
                filepath = f"conversation_{timestamp}.md"
                self.memory.export_markdown(filepath)
            
            self.console.print(f"[green]Conversation exported to {filepath}[/green]")
        
        elif command == "/memory":
            context = self.memory.get_context(self.ui_config.max_history_display)
            table = Table(title="Conversation Memory")
            table.add_column("Role", style="cyan")
            table.add_column("Content", style="green")
            
            for msg in context:
                content = msg["content"][:50] + "..." if len(msg["content"]) > 50 else msg["content"]
                table.add_row(msg["role"], content)
            
            self.console.print(table)
        
        elif command == "/stats":
            stats_text = f"""
            [cyan]Statistics[/cyan]
            - Total messages: {len(self.memory.messages)}
            - Tokens used: {self.status_bar.tokens_used}
            - Current provider: {self.current_provider}
            - Current model: {self.current_model}
            - Session duration: {datetime.now() - self.memory.created_at}
            """
            self.console.print(Markdown(stats_text))
        
        elif command == "/quit":
            self.is_running = False
    
    async def run(self):
        """Main event loop for terminal UI"""
        self.is_running = True
        self.display_header()
        self.display_welcome()
        
        try:
            while self.is_running:
                # Get user input
                user_input = await self.get_user_input()
                
                if not user_input:
                    continue
                
                # Check if it's a command
                if self.command_parser.is_command(user_input):
                    self.handle_command(user_input)
                    if user_input == "/quit":
                        break
                    continue
                
                # Add to memory
                self.memory.add_message("user", user_input)
                
                # Display thinking
                self.display_thinking()
                
                # Simulate response (in real implementation, this would call LLM)
                simulated_response = f"This is a response to: {user_input[:50]}...\n\n"
                simulated_response += "```python\n"
                simulated_response += "def hello():\n    print('Hello from Minimol!')\n"
                simulated_response += "```\n\n"
                simulated_response += "This is a demonstration response."
                
                # Stream and display response
                streamed_response = ""
                async for chunk in self.stream_response(simulated_response):
                    streamed_response += chunk
                
                self.display_response(streamed_response)
                self.memory.add_message("assistant", streamed_response)
                
                # Update stats
                self.update_stats(tokens=len(streamed_response.split()), memory_percent=45.0, latency_ms=150.0)
                self.display_status()
        
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Interrupted. Exiting...[/yellow]")
        except EOFError:
            pass
        finally:
            self.is_running = False
            self.console.print("[cyan]Goodbye![/cyan]")


async def main():
    """Main entry point for terminal UI"""
    # Load configuration
    config_path = Path.home() / ".minimol" / "ui.yaml"
    ui_config = UIConfig.from_yaml(str(config_path))
    
    # Create and run UI
    ui = TerminalUI(ui_config=ui_config)
    await ui.run()


def main_sync():
    """Synchronous entry point"""
    asyncio.run(main())


if __name__ == "__main__":
    asyncio.run(main())
