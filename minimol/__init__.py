"""
Minimol - Advanced 70B Parameter Neural Network
A powerful transformer-based neural network with multi-provider LLM support
and beautiful Claude-style terminal UI.
"""

__version__ = "0.1.0"
__author__ = "Fahhed"
__email__ = "fahhedhh@gmail.com"

from minimol.neural_network import (
    Minimol70B,
    MinimolTrainer,
    RotaryPositionalEmbedding,
    MultiHeadAttention,
    TransformerBlock,
)

from minimol.terminal_ui import (
    TerminalUI,
    UIConfig,
    Theme,
    ThemeConfig,
    ConversationMemory,
    CommandParser,
    ResponseRenderer,
    StatusBar,
)

from minimol.llm_router import (
    LLMRouter,
    LLMConfig,
    ProviderConfig,
    ProviderType,
    UseCase,
    OllamaAdapter,
    ClaudeAdapter,
    OpenAIAdapter,
    GeminiAdapter,
    CostOptimizer,
    FallbackChain,
)

__all__ = [
    # Core model
    "Minimol70B",
    "MinimolTrainer",
    "RotaryPositionalEmbedding",
    "MultiHeadAttention",
    "TransformerBlock",
    # Terminal UI
    "TerminalUI",
    "UIConfig",
    "Theme",
    "ThemeConfig",
    "ConversationMemory",
    "CommandParser",
    "ResponseRenderer",
    "StatusBar",
    # LLM Router
    "LLMRouter",
    "LLMConfig",
    "ProviderConfig",
    "ProviderType",
    "UseCase",
    "OllamaAdapter",
    "ClaudeAdapter",
    "OpenAIAdapter",
    "GeminiAdapter",
    "CostOptimizer",
    "FallbackChain",
]
