"""Minimol: Advanced 70B parameter neural network with terminal UI and multi-provider LLM support."""

__version__ = "0.1.0"
__author__ = "Fahhed"
__license__ = "MIT"

from minimol.neural_network import NeuralNetwork
from minimol.llm_router import LLMRouter

__all__ = [
    "NeuralNetwork",
    "LLMRouter",
]
