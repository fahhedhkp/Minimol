# Minimol: Advanced 70B Parameter Neural Network with Terminal UI

A powerful transformer-based neural network with 70 billion parameters, designed for advanced sequence modeling, language understanding, and autonomous agent capabilities. Comparable to Hermes 2 Pro with extended context and specialized agent features. Seamlessly integrates with leading LLM providers like Anthropic's Claude, OpenAI's GPT models, Google Gemini, and local models via Ollama for enhanced reasoning capabilities. Features a beautiful Claude-style terminal UI for interactive agent interactions.

## 🚀 Features

- **Scale**: 70B parameters for enterprise-grade performance and reasoning
- **Architecture**: Transformer encoder with rotary positional embeddings (RoPE)
- **Attention**: Multi-head self-attention with 64 heads
- **Context**: 32,000 token context window (32K) for long-document understanding
- **Optimization**: Layer normalization, gradient clipping, AdamW optimizer
- **Inference**: Fast text generation with top-k and nucleus sampling
- **Training**: Complete training pipeline with checkpointing and validation
- **Multi-Model Support**: Integrate with Claude, GPT-4, Gemini, Ollama local models, and other external LLMs for enhanced reasoning
- **Beautiful Terminal UI**: Claude-style interactive interface with syntax highlighting, streaming responses, and real-time agent actions

### ✨ Advanced Features (Hermes + OpenClaw Inspired)

#### **Hermes-Inspired: Instruction Following & Reasoning**
- **Multi-turn Dialogue**: Maintain context across extended conversations
- **Instruction Following**: Optimized for natural language commands and complex instructions
- **Strong Reasoning**: Step-by-step problem-solving and logical inference
- **Multi-Role Support**: System, user, assistant, and tool message roles (ChatML format)
- **Safety & Clarity**: Fine-tuned to minimize bias, hallucinations, and toxicity
- **Long-Context Processing**: Handle extended sequences (32K tokens) for document analysis, research, code review

#### **OpenClaw-Inspired: Autonomous Agent Capabilities**
- **Actionable Agent**: Not just a chatbot—execute real tasks and workflows
- **Tool Integration**: Built-in support for function calling and tool use
- **Skill/Plugin Architecture**: Extensible system for adding custom capabilities
- **Memory & Context Management**: Persistent conversation memory with contextual awareness
- **Workflow Automation**: Task chains and conditional logic execution
- **Multi-Channel Support**: Designed to integrate with multiple communication platforms
- **External LLM Integration**: Route complex reasoning tasks to Claude, GPT-4, Gemini, Ollama, or other providers
- **Interactive Terminal UI**: Modern Claude-like interface with real-time streaming, syntax highlighting, and visual feedback

## Installation

### Quick Start

```bash
pip install minimol[ui]
```

### From Source

```bash
git clone https://github.com/fahhedhh-ctrl/Minimol.git
cd Minimol
pip install -e .[ui]
```

## Usage

```bash
minimol-ui
```

## License

MIT License
