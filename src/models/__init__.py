"""
Local LLM models module.
Supports Llama, Qwen, and Mistral via llama.cpp or Ollama.
"""

from .base import BaseLLM
from .llama_cpp_model import LlamaCppModel
from .ollama_model import OllamaModel

__all__ = ["BaseLLM", "LlamaCppModel", "OllamaModel"]

