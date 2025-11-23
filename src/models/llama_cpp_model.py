"""
LLM models using llama.cpp (GGUF format).
"""

from typing import List, Optional
import os

from .base import BaseLLM
from ..config import config


class LlamaCppModel(BaseLLM):
    """LLM model using llama.cpp for GGUF models."""
    
    def __init__(
        self,
        model_path: str = None,
        context_size: int = None,
        n_gpu_layers: int = None,
        temperature: float = None,
        max_tokens: int = None
    ):
        """
        Initialize llama.cpp model.
        
        Args:
            model_path: Path to GGUF model file
            context_size: Context window size
            n_gpu_layers: Number of layers to offload to GPU
            temperature: Default temperature
            max_tokens: Default max tokens
        """
        self.model_path = model_path or config.llm.model_path
        self.context_size = context_size or config.llm.context_size
        self.n_gpu_layers = n_gpu_layers or config.llm.n_gpu_layers
        self.temperature = temperature or config.llm.temperature
        self.max_tokens = max_tokens or config.llm.max_tokens
        
        # Lazy loading
        self._llm = None
    
    def _load_model(self):
        """Lazy load the model."""
        if self._llm is not None:
            return
        
        try:
            from llama_cpp import Llama
            
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"Model file not found: {self.model_path}\n"
                    "Please download a GGUF model and update LLM_MODEL_PATH in .env"
                )
            
            print(f"Loading GGUF model from {self.model_path}...")
            self._llm = Llama(
                model_path=self.model_path,
                n_ctx=self.context_size,
                n_gpu_layers=self.n_gpu_layers,
                verbose=False
            )
            print("Model loaded successfully")
        
        except ImportError:
            raise ImportError(
                "llama-cpp-python is required. Install with: pip install llama-cpp-python"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None
    ) -> str:
        """Generate text from prompt."""
        self._load_model()
        
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature if temperature is not None else self.temperature
        
        response = self._llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            echo=False
        )
        
        return response["choices"][0]["text"]
    
    def stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None
    ):
        """Stream generated text."""
        self._load_model()
        
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature if temperature is not None else self.temperature
        
        stream = self._llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            echo=False,
            stream=True
        )
        
        for chunk in stream:
            if "choices" in chunk and len(chunk["choices"]) > 0:
                text = chunk["choices"][0].get("text", "")
                if text:
                    yield text

