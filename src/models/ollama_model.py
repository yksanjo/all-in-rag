"""
LLM models using Ollama (alternative to llama.cpp).
"""

from typing import List, Optional

from .base import BaseLLM
from ..config import config


class OllamaModel(BaseLLM):
    """LLM model using Ollama."""
    
    def __init__(
        self,
        model_name: str = None,
        base_url: str = "http://localhost:11434",
        temperature: float = None,
        max_tokens: int = None
    ):
        """
        Initialize Ollama model.
        
        Args:
            model_name: Name of the Ollama model
            base_url: Ollama API base URL
            temperature: Default temperature
            max_tokens: Default max tokens
        """
        self.model_name = model_name or config.llm.model_type
        self.base_url = base_url
        self.temperature = temperature or config.llm.temperature
        self.max_tokens = max_tokens or config.llm.max_tokens
        
        # Lazy loading
        self._client = None
    
    def _get_client(self):
        """Get or create Ollama client."""
        if self._client is not None:
            return self._client
        
        try:
            import ollama
            
            self._client = ollama.Client(host=self.base_url)
            return self._client
        except ImportError:
            raise ImportError(
                "ollama library is required. Install with: pip install ollama"
            )
    
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None
    ) -> str:
        """Generate text from prompt."""
        client = self._get_client()
        
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature if temperature is not None else self.temperature
        
        response = client.generate(
            model=self.model_name,
            prompt=prompt,
            options={
                "num_predict": max_tokens,
                "temperature": temperature,
                "stop": stop or []
            }
        )
        
        return response["response"]
    
    def stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None
    ):
        """Stream generated text."""
        client = self._get_client()
        
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature if temperature is not None else self.temperature
        
        stream = client.generate(
            model=self.model_name,
            prompt=prompt,
            stream=True,
            options={
                "num_predict": max_tokens,
                "temperature": temperature,
                "stop": stop or []
            }
        )
        
        for chunk in stream:
            if "response" in chunk:
                yield chunk["response"]

