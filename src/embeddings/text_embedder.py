"""
Text embedding models: BGE-M3 and GTE-Large.
"""

import os
from typing import List, Union
import numpy as np
import torch

from .base import BaseEmbedder
from ..config import config


class BGE_M3_Embedder(BaseEmbedder):
    """BGE-M3 text embedder."""
    
    def __init__(self, model_path: str = None, device: str = None):
        """
        Initialize BGE-M3 embedder.
        
        Args:
            model_path: Path to BGE-M3 model
            device: Device to run on ('cuda' or 'cpu')
        """
        self.model_path = model_path or config.embedding.model_path
        self.device = device or config.embedding.device
        
        # Lazy loading - model loaded on first use
        self._model = None
        self._tokenizer = None
    
    def _load_model(self):
        """Lazy load the model."""
        if self._model is not None:
            return
        
        try:
            from transformers import AutoModel, AutoTokenizer
            
            print(f"Loading BGE-M3 model from {self.model_path}...")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self._model = AutoModel.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            if self.device == "cuda" and torch.cuda.is_available():
                self._model = self._model.to(self.device)
            else:
                self.device = "cpu"
                self._model = self._model.to(self.device)
            
            self._model.eval()
            print(f"BGE-M3 model loaded on {self.device}")
        
        except ImportError:
            raise ImportError(
                "transformers library is required. Install with: pip install transformers"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load BGE-M3 model: {e}")
    
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings."""
        self._load_model()
        
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize
        encoded_input = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=512
        ).to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            model_output = self._model(**encoded_input)
            # Use mean pooling
            embeddings = model_output.last_hidden_state
            attention_mask = encoded_input['attention_mask']
            embeddings = (embeddings * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1, keepdim=True)
            embeddings = embeddings.cpu().numpy()
        
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)
        
        return embeddings
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return 1024


class GTE_Large_Embedder(BaseEmbedder):
    """GTE-Large text embedder."""
    
    def __init__(self, model_path: str = None, device: str = None):
        """
        Initialize GTE-Large embedder.
        
        Args:
            model_path: Path to GTE-Large model
            device: Device to run on ('cuda' or 'cpu')
        """
        self.model_path = model_path or config.embedding.model_path
        self.device = device or config.embedding.device
        
        # Lazy loading
        self._model = None
        self._tokenizer = None
    
    def _load_model(self):
        """Lazy load the model."""
        if self._model is not None:
            return
        
        try:
            from transformers import AutoModel, AutoTokenizer
            
            print(f"Loading GTE-Large model from {self.model_path}...")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self._model = AutoModel.from_pretrained(self.model_path)
            
            if self.device == "cuda" and torch.cuda.is_available():
                self._model = self._model.to(self.device)
            else:
                self.device = "cpu"
                self._model = self._model.to(self.device)
            
            self._model.eval()
            print(f"GTE-Large model loaded on {self.device}")
        
        except ImportError:
            raise ImportError(
                "transformers library is required. Install with: pip install transformers"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load GTE-Large model: {e}")
    
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings."""
        self._load_model()
        
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize
        encoded_input = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=512
        ).to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            model_output = self._model(**encoded_input)
            # Mean pooling
            embeddings = model_output.last_hidden_state
            attention_mask = encoded_input['attention_mask']
            embeddings = (embeddings * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1, keepdim=True)
            embeddings = embeddings.cpu().numpy()
        
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)
        
        return embeddings
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return 1024


class TextEmbedder:
    """Text embedder factory."""
    
    def __init__(
        self,
        model_type: str = None,
        model_path: str = None,
        device: str = None
    ):
        """
        Initialize text embedder.
        
        Args:
            model_type: Type of model ('bge-m3' or 'gte-large')
            model_path: Path to model
            device: Device to run on
        """
        self.model_type = model_type or config.embedding.model_type
        self.model_path = model_path or config.embedding.model_path
        self.device = device or config.embedding.device
        
        if self.model_type == "bge-m3":
            self.embedder = BGE_M3_Embedder(self.model_path, self.device)
        elif self.model_type == "gte-large":
            self.embedder = GTE_Large_Embedder(self.model_path, self.device)
        else:
            raise ValueError(
                f"Unsupported embedding model type: {self.model_type}. "
                "Supported: 'bge-m3', 'gte-large'"
            )
    
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings."""
        return self.embedder.embed(texts)
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.embedder.get_dimension()

