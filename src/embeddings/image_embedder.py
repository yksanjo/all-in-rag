"""
Image embedding using OpenCLIP.
"""

from typing import List, Union
import numpy as np
import torch
from PIL import Image

from .base import BaseEmbedder
from ..config import config


class OpenCLIP_Embedder(BaseEmbedder):
    """OpenCLIP image embedder."""
    
    def __init__(self, model_path: str = None, device: str = None):
        """
        Initialize OpenCLIP embedder.
        
        Args:
            model_path: Path to OpenCLIP model (or model name)
            device: Device to run on ('cuda' or 'cpu')
        """
        self.model_path = model_path or config.image_rag.embedding_path
        self.device = device or config.embedding.device
        
        # Lazy loading
        self._model = None
        self._preprocess = None
    
    def _load_model(self):
        """Lazy load the model."""
        if self._model is not None:
            return
        
        try:
            import open_clip
            
            print(f"Loading OpenCLIP model...")
            
            # Try to load from path, otherwise use default model
            try:
                model, _, preprocess = open_clip.create_model_and_transforms(
                    'ViT-B-32',
                    pretrained='openai'
                )
            except:
                # Fallback to local path if provided
                model, _, preprocess = open_clip.create_model_and_transforms(
                    'ViT-B-32',
                    pretrained=self.model_path
                )
            
            if self.device == "cuda" and torch.cuda.is_available():
                self._model = model.to(self.device)
                self.device = "cuda"
            else:
                self._model = model.to("cpu")
                self.device = "cpu"
            
            self._preprocess = preprocess
            self._model.eval()
            print(f"OpenCLIP model loaded on {self.device}")
        
        except ImportError:
            raise ImportError(
                "open-clip-torch library is required. Install with: pip install open-clip-torch"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load OpenCLIP model: {e}")
    
    def embed(self, images: Union[str, List[str], Image.Image, List[Image.Image]]) -> np.ndarray:
        """
        Generate embeddings for images.
        
        Args:
            images: Image path(s) or PIL Image(s)
            
        Returns:
            Numpy array of embeddings
        """
        self._load_model()
        
        if not isinstance(images, list):
            images = [images]
        
        # Load and preprocess images
        processed_images = []
        for img in images:
            if isinstance(img, str):
                img = Image.open(img)
            elif not isinstance(img, Image.Image):
                raise ValueError(f"Invalid image type: {type(img)}")
            
            processed = self._preprocess(img).unsqueeze(0)
            processed_images.append(processed)
        
        # Batch process
        batch = torch.cat(processed_images, dim=0).to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            embeddings = self._model.encode_image(batch)
            embeddings = embeddings.cpu().numpy()
        
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)
        
        return embeddings
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return 512


class ImageEmbedder:
    """Image embedder factory."""
    
    def __init__(
        self,
        model_type: str = "openclip",
        model_path: str = None,
        device: str = None
    ):
        """
        Initialize image embedder.
        
        Args:
            model_type: Type of model (currently only 'openclip')
            model_path: Path to model
            device: Device to run on
        """
        self.model_type = model_type
        self.model_path = model_path or config.image_rag.embedding_path
        self.device = device or config.embedding.device
        
        if self.model_type == "openclip":
            self.embedder = OpenCLIP_Embedder(self.model_path, self.device)
        else:
            raise ValueError(
                f"Unsupported image embedding model type: {self.model_type}. "
                "Supported: 'openclip'"
            )
    
    def embed(self, images: Union[str, List[str], Image.Image, List[Image.Image]]) -> np.ndarray:
        """Generate embeddings."""
        return self.embedder.embed(images)
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.embedder.get_dimension()

