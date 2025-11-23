"""
Embedding models module.
Supports BGE-M3, GTE-Large, and OpenCLIP for image embeddings.
"""

from .base import BaseEmbedder
from .text_embedder import TextEmbedder
from .image_embedder import ImageEmbedder

__all__ = ["BaseEmbedder", "TextEmbedder", "ImageEmbedder"]

