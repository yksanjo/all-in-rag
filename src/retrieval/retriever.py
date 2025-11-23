"""
Document retriever.
"""

from typing import List, Dict, Any, Optional
import numpy as np

from ..vectorstore import BaseVectorStore
from ..embeddings import TextEmbedder, ImageEmbedder
from ..config import config


class Retriever:
    """Document retriever for RAG."""
    
    def __init__(
        self,
        vector_store: BaseVectorStore,
        text_embedder: TextEmbedder,
        image_embedder: Optional[ImageEmbedder] = None,
        top_k: int = None
    ):
        """
        Initialize retriever.
        
        Args:
            vector_store: Vector store instance
            text_embedder: Text embedding model
            image_embedder: Optional image embedding model
            top_k: Number of documents to retrieve
        """
        self.vector_store = vector_store
        self.text_embedder = text_embedder
        self.image_embedder = image_embedder
        self.top_k = top_k or config.vectorstore.top_k
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Query text
            top_k: Number of documents to retrieve
            filter: Optional metadata filter
            
        Returns:
            List of retrieved documents with scores
        """
        top_k = top_k or self.top_k
        
        # Generate query embedding
        query_embedding = self.text_embedder.embed(query)
        
        # Search vector store
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter=filter
        )
        
        return results
    
    def retrieve_by_image(
        self,
        image_path: str,
        top_k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents using an image query.
        
        Args:
            image_path: Path to query image
            top_k: Number of documents to retrieve
            filter: Optional metadata filter
            
        Returns:
            List of retrieved documents with scores
        """
        if self.image_embedder is None:
            raise ValueError("Image embedder not initialized")
        
        top_k = top_k or self.top_k
        
        # Generate image embedding
        query_embedding = self.image_embedder.embed(image_path)
        
        # Search vector store
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter=filter
        )
        
        return results

