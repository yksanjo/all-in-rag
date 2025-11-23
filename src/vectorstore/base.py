"""
Base vector store interface.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np


class BaseVectorStore(ABC):
    """Base class for vector stores."""
    
    @abstractmethod
    def add_documents(
        self,
        embeddings: np.ndarray,
        documents: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            embeddings: Numpy array of embeddings (n_docs, embedding_dim)
            documents: List of document dictionaries
            ids: Optional list of document IDs
            
        Returns:
            List of document IDs
        """
        pass
    
    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of search results with 'document', 'score', and 'id' keys
        """
        pass
    
    @abstractmethod
    def delete(self, ids: List[str]) -> bool:
        """
        Delete documents by IDs.
        
        Args:
            ids: List of document IDs to delete
            
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> bool:
        """
        Save the vector store to disk.
        
        Args:
            path: Path to save to
            
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> bool:
        """
        Load the vector store from disk.
        
        Args:
            path: Path to load from
            
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        pass

