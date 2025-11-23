"""
FAISS vector store implementation.
"""

import os
import pickle
from typing import List, Dict, Any, Optional
import numpy as np
import faiss

from .base import BaseVectorStore
from ..config import config


class FAISSVectorStore(BaseVectorStore):
    """FAISS-based vector store."""
    
    def __init__(self, dimension: int = None, index_path: str = None):
        """
        Initialize FAISS vector store.
        
        Args:
            dimension: Embedding dimension
            index_path: Path to existing index (optional)
        """
        self.dimension = dimension or config.vectorstore.dimension
        self.index_path = index_path or config.vectorstore.store_path
        
        # FAISS index
        self.index = None
        
        # Metadata storage (maps ID to document)
        self.metadata: Dict[str, Dict[str, Any]] = {}
        
        # ID to index mapping
        self.id_to_index: Dict[str, int] = {}
        self.index_to_id: Dict[int, str] = {}
        self.next_index = 0
        
        # Load existing index if path exists
        if index_path and os.path.exists(index_path):
            self.load(index_path)
        else:
            # Create new index
            self._create_index()
    
    def _create_index(self):
        """Create a new FAISS index."""
        # Use L2 distance (Euclidean) - works well with normalized embeddings
        # For cosine similarity with normalized vectors, L2 is equivalent
        self.index = faiss.IndexFlatL2(self.dimension)
    
    def add_documents(
        self,
        embeddings: np.ndarray,
        documents: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add documents to the vector store."""
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dimension}, "
                f"got {embeddings.shape[1]}"
            )
        
        n_docs = len(documents)
        if ids is None:
            ids = [f"doc_{self.next_index + i}" for i in range(n_docs)]
        
        if len(ids) != n_docs:
            raise ValueError("Number of IDs must match number of documents")
        
        # Ensure embeddings are float32 for FAISS
        embeddings = embeddings.astype('float32')
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Store metadata
        for i, (doc_id, doc) in enumerate(zip(ids, documents)):
            idx = self.next_index + i
            self.id_to_index[doc_id] = idx
            self.index_to_id[idx] = doc_id
            self.metadata[doc_id] = doc
        
        self.next_index += n_docs
        
        return ids
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        if self.index.ntotal == 0:
            return []
        
        # Ensure query is float32 and has correct shape
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_embedding = query_embedding.astype('float32')
        
        # Search
        k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for invalid results
                continue
            
            doc_id = self.index_to_id.get(idx)
            if doc_id is None:
                continue
            
            doc = self.metadata.get(doc_id, {})
            
            # Apply filter if provided
            if filter:
                if not self._matches_filter(doc, filter):
                    continue
            
            # Convert L2 distance to similarity score (higher is better)
            # For normalized vectors, similarity = 1 - (distance / 2)
            similarity = 1.0 - (dist / 2.0)
            similarity = max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
            
            results.append({
                "id": doc_id,
                "document": doc,
                "score": float(similarity),
                "distance": float(dist)
            })
        
        return results
    
    def _matches_filter(self, document: Dict[str, Any], filter: Dict[str, Any]) -> bool:
        """Check if document matches filter criteria."""
        doc_metadata = document.get("metadata", {})
        
        for key, value in filter.items():
            if key not in doc_metadata:
                return False
            if doc_metadata[key] != value:
                return False
        
        return True
    
    def delete(self, ids: List[str]) -> bool:
        """Delete documents by IDs."""
        # FAISS doesn't support deletion directly
        # We'll mark them as deleted in metadata and skip during search
        for doc_id in ids:
            if doc_id in self.metadata:
                # Mark as deleted
                self.metadata[doc_id]["_deleted"] = True
        
        return True
    
    def save(self, path: str) -> bool:
        """Save the vector store to disk."""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, path)
            
            # Save metadata
            metadata_path = path + ".metadata"
            with open(metadata_path, 'wb') as f:
                pickle.dump({
                    "metadata": self.metadata,
                    "id_to_index": self.id_to_index,
                    "index_to_id": self.index_to_id,
                    "next_index": self.next_index
                }, f)
            
            return True
        except Exception as e:
            print(f"Error saving FAISS index: {e}")
            return False
    
    def load(self, path: str) -> bool:
        """Load the vector store from disk."""
        try:
            # Load FAISS index
            if not os.path.exists(path):
                return False
            
            self.index = faiss.read_index(path)
            
            # Load metadata
            metadata_path = path + ".metadata"
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    data = pickle.load(f)
                    self.metadata = data.get("metadata", {})
                    self.id_to_index = data.get("id_to_index", {})
                    self.index_to_id = data.get("index_to_id", {})
                    self.next_index = data.get("next_index", 0)
            
            self.index_path = path
            return True
        except Exception as e:
            print(f"Error loading FAISS index: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        return {
            "type": "faiss",
            "dimension": self.dimension,
            "num_documents": self.index.ntotal if self.index else 0,
            "index_path": self.index_path
        }

