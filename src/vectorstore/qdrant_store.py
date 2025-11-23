"""
Qdrant vector store implementation (local mode).
"""

import os
from typing import List, Dict, Any, Optional
import numpy as np

from .base import BaseVectorStore
from ..config import config


class QdrantVectorStore(BaseVectorStore):
    """Qdrant-based vector store (local mode)."""
    
    def __init__(self, dimension: int = None, collection_name: str = "rag_collection", path: str = None):
        """
        Initialize Qdrant vector store.
        
        Args:
            dimension: Embedding dimension
            collection_name: Name of the collection
            path: Path to Qdrant data directory
        """
        self.dimension = dimension or config.vectorstore.dimension
        self.collection_name = collection_name
        self.path = path or os.path.join(config.vectorstore.store_path, "qdrant")
        
        # Lazy load Qdrant client
        self._client = None
        self._collection_created = False
    
    def _get_client(self):
        """Get or create Qdrant client."""
        if self._client is not None:
            return self._client
        
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams, PointStruct
            
            self._client = QdrantClient(path=self.path)
            self._PointStruct = PointStruct
            self._Distance = Distance
            self._VectorParams = VectorParams
            
            return self._client
        except ImportError:
            raise ImportError(
                "qdrant-client is required. Install with: pip install qdrant-client"
            )
    
    def _ensure_collection(self):
        """Ensure collection exists."""
        if self._collection_created:
            return
        
        client = self._get_client()
        
        try:
            # Check if collection exists
            collections = client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                # Create collection
                client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=self._VectorParams(
                        size=self.dimension,
                        distance=self._Distance.COSINE
                    )
                )
            
            self._collection_created = True
        except Exception as e:
            raise RuntimeError(f"Failed to create Qdrant collection: {e}")
    
    def add_documents(
        self,
        embeddings: np.ndarray,
        documents: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add documents to the vector store."""
        self._ensure_collection()
        
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dimension}, "
                f"got {embeddings.shape[1]}"
            )
        
        n_docs = len(documents)
        if ids is None:
            ids = [f"doc_{i}" for i in range(n_docs)]
        
        if len(ids) != n_docs:
            raise ValueError("Number of IDs must match number of documents")
        
        client = self._get_client()
        
        # Prepare points
        points = []
        for i, (doc_id, doc, embedding) in enumerate(zip(ids, documents, embeddings)):
            points.append(
                self._PointStruct(
                    id=i,  # Qdrant uses integer IDs
                    vector=embedding.tolist(),
                    payload={
                        "doc_id": doc_id,
                        "document": doc
                    }
                )
            )
        
        # Upsert points
        client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        return ids
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        self._ensure_collection()
        
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        client = self._get_client()
        
        # Build query filter if provided
        query_filter = None
        if filter:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            conditions = []
            for key, value in filter.items():
                conditions.append(
                    FieldCondition(
                        key=f"document.metadata.{key}",
                        match=MatchValue(value=value)
                    )
                )
            query_filter = Filter(must=conditions)
        
        # Search
        search_results = client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding[0].tolist(),
            limit=top_k,
            query_filter=query_filter
        )
        
        results = []
        for result in search_results:
            payload = result.payload
            doc = payload.get("document", {})
            
            results.append({
                "id": payload.get("doc_id"),
                "document": doc,
                "score": float(result.score),
                "distance": None  # Qdrant returns similarity score
            })
        
        return results
    
    def delete(self, ids: List[str]) -> bool:
        """Delete documents by IDs."""
        self._ensure_collection()
        
        client = self._get_client()
        
        # Find point IDs by doc_id
        # Note: This is simplified - in production you'd maintain a mapping
        try:
            # Scroll to find points with matching doc_ids
            from qdrant_client.models import Filter, FieldCondition, MatchAny
            
            point_ids = []
            scroll_result = client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="doc_id",
                            match=MatchAny(any=ids)
                        )
                    ]
                ),
                limit=10000
            )
            
            for point in scroll_result[0]:
                point_ids.append(point.id)
            
            if point_ids:
                client.delete(
                    collection_name=self.collection_name,
                    points_selector=point_ids
                )
            
            return True
        except Exception as e:
            print(f"Error deleting from Qdrant: {e}")
            return False
    
    def save(self, path: str) -> bool:
        """Save the vector store to disk."""
        # Qdrant saves automatically to the path directory
        # This method is mainly for API compatibility
        return True
    
    def load(self, path: str) -> bool:
        """Load the vector store from disk."""
        # Qdrant loads automatically from the path directory
        self.path = path
        self._client = None  # Reset client to reload from new path
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        try:
            self._ensure_collection()
            client = self._get_client()
            collection_info = client.get_collection(self.collection_name)
            
            return {
                "type": "qdrant",
                "dimension": self.dimension,
                "num_documents": collection_info.points_count,
                "collection_name": self.collection_name,
                "path": self.path
            }
        except Exception:
            return {
                "type": "qdrant",
                "dimension": self.dimension,
                "num_documents": 0,
                "collection_name": self.collection_name,
                "path": self.path
            }

