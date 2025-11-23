"""
Reranking module for improving retrieval quality.
"""

from typing import List, Dict, Any
import torch

from ..config import config


class Reranker:
    """Reranker for improving retrieval results."""
    
    def __init__(self, model_path: str = None):
        """
        Initialize reranker.
        
        Args:
            model_path: Path to reranker model
        """
        self.model_path = model_path or config.rag_pipeline.reranker_model_path
        self._model = None
        self._tokenizer = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def _load_model(self):
        """Lazy load the reranker model."""
        if self._model is not None:
            return
        
        if not self.model_path:
            raise ValueError("Reranker model path not specified")
        
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            
            print(f"Loading reranker model from {self.model_path}...")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path
            )
            self._model = self._model.to(self.device)
            self._model.eval()
            print(f"Reranker model loaded on {self._device}")
        
        except ImportError:
            raise ImportError(
                "transformers library is required. Install with: pip install transformers"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load reranker model: {e}")
    
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank retrieved documents.
        
        Args:
            query: Query text
            documents: List of retrieved documents
            top_k: Number of top documents to return after reranking
            
        Returns:
            Reranked list of documents
        """
        if not documents:
            return []
        
        self._load_model()
        
        # Extract text from documents
        doc_texts = []
        for doc in documents:
            doc_dict = doc.get("document", {})
            if isinstance(doc_dict, dict):
                text = doc_dict.get("text", "") or doc_dict.get("content", "")
            else:
                text = str(doc_dict)
            doc_texts.append(text)
        
        # Create query-document pairs
        pairs = [[query, doc_text] for doc_text in doc_texts]
        
        # Tokenize
        encoded = self._tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=512
        ).to(self._device)
        
        # Score pairs
        with torch.no_grad():
            scores = self._model(**encoded).logits.squeeze(-1).cpu().numpy()
        
        # Sort by score (higher is better)
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Update scores and return
        reranked = []
        for doc, score in scored_docs:
            doc["score"] = float(score)
            reranked.append(doc)
        
        if top_k:
            reranked = reranked[:top_k]
        
        return reranked
    
    @property
    def device(self):
        """Get device."""
        return self._device

