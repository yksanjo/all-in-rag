"""
Main RAG pipeline.
Orchestrates retrieval and generation.
"""

from typing import List, Dict, Any, Optional

from ..models import BaseLLM
from ..retrieval import Retriever, Reranker
from ..config import config


class RAGPipeline:
    """Complete RAG pipeline."""
    
    def __init__(
        self,
        llm: BaseLLM,
        retriever: Retriever,
        reranker: Optional[Reranker] = None
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            llm: Language model instance
            retriever: Document retriever
            reranker: Optional reranker for improving retrieval
        """
        self.llm = llm
        self.retriever = retriever
        self.reranker = reranker
    
    def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
        use_reranking: bool = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Process a query through the RAG pipeline.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            filter: Optional metadata filter
            use_reranking: Whether to use reranking
            max_tokens: Max tokens for generation
            temperature: Temperature for generation
            
        Returns:
            Dictionary with 'answer', 'sources', and 'retrieved_docs'
        """
        # Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(
            query=query,
            top_k=top_k,
            filter=filter
        )
        
        # Rerank if enabled
        use_reranking = use_reranking if use_reranking is not None else config.rag_pipeline.enable_reranking
        if use_reranking and self.reranker and retrieved_docs:
            retrieved_docs = self.reranker.rerank(
                query=query,
                documents=retrieved_docs,
                top_k=top_k
            )
        
        # Build context from retrieved documents
        context = self._build_context(retrieved_docs)
        
        # Build prompt
        prompt = self._build_prompt(query, context)
        
        # Generate answer
        answer = self.llm.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # Extract sources
        sources = [
            {
                "id": doc.get("id"),
                "score": doc.get("score"),
                "metadata": doc.get("document", {}).get("metadata", {})
            }
            for doc in retrieved_docs
        ]
        
        return {
            "answer": answer,
            "sources": sources,
            "retrieved_docs": retrieved_docs,
            "query": query
        }
    
    def stream_query(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
        use_reranking: bool = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ):
        """
        Stream query processing.
        
        Yields:
            Dictionary chunks with 'answer_chunk' and 'done' keys
        """
        # Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(
            query=query,
            top_k=top_k,
            filter=filter
        )
        
        # Rerank if enabled
        use_reranking = use_reranking if use_reranking is not None else config.rag_pipeline.enable_reranking
        if use_reranking and self.reranker and retrieved_docs:
            retrieved_docs = self.reranker.rerank(
                query=query,
                documents=retrieved_docs,
                top_k=top_k
            )
        
        # Build context
        context = self._build_context(retrieved_docs)
        
        # Build prompt
        prompt = self._build_prompt(query, context)
        
        # Stream generation
        full_answer = ""
        for chunk in self.llm.stream(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature
        ):
            full_answer += chunk
            yield {
                "answer_chunk": chunk,
                "answer": full_answer,
                "done": False
            }
        
        # Extract sources
        sources = [
            {
                "id": doc.get("id"),
                "score": doc.get("score"),
                "metadata": doc.get("document", {}).get("metadata", {})
            }
            for doc in retrieved_docs
        ]
        
        yield {
            "answer_chunk": "",
            "answer": full_answer,
            "sources": sources,
            "retrieved_docs": retrieved_docs,
            "done": True
        }
    
    def _build_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Build context string from retrieved documents."""
        if not retrieved_docs:
            return "No relevant documents found."
        
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            doc_dict = doc.get("document", {})
            
            # Extract text content
            if isinstance(doc_dict, dict):
                text = doc_dict.get("text", "") or doc_dict.get("content", "")
            else:
                text = str(doc_dict)
            
            # Add metadata info
            metadata = doc_dict.get("metadata", {})
            source = metadata.get("source", "Unknown")
            
            context_parts.append(f"[Document {i} - Source: {source}]\n{text}\n")
        
        return "\n---\n".join(context_parts)
    
    def _build_prompt(self, query: str, context: str) -> str:
        """Build prompt for LLM."""
        prompt = f"""You are a helpful assistant that answers questions based on the provided context documents.

Context Documents:
{context}

Question: {query}

Please provide a comprehensive answer based on the context above. If the context doesn't contain enough information to answer the question, please say so.

Answer:"""
        return prompt

