"""
RAG system evaluator.
"""

from typing import List, Dict, Any, Optional
import time

from .metrics import EvaluationMetrics
from ..rag_pipeline import RAGPipeline


class RAGEvaluator:
    """Evaluator for RAG pipeline."""
    
    def __init__(self, pipeline: RAGPipeline):
        """
        Initialize evaluator.
        
        Args:
            pipeline: RAG pipeline to evaluate
        """
        self.pipeline = pipeline
        self.metrics = EvaluationMetrics()
    
    def evaluate_query(
        self,
        query: str,
        expected_answer: Optional[str] = None,
        relevant_doc_ids: Optional[List[str]] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single query.
        
        Args:
            query: Query text
            expected_answer: Optional expected answer (for future answer quality metrics)
            relevant_doc_ids: Optional list of relevant document IDs
            filter: Optional metadata filter
            
        Returns:
            Dictionary with evaluation results
        """
        # Measure latency
        start_time = time.time()
        result = self.pipeline.query(query=query, filter=filter)
        latency = time.time() - start_time
        
        # Calculate retrieval metrics if relevant docs provided
        retrieval_metrics = {}
        if relevant_doc_ids:
            retrieval_metrics = self.metrics.calculate_all_retrieval_metrics(
                retrieved_docs=result["retrieved_docs"],
                relevant_doc_ids=relevant_doc_ids
            )
        
        return {
            "query": query,
            "answer": result["answer"],
            "sources": result["sources"],
            "retrieved_docs": result["retrieved_docs"],
            "latency_seconds": latency,
            "retrieval_metrics": retrieval_metrics,
            "num_retrieved": len(result["retrieved_docs"])
        }
    
    def evaluate_batch(
        self,
        queries: List[str],
        relevant_doc_ids_list: Optional[List[List[str]]] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a batch of queries.
        
        Args:
            queries: List of query texts
            relevant_doc_ids_list: Optional list of relevant doc ID lists (one per query)
            filter: Optional metadata filter
            
        Returns:
            Dictionary with aggregated evaluation results
        """
        results = []
        
        for i, query in enumerate(queries):
            relevant_doc_ids = None
            if relevant_doc_ids_list and i < len(relevant_doc_ids_list):
                relevant_doc_ids = relevant_doc_ids_list[i]
            
            result = self.evaluate_query(
                query=query,
                relevant_doc_ids=relevant_doc_ids,
                filter=filter
            )
            results.append(result)
        
        # Aggregate metrics
        latencies = [r["latency_seconds"] for r in results]
        
        aggregated = {
            "num_queries": len(queries),
            "avg_latency": sum(latencies) / len(latencies) if latencies else 0.0,
            "min_latency": min(latencies) if latencies else 0.0,
            "max_latency": max(latencies) if latencies else 0.0,
            "results": results
        }
        
        # Aggregate retrieval metrics if available
        if relevant_doc_ids_list:
            all_precisions = []
            all_recalls = []
            all_f1s = []
            
            for result in results:
                metrics = result.get("retrieval_metrics", {})
                if metrics:
                    all_precisions.append(metrics.get("precision", 0.0))
                    all_recalls.append(metrics.get("recall", 0.0))
                    all_f1s.append(metrics.get("f1", 0.0))
            
            if all_precisions:
                aggregated["avg_precision"] = sum(all_precisions) / len(all_precisions)
                aggregated["avg_recall"] = sum(all_recalls) / len(all_recalls)
                aggregated["avg_f1"] = sum(all_f1s) / len(all_f1s)
        
        return aggregated

