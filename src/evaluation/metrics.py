"""
Evaluation metrics for RAG systems.
"""

from typing import List, Dict, Any
import numpy as np


class EvaluationMetrics:
    """Collection of evaluation metrics."""
    
    @staticmethod
    def retrieval_precision(
        retrieved_docs: List[Dict[str, Any]],
        relevant_doc_ids: List[str]
    ) -> float:
        """
        Calculate retrieval precision.
        
        Args:
            retrieved_docs: List of retrieved documents
            relevant_doc_ids: List of relevant document IDs
            
        Returns:
            Precision score (0-1)
        """
        if not retrieved_docs:
            return 0.0
        
        relevant_set = set(relevant_doc_ids)
        retrieved_ids = [doc.get("id") for doc in retrieved_docs]
        
        relevant_retrieved = sum(1 for doc_id in retrieved_ids if doc_id in relevant_set)
        
        return relevant_retrieved / len(retrieved_docs)
    
    @staticmethod
    def retrieval_recall(
        retrieved_docs: List[Dict[str, Any]],
        relevant_doc_ids: List[str]
    ) -> float:
        """
        Calculate retrieval recall.
        
        Args:
            retrieved_docs: List of retrieved documents
            relevant_doc_ids: List of all relevant document IDs
            
        Returns:
            Recall score (0-1)
        """
        if not relevant_doc_ids:
            return 1.0
        
        relevant_set = set(relevant_doc_ids)
        retrieved_ids = [doc.get("id") for doc in retrieved_docs]
        
        relevant_retrieved = sum(1 for doc_id in retrieved_ids if doc_id in relevant_set)
        
        return relevant_retrieved / len(relevant_doc_ids)
    
    @staticmethod
    def retrieval_f1(
        retrieved_docs: List[Dict[str, Any]],
        relevant_doc_ids: List[str]
    ) -> float:
        """
        Calculate retrieval F1 score.
        
        Args:
            retrieved_docs: List of retrieved documents
            relevant_doc_ids: List of relevant document IDs
            
        Returns:
            F1 score (0-1)
        """
        precision = EvaluationMetrics.retrieval_precision(retrieved_docs, relevant_doc_ids)
        recall = EvaluationMetrics.retrieval_recall(retrieved_docs, relevant_doc_ids)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def mean_reciprocal_rank(
        retrieved_docs: List[Dict[str, Any]],
        relevant_doc_ids: List[str]
    ) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).
        
        Args:
            retrieved_docs: List of retrieved documents
            relevant_doc_ids: List of relevant document IDs
            
        Returns:
            MRR score (0-1)
        """
        if not relevant_doc_ids:
            return 0.0
        
        relevant_set = set(relevant_doc_ids)
        
        for rank, doc in enumerate(retrieved_docs, 1):
            if doc.get("id") in relevant_set:
                return 1.0 / rank
        
        return 0.0
    
    @staticmethod
    def mean_average_precision(
        retrieved_docs: List[Dict[str, Any]],
        relevant_doc_ids: List[str]
    ) -> float:
        """
        Calculate Mean Average Precision (MAP).
        
        Args:
            retrieved_docs: List of retrieved documents
            relevant_doc_ids: List of relevant document IDs
            
        Returns:
            MAP score (0-1)
        """
        if not relevant_doc_ids:
            return 0.0
        
        relevant_set = set(relevant_doc_ids)
        precision_scores = []
        relevant_count = 0
        
        for rank, doc in enumerate(retrieved_docs, 1):
            if doc.get("id") in relevant_set:
                relevant_count += 1
                precision = relevant_count / rank
                precision_scores.append(precision)
        
        if not precision_scores:
            return 0.0
        
        return np.mean(precision_scores)
    
    @staticmethod
    def average_retrieval_score(retrieved_docs: List[Dict[str, Any]]) -> float:
        """
        Calculate average retrieval score.
        
        Args:
            retrieved_docs: List of retrieved documents with scores
            
        Returns:
            Average score
        """
        if not retrieved_docs:
            return 0.0
        
        scores = [doc.get("score", 0.0) for doc in retrieved_docs]
        return np.mean(scores)
    
    @staticmethod
    def calculate_all_retrieval_metrics(
        retrieved_docs: List[Dict[str, Any]],
        relevant_doc_ids: List[str]
    ) -> Dict[str, float]:
        """
        Calculate all retrieval metrics.
        
        Args:
            retrieved_docs: List of retrieved documents
            relevant_doc_ids: List of relevant document IDs
            
        Returns:
            Dictionary of metric names and values
        """
        return {
            "precision": EvaluationMetrics.retrieval_precision(retrieved_docs, relevant_doc_ids),
            "recall": EvaluationMetrics.retrieval_recall(retrieved_docs, relevant_doc_ids),
            "f1": EvaluationMetrics.retrieval_f1(retrieved_docs, relevant_doc_ids),
            "mrr": EvaluationMetrics.mean_reciprocal_rank(retrieved_docs, relevant_doc_ids),
            "map": EvaluationMetrics.mean_average_precision(retrieved_docs, relevant_doc_ids),
            "avg_score": EvaluationMetrics.average_retrieval_score(retrieved_docs)
        }

