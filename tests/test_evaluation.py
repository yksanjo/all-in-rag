"""
Tests for evaluation metrics.
"""

import pytest
from src.evaluation.metrics import EvaluationMetrics


def test_retrieval_precision():
    """Test precision calculation."""
    retrieved_docs = [
        {"id": "doc1", "score": 0.9},
        {"id": "doc2", "score": 0.8},
        {"id": "doc3", "score": 0.7}
    ]
    relevant_doc_ids = ["doc1", "doc2", "doc4"]
    
    precision = EvaluationMetrics.retrieval_precision(retrieved_docs, relevant_doc_ids)
    assert precision == 2 / 3  # 2 out of 3 retrieved are relevant


def test_retrieval_recall():
    """Test recall calculation."""
    retrieved_docs = [
        {"id": "doc1", "score": 0.9},
        {"id": "doc2", "score": 0.8}
    ]
    relevant_doc_ids = ["doc1", "doc2", "doc3", "doc4"]
    
    recall = EvaluationMetrics.retrieval_recall(retrieved_docs, relevant_doc_ids)
    assert recall == 0.5  # 2 out of 4 relevant are retrieved


def test_retrieval_f1():
    """Test F1 calculation."""
    retrieved_docs = [
        {"id": "doc1", "score": 0.9},
        {"id": "doc2", "score": 0.8}
    ]
    relevant_doc_ids = ["doc1", "doc2", "doc3"]
    
    f1 = EvaluationMetrics.retrieval_f1(retrieved_docs, relevant_doc_ids)
    precision = 2 / 2  # 2/2
    recall = 2 / 3  # 2/3
    expected_f1 = 2 * (precision * recall) / (precision + recall)
    assert abs(f1 - expected_f1) < 0.001


def test_mean_reciprocal_rank():
    """Test MRR calculation."""
    retrieved_docs = [
        {"id": "doc1", "score": 0.9},
        {"id": "doc2", "score": 0.8},
        {"id": "doc3", "score": 0.7}
    ]
    relevant_doc_ids = ["doc2"]
    
    mrr = EvaluationMetrics.mean_reciprocal_rank(retrieved_docs, relevant_doc_ids)
    assert mrr == 1 / 2  # First relevant doc at rank 2

