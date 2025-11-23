"""
Tests for document ingestion.
"""

import pytest
from pathlib import Path
import tempfile
import os

from src.ingest import DocumentLoader, DocumentChunker


def test_txt_loader():
    """Test text file loading."""
    loader = DocumentLoader()
    
    # Create temporary text file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is a test document.\nIt has multiple lines.")
        temp_path = f.name
    
    try:
        doc = loader.load(temp_path)
        assert doc["type"] == "text"
        assert "test document" in doc["content"]
        assert doc["metadata"]["file_type"] == "txt"
    finally:
        os.unlink(temp_path)


def test_chunker():
    """Test document chunking."""
    chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)
    
    text = "This is a long document that needs to be chunked. " * 10
    chunks = chunker.chunk_text(text)
    
    assert len(chunks) > 0
    assert all(len(chunk.text) <= 50 + 20 for chunk in chunks)  # Allow some flexibility


def test_chunk_document():
    """Test chunking a document."""
    loader = DocumentLoader()
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("Test content. " * 20)
        temp_path = f.name
    
    try:
        doc = loader.load(temp_path)
        chunks = chunker.chunk_document(doc)
        assert len(chunks) > 0
    finally:
        os.unlink(temp_path)

