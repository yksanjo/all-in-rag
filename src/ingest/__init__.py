"""
Document ingestion module.
Supports PDF, DOCX, TXT, and image files.
"""

from .document_loader import DocumentLoader
from .chunker import DocumentChunker

__all__ = ["DocumentLoader", "DocumentChunker"]

