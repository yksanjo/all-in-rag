"""
Document chunking module.
Splits documents into smaller chunks for embedding and retrieval.
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from ..config import config


@dataclass
class Chunk:
    """Represents a document chunk."""
    
    text: str
    metadata: Dict[str, Any]
    chunk_id: str
    start_index: int
    end_index: int


class DocumentChunker:
    """Chunks documents into smaller pieces for RAG."""
    
    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ):
        """
        Initialize document chunker.
        
        Args:
            chunk_size: Size of each chunk in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size or config.rag_pipeline.chunk_size
        self.chunk_overlap = chunk_overlap or config.rag_pipeline.chunk_overlap
        
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
    
    def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Split text into chunks.
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to chunks
            
        Returns:
            List of Chunk objects
        """
        if not text or not text.strip():
            return []
        
        metadata = metadata or {}
        chunks = []
        
        # Try to split on sentences first
        sentences = self._split_sentences(text)
        
        current_chunk = []
        current_length = 0
        chunk_start = 0
        
        for i, sentence in enumerate(sentences):
            sentence_length = len(sentence)
            
            # If adding this sentence would exceed chunk size
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = " ".join(current_chunk)
                chunk = Chunk(
                    text=chunk_text,
                    metadata={**metadata, "chunk_index": len(chunks)},
                    chunk_id=f"{metadata.get('source', 'doc')}_chunk_{len(chunks)}",
                    start_index=chunk_start,
                    end_index=chunk_start + len(chunk_text)
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(chunk_text)
                current_chunk = [overlap_text] if overlap_text else []
                current_length = len(overlap_text) if overlap_text else 0
                chunk_start = chunk.end_index - len(overlap_text) if overlap_text else chunk.end_index
            
            current_chunk.append(sentence)
            current_length += sentence_length + 1  # +1 for space
        
        # Add remaining chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunk = Chunk(
                text=chunk_text,
                metadata={**metadata, "chunk_index": len(chunks)},
                chunk_id=f"{metadata.get('source', 'doc')}_chunk_{len(chunks)}",
                start_index=chunk_start,
                end_index=chunk_start + len(chunk_text)
            )
            chunks.append(chunk)
        
        return chunks
    
    def chunk_document(self, document: Dict[str, Any]) -> List[Chunk]:
        """
        Chunk a document (from DocumentLoader).
        
        Args:
            document: Document dictionary from DocumentLoader
            
        Returns:
            List of Chunk objects
        """
        doc_type = document.get("type", "text")
        metadata = document.get("metadata", {})
        
        if doc_type == "text":
            content = document.get("content", "")
            
            # Handle PDF content (list of page dicts)
            if isinstance(content, list):
                all_text = []
                for page in content:
                    if isinstance(page, dict):
                        all_text.append(page.get("text", ""))
                    else:
                        all_text.append(str(page))
                content = "\n\n".join(all_text)
            
            return self.chunk_text(str(content), metadata)
        
        elif doc_type == "image":
            # For images, return single chunk with image path
            return [Chunk(
                text=document.get("content", ""),
                metadata=metadata,
                chunk_id=f"{metadata.get('source', 'image')}_chunk_0",
                start_index=0,
                end_index=0
            )]
        
        else:
            return []
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentence_endings = re.compile(r'[.!?]+\s+')
        sentences = sentence_endings.split(text)
        
        # Clean up sentences
        cleaned = []
        for sent in sentences:
            sent = sent.strip()
            if sent:
                cleaned.append(sent)
        
        return cleaned if cleaned else [text]
    
    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text from the end of a chunk."""
        if self.chunk_overlap == 0:
            return ""
        
        # Take last chunk_overlap characters, but try to start at word boundary
        overlap = text[-self.chunk_overlap:]
        
        # Find first space in overlap to start at word boundary
        first_space = overlap.find(' ')
        if first_space > 0:
            overlap = overlap[first_space + 1:]
        
        return overlap

