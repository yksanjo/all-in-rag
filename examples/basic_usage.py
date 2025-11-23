"""
Basic usage example for Enterprise RAG system.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.factory import create_pipeline
from src.ingest import DocumentLoader, DocumentChunker
from src.embeddings import TextEmbedder
from src.vectorstore import FAISSVectorStore


def main():
    """Example usage of the RAG pipeline."""
    
    print("Initializing RAG pipeline...")
    
    # Create pipeline using factory
    pipeline = create_pipeline()
    
    print("Pipeline created successfully!")
    
    # Example: Load and index a document
    print("\nLoading document...")
    loader = DocumentLoader()
    chunker = DocumentChunker()
    
    # Example document path (replace with your document)
    doc_path = "data/documents/example.txt"
    
    if Path(doc_path).exists():
        # Load document
        doc = loader.load(doc_path)
        print(f"Loaded document: {doc['metadata']['file_name']}")
        
        # Chunk document
        chunks = chunker.chunk_document(doc)
        print(f"Created {len(chunks)} chunks")
        
        # Generate embeddings
        print("Generating embeddings...")
        text_embedder = TextEmbedder()
        texts = [chunk.text for chunk in chunks]
        embeddings = text_embedder.embed(texts)
        
        # Add to vector store
        print("Indexing documents...")
        vector_store = pipeline.retriever.vector_store
        documents = [
            {
                "text": chunk.text,
                "metadata": chunk.metadata
            }
            for chunk in chunks
        ]
        vector_store.add_documents(embeddings, documents)
        
        # Save index
        if hasattr(vector_store, 'save'):
            vector_store.save(vector_store.index_path)
        
        print("Documents indexed successfully!")
    else:
        print(f"Document not found: {doc_path}")
        print("Please add a document to index first.")
    
    # Example: Query the system
    print("\nQuerying the system...")
    query = "What is the main topic of the documents?"
    
    result = pipeline.query(query=query, top_k=3)
    
    print(f"\nQuery: {query}")
    print(f"\nAnswer: {result['answer']}")
    print(f"\nSources ({len(result['sources'])}):")
    for i, source in enumerate(result['sources'], 1):
        print(f"  {i}. Score: {source['score']:.3f}")
        print(f"     Metadata: {source.get('metadata', {})}")


if __name__ == "__main__":
    main()

