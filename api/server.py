"""
FastAPI server for Enterprise RAG system.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
import uvicorn

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.ingest import DocumentLoader, DocumentChunker
from src.embeddings import TextEmbedder, ImageEmbedder
from src.vectorstore import FAISSVectorStore, QdrantVectorStore
from src.retrieval import Retriever, Reranker
from src.models import LlamaCppModel, OllamaModel
from src.rag_pipeline import RAGPipeline


# Initialize FastAPI app
app = FastAPI(
    title="Enterprise Offline RAG API",
    description="Fully offline RAG system API",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline (initialized on startup)
pipeline: Optional[RAGPipeline] = None
vector_store = None
text_embedder = None
image_embedder = None


def initialize_pipeline():
    """Initialize the RAG pipeline."""
    global pipeline, vector_store, text_embedder, image_embedder
    
    print("Initializing RAG pipeline...")
    
    # Initialize embedders
    print("Loading text embedder...")
    text_embedder = TextEmbedder()
    
    if config.image_rag.enable_image_rag:
        print("Loading image embedder...")
        image_embedder = ImageEmbedder()
    else:
        image_embedder = None
    
    # Initialize vector store
    print("Initializing vector store...")
    if config.vectorstore.store_type == "faiss":
        vector_store = FAISSVectorStore(
            dimension=text_embedder.get_dimension()
        )
        # Try to load existing index
        if os.path.exists(config.vectorstore.store_path):
            vector_store.load(config.vectorstore.store_path)
    elif config.vectorstore.store_type == "qdrant":
        vector_store = QdrantVectorStore(
            dimension=text_embedder.get_dimension()
        )
    else:
        raise ValueError(f"Unsupported vector store type: {config.vectorstore.store_type}")
    
    # Initialize retriever
    retriever = Retriever(
        vector_store=vector_store,
        text_embedder=text_embedder,
        image_embedder=image_embedder
    )
    
    # Initialize reranker if enabled
    reranker = None
    if config.rag_pipeline.enable_reranking:
        print("Loading reranker...")
        reranker = Reranker()
    
    # Initialize LLM
    print("Loading LLM...")
    if config.llm.model_type in ["qwen2.5", "llama3.1", "mistral"]:
        llm = LlamaCppModel()
    else:
        # Try Ollama as fallback
        llm = OllamaModel()
    
    # Create pipeline
    pipeline = RAGPipeline(
        llm=llm,
        retriever=retriever,
        reranker=reranker
    )
    
    print("RAG pipeline initialized successfully!")


@app.on_event("startup")
async def startup_event():
    """Initialize pipeline on startup."""
    try:
        initialize_pipeline()
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        print("Pipeline will be initialized on first request.")


# Request/Response models
class QueryRequest(BaseModel):
    """Query request model."""
    query: str
    top_k: Optional[int] = None
    use_reranking: Optional[bool] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    filter: Optional[Dict[str, Any]] = None


class QueryResponse(BaseModel):
    """Query response model."""
    answer: str
    sources: List[Dict[str, Any]]
    query: str


class UploadResponse(BaseModel):
    """Upload response model."""
    message: str
    num_documents: int
    document_ids: List[str]


# API endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "pipeline_initialized": pipeline is not None,
        "vector_store_stats": vector_store.get_stats() if vector_store else None
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the RAG system.
    
    Args:
        request: Query request with question and optional parameters
        
    Returns:
        Query response with answer and sources
    """
    if pipeline is None:
        try:
            initialize_pipeline()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Pipeline initialization failed: {e}")
    
    try:
        result = pipeline.query(
            query=request.query,
            top_k=request.top_k,
            filter=request.filter,
            use_reranking=request.use_reranking,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            query=result["query"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.post("/upload", response_model=UploadResponse)
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Upload and index documents.
    
    Args:
        files: List of files to upload
        
    Returns:
        Upload response with number of documents indexed
    """
    if pipeline is None:
        try:
            initialize_pipeline()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Pipeline initialization failed: {e}")
    
    # Create upload directory
    upload_dir = Path(config.data.documents_path)
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    document_ids = []
    loader = DocumentLoader()
    chunker = DocumentChunker()
    
    for file in files:
        # Save uploaded file
        file_path = upload_dir / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        try:
            # Load document
            doc = loader.load(str(file_path))
            
            # Chunk document
            chunks = chunker.chunk_document(doc)
            
            # Generate embeddings
            texts = [chunk.text for chunk in chunks]
            if texts:
                embeddings = text_embedder.embed(texts)
                
                # Prepare documents for vector store
                documents = []
                for chunk in chunks:
                    documents.append({
                        "text": chunk.text,
                        "metadata": {
                            **chunk.metadata,
                            "chunk_id": chunk.chunk_id
                        }
                    })
                
                # Add to vector store
                ids = vector_store.add_documents(
                    embeddings=embeddings,
                    documents=documents
                )
                document_ids.extend(ids)
        
        except Exception as e:
            print(f"Error processing file {file.filename}: {e}")
            continue
    
    # Save vector store
    if config.vectorstore.store_type == "faiss":
        vector_store.save(config.vectorstore.store_path)
    
    return UploadResponse(
        message=f"Successfully indexed {len(document_ids)} document chunks",
        num_documents=len(document_ids),
        document_ids=document_ids
    )


@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    if vector_store is None:
        raise HTTPException(status_code=500, detail="Vector store not initialized")
    
    return {
        "vector_store": vector_store.get_stats(),
        "config": {
            "llm_model": config.llm.model_type,
            "embedding_model": config.embedding.model_type,
            "vector_store_type": config.vectorstore.store_type
        }
    }


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host=config.api.host,
        port=config.api.port,
        reload=False
    )

