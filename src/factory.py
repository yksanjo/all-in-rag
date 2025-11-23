"""
Factory module for easy pipeline initialization.
"""

from typing import Optional

from .config import config
from .models import LlamaCppModel, OllamaModel, BaseLLM
from .embeddings import TextEmbedder, ImageEmbedder
from .vectorstore import FAISSVectorStore, QdrantVectorStore, BaseVectorStore
from .retrieval import Retriever, Reranker
from .rag_pipeline import RAGPipeline


def create_llm(
    model_type: Optional[str] = None,
    model_path: Optional[str] = None
) -> BaseLLM:
    """
    Create LLM instance.
    
    Args:
        model_type: Model type (qwen2.5, llama3.1, mistral)
        model_path: Path to model file
        
    Returns:
        LLM instance
    """
    model_type = model_type or config.llm.model_type
    model_path = model_path or config.llm.model_path
    
    if model_type in ["qwen2.5", "llama3.1", "mistral"]:
        return LlamaCppModel(model_path=model_path)
    else:
        # Try Ollama as fallback
        return OllamaModel(model_name=model_type)


def create_vector_store(
    store_type: Optional[str] = None,
    dimension: Optional[int] = None
) -> BaseVectorStore:
    """
    Create vector store instance.
    
    Args:
        store_type: Store type (faiss, qdrant)
        dimension: Embedding dimension
        
    Returns:
        Vector store instance
    """
    store_type = store_type or config.vectorstore.store_type
    dimension = dimension or config.vectorstore.dimension
    
    if store_type == "faiss":
        return FAISSVectorStore(
            dimension=dimension,
            index_path=config.vectorstore.store_path
        )
    elif store_type == "qdrant":
        return QdrantVectorStore(
            dimension=dimension,
            path=config.vectorstore.store_path
        )
    else:
        raise ValueError(f"Unsupported vector store type: {store_type}")


def create_pipeline(
    llm: Optional[BaseLLM] = None,
    vector_store: Optional[BaseVectorStore] = None,
    text_embedder: Optional[TextEmbedder] = None,
    image_embedder: Optional[ImageEmbedder] = None,
    retriever: Optional[Retriever] = None,
    reranker: Optional[Reranker] = None
) -> RAGPipeline:
    """
    Create RAG pipeline with all components.
    
    Args:
        llm: Optional LLM instance
        vector_store: Optional vector store instance
        text_embedder: Optional text embedder
        image_embedder: Optional image embedder
        retriever: Optional retriever
        reranker: Optional reranker
        
    Returns:
        RAG pipeline instance
    """
    # Create LLM if not provided
    if llm is None:
        llm = create_llm()
    
    # Create embedders if not provided
    if text_embedder is None:
        text_embedder = TextEmbedder()
    
    if image_embedder is None and config.image_rag.enable_image_rag:
        image_embedder = ImageEmbedder()
    
    # Create vector store if not provided
    if vector_store is None:
        vector_store = create_vector_store(dimension=text_embedder.get_dimension())
    
    # Create retriever if not provided
    if retriever is None:
        retriever = Retriever(
            vector_store=vector_store,
            text_embedder=text_embedder,
            image_embedder=image_embedder
        )
    
    # Create reranker if not provided and enabled
    if reranker is None and config.rag_pipeline.enable_reranking:
        reranker = Reranker()
    
    # Create pipeline
    return RAGPipeline(
        llm=llm,
        retriever=retriever,
        reranker=reranker
    )

