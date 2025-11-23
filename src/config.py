"""
Configuration management using pydantic-settings.
All configuration is loaded from environment variables.
"""

from typing import Literal, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class LLMConfig(BaseSettings):
    """LLM model configuration."""
    
    model_type: Literal["qwen2.5", "llama3.1", "mistral"] = Field(
        default="qwen2.5",
        alias="LLM_MODEL_TYPE"
    )
    model_path: str = Field(
        default="./data/models/qwen2.5-7b-instruct.gguf",
        alias="LLM_MODEL_PATH"
    )
    context_size: int = Field(default=4096, alias="LLM_CONTEXT_SIZE")
    temperature: float = Field(default=0.7, alias="LLM_TEMPERATURE")
    max_tokens: int = Field(default=2048, alias="LLM_MAX_TOKENS")
    n_gpu_layers: int = Field(default=35, alias="LLM_N_GPU_LAYERS")
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class EmbeddingConfig(BaseSettings):
    """Embedding model configuration."""
    
    model_type: Literal["bge-m3", "gte-large", "openclip"] = Field(
        default="bge-m3",
        alias="EMBEDDING_MODEL_TYPE"
    )
    model_path: str = Field(
        default="./data/models/bge-m3",
        alias="EMBEDDING_MODEL_PATH"
    )
    device: Literal["cuda", "cpu"] = Field(
        default="cuda",
        alias="EMBEDDING_DEVICE"
    )
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class VectorStoreConfig(BaseSettings):
    """Vector store configuration."""
    
    store_type: Literal["faiss", "qdrant"] = Field(
        default="faiss",
        alias="VECTOR_STORE_TYPE"
    )
    store_path: str = Field(
        default="./data/indices/faiss_index",
        alias="VECTOR_STORE_PATH"
    )
    dimension: int = Field(default=1024, alias="VECTOR_DIMENSION")
    top_k: int = Field(default=5, alias="TOP_K_RETRIEVAL")
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class RAGPipelineConfig(BaseSettings):
    """RAG pipeline configuration."""
    
    chunk_size: int = Field(default=512, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, alias="CHUNK_OVERLAP")
    enable_reranking: bool = Field(
        default=False,
        alias="ENABLE_RERANKING"
    )
    reranker_model_path: Optional[str] = Field(
        default=None,
        alias="RERANKER_MODEL_PATH"
    )
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class ImageRAGConfig(BaseSettings):
    """Image RAG configuration."""
    
    enable_image_rag: bool = Field(
        default=True,
        alias="ENABLE_IMAGE_RAG"
    )
    embedding_model: str = Field(
        default="openclip",
        alias="IMAGE_EMBEDDING_MODEL"
    )
    embedding_path: str = Field(
        default="./data/models/openclip",
        alias="IMAGE_EMBEDDING_PATH"
    )
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class APIConfig(BaseSettings):
    """API server configuration."""
    
    host: str = Field(default="0.0.0.0", alias="API_HOST")
    port: int = Field(default=8000, alias="API_PORT")
    workers: int = Field(default=1, alias="API_WORKERS")
    log_level: str = Field(default="info", alias="API_LOG_LEVEL")
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class UIConfig(BaseSettings):
    """UI configuration."""
    
    streamlit_port: int = Field(default=8501, alias="STREAMLIT_PORT")
    streamlit_host: str = Field(default="0.0.0.0", alias="STREAMLIT_HOST")
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class DataConfig(BaseSettings):
    """Data paths configuration."""
    
    documents_path: str = Field(
        default="./data/documents",
        alias="DOCUMENTS_PATH"
    )
    indices_path: str = Field(
        default="./data/indices",
        alias="INDICES_PATH"
    )
    models_path: str = Field(
        default="./data/models",
        alias="MODELS_PATH"
    )
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class LoggingConfig(BaseSettings):
    """Logging configuration."""
    
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_file: str = Field(
        default="./logs/rag_system.log",
        alias="LOG_FILE"
    )
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class SecurityConfig(BaseSettings):
    """Security configuration (for future RBAC)."""
    
    enable_auth: bool = Field(default=False, alias="ENABLE_AUTH")
    secret_key: str = Field(
        default="your-secret-key-here-change-in-production",
        alias="SECRET_KEY"
    )
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class Config(BaseSettings):
    """Main configuration class combining all configs."""
    
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vectorstore: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    rag_pipeline: RAGPipelineConfig = Field(default_factory=RAGPipelineConfig)
    image_rag: ImageRAGConfig = Field(default_factory=ImageRAGConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    
    @classmethod
    def load(cls) -> "Config":
        """Load configuration from environment."""
        return cls()


# Global config instance
config = Config.load()

