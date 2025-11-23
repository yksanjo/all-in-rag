# Configuration

The system is configured via environment variables in the `.env` file.

## LLM Configuration

```env
LLM_MODEL_TYPE=qwen2.5  # Options: qwen2.5, llama3.1, mistral
LLM_MODEL_PATH=./data/models/qwen2.5-7b-instruct.gguf
LLM_CONTEXT_SIZE=4096
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=2048
LLM_N_GPU_LAYERS=35  # Set to 0 for CPU-only
```

## Embedding Configuration

```env
EMBEDDING_MODEL_TYPE=bge-m3  # Options: bge-m3, gte-large
EMBEDDING_MODEL_PATH=./data/models/bge-m3
EMBEDDING_DEVICE=cuda  # Options: cuda, cpu
```

## Vector Store Configuration

```env
VECTOR_STORE_TYPE=faiss  # Options: faiss, qdrant
VECTOR_STORE_PATH=./data/indices/faiss_index
VECTOR_DIMENSION=1024
TOP_K_RETRIEVAL=5
```

## RAG Pipeline Configuration

```env
CHUNK_SIZE=512
CHUNK_OVERLAP=50
ENABLE_RERANKING=false
```

## API Configuration

```env
API_HOST=0.0.0.0
API_PORT=8000
```

