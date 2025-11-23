# Enterprise Offline RAG System

A private, fully offline Retrieval-Augmented Generation framework for companies.

> **Note**: This is an enterprise-ready fork of [Datawhale's All-in-RAG](https://github.com/datawhalechina/all-in-rag), redesigned as a production-grade, fully offline RAG system that companies can deploy within their own infrastructure.

## 🚀 Overview

This project is a production-grade, offline-capable RAG framework that companies can run inside their own firewall. It transforms the educational All-in-RAG tutorial into a complete, deployable enterprise solution.

**No internet. No external APIs. 100% private. 100% local.**

### Features

- ✅ **100% Offline** - No external API calls
- ✅ **Private** - All data stays within your infrastructure
- ✅ **Local LLMs** - Supports Llama, Qwen, Mistral (GGUF format)
- ✅ **Local Embeddings** - BGE-M3, GTE-Large, OpenCLIP
- ✅ **Local Vector DB** - FAISS or Qdrant
- ✅ **Text + Image RAG** - Multimodal document retrieval
- ✅ **Modular Architecture** - Easy to customize and extend
- ✅ **FastAPI Backend** - RESTful API for integration
- ✅ **Streamlit UI** - User-friendly interface
- ✅ **Docker Support** - Easy deployment

## 📁 Project Structure

```
enterprise-rag/
├── src/
│   ├── ingest/          # Document loading and chunking
│   ├── embeddings/      # Text and image embedding models
│   ├── vectorstore/     # FAISS and Qdrant implementations
│   ├── retrieval/       # Document retrieval and reranking
│   ├── models/          # Local LLM models (llama.cpp, Ollama)
│   ├── rag_pipeline/    # Main RAG orchestration
│   ├── evaluation/      # Evaluation metrics and tools
│   └── config.py        # Configuration management
├── api/
│   └── server.py        # FastAPI server
├── ui/
│   └── app.py           # Streamlit UI
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── docs/                # Documentation
├── data/
│   ├── documents/       # Uploaded documents
│   ├── indices/         # Vector store indices
│   └── models/          # Model files
├── tests/               # Unit tests
├── .env.example         # Environment configuration template
├── requirements.txt     # Python dependencies
└── README.md
```

## 🛠️ Installation

### Prerequisites

- Python 3.10+
- CUDA (optional, for GPU acceleration)
- 16GB+ RAM recommended
- 20GB+ disk space for models

### Step 1: Clone and Setup

```bash
cd enterprise-rag
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2: Configure Environment

```bash
cp .env.example .env
# Edit .env with your configuration
```

### Step 3: Download Models

#### LLM Models (GGUF format)

Download one of the following models:

- **Qwen2.5-7B-Instruct**: [HuggingFace](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF)
- **Llama3.1-8B-Instruct**: [HuggingFace](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct-GGUF)
- **Mistral-Nemo-12B**: [HuggingFace](https://huggingface.co/mistralai/Mistral-Nemo-12B-GGUF)

Place the `.gguf` file in `data/models/` and update `LLM_MODEL_PATH` in `.env`.

#### Embedding Models

```bash
# BGE-M3
python -c "from transformers import AutoModel; AutoModel.from_pretrained('BAAI/bge-m3').save_pretrained('data/models/bge-m3')"

# GTE-Large
python -c "from transformers import AutoModel; AutoModel.from_pretrained('thenlper/gte-large').save_pretrained('data/models/gte-large')"
```

Update `EMBEDDING_MODEL_PATH` in `.env`.

## 🚀 Quick Start

### Option 1: Run with Docker

```bash
# Build and start services
docker-compose up -d

# View logs
docker-compose logs -f

# Access API: http://localhost:8000
# Access UI: http://localhost:8501
```

### Option 2: Run Locally

#### Start API Server

```bash
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000
```

#### Start Streamlit UI (separate terminal)

```bash
streamlit run ui/app.py
```

## 📖 Usage

### API Endpoints

#### Query

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the main topic of the documents?",
    "top_k": 5
  }'
```

#### Upload Documents

```bash
curl -X POST "http://localhost:8000/upload" \
  -F "files=@document1.pdf" \
  -F "files=@document2.txt"
```

#### Health Check

```bash
curl http://localhost:8000/health
```

### Python SDK

```python
from src.rag_pipeline import RAGPipeline
from src.models import LlamaCppModel
from src.embeddings import TextEmbedder
from src.vectorstore import FAISSVectorStore
from src.retrieval import Retriever

# Initialize components
llm = LlamaCppModel()
embedder = TextEmbedder()
vector_store = FAISSVectorStore()
retriever = Retriever(vector_store, embedder)

# Create pipeline
pipeline = RAGPipeline(llm, retriever)

# Query
result = pipeline.query("What is RAG?")
print(result["answer"])
```

## 🔧 Configuration

Key configuration options in `.env`:

```env
# LLM Configuration
LLM_MODEL_TYPE=qwen2.5
LLM_MODEL_PATH=./data/models/qwen2.5-7b-instruct.gguf
LLM_N_GPU_LAYERS=35  # Set to 0 for CPU-only

# Embedding Configuration
EMBEDDING_MODEL_TYPE=bge-m3
EMBEDDING_DEVICE=cuda  # or cpu

# Vector Store
VECTOR_STORE_TYPE=faiss
VECTOR_STORE_PATH=./data/indices/faiss_index

# RAG Pipeline
CHUNK_SIZE=512
CHUNK_OVERLAP=50
TOP_K_RETRIEVAL=5
```

## 🧪 Evaluation

Run evaluation on your RAG system:

```python
from src.evaluation import RAGEvaluator

evaluator = RAGEvaluator(pipeline)

# Evaluate single query
result = evaluator.evaluate_query(
    query="What is machine learning?",
    relevant_doc_ids=["doc1", "doc2"]
)

# Evaluate batch
results = evaluator.evaluate_batch(
    queries=["query1", "query2"],
    relevant_doc_ids_list=[["doc1"], ["doc2"]]
)
```

## 🐳 Docker Deployment

### Build Image

```bash
docker build -t enterprise-rag .
```

### Run Container

```bash
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/.env:/app/.env \
  enterprise-rag
```

## 📚 Documentation

Generate documentation:

```bash
mkdocs serve
```

Access at `http://localhost:8000`

## 🧩 Architecture

### Pipeline Flow

1. **Ingestion**: Documents are loaded and chunked
2. **Embedding**: Chunks are embedded using local models
3. **Indexing**: Embeddings are stored in vector database
4. **Retrieval**: Query is embedded and similar chunks are retrieved
5. **Reranking** (optional): Retrieved chunks are reranked
6. **Generation**: LLM generates answer using retrieved context

### Component Overview

- **Document Loader**: Supports PDF, DOCX, TXT, images
- **Chunker**: Intelligent text chunking with overlap
- **Embedders**: BGE-M3, GTE-Large (text), OpenCLIP (images)
- **Vector Stores**: FAISS (fast, local) or Qdrant (advanced features)
- **LLMs**: llama.cpp (GGUF) or Ollama
- **Reranker**: Optional BGE reranker for improved precision

## 🔒 Security

- All processing happens locally
- No data leaves your infrastructure
- Configurable authentication (future RBAC support)
- Audit logging support

## 🛣️ Roadmap

### v1 - Foundation ✅
- [x] Repo restructure
- [x] Basic text RAG
- [x] Local FAISS index
- [x] Local LLM models
- [x] Document upload
- [x] /query endpoint
- [x] Basic Streamlit UI

### v2 - Multimodal + Evaluation (In Progress)
- [ ] Image RAG with OpenCLIP
- [ ] Metadata store
- [ ] Chunking improvements
- [ ] Reranking model
- [ ] Offline evaluation suite
- [ ] Unit tests
- [ ] Model adapters
- [ ] CI pipeline

### v3 - Enterprise Ready
- [ ] RBAC (user roles)
- [ ] Audit logs
- [ ] Multi-tenant index
- [ ] Fine-grained permissions
- [ ] Model hot-swapping
- [ ] Caching layer
- [ ] Document versioning
- [ ] Production monitoring
- [ ] Scalability tuning

## 🤝 Contributing

This is an enterprise template. Customize as needed for your organization.

## 📄 License

[Your License Here]

## 🙏 Acknowledgments

- **Based on [Datawhale's All-in-RAG](https://github.com/datawhalechina/all-in-rag)** - Original RAG tutorial and learning materials
- Uses llama.cpp for efficient LLM inference
- FAISS for fast similarity search
- HuggingFace Transformers for embeddings
- Qdrant for advanced vector database features

## 📞 Support

For issues and questions, please open an issue in the repository.

---

**Built for enterprises that value privacy and control.**

