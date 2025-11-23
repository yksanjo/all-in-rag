# Quick Start Guide

Get up and running with Enterprise RAG in 5 minutes.

## Prerequisites

- Python 3.10+
- 16GB+ RAM
- 20GB+ disk space

## Step 1: Install

```bash
cd enterprise-rag
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Step 2: Configure

```bash
cp .env.example .env
# Edit .env - at minimum, set your model paths
```

## Step 3: Download Models

### Option A: Use Script

```bash
bash scripts/download_models.sh
```

### Option B: Manual Download

1. **Embedding Model** (required):
   ```bash
   python -c "from transformers import AutoModel; AutoModel.from_pretrained('BAAI/bge-m3').save_pretrained('data/models/bge-m3')"
   ```

2. **LLM Model** (required):
   - Download a GGUF model from HuggingFace
   - Place in `data/models/`
   - Update `LLM_MODEL_PATH` in `.env`

## Step 4: Start API

```bash
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000
```

## Step 5: Start UI (Optional)

In a new terminal:

```bash
streamlit run ui/app.py
```

## Step 6: Upload Documents

Visit `http://localhost:8501` and upload documents, or use the API:

```bash
curl -X POST "http://localhost:8000/upload" \
  -F "files=@your_document.pdf"
```

## Step 7: Query

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is this document about?"}'
```

## Troubleshooting

### Model Not Found

- Check `LLM_MODEL_PATH` in `.env` points to a valid GGUF file
- Ensure embedding model is downloaded to `data/models/`

### Out of Memory

- Reduce `LLM_N_GPU_LAYERS` in `.env` (set to 0 for CPU-only)
- Use smaller models
- Reduce `CHUNK_SIZE` and `TOP_K_RETRIEVAL`

### API Not Starting

- Check port 8000 is not in use
- Verify all dependencies are installed
- Check logs for error messages

## Next Steps

- Read the [full README](README.md)
- Check [documentation](docs/index.md)
- Explore the [API endpoints](docs/api/endpoints.md)

