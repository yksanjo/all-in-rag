# Quick Start

## Start the API Server

```bash
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000
```

## Start the Streamlit UI

```bash
streamlit run ui/app.py
```

## Upload Documents

Use the UI or API to upload documents:

```bash
curl -X POST "http://localhost:8000/upload" \
  -F "files=@document.pdf"
```

## Query the System

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the main topic?"}'
```

## Next Steps

- [API Reference](../api/endpoints.md)
- [Architecture](../architecture/overview.md)

