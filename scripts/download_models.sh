#!/bin/bash
# Script to download models for Enterprise RAG system

set -e

MODELS_DIR="./data/models"
mkdir -p "$MODELS_DIR"

echo "Downloading models for Enterprise RAG system..."
echo "Note: This script requires huggingface-cli or manual download"
echo ""

# BGE-M3 Embedding Model
echo "Downloading BGE-M3 embedding model..."
python -c "
from transformers import AutoModel, AutoTokenizer
import os

model_name = 'BAAI/bge-m3'
save_path = '$MODELS_DIR/bge-m3'

print(f'Downloading {model_name} to {save_path}...')
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

os.makedirs(save_path, exist_ok=True)
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)
print('BGE-M3 downloaded successfully!')
"

# GTE-Large Embedding Model (optional)
read -p "Download GTE-Large model? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Downloading GTE-Large embedding model..."
    python -c "
from transformers import AutoModel, AutoTokenizer
import os

model_name = 'thenlper/gte-large'
save_path = '$MODELS_DIR/gte-large'

print(f'Downloading {model_name} to {save_path}...')
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

os.makedirs(save_path, exist_ok=True)
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)
print('GTE-Large downloaded successfully!')
"
fi

echo ""
echo "Model download complete!"
echo ""
echo "For LLM models (GGUF format), please download manually:"
echo "  - Qwen2.5-7B-Instruct: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF"
echo "  - Llama3.1-8B-Instruct: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct-GGUF"
echo "  - Mistral-Nemo-12B: https://huggingface.co/mistralai/Mistral-Nemo-12B-GGUF"
echo ""
echo "Place GGUF files in $MODELS_DIR and update LLM_MODEL_PATH in .env"

