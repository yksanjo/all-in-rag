# Installation

## Prerequisites

- Python 3.10 or higher
- CUDA (optional, for GPU acceleration)
- 16GB+ RAM recommended
- 20GB+ disk space for models

## Step 1: Clone Repository

```bash
git clone <repository-url>
cd enterprise-rag
```

## Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 4: Configure Environment

```bash
cp .env.example .env
# Edit .env with your configuration
```

## Step 5: Download Models

See the main README for model download instructions.

## Next Steps

- [Configuration](configuration.md)
- [Quick Start](quick-start.md)

