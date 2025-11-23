#!/usr/bin/env python3
"""
Project initialization script.
Creates necessary directories and checks configuration.
"""

import os
from pathlib import Path


def create_directories():
    """Create necessary directories."""
    directories = [
        "data/documents",
        "data/indices",
        "data/models",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")


def check_env_file():
    """Check if .env file exists."""
    if not Path(".env").exists():
        print("⚠ .env file not found!")
        print("  Please copy .env.example to .env and configure it:")
        print("  cp .env.example .env")
        return False
    else:
        print("✓ .env file exists")
        return True


def check_models():
    """Check if model directories exist."""
    models_dir = Path("data/models")
    if not models_dir.exists():
        print("⚠ Models directory not found")
        return False
    
    # Check for common model files
    has_models = False
    for item in models_dir.iterdir():
        if item.is_dir() or item.suffix == ".gguf":
            has_models = True
            break
    
    if not has_models:
        print("⚠ No models found in data/models/")
        print("  Please download models:")
        print("  - LLM model (GGUF format)")
        print("  - Embedding model (BGE-M3 or GTE-Large)")
        return False
    else:
        print("✓ Models directory contains files")
        return True


def main():
    """Main initialization function."""
    print("Enterprise RAG System - Project Initialization")
    print("=" * 50)
    print()
    
    # Create directories
    print("Creating directories...")
    create_directories()
    print()
    
    # Check configuration
    print("Checking configuration...")
    env_ok = check_env_file()
    print()
    
    # Check models
    print("Checking models...")
    models_ok = check_models()
    print()
    
    # Summary
    print("=" * 50)
    if env_ok and models_ok:
        print("✓ Project is ready!")
        print()
        print("Next steps:")
        print("  1. Review and update .env configuration")
        print("  2. Start the API: python -m uvicorn api.server:app")
        print("  3. Start the UI: streamlit run ui/app.py")
    else:
        print("⚠ Project setup incomplete")
        print()
        if not env_ok:
            print("  - Create and configure .env file")
        if not models_ok:
            print("  - Download required models")
        print()
        print("See QUICKSTART.md for detailed instructions")


if __name__ == "__main__":
    main()

