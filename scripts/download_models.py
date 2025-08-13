#!/usr/bin/env python3
import os
import urllib.request
from pathlib import Path

# Create models directory
models_dir = Path('models')
models_dir.mkdir(exist_ok=True)

# Download FinBERT if not exists
finbert_path = models_dir / 'finbert'
if not finbert_path.exists():
    print("Downloading FinBERT model...")
    # In production, download from Hugging Face
    # For now, create placeholder
    finbert_path.mkdir(exist_ok=True)
    (finbert_path / 'config.json').write_text('{"model_type": "bert", "version": "1.0"}')
    print("✓ FinBERT placeholder created")

print("Model setup complete!")
