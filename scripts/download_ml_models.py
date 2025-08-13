#!/usr/bin/env python3
"""
Download and initialize ML models
"""

import os
import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from backend.ml.model_manager import get_model_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Download and initialize all ML models"""
    logger.info("Starting ML model initialization...")
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Initialize model manager
    model_manager = get_model_manager()
    
    # Download models
    logger.info("Downloading pre-trained models...")
    model_manager.download_models()
    
    # Train initial models if needed
    logger.info("Training initial models...")
    model_manager.train_initial_models()
    
    logger.info("ML model initialization complete!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
