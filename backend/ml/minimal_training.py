#!/usr/bin/env python3
"""
Minimal ML Training Script for Testing
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import pickle

def create_sample_model():
    """Create a minimal sample model"""
    print("Creating minimal sample model...")
    
    # Create directories
    os.makedirs('backend/ml_models', exist_ok=True)
    os.makedirs('backend/ml_logs', exist_ok=True)
    
    # Create simple dummy model (using sklearn)
    from sklearn.linear_model import LinearRegression
    
    # Generate sample data
    X = np.random.randn(100, 5)
    y = np.random.randn(100)
    
    # Train model
    model = LinearRegression()
    model.fit(X, y)
    
    # Save model
    model_path = 'backend/ml_models/sample_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'model_type': 'LinearRegression',
        'model_path': model_path,
        'features': 5,
        'samples': 100,
        'score': float(model.score(X, y))
    }
    
    metadata_path = 'backend/ml_logs/sample_model_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Model saved to {model_path}")
    print(f"✓ Metadata saved to {metadata_path}")
    print(f"✓ Model score: {metadata['score']:.3f}")
    
    return metadata

if __name__ == "__main__":
    result = create_sample_model()
    print("Training completed successfully!")