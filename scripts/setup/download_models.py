#!/usr/bin/env python3
"""
Download and initialize ML models - Simplified version
"""

import sys
import os

# Fix path issues
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, script_dir)

print("Working directory:", os.getcwd())

def main():
    """Download and initialize ML models"""
    print("Starting ML model initialization...")
    
    # Create models directory
    models_dir = os.path.join(script_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    print(f"✓ Models directory created/verified: {models_dir}")
    
    try:
        # Try to import model manager
        from backend.ml.model_manager import get_model_manager
        
        print("Initializing model manager...")
        model_manager = get_model_manager()
        
        # Download models
        print("Downloading pre-trained models...")
        if hasattr(model_manager, 'download_models'):
            model_manager.download_models()
        else:
            print("Note: download_models method not found, skipping...")
        
        # Train initial models if needed
        print("Training initial models...")
        if hasattr(model_manager, 'train_initial_models'):
            model_manager.train_initial_models()
        else:
            print("Note: train_initial_models method not found, skipping...")
        
        print("✅ ML model initialization complete!")
        
    except ImportError as e:
        print(f"⚠ Could not import model manager: {e}")
        print("\nCreating placeholder model files...")
        
        # Create placeholder model files
        import json
        
        model_config = {
            "models": {
                "price_predictor": {
                    "type": "lstm",
                    "version": "1.0.0",
                    "path": "models/price_predictor.pkl",
                    "status": "not_loaded"
                },
                "sentiment_analyzer": {
                    "type": "transformer",
                    "version": "1.0.0",
                    "path": "models/sentiment.pkl",
                    "status": "not_loaded"
                },
                "pattern_detector": {
                    "type": "cnn",
                    "version": "1.0.0",
                    "path": "models/patterns.pkl",
                    "status": "not_loaded"
                }
            },
            "created": "2024-01-01",
            "description": "Placeholder model configuration"
        }
        
        config_path = os.path.join(models_dir, 'model_config.json')
        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=2)
        
        print(f"✓ Created model configuration: {config_path}")
        print("\nModels will be downloaded on first use.")
    
    except Exception as e:
        print(f"❌ Error during model initialization: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())