#!/usr/bin/env python3
"""
Load and test trained ML models
Validates that all ensemble models are working correctly
"""

import os
import sys
import asyncio
from pathlib import Path
import logging
import torch
import joblib
import json

# Add backend to path
current_dir = Path(__file__).parent.parent
sys.path.append(str(current_dir))

from backend.ml.model_manager import get_model_manager
from backend.models.ml_models import ModelManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_model_manager():
    """Test the existing ModelManager with trained models"""
    
    logger.info("Testing ModelManager with trained models...")
    
    # Get the model manager
    manager = get_model_manager()
    
    # Check model status
    status = manager.get_model_status()
    logger.info("Model Status:")
    for model_name, model_status in status.items():
        logger.info(f"  {model_name}: {model_status}")
    
    # Health check
    health = manager.health_check()
    logger.info(f"\nHealth Check Results:")
    logger.info(f"  Overall Healthy: {health['healthy']}")
    logger.info(f"  Total Models: {health['total_models']}")
    logger.info(f"  Loaded Models: {health['loaded_models']}")
    logger.info(f"  Fallback Models: {health['fallback_models']}")
    
    for model_name, model_health in health['models'].items():
        logger.info(f"  {model_name}: {model_health['health']}")
    
    # Test predictions
    logger.info("\nTesting predictions...")
    
    # Test each model individually
    test_models = ['lstm_price_predictor', 'xgboost_classifier', 'prophet_forecaster']
    
    for model_name in test_models:
        try:
            result = manager.predict(model_name, manager._get_test_data(model_name))
            logger.info(f"  {model_name}: ✅ Prediction successful - {result}")
        except Exception as e:
            logger.error(f"  {model_name}: ❌ Prediction failed - {e}")


def update_model_manager_config():
    """Update ModelManager to use our trained model files"""
    
    logger.info("Updating ModelManager configuration...")
    
    # Check if models exist
    models_dir = Path("/app/ml_models")
    
    expected_files = {
        "lstm_model.pt": "lstm_price_predictor",
        "xgboost_model.pkl": "xgboost_classifier", 
        "rf_model.joblib": "risk_assessor",
        "feature_scaler.pkl": "scaler",
        "model_registry.json": "registry"
    }
    
    logger.info("Checking for trained model files...")
    for filename, description in expected_files.items():
        filepath = models_dir / filename
        if filepath.exists():
            logger.info(f"  ✅ {description}: {filepath}")
        else:
            logger.warning(f"  ❌ {description}: Missing {filepath}")
    
    return True


async def create_new_model_manager():
    """Create and test new ModelManager instance"""
    
    logger.info("Creating new ModelManager instance...")
    
    # Import the updated ModelManager
    manager = ModelManager()
    await manager.load_models()
    
    # Test sample prediction
    logger.info("Testing sample predictions...")
    
    # Create sample data
    import pandas as pd
    import numpy as np
    
    sample_data = pd.DataFrame({
        'close': np.random.randn(100) * 10 + 100,
        'volume': np.random.randint(1000, 10000, 100)
    })
    sample_data.index = pd.date_range('2023-01-01', periods=100, freq='D')
    
    try:
        predictions = await manager.predict("AAPL", sample_data, horizon=5)
        logger.info(f"Sample predictions: {list(predictions.keys())}")
        
        for model_name, prediction in predictions.items():
            if prediction:
                logger.info(f"  {model_name}: Price={getattr(prediction, 'predicted_price', 'N/A'):.2f}, "
                          f"Return={getattr(prediction, 'predicted_return', 'N/A'):.4f}")
    except Exception as e:
        logger.error(f"Error testing predictions: {e}")


def check_model_files():
    """Check and validate model files"""
    
    models_dir = Path("/app/ml_models")
    
    if not models_dir.exists():
        logger.error(f"Models directory does not exist: {models_dir}")
        return False
    
    logger.info(f"Checking model files in {models_dir}...")
    
    # Check PyTorch models
    pytorch_models = ["lstm_model.pt", "transformer_model.pt"]
    for model_file in pytorch_models:
        model_path = models_dir / model_file
        if model_path.exists():
            try:
                # Try to load the model
                state_dict = torch.load(model_path, map_location='cpu')
                logger.info(f"  ✅ {model_file}: Valid PyTorch model ({len(state_dict)} parameters)")
            except Exception as e:
                logger.error(f"  ❌ {model_file}: Invalid PyTorch model - {e}")
        else:
            logger.warning(f"  ❌ {model_file}: Missing")
    
    # Check sklearn/joblib models
    sklearn_models = ["xgboost_model.pkl", "lightgbm_model.pkl", "rf_model.joblib"]
    for model_file in sklearn_models:
        model_path = models_dir / model_file
        if model_path.exists():
            try:
                # Try to load the model
                model = joblib.load(model_path)
                logger.info(f"  ✅ {model_file}: Valid sklearn model ({type(model).__name__})")
            except Exception as e:
                logger.error(f"  ❌ {model_file}: Invalid sklearn model - {e}")
        else:
            logger.warning(f"  ❌ {model_file}: Missing")
    
    # Check other files
    other_files = ["feature_scaler.pkl", "feature_names.pkl", "model_registry.json"]
    for other_file in other_files:
        file_path = models_dir / other_file
        if file_path.exists():
            logger.info(f"  ✅ {other_file}: Present")
            
            if other_file == "model_registry.json":
                try:
                    with open(file_path) as f:
                        registry = json.load(f)
                    logger.info(f"    Registry contains {len(registry.get('model_versions', {}))} models")
                except Exception as e:
                    logger.error(f"    Invalid JSON: {e}")
        else:
            logger.warning(f"  ❌ {other_file}: Missing")
    
    return True


async def main():
    """Main testing function"""
    
    logger.info("="*60)
    logger.info("TRAINED ML MODELS VALIDATION")
    logger.info("="*60)
    
    # 1. Check model files
    logger.info("\n1. Checking model files...")
    check_model_files()
    
    # 2. Update configuration
    logger.info("\n2. Updating configuration...")
    update_model_manager_config()
    
    # 3. Test existing ModelManager
    logger.info("\n3. Testing existing ModelManager...")
    try:
        await test_model_manager()
    except Exception as e:
        logger.error(f"Error testing existing ModelManager: {e}")
    
    # 4. Test new ModelManager instance
    logger.info("\n4. Testing new ModelManager instance...")
    try:
        await create_new_model_manager()
    except Exception as e:
        logger.error(f"Error testing new ModelManager: {e}")
    
    logger.info("\n" + "="*60)
    logger.info("VALIDATION COMPLETE")
    logger.info("="*60)
    
    # Summary
    models_dir = Path("/app/ml_models")
    if models_dir.exists():
        files = list(models_dir.glob("*"))
        logger.info(f"\nTotal model files: {len(files)}")
        logger.info("Next steps:")
        logger.info("1. Run the training script: python scripts/train_ml_models.py")
        logger.info("2. Start the application to test ensemble predictions")
        logger.info("3. Monitor model performance in production")
    else:
        logger.error("❌ Models directory not found. Run training script first!")


if __name__ == "__main__":
    asyncio.run(main())