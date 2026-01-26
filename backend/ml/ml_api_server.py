#!/usr/bin/env python3
"""
ML API Server
Serves ML models via FastAPI on port 8001
"""

import os
import sys
import logging
import uvicorn
import joblib
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="ML Inference API",
    description="Machine Learning Model Inference Service",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
loaded_models = {}
model_metadata = {}

class PredictionRequest(BaseModel):
    features: List[float]
    model_name: Optional[str] = "sample_model"

class PredictionResponse(BaseModel):
    prediction: float
    model_name: str
    timestamp: str
    confidence: Optional[float] = None

class ModelInfo(BaseModel):
    name: str
    type: str
    features: int
    score: float
    loaded_at: str

def load_model(model_name: str):
    """Load a model from disk"""
    try:
        model_path = Path(f"backend/ml_models/{model_name}.pkl")
        metadata_path = Path(f"backend/ml_logs/{model_name}_metadata.json")
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # SECURITY: Use joblib instead of pickle for safer deserialization
        model = joblib.load(model_path)
        
        # Load metadata if available
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        loaded_models[model_name] = model
        model_metadata[model_name] = {
            **metadata,
            'loaded_at': datetime.now().isoformat()
        }
        
        logger.info(f"Model {model_name} loaded successfully")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        raise

def ensure_model_loaded(model_name: str):
    """Ensure model is loaded"""
    if model_name not in loaded_models:
        load_model(model_name)
    return loaded_models[model_name]

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    logger.info("Starting ML API Server...")
    
    # Try to load default models
    models_dir = Path("backend/ml_models")
    if models_dir.exists():
        for model_file in models_dir.glob("*.pkl"):
            model_name = model_file.stem
            try:
                load_model(model_name)
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {e}")
    
    logger.info(f"ML API Server started with {len(loaded_models)} models")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "ML Inference API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "loaded_models": list(loaded_models.keys())
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": len(loaded_models),
        "models": list(loaded_models.keys())
    }

@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List available models"""
    models = []
    for name, metadata in model_metadata.items():
        models.append(ModelInfo(
            name=name,
            type=metadata.get('model_type', 'unknown'),
            features=metadata.get('features', 0),
            score=metadata.get('score', 0.0),
            loaded_at=metadata.get('loaded_at', 'unknown')
        ))
    return models

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make prediction"""
    try:
        model = ensure_model_loaded(request.model_name)
        
        # Prepare input
        input_array = np.array(request.features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(input_array)[0]
        
        # Calculate confidence if possible
        confidence = None
        if hasattr(model, 'predict_proba'):
            try:
                proba = model.predict_proba(input_array)[0]
                confidence = float(np.max(proba))
            except:
                pass
        
        return PredictionResponse(
            prediction=float(prediction),
            model_name=request.model_name,
            timestamp=datetime.now().isoformat(),
            confidence=confidence
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/{model_name}/load")
async def load_model_endpoint(model_name: str):
    """Load a specific model"""
    try:
        load_model(model_name)
        return {
            "message": f"Model {model_name} loaded successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.delete("/models/{model_name}")
async def unload_model(model_name: str):
    """Unload a model from memory"""
    if model_name in loaded_models:
        del loaded_models[model_name]
        del model_metadata[model_name]
        logger.info(f"Model {model_name} unloaded")
        return {
            "message": f"Model {model_name} unloaded successfully",
            "timestamp": datetime.now().isoformat()
        }
    else:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

@app.get("/models/{model_name}/info")
async def get_model_info(model_name: str):
    """Get model information"""
    if model_name not in model_metadata:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    return model_metadata[model_name]

@app.post("/retrain")
async def trigger_retrain(background_tasks: BackgroundTasks):
    """Trigger model retraining"""
    try:
        background_tasks.add_task(run_training_task)
        return {
            "message": "Training task queued",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def run_training_task():
    """Background task for training"""
    try:
        import subprocess
        result = subprocess.run(
            ['python3', 'backend/ml/minimal_training.py'],
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes
        )
        
        if result.returncode == 0:
            logger.info("Background training completed successfully")
        else:
            logger.error(f"Background training failed: {result.stderr}")
            
    except Exception as e:
        logger.error(f"Background training error: {e}")

if __name__ == "__main__":
    # Set environment
    os.environ["ML_MODELS_PATH"] = "backend/ml_models"
    os.environ["ML_LOGS_PATH"] = "backend/ml_logs"
    
    # Run server
    uvicorn.run(
        "ml_api_server:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info"
    )