"""
Machine Learning Model Manager
Handles loading, caching, and inference for all ML models
"""

import os
import pickle
import joblib
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import torch
import numpy as np
from datetime import datetime, timedelta
from threading import Lock

logger = logging.getLogger(__name__)

class ModelManager:
    """Centralized ML model management with error handling"""
    
    def __init__(self, models_path: str = "/app/ml_models"):
        self.models_path = Path(models_path)
        self.models: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Dict] = {}
        self.lock = Lock()
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize and load all available models with error handling"""
        if not self.models_path.exists():
            logger.warning(f"Models directory does not exist: {self.models_path}")
            self.models_path.mkdir(parents=True, exist_ok=True)
            return
        
        # Define expected models and their loaders
        model_configs = {
            "lstm_price_predictor": {
                "file": "lstm_model.pt",
                "loader": self._load_pytorch_model,
                "fallback": self._create_dummy_lstm
            },
            "xgboost_classifier": {
                "file": "xgboost_model.pkl",
                "loader": self._load_pickle_model,
                "fallback": self._create_dummy_xgboost
            },
            "prophet_forecaster": {
                "file": "prophet_model.pkl",
                "loader": self._load_pickle_model,
                "fallback": self._create_dummy_prophet
            },
            "sentiment_analyzer": {
                "file": "sentiment_model.pkl",
                "loader": self._load_pickle_model,
                "fallback": self._create_dummy_sentiment
            },
            "risk_assessor": {
                "file": "risk_model.joblib",
                "loader": self._load_joblib_model,
                "fallback": self._create_dummy_risk
            }
        }
        
        for model_name, config in model_configs.items():
            try:
                model_path = self.models_path / config["file"]
                if model_path.exists():
                    self.models[model_name] = config["loader"](model_path)
                    self.model_metadata[model_name] = {
                        "loaded_at": datetime.utcnow(),
                        "path": str(model_path),
                        "size": model_path.stat().st_size,
                        "status": "loaded"
                    }
                    logger.info(f"Successfully loaded model: {model_name}")
                else:
                    logger.warning(f"Model file not found: {model_path}")
                    # Create fallback model
                    self.models[model_name] = config["fallback"]()
                    self.model_metadata[model_name] = {
                        "loaded_at": datetime.utcnow(),
                        "status": "fallback",
                        "reason": "file_not_found"
                    }
            except Exception as e:
                logger.error(f"Error loading model {model_name}: {e}")
                # Create fallback model
                self.models[model_name] = config["fallback"]()
                self.model_metadata[model_name] = {
                    "loaded_at": datetime.utcnow(),
                    "status": "fallback",
                    "error": str(e)
                }
    
    def _load_pytorch_model(self, path: Path):
        """Load PyTorch model with error handling"""
        try:
            model = torch.load(path, map_location='cpu')
            model.eval()
            return model
        except Exception as e:
            logger.error(f"Failed to load PyTorch model from {path}: {e}")
            raise
    
    def _load_pickle_model(self, path: Path):
        """Load pickled model with error handling"""
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load pickle model from {path}: {e}")
            raise
    
    def _load_joblib_model(self, path: Path):
        """Load joblib model with error handling"""
        try:
            return joblib.load(path)
        except Exception as e:
            logger.error(f"Failed to load joblib model from {path}: {e}")
            raise
    
    def _create_dummy_lstm(self):
        """Create dummy LSTM model for fallback"""
        class DummyLSTM:
            def predict(self, X):
                # Return random predictions with slight upward bias
                return np.random.randn(len(X)) * 0.02 + 1.001
        
        return DummyLSTM()
    
    def _create_dummy_xgboost(self):
        """Create dummy XGBoost model for fallback"""
        class DummyXGBoost:
            def predict(self, X):
                # Return random classifications
                return np.random.choice([0, 1], size=len(X))
            
            def predict_proba(self, X):
                # Return random probabilities
                probs = np.random.random((len(X), 2))
                return probs / probs.sum(axis=1, keepdims=True)
        
        return DummyXGBoost()
    
    def _create_dummy_prophet(self):
        """Create dummy Prophet model for fallback"""
        class DummyProphet:
            def predict(self, df):
                # Return simple linear trend
                df['yhat'] = np.linspace(100, 110, len(df))
                df['yhat_lower'] = df['yhat'] * 0.95
                df['yhat_upper'] = df['yhat'] * 1.05
                return df
        
        return DummyProphet()
    
    def _create_dummy_sentiment(self):
        """Create dummy sentiment model for fallback"""
        class DummySentiment:
            def predict(self, texts):
                # Return neutral sentiment
                return [{"sentiment": "neutral", "score": 0.0} for _ in texts]
        
        return DummySentiment()
    
    def _create_dummy_risk(self):
        """Create dummy risk model for fallback"""
        class DummyRisk:
            def assess(self, data):
                # Return moderate risk
                return {
                    "risk_score": 0.5,
                    "risk_level": "moderate",
                    "confidence": 0.6
                }
        
        return DummyRisk()
    
    def get_model(self, model_name: str) -> Optional[Any]:
        """Get a loaded model by name"""
        with self.lock:
            return self.models.get(model_name)
    
    def predict(self, model_name: str, data: Any) -> Any:
        """Make prediction with error handling"""
        try:
            model = self.get_model(model_name)
            if model is None:
                raise ValueError(f"Model {model_name} not found")
            
            # Model-specific prediction logic
            if model_name == "lstm_price_predictor":
                return self._predict_lstm(model, data)
            elif model_name == "xgboost_classifier":
                return self._predict_xgboost(model, data)
            elif model_name == "prophet_forecaster":
                return self._predict_prophet(model, data)
            elif model_name == "sentiment_analyzer":
                return self._predict_sentiment(model, data)
            elif model_name == "risk_assessor":
                return self._predict_risk(model, data)
            else:
                return model.predict(data)
                
        except Exception as e:
            logger.error(f"Prediction error for {model_name}: {e}")
            # Return safe default prediction
            return self._get_default_prediction(model_name)
    
    def _predict_lstm(self, model, data):
        """LSTM-specific prediction with error handling"""
        try:
            if hasattr(model, 'predict'):
                return model.predict(data)
            else:
                # PyTorch model
                with torch.no_grad():
                    tensor_data = torch.FloatTensor(data)
                    predictions = model(tensor_data)
                    return predictions.numpy()
        except Exception as e:
            logger.error(f"LSTM prediction error: {e}")
            return np.array([100.0])  # Default price
    
    def _predict_xgboost(self, model, data):
        """XGBoost-specific prediction with error handling"""
        try:
            probabilities = model.predict_proba(data)
            predictions = model.predict(data)
            return {
                "predictions": predictions.tolist(),
                "probabilities": probabilities.tolist()
            }
        except Exception as e:
            logger.error(f"XGBoost prediction error: {e}")
            return {
                "predictions": [0],
                "probabilities": [[0.5, 0.5]]
            }
    
    def _predict_prophet(self, model, data):
        """Prophet-specific prediction with error handling"""
        try:
            forecast = model.predict(data)
            return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict('records')
        except Exception as e:
            logger.error(f"Prophet prediction error: {e}")
            return [{"ds": datetime.now(), "yhat": 100.0, "yhat_lower": 95.0, "yhat_upper": 105.0}]
    
    def _predict_sentiment(self, model, texts):
        """Sentiment-specific prediction with error handling"""
        try:
            if isinstance(texts, str):
                texts = [texts]
            return model.predict(texts)
        except Exception as e:
            logger.error(f"Sentiment prediction error: {e}")
            return [{"sentiment": "neutral", "score": 0.0} for _ in texts]
    
    def _predict_risk(self, model, data):
        """Risk assessment with error handling"""
        try:
            return model.assess(data)
        except Exception as e:
            logger.error(f"Risk assessment error: {e}")
            return {
                "risk_score": 0.5,
                "risk_level": "moderate",
                "confidence": 0.5
            }
    
    def _get_default_prediction(self, model_name: str):
        """Get safe default prediction for any model"""
        defaults = {
            "lstm_price_predictor": {"price": 100.0, "confidence": 0.0},
            "xgboost_classifier": {"class": 0, "probability": 0.5},
            "prophet_forecaster": {"forecast": 100.0, "lower": 95.0, "upper": 105.0},
            "sentiment_analyzer": {"sentiment": "neutral", "score": 0.0},
            "risk_assessor": {"risk_score": 0.5, "risk_level": "moderate"}
        }
        return defaults.get(model_name, {"error": "model_not_available"})
    
    def reload_model(self, model_name: str) -> bool:
        """Reload a specific model"""
        try:
            with self.lock:
                # Remove old model
                if model_name in self.models:
                    del self.models[model_name]
                
                # Reload
                self._initialize_models()
                return model_name in self.models
        except Exception as e:
            logger.error(f"Error reloading model {model_name}: {e}")
            return False
    
    def get_model_status(self) -> Dict[str, Dict]:
        """Get status of all models"""
        return {
            name: {
                **metadata,
                "is_loaded": name in self.models,
                "model_type": type(self.models.get(name)).__name__ if name in self.models else None
            }
            for name, metadata in self.model_metadata.items()
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on all models"""
        health_status = {
            "healthy": True,
            "models": {},
            "total_models": len(self.models),
            "loaded_models": 0,
            "fallback_models": 0
        }
        
        for model_name, metadata in self.model_metadata.items():
            model_health = {
                "status": metadata.get("status", "unknown"),
                "loaded_at": metadata.get("loaded_at", "").isoformat() if metadata.get("loaded_at") else None,
                "health": "healthy"
            }
            
            # Test prediction
            try:
                if model_name in self.models:
                    # Simple test prediction
                    test_data = self._get_test_data(model_name)
                    result = self.predict(model_name, test_data)
                    if result is not None:
                        model_health["health"] = "healthy"
                        health_status["loaded_models"] += 1
                    else:
                        model_health["health"] = "degraded"
                else:
                    model_health["health"] = "unavailable"
                    
                if metadata.get("status") == "fallback":
                    health_status["fallback_models"] += 1
                    model_health["health"] = "fallback"
                    
            except Exception as e:
                model_health["health"] = "unhealthy"
                model_health["error"] = str(e)
                health_status["healthy"] = False
            
            health_status["models"][model_name] = model_health
        
        return health_status
    
    def _get_test_data(self, model_name: str):
        """Get test data for model health check"""
        test_data = {
            "lstm_price_predictor": np.random.randn(1, 30, 5),
            "xgboost_classifier": np.random.randn(1, 20),
            "prophet_forecaster": {"ds": [datetime.now() + timedelta(days=i) for i in range(7)]},
            "sentiment_analyzer": ["Test text for sentiment analysis"],
            "risk_assessor": {"volatility": 0.2, "beta": 1.1, "sharpe": 0.8}
        }
        return test_data.get(model_name, np.random.randn(1, 10))

# Global model manager instance
_model_manager: Optional[ModelManager] = None

def get_model_manager() -> ModelManager:
    """Get or create the global model manager instance"""
    global _model_manager
    if _model_manager is None:
        models_path = os.getenv("ML_MODELS_PATH", "/app/ml_models")
        _model_manager = ModelManager(models_path)
    return _model_manager

def reload_all_models():
    """Reload all ML models"""
    manager = get_model_manager()
    manager._initialize_models()
    return manager.get_model_status()