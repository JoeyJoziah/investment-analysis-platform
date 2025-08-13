"""
Ensemble Voting Classifier for Stock Predictions
Combines multiple models for robust predictions
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.base import BaseEstimator
import joblib
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class StockPredictionEnsemble:
    """Ensemble model for stock price predictions"""
    
    def __init__(self, models: Dict[str, BaseEstimator], voting: str = 'soft'):
        """
        Initialize ensemble with multiple models
        
        Args:
            models: Dictionary of model_name -> model instance
            voting: 'hard' or 'soft' voting
        """
        self.models = models
        self.voting = voting
        self.weights = self._calculate_model_weights()
        self.ensemble_classifier = None
        self.ensemble_regressor = None
        self.feature_importance = {}
        
    def _calculate_model_weights(self) -> List[float]:
        """Calculate weights for each model based on past performance"""
        # In production, load historical performance metrics
        # For now, use equal weights
        return [1.0] * len(self.models)
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
             task_type: str = 'classification') -> None:
        """Train the ensemble model"""
        
        if task_type == 'classification':
            # Create voting classifier
            estimators = [(name, model) for name, model in self.models.items()]
            self.ensemble_classifier = VotingClassifier(
                estimators=estimators,
                voting=self.voting,
                weights=self.weights
            )
            self.ensemble_classifier.fit(X_train, y_train)
            
        elif task_type == 'regression':
            # Create voting regressor
            estimators = [(name, model) for name, model in self.models.items()]
            self.ensemble_regressor = VotingRegressor(
                estimators=estimators,
                weights=self.weights
            )
            self.ensemble_regressor.fit(X_train, y_train)
        
        # Calculate feature importance
        self._calculate_feature_importance(X_train)
    
    def predict(self, X: pd.DataFrame, task_type: str = 'classification') -> np.array:
        """Make predictions using ensemble"""
        
        if task_type == 'classification':
            if self.ensemble_classifier is None:
                raise ValueError("Classifier not trained. Call train() first.")
            return self.ensemble_classifier.predict(X)
            
        elif task_type == 'regression':
            if self.ensemble_regressor is None:
                raise ValueError("Regressor not trained. Call train() first.")
            return self.ensemble_regressor.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.array:
        """Get probability predictions (classification only)"""
        if self.ensemble_classifier is None:
            raise ValueError("Classifier not trained. Call train() first.")
        
        if self.voting == 'soft':
            return self.ensemble_classifier.predict_proba(X)
        else:
            # For hard voting, calculate pseudo-probabilities
            predictions = []
            for name, model in self.models.items():
                if hasattr(model, 'predict_proba'):
                    predictions.append(model.predict_proba(X))
            
            if predictions:
                return np.mean(predictions, axis=0)
            else:
                raise ValueError("No models support probability predictions")
    
    def get_individual_predictions(self, X: pd.DataFrame) -> Dict[str, np.array]:
        """Get predictions from each individual model"""
        predictions = {}
        
        for name, model in self.models.items():
            try:
                predictions[name] = model.predict(X)
            except Exception as e:
                logger.error(f"Error getting prediction from {name}: {e}")
                predictions[name] = None
        
        return predictions
    
    def _calculate_feature_importance(self, X: pd.DataFrame) -> None:
        """Calculate aggregate feature importance across models"""
        importance_scores = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                importance_scores[name] = model.feature_importances_
            elif hasattr(model, 'coef_'):
                # Linear models
                importance_scores[name] = np.abs(model.coef_).flatten()
        
        if importance_scores:
            # Average importance across models
            avg_importance = np.mean(list(importance_scores.values()), axis=0)
            self.feature_importance = dict(zip(X.columns, avg_importance))
            
            # Sort by importance
            self.feature_importance = dict(
                sorted(self.feature_importance.items(), 
                      key=lambda x: x[1], 
                      reverse=True)
            )
    
    def save_ensemble(self, filepath: str) -> None:
        """Save ensemble model to disk"""
        ensemble_data = {
            'models': self.models,
            'weights': self.weights,
            'voting': self.voting,
            'feature_importance': self.feature_importance,
            'ensemble_classifier': self.ensemble_classifier,
            'ensemble_regressor': self.ensemble_regressor,
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(ensemble_data, filepath)
        logger.info(f"Ensemble model saved to {filepath}")
    
    def load_ensemble(self, filepath: str) -> None:
        """Load ensemble model from disk"""
        ensemble_data = joblib.load(filepath)
        
        self.models = ensemble_data['models']
        self.weights = ensemble_data['weights']
        self.voting = ensemble_data['voting']
        self.feature_importance = ensemble_data['feature_importance']
        self.ensemble_classifier = ensemble_data.get('ensemble_classifier')
        self.ensemble_regressor = ensemble_data.get('ensemble_regressor')
        
        logger.info(f"Ensemble model loaded from {filepath}")
    
    def update_weights(self, performance_metrics: Dict[str, float]) -> None:
        """Update model weights based on recent performance"""
        
        # Normalize performance metrics to weights
        total_performance = sum(performance_metrics.values())
        
        if total_performance > 0:
            new_weights = []
            for name in self.models.keys():
                weight = performance_metrics.get(name, 0) / total_performance
                new_weights.append(weight)
            
            self.weights = new_weights
            logger.info(f"Updated ensemble weights: {dict(zip(self.models.keys(), self.weights))}")
    
    def get_confidence_score(self, X: pd.DataFrame) -> np.array:
        """Calculate confidence scores for predictions"""
        
        if self.voting == 'soft' and self.ensemble_classifier:
            # Use probability as confidence
            proba = self.predict_proba(X)
            return np.max(proba, axis=1)
        else:
            # Calculate agreement between models
            predictions = self.get_individual_predictions(X)
            predictions_array = np.array([pred for pred in predictions.values() if pred is not None])
            
            # Calculate mode (most common prediction) for each sample
            confidence_scores = []
            for i in range(predictions_array.shape[1]):
                values, counts = np.unique(predictions_array[:, i], return_counts=True)
                max_count = np.max(counts)
                confidence = max_count / len(predictions_array)
                confidence_scores.append(confidence)
            
            return np.array(confidence_scores)
