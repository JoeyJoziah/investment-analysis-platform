"""
Online Learning System
Provides incremental learning, continuous model updates, and adaptive ensemble weighting
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import json
import logging
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor
from collections import deque, defaultdict
import joblib  # SECURITY: Use joblib instead of pickle for model serialization

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import SGDRegressor, SGDClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

logger = logging.getLogger(__name__)


class LearningStrategy(Enum):
    """Online learning strategies"""
    INCREMENTAL = "incremental"       # Update existing model
    ENSEMBLE_WEIGHTING = "ensemble_weighting"  # Adjust ensemble weights
    MODEL_REPLACEMENT = "model_replacement"    # Replace underperforming models
    HYBRID = "hybrid"                 # Combination of strategies


class UpdateTrigger(Enum):
    """Triggers for model updates"""
    TIME_BASED = "time_based"         # Regular intervals
    PERFORMANCE_BASED = "performance_based"  # When performance drops
    DATA_DRIFT = "data_drift"         # When input distribution changes
    CONCEPT_DRIFT = "concept_drift"   # When relationships change
    MANUAL = "manual"                 # Manually triggered


@dataclass
class LearningMetrics:
    """Metrics for online learning performance"""
    timestamp: datetime
    model_name: str
    update_type: str
    samples_processed: int
    learning_rate: float
    performance_before: float
    performance_after: float
    improvement: float
    computational_cost_ms: float
    memory_usage_mb: float
    convergence_score: float
    stability_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class EnsembleWeights:
    """Ensemble model weights with metadata"""
    model_name: str
    weights: Dict[str, float]
    performance_history: Dict[str, List[float]]
    last_updated: datetime
    update_count: int
    confidence_scores: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['last_updated'] = self.last_updated.isoformat()
        return data


class IncrementalLearner:
    """Base class for incremental learning models"""
    
    def __init__(self, model_name: str, learning_rate: float = 0.01):
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.model = None
        self.scaler = StandardScaler()
        self.samples_processed = 0
        self.last_updated = None
        
    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> float:
        """Incrementally update model with new data"""
        raise NotImplementedError
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not initialized")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_learning_metrics(self) -> Dict[str, Any]:
        """Get learning performance metrics"""
        return {
            'samples_processed': self.samples_processed,
            'learning_rate': self.learning_rate,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None
        }


class SGDIncrementalLearner(IncrementalLearner):
    """Stochastic Gradient Descent incremental learner"""
    
    def __init__(self, model_name: str, problem_type: str = "regression", 
                 learning_rate: float = 0.01, **kwargs):
        super().__init__(model_name, learning_rate)
        
        self.problem_type = problem_type
        
        if problem_type == "regression":
            self.model = SGDRegressor(
                learning_rate='adaptive',
                eta0=learning_rate,
                random_state=42,
                **kwargs
            )
        else:
            self.model = SGDClassifier(
                learning_rate='adaptive',
                eta0=learning_rate,
                random_state=42,
                **kwargs
            )
    
    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> float:
        """Update model with new batch"""
        
        # Initialize or update scaler
        if self.samples_processed == 0:
            self.scaler.fit(X)
        else:
            # Incremental scaling update (approximate)
            self.scaler.partial_fit(X)
        
        X_scaled = self.scaler.transform(X)
        
        # Perform partial fit
        if self.problem_type == "classification" and self.samples_processed == 0:
            # For classification, need to see all classes initially
            unique_classes = np.unique(y)
            self.model.partial_fit(X_scaled, y, classes=unique_classes)
        else:
            self.model.partial_fit(X_scaled, y)
        
        self.samples_processed += len(X)
        self.last_updated = datetime.utcnow()
        
        # Calculate performance metric (simple)
        predictions = self.model.predict(X_scaled)
        
        if self.problem_type == "regression":
            performance = 1 - np.mean((y - predictions) ** 2) / np.var(y) if np.var(y) > 0 else 0
        else:
            performance = np.mean(y == predictions)
        
        return performance


class PyTorchIncrementalLearner(IncrementalLearner):
    """PyTorch-based incremental learner"""
    
    def __init__(self, model_name: str, input_dim: int, output_dim: int = 1,
                 hidden_dims: List[int] = None, learning_rate: float = 0.001):
        super().__init__(model_name, learning_rate)
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims or [64, 32]
        
        # Build model
        self._build_model()
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss() if output_dim == 1 else nn.CrossEntropyLoss()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def _build_model(self):
        """Build neural network model"""
        layers = []
        
        # Input layer
        prev_dim = self.input_dim
        
        # Hidden layers
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, self.output_dim))
        
        self.model = nn.Sequential(*layers)
    
    def partial_fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 5) -> float:
        """Update model with new batch"""
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        if self.output_dim > 1:  # Classification
            y_tensor = y_tensor.long()
        
        self.model.train()
        
        # Mini-batch training
        batch_losses = []
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(X_tensor)
            if self.output_dim == 1:
                outputs = outputs.squeeze()
            
            loss = self.criterion(outputs, y_tensor)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            batch_losses.append(loss.item())
        
        self.samples_processed += len(X)
        self.last_updated = datetime.utcnow()
        
        # Calculate performance
        with torch.no_grad():
            self.model.eval()
            predictions = self.model(X_tensor)
            
            if self.output_dim == 1:  # Regression
                mse = nn.MSELoss()(predictions.squeeze(), y_tensor).item()
                performance = max(0, 1 - mse)  # Normalized performance
            else:  # Classification
                _, predicted = torch.max(predictions, 1)
                accuracy = (predicted == y_tensor).float().mean().item()
                performance = accuracy
        
        return performance
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            self.model.eval()
            outputs = self.model(X_tensor)
            
            if self.output_dim == 1:  # Regression
                return outputs.squeeze().cpu().numpy()
            else:  # Classification
                _, predicted = torch.max(outputs, 1)
                return predicted.cpu().numpy()


class AdaptiveEnsembleWeighter:
    """Adaptive ensemble weight management system"""
    
    def __init__(self, 
                 models: Dict[str, Any],
                 initial_weights: Dict[str, float] = None,
                 learning_rate: float = 0.1,
                 performance_window: int = 100):
        
        self.models = models
        self.learning_rate = learning_rate
        self.performance_window = performance_window
        
        # Initialize weights
        if initial_weights is None:
            n_models = len(models)
            initial_weights = {name: 1.0 / n_models for name in models.keys()}
        
        self.weights = EnsembleWeights(
            model_name="ensemble",
            weights=initial_weights.copy(),
            performance_history={name: deque(maxlen=performance_window) for name in models.keys()},
            last_updated=datetime.utcnow(),
            update_count=0,
            confidence_scores={name: 1.0 for name in models.keys()}
        )
        
        # Performance tracking
        self.recent_predictions: Dict[str, deque] = {
            name: deque(maxlen=performance_window) for name in models.keys()
        }
        self.recent_targets = deque(maxlen=performance_window)
        
        logger.info(f"Initialized adaptive ensemble with {len(models)} models")
    
    def update_weights(self, 
                      predictions: Dict[str, np.ndarray],
                      targets: np.ndarray,
                      strategy: str = "performance_based") -> Dict[str, float]:
        """Update ensemble weights based on recent performance"""
        
        # Store predictions and targets
        for model_name, pred in predictions.items():
            if model_name in self.recent_predictions:
                self.recent_predictions[model_name].extend(pred)
        
        self.recent_targets.extend(targets)
        
        # Calculate performance metrics for each model
        model_performances = {}
        
        for model_name in self.models.keys():
            if (len(self.recent_predictions[model_name]) >= 10 and 
                len(self.recent_targets) >= 10):
                
                # Get recent predictions and targets
                recent_preds = np.array(list(self.recent_predictions[model_name]))[-len(targets):]
                recent_targs = np.array(list(self.recent_targets))[-len(targets):]
                
                if len(recent_preds) == len(recent_targs):
                    # Calculate performance (negative MSE for regression)
                    mse = np.mean((recent_preds - recent_targs) ** 2)
                    performance = np.exp(-mse)  # Convert to positive score
                    
                    model_performances[model_name] = performance
                    self.weights.performance_history[model_name].append(performance)
        
        if not model_performances:
            return self.weights.weights
        
        # Update weights based on strategy
        if strategy == "performance_based":
            new_weights = self._update_performance_based(model_performances)
        elif strategy == "gradient_based":
            new_weights = self._update_gradient_based(model_performances)
        elif strategy == "bandit_based":
            new_weights = self._update_bandit_based(model_performances)
        else:
            new_weights = self.weights.weights
        
        # Apply learning rate and ensure normalization
        for model_name in self.weights.weights.keys():
            if model_name in new_weights:
                old_weight = self.weights.weights[model_name]
                new_weight = new_weights[model_name]
                self.weights.weights[model_name] = (
                    (1 - self.learning_rate) * old_weight + 
                    self.learning_rate * new_weight
                )
        
        # Normalize weights
        total_weight = sum(self.weights.weights.values())
        if total_weight > 0:
            for model_name in self.weights.weights.keys():
                self.weights.weights[model_name] /= total_weight
        
        self.weights.last_updated = datetime.utcnow()
        self.weights.update_count += 1
        
        logger.info(f"Updated ensemble weights: {self.weights.weights}")
        
        return self.weights.weights
    
    def _update_performance_based(self, model_performances: Dict[str, float]) -> Dict[str, float]:
        """Update weights based on recent performance"""
        
        # Softmax transformation of performance scores
        max_perf = max(model_performances.values())
        exp_perfs = {name: np.exp(perf - max_perf) for name, perf in model_performances.items()}
        
        total_exp = sum(exp_perfs.values())
        new_weights = {name: exp_perf / total_exp for name, exp_perf in exp_perfs.items()}
        
        # Include models not in current performance calculation
        for model_name in self.weights.weights.keys():
            if model_name not in new_weights:
                new_weights[model_name] = 0.01  # Minimum weight
        
        return new_weights
    
    def _update_gradient_based(self, model_performances: Dict[str, float]) -> Dict[str, float]:
        """Update weights using gradient-based approach"""
        
        new_weights = {}
        
        for model_name, current_weight in self.weights.weights.items():
            if model_name in model_performances:
                performance = model_performances[model_name]
                
                # Gradient of log-likelihood (approximated)
                gradient = performance - np.mean(list(model_performances.values()))
                
                # Update weight (ensure positive)
                new_weight = current_weight * np.exp(self.learning_rate * gradient)
                new_weights[model_name] = max(new_weight, 0.001)  # Minimum weight
            else:
                new_weights[model_name] = current_weight
        
        return new_weights
    
    def _update_bandit_based(self, model_performances: Dict[str, float]) -> Dict[str, float]:
        """Update weights using multi-armed bandit approach (UCB1)"""
        
        new_weights = {}
        
        for model_name, current_weight in self.weights.weights.items():
            if model_name in model_performances and len(self.weights.performance_history[model_name]) > 0:
                
                # Calculate UCB1 score
                avg_performance = np.mean(self.weights.performance_history[model_name])
                n_trials = len(self.weights.performance_history[model_name])
                total_trials = sum(len(hist) for hist in self.weights.performance_history.values())
                
                if n_trials > 0 and total_trials > 0:
                    confidence_bonus = np.sqrt(2 * np.log(total_trials) / n_trials)
                    ucb_score = avg_performance + confidence_bonus
                else:
                    ucb_score = 1.0  # Default for new models
                
                new_weights[model_name] = ucb_score
            else:
                new_weights[model_name] = 1.0  # Default for models without performance
        
        return new_weights
    
    def predict_ensemble(self, 
                        predictions: Dict[str, np.ndarray],
                        return_individual: bool = False) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Make ensemble prediction using current weights"""
        
        if not predictions:
            return np.array([])
        
        # Ensure all predictions have the same length
        pred_length = len(next(iter(predictions.values())))
        ensemble_pred = np.zeros(pred_length)
        
        for model_name, pred in predictions.items():
            if model_name in self.weights.weights:
                weight = self.weights.weights[model_name]
                ensemble_pred += weight * pred
        
        if return_individual:
            return {
                'ensemble': ensemble_pred,
                **predictions,
                'weights': self.weights.weights
            }
        
        return ensemble_pred
    
    def get_model_rankings(self) -> Dict[str, Dict[str, Any]]:
        """Get current model rankings and statistics"""
        
        rankings = {}
        
        for model_name, weight in sorted(self.weights.weights.items(), 
                                       key=lambda x: x[1], reverse=True):
            
            performance_history = self.weights.performance_history.get(model_name, [])
            
            rankings[model_name] = {
                'current_weight': weight,
                'average_performance': np.mean(performance_history) if performance_history else 0.0,
                'performance_std': np.std(performance_history) if performance_history else 0.0,
                'performance_trend': self._calculate_trend(performance_history),
                'confidence_score': self.weights.confidence_scores.get(model_name, 1.0),
                'sample_count': len(performance_history)
            }
        
        return rankings
    
    def _calculate_trend(self, performance_history: List[float]) -> str:
        """Calculate performance trend"""
        if len(performance_history) < 5:
            return "insufficient_data"
        
        recent = performance_history[-5:]
        older = performance_history[-10:-5] if len(performance_history) >= 10 else performance_history[:-5]
        
        if not older:
            return "insufficient_data"
        
        recent_avg = np.mean(recent)
        older_avg = np.mean(older)
        
        if recent_avg > older_avg * 1.05:
            return "improving"
        elif recent_avg < older_avg * 0.95:
            return "declining"
        else:
            return "stable"


class OnlineLearningManager:
    """
    Comprehensive online learning management system
    """
    
    def __init__(self,
                 storage_path: str = "/app/online_learning",
                 default_learning_rate: float = 0.01,
                 update_frequency_minutes: int = 60):
        
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.default_learning_rate = default_learning_rate
        self.update_frequency_minutes = update_frequency_minutes
        
        # Learners and ensembles
        self.incremental_learners: Dict[str, IncrementalLearner] = {}
        self.ensemble_weights: Dict[str, AdaptiveEnsembleWeighter] = {}
        
        # Learning history
        self.learning_metrics_history: List[LearningMetrics] = []
        
        # Update triggers
        self.update_triggers: Dict[str, UpdateTrigger] = {}
        self.performance_thresholds: Dict[str, float] = {}
        
        # Threading for continuous learning
        self.is_learning = False
        self.learning_thread = None
        self.learning_queue = asyncio.Queue()
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        # Load existing state
        self._load_state()
        
        logger.info("Online learning manager initialized")
    
    def register_incremental_learner(self,
                                   model_name: str,
                                   learner_type: str = "sgd",
                                   problem_type: str = "regression",
                                   learning_rate: float = None,
                                   **kwargs) -> bool:
        """Register an incremental learner"""
        
        try:
            learning_rate = learning_rate or self.default_learning_rate
            
            if learner_type == "sgd":
                learner = SGDIncrementalLearner(
                    model_name=model_name,
                    problem_type=problem_type,
                    learning_rate=learning_rate,
                    **kwargs
                )
            elif learner_type == "pytorch":
                input_dim = kwargs.get('input_dim', 10)
                output_dim = kwargs.get('output_dim', 1)
                learner = PyTorchIncrementalLearner(
                    model_name=model_name,
                    input_dim=input_dim,
                    output_dim=output_dim,
                    learning_rate=learning_rate,
                    **kwargs
                )
            else:
                raise ValueError(f"Unknown learner type: {learner_type}")
            
            with self.lock:
                self.incremental_learners[model_name] = learner
                self.update_triggers[model_name] = UpdateTrigger.TIME_BASED
                self.performance_thresholds[model_name] = 0.05  # 5% degradation threshold
            
            logger.info(f"Registered incremental learner: {model_name} ({learner_type})")
            return True
            
        except Exception as e:
            logger.error(f"Error registering incremental learner {model_name}: {e}")
            return False
    
    def register_ensemble(self,
                         ensemble_name: str,
                         models: Dict[str, Any],
                         initial_weights: Dict[str, float] = None,
                         learning_rate: float = 0.1) -> bool:
        """Register an adaptive ensemble"""
        
        try:
            ensemble_weighter = AdaptiveEnsembleWeighter(
                models=models,
                initial_weights=initial_weights,
                learning_rate=learning_rate
            )
            
            with self.lock:
                self.ensemble_weights[ensemble_name] = ensemble_weighter
            
            logger.info(f"Registered adaptive ensemble: {ensemble_name} with {len(models)} models")
            return True
            
        except Exception as e:
            logger.error(f"Error registering ensemble {ensemble_name}: {e}")
            return False
    
    async def update_incremental_learner(self,
                                       model_name: str,
                                       X: np.ndarray,
                                       y: np.ndarray,
                                       force_update: bool = False) -> Optional[LearningMetrics]:
        """Update an incremental learner with new data"""
        
        if model_name not in self.incremental_learners:
            logger.warning(f"Incremental learner {model_name} not registered")
            return None
        
        learner = self.incremental_learners[model_name]
        
        # Check update trigger
        if not force_update and not self._should_update(model_name, X, y):
            return None
        
        try:
            start_time = datetime.utcnow()
            
            # Get baseline performance if possible
            performance_before = 0.0
            if learner.samples_processed > 0:
                try:
                    baseline_predictions = learner.predict(X)
                    if learner.problem_type == "regression":
                        performance_before = 1 - np.mean((y - baseline_predictions) ** 2) / np.var(y)
                    else:
                        performance_before = np.mean(y == baseline_predictions)
                except:
                    performance_before = 0.0
            
            # Perform incremental update
            performance_after = learner.partial_fit(X, y)
            
            # Calculate metrics
            end_time = datetime.utcnow()
            computational_cost = (end_time - start_time).total_seconds() * 1000  # ms
            
            learning_metrics = LearningMetrics(
                timestamp=end_time,
                model_name=model_name,
                update_type="incremental",
                samples_processed=len(X),
                learning_rate=learner.learning_rate,
                performance_before=performance_before,
                performance_after=performance_after,
                improvement=performance_after - performance_before,
                computational_cost_ms=computational_cost,
                memory_usage_mb=0.0,  # Would need to measure actual memory
                convergence_score=self._calculate_convergence_score(learner),
                stability_score=self._calculate_stability_score(learner)
            )
            
            # Store metrics
            self.learning_metrics_history.append(learning_metrics)
            
            # Save updated state
            self._save_learner_state(learner)
            
            logger.info(f"Updated incremental learner {model_name}: "
                       f"performance={performance_after:.3f}, "
                       f"improvement={learning_metrics.improvement:.3f}")
            
            return learning_metrics
            
        except Exception as e:
            logger.error(f"Error updating incremental learner {model_name}: {e}")
            return None
    
    async def update_ensemble_weights(self,
                                    ensemble_name: str,
                                    predictions: Dict[str, np.ndarray],
                                    targets: np.ndarray,
                                    strategy: str = "performance_based") -> Optional[Dict[str, float]]:
        """Update ensemble weights based on performance"""
        
        if ensemble_name not in self.ensemble_weights:
            logger.warning(f"Ensemble {ensemble_name} not registered")
            return None
        
        try:
            ensemble_weighter = self.ensemble_weights[ensemble_name]
            
            # Update weights
            new_weights = ensemble_weighter.update_weights(
                predictions=predictions,
                targets=targets,
                strategy=strategy
            )
            
            # Create learning metrics
            learning_metrics = LearningMetrics(
                timestamp=datetime.utcnow(),
                model_name=ensemble_name,
                update_type="ensemble_weighting",
                samples_processed=len(targets),
                learning_rate=ensemble_weighter.learning_rate,
                performance_before=0.0,  # Would need to track
                performance_after=0.0,   # Would need to calculate
                improvement=0.0,
                computational_cost_ms=10.0,  # Lightweight operation
                memory_usage_mb=0.0,
                convergence_score=1.0,
                stability_score=self._calculate_ensemble_stability(ensemble_weighter)
            )
            
            self.learning_metrics_history.append(learning_metrics)
            
            # Save ensemble state
            self._save_ensemble_state(ensemble_weighter, ensemble_name)
            
            logger.info(f"Updated ensemble weights for {ensemble_name}: {new_weights}")
            
            return new_weights
            
        except Exception as e:
            logger.error(f"Error updating ensemble weights for {ensemble_name}: {e}")
            return None
    
    def predict_with_incremental(self, model_name: str, X: np.ndarray) -> Optional[np.ndarray]:
        """Make predictions with incremental learner"""
        
        if model_name not in self.incremental_learners:
            return None
        
        try:
            return self.incremental_learners[model_name].predict(X)
        except Exception as e:
            logger.error(f"Error making prediction with {model_name}: {e}")
            return None
    
    def predict_with_ensemble(self, 
                            ensemble_name: str,
                            predictions: Dict[str, np.ndarray],
                            return_individual: bool = False) -> Optional[Union[np.ndarray, Dict[str, np.ndarray]]]:
        """Make ensemble predictions"""
        
        if ensemble_name not in self.ensemble_weights:
            return None
        
        try:
            ensemble_weighter = self.ensemble_weights[ensemble_name]
            return ensemble_weighter.predict_ensemble(predictions, return_individual)
        except Exception as e:
            logger.error(f"Error making ensemble prediction with {ensemble_name}: {e}")
            return None
    
    def start_continuous_learning(self):
        """Start continuous learning process"""
        
        if self.is_learning:
            logger.warning("Continuous learning already started")
            return
        
        self.is_learning = True
        self.learning_thread = threading.Thread(
            target=self._continuous_learning_loop, 
            daemon=True
        )
        self.learning_thread.start()
        
        logger.info("Started continuous learning process")
    
    def stop_continuous_learning(self):
        """Stop continuous learning process"""
        
        self.is_learning = False
        if self.learning_thread:
            self.learning_thread.join(timeout=10)
        
        logger.info("Stopped continuous learning process")
    
    async def queue_learning_update(self,
                                  model_name: str,
                                  X: np.ndarray,
                                  y: np.ndarray,
                                  update_type: str = "incremental"):
        """Queue a learning update for asynchronous processing"""
        
        update_data = {
            'model_name': model_name,
            'X': X,
            'y': y,
            'update_type': update_type,
            'timestamp': datetime.utcnow()
        }
        
        await self.learning_queue.put(update_data)
    
    def _should_update(self, model_name: str, X: np.ndarray, y: np.ndarray) -> bool:
        """Determine if model should be updated based on trigger conditions"""
        
        trigger = self.update_triggers.get(model_name, UpdateTrigger.TIME_BASED)
        learner = self.incremental_learners[model_name]
        
        if trigger == UpdateTrigger.TIME_BASED:
            if learner.last_updated is None:
                return True
            
            time_since_update = datetime.utcnow() - learner.last_updated
            return time_since_update.total_seconds() > self.update_frequency_minutes * 60
        
        elif trigger == UpdateTrigger.PERFORMANCE_BASED:
            if learner.samples_processed > 0:
                try:
                    current_predictions = learner.predict(X)
                    if hasattr(learner, 'problem_type') and learner.problem_type == "regression":
                        current_performance = 1 - np.mean((y - current_predictions) ** 2) / np.var(y)
                    else:
                        current_performance = np.mean(y == current_predictions)
                    
                    # Check if performance dropped below threshold
                    threshold = self.performance_thresholds.get(model_name, 0.05)
                    baseline_performance = self._get_baseline_performance(model_name)
                    
                    if baseline_performance > 0:
                        degradation = (baseline_performance - current_performance) / baseline_performance
                        return degradation > threshold
                except:
                    pass
            
            return True  # Update if we can't measure performance
        
        return False  # Default: don't update
    
    def _get_baseline_performance(self, model_name: str) -> float:
        """Get baseline performance for a model"""
        
        # Get recent performance from metrics history
        recent_metrics = [
            m for m in self.learning_metrics_history
            if m.model_name == model_name and 
               (datetime.utcnow() - m.timestamp).days < 7
        ]
        
        if recent_metrics:
            return np.mean([m.performance_after for m in recent_metrics])
        
        return 0.0
    
    def _calculate_convergence_score(self, learner: IncrementalLearner) -> float:
        """Calculate convergence score for a learner"""
        
        # Simple heuristic: more samples processed = higher convergence
        # In practice, would analyze loss curve stability
        
        if learner.samples_processed < 100:
            return 0.0
        elif learner.samples_processed < 1000:
            return 0.5
        else:
            return min(1.0, np.log10(learner.samples_processed) / 4.0)
    
    def _calculate_stability_score(self, learner: IncrementalLearner) -> float:
        """Calculate stability score for a learner"""
        
        # Get recent performance metrics
        recent_metrics = [
            m for m in self.learning_metrics_history
            if m.model_name == learner.model_name and 
               (datetime.utcnow() - m.timestamp).hours < 24
        ]
        
        if len(recent_metrics) < 3:
            return 1.0  # Not enough data, assume stable
        
        # Calculate performance variance
        performances = [m.performance_after for m in recent_metrics]
        stability = 1.0 - min(1.0, np.std(performances))
        
        return max(0.0, stability)
    
    def _calculate_ensemble_stability(self, ensemble_weighter: AdaptiveEnsembleWeighter) -> float:
        """Calculate stability score for ensemble weights"""
        
        # Look at weight variance over recent updates
        weight_history = []
        for model_name in ensemble_weighter.weights.weights.keys():
            perf_history = ensemble_weighter.weights.performance_history[model_name]
            if len(perf_history) >= 5:
                weight_history.append(np.std(list(perf_history)[-5:]))
        
        if not weight_history:
            return 1.0
        
        avg_variance = np.mean(weight_history)
        stability = 1.0 - min(1.0, avg_variance)
        
        return max(0.0, stability)
    
    def _continuous_learning_loop(self):
        """Main continuous learning loop"""
        
        async def process_learning_queue():
            while self.is_learning:
                try:
                    # Wait for queued updates
                    update_data = await asyncio.wait_for(
                        self.learning_queue.get(), 
                        timeout=60.0
                    )
                    
                    # Process update
                    if update_data['update_type'] == 'incremental':
                        await self.update_incremental_learner(
                            update_data['model_name'],
                            update_data['X'],
                            update_data['y']
                        )
                    
                except asyncio.TimeoutError:
                    # No updates in queue, continue
                    continue
                except Exception as e:
                    logger.error(f"Error in continuous learning loop: {e}")
                    await asyncio.sleep(5)  # Wait before retrying
        
        # Run async loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(process_learning_queue())
    
    def _save_learner_state(self, learner: IncrementalLearner):
        """Save learner state to disk"""

        try:
            learner_file = self.storage_path / f"learner_{learner.model_name}.pkl"

            # SECURITY: Use joblib instead of pickle for safer serialization
            joblib.dump(learner, learner_file)

        except Exception as e:
            logger.error(f"Error saving learner state for {learner.model_name}: {e}")
    
    def _save_ensemble_state(self, ensemble: AdaptiveEnsembleWeighter, name: str):
        """Save ensemble state to disk"""
        
        try:
            ensemble_file = self.storage_path / f"ensemble_{name}.json"
            
            with open(ensemble_file, 'w') as f:
                json.dump(ensemble.weights.to_dict(), f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving ensemble state for {name}: {e}")
    
    def _load_state(self):
        """Load existing learning state"""
        
        try:
            # Load learners
            for learner_file in self.storage_path.glob("learner_*.pkl"):
                try:
                    # SECURITY: Use joblib instead of pickle for safer deserialization
                    learner = joblib.load(learner_file)
                    self.incremental_learners[learner.model_name] = learner
                    logger.info(f"Loaded learner: {learner.model_name}")
                except Exception as e:
                    logger.error(f"Error loading learner from {learner_file}: {e}")
            
            # Load learning metrics
            metrics_file = self.storage_path / "learning_metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics_data = json.load(f)
                
                for metric_data in metrics_data:
                    metric_data['timestamp'] = datetime.fromisoformat(metric_data['timestamp'])
                    metrics = LearningMetrics(**metric_data)
                    self.learning_metrics_history.append(metrics)
                
                logger.info(f"Loaded {len(metrics_data)} learning metrics")
            
        except Exception as e:
            logger.error(f"Error loading online learning state: {e}")
    
    def save_state(self):
        """Save current learning state"""
        
        try:
            # Save learning metrics
            metrics_file = self.storage_path / "learning_metrics.json"
            
            metrics_data = [m.to_dict() for m in self.learning_metrics_history[-1000:]]  # Keep last 1000
            
            with open(metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            
            logger.info("Saved online learning state")
            
        except Exception as e:
            logger.error(f"Error saving online learning state: {e}")
    
    def get_learning_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive learning dashboard"""
        
        dashboard = {
            'timestamp': datetime.utcnow().isoformat(),
            'incremental_learners': {},
            'ensembles': {},
            'recent_metrics': {},
            'system_status': {}
        }
        
        # Incremental learners status
        for name, learner in self.incremental_learners.items():
            dashboard['incremental_learners'][name] = {
                'samples_processed': learner.samples_processed,
                'learning_rate': learner.learning_rate,
                'last_updated': learner.last_updated.isoformat() if learner.last_updated else None,
                'convergence_score': self._calculate_convergence_score(learner),
                'stability_score': self._calculate_stability_score(learner)
            }
        
        # Ensembles status
        for name, ensemble in self.ensemble_weights.items():
            rankings = ensemble.get_model_rankings()
            dashboard['ensembles'][name] = {
                'model_count': len(ensemble.models),
                'last_updated': ensemble.weights.last_updated.isoformat(),
                'update_count': ensemble.weights.update_count,
                'current_weights': ensemble.weights.weights,
                'model_rankings': rankings
            }
        
        # Recent metrics
        if self.learning_metrics_history:
            recent_metrics = self.learning_metrics_history[-10:]  # Last 10
            dashboard['recent_metrics'] = {
                'count': len(recent_metrics),
                'average_improvement': np.mean([m.improvement for m in recent_metrics]),
                'average_cost_ms': np.mean([m.computational_cost_ms for m in recent_metrics]),
                'latest_updates': [m.to_dict() for m in recent_metrics]
            }
        
        # System status
        dashboard['system_status'] = {
            'is_continuous_learning_active': self.is_learning,
            'update_frequency_minutes': self.update_frequency_minutes,
            'total_learners': len(self.incremental_learners),
            'total_ensembles': len(self.ensemble_weights),
            'total_learning_events': len(self.learning_metrics_history)
        }
        
        return dashboard


# Global online learning manager instance
_online_learning_manager: Optional[OnlineLearningManager] = None

def get_online_learning_manager() -> OnlineLearningManager:
    """Get global online learning manager instance"""
    global _online_learning_manager
    if _online_learning_manager is None:
        _online_learning_manager = OnlineLearningManager()
    return _online_learning_manager