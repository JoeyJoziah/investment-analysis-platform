"""
Model Performance Monitoring System
Provides model drift detection, performance degradation monitoring, and automated retraining
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import json
import logging
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    mean_squared_error, mean_absolute_error, r2_score
)
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    ERROR = "error"


class ModelHealth(Enum):
    """Model health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILING = "failing"
    UNKNOWN = "unknown"


class DriftType(Enum):
    """Types of model drift"""
    DATA_DRIFT = "data_drift"          # Input feature distribution changes
    PREDICTION_DRIFT = "prediction_drift"  # Model output distribution changes
    CONCEPT_DRIFT = "concept_drift"     # True relationship changes
    PERFORMANCE_DRIFT = "performance_drift"  # Model accuracy degradation


@dataclass
class PerformanceMetrics:
    """Model performance metrics snapshot"""
    timestamp: datetime
    model_name: str
    model_version: str
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    mse: Optional[float] = None
    mae: Optional[float] = None
    r2_score: Optional[float] = None
    auc_roc: Optional[float] = None
    directional_accuracy: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    sample_size: int = 0
    prediction_latency_ms: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class DriftDetectionResult:
    """Drift detection result"""
    timestamp: datetime
    model_name: str
    drift_type: DriftType
    drift_score: float
    threshold: float
    is_drift_detected: bool
    feature_drifts: Dict[str, float]
    statistical_test_results: Dict[str, Any]
    confidence_level: float
    sample_size: int
    reference_period: str
    detection_period: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['drift_type'] = self.drift_type.value
        return data


@dataclass
class ModelAlert:
    """Model monitoring alert"""
    id: str
    timestamp: datetime
    model_name: str
    alert_type: str
    severity: AlertSeverity
    message: str
    details: Dict[str, Any]
    is_resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['severity'] = self.severity.value
        if self.resolved_at:
            data['resolved_at'] = self.resolved_at.isoformat()
        return data


class ModelPerformanceTracker:
    """Tracks model performance metrics over time"""
    
    def __init__(self, storage_path: str = "/app/monitoring"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Performance history
        self.performance_history: Dict[str, List[PerformanceMetrics]] = defaultdict(list)
        self.lock = threading.Lock()
        
        # Load existing history
        self._load_performance_history()
    
    def record_performance(self, 
                          model_name: str,
                          model_version: str,
                          predictions: np.ndarray,
                          true_values: np.ndarray,
                          prediction_latency_ms: float = None,
                          memory_usage_mb: float = None,
                          cpu_usage_percent: float = None) -> PerformanceMetrics:
        """Record model performance metrics"""
        
        metrics = PerformanceMetrics(
            timestamp=datetime.utcnow(),
            model_name=model_name,
            model_version=model_version,
            sample_size=len(predictions),
            prediction_latency_ms=prediction_latency_ms,
            memory_usage_mb=memory_usage_mb,
            cpu_usage_percent=cpu_usage_percent
        )
        
        # Calculate appropriate metrics based on problem type
        if self._is_classification_problem(predictions, true_values):
            metrics.accuracy = accuracy_score(true_values, predictions)
            metrics.precision = precision_score(true_values, predictions, average='weighted', zero_division=0)
            metrics.recall = recall_score(true_values, predictions, average='weighted', zero_division=0)
            metrics.f1_score = f1_score(true_values, predictions, average='weighted', zero_division=0)
            
            # Directional accuracy for financial predictions
            if len(predictions) > 1:
                pred_direction = np.sign(predictions - np.mean(predictions))
                true_direction = np.sign(true_values - np.mean(true_values))
                metrics.directional_accuracy = accuracy_score(true_direction, pred_direction)
                
        else:
            # Regression metrics
            metrics.mse = mean_squared_error(true_values, predictions)
            metrics.mae = mean_absolute_error(true_values, predictions)
            metrics.r2_score = r2_score(true_values, predictions)
            
            # Financial metrics for return predictions
            if np.std(predictions) > 0:
                metrics.sharpe_ratio = np.mean(predictions) / np.std(predictions) * np.sqrt(252)
        
        # Store metrics
        with self.lock:
            self.performance_history[model_name].append(metrics)
            
            # Keep only recent history (last 1000 records)
            if len(self.performance_history[model_name]) > 1000:
                self.performance_history[model_name] = self.performance_history[model_name][-1000:]
        
        # Save to disk
        self._save_performance_metrics(metrics)
        
        logger.info(f"Recorded performance for model {model_name}: "
                   f"accuracy={metrics.accuracy}, f1={metrics.f1_score}, "
                   f"mse={metrics.mse}, r2={metrics.r2_score}")
        
        return metrics
    
    def _is_classification_problem(self, predictions: np.ndarray, true_values: np.ndarray) -> bool:
        """Detect if this is a classification or regression problem"""
        # Simple heuristic: if all values are integers and range is small, likely classification
        pred_unique = len(np.unique(predictions))
        true_unique = len(np.unique(true_values))
        
        return (pred_unique <= 20 and true_unique <= 20 and 
                np.all(predictions == predictions.astype(int)) and 
                np.all(true_values == true_values.astype(int)))
    
    def get_performance_trend(self, 
                            model_name: str, 
                            metric: str = "accuracy",
                            days_back: int = 30) -> pd.DataFrame:
        """Get performance trend for a specific metric"""
        
        if model_name not in self.performance_history:
            return pd.DataFrame()
        
        # Filter recent history
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        recent_metrics = [
            m for m in self.performance_history[model_name] 
            if m.timestamp >= cutoff_date
        ]
        
        if not recent_metrics:
            return pd.DataFrame()
        
        # Create DataFrame
        data = []
        for m in recent_metrics:
            metric_value = getattr(m, metric, None)
            if metric_value is not None:
                data.append({
                    'timestamp': m.timestamp,
                    'model_version': m.model_version,
                    'metric_value': metric_value,
                    'sample_size': m.sample_size
                })
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        df = df.sort_values('timestamp')
        
        return df
    
    def detect_performance_degradation(self,
                                     model_name: str,
                                     metric: str = "accuracy",
                                     degradation_threshold: float = 0.05,
                                     min_samples: int = 10) -> Optional[Dict[str, Any]]:
        """Detect performance degradation using statistical tests"""
        
        trend_df = self.get_performance_trend(model_name, metric, days_back=30)
        
        if len(trend_df) < min_samples:
            return None
        
        # Split into recent and historical periods
        split_point = len(trend_df) // 2
        historical = trend_df.iloc[:split_point]['metric_value'].values
        recent = trend_df.iloc[split_point:]['metric_value'].values
        
        if len(historical) < 5 or len(recent) < 5:
            return None
        
        # Statistical tests
        historical_mean = np.mean(historical)
        recent_mean = np.mean(recent)
        
        # T-test for significant difference
        t_stat, p_value = stats.ttest_ind(historical, recent)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(historical) - 1) * np.var(historical) + 
                             (len(recent) - 1) * np.var(recent)) / 
                            (len(historical) + len(recent) - 2))
        cohens_d = (recent_mean - historical_mean) / pooled_std if pooled_std > 0 else 0
        
        # Degradation detection
        relative_change = (recent_mean - historical_mean) / historical_mean if historical_mean > 0 else 0
        is_degradation = relative_change < -degradation_threshold and p_value < 0.05
        
        return {
            'model_name': model_name,
            'metric': metric,
            'historical_mean': historical_mean,
            'recent_mean': recent_mean,
            'relative_change': relative_change,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'is_significant_degradation': is_degradation,
            'degradation_threshold': degradation_threshold
        }
    
    def _load_performance_history(self):
        """Load performance history from disk"""
        try:
            history_file = self.storage_path / "performance_history.json"
            if history_file.exists():
                with open(history_file, 'r') as f:
                    data = json.load(f)
                
                for model_name, metrics_list in data.items():
                    for metric_data in metrics_list:
                        metric_data['timestamp'] = datetime.fromisoformat(metric_data['timestamp'])
                        metrics = PerformanceMetrics(**metric_data)
                        self.performance_history[model_name].append(metrics)
                        
                logger.info(f"Loaded performance history for {len(data)} models")
                
        except Exception as e:
            logger.error(f"Error loading performance history: {e}")
    
    def _save_performance_metrics(self, metrics: PerformanceMetrics):
        """Save individual performance metrics"""
        try:
            metrics_file = self.storage_path / f"metrics_{metrics.model_name}_{datetime.utcnow().strftime('%Y%m%d')}.json"
            
            # Append to daily file
            daily_metrics = []
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    daily_metrics = json.load(f)
            
            daily_metrics.append(metrics.to_dict())
            
            with open(metrics_file, 'w') as f:
                json.dump(daily_metrics, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving performance metrics: {e}")


class DriftDetector:
    """Advanced drift detection for model inputs and outputs"""
    
    def __init__(self):
        self.reference_distributions = {}
        self.drift_thresholds = {
            DriftType.DATA_DRIFT: 0.2,
            DriftType.PREDICTION_DRIFT: 0.15,
            DriftType.CONCEPT_DRIFT: 0.25
        }
    
    def update_reference_distribution(self, 
                                    model_name: str,
                                    feature_name: str,
                                    reference_data: np.ndarray):
        """Update reference distribution for drift detection"""
        
        if model_name not in self.reference_distributions:
            self.reference_distributions[model_name] = {}
        
        self.reference_distributions[model_name][feature_name] = {
            'mean': np.mean(reference_data),
            'std': np.std(reference_data),
            'quantiles': np.percentile(reference_data, [5, 25, 50, 75, 95]),
            'histogram': np.histogram(reference_data, bins=50),
            'updated_at': datetime.utcnow()
        }
    
    def detect_data_drift(self,
                         model_name: str,
                         current_features: Dict[str, np.ndarray],
                         confidence_level: float = 0.95) -> DriftDetectionResult:
        """Detect drift in input features"""
        
        feature_drifts = {}
        statistical_tests = {}
        overall_drift_score = 0.0
        
        if model_name not in self.reference_distributions:
            logger.warning(f"No reference distribution for model {model_name}")
            return self._create_empty_drift_result(model_name, DriftType.DATA_DRIFT)
        
        reference_dist = self.reference_distributions[model_name]
        
        for feature_name, current_data in current_features.items():
            if feature_name not in reference_dist:
                continue
            
            ref_data = reference_dist[feature_name]
            
            # Statistical tests
            drift_scores = {}
            
            # Kolmogorov-Smirnov test
            try:
                # Generate reference sample from stored statistics
                ref_sample = np.random.normal(
                    ref_data['mean'], 
                    ref_data['std'], 
                    size=len(current_data)
                )
                
                ks_stat, ks_p_value = stats.ks_2samp(ref_sample, current_data)
                drift_scores['ks_statistic'] = ks_stat
                drift_scores['ks_p_value'] = ks_p_value
                
            except Exception as e:
                logger.error(f"Error in KS test for {feature_name}: {e}")
                drift_scores['ks_statistic'] = 0.0
                drift_scores['ks_p_value'] = 1.0
            
            # Population Stability Index (PSI)
            try:
                psi_score = self._calculate_psi(ref_data, current_data)
                drift_scores['psi'] = psi_score
            except Exception as e:
                logger.error(f"Error calculating PSI for {feature_name}: {e}")
                drift_scores['psi'] = 0.0
            
            # Jensen-Shannon Distance
            try:
                js_distance = self._calculate_js_distance(ref_data, current_data)
                drift_scores['js_distance'] = js_distance
            except Exception as e:
                logger.error(f"Error calculating JS distance for {feature_name}: {e}")
                drift_scores['js_distance'] = 0.0
            
            # Combined drift score for this feature
            feature_drift_score = (
                0.4 * drift_scores['psi'] + 
                0.3 * drift_scores['ks_statistic'] + 
                0.3 * drift_scores['js_distance']
            )
            
            feature_drifts[feature_name] = feature_drift_score
            statistical_tests[feature_name] = drift_scores
            
            # Add to overall score
            overall_drift_score += feature_drift_score
        
        # Average drift score across features
        if feature_drifts:
            overall_drift_score /= len(feature_drifts)
        
        threshold = self.drift_thresholds[DriftType.DATA_DRIFT]
        is_drift_detected = overall_drift_score > threshold
        
        return DriftDetectionResult(
            timestamp=datetime.utcnow(),
            model_name=model_name,
            drift_type=DriftType.DATA_DRIFT,
            drift_score=overall_drift_score,
            threshold=threshold,
            is_drift_detected=is_drift_detected,
            feature_drifts=feature_drifts,
            statistical_test_results=statistical_tests,
            confidence_level=confidence_level,
            sample_size=len(next(iter(current_features.values()))),
            reference_period="last_30_days",
            detection_period="current"
        )
    
    def detect_prediction_drift(self,
                               model_name: str,
                               current_predictions: np.ndarray,
                               reference_predictions: np.ndarray = None) -> DriftDetectionResult:
        """Detect drift in model predictions"""
        
        if reference_predictions is None:
            # Use stored reference if available
            if (model_name in self.reference_distributions and 
                'predictions' in self.reference_distributions[model_name]):
                ref_data = self.reference_distributions[model_name]['predictions']
                # Generate reference sample
                reference_predictions = np.random.normal(
                    ref_data['mean'],
                    ref_data['std'],
                    size=len(current_predictions)
                )
            else:
                return self._create_empty_drift_result(model_name, DriftType.PREDICTION_DRIFT)
        
        # Statistical tests
        ks_stat, ks_p_value = stats.ks_2samp(reference_predictions, current_predictions)
        
        # Distribution comparison
        psi_score = self._calculate_psi_arrays(reference_predictions, current_predictions)
        js_distance = self._calculate_js_distance_arrays(reference_predictions, current_predictions)
        
        # Combined drift score
        drift_score = 0.4 * psi_score + 0.3 * ks_stat + 0.3 * js_distance
        
        threshold = self.drift_thresholds[DriftType.PREDICTION_DRIFT]
        is_drift_detected = drift_score > threshold
        
        return DriftDetectionResult(
            timestamp=datetime.utcnow(),
            model_name=model_name,
            drift_type=DriftType.PREDICTION_DRIFT,
            drift_score=drift_score,
            threshold=threshold,
            is_drift_detected=is_drift_detected,
            feature_drifts={'predictions': drift_score},
            statistical_test_results={
                'ks_statistic': ks_stat,
                'ks_p_value': ks_p_value,
                'psi': psi_score,
                'js_distance': js_distance
            },
            confidence_level=0.95,
            sample_size=len(current_predictions),
            reference_period="reference",
            detection_period="current"
        )
    
    def _calculate_psi(self, ref_data: Dict[str, Any], current_data: np.ndarray) -> float:
        """Calculate Population Stability Index using stored reference data"""
        try:
            # Use quantiles to create bins
            quantiles = ref_data['quantiles']
            bins = [-np.inf] + list(quantiles) + [np.inf]
            
            # Create reference frequencies from stored histogram
            ref_hist, _ = ref_data['histogram']
            ref_freq = ref_hist / ref_hist.sum()
            
            # Calculate current frequencies
            cur_hist, _ = np.histogram(current_data, bins=bins[1:-1])
            cur_freq = cur_hist / cur_hist.sum() if cur_hist.sum() > 0 else cur_hist
            
            # Ensure same length
            min_len = min(len(ref_freq), len(cur_freq))
            ref_freq = ref_freq[:min_len]
            cur_freq = cur_freq[:min_len]
            
            # Avoid division by zero
            ref_freq = np.where(ref_freq == 0, 0.0001, ref_freq)
            cur_freq = np.where(cur_freq == 0, 0.0001, cur_freq)
            
            # Calculate PSI
            psi = np.sum((cur_freq - ref_freq) * np.log(cur_freq / ref_freq))
            
            return min(abs(psi), 1.0)  # Cap at 1.0
            
        except Exception as e:
            logger.error(f"Error calculating PSI: {e}")
            return 0.0
    
    def _calculate_psi_arrays(self, reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
        """Calculate PSI between two arrays"""
        try:
            # Create bins
            combined_data = np.concatenate([reference, current])
            _, bin_edges = np.histogram(combined_data, bins=bins)
            
            # Calculate frequencies
            ref_hist, _ = np.histogram(reference, bins=bin_edges)
            cur_hist, _ = np.histogram(current, bins=bin_edges)
            
            # Convert to probabilities
            ref_freq = ref_hist / ref_hist.sum() if ref_hist.sum() > 0 else ref_hist
            cur_freq = cur_hist / cur_hist.sum() if cur_hist.sum() > 0 else cur_hist
            
            # Avoid division by zero
            ref_freq = np.where(ref_freq == 0, 0.0001, ref_freq)
            cur_freq = np.where(cur_freq == 0, 0.0001, cur_freq)
            
            # Calculate PSI
            psi = np.sum((cur_freq - ref_freq) * np.log(cur_freq / ref_freq))
            
            return min(abs(psi), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating PSI: {e}")
            return 0.0
    
    def _calculate_js_distance(self, ref_data: Dict[str, Any], current_data: np.ndarray) -> float:
        """Calculate Jensen-Shannon distance using stored reference data"""
        try:
            # Use stored histogram
            ref_hist, bin_edges = ref_data['histogram']
            
            # Calculate current histogram with same bins
            cur_hist, _ = np.histogram(current_data, bins=bin_edges)
            
            # Normalize
            ref_prob = ref_hist / ref_hist.sum() if ref_hist.sum() > 0 else ref_hist
            cur_prob = cur_hist / cur_hist.sum() if cur_hist.sum() > 0 else cur_hist
            
            # JS distance
            m = 0.5 * (ref_prob + cur_prob)
            
            def kl_div(p, q):
                return np.sum(p * np.log(p / (q + 1e-10) + 1e-10))
            
            js_distance = 0.5 * kl_div(ref_prob, m) + 0.5 * kl_div(cur_prob, m)
            
            return min(js_distance, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating JS distance: {e}")
            return 0.0
    
    def _calculate_js_distance_arrays(self, reference: np.ndarray, current: np.ndarray, bins: int = 50) -> float:
        """Calculate Jensen-Shannon distance between two arrays"""
        try:
            # Create common bins
            min_val = min(reference.min(), current.min())
            max_val = max(reference.max(), reference.max())
            
            ref_hist, _ = np.histogram(reference, bins=bins, range=(min_val, max_val))
            cur_hist, _ = np.histogram(current, bins=bins, range=(min_val, max_val))
            
            # Normalize
            ref_prob = ref_hist / ref_hist.sum() if ref_hist.sum() > 0 else ref_hist
            cur_prob = cur_hist / cur_hist.sum() if cur_hist.sum() > 0 else cur_hist
            
            # JS distance
            m = 0.5 * (ref_prob + cur_prob)
            
            def kl_div(p, q):
                return np.sum(p * np.log(p / (q + 1e-10) + 1e-10))
            
            js_distance = 0.5 * kl_div(ref_prob, m) + 0.5 * kl_div(cur_prob, m)
            
            return min(js_distance, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating JS distance: {e}")
            return 0.0
    
    def _create_empty_drift_result(self, model_name: str, drift_type: DriftType) -> DriftDetectionResult:
        """Create empty drift result when no reference data available"""
        return DriftDetectionResult(
            timestamp=datetime.utcnow(),
            model_name=model_name,
            drift_type=drift_type,
            drift_score=0.0,
            threshold=self.drift_thresholds[drift_type],
            is_drift_detected=False,
            feature_drifts={},
            statistical_test_results={},
            confidence_level=0.95,
            sample_size=0,
            reference_period="unavailable",
            detection_period="current"
        )


class AlertManager:
    """Manages model monitoring alerts"""
    
    def __init__(self, storage_path: str = "/app/monitoring/alerts"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.alerts: List[ModelAlert] = []
        self.alert_handlers: Dict[str, Callable] = {}
        self.lock = threading.Lock()
        
        # Load existing alerts
        self._load_alerts()
    
    def create_alert(self,
                    model_name: str,
                    alert_type: str,
                    severity: AlertSeverity,
                    message: str,
                    details: Dict[str, Any] = None) -> str:
        """Create a new alert"""
        
        alert_id = f"{model_name}_{alert_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        alert = ModelAlert(
            id=alert_id,
            timestamp=datetime.utcnow(),
            model_name=model_name,
            alert_type=alert_type,
            severity=severity,
            message=message,
            details=details or {}
        )
        
        with self.lock:
            self.alerts.append(alert)
        
        # Save alert
        self._save_alert(alert)
        
        # Trigger alert handlers
        self._trigger_alert_handlers(alert)
        
        logger.warning(f"Alert created: {alert_id} - {message}")
        
        return alert_id
    
    def resolve_alert(self, alert_id: str, resolution_notes: str = ""):
        """Resolve an alert"""
        
        with self.lock:
            for alert in self.alerts:
                if alert.id == alert_id:
                    alert.is_resolved = True
                    alert.resolved_at = datetime.utcnow()
                    alert.resolution_notes = resolution_notes
                    
                    self._save_alert(alert)
                    
                    logger.info(f"Alert resolved: {alert_id}")
                    break
    
    def get_active_alerts(self, model_name: str = None) -> List[ModelAlert]:
        """Get active (unresolved) alerts"""
        
        active_alerts = [alert for alert in self.alerts if not alert.is_resolved]
        
        if model_name:
            active_alerts = [alert for alert in active_alerts if alert.model_name == model_name]
        
        return active_alerts
    
    def register_alert_handler(self, alert_type: str, handler: Callable):
        """Register custom alert handler"""
        self.alert_handlers[alert_type] = handler
    
    def _trigger_alert_handlers(self, alert: ModelAlert):
        """Trigger registered alert handlers"""
        
        # Generic handler
        if 'generic' in self.alert_handlers:
            try:
                self.alert_handlers['generic'](alert)
            except Exception as e:
                logger.error(f"Error in generic alert handler: {e}")
        
        # Specific handler
        if alert.alert_type in self.alert_handlers:
            try:
                self.alert_handlers[alert.alert_type](alert)
            except Exception as e:
                logger.error(f"Error in {alert.alert_type} alert handler: {e}")
    
    def _load_alerts(self):
        """Load alerts from storage"""
        try:
            alerts_file = self.storage_path / "alerts.json"
            if alerts_file.exists():
                with open(alerts_file, 'r') as f:
                    alerts_data = json.load(f)
                
                for alert_data in alerts_data:
                    alert_data['timestamp'] = datetime.fromisoformat(alert_data['timestamp'])
                    alert_data['severity'] = AlertSeverity(alert_data['severity'])
                    if alert_data.get('resolved_at'):
                        alert_data['resolved_at'] = datetime.fromisoformat(alert_data['resolved_at'])
                    
                    alert = ModelAlert(**alert_data)
                    self.alerts.append(alert)
                
                logger.info(f"Loaded {len(self.alerts)} alerts")
                
        except Exception as e:
            logger.error(f"Error loading alerts: {e}")
    
    def _save_alert(self, alert: ModelAlert):
        """Save individual alert"""
        try:
            alert_file = self.storage_path / f"alert_{alert.id}.json"
            
            with open(alert_file, 'w') as f:
                json.dump(alert.to_dict(), f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving alert: {e}")


class ModelMonitor:
    """
    Comprehensive model monitoring system
    """
    
    def __init__(self, 
                 storage_path: str = "/app/monitoring",
                 monitoring_interval_hours: float = 1.0):
        
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.monitoring_interval_hours = monitoring_interval_hours
        
        # Components
        self.performance_tracker = ModelPerformanceTracker(str(self.storage_path / "performance"))
        self.drift_detector = DriftDetector()
        self.alert_manager = AlertManager(str(self.storage_path / "alerts"))
        
        # Monitoring state
        self.monitored_models: Dict[str, Dict[str, Any]] = {}
        self.is_monitoring = False
        self.monitoring_thread = None
        
        # Thresholds
        self.performance_thresholds = {
            'accuracy_degradation': 0.05,
            'f1_degradation': 0.05,
            'mse_increase': 0.2,
            'latency_increase': 0.5
        }
        
        logger.info("Model monitor initialized")
    
    def register_model(self,
                      model_name: str,
                      model_version: str,
                      monitoring_config: Dict[str, Any] = None):
        """Register a model for monitoring"""
        
        config = monitoring_config or {}
        
        self.monitored_models[model_name] = {
            'version': model_version,
            'registered_at': datetime.utcnow(),
            'last_monitored': None,
            'monitoring_config': config,
            'health_status': ModelHealth.UNKNOWN,
            'performance_baseline': None
        }
        
        logger.info(f"Registered model {model_name} v{model_version} for monitoring")
    
    def start_monitoring(self):
        """Start continuous monitoring"""
        
        if self.is_monitoring:
            logger.warning("Monitoring already started")
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Started continuous model monitoring")
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        logger.info("Stopped model monitoring")
    
    def monitor_model_performance(self,
                                model_name: str,
                                predictions: np.ndarray,
                                true_values: np.ndarray,
                                features: Dict[str, np.ndarray] = None,
                                metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Monitor a single model's performance"""
        
        if model_name not in self.monitored_models:
            logger.warning(f"Model {model_name} not registered for monitoring")
            return {}
        
        model_info = self.monitored_models[model_name]
        monitoring_results = {}
        
        # Record performance metrics
        performance_metrics = self.performance_tracker.record_performance(
            model_name=model_name,
            model_version=model_info['version'],
            predictions=predictions,
            true_values=true_values,
            prediction_latency_ms=metadata.get('latency_ms') if metadata else None,
            memory_usage_mb=metadata.get('memory_mb') if metadata else None,
            cpu_usage_percent=metadata.get('cpu_percent') if metadata else None
        )
        
        monitoring_results['performance_metrics'] = performance_metrics
        
        # Check for performance degradation
        degradation_result = self.performance_tracker.detect_performance_degradation(
            model_name=model_name,
            metric='accuracy' if performance_metrics.accuracy else 'mse',
            degradation_threshold=self.performance_thresholds.get('accuracy_degradation', 0.05)
        )
        
        if degradation_result and degradation_result['is_significant_degradation']:
            self.alert_manager.create_alert(
                model_name=model_name,
                alert_type="performance_degradation",
                severity=AlertSeverity.WARNING,
                message=f"Performance degradation detected: {degradation_result['relative_change']:.1%} decrease",
                details=degradation_result
            )
        
        monitoring_results['degradation_check'] = degradation_result
        
        # Drift detection
        if features:
            # Update reference distributions if this is the first run
            if model_info['performance_baseline'] is None:
                for feature_name, feature_data in features.items():
                    self.drift_detector.update_reference_distribution(
                        model_name, feature_name, feature_data
                    )
                
                # Store predictions as reference
                self.drift_detector.update_reference_distribution(
                    model_name, 'predictions', predictions
                )
                
                model_info['performance_baseline'] = performance_metrics
            
            # Detect data drift
            data_drift_result = self.drift_detector.detect_data_drift(
                model_name=model_name,
                current_features=features
            )
            
            monitoring_results['data_drift'] = data_drift_result
            
            if data_drift_result.is_drift_detected:
                self.alert_manager.create_alert(
                    model_name=model_name,
                    alert_type="data_drift",
                    severity=AlertSeverity.WARNING,
                    message=f"Data drift detected: drift_score={data_drift_result.drift_score:.3f}",
                    details=data_drift_result.to_dict()
                )
            
            # Detect prediction drift
            if model_info['performance_baseline']:
                prediction_drift_result = self.drift_detector.detect_prediction_drift(
                    model_name=model_name,
                    current_predictions=predictions
                )
                
                monitoring_results['prediction_drift'] = prediction_drift_result
                
                if prediction_drift_result.is_drift_detected:
                    self.alert_manager.create_alert(
                        model_name=model_name,
                        alert_type="prediction_drift", 
                        severity=AlertSeverity.WARNING,
                        message=f"Prediction drift detected: drift_score={prediction_drift_result.drift_score:.3f}",
                        details=prediction_drift_result.to_dict()
                    )
        
        # Update model health status
        health_status = self._assess_model_health(model_name, monitoring_results)
        model_info['health_status'] = health_status
        model_info['last_monitored'] = datetime.utcnow()
        
        monitoring_results['health_status'] = health_status
        
        return monitoring_results
    
    def _assess_model_health(self, model_name: str, monitoring_results: Dict[str, Any]) -> ModelHealth:
        """Assess overall model health"""
        
        # Check for critical issues
        active_alerts = self.alert_manager.get_active_alerts(model_name)
        critical_alerts = [a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]
        error_alerts = [a for a in active_alerts if a.severity == AlertSeverity.ERROR]
        
        if critical_alerts or error_alerts:
            return ModelHealth.FAILING
        
        # Check performance metrics
        performance = monitoring_results.get('performance_metrics')
        if performance:
            # Check if performance is below acceptable thresholds
            if (performance.accuracy and performance.accuracy < 0.5) or \
               (performance.f1_score and performance.f1_score < 0.5):
                return ModelHealth.FAILING
        
        # Check for drift
        data_drift = monitoring_results.get('data_drift')
        prediction_drift = monitoring_results.get('prediction_drift')
        
        if ((data_drift and data_drift.is_drift_detected) or 
            (prediction_drift and prediction_drift.is_drift_detected)):
            return ModelHealth.DEGRADED
        
        # Check for performance degradation
        degradation = monitoring_results.get('degradation_check')
        if degradation and degradation['is_significant_degradation']:
            return ModelHealth.DEGRADED
        
        return ModelHealth.HEALTHY
    
    def _monitoring_loop(self):
        """Continuous monitoring loop"""
        
        while self.is_monitoring:
            try:
                # Monitor each registered model
                for model_name, model_info in self.monitored_models.items():
                    
                    # Check if it's time to monitor this model
                    if model_info['last_monitored']:
                        time_since_monitor = datetime.utcnow() - model_info['last_monitored']
                        if time_since_monitor.total_seconds() < self.monitoring_interval_hours * 3600:
                            continue
                    
                    # Perform automated checks
                    self._automated_model_check(model_name, model_info)
                
                # Sleep until next check
                import time
                time.sleep(self.monitoring_interval_hours * 3600)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                import time
                time.sleep(60)  # Wait 1 minute before retrying
    
    def _automated_model_check(self, model_name: str, model_info: Dict[str, Any]):
        """Perform automated checks for a model"""
        
        try:
            # Check for stale data
            if model_info['last_monitored']:
                hours_since_update = (datetime.utcnow() - model_info['last_monitored']).total_seconds() / 3600
                if hours_since_update > 24:  # No updates in 24 hours
                    self.alert_manager.create_alert(
                        model_name=model_name,
                        alert_type="stale_data",
                        severity=AlertSeverity.WARNING,
                        message=f"No performance data received for {hours_since_update:.1f} hours",
                        details={'hours_since_update': hours_since_update}
                    )
            
            # Check active alerts count
            active_alerts = self.alert_manager.get_active_alerts(model_name)
            if len(active_alerts) > 10:  # Too many unresolved alerts
                self.alert_manager.create_alert(
                    model_name=model_name,
                    alert_type="alert_overflow",
                    severity=AlertSeverity.CRITICAL,
                    message=f"Too many unresolved alerts: {len(active_alerts)}",
                    details={'active_alert_count': len(active_alerts)}
                )
            
            # Update last monitored timestamp
            model_info['last_monitored'] = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Error in automated check for {model_name}: {e}")
    
    def get_model_health_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive model health dashboard"""
        
        dashboard = {
            'timestamp': datetime.utcnow().isoformat(),
            'monitored_models_count': len(self.monitored_models),
            'models': {},
            'overall_health': {},
            'recent_alerts': []
        }
        
        health_counts = defaultdict(int)
        
        for model_name, model_info in self.monitored_models.items():
            
            # Get recent performance trend
            performance_trend = self.performance_tracker.get_performance_trend(
                model_name, days_back=7
            )
            
            # Get active alerts
            active_alerts = self.alert_manager.get_active_alerts(model_name)
            
            model_dashboard = {
                'version': model_info['version'],
                'health_status': model_info['health_status'].value,
                'last_monitored': model_info['last_monitored'].isoformat() if model_info['last_monitored'] else None,
                'active_alerts_count': len(active_alerts),
                'performance_trend_points': len(performance_trend),
                'recent_performance': {}
            }
            
            # Add recent performance metrics
            if not performance_trend.empty:
                latest_performance = performance_trend.iloc[-1]
                model_dashboard['recent_performance'] = {
                    'metric_value': latest_performance['metric_value'],
                    'sample_size': latest_performance['sample_size'],
                    'timestamp': latest_performance['timestamp'].isoformat()
                }
            
            dashboard['models'][model_name] = model_dashboard
            health_counts[model_info['health_status'].value] += 1
        
        # Overall health summary
        dashboard['overall_health'] = dict(health_counts)
        
        # Recent alerts (last 24 hours)
        recent_cutoff = datetime.utcnow() - timedelta(hours=24)
        recent_alerts = [
            alert.to_dict() for alert in self.alerts 
            if alert.timestamp >= recent_cutoff
        ]
        dashboard['recent_alerts'] = recent_alerts[-20:]  # Last 20 alerts
        
        return dashboard
    
    def generate_monitoring_report(self, model_name: str, days_back: int = 30) -> Dict[str, Any]:
        """Generate comprehensive monitoring report for a model"""
        
        if model_name not in self.monitored_models:
            return {}
        
        model_info = self.monitored_models[model_name]
        
        report = {
            'model_name': model_name,
            'model_version': model_info['version'],
            'report_period_days': days_back,
            'generated_at': datetime.utcnow().isoformat(),
            'summary': {},
            'performance_analysis': {},
            'drift_analysis': {},
            'alert_summary': {},
            'recommendations': []
        }
        
        # Performance analysis
        performance_trend = self.performance_tracker.get_performance_trend(
            model_name, days_back=days_back
        )
        
        if not performance_trend.empty:
            report['performance_analysis'] = {
                'data_points': len(performance_trend),
                'latest_performance': performance_trend.iloc[-1]['metric_value'],
                'average_performance': performance_trend['metric_value'].mean(),
                'performance_stability': performance_trend['metric_value'].std(),
                'trend_direction': 'improving' if performance_trend['metric_value'].iloc[-1] > performance_trend['metric_value'].iloc[0] else 'declining'
            }
        
        # Alert summary
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        period_alerts = [
            alert for alert in self.alert_manager.alerts 
            if alert.model_name == model_name and alert.timestamp >= cutoff_date
        ]
        
        alert_counts = defaultdict(int)
        for alert in period_alerts:
            alert_counts[alert.alert_type] += 1
        
        report['alert_summary'] = {
            'total_alerts': len(period_alerts),
            'alerts_by_type': dict(alert_counts),
            'resolved_alerts': len([a for a in period_alerts if a.is_resolved]),
            'active_alerts': len([a for a in period_alerts if not a.is_resolved])
        }
        
        # Generate recommendations
        recommendations = self._generate_recommendations(model_name, report)
        report['recommendations'] = recommendations
        
        return report
    
    def _generate_recommendations(self, model_name: str, report: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on monitoring data"""
        
        recommendations = []
        
        # Performance-based recommendations
        performance = report.get('performance_analysis', {})
        if performance.get('trend_direction') == 'declining':
            recommendations.append("Consider retraining the model due to declining performance trend")
        
        if performance.get('performance_stability', 0) > 0.1:
            recommendations.append("High performance variability detected - investigate data quality")
        
        # Alert-based recommendations
        alert_summary = report.get('alert_summary', {})
        if alert_summary.get('total_alerts', 0) > 10:
            recommendations.append("High alert volume - review monitoring thresholds")
        
        if 'data_drift' in alert_summary.get('alerts_by_type', {}):
            recommendations.append("Data drift detected - consider feature engineering or model updates")
        
        if 'performance_degradation' in alert_summary.get('alerts_by_type', {}):
            recommendations.append("Performance degradation alerts - schedule model retraining")
        
        # General recommendations
        model_info = self.monitored_models[model_name]
        if model_info['health_status'] == ModelHealth.DEGRADED:
            recommendations.append("Model health is degraded - immediate attention required")
        elif model_info['health_status'] == ModelHealth.FAILING:
            recommendations.append("Model is failing - consider taking offline and investigating")
        
        return recommendations


# Global monitor instance
_model_monitor: Optional[ModelMonitor] = None

def get_model_monitor() -> ModelMonitor:
    """Get global model monitor instance"""
    global _model_monitor
    if _model_monitor is None:
        _model_monitor = ModelMonitor()
    return _model_monitor