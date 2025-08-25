"""
Model Monitoring - Performance tracking, drift detection, and alerting
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score, confusion_matrix, classification_report
)
import json
from pathlib import Path
import asyncio
import aiohttp
from collections import deque
import warnings

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class DriftType(Enum):
    """Types of drift detection"""
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    PREDICTION_DRIFT = "prediction_drift"
    PERFORMANCE_DRIFT = "performance_drift"


@dataclass
class PerformanceMetrics:
    """Container for model performance metrics"""
    model_name: str
    model_version: str
    timestamp: datetime
    
    # Classification metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    roc_auc: Optional[float] = None
    confusion_matrix: Optional[List[List[int]]] = None
    
    # Regression metrics
    mse: Optional[float] = None
    mae: Optional[float] = None
    rmse: Optional[float] = None
    r2: Optional[float] = None
    mape: Optional[float] = None
    
    # Business metrics
    business_value: Optional[float] = None
    cost_saved: Optional[float] = None
    revenue_impact: Optional[float] = None
    
    # Operational metrics
    latency_ms: Optional[float] = None
    throughput_rps: Optional[float] = None
    error_rate: Optional[float] = None
    availability: Optional[float] = None
    
    # Data quality metrics
    missing_features_rate: Optional[float] = None
    out_of_range_rate: Optional[float] = None
    
    # Sample information
    sample_count: int = 0
    prediction_count: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "timestamp": self.timestamp.isoformat(),
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "roc_auc": self.roc_auc,
            "mse": self.mse,
            "mae": self.mae,
            "rmse": self.rmse,
            "r2": self.r2,
            "mape": self.mape,
            "latency_ms": self.latency_ms,
            "throughput_rps": self.throughput_rps,
            "error_rate": self.error_rate,
            "sample_count": self.sample_count,
            "prediction_count": self.prediction_count
        }


@dataclass
class DriftReport:
    """Report for drift detection results"""
    drift_type: DriftType
    timestamp: datetime
    is_drift_detected: bool
    drift_score: float
    threshold: float
    
    # Statistical tests
    statistical_test: str = ""
    p_value: Optional[float] = None
    test_statistic: Optional[float] = None
    
    # Feature-level drift
    feature_drift_scores: Dict[str, float] = field(default_factory=dict)
    drifted_features: List[str] = field(default_factory=list)
    
    # Distribution comparison
    reference_distribution: Optional[Dict] = None
    current_distribution: Optional[Dict] = None
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "drift_type": self.drift_type.value,
            "timestamp": self.timestamp.isoformat(),
            "is_drift_detected": self.is_drift_detected,
            "drift_score": self.drift_score,
            "threshold": self.threshold,
            "statistical_test": self.statistical_test,
            "p_value": self.p_value,
            "test_statistic": self.test_statistic,
            "feature_drift_scores": self.feature_drift_scores,
            "drifted_features": self.drifted_features,
            "recommendations": self.recommendations
        }


@dataclass
class Alert:
    """Alert notification"""
    alert_id: str
    severity: AlertSeverity
    title: str
    message: str
    timestamp: datetime
    
    # Context
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    
    # Actions
    recommended_actions: List[str] = field(default_factory=list)
    auto_remediation: bool = False
    
    # Status
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None


class DriftDetector:
    """Detects various types of drift in model and data"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.reference_data: Optional[pd.DataFrame] = None
        self.reference_predictions: Optional[np.ndarray] = None
        self.reference_stats: Dict[str, Dict] = {}
        
    def _default_config(self) -> Dict:
        """Default drift detection configuration"""
        return {
            "data_drift_threshold": 0.3,
            "concept_drift_threshold": 0.05,  # p-value
            "prediction_drift_threshold": 0.2,
            "performance_drift_threshold": 0.1,
            "min_samples": 100,
            "window_size": 1000,
            "statistical_tests": {
                "numerical": "kolmogorov_smirnov",
                "categorical": "chi_squared"
            }
        }
    
    def set_reference_data(self, data: pd.DataFrame, predictions: Optional[np.ndarray] = None):
        """Set reference data for drift detection"""
        self.reference_data = data
        self.reference_predictions = predictions
        
        # Calculate reference statistics
        self.reference_stats = self._calculate_statistics(data)
        logger.info(f"Reference data set with {len(data)} samples and {len(data.columns)} features")
    
    def _calculate_statistics(self, data: pd.DataFrame) -> Dict[str, Dict]:
        """Calculate statistics for each feature"""
        stats = {}
        
        for column in data.columns:
            if data[column].dtype in ['float64', 'int64']:
                stats[column] = {
                    "mean": data[column].mean(),
                    "std": data[column].std(),
                    "min": data[column].min(),
                    "max": data[column].max(),
                    "median": data[column].median(),
                    "q1": data[column].quantile(0.25),
                    "q3": data[column].quantile(0.75)
                }
            else:
                # Categorical features
                value_counts = data[column].value_counts()
                stats[column] = {
                    "unique_values": len(value_counts),
                    "mode": value_counts.index[0] if len(value_counts) > 0 else None,
                    "distribution": value_counts.to_dict()
                }
        
        return stats
    
    async def detect_data_drift(self, current_data: pd.DataFrame) -> DriftReport:
        """Detect data drift between reference and current data"""
        if self.reference_data is None:
            raise ValueError("Reference data not set")
        
        if len(current_data) < self.config["min_samples"]:
            logger.warning(f"Insufficient samples for drift detection: {len(current_data)}")
            return DriftReport(
                drift_type=DriftType.DATA_DRIFT,
                timestamp=datetime.utcnow(),
                is_drift_detected=False,
                drift_score=0,
                threshold=self.config["data_drift_threshold"]
            )
        
        feature_drift_scores = {}
        drifted_features = []
        
        for column in self.reference_data.columns:
            if column not in current_data.columns:
                continue
            
            # Perform statistical test based on feature type
            if self.reference_data[column].dtype in ['float64', 'int64']:
                drift_score, p_value = self._kolmogorov_smirnov_test(
                    self.reference_data[column],
                    current_data[column]
                )
            else:
                drift_score, p_value = self._chi_squared_test(
                    self.reference_data[column],
                    current_data[column]
                )
            
            feature_drift_scores[column] = drift_score
            
            if drift_score > self.config["data_drift_threshold"]:
                drifted_features.append(column)
        
        # Calculate overall drift score
        overall_drift_score = np.mean(list(feature_drift_scores.values()))
        is_drift_detected = overall_drift_score > self.config["data_drift_threshold"]
        
        # Generate recommendations
        recommendations = []
        if is_drift_detected:
            recommendations.append("Consider retraining the model with recent data")
            recommendations.append(f"Features with significant drift: {', '.join(drifted_features[:5])}")
            if len(drifted_features) > 5:
                recommendations.append(f"And {len(drifted_features) - 5} more features")
        
        return DriftReport(
            drift_type=DriftType.DATA_DRIFT,
            timestamp=datetime.utcnow(),
            is_drift_detected=is_drift_detected,
            drift_score=overall_drift_score,
            threshold=self.config["data_drift_threshold"],
            statistical_test="mixed",
            feature_drift_scores=feature_drift_scores,
            drifted_features=drifted_features,
            recommendations=recommendations
        )
    
    def _kolmogorov_smirnov_test(self, reference: pd.Series, current: pd.Series) -> Tuple[float, float]:
        """Perform Kolmogorov-Smirnov test for numerical features"""
        try:
            statistic, p_value = stats.ks_2samp(reference.dropna(), current.dropna())
            return statistic, p_value
        except Exception as e:
            logger.warning(f"KS test failed: {e}")
            return 0, 1
    
    def _chi_squared_test(self, reference: pd.Series, current: pd.Series) -> Tuple[float, float]:
        """Perform Chi-squared test for categorical features"""
        try:
            # Get value counts
            ref_counts = reference.value_counts()
            curr_counts = current.value_counts()
            
            # Align categories
            all_categories = set(ref_counts.index) | set(curr_counts.index)
            ref_aligned = pd.Series([ref_counts.get(cat, 0) for cat in all_categories])
            curr_aligned = pd.Series([curr_counts.get(cat, 0) for cat in all_categories])
            
            # Perform chi-squared test
            statistic, p_value = stats.chisquare(curr_aligned, ref_aligned)
            
            # Normalize statistic to [0, 1]
            normalized_statistic = 1 - p_value
            
            return normalized_statistic, p_value
        except Exception as e:
            logger.warning(f"Chi-squared test failed: {e}")
            return 0, 1
    
    async def detect_concept_drift(
        self,
        predictions: np.ndarray,
        actual_labels: np.ndarray
    ) -> DriftReport:
        """Detect concept drift using prediction error analysis"""
        if len(predictions) != len(actual_labels):
            raise ValueError("Predictions and labels must have same length")
        
        # Calculate error rate over time windows
        window_size = self.config["window_size"]
        n_windows = len(predictions) // window_size
        
        if n_windows < 2:
            return DriftReport(
                drift_type=DriftType.CONCEPT_DRIFT,
                timestamp=datetime.utcnow(),
                is_drift_detected=False,
                drift_score=0,
                threshold=self.config["concept_drift_threshold"]
            )
        
        window_errors = []
        for i in range(n_windows):
            start_idx = i * window_size
            end_idx = (i + 1) * window_size
            
            window_pred = predictions[start_idx:end_idx]
            window_actual = actual_labels[start_idx:end_idx]
            
            # Calculate error rate
            errors = np.abs(window_pred - window_actual)
            error_rate = np.mean(errors)
            window_errors.append(error_rate)
        
        # Test for trend in error rates
        x = np.arange(len(window_errors))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, window_errors)
        
        # Detect if there's significant trend
        is_drift_detected = p_value < self.config["concept_drift_threshold"]
        
        recommendations = []
        if is_drift_detected:
            if slope > 0:
                recommendations.append("Model performance is degrading over time")
            else:
                recommendations.append("Model performance is improving over time")
            recommendations.append("Consider investigating the cause of concept drift")
            recommendations.append("Retrain with recent data to adapt to new patterns")
        
        return DriftReport(
            drift_type=DriftType.CONCEPT_DRIFT,
            timestamp=datetime.utcnow(),
            is_drift_detected=is_drift_detected,
            drift_score=abs(slope),
            threshold=self.config["concept_drift_threshold"],
            statistical_test="linear_regression_trend",
            p_value=p_value,
            test_statistic=slope,
            recommendations=recommendations
        )
    
    async def detect_prediction_drift(
        self,
        current_predictions: np.ndarray
    ) -> DriftReport:
        """Detect drift in prediction distribution"""
        if self.reference_predictions is None:
            raise ValueError("Reference predictions not set")
        
        # Compare prediction distributions
        statistic, p_value = stats.ks_2samp(
            self.reference_predictions,
            current_predictions
        )
        
        is_drift_detected = statistic > self.config["prediction_drift_threshold"]
        
        recommendations = []
        if is_drift_detected:
            recommendations.append("Prediction distribution has changed significantly")
            recommendations.append("Check if input data distribution has changed")
            recommendations.append("Model may need recalibration or retraining")
        
        return DriftReport(
            drift_type=DriftType.PREDICTION_DRIFT,
            timestamp=datetime.utcnow(),
            is_drift_detected=is_drift_detected,
            drift_score=statistic,
            threshold=self.config["prediction_drift_threshold"],
            statistical_test="kolmogorov_smirnov",
            p_value=p_value,
            test_statistic=statistic,
            recommendations=recommendations
        )


class ModelMonitor:
    """Comprehensive model monitoring system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.drift_detector = DriftDetector()
        self.alert_manager = AlertManager()
        
        # Metrics storage
        self.metrics_history: Dict[str, deque] = {}
        self.max_history_size = 10000
        
        # Monitoring state
        self.monitored_models: Dict[str, Dict] = {}
        self.last_check_time: Dict[str, datetime] = {}
        
    def _default_config(self) -> Dict:
        """Default monitoring configuration"""
        return {
            "check_interval_seconds": 300,  # 5 minutes
            "metrics_aggregation_window": 3600,  # 1 hour
            "alert_thresholds": {
                "accuracy_min": 0.8,
                "latency_max_ms": 1000,
                "error_rate_max": 0.05,
                "availability_min": 0.99
            },
            "enable_auto_remediation": False
        }
    
    async def register_model(
        self,
        model_name: str,
        model_version: str,
        endpoint: str,
        reference_data: Optional[pd.DataFrame] = None
    ):
        """Register a model for monitoring"""
        self.monitored_models[model_name] = {
            "version": model_version,
            "endpoint": endpoint,
            "registered_at": datetime.utcnow(),
            "status": "active"
        }
        
        # Initialize metrics history
        if model_name not in self.metrics_history:
            self.metrics_history[model_name] = deque(maxlen=self.max_history_size)
        
        # Set reference data for drift detection
        if reference_data is not None:
            self.drift_detector.set_reference_data(reference_data)
        
        logger.info(f"Registered model {model_name} v{model_version} for monitoring")
    
    async def collect_metrics(
        self,
        model_name: str,
        endpoint: str
    ) -> PerformanceMetrics:
        """Collect metrics from model endpoint"""
        try:
            # Make request to metrics endpoint
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{endpoint}/metrics") as response:
                    if response.status == 200:
                        metrics_data = await response.json()
                    else:
                        logger.error(f"Failed to collect metrics from {endpoint}: {response.status}")
                        metrics_data = {}
            
            # Create metrics object
            metrics = PerformanceMetrics(
                model_name=model_name,
                model_version=self.monitored_models[model_name]["version"],
                timestamp=datetime.utcnow(),
                accuracy=metrics_data.get("accuracy"),
                precision=metrics_data.get("precision"),
                recall=metrics_data.get("recall"),
                f1_score=metrics_data.get("f1_score"),
                latency_ms=metrics_data.get("latency_ms"),
                throughput_rps=metrics_data.get("throughput_rps"),
                error_rate=metrics_data.get("error_rate"),
                sample_count=metrics_data.get("sample_count", 0),
                prediction_count=metrics_data.get("prediction_count", 0)
            )
            
            # Store in history
            self.metrics_history[model_name].append(metrics)
            
            # Check for alerts
            await self._check_alerts(model_name, metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting metrics for {model_name}: {e}")
            return PerformanceMetrics(
                model_name=model_name,
                model_version="unknown",
                timestamp=datetime.utcnow()
            )
    
    async def _check_alerts(self, model_name: str, metrics: PerformanceMetrics):
        """Check if any alert conditions are met"""
        thresholds = self.config["alert_thresholds"]
        
        # Check accuracy
        if metrics.accuracy and metrics.accuracy < thresholds["accuracy_min"]:
            await self.alert_manager.create_alert(
                severity=AlertSeverity.WARNING,
                title=f"Low accuracy for {model_name}",
                message=f"Accuracy {metrics.accuracy:.3f} below threshold {thresholds['accuracy_min']}",
                model_name=model_name,
                metric_name="accuracy",
                metric_value=metrics.accuracy,
                threshold=thresholds["accuracy_min"]
            )
        
        # Check latency
        if metrics.latency_ms and metrics.latency_ms > thresholds["latency_max_ms"]:
            await self.alert_manager.create_alert(
                severity=AlertSeverity.WARNING,
                title=f"High latency for {model_name}",
                message=f"Latency {metrics.latency_ms}ms exceeds threshold {thresholds['latency_max_ms']}ms",
                model_name=model_name,
                metric_name="latency_ms",
                metric_value=metrics.latency_ms,
                threshold=thresholds["latency_max_ms"]
            )
        
        # Check error rate
        if metrics.error_rate and metrics.error_rate > thresholds["error_rate_max"]:
            await self.alert_manager.create_alert(
                severity=AlertSeverity.ERROR,
                title=f"High error rate for {model_name}",
                message=f"Error rate {metrics.error_rate:.3f} exceeds threshold {thresholds['error_rate_max']}",
                model_name=model_name,
                metric_name="error_rate",
                metric_value=metrics.error_rate,
                threshold=thresholds["error_rate_max"]
            )
    
    async def get_model_metrics(self, model_name: str) -> Dict[str, Any]:
        """Get current metrics for a model"""
        if model_name not in self.metrics_history:
            return {}
        
        recent_metrics = list(self.metrics_history[model_name])
        if not recent_metrics:
            return {}
        
        # Get latest metrics
        latest = recent_metrics[-1]
        
        # Calculate aggregated metrics over window
        window_start = datetime.utcnow() - timedelta(seconds=self.config["metrics_aggregation_window"])
        window_metrics = [m for m in recent_metrics if m.timestamp >= window_start]
        
        if window_metrics:
            # Calculate aggregates
            avg_accuracy = np.mean([m.accuracy for m in window_metrics if m.accuracy])
            avg_latency = np.mean([m.latency_ms for m in window_metrics if m.latency_ms])
            avg_error_rate = np.mean([m.error_rate for m in window_metrics if m.error_rate])
            total_predictions = sum(m.prediction_count for m in window_metrics)
        else:
            avg_accuracy = latest.accuracy
            avg_latency = latest.latency_ms
            avg_error_rate = latest.error_rate
            total_predictions = latest.prediction_count
        
        return {
            "model_name": model_name,
            "model_version": self.monitored_models[model_name]["version"],
            "latest_metrics": latest.to_dict(),
            "aggregated_metrics": {
                "avg_accuracy": avg_accuracy,
                "avg_latency_ms": avg_latency,
                "avg_error_rate": avg_error_rate,
                "total_predictions": total_predictions,
                "window_size_seconds": self.config["metrics_aggregation_window"]
            },
            "metrics_count": len(window_metrics),
            "monitoring_status": self.monitored_models[model_name]["status"]
        }
    
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_type: str = "classification"
    ) -> PerformanceMetrics:
        """Calculate performance metrics from predictions"""
        metrics = PerformanceMetrics(
            model_name="",
            model_version="",
            timestamp=datetime.utcnow()
        )
        
        if model_type == "classification":
            metrics.accuracy = accuracy_score(y_true, y_pred)
            metrics.precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics.recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics.f1_score = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            # ROC AUC for binary classification
            if len(np.unique(y_true)) == 2:
                try:
                    metrics.roc_auc = roc_auc_score(y_true, y_pred)
                except:
                    pass
            
            # Confusion matrix
            metrics.confusion_matrix = confusion_matrix(y_true, y_pred).tolist()
            
        elif model_type == "regression":
            metrics.mse = mean_squared_error(y_true, y_pred)
            metrics.mae = mean_absolute_error(y_true, y_pred)
            metrics.rmse = np.sqrt(metrics.mse)
            metrics.r2 = r2_score(y_true, y_pred)
            
            # MAPE
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if np.all(y_true != 0):
                    metrics.mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        metrics.sample_count = len(y_true)
        metrics.prediction_count = len(y_pred)
        
        return metrics


class AlertManager:
    """Manages alerts and notifications"""
    
    def __init__(self):
        self.alerts: List[Alert] = []
        self.alert_handlers: List[Callable] = []
        
    async def create_alert(
        self,
        severity: AlertSeverity,
        title: str,
        message: str,
        **kwargs
    ) -> Alert:
        """Create and send an alert"""
        alert = Alert(
            alert_id=f"alert_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{len(self.alerts)}",
            severity=severity,
            title=title,
            message=message,
            timestamp=datetime.utcnow(),
            **kwargs
        )
        
        self.alerts.append(alert)
        
        # Send to handlers
        for handler in self.alert_handlers:
            try:
                await handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")
        
        logger.info(f"Alert created: {title} ({severity.value})")
        return alert
    
    def register_handler(self, handler: Callable):
        """Register an alert handler"""
        self.alert_handlers.append(handler)
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all unresolved alerts"""
        return [a for a in self.alerts if not a.resolved]
    
    def acknowledge_alert(self, alert_id: str, user: str = "system"):
        """Acknowledge an alert"""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                alert.acknowledged_by = user
                break
    
    def resolve_alert(self, alert_id: str):
        """Resolve an alert"""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                alert.resolved_at = datetime.utcnow()
                break