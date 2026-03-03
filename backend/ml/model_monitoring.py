"""
Model Performance Monitoring System
Provides model drift detection, performance degradation monitoring, and automated retraining.

This file is the thin orchestrator / backward-compatibility facade.
Heavy implementation lives in the extracted sub-modules:
  - monitoring_types.py  - enums and dataclasses
  - drift_detection.py   - DriftDetector
  - alert_management.py  - AlertManager

All names that existed in the original module are re-exported here so that
any existing ``from backend.ml.model_monitoring import X`` statement continues
to work unchanged.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        mean_absolute_error,
        mean_squared_error,
        precision_score,
        r2_score,
        recall_score,
    )
except ImportError:  # pragma: no cover
    accuracy_score = precision_score = recall_score = f1_score = None  # type: ignore[assignment]
    mean_squared_error = mean_absolute_error = r2_score = None  # type: ignore[assignment]

try:
    from scipy import stats
except ImportError:  # pragma: no cover
    stats = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Sub-module imports (with fallbacks for direct-file execution)
# ---------------------------------------------------------------------------

try:
    from backend.ml.monitoring_types import (
        AlertSeverity,
        DriftDetectionResult,
        DriftType,
        ModelAlert,
        ModelHealth,
        PerformanceMetrics,
    )
    from backend.ml.drift_detection import DriftDetector
    from backend.ml.alert_management import AlertManager
except ImportError:  # pragma: no cover
    from monitoring_types import (  # type: ignore[no-redef]
        AlertSeverity,
        DriftDetectionResult,
        DriftType,
        ModelAlert,
        ModelHealth,
        PerformanceMetrics,
    )
    from drift_detection import DriftDetector  # type: ignore[no-redef]
    from alert_management import AlertManager  # type: ignore[no-redef]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ModelPerformanceTracker - kept in this file (orchestrator role)
# ---------------------------------------------------------------------------

class ModelPerformanceTracker:
    """Tracks model performance metrics over time"""

    def __init__(self, storage_path: str = "/app/monitoring") -> None:
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.performance_history: Dict[str, List[PerformanceMetrics]] = defaultdict(list)
        self.lock = threading.Lock()

        self._load_performance_history()

    def record_performance(
        self,
        model_name: str,
        model_version: str,
        predictions: np.ndarray,
        true_values: np.ndarray,
        prediction_latency_ms: Optional[float] = None,
        memory_usage_mb: Optional[float] = None,
        cpu_usage_percent: Optional[float] = None,
    ) -> PerformanceMetrics:
        """Record model performance metrics"""

        metrics = PerformanceMetrics(
            timestamp=datetime.now(timezone.utc),
            model_name=model_name,
            model_version=model_version,
            sample_size=len(predictions),
            prediction_latency_ms=prediction_latency_ms,
            memory_usage_mb=memory_usage_mb,
            cpu_usage_percent=cpu_usage_percent,
        )

        if self._is_classification_problem(predictions, true_values):
            if accuracy_score is not None:
                metrics.accuracy = accuracy_score(true_values, predictions)
                metrics.precision = precision_score(
                    true_values, predictions, average='weighted', zero_division=0
                )
                metrics.recall = recall_score(
                    true_values, predictions, average='weighted', zero_division=0
                )
                metrics.f1_score = f1_score(
                    true_values, predictions, average='weighted', zero_division=0
                )

            if len(predictions) > 1:
                pred_direction = np.sign(predictions - np.mean(predictions))
                true_direction = np.sign(true_values - np.mean(true_values))
                if accuracy_score is not None:
                    metrics.directional_accuracy = accuracy_score(true_direction, pred_direction)
        else:
            if mean_squared_error is not None:
                metrics.mse = mean_squared_error(true_values, predictions)
                metrics.mae = mean_absolute_error(true_values, predictions)
                metrics.r2_score = r2_score(true_values, predictions)

            if np.std(predictions) > 0:
                metrics.sharpe_ratio = (
                    np.mean(predictions) / np.std(predictions) * np.sqrt(252)
                )

        with self.lock:
            self.performance_history[model_name].append(metrics)
            if len(self.performance_history[model_name]) > 1000:
                self.performance_history[model_name] = (
                    self.performance_history[model_name][-1000:]
                )

        self._save_performance_metrics(metrics)

        logger.info(
            f"Recorded performance for model {model_name}: "
            f"accuracy={metrics.accuracy}, f1={metrics.f1_score}, "
            f"mse={metrics.mse}, r2={metrics.r2_score}"
        )

        return metrics

    def _is_classification_problem(
        self, predictions: np.ndarray, true_values: np.ndarray
    ) -> bool:
        """Detect if this is a classification or regression problem"""
        pred_unique = len(np.unique(predictions))
        true_unique = len(np.unique(true_values))
        return (
            pred_unique <= 20
            and true_unique <= 20
            and np.all(predictions == predictions.astype(int))
            and np.all(true_values == true_values.astype(int))
        )

    def get_performance_trend(
        self,
        model_name: str,
        metric: str = "accuracy",
        days_back: int = 30,
    ) -> pd.DataFrame:
        """Get performance trend for a specific metric"""

        if model_name not in self.performance_history:
            return pd.DataFrame()

        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_back)
        recent_metrics = [
            m for m in self.performance_history[model_name] if m.timestamp >= cutoff_date
        ]

        if not recent_metrics:
            return pd.DataFrame()

        data = []
        for m in recent_metrics:
            metric_value = getattr(m, metric, None)
            if metric_value is not None:
                data.append(
                    {
                        'timestamp': m.timestamp,
                        'model_version': m.model_version,
                        'metric_value': metric_value,
                        'sample_size': m.sample_size,
                    }
                )

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df = df.sort_values('timestamp')
        return df

    def detect_performance_degradation(
        self,
        model_name: str,
        metric: str = "accuracy",
        degradation_threshold: float = 0.05,
        min_samples: int = 10,
    ) -> Optional[Dict[str, Any]]:
        """Detect performance degradation using statistical tests"""

        trend_df = self.get_performance_trend(model_name, metric, days_back=30)

        if len(trend_df) < min_samples:
            return None

        split_point = len(trend_df) // 2
        historical = trend_df.iloc[:split_point]['metric_value'].values
        recent = trend_df.iloc[split_point:]['metric_value'].values

        if len(historical) < 5 or len(recent) < 5:
            return None

        historical_mean = np.mean(historical)
        recent_mean = np.mean(recent)

        if stats is not None:
            t_stat, p_value = stats.ttest_ind(historical, recent)
        else:
            t_stat, p_value = 0.0, 1.0

        pooled_std = np.sqrt(
            (
                (len(historical) - 1) * np.var(historical)
                + (len(recent) - 1) * np.var(recent)
            )
            / (len(historical) + len(recent) - 2)
        )
        cohens_d = (recent_mean - historical_mean) / pooled_std if pooled_std > 0 else 0

        relative_change = (
            (recent_mean - historical_mean) / historical_mean if historical_mean > 0 else 0
        )
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
            'degradation_threshold': degradation_threshold,
        }

    def _load_performance_history(self) -> None:
        """Load performance history from disk"""
        try:
            history_file = self.storage_path / "performance_history.json"
            if history_file.exists():
                with open(history_file, 'r') as fh:
                    data = json.load(fh)

                for model_name, metrics_list in data.items():
                    for metric_data in metrics_list:
                        metric_data['timestamp'] = datetime.fromisoformat(
                            metric_data['timestamp']
                        )
                        self.performance_history[model_name].append(
                            PerformanceMetrics(**metric_data)
                        )

                logger.info(f"Loaded performance history for {len(data)} models")
        except Exception as exc:
            logger.error(f"Error loading performance history: {exc}")

    def _save_performance_metrics(self, metrics: PerformanceMetrics) -> None:
        """Save individual performance metrics"""
        try:
            metrics_file = (
                self.storage_path
                / f"metrics_{metrics.model_name}_{datetime.now(timezone.utc).strftime('%Y%m%d')}.json"
            )

            daily_metrics: List[Dict[str, Any]] = []
            if metrics_file.exists():
                with open(metrics_file, 'r') as fh:
                    daily_metrics = json.load(fh)

            daily_metrics.append(metrics.to_dict())

            with open(metrics_file, 'w') as fh:
                json.dump(daily_metrics, fh, indent=2)
        except Exception as exc:
            logger.error(f"Error saving performance metrics: {exc}")


# ---------------------------------------------------------------------------
# ModelMonitor - orchestrator (kept in this file)
# ---------------------------------------------------------------------------

class ModelMonitor:
    """
    Comprehensive model monitoring system.

    Orchestrates ModelPerformanceTracker, DriftDetector, and AlertManager.
    """

    def __init__(
        self,
        storage_path: str = "/app/monitoring",
        monitoring_interval_hours: float = 1.0,
    ) -> None:
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.monitoring_interval_hours = monitoring_interval_hours

        # Components (injected via sub-modules)
        self.performance_tracker = ModelPerformanceTracker(str(self.storage_path / "performance"))
        self.drift_detector = DriftDetector()
        self.alert_manager = AlertManager(str(self.storage_path / "alerts"))

        # Monitoring state
        self.monitored_models: Dict[str, Dict[str, Any]] = {}
        self.is_monitoring = False
        self.monitoring_thread: Optional[threading.Thread] = None

        self.performance_thresholds = {
            'accuracy_degradation': 0.05,
            'f1_degradation': 0.05,
            'mse_increase': 0.2,
            'latency_increase': 0.5,
        }

        logger.info("Model monitor initialized")

    # Convenience accessor so callers can do monitor.alerts.
    # The setter is a no-op: alert_manager.alerts is the authoritative list.
    # It exists so that test code can do ``monitor.alerts = ...`` without error.
    @property
    def alerts(self) -> List[ModelAlert]:
        return self.alert_manager.alerts

    @alerts.setter
    def alerts(self, value: List[ModelAlert]) -> None:  # no-op: list is owned by alert_manager
        pass

    def register_model(
        self,
        model_name: str,
        model_version: str,
        monitoring_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a model for monitoring"""

        config = monitoring_config or {}
        self.monitored_models[model_name] = {
            'version': model_version,
            'registered_at': datetime.now(timezone.utc),
            'last_monitored': None,
            'monitoring_config': config,
            'health_status': ModelHealth.UNKNOWN,
            'performance_baseline': None,
        }
        logger.info(f"Registered model {model_name} v{model_version} for monitoring")

    def start_monitoring(self) -> None:
        """Start continuous monitoring"""

        if self.is_monitoring:
            logger.warning("Monitoring already started")
            return

        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self.monitoring_thread.start()
        logger.info("Started continuous model monitoring")

    def stop_monitoring(self) -> None:
        """Stop continuous monitoring"""

        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Stopped model monitoring")

    def monitor_model_performance(
        self,
        model_name: str,
        predictions: np.ndarray,
        true_values: np.ndarray,
        features: Optional[Dict[str, np.ndarray]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Monitor a single model's performance"""

        if model_name not in self.monitored_models:
            logger.warning(f"Model {model_name} not registered for monitoring")
            return {}

        model_info = self.monitored_models[model_name]
        monitoring_results: Dict[str, Any] = {}

        performance_metrics = self.performance_tracker.record_performance(
            model_name=model_name,
            model_version=model_info['version'],
            predictions=predictions,
            true_values=true_values,
            prediction_latency_ms=metadata.get('latency_ms') if metadata else None,
            memory_usage_mb=metadata.get('memory_mb') if metadata else None,
            cpu_usage_percent=metadata.get('cpu_percent') if metadata else None,
        )

        monitoring_results['performance_metrics'] = performance_metrics

        degradation_result = self.performance_tracker.detect_performance_degradation(
            model_name=model_name,
            metric='accuracy' if performance_metrics.accuracy else 'mse',
            degradation_threshold=self.performance_thresholds.get('accuracy_degradation', 0.05),
        )

        if degradation_result and degradation_result['is_significant_degradation']:
            self.alert_manager.create_alert(
                model_name=model_name,
                alert_type="performance_degradation",
                severity=AlertSeverity.WARNING,
                message=(
                    f"Performance degradation detected: "
                    f"{degradation_result['relative_change']:.1%} decrease"
                ),
                details=degradation_result,
            )

        monitoring_results['degradation_check'] = degradation_result

        if features:
            if model_info['performance_baseline'] is None:
                for feature_name, feature_data in features.items():
                    self.drift_detector.update_reference_distribution(
                        model_name, feature_name, feature_data
                    )
                self.drift_detector.update_reference_distribution(
                    model_name, 'predictions', predictions
                )
                model_info['performance_baseline'] = performance_metrics

            data_drift_result = self.drift_detector.detect_data_drift(
                model_name=model_name, current_features=features
            )
            monitoring_results['data_drift'] = data_drift_result

            if data_drift_result.is_drift_detected:
                self.alert_manager.create_alert(
                    model_name=model_name,
                    alert_type="data_drift",
                    severity=AlertSeverity.WARNING,
                    message=f"Data drift detected: drift_score={data_drift_result.drift_score:.3f}",
                    details=data_drift_result.to_dict(),
                )

            if model_info['performance_baseline']:
                prediction_drift_result = self.drift_detector.detect_prediction_drift(
                    model_name=model_name, current_predictions=predictions
                )
                monitoring_results['prediction_drift'] = prediction_drift_result

                if prediction_drift_result.is_drift_detected:
                    self.alert_manager.create_alert(
                        model_name=model_name,
                        alert_type="prediction_drift",
                        severity=AlertSeverity.WARNING,
                        message=(
                            f"Prediction drift detected: "
                            f"drift_score={prediction_drift_result.drift_score:.3f}"
                        ),
                        details=prediction_drift_result.to_dict(),
                    )

        health_status = self._assess_model_health(model_name, monitoring_results)
        model_info['health_status'] = health_status
        model_info['last_monitored'] = datetime.now(timezone.utc)
        monitoring_results['health_status'] = health_status

        return monitoring_results

    def _assess_model_health(
        self, model_name: str, monitoring_results: Dict[str, Any]
    ) -> ModelHealth:
        """Assess overall model health"""

        active_alerts = self.alert_manager.get_active_alerts(model_name)
        critical_alerts = [a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]
        error_alerts = [a for a in active_alerts if a.severity == AlertSeverity.ERROR]

        if critical_alerts or error_alerts:
            return ModelHealth.FAILING

        performance = monitoring_results.get('performance_metrics')
        if performance:
            if (performance.accuracy and performance.accuracy < 0.5) or (
                performance.f1_score and performance.f1_score < 0.5
            ):
                return ModelHealth.FAILING

        data_drift = monitoring_results.get('data_drift')
        prediction_drift = monitoring_results.get('prediction_drift')

        if (data_drift and data_drift.is_drift_detected) or (
            prediction_drift and prediction_drift.is_drift_detected
        ):
            return ModelHealth.DEGRADED

        degradation = monitoring_results.get('degradation_check')
        if degradation and degradation['is_significant_degradation']:
            return ModelHealth.DEGRADED

        return ModelHealth.HEALTHY

    def _monitoring_loop(self) -> None:
        """Continuous monitoring loop"""

        while self.is_monitoring:
            try:
                for model_name, model_info in self.monitored_models.items():
                    if model_info['last_monitored']:
                        time_since_monitor = (
                            datetime.now(timezone.utc) - model_info['last_monitored']
                        )
                        if (
                            time_since_monitor.total_seconds()
                            < self.monitoring_interval_hours * 3600
                        ):
                            continue

                    self._automated_model_check(model_name, model_info)

                # Blocking sleep OK: runs in dedicated daemon thread
                time.sleep(self.monitoring_interval_hours * 3600)

            except Exception as exc:
                logger.error(f"Error in monitoring loop: {exc}")
                time.sleep(60)  # Blocking sleep OK: runs in dedicated daemon thread

    def _automated_model_check(
        self, model_name: str, model_info: Dict[str, Any]
    ) -> None:
        """Perform automated checks for a model"""

        try:
            if model_info['last_monitored']:
                hours_since_update = (
                    datetime.now(timezone.utc) - model_info['last_monitored']
                ).total_seconds() / 3600

                if hours_since_update > 24:
                    self.alert_manager.create_alert(
                        model_name=model_name,
                        alert_type="stale_data",
                        severity=AlertSeverity.WARNING,
                        message=(
                            f"No performance data received for {hours_since_update:.1f} hours"
                        ),
                        details={'hours_since_update': hours_since_update},
                    )

            active_alerts = self.alert_manager.get_active_alerts(model_name)
            if len(active_alerts) > 10:
                self.alert_manager.create_alert(
                    model_name=model_name,
                    alert_type="alert_overflow",
                    severity=AlertSeverity.CRITICAL,
                    message=f"Too many unresolved alerts: {len(active_alerts)}",
                    details={'active_alert_count': len(active_alerts)},
                )

            model_info['last_monitored'] = datetime.now(timezone.utc)

        except Exception as exc:
            logger.error(f"Error in automated check for {model_name}: {exc}")

    def get_model_health_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive model health dashboard"""

        dashboard: Dict[str, Any] = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'monitored_models_count': len(self.monitored_models),
            'models': {},
            'overall_health': {},
            'recent_alerts': [],
        }

        health_counts: Dict[str, int] = defaultdict(int)

        for model_name, model_info in self.monitored_models.items():
            performance_trend = self.performance_tracker.get_performance_trend(
                model_name, days_back=7
            )
            active_alerts = self.alert_manager.get_active_alerts(model_name)

            model_dashboard: Dict[str, Any] = {
                'version': model_info['version'],
                'health_status': model_info['health_status'].value,
                'last_monitored': (
                    model_info['last_monitored'].isoformat()
                    if model_info['last_monitored']
                    else None
                ),
                'active_alerts_count': len(active_alerts),
                'performance_trend_points': len(performance_trend),
                'recent_performance': {},
            }

            if not performance_trend.empty:
                latest_performance = performance_trend.iloc[-1]
                model_dashboard['recent_performance'] = {
                    'metric_value': latest_performance['metric_value'],
                    'sample_size': latest_performance['sample_size'],
                    'timestamp': latest_performance['timestamp'].isoformat(),
                }

            dashboard['models'][model_name] = model_dashboard
            health_counts[model_info['health_status'].value] += 1

        dashboard['overall_health'] = dict(health_counts)

        recent_cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        recent_alerts = [
            alert.to_dict()
            for alert in self.alert_manager.alerts
            if alert.timestamp >= recent_cutoff
        ]
        dashboard['recent_alerts'] = recent_alerts[-20:]

        return dashboard

    def generate_monitoring_report(
        self, model_name: str, days_back: int = 30
    ) -> Dict[str, Any]:
        """Generate comprehensive monitoring report for a model"""

        if model_name not in self.monitored_models:
            return {}

        model_info = self.monitored_models[model_name]

        report: Dict[str, Any] = {
            'model_name': model_name,
            'model_version': model_info['version'],
            'report_period_days': days_back,
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'summary': {},
            'performance_analysis': {},
            'drift_analysis': {},
            'alert_summary': {},
            'recommendations': [],
        }

        performance_trend = self.performance_tracker.get_performance_trend(
            model_name, days_back=days_back
        )

        if not performance_trend.empty:
            report['performance_analysis'] = {
                'data_points': len(performance_trend),
                'latest_performance': performance_trend.iloc[-1]['metric_value'],
                'average_performance': performance_trend['metric_value'].mean(),
                'performance_stability': performance_trend['metric_value'].std(),
                'trend_direction': (
                    'improving'
                    if performance_trend['metric_value'].iloc[-1]
                    > performance_trend['metric_value'].iloc[0]
                    else 'declining'
                ),
            }

        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_back)
        period_alerts = [
            alert
            for alert in self.alert_manager.alerts
            if alert.model_name == model_name and alert.timestamp >= cutoff_date
        ]

        alert_counts: Dict[str, int] = defaultdict(int)
        for alert in period_alerts:
            alert_counts[alert.alert_type] += 1

        report['alert_summary'] = {
            'total_alerts': len(period_alerts),
            'alerts_by_type': dict(alert_counts),
            'resolved_alerts': len([a for a in period_alerts if a.is_resolved]),
            'active_alerts': len([a for a in period_alerts if not a.is_resolved]),
        }

        report['recommendations'] = self._generate_recommendations(model_name, report)
        return report

    def _generate_recommendations(
        self, model_name: str, report: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable recommendations based on monitoring data"""

        recommendations: List[str] = []

        performance = report.get('performance_analysis', {})
        if performance.get('trend_direction') == 'declining':
            recommendations.append(
                "Consider retraining the model due to declining performance trend"
            )

        if performance.get('performance_stability', 0) > 0.1:
            recommendations.append(
                "High performance variability detected - investigate data quality"
            )

        alert_summary = report.get('alert_summary', {})
        if alert_summary.get('total_alerts', 0) > 10:
            recommendations.append("High alert volume - review monitoring thresholds")

        if 'data_drift' in alert_summary.get('alerts_by_type', {}):
            recommendations.append(
                "Data drift detected - consider feature engineering or model updates"
            )

        if 'performance_degradation' in alert_summary.get('alerts_by_type', {}):
            recommendations.append(
                "Performance degradation alerts - schedule model retraining"
            )

        model_info = self.monitored_models[model_name]
        if model_info['health_status'] == ModelHealth.DEGRADED:
            recommendations.append("Model health is degraded - immediate attention required")
        elif model_info['health_status'] == ModelHealth.FAILING:
            recommendations.append(
                "Model is failing - consider taking offline and investigating"
            )

        return recommendations


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_model_monitor: Optional[ModelMonitor] = None


def get_model_monitor() -> ModelMonitor:
    """Get global model monitor instance"""
    global _model_monitor
    if _model_monitor is None:
        _model_monitor = ModelMonitor()
    return _model_monitor


# ---------------------------------------------------------------------------
# Re-export everything so ``from backend.ml.model_monitoring import X`` works
# ---------------------------------------------------------------------------

__all__ = [
    "AlertSeverity",
    "ModelHealth",
    "DriftType",
    "PerformanceMetrics",
    "DriftDetectionResult",
    "ModelAlert",
    "DriftDetector",
    "AlertManager",
    "ModelPerformanceTracker",
    "ModelMonitor",
    "get_model_monitor",
]
