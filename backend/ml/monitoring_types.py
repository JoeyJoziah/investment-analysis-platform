"""
Monitoring types: enums and dataclasses for model_monitoring sub-modules.

Extracted from model_monitoring.py to keep it under 500 lines.
Original import path (backend.ml.model_monitoring) remains fully backward-compatible
because model_monitoring.py re-exports everything from here.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional


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
    DATA_DRIFT = "data_drift"              # Input feature distribution changes
    PREDICTION_DRIFT = "prediction_drift"  # Model output distribution changes
    CONCEPT_DRIFT = "concept_drift"        # True relationship changes
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


__all__ = [
    "AlertSeverity",
    "ModelHealth",
    "DriftType",
    "PerformanceMetrics",
    "DriftDetectionResult",
    "ModelAlert",
]
