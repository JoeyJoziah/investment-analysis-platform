"""
Pipeline types: enums and dataclasses for pipeline_optimization sub-modules.

Extracted from pipeline_optimization.py to keep it under 500 lines.
Original import path (backend.ml.pipeline_optimization) remains fully backward-compatible
because pipeline_optimization.py re-exports everything from here.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional


class OptimizationStrategy(Enum):
    """Pipeline optimization strategies"""
    CACHING = "caching"
    PARALLELIZATION = "parallelization"
    QUANTIZATION = "quantization"
    BATCHING = "batching"
    LOAD_BALANCING = "load_balancing"
    PREPROCESSING_CACHE = "preprocessing_cache"


class ModelFormat(Enum):
    """Supported model formats"""
    PYTORCH = "pytorch"
    ONNX = "onnx"
    TENSORRT = "tensorrt"
    SKLEARN_JOBLIB = "sklearn_joblib"
    XGBOOST = "xgboost"
    PICKLE = "pickle"


@dataclass
class InferenceMetrics:
    """Inference performance metrics"""
    model_name: str
    timestamp: datetime
    batch_size: int
    inference_time_ms: float
    preprocessing_time_ms: float
    postprocessing_time_ms: float
    total_time_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    gpu_usage_percent: Optional[float]
    throughput_samples_per_sec: float
    cache_hit_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class LoadBalancingConfig:
    """Load balancing configuration"""
    strategy: str = "round_robin"           # round_robin, weighted, least_connections
    health_check_interval: int = 30         # seconds
    max_connections_per_worker: int = 100
    timeout_seconds: int = 30
    retry_attempts: int = 3
    circuit_breaker_threshold: int = 5      # failures before circuit break


__all__ = [
    "OptimizationStrategy",
    "ModelFormat",
    "InferenceMetrics",
    "LoadBalancingConfig",
]
