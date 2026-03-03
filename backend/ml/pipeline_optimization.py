"""
ML Pipeline Optimization System
Provides efficient model inference, caching, load balancing, and artifact management.

This file is the thin orchestrator / backward-compatibility facade.
Heavy implementation lives in the extracted sub-modules:
  - pipeline_types.py    - enums and dataclasses
  - artifact_manager.py  - ModelArtifactManager
  - inference_cache.py   - InferenceCache
  - load_balancer.py     - LoadBalancer

All names that existed in the original module are re-exported here so that
any existing ``from backend.ml.pipeline_optimization import X`` statement
continues to work unchanged.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import numpy as np

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Sub-module imports (with fallbacks for direct-file execution)
# ---------------------------------------------------------------------------

try:
    from backend.ml.pipeline_types import (
        InferenceMetrics,
        LoadBalancingConfig,
        ModelFormat,
        OptimizationStrategy,
    )
    from backend.ml.artifact_manager import ModelArtifactManager
    from backend.ml.inference_cache import InferenceCache
    from backend.ml.load_balancer import LoadBalancer
except ImportError:  # pragma: no cover
    from pipeline_types import (  # type: ignore[no-redef]
        InferenceMetrics,
        LoadBalancingConfig,
        ModelFormat,
        OptimizationStrategy,
    )
    from artifact_manager import ModelArtifactManager  # type: ignore[no-redef]
    from inference_cache import InferenceCache  # type: ignore[no-redef]
    from load_balancer import LoadBalancer  # type: ignore[no-redef]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MLPipelineOptimizer - orchestrator (kept in this file)
# ---------------------------------------------------------------------------

class MLPipelineOptimizer:
    """
    Main ML pipeline optimization system.

    Orchestrates ModelArtifactManager, InferenceCache, and LoadBalancer.
    """

    def __init__(
        self,
        storage_path: str = "/app/pipeline_optimization",
        enable_caching: bool = True,
        enable_load_balancing: bool = False,
    ) -> None:
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Components (injected via sub-modules)
        self.artifact_manager = ModelArtifactManager(str(self.storage_path / "artifacts"))

        self.cache: Optional[InferenceCache] = None
        if enable_caching:
            self.cache = InferenceCache()

        self.load_balancer: Optional[LoadBalancer] = None
        if enable_load_balancing:
            self.load_balancer = LoadBalancer()
            self.load_balancer.start_health_checks()

        # Optimization strategies flag map
        self.optimization_strategies: Dict[str, bool] = {
            OptimizationStrategy.CACHING.value: enable_caching,
            OptimizationStrategy.PARALLELIZATION.value: True,
            OptimizationStrategy.QUANTIZATION.value: True,
            OptimizationStrategy.BATCHING.value: True,
            OptimizationStrategy.LOAD_BALANCING.value: enable_load_balancing,
            OptimizationStrategy.PREPROCESSING_CACHE.value: True,
        }

        # Performance tracking
        self.inference_metrics: List[InferenceMetrics] = []
        self.preprocessing_cache: Dict[str, Any] = {}

        # Execution pools
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.process_pool = ProcessPoolExecutor(max_workers=2)

        logger.info("ML Pipeline Optimizer initialized")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def optimized_inference(
        self,
        model_name: str,
        input_data: np.ndarray,
        model_version: str = "latest",
        optimization_level: str = "balanced",
        batch_size: Optional[int] = None,
        use_cache: bool = True,
    ) -> Tuple[np.ndarray, InferenceMetrics]:
        """
        Perform optimized model inference.

        Args:
            model_name: Name of the model
            input_data: Input data for inference
            model_version: Model version to use
            optimization_level: Level of optimization (fast, balanced, accurate)
            batch_size: Batch size for inference
            use_cache: Whether to use caching

        Returns:
            Predictions and inference metrics
        """

        start_time = time.time()

        cache_key: Optional[str] = None
        if use_cache and self.cache:
            cache_key = self.cache.generate_cache_key(model_name, input_data)
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                inference_time = (time.time() - start_time) * 1000
                metrics = InferenceMetrics(
                    model_name=model_name,
                    timestamp=datetime.now(timezone.utc),
                    batch_size=len(input_data),
                    inference_time_ms=inference_time,
                    preprocessing_time_ms=0,
                    postprocessing_time_ms=0,
                    total_time_ms=inference_time,
                    memory_usage_mb=0,
                    cpu_usage_percent=0,
                    gpu_usage_percent=None,
                    throughput_samples_per_sec=len(input_data) / (inference_time / 1000),
                    cache_hit_ratio=1.0,
                )
                return cached_result, metrics

        preprocessing_start = time.time()
        processed_data = await self._optimized_preprocessing(input_data, model_name)
        preprocessing_time = (time.time() - preprocessing_start) * 1000

        inference_start = time.time()

        model_artifact_id = f"{model_name}_{model_version}_pytorch"
        optimization = self._select_optimization(optimization_level)

        model = self.artifact_manager.load_artifact(model_artifact_id, optimization)
        if model is None:
            raise ValueError(f"Could not load model {model_name}")

        if batch_size and len(processed_data) > batch_size:
            predictions = await self._batch_inference(model, processed_data, batch_size)
        else:
            predictions = await self._single_inference(model, processed_data)

        inference_time = (time.time() - inference_start) * 1000

        postprocessing_start = time.time()
        final_predictions = await self._optimized_postprocessing(predictions, model_name)
        postprocessing_time = (time.time() - postprocessing_start) * 1000

        total_time = (time.time() - start_time) * 1000

        if use_cache and self.cache and cache_key:
            self.cache.set(cache_key, final_predictions)

        memory_usage = (
            psutil.Process().memory_info().rss / (1024 * 1024) if psutil else 0.0
        )
        cpu_usage = psutil.cpu_percent() if psutil else 0.0

        metrics = InferenceMetrics(
            model_name=model_name,
            timestamp=datetime.now(timezone.utc),
            batch_size=len(input_data),
            inference_time_ms=inference_time,
            preprocessing_time_ms=preprocessing_time,
            postprocessing_time_ms=postprocessing_time,
            total_time_ms=total_time,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage,
            gpu_usage_percent=self._get_gpu_usage(),
            throughput_samples_per_sec=len(input_data) / (total_time / 1000),
            cache_hit_ratio=self.cache.get_stats()['hit_ratio'] if self.cache else 0.0,
        )

        self.inference_metrics.append(metrics)
        return final_predictions, metrics

    async def parallel_inference(
        self,
        models: Dict[str, str],
        input_data: np.ndarray,
        combine_strategy: str = "average",
    ) -> Dict[str, np.ndarray]:
        """Run inference on multiple models in parallel"""

        if not self.optimization_strategies[OptimizationStrategy.PARALLELIZATION.value]:
            results: Dict[str, np.ndarray] = {}
            for model_name, model_version in models.items():
                pred, _ = await self.optimized_inference(model_name, input_data, model_version)
                results[model_name] = pred
            return results

        tasks = []
        for model_name, model_version in models.items():
            task = asyncio.create_task(
                self.optimized_inference(model_name, input_data, model_version)
            )
            tasks.append((model_name, task))

        results = {}
        for model_name, task in tasks:
            pred, _ = await task
            results[model_name] = pred

        return results

    def register_model_for_optimization(
        self,
        model_name: str,
        model_version: str,
        model_object: Any,
        model_format: ModelFormat,
    ) -> str:
        """Register model and create optimized versions"""

        return self.artifact_manager.store_artifact(
            model_name=model_name,
            model_version=model_version,
            model_object=model_object,
            model_format=model_format,
        )

    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report"""

        report: Dict[str, Any] = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'optimization_strategies': self.optimization_strategies,
            'cache_stats': self.cache.get_stats() if self.cache else {},
            'load_balancer_stats': (
                self.load_balancer.get_worker_stats() if self.load_balancer else {}
            ),
            'recent_inference_metrics': [],
            'performance_summary': {},
            'artifact_summary': {},
        }

        recent_metrics = self.inference_metrics[-100:]
        report['recent_inference_metrics'] = [m.to_dict() for m in recent_metrics]

        if recent_metrics:
            report['performance_summary'] = {
                'average_inference_time_ms': np.mean(
                    [m.inference_time_ms for m in recent_metrics]
                ),
                'average_total_time_ms': np.mean([m.total_time_ms for m in recent_metrics]),
                'average_throughput': np.mean(
                    [m.throughput_samples_per_sec for m in recent_metrics]
                ),
                'p95_inference_time_ms': np.percentile(
                    [m.inference_time_ms for m in recent_metrics], 95
                ),
                'average_memory_usage_mb': np.mean(
                    [m.memory_usage_mb for m in recent_metrics]
                ),
                'average_cpu_usage': np.mean([m.cpu_usage_percent for m in recent_metrics]),
            }

        report['artifact_summary'] = {
            'total_artifacts': len(self.artifact_manager.artifacts),
            'total_storage_mb': sum(
                artifact.get('original_size_mb', 0)
                for artifact in self.artifact_manager.artifacts.values()
            ),
        }

        return report

    def cleanup(self) -> None:
        """Clean up resources"""

        if self.load_balancer:
            self.load_balancer.stop_health_checks()

        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)

        logger.info("ML Pipeline Optimizer cleaned up")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _optimized_preprocessing(
        self, input_data: np.ndarray, model_name: str
    ) -> np.ndarray:
        """Optimized preprocessing with caching"""

        if not self.optimization_strategies[OptimizationStrategy.PREPROCESSING_CACHE.value]:
            return self._basic_preprocessing(input_data)

        data_hash = hashlib.md5(input_data.tobytes()).hexdigest()
        cache_key = f"preprocessing:{model_name}:{data_hash}"

        if cache_key in self.preprocessing_cache:
            return self.preprocessing_cache[cache_key]

        processed_data = self._basic_preprocessing(input_data)
        self.preprocessing_cache[cache_key] = processed_data

        if len(self.preprocessing_cache) > 1000:
            oldest_keys = list(self.preprocessing_cache.keys())[:100]
            for key in oldest_keys:
                del self.preprocessing_cache[key]

        return processed_data

    def _basic_preprocessing(self, input_data: np.ndarray) -> np.ndarray:
        """Basic preprocessing operations"""

        if len(input_data.shape) == 2:
            mean = np.mean(input_data, axis=0)
            std = np.std(input_data, axis=0)
            return (input_data - mean) / (std + 1e-8)

        return input_data

    async def _single_inference(self, model: Any, input_data: np.ndarray) -> np.ndarray:
        """Single model inference"""

        if hasattr(model, 'predict'):
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self.thread_pool, model.predict, input_data)

        elif hasattr(model, 'forward'):
            try:
                import torch
                model.eval()
                with torch.no_grad():
                    tensor_input = torch.FloatTensor(input_data)
                    output = model(tensor_input)
                    return output.numpy()
            except ImportError:
                pass

        return np.array([0.0] * len(input_data))

    async def _batch_inference(
        self, model: Any, input_data: np.ndarray, batch_size: int
    ) -> np.ndarray:
        """Batch model inference for large inputs"""

        num_samples = len(input_data)
        num_batches = (num_samples + batch_size - 1) // batch_size

        batch_results = []
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)
            batch_data = input_data[start_idx:end_idx]
            batch_prediction = await self._single_inference(model, batch_data)
            batch_results.append(batch_prediction)

        return np.concatenate(batch_results)

    async def _optimized_postprocessing(
        self, predictions: np.ndarray, model_name: str
    ) -> np.ndarray:
        """Optimized postprocessing"""

        if 'classification' in model_name.lower():
            exp_pred = np.exp(predictions - np.max(predictions, axis=1, keepdims=True))
            return exp_pred / np.sum(exp_pred, axis=1, keepdims=True)

        return predictions

    def _select_optimization(self, optimization_level: str) -> str:
        """Select optimization strategy based on level"""

        if optimization_level == "fast":
            return "quantized"
        elif optimization_level == "accurate":
            return "original"
        else:
            return "onnx"

    def _get_gpu_usage(self) -> Optional[float]:
        """Get GPU usage if available"""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return float(info.gpu)
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_pipeline_optimizer: Optional[MLPipelineOptimizer] = None


def get_pipeline_optimizer() -> MLPipelineOptimizer:
    """Get global pipeline optimizer instance"""
    global _pipeline_optimizer
    if _pipeline_optimizer is None:
        _pipeline_optimizer = MLPipelineOptimizer()
    return _pipeline_optimizer


# ---------------------------------------------------------------------------
# Re-export everything so ``from backend.ml.pipeline_optimization import X`` works
# ---------------------------------------------------------------------------

__all__ = [
    "OptimizationStrategy",
    "ModelFormat",
    "InferenceMetrics",
    "LoadBalancingConfig",
    "ModelArtifactManager",
    "InferenceCache",
    "LoadBalancer",
    "MLPipelineOptimizer",
    "get_pipeline_optimizer",
]
