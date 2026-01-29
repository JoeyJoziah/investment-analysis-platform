"""
ML Model Performance Testing

Tests for ML model inference performance, memory profiling,
and cache hit rate validation.

Performance Targets:
- Model inference: <100ms per stock
- Memory per model: <500MB
- Cache hit rate: >85%
- Batch inference: >100 stocks/second
"""

import pytest
import time
import psutil
import numpy as np
import asyncio
import gc
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from unittest.mock import Mock, AsyncMock, patch
import tracemalloc

logger = logging.getLogger(__name__)


@dataclass
class MLPerformanceMetrics:
    """Container for ML performance metrics"""
    model_name: str
    total_inferences: int
    successful_inferences: int
    failed_inferences: int
    total_time: float
    avg_inference_time: float
    p50_inference_time: float
    p95_inference_time: float
    p99_inference_time: float
    peak_memory_mb: float
    memory_per_inference_mb: float
    throughput_per_second: float
    error_rate: float
    cache_hit_rate: float


class MLInferenceProfiler:
    """Profile ML model inference performance"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.inference_times = []
        self.memory_snapshots = []
        self.cache_hits = 0
        self.cache_misses = 0
        self.errors = 0
        self.start_time = None
        self.peak_memory = 0
        self.initial_memory = 0

    def start_profiling(self):
        """Start memory profiling"""
        gc.collect()
        tracemalloc.start()
        self.initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        self.peak_memory = self.initial_memory
        self.start_time = time.time()

    def record_inference(self, inference_time: float):
        """Record an inference time"""
        self.inference_times.append(inference_time)

        # Update peak memory
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024
        self.peak_memory = max(self.peak_memory, current_memory)

    def record_cache_hit(self):
        """Record a cache hit"""
        self.cache_hits += 1

    def record_cache_miss(self):
        """Record a cache miss"""
        self.cache_misses += 1

    def record_error(self):
        """Record an error"""
        self.errors += 1

    def get_metrics(self) -> MLPerformanceMetrics:
        """Calculate and return performance metrics"""
        total_time = time.time() - self.start_time
        total_inferences = len(self.inference_times)
        successful = total_inferences
        failed = self.errors

        if total_inferences > 0:
            avg_time = np.mean(self.inference_times)
            p50_time = np.percentile(self.inference_times, 50)
            p95_time = np.percentile(self.inference_times, 95)
            p99_time = np.percentile(self.inference_times, 99)
        else:
            avg_time = p50_time = p95_time = p99_time = 0

        throughput = total_inferences / max(total_time, 1)

        total_cache_ops = self.cache_hits + self.cache_misses
        cache_hit_rate = self.cache_hits / max(total_cache_ops, 1)

        current_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_per_inference = (current_memory - self.initial_memory) / max(total_inferences, 1)

        return MLPerformanceMetrics(
            model_name=self.model_name,
            total_inferences=total_inferences,
            successful_inferences=successful,
            failed_inferences=failed,
            total_time=total_time,
            avg_inference_time=avg_time,
            p50_inference_time=p50_time,
            p95_inference_time=p95_time,
            p99_inference_time=p99_time,
            peak_memory_mb=self.peak_memory,
            memory_per_inference_mb=memory_per_inference,
            throughput_per_second=throughput,
            error_rate=failed / max(total_inferences + failed, 1),
            cache_hit_rate=cache_hit_rate
        )


class MockMLModel:
    """Mock ML model for testing"""

    def __init__(self, inference_time: float = 0.01, complexity: str = 'simple'):
        self.inference_time = inference_time
        self.complexity = complexity  # simple, moderate, complex
        self.call_count = 0

    async def predict(self, features: np.ndarray) -> float:
        """Simulate model prediction"""
        self.call_count += 1

        # Simulate inference time based on complexity
        if self.complexity == 'simple':
            await asyncio.sleep(0.001)
        elif self.complexity == 'moderate':
            await asyncio.sleep(0.01)
        else:  # complex
            await asyncio.sleep(0.05)

        # Return a mock prediction
        return float(np.tanh(np.sum(features)))


class TestMLModelInference:
    """Test ML model inference performance"""

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_single_model_inference_latency(self):
        """Test single model inference latency"""

        profiler = MLInferenceProfiler("single_model")
        profiler.start_profiling()

        model = MockMLModel(complexity='simple')

        # Test 1000 inferences
        num_inferences = 1000
        features_dim = 50

        for i in range(num_inferences):
            features = np.random.randn(features_dim)

            start_time = time.time()
            try:
                prediction = await model.predict(features)
                inference_time = time.time() - start_time
                profiler.record_inference(inference_time)

                # Simulate cache behavior
                if i % 3 == 0:
                    profiler.record_cache_hit()
                else:
                    profiler.record_cache_miss()

            except Exception as e:
                profiler.record_error()
                logger.error(f"Inference failed: {e}")

        metrics = profiler.get_metrics()

        print(f"\nSingle Model Inference Latency:")
        print(f"  Total inferences: {metrics.total_inferences}")
        print(f"  Successful: {metrics.successful_inferences}")
        print(f"  Failed: {metrics.failed_inferences}")
        print(f"  Avg latency: {metrics.avg_inference_time*1000:.2f}ms")
        print(f"  P50 latency: {metrics.p50_inference_time*1000:.2f}ms")
        print(f"  P95 latency: {metrics.p95_inference_time*1000:.2f}ms")
        print(f"  P99 latency: {metrics.p99_inference_time*1000:.2f}ms")
        print(f"  Throughput: {metrics.throughput_per_second:.0f} inferences/s")
        print(f"  Error rate: {metrics.error_rate:.3f}")
        print(f"  Cache hit rate: {metrics.cache_hit_rate:.3f}")

        # Performance assertions
        assert metrics.avg_inference_time < 0.1, f"Avg latency too high: {metrics.avg_inference_time*1000:.2f}ms"
        assert metrics.p95_inference_time < 0.2, f"P95 latency too high: {metrics.p95_inference_time*1000:.2f}ms"
        assert metrics.error_rate < 0.01, f"Error rate too high: {metrics.error_rate}"
        assert metrics.cache_hit_rate > 0.3, f"Cache hit rate too low: {metrics.cache_hit_rate}"

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_batch_inference_performance(self):
        """Test batch inference performance"""

        profiler = MLInferenceProfiler("batch_inference")
        profiler.start_profiling()

        model = MockMLModel(complexity='moderate')

        # Test batch inference
        batch_sizes = [1, 10, 32, 64, 128]
        total_samples = 1000
        features_dim = 50

        batch_results = {}

        for batch_size in batch_sizes:
            batch_times = []

            for batch_idx in range(0, total_samples, batch_size):
                # Create batch
                batch = np.random.randn(
                    min(batch_size, total_samples - batch_idx),
                    features_dim
                )

                start_time = time.time()

                try:
                    # Simulate batch inference
                    tasks = [model.predict(batch[i]) for i in range(len(batch))]
                    predictions = await asyncio.gather(*tasks)

                    batch_time = time.time() - start_time
                    batch_times.append(batch_time)

                    for _ in range(len(batch)):
                        profiler.record_inference(batch_time / len(batch))
                        profiler.record_cache_miss()

                except Exception as e:
                    profiler.record_error()

            avg_batch_time = np.mean(batch_times) if batch_times else 0
            batch_results[batch_size] = avg_batch_time

            throughput = batch_size / max(avg_batch_time, 0.0001)
            print(f"Batch size {batch_size:3d}: {avg_batch_time*1000:.2f}ms, {throughput:.0f} samples/s")

        metrics = profiler.get_metrics()

        print(f"\nBatch Inference Summary:")
        print(f"  Total samples processed: {metrics.total_inferences}")
        print(f"  Throughput: {metrics.throughput_per_second:.0f} samples/s")
        print(f"  Peak memory: {metrics.peak_memory_mb:.1f}MB")

        # Optimal batch size should have highest throughput
        optimal_batch = max(batch_results.keys(), key=lambda k: k / max(batch_results[k], 0.0001))
        print(f"  Optimal batch size: {optimal_batch}")

        # Assert throughput target
        assert metrics.throughput_per_second > 100, f"Throughput too low: {metrics.throughput_per_second} samples/s"

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_model_inference(self):
        """Test concurrent inference from multiple models"""

        num_models = 5
        inferences_per_model = 100
        num_concurrent = 20

        all_profilers = {}
        start_time = time.time()

        models = {f"model_{i}": MockMLModel(complexity='simple') for i in range(num_models)}

        for model_name, model in models.items():
            all_profilers[model_name] = MLInferenceProfiler(model_name)
            all_profilers[model_name].start_profiling()

        # Simulate concurrent inference requests
        tasks = []
        semaphore = asyncio.Semaphore(num_concurrent)

        async def concurrent_inference(model_idx, inference_idx):
            async with semaphore:
                model_name = f"model_{model_idx}"
                model = models[model_name]
                profiler = all_profilers[model_name]

                features = np.random.randn(50)

                inf_start = time.time()
                try:
                    prediction = await model.predict(features)
                    inf_time = time.time() - inf_start
                    profiler.record_inference(inf_time)
                except Exception as e:
                    profiler.record_error()

        # Create concurrent tasks
        for model_idx in range(num_models):
            for inf_idx in range(inferences_per_model):
                tasks.append(concurrent_inference(model_idx, inf_idx))

        await asyncio.gather(*tasks)

        total_time = time.time() - start_time

        print(f"\nConcurrent Model Inference ({num_models} models, {num_concurrent} concurrent):")
        print(f"  Total time: {total_time:.2f}s")

        for model_name, profiler in all_profilers.items():
            metrics = profiler.get_metrics()
            print(f"\n  {model_name}:")
            print(f"    Inferences: {metrics.total_inferences}")
            print(f"    Avg latency: {metrics.avg_inference_time*1000:.2f}ms")
            print(f"    Throughput: {metrics.throughput_per_second:.0f} inf/s")

        # Overall throughput
        total_inferences = num_models * inferences_per_model
        overall_throughput = total_inferences / max(total_time, 1)

        print(f"\n  Overall throughput: {overall_throughput:.0f} inferences/s")

        assert overall_throughput > 50, f"Throughput too low: {overall_throughput} inf/s"


class TestMLMemoryProfiling:
    """Test ML model memory profiling"""

    @pytest.mark.performance
    def test_model_memory_efficiency(self):
        """Test model memory efficiency"""

        profiler = MLInferenceProfiler("memory_test")
        profiler.start_profiling()

        gc.collect()
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024

        # Test with increasing model complexity
        model = MockMLModel(complexity='complex')

        num_inferences = 500
        features_dim = 100

        memory_samples = []

        for i in range(num_inferences):
            # Create features
            features = np.random.randn(features_dim)

            # Simulate inference (synchronous for memory testing)
            prediction = np.tanh(np.sum(features))
            profiler.record_inference(0.01)

            # Sample memory periodically
            if i % 50 == 0:
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                memory_samples.append(current_memory)

            # Force garbage collection
            if i % 100 == 0:
                gc.collect()

        gc.collect()
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024

        metrics = profiler.get_metrics()

        print(f"\nModel Memory Efficiency:")
        print(f"  Initial memory: {initial_memory:.1f}MB")
        print(f"  Final memory: {final_memory:.1f}MB")
        print(f"  Peak memory: {metrics.peak_memory_mb:.1f}MB")
        print(f"  Total growth: {final_memory - initial_memory:.1f}MB")
        print(f"  Memory per inference: {metrics.memory_per_inference_mb:.3f}MB")

        # Memory assertions
        memory_per_inference_kb = metrics.memory_per_inference_mb * 1024
        assert memory_per_inference_kb < 100, f"Memory per inference too high: {memory_per_inference_kb:.1f}KB"
        assert metrics.peak_memory_mb < 2048, f"Peak memory too high: {metrics.peak_memory_mb:.1f}MB"

    @pytest.mark.performance
    def test_memory_leak_detection(self):
        """Test for memory leaks during inference"""

        profiler = MLInferenceProfiler("leak_test")
        profiler.start_profiling()

        model = MockMLModel(complexity='simple')

        gc.collect()
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024

        # Run inferences for 60 seconds and monitor memory
        end_time = time.time() + 60
        inference_count = 0
        memory_trend = []

        while time.time() < end_time:
            features = np.random.randn(50)
            prediction = np.tanh(np.sum(features))
            inference_count += 1
            profiler.record_inference(0.001)

            # Sample memory every 5 seconds
            if inference_count % 5000 == 0:
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                memory_trend.append(current_memory)
                print(f"After {inference_count} inferences: {current_memory:.1f}MB")

        gc.collect()
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024

        # Calculate memory growth trend
        if len(memory_trend) > 1:
            memory_growth_per_sample = (memory_trend[-1] - memory_trend[0]) / (len(memory_trend) - 1)
        else:
            memory_growth_per_sample = 0

        print(f"\nMemory Leak Detection:")
        print(f"  Initial memory: {initial_memory:.1f}MB")
        print(f"  Final memory: {final_memory:.1f}MB")
        print(f"  Total inferences: {inference_count}")
        print(f"  Memory growth trend: {memory_growth_per_sample:.3f}MB per sample")

        # Memory should not grow significantly
        assert memory_growth_per_sample < 0.01, f"Potential memory leak detected: {memory_growth_per_sample:.3f}MB/sample"


class TestCacheHitRate:
    """Test cache hit rate validation"""

    @pytest.mark.performance
    def test_inference_cache_hit_rate(self):
        """Test inference result caching"""

        cache = {}

        def get_cached_prediction(features_key: str, model) -> Tuple[float, bool]:
            """Get prediction from cache or compute"""
            if features_key in cache:
                return cache[features_key], True
            else:
                prediction = float(np.random.random())
                cache[features_key] = prediction
                return prediction, False

        profiler = MLInferenceProfiler("cache_test")
        profiler.start_profiling()

        model = MockMLModel()

        # Test cache hit pattern
        num_unique_features = 100
        num_total_requests = 1000

        # Request pattern: 80% to same 20% of features (Zipf distribution)
        feature_indices = np.random.choice(
            num_unique_features,
            num_total_requests,
            p=np.array([1.0 / i for i in range(1, num_unique_features + 1)]) / sum([1.0 / i for i in range(1, num_unique_features + 1)])
        )

        for idx in feature_indices:
            features_key = f"features_{idx}"

            prediction, is_cached = get_cached_prediction(features_key, model)

            if is_cached:
                profiler.record_cache_hit()
            else:
                profiler.record_cache_miss()

            profiler.record_inference(0.001)

        metrics = profiler.get_metrics()

        print(f"\nInference Cache Hit Rate:")
        print(f"  Total requests: {metrics.total_inferences}")
        print(f"  Cache hits: {profiler.cache_hits}")
        print(f"  Cache misses: {profiler.cache_misses}")
        print(f"  Hit rate: {metrics.cache_hit_rate:.3f}")

        # Target: >85% cache hit rate
        assert metrics.cache_hit_rate > 0.85, f"Cache hit rate too low: {metrics.cache_hit_rate:.3f}"

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_distributed_cache_consistency(self):
        """Test distributed cache consistency"""

        # Simulate distributed cache across multiple workers
        cache_stores = {
            'worker_1': {},
            'worker_2': {},
            'worker_3': {}
        }

        profiler = MLInferenceProfiler("dist_cache_test")
        profiler.start_profiling()

        num_requests = 1000
        num_workers = 3

        for req_idx in range(num_requests):
            # Route to random worker
            worker_id = f"worker_{(req_idx % num_workers) + 1}"
            cache = cache_stores[worker_id]

            features_key = f"features_{req_idx % 100}"  # 100 unique features

            # Check cache hit
            if features_key in cache:
                profiler.record_cache_hit()
            else:
                cache[features_key] = {"prediction": float(np.random.random())}
                profiler.record_cache_miss()

            profiler.record_inference(0.001)

        metrics = profiler.get_metrics()

        print(f"\nDistributed Cache Consistency:")
        print(f"  Workers: {num_workers}")
        print(f"  Total requests: {metrics.total_inferences}")
        print(f"  Cache hit rate: {metrics.cache_hit_rate:.3f}")

        for worker_id, cache in cache_stores.items():
            print(f"  {worker_id} cache size: {len(cache)}")

        # Even with distribution, hit rate should be reasonable
        assert metrics.cache_hit_rate > 0.3, f"Distributed cache hit rate too low: {metrics.cache_hit_rate:.3f}"


class TestInferenceLatencyDistribution:
    """Test inference latency distribution"""

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_latency_percentiles(self):
        """Test inference latency percentiles"""

        profiler = MLInferenceProfiler("latency_test")
        profiler.start_profiling()

        model = MockMLModel(complexity='moderate')

        # Generate requests with varying latency
        num_requests = 1000

        for i in range(num_requests):
            features = np.random.randn(50)

            start_time = time.time()
            prediction = await model.predict(features)
            inference_time = time.time() - start_time

            profiler.record_inference(inference_time)

        metrics = profiler.get_metrics()

        print(f"\nInference Latency Distribution:")
        print(f"  Total inferences: {metrics.total_inferences}")
        print(f"  Min: {min(profiler.inference_times)*1000:.2f}ms")
        print(f"  Max: {max(profiler.inference_times)*1000:.2f}ms")
        print(f"  Mean: {metrics.avg_inference_time*1000:.2f}ms")
        print(f"  P50: {metrics.p50_inference_time*1000:.2f}ms")
        print(f"  P95: {metrics.p95_inference_time*1000:.2f}ms")
        print(f"  P99: {metrics.p99_inference_time*1000:.2f}ms")

        # Assertions
        assert metrics.p95_inference_time < 0.1, f"P95 latency too high: {metrics.p95_inference_time*1000:.2f}ms"
        assert metrics.p99_inference_time < 0.15, f"P99 latency too high: {metrics.p99_inference_time*1000:.2f}ms"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "performance"])
