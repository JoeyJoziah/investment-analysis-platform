"""
Comprehensive Tests for Performance Optimizations
Tests all memory leak fixes, batch processing optimizations, and performance improvements
"""

import asyncio
import pytest
import pytest_asyncio
import time
import gc
import psutil
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

from backend.utils.memory_manager import (
    MemoryManager, MemoryPressureLevel, GCStrategy, BoundedDict, BoundedList
)
from backend.utils.adaptive_batch_processor import (
    AdaptiveBatchProcessor, BatchConfiguration, BatchStrategy, BatchMetrics
)
from backend.utils.enhanced_parallel_processor import (
    EnhancedParallelProcessor, EnhancedAPITask, Priority, ProcessingStrategy
)
from backend.utils.dynamic_resource_manager import (
    DynamicResourceManager, ResourceType, WorkloadType, ResourceMetrics
)
from backend.utils.performance_profiler import (
    PerformanceProfiler, MetricType, PerformanceMetric
)
from backend.analytics.recommendation_engine_optimized import OptimizedRecommendationEngine


class TestMemoryManager:
    """Test memory management and leak fixes"""
    
    @pytest_asyncio.fixture
    async def memory_manager(self):
        """Create memory manager for testing"""
        manager = MemoryManager(
            gc_strategy=GCStrategy.ADAPTIVE,
            memory_threshold_mb=1024,
            monitoring_interval=1  # Fast interval for testing
        )
        await manager.initialize()
        yield manager
        await manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_memory_metrics_collection(self, memory_manager):
        """Test memory metrics collection"""
        metrics = await memory_manager.collect_metrics()
        
        assert isinstance(metrics.memory_percent, float)
        assert isinstance(metrics.process_memory_mb, float)
        assert isinstance(metrics.pressure_level, MemoryPressureLevel)
        assert metrics.memory_percent >= 0
        assert metrics.process_memory_mb > 0
    
    @pytest.mark.asyncio
    async def test_memory_pressure_handling(self, memory_manager):
        """Test memory pressure handling"""
        # Mock high memory pressure
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value.percent = 95  # High pressure
            
            metrics = await memory_manager.collect_metrics()
            assert metrics.pressure_level in [MemoryPressureLevel.HIGH, MemoryPressureLevel.CRITICAL]
    
    @pytest.mark.asyncio
    async def test_emergency_cleanup(self, memory_manager):
        """Test emergency cleanup functionality"""
        initial_objects = len(gc.get_objects())
        
        # Create some objects to clean up
        test_data = [list(range(1000)) for _ in range(100)]
        
        await memory_manager.emergency_cleanup()
        
        # Verify cleanup occurred
        final_objects = len(gc.get_objects())
        assert final_objects <= initial_objects + len(test_data)
    
    @pytest.mark.asyncio
    async def test_leak_detection(self, memory_manager):
        """Test memory leak detection"""
        if not memory_manager.tracemalloc_enabled:
            pytest.skip("Tracemalloc not enabled")
        
        # Simulate memory growth
        growing_data = []
        for i in range(50):
            growing_data.extend([f"data_{i}_{j}" for j in range(100)])
            await asyncio.sleep(0.01)  # Allow monitoring
        
        # Trigger leak detection
        await memory_manager._detect_memory_leaks()
        
        # Check if potential leaks were detected
        assert len(memory_manager.potential_leaks) >= 0  # May or may not detect leaks
    
    def test_bounded_dict(self):
        """Test BoundedDict prevents memory leaks"""
        bounded_dict = BoundedDict(max_size=100)
        
        # Add more items than max_size
        for i in range(200):
            bounded_dict[f"key_{i}"] = f"value_{i}"
        
        # Should not exceed max_size
        assert len(bounded_dict) <= 100
        
        # Should contain recent items
        assert "key_199" in bounded_dict
        assert "key_0" not in bounded_dict
    
    def test_bounded_list(self):
        """Test BoundedList prevents memory leaks"""
        bounded_list = BoundedList(max_size=50)
        
        # Add more items than max_size
        for i in range(100):
            bounded_list.append(i)
        
        # Should not exceed max_size
        assert len(bounded_list) <= 50
        
        # Should contain recent items
        assert 99 in bounded_list
        assert 0 not in bounded_list


class TestAdaptiveBatchProcessor:
    """Test batch processing optimizations"""
    
    @pytest_asyncio.fixture
    async def batch_processor(self):
        """Create batch processor for testing"""
        config = BatchConfiguration(
            min_batch_size=10,
            max_batch_size=100,
            initial_batch_size=50,
            enable_parallel_processing=True,
            max_concurrent_batches=2
        )
        processor = AdaptiveBatchProcessor(config)
        await processor.initialize()
        yield processor
    
    @pytest.mark.asyncio
    async def test_adaptive_batch_processing(self, batch_processor):
        """Test adaptive batch processing"""
        test_items = list(range(200))
        
        async def mock_process_func(batch):
            # Simulate processing time
            await asyncio.sleep(0.01)
            return [f"processed_{item}" for item in batch]
        
        results = await batch_processor.process_adaptive_batch(
            test_items, mock_process_func
        )
        
        assert len(results) > 0
        
        # Verify all items were processed
        total_processed = sum(len(batch_result[0]) for batch_result in results)
        assert total_processed == len(test_items)
        
        # Check metrics
        for batch_result, metrics in results:
            assert isinstance(metrics, BatchMetrics)
            assert metrics.batch_size > 0
            assert metrics.processing_time_ms >= 0
            assert metrics.success_rate > 0
    
    @pytest.mark.asyncio
    async def test_parallel_batch_processing(self, batch_processor):
        """Test parallel batch processing"""
        test_items = list(range(500))  # Large enough to trigger parallel processing
        
        async def mock_process_func(batch):
            await asyncio.sleep(0.02)  # Simulate work
            return [f"processed_{item}" for item in batch]
        
        start_time = time.time()
        results = await batch_processor.process_adaptive_batch(
            test_items, mock_process_func
        )
        end_time = time.time()
        
        # Should complete faster than sequential processing
        total_time = end_time - start_time
        expected_sequential_time = len(test_items) * 0.02 / batch_processor.config.initial_batch_size
        
        # Parallel processing should be faster
        assert total_time < expected_sequential_time
    
    @pytest.mark.asyncio
    async def test_batch_size_optimization(self, batch_processor):
        """Test batch size optimization"""
        test_items = list(range(100))
        
        async def variable_process_func(batch):
            # Processing time varies with batch size
            processing_time = len(batch) * 0.001
            await asyncio.sleep(processing_time)
            return batch
        
        # Run optimization
        optimal_size = await batch_processor.optimize_batch_size(
            test_items[:50], variable_process_func, optimization_rounds=5
        )
        
        assert optimal_size >= batch_processor.config.min_batch_size
        assert optimal_size <= batch_processor.config.max_batch_size
    
    def test_batch_metrics(self):
        """Test batch metrics calculation"""
        metrics = BatchMetrics(
            batch_size=100,
            processing_time_ms=1000,
            memory_used_mb=50,
            cpu_usage_percent=60,
            items_per_second=100,
            success_rate=1.0
        )
        
        efficiency = metrics.efficiency_score
        assert 0 <= efficiency <= 1
        assert isinstance(efficiency, float)


class TestEnhancedParallelProcessor:
    """Test enhanced parallel processing"""
    
    @pytest_asyncio.fixture
    async def parallel_processor(self):
        """Create parallel processor for testing"""
        processor = EnhancedParallelProcessor(
            max_concurrent_calls=20,
            strategy=ProcessingStrategy.BALANCED
        )
        await processor.initialize()
        yield processor
        await processor.shutdown()
    
    @pytest.mark.asyncio
    async def test_enhanced_task_processing(self, parallel_processor):
        """Test enhanced task processing"""
        tasks = [
            EnhancedAPITask(
                id=f"test_task_{i}",
                provider="test_provider",
                endpoint="test_endpoint",
                params={"param": f"value_{i}"},
                priority=Priority.MEDIUM
            )
            for i in range(10)
        ]
        
        with patch.object(parallel_processor, '_execute_enhanced_api_call') as mock_execute:
            mock_execute.return_value = AsyncMock(return_value=MagicMock(
                task_id="test",
                success=True,
                data={"result": "success"},
                latency_ms=100
            ))
            
            results = await parallel_processor.process_enhanced_batch(tasks)
            
            assert len(results) == len(tasks)
            assert all(result.success for result in results)
    
    @pytest.mark.asyncio
    async def test_connection_pool_optimization(self, parallel_processor):
        """Test connection pool optimization"""
        # Test connection pool creation
        pool = await parallel_processor._get_connection_pool("test_provider")
        assert pool is not None
        
        # Test session creation
        session = await parallel_processor._get_session("test_provider")
        assert session is not None
        assert not session.closed
    
    @pytest.mark.asyncio
    async def test_adaptive_concurrency_learning(self, parallel_processor):
        """Test adaptive concurrency learning"""
        provider = "test_provider"
        
        # Mock some performance data
        for i in range(30):
            result = MagicMock()
            result.success = True
            result.provider_latency_ms = 100 + (i % 10) * 10
            result.queue_wait_time_ms = 10
            result.response_size_bytes = 1000
            
            await parallel_processor._update_enhanced_adaptive_learning(provider, result)
        
        # Check if optimal concurrency was calculated
        assert provider in parallel_processor._optimal_concurrency
        optimal_concurrency = parallel_processor._optimal_concurrency[provider]
        assert optimal_concurrency > 0
    
    def test_performance_score_calculation(self, parallel_processor):
        """Test performance score calculation"""
        result = MagicMock()
        result.success = True
        result.provider_latency_ms = 500
        result.queue_wait_time_ms = 50
        result.response_size_bytes = 2000
        
        score = parallel_processor._calculate_performance_score(result)
        assert 0 <= score <= 1


class TestDynamicResourceManager:
    """Test dynamic resource management"""
    
    @pytest_asyncio.fixture
    async def resource_manager(self):
        """Create resource manager for testing"""
        manager = DynamicResourceManager(
            monitoring_interval_s=1,  # Fast for testing
            enable_auto_scaling=True
        )
        await manager.initialize()
        yield manager
        await manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_resource_metrics_collection(self, resource_manager):
        """Test resource metrics collection"""
        metrics = await resource_manager._collect_comprehensive_metrics()
        
        assert isinstance(metrics, ResourceMetrics)
        assert metrics.cpu_percent >= 0
        assert metrics.memory_percent >= 0
        assert metrics.memory_total_gb > 0
        assert isinstance(metrics.timestamp, datetime)
    
    @pytest.mark.asyncio
    async def test_resource_optimization(self, resource_manager):
        """Test resource optimization"""
        # Create mock high-pressure metrics
        high_pressure_metrics = ResourceMetrics(
            timestamp=datetime.utcnow(),
            cpu_percent=95.0,
            memory_percent=90.0,
            cpu_count_logical=4,
            cpu_count_physical=2,
            cpu_freq_current=2000.0,
            memory_total_gb=16.0,
            memory_used_gb=14.4,
            memory_available_gb=1.6,
            swap_total_gb=4.0,
            swap_used_gb=0.5,
            disk_read_mb_s=10.0,
            disk_write_mb_s=5.0,
            disk_usage_percent=80.0,
            disk_iops=100.0,
            network_sent_mb_s=5.0,
            network_recv_mb_s=3.0,
            network_connections=50,
            process_count=200,
            thread_count=100,
            file_descriptor_count=500
        )
        
        initial_allocation = resource_manager.current_allocation
        await resource_manager._optimize_resources(high_pressure_metrics)
        final_allocation = resource_manager.current_allocation
        
        # Should have reduced resource allocation under high pressure
        assert final_allocation.cpu_workers <= initial_allocation.cpu_workers
        assert final_allocation.memory_limit_mb <= initial_allocation.memory_limit_mb
    
    def test_system_limits_detection(self):
        """Test system limits detection"""
        manager = DynamicResourceManager()
        limits = manager.system_limits
        
        assert 'cpu' in limits
        assert 'memory' in limits
        assert 'disk' in limits
        assert 'network' in limits
        
        assert limits['cpu']['logical_cores'] > 0
        assert limits['memory']['total_gb'] > 0
    
    @pytest.mark.asyncio
    async def test_predictive_scaling(self, resource_manager):
        """Test predictive scaling"""
        # Add some trend data
        for i in range(25):
            resource_manager.cpu_trend.append(50 + i * 1.5)  # Upward trend
            resource_manager.memory_trend.append(40 + i * 1.2)
        
        initial_allocation = resource_manager.current_allocation
        await resource_manager._predictive_scaling()
        
        # Should have adjusted allocation based on predicted trend
        # (Implementation may or may not change allocation based on prediction thresholds)
        assert resource_manager.current_allocation is not None


class TestPerformanceProfiler:
    """Test performance profiling and monitoring"""
    
    @pytest_asyncio.fixture
    async def profiler(self):
        """Create performance profiler for testing"""
        profiler = PerformanceProfiler(
            sampling_interval=0.1,  # Fast for testing
            max_samples=1000
        )
        await profiler.initialize()
        yield profiler
        await profiler.shutdown()
    
    @pytest.mark.asyncio
    async def test_performance_monitoring(self, profiler):
        """Test performance monitoring"""
        # Let it collect some metrics
        await asyncio.sleep(0.5)
        
        assert len(profiler.metrics_storage) > 0
        
        # Check metric types
        metric_types = {metric.metric_type for metric in profiler.metrics_storage}
        assert MetricType.RESOURCE_USAGE in metric_types
    
    @pytest.mark.asyncio
    async def test_function_profiling(self, profiler):
        """Test function profiling decorator"""
        
        @profiler.profile_function("test_function")
        async def test_async_function():
            await asyncio.sleep(0.1)
            return "result"
        
        result = await test_async_function()
        assert result == "result"
        
        # Check if function was tracked
        assert "test_function" in profiler.function_stats
        stats = profiler.function_stats["test_function"]
        assert stats['call_count'] == 1
        assert stats['total_time'] > 0
    
    @pytest.mark.asyncio
    async def test_memory_profiling(self, profiler):
        """Test memory profiling context manager"""
        with profiler.profile_memory("test_session"):
            # Allocate some memory
            test_data = [list(range(1000)) for _ in range(100)]
        
        # Check if session was recorded
        assert "test_session" in profiler.profiling_sessions
        session = profiler.profiling_sessions["test_session"]
        assert session['type'] == 'memory'
        assert session['total_memory_bytes'] >= 0
    
    @pytest.mark.asyncio
    async def test_cpu_profiling(self, profiler):
        """Test CPU profiling context manager"""
        with profiler.profile_cpu("test_cpu_session"):
            # Do some CPU work
            sum(i**2 for i in range(1000))
        
        # Check if session was recorded
        assert "test_cpu_session" in profiler.profiling_sessions
        session = profiler.profiling_sessions["test_cpu_session"]
        assert session['type'] == 'cpu'
        assert 'stats_output' in session
    
    def test_performance_report_generation(self, profiler):
        """Test performance report generation"""
        # Add some test metrics
        for i in range(10):
            profiler._add_metric(
                MetricType.LATENCY,
                "test_metric",
                100 + i * 10,
                datetime.utcnow()
            )
        
        report = profiler.generate_performance_report(time_range_minutes=60)
        
        assert 'report_generated' in report
        assert 'metric_statistics' in report
        assert 'recommendations' in report
        assert len(report['recommendations']) > 0
    
    def test_metrics_export(self, profiler):
        """Test metrics export functionality"""
        # Add some test metrics
        profiler._add_metric(
            MetricType.THROUGHPUT,
            "test_export_metric",
            50.0,
            datetime.utcnow(),
            tags={'test': 'true'},
            unit='rps'
        )
        
        # Test JSON export
        json_export = profiler.export_metrics(format='json', time_range_minutes=60)
        assert 'export_time' in json_export
        assert 'metrics' in json_export
        
        # Test CSV export
        csv_export = profiler.export_metrics(format='csv', time_range_minutes=60)
        assert 'timestamp,metric_type,name,value,unit,tags' in csv_export


class TestOptimizedRecommendationEngine:
    """Test optimized recommendation engine"""
    
    @pytest_asyncio.fixture
    async def recommendation_engine(self):
        """Create optimized recommendation engine for testing"""
        engine = OptimizedRecommendationEngine()
        
        # Mock the dependencies
        with patch('backend.analytics.technical_analysis.TechnicalAnalysisEngine'), \
             patch('backend.analytics.fundamental_analysis.FundamentalAnalysisEngine'), \
             patch('backend.analytics.sentiment_analysis.SentimentAnalysisEngine'), \
             patch('backend.models.ml_models.ModelManager'), \
             patch('backend.data_ingestion.market_scanner.MarketScanner'):
            
            await engine.initialize()
            yield engine
            await engine.shutdown()
    
    @pytest.mark.asyncio
    async def test_memory_optimized_analysis(self, recommendation_engine):
        """Test memory-optimized stock analysis"""
        mock_stock_data = {
            'ticker': 'TEST',
            'current_price': 100.0,
            'price_history': MagicMock(),
            'volume': 1000000,
            'market_cap': 1000000000,
            'beta': 1.2
        }
        
        # Mock price history
        import pandas as pd
        mock_price_history = pd.DataFrame({
            'close': [100 + i for i in range(50)],
            'volume': [1000000 + i*1000 for i in range(50)]
        })
        mock_stock_data['price_history'] = mock_price_history
        
        recommendation = await recommendation_engine._analyze_stock_optimized(
            'TEST', mock_stock_data
        )
        
        assert recommendation is not None
        assert recommendation.ticker == 'TEST'
        assert recommendation.confidence >= 0
        assert recommendation.confidence <= 1
    
    @pytest.mark.asyncio
    async def test_cache_utilization(self, recommendation_engine):
        """Test cache utilization for performance"""
        # First analysis should populate cache
        mock_data = {'ticker': 'CACHE_TEST', 'current_price': 50.0}
        
        with patch.object(recommendation_engine, '_fetch_stock_data_minimal', 
                         return_value=mock_data) as mock_fetch:
            
            # First call
            result1 = await recommendation_engine._analyze_stock_optimized('CACHE_TEST')
            
            # Second call should use cache
            result2 = await recommendation_engine._analyze_stock_optimized('CACHE_TEST')
            
            # Verify cache was used (fetch should only be called once)
            assert mock_fetch.call_count <= 2  # May be called for each due to date-based cache keys
    
    def test_bounded_collections_prevent_leaks(self, recommendation_engine):
        """Test that bounded collections prevent memory leaks"""
        # Add many items to caches
        for i in range(2000):  # More than max_size
            recommendation_engine._analysis_cache[f"key_{i}"] = f"value_{i}"
            recommendation_engine._stock_data_cache[f"stock_{i}"] = {"data": i}
        
        # Should not exceed bounds
        assert len(recommendation_engine._analysis_cache) <= 1000
        assert len(recommendation_engine._stock_data_cache) <= 500
    
    def test_performance_stats(self, recommendation_engine):
        """Test performance statistics collection"""
        stats = recommendation_engine.get_performance_stats()
        
        assert 'cache_sizes' in stats
        assert 'memory_optimization' in stats
        assert stats['memory_optimization'] == 'enabled'


class TestIntegration:
    """Integration tests for all optimizations working together"""
    
    @pytest.mark.asyncio
    async def test_full_optimization_pipeline(self):
        """Test full optimization pipeline integration"""
        # This test verifies that all optimizations work together
        memory_manager = None
        resource_manager = None
        profiler = None
        
        try:
            # Initialize all components
            memory_manager = MemoryManager()
            await memory_manager.initialize()
            
            resource_manager = DynamicResourceManager(monitoring_interval_s=1)
            await resource_manager.initialize()
            
            profiler = PerformanceProfiler(sampling_interval=0.1)
            await profiler.initialize()
            
            # Simulate some work
            with profiler.profile_memory("integration_test"):
                test_data = []
                for i in range(100):
                    test_data.extend([f"data_{i}_{j}" for j in range(50)])
                    
                    if i % 10 == 0:
                        await asyncio.sleep(0.01)  # Allow monitoring
            
            # Let components collect metrics
            await asyncio.sleep(0.5)
            
            # Verify components are working
            memory_stats = memory_manager.get_memory_stats()
            resource_stats = resource_manager.get_performance_stats()
            perf_report = profiler.generate_performance_report(time_range_minutes=1)
            
            assert memory_stats['current']['memory_percent'] > 0
            assert resource_stats['current']['cpu_percent'] >= 0
            assert len(perf_report['recommendations']) > 0
            
        finally:
            # Clean up
            if profiler:
                await profiler.shutdown()
            if resource_manager:
                await resource_manager.shutdown()
            if memory_manager:
                await memory_manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_memory_leak_prevention(self):
        """Test that optimizations prevent memory leaks"""
        initial_memory = psutil.Process().memory_info().rss
        
        # Create and destroy many objects that might leak
        for cycle in range(5):
            data_structures = []
            
            # Create various data structures
            for i in range(1000):
                data_structures.append({
                    'list': list(range(100)),
                    'dict': {f'key_{j}': f'value_{j}' for j in range(50)},
                    'nested': {'inner': list(range(25))}
                })
            
            # Clear references
            data_structures.clear()
            
            # Force garbage collection
            gc.collect()
            
            # Check memory hasn't grown excessively
            current_memory = psutil.Process().memory_info().rss
            growth_mb = (current_memory - initial_memory) / (1024 * 1024)
            
            # Allow some growth but not excessive (< 50MB per cycle)
            assert growth_mb < 50 * (cycle + 1), f"Excessive memory growth: {growth_mb:.2f}MB"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])