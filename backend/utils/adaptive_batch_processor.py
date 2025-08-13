"""
Adaptive Batch Processing System
Dynamically adjusts batch sizes based on system performance and data characteristics
"""

import asyncio
import psutil
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import numpy as np
from collections import deque
import statistics

from backend.utils.cache import get_redis
from backend.config.settings import settings

logger = logging.getLogger(__name__)


class BatchStrategy(Enum):
    """Batch sizing strategies"""
    FIXED = "fixed"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    ADAPTIVE = "adaptive"
    MEMORY_BASED = "memory_based"
    LATENCY_BASED = "latency_based"
    THROUGHPUT_OPTIMIZED = "throughput_optimized"


@dataclass
class BatchMetrics:
    """Metrics for a processed batch"""
    batch_size: int
    processing_time_ms: float
    memory_used_mb: float
    cpu_usage_percent: float
    items_per_second: float
    success_rate: float
    error_count: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def efficiency_score(self) -> float:
        """Calculate batch efficiency score"""
        if self.processing_time_ms == 0:
            return 0
        
        throughput_score = min(1.0, self.items_per_second / 1000)
        memory_score = max(0, 1 - (self.memory_used_mb / 1024))  # Penalty for > 1GB
        cpu_score = max(0, 1 - (self.cpu_usage_percent / 100))
        success_score = self.success_rate
        
        return (
            throughput_score * 0.4 +
            memory_score * 0.2 +
            cpu_score * 0.2 +
            success_score * 0.2
        )


@dataclass
class BatchConfiguration:
    """Configuration for batch processing"""
    min_batch_size: int = 20  # Increased for better throughput
    max_batch_size: int = 2000  # Increased for large datasets
    initial_batch_size: int = 200  # Increased starting size
    target_processing_time_ms: float = 2000  # Increased for better throughput
    max_memory_mb: float = 1024  # Increased memory limit
    max_cpu_percent: float = 85  # Increased CPU utilization
    adjustment_factor: float = 0.3  # More aggressive adjustments
    stability_window: int = 5  # Reduced for faster adaptation
    strategy: BatchStrategy = BatchStrategy.THROUGHPUT_OPTIMIZED  # Focus on throughput
    
    # New performance optimization settings
    enable_parallel_processing: bool = True
    max_concurrent_batches: int = 4
    stream_processing: bool = True
    prefetch_batches: int = 2


class AdaptiveBatchProcessor:
    """
    Adaptive batch processing system that optimizes batch sizes
    """
    
    def __init__(self, config: Optional[BatchConfiguration] = None):
        self.config = config or BatchConfiguration()
        self.current_batch_size = self.config.initial_batch_size
        self.metrics_history = deque(maxlen=100)
        self.batch_size_history = deque(maxlen=50)
        self.performance_model = None
        self.redis = None
        
        # Performance tracking
        self.total_items_processed = 0
        self.total_processing_time = 0
        self.batch_count = 0
        
        # Adaptive parameters
        self.stable_performance_window = deque(maxlen=self.config.stability_window)
        self.last_adjustment_time = datetime.utcnow()
        self.adjustment_cooldown = timedelta(seconds=5)
        
    async def initialize(self):
        """Initialize the batch processor"""
        self.redis = await get_redis()
        await self._load_performance_model()
        logger.info(f"Adaptive batch processor initialized with strategy: {self.config.strategy.value}")
    
    async def process_adaptive_batch(
        self,
        items: List[Any],
        process_func: Callable,
        item_size_estimator: Optional[Callable] = None
    ) -> List[Tuple[List[Any], BatchMetrics]]:
        """
        Process items in adaptive batches with parallel processing optimization
        
        Args:
            items: Items to process
            process_func: Async function to process a batch
            item_size_estimator: Optional function to estimate item size
            
        Returns:
            List of (batch results, metrics) tuples
        """
        results = []
        remaining_items = items[:]
        
        # Enable parallel processing if configured
        if self.config.enable_parallel_processing and len(remaining_items) > self.config.max_batch_size:
            return await self._process_parallel_batches(
                remaining_items, process_func, item_size_estimator
            )
        
        # Regular sequential processing
        while remaining_items:
            # Determine optimal batch size
            batch_size = await self._determine_batch_size(
                remaining_items,
                item_size_estimator
            )
            
            # Extract batch
            batch = remaining_items[:batch_size]
            remaining_items = remaining_items[batch_size:]
            
            # Process batch with metrics collection
            batch_result, metrics = await self._process_with_metrics(
                batch,
                process_func
            )
            
            # Update batch size based on performance
            await self._adjust_batch_size(metrics)
            
            # Store metrics
            self.metrics_history.append(metrics)
            results.append((batch_result, metrics))
            
            # Update totals
            self.total_items_processed += len(batch)
            self.total_processing_time += metrics.processing_time_ms
            self.batch_count += 1
            
            # Reduced delay for better throughput
            if remaining_items:
                await asyncio.sleep(0.001)  # Reduced from 0.01
        
        return results
    
    async def _process_parallel_batches(
        self,
        items: List[Any],
        process_func: Callable,
        item_size_estimator: Optional[Callable] = None
    ) -> List[Tuple[List[Any], BatchMetrics]]:
        """Process multiple batches in parallel for improved throughput"""
        results = []
        remaining_items = items[:]
        
        # Create semaphore for concurrent batch processing
        batch_semaphore = asyncio.Semaphore(self.config.max_concurrent_batches)
        
        async def process_single_batch(batch_items):
            async with batch_semaphore:
                batch_size = len(batch_items)
                batch_result, metrics = await self._process_with_metrics(
                    batch_items, process_func
                )
                
                # Update metrics
                self.metrics_history.append(metrics)
                self.total_items_processed += batch_size
                self.total_processing_time += metrics.processing_time_ms
                self.batch_count += 1
                
                return (batch_result, metrics)
        
        # Create batches
        batch_tasks = []
        while remaining_items:
            batch_size = await self._determine_batch_size(
                remaining_items, item_size_estimator
            )
            batch = remaining_items[:batch_size]
            remaining_items = remaining_items[batch_size:]
            
            # Create task for this batch
            task = asyncio.create_task(process_single_batch(batch))
            batch_tasks.append(task)
            
            # Limit concurrent tasks
            if len(batch_tasks) >= self.config.max_concurrent_batches:
                # Wait for some tasks to complete
                done, pending = await asyncio.wait(
                    batch_tasks, return_when=asyncio.FIRST_COMPLETED
                )
                
                # Collect completed results
                for task in done:
                    try:
                        result = await task
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Parallel batch processing error: {e}")
                
                # Keep pending tasks for next iteration
                batch_tasks = list(pending)
        
        # Wait for remaining tasks
        if batch_tasks:
            remaining_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            for result in remaining_results:
                if isinstance(result, tuple):
                    results.append(result)
                else:
                    logger.error(f"Parallel batch error: {result}")
        
        return results
    
    async def optimize_batch_size(
        self,
        sample_data: List[Any],
        process_func: Callable,
        optimization_rounds: int = 10
    ) -> int:
        """
        Optimize batch size through experimentation
        
        Args:
            sample_data: Sample data for testing
            process_func: Processing function
            optimization_rounds: Number of optimization rounds
            
        Returns:
            Optimal batch size
        """
        logger.info("Starting batch size optimization...")
        
        # Test different batch sizes
        test_sizes = self._generate_test_sizes()
        size_performance = {}
        
        for size in test_sizes:
            if size > len(sample_data):
                continue
            
            # Test this batch size multiple times
            metrics_list = []
            
            for _ in range(min(3, optimization_rounds)):
                batch = sample_data[:size]
                _, metrics = await self._process_with_metrics(batch, process_func)
                metrics_list.append(metrics)
                
                # Small delay between tests
                await asyncio.sleep(0.1)
            
            # Calculate average performance
            avg_efficiency = np.mean([m.efficiency_score for m in metrics_list])
            size_performance[size] = avg_efficiency
            
            logger.debug(f"Batch size {size}: efficiency={avg_efficiency:.3f}")
        
        # Find optimal size
        if size_performance:
            optimal_size = max(size_performance, key=size_performance.get)
            
            # Update configuration
            self.current_batch_size = optimal_size
            self.config.initial_batch_size = optimal_size
            
            logger.info(f"Optimal batch size determined: {optimal_size}")
            return optimal_size
        
        return self.config.initial_batch_size
    
    async def _determine_batch_size(
        self,
        items: List[Any],
        size_estimator: Optional[Callable] = None
    ) -> int:
        """Determine optimal batch size based on strategy"""
        
        if self.config.strategy == BatchStrategy.FIXED:
            return min(self.config.initial_batch_size, len(items))
        
        elif self.config.strategy == BatchStrategy.LINEAR:
            return await self._linear_batch_size(items)
        
        elif self.config.strategy == BatchStrategy.EXPONENTIAL:
            return await self._exponential_batch_size(items)
        
        elif self.config.strategy == BatchStrategy.MEMORY_BASED:
            return await self._memory_based_batch_size(items, size_estimator)
        
        elif self.config.strategy == BatchStrategy.LATENCY_BASED:
            return await self._latency_based_batch_size(items)
        
        elif self.config.strategy == BatchStrategy.THROUGHPUT_OPTIMIZED:
            return await self._throughput_optimized_batch_size(items)
        
        elif self.config.strategy == BatchStrategy.ADAPTIVE:
            return await self._fully_adaptive_batch_size(items, size_estimator)
        
        return min(self.current_batch_size, len(items))
    
    async def _linear_batch_size(self, items: List[Any]) -> int:
        """Linear batch size adjustment"""
        if not self.metrics_history:
            return min(self.current_batch_size, len(items))
        
        last_metric = self.metrics_history[-1]
        
        # Adjust based on processing time
        if last_metric.processing_time_ms > self.config.target_processing_time_ms:
            # Decrease batch size
            adjustment = int(self.current_batch_size * 0.1)
            self.current_batch_size = max(
                self.config.min_batch_size,
                self.current_batch_size - adjustment
            )
        else:
            # Increase batch size
            adjustment = int(self.current_batch_size * 0.1)
            self.current_batch_size = min(
                self.config.max_batch_size,
                self.current_batch_size + adjustment
            )
        
        return min(self.current_batch_size, len(items))
    
    async def _exponential_batch_size(self, items: List[Any]) -> int:
        """Exponential batch size adjustment"""
        if not self.metrics_history:
            return min(self.current_batch_size, len(items))
        
        # Calculate performance trend
        recent_metrics = list(self.metrics_history)[-5:]
        efficiency_trend = [m.efficiency_score for m in recent_metrics]
        
        if len(efficiency_trend) >= 2:
            if efficiency_trend[-1] > efficiency_trend[-2]:
                # Performance improving, increase exponentially
                self.current_batch_size = min(
                    self.config.max_batch_size,
                    int(self.current_batch_size * 1.5)
                )
            else:
                # Performance degrading, decrease exponentially
                self.current_batch_size = max(
                    self.config.min_batch_size,
                    int(self.current_batch_size * 0.7)
                )
        
        return min(self.current_batch_size, len(items))
    
    async def _memory_based_batch_size(
        self,
        items: List[Any],
        size_estimator: Optional[Callable] = None
    ) -> int:
        """Determine batch size based on memory constraints"""
        # Get current memory usage
        memory_info = psutil.virtual_memory()
        available_memory_mb = memory_info.available / (1024 * 1024)
        
        # Estimate item size
        if size_estimator:
            avg_item_size = size_estimator(items[0]) if items else 1024
        else:
            # Default estimation
            avg_item_size = 1024  # 1KB default
        
        # Calculate max items that fit in memory budget
        memory_budget_mb = min(
            self.config.max_memory_mb,
            available_memory_mb * 0.5  # Use max 50% of available memory
        )
        
        max_items = int((memory_budget_mb * 1024 * 1024) / avg_item_size)
        
        # Apply bounds
        batch_size = max(
            self.config.min_batch_size,
            min(max_items, self.config.max_batch_size)
        )
        
        return min(batch_size, len(items))
    
    async def _latency_based_batch_size(self, items: List[Any]) -> int:
        """Determine batch size based on latency requirements"""
        if not self.metrics_history:
            return min(self.current_batch_size, len(items))
        
        # Calculate average processing time per item
        recent_metrics = list(self.metrics_history)[-10:]
        avg_time_per_item = np.mean([
            m.processing_time_ms / m.batch_size
            for m in recent_metrics
            if m.batch_size > 0
        ])
        
        # Calculate batch size to meet target latency
        if avg_time_per_item > 0:
            target_batch_size = int(
                self.config.target_processing_time_ms / avg_time_per_item
            )
            
            # Apply bounds and smoothing
            target_batch_size = max(
                self.config.min_batch_size,
                min(target_batch_size, self.config.max_batch_size)
            )
            
            # Smooth adjustment
            self.current_batch_size = int(
                self.current_batch_size * 0.7 + target_batch_size * 0.3
            )
        
        return min(self.current_batch_size, len(items))
    
    async def _throughput_optimized_batch_size(self, items: List[Any]) -> int:
        """Optimize for maximum throughput"""
        if len(self.metrics_history) < 5:
            return min(self.current_batch_size, len(items))
        
        # Analyze throughput at different batch sizes
        size_throughput = {}
        
        for metric in list(self.metrics_history)[-20:]:
            size = metric.batch_size
            throughput = metric.items_per_second
            
            if size not in size_throughput:
                size_throughput[size] = []
            size_throughput[size].append(throughput)
        
        # Find size with best average throughput
        best_size = self.current_batch_size
        best_throughput = 0
        
        for size, throughputs in size_throughput.items():
            avg_throughput = np.mean(throughputs)
            if avg_throughput > best_throughput:
                best_throughput = avg_throughput
                best_size = size
        
        # Adjust towards best size
        if best_size != self.current_batch_size:
            # Gradual adjustment
            direction = 1 if best_size > self.current_batch_size else -1
            adjustment = int(abs(best_size - self.current_batch_size) * 0.3)
            self.current_batch_size += direction * adjustment
        
        # Apply bounds
        self.current_batch_size = max(
            self.config.min_batch_size,
            min(self.current_batch_size, self.config.max_batch_size)
        )
        
        return min(self.current_batch_size, len(items))
    
    async def _fully_adaptive_batch_size(
        self,
        items: List[Any],
        size_estimator: Optional[Callable] = None
    ) -> int:
        """Fully adaptive batch size using multiple factors"""
        
        # Start with current size
        candidate_size = self.current_batch_size
        
        # Factor 1: Memory constraints
        memory_size = await self._memory_based_batch_size(items, size_estimator)
        
        # Factor 2: Latency requirements
        latency_size = await self._latency_based_batch_size(items)
        
        # Factor 3: System load
        cpu_percent = psutil.cpu_percent(interval=0.1)
        load_factor = max(0.5, 1 - (cpu_percent / 100))
        
        # Factor 4: Historical performance
        if self.metrics_history:
            recent_efficiency = np.mean([
                m.efficiency_score for m in list(self.metrics_history)[-5:]
            ])
            
            if recent_efficiency > 0.8:
                # Good performance, can increase
                performance_factor = 1.2
            elif recent_efficiency < 0.5:
                # Poor performance, decrease
                performance_factor = 0.8
            else:
                performance_factor = 1.0
        else:
            performance_factor = 1.0
        
        # Combine factors
        candidate_size = int(
            min(memory_size, latency_size) * load_factor * performance_factor
        )
        
        # Apply bounds
        candidate_size = max(
            self.config.min_batch_size,
            min(candidate_size, self.config.max_batch_size)
        )
        
        # Smooth adjustment to prevent oscillation
        if self.batch_size_history:
            # Use exponential moving average
            alpha = 0.3
            self.current_batch_size = int(
                alpha * candidate_size + (1 - alpha) * self.current_batch_size
            )
        else:
            self.current_batch_size = candidate_size
        
        self.batch_size_history.append(self.current_batch_size)
        
        return min(self.current_batch_size, len(items))
    
    async def _process_with_metrics(
        self,
        batch: List[Any],
        process_func: Callable
    ) -> Tuple[Any, BatchMetrics]:
        """Process batch and collect metrics"""
        
        # Record initial state
        start_time = time.perf_counter()
        start_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        start_cpu = psutil.cpu_percent(interval=None)
        
        # Process batch
        success_count = 0
        error_count = 0
        result = None
        
        try:
            result = await process_func(batch)
            success_count = len(batch)
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            error_count = len(batch)
        
        # Record final state
        end_time = time.perf_counter()
        end_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        end_cpu = psutil.cpu_percent(interval=None)
        
        # Calculate metrics
        processing_time_ms = (end_time - start_time) * 1000
        memory_used_mb = max(0, end_memory - start_memory)
        cpu_usage_percent = max(0, end_cpu - start_cpu)
        items_per_second = (len(batch) / processing_time_ms * 1000) if processing_time_ms > 0 else 0
        success_rate = success_count / len(batch) if batch else 0
        
        metrics = BatchMetrics(
            batch_size=len(batch),
            processing_time_ms=processing_time_ms,
            memory_used_mb=memory_used_mb,
            cpu_usage_percent=cpu_usage_percent,
            items_per_second=items_per_second,
            success_rate=success_rate,
            error_count=error_count
        )
        
        return result, metrics
    
    async def _adjust_batch_size(self, metrics: BatchMetrics):
        """Adjust batch size based on metrics"""
        
        # Check cooldown
        if datetime.utcnow() - self.last_adjustment_time < self.adjustment_cooldown:
            return
        
        # Add to stability window
        self.stable_performance_window.append(metrics.efficiency_score)
        
        # Check if performance is stable
        if len(self.stable_performance_window) >= self.config.stability_window:
            std_dev = statistics.stdev(self.stable_performance_window)
            
            if std_dev < 0.1:  # Stable performance
                # Make larger adjustments
                adjustment_factor = self.config.adjustment_factor * 2
            else:
                # Make smaller adjustments
                adjustment_factor = self.config.adjustment_factor
        else:
            adjustment_factor = self.config.adjustment_factor
        
        # Determine adjustment direction
        if metrics.processing_time_ms > self.config.target_processing_time_ms * 1.2:
            # Too slow, decrease batch size
            self.current_batch_size = max(
                self.config.min_batch_size,
                int(self.current_batch_size * (1 - adjustment_factor))
            )
        elif metrics.processing_time_ms < self.config.target_processing_time_ms * 0.8:
            # Too fast, increase batch size
            if metrics.memory_used_mb < self.config.max_memory_mb * 0.8:
                self.current_batch_size = min(
                    self.config.max_batch_size,
                    int(self.current_batch_size * (1 + adjustment_factor))
                )
        
        # Check resource constraints
        if metrics.memory_used_mb > self.config.max_memory_mb:
            # Memory constraint violated
            self.current_batch_size = max(
                self.config.min_batch_size,
                int(self.current_batch_size * 0.8)
            )
        
        if metrics.cpu_usage_percent > self.config.max_cpu_percent:
            # CPU constraint violated
            self.current_batch_size = max(
                self.config.min_batch_size,
                int(self.current_batch_size * 0.9)
            )
        
        self.last_adjustment_time = datetime.utcnow()
    
    def _generate_test_sizes(self) -> List[int]:
        """Generate test batch sizes for optimization"""
        sizes = []
        
        # Logarithmic scale
        current = self.config.min_batch_size
        while current <= self.config.max_batch_size:
            sizes.append(current)
            current = int(current * 1.5)
        
        # Add some intermediate sizes
        if len(sizes) > 1:
            for i in range(len(sizes) - 1):
                mid = (sizes[i] + sizes[i + 1]) // 2
                sizes.append(mid)
        
        return sorted(set(sizes))
    
    async def _load_performance_model(self):
        """Load historical performance model"""
        # In production, load ML model for batch size prediction
        pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        if not self.metrics_history:
            return {}
        
        recent_metrics = list(self.metrics_history)[-20:]
        
        return {
            'current_batch_size': self.current_batch_size,
            'total_items_processed': self.total_items_processed,
            'total_batches': self.batch_count,
            'average_batch_size': np.mean([m.batch_size for m in recent_metrics]),
            'average_processing_time_ms': np.mean([m.processing_time_ms for m in recent_metrics]),
            'average_throughput': np.mean([m.items_per_second for m in recent_metrics]),
            'average_efficiency': np.mean([m.efficiency_score for m in recent_metrics]),
            'batch_size_stability': statistics.stdev(self.batch_size_history) if len(self.batch_size_history) > 1 else 0
        }


# Global instance
_batch_processor: Optional[AdaptiveBatchProcessor] = None


async def get_batch_processor(
    config: Optional[BatchConfiguration] = None
) -> AdaptiveBatchProcessor:
    """Get or create the global batch processor"""
    global _batch_processor
    if _batch_processor is None:
        _batch_processor = AdaptiveBatchProcessor(config)
        await _batch_processor.initialize()
    return _batch_processor