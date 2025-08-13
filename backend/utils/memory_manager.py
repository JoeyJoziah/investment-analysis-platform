"""
Comprehensive Memory Management and Performance Optimization System
Fixes memory leaks, optimizes performance, and provides advanced monitoring
"""

import gc
import psutil
import asyncio
import weakref
import sys
import threading
from typing import Dict, List, Optional, Any, Callable, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from collections import defaultdict, deque
import tracemalloc
import linecache
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import resource
import objgraph

from backend.utils.cache import get_redis
from backend.config.settings import settings

logger = logging.getLogger(__name__)

class MemoryPressureLevel(Enum):
    """Memory pressure levels"""
    LOW = "low"
    MODERATE = "moderate" 
    HIGH = "high"
    CRITICAL = "critical"

class GCStrategy(Enum):
    """Garbage collection strategies"""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    ADAPTIVE = "adaptive"

@dataclass
class MemoryMetrics:
    """Memory usage metrics"""
    total_memory_mb: float
    used_memory_mb: float
    available_memory_mb: float
    memory_percent: float
    process_memory_mb: float
    process_memory_percent: float
    gc_counts: Dict[int, int]
    pressure_level: MemoryPressureLevel
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def memory_efficiency(self) -> float:
        """Calculate memory efficiency score (0-1)"""
        if self.total_memory_mb == 0:
            return 0
        
        # Lower memory usage = higher efficiency
        efficiency = 1 - (self.used_memory_mb / self.total_memory_mb)
        return max(0, min(1, efficiency))

@dataclass
class LeakDetectionResult:
    """Memory leak detection result"""
    object_type: str
    count: int
    size_mb: float
    growth_rate: float
    is_potential_leak: bool
    last_seen: datetime
    stack_trace: Optional[str] = None

class WeakRefCleaner:
    """Automatic cleanup for weak references"""
    
    def __init__(self):
        self._refs: Set[weakref.ReferenceType] = set()
        self._cleanup_callbacks: List[Callable] = []
    
    def register(self, obj: Any, callback: Optional[Callable] = None) -> weakref.ReferenceType:
        """Register object for automatic cleanup"""
        def cleanup_ref(ref):
            self._refs.discard(ref)
            if callback:
                try:
                    callback()
                except Exception as e:
                    logger.error(f"Error in cleanup callback: {e}")
        
        ref = weakref.ref(obj, cleanup_ref)
        self._refs.add(ref)
        return ref
    
    def cleanup(self):
        """Force cleanup of dead references"""
        dead_refs = [ref for ref in self._refs if ref() is None]
        for ref in dead_refs:
            self._refs.discard(ref)

class MemoryManager:
    """
    Comprehensive memory management system
    """
    
    def __init__(
        self,
        gc_strategy: GCStrategy = GCStrategy.ADAPTIVE,
        memory_threshold_mb: int = 2048,
        leak_detection_enabled: bool = True,
        monitoring_interval: int = 30
    ):
        self.gc_strategy = gc_strategy
        self.memory_threshold_mb = memory_threshold_mb
        self.leak_detection_enabled = leak_detection_enabled
        self.monitoring_interval = monitoring_interval
        
        # Memory tracking
        self.metrics_history: deque = deque(maxlen=1000)
        self.object_counts: Dict[str, List[int]] = defaultdict(list)
        self.object_sizes: Dict[str, List[float]] = defaultdict(list)
        
        # Leak detection
        self.potential_leaks: Dict[str, LeakDetectionResult] = {}
        self.tracemalloc_enabled = False
        
        # Cleanup management
        self.weak_ref_cleaner = WeakRefCleaner()
        self.cleanup_tasks: List[Callable] = []
        self.bounded_collections: Dict[str, Any] = {}
        
        # Performance tracking
        self.gc_performance: deque = deque(maxlen=100)
        self.cleanup_performance: deque = deque(maxlen=100)
        
        # Threading
        self._monitor_task: Optional[asyncio.Task] = None
        self._shutdown_event = threading.Event()
        
    async def initialize(self):
        """Initialize memory manager"""
        # Enable tracemalloc for leak detection
        if self.leak_detection_enabled:
            tracemalloc.start(25)  # Keep 25 frames
            self.tracemalloc_enabled = True
        
        # Configure garbage collection
        self._configure_gc()
        
        # Start monitoring
        self._monitor_task = asyncio.create_task(self._memory_monitor_loop())
        
        logger.info(f"Memory manager initialized with strategy: {self.gc_strategy.value}")
    
    def _configure_gc(self):
        """Configure garbage collection based on strategy"""
        if self.gc_strategy == GCStrategy.CONSERVATIVE:
            # Less frequent GC
            gc.set_threshold(1000, 15, 15)
        elif self.gc_strategy == GCStrategy.BALANCED:
            # Default thresholds
            gc.set_threshold(700, 10, 10)
        elif self.gc_strategy == GCStrategy.AGGRESSIVE:
            # More frequent GC
            gc.set_threshold(500, 8, 8)
        else:  # ADAPTIVE
            # Will adjust dynamically
            gc.set_threshold(700, 10, 10)
    
    async def _memory_monitor_loop(self):
        """Continuous memory monitoring loop"""
        while not self._shutdown_event.is_set():
            try:
                # Collect metrics
                metrics = await self.collect_metrics()
                self.metrics_history.append(metrics)
                
                # Check pressure level and respond
                await self._handle_memory_pressure(metrics)
                
                # Detect leaks
                if self.leak_detection_enabled:
                    await self._detect_memory_leaks()
                
                # Adaptive GC tuning
                if self.gc_strategy == GCStrategy.ADAPTIVE:
                    await self._tune_gc_adaptive(metrics)
                
                # Cleanup bounded collections
                await self._cleanup_bounded_collections()
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in memory monitoring: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def collect_metrics(self) -> MemoryMetrics:
        """Collect comprehensive memory metrics"""
        # System memory
        memory = psutil.virtual_memory()
        
        # Process memory
        process = psutil.Process()
        process_memory = process.memory_info()
        
        # Garbage collection stats
        gc_counts = {i: gc.get_count()[i] for i in range(3)}
        
        # Determine pressure level
        if memory.percent > 90:
            pressure = MemoryPressureLevel.CRITICAL
        elif memory.percent > 80:
            pressure = MemoryPressureLevel.HIGH
        elif memory.percent > 70:
            pressure = MemoryPressureLevel.MODERATE
        else:
            pressure = MemoryPressureLevel.LOW
        
        return MemoryMetrics(
            total_memory_mb=memory.total / (1024 * 1024),
            used_memory_mb=memory.used / (1024 * 1024),
            available_memory_mb=memory.available / (1024 * 1024),
            memory_percent=memory.percent,
            process_memory_mb=process_memory.rss / (1024 * 1024),
            process_memory_percent=process_memory.rss / memory.total * 100,
            gc_counts=gc_counts,
            pressure_level=pressure
        )
    
    async def _handle_memory_pressure(self, metrics: MemoryMetrics):
        """Handle different memory pressure levels"""
        if metrics.pressure_level == MemoryPressureLevel.CRITICAL:
            logger.warning(f"Critical memory pressure: {metrics.memory_percent:.1f}%")
            await self.emergency_cleanup()
            
        elif metrics.pressure_level == MemoryPressureLevel.HIGH:
            logger.info(f"High memory pressure: {metrics.memory_percent:.1f}%")
            await self.aggressive_cleanup()
            
        elif metrics.pressure_level == MemoryPressureLevel.MODERATE:
            await self.standard_cleanup()
    
    async def emergency_cleanup(self):
        """Emergency memory cleanup"""
        start_time = datetime.utcnow()
        
        # Force garbage collection
        collected = await self._force_gc_all_generations()
        
        # Clear caches
        await self._clear_all_caches()
        
        # Cleanup bounded collections
        await self._cleanup_bounded_collections(aggressive=True)
        
        # Force weak reference cleanup
        self.weak_ref_cleaner.cleanup()
        
        # Clear object tracking
        self._clear_object_tracking()
        
        # Run custom cleanup tasks
        await self._run_cleanup_tasks()
        
        elapsed = (datetime.utcnow() - start_time).total_seconds()
        logger.info(f"Emergency cleanup completed in {elapsed:.2f}s, collected {collected} objects")
    
    async def aggressive_cleanup(self):
        """Aggressive memory cleanup"""
        # Collect all generations
        collected = gc.collect()
        
        # Cleanup bounded collections
        await self._cleanup_bounded_collections(aggressive=True)
        
        # Clear old metrics
        if len(self.metrics_history) > 500:
            # Keep only recent metrics
            recent_metrics = list(self.metrics_history)[-500:]
            self.metrics_history.clear()
            self.metrics_history.extend(recent_metrics)
        
        logger.debug(f"Aggressive cleanup collected {collected} objects")
    
    async def standard_cleanup(self):
        """Standard memory cleanup"""
        # Regular garbage collection
        collected = gc.collect()
        
        # Moderate cleanup of collections
        await self._cleanup_bounded_collections()
        
        logger.debug(f"Standard cleanup collected {collected} objects")
    
    async def _force_gc_all_generations(self) -> int:
        """Force garbage collection on all generations"""
        total_collected = 0
        
        # Collect from highest to lowest generation
        for generation in range(2, -1, -1):
            try:
                collected = gc.collect(generation)
                total_collected += collected
            except Exception as e:
                logger.error(f"Error collecting generation {generation}: {e}")
        
        return total_collected
    
    async def _clear_all_caches(self):
        """Clear all available caches"""
        try:
            # Clear Redis cache if available
            redis = await get_redis()
            if redis:
                await redis.flushdb()
        except Exception as e:
            logger.error(f"Error clearing Redis cache: {e}")
        
        # Clear function caches
        import functools
        for obj in gc.get_objects():
            if hasattr(obj, 'cache_clear') and callable(obj.cache_clear):
                try:
                    obj.cache_clear()
                except Exception:
                    pass
    
    async def _cleanup_bounded_collections(self, aggressive: bool = False):
        """Cleanup bounded collections to prevent memory leaks"""
        cleanup_ratio = 0.5 if aggressive else 0.3
        
        for name, collection in self.bounded_collections.items():
            try:
                if hasattr(collection, '__len__') and len(collection) > 0:
                    if hasattr(collection, 'clear'):
                        # Clear everything if aggressive
                        if aggressive:
                            collection.clear()
                        else:
                            # Keep recent items
                            if hasattr(collection, 'maxlen'):
                                target_size = int(collection.maxlen * (1 - cleanup_ratio))
                                while len(collection) > target_size:
                                    if hasattr(collection, 'popleft'):
                                        collection.popleft()
                                    elif hasattr(collection, 'pop'):
                                        collection.pop(0)
                                    else:
                                        break
            except Exception as e:
                logger.error(f"Error cleaning collection {name}: {e}")
    
    def _clear_object_tracking(self):
        """Clear object tracking data"""
        # Keep only recent tracking data
        for obj_type in list(self.object_counts.keys()):
            if len(self.object_counts[obj_type]) > 100:
                self.object_counts[obj_type] = self.object_counts[obj_type][-50:]
                self.object_sizes[obj_type] = self.object_sizes[obj_type][-50:]
    
    async def _run_cleanup_tasks(self):
        """Run registered cleanup tasks"""
        for cleanup_task in self.cleanup_tasks:
            try:
                if asyncio.iscoroutinefunction(cleanup_task):
                    await cleanup_task()
                else:
                    cleanup_task()
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
    
    async def _detect_memory_leaks(self):
        """Detect potential memory leaks"""
        if not self.tracemalloc_enabled:
            return
        
        try:
            # Get current memory snapshot
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')
            
            # Track object counts and sizes
            for stat in top_stats[:20]:  # Top 20 memory consumers
                filename = stat.traceback.format()[-1] if stat.traceback else "unknown"
                size_mb = stat.size / (1024 * 1024)
                
                # Track growth
                if filename in self.object_sizes:
                    self.object_sizes[filename].append(size_mb)
                    if len(self.object_sizes[filename]) > 10:
                        # Calculate growth rate
                        recent_sizes = self.object_sizes[filename][-5:]
                        old_sizes = self.object_sizes[filename][-10:-5]
                        
                        if len(recent_sizes) >= 3 and len(old_sizes) >= 3:
                            recent_avg = sum(recent_sizes) / len(recent_sizes)
                            old_avg = sum(old_sizes) / len(old_sizes)
                            growth_rate = (recent_avg - old_avg) / old_avg if old_avg > 0 else 0
                            
                            # Check for potential leak
                            if growth_rate > 0.2 and size_mb > 10:  # 20% growth and > 10MB
                                self.potential_leaks[filename] = LeakDetectionResult(
                                    object_type=filename,
                                    count=stat.count,
                                    size_mb=size_mb,
                                    growth_rate=growth_rate,
                                    is_potential_leak=True,
                                    last_seen=datetime.utcnow(),
                                    stack_trace=stat.traceback.format()[0] if stat.traceback else None
                                )
                else:
                    self.object_sizes[filename] = [size_mb]
                
        except Exception as e:
            logger.error(f"Error in leak detection: {e}")
    
    async def _tune_gc_adaptive(self, metrics: MemoryMetrics):
        """Adaptively tune garbage collection"""
        # Adjust GC thresholds based on memory pressure
        current_thresholds = gc.get_threshold()
        
        if metrics.pressure_level in [MemoryPressureLevel.HIGH, MemoryPressureLevel.CRITICAL]:
            # More aggressive GC
            new_thresholds = (
                max(300, current_thresholds[0] - 50),
                max(5, current_thresholds[1] - 1),
                max(5, current_thresholds[2] - 1)
            )
        elif metrics.pressure_level == MemoryPressureLevel.LOW:
            # Less aggressive GC
            new_thresholds = (
                min(1000, current_thresholds[0] + 50),
                min(15, current_thresholds[1] + 1),
                min(15, current_thresholds[2] + 1)
            )
        else:
            return  # No change needed
        
        gc.set_threshold(*new_thresholds)
        logger.debug(f"Adjusted GC thresholds: {current_thresholds} -> {new_thresholds}")
    
    def register_bounded_collection(self, name: str, collection: Any):
        """Register a collection for automatic cleanup"""
        self.bounded_collections[name] = collection
    
    def register_cleanup_task(self, task: Callable):
        """Register a cleanup task"""
        self.cleanup_tasks.append(task)
    
    def create_bounded_cache(self, name: str, max_size: int = 1000) -> deque:
        """Create a bounded cache that prevents memory leaks"""
        cache = deque(maxlen=max_size)
        self.register_bounded_collection(name, cache)
        return cache
    
    def create_weak_ref(self, obj: Any, callback: Optional[Callable] = None) -> weakref.ReferenceType:
        """Create a weak reference with automatic cleanup"""
        return self.weak_ref_cleaner.register(obj, callback)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        if not self.metrics_history:
            return {}
        
        current_metrics = self.metrics_history[-1]
        recent_metrics = list(self.metrics_history)[-20:]
        
        return {
            'current': {
                'memory_percent': current_metrics.memory_percent,
                'process_memory_mb': current_metrics.process_memory_mb,
                'pressure_level': current_metrics.pressure_level.value,
                'gc_counts': current_metrics.gc_counts
            },
            'trends': {
                'avg_memory_percent': np.mean([m.memory_percent for m in recent_metrics]),
                'max_memory_percent': max([m.memory_percent for m in recent_metrics]),
                'memory_efficiency': np.mean([m.memory_efficiency for m in recent_metrics])
            },
            'leaks': {
                'potential_leaks_count': len(self.potential_leaks),
                'potential_leaks': [
                    {
                        'type': leak.object_type,
                        'size_mb': leak.size_mb,
                        'growth_rate': leak.growth_rate
                    }
                    for leak in self.potential_leaks.values()
                    if leak.is_potential_leak
                ]
            },
            'gc_performance': {
                'avg_collection_time_ms': np.mean(self.gc_performance) if self.gc_performance else 0,
                'max_collection_time_ms': max(self.gc_performance) if self.gc_performance else 0
            }
        }
    
    def get_leak_report(self) -> Dict[str, Any]:
        """Generate detailed memory leak report"""
        if not self.potential_leaks:
            return {'status': 'no_leaks_detected', 'leaks': []}
        
        leak_details = []
        for filename, leak in self.potential_leaks.items():
            if leak.is_potential_leak:
                leak_details.append({
                    'location': filename,
                    'size_mb': leak.size_mb,
                    'count': leak.count,
                    'growth_rate': f"{leak.growth_rate*100:.1f}%",
                    'last_seen': leak.last_seen.isoformat(),
                    'stack_trace': leak.stack_trace
                })
        
        return {
            'status': 'leaks_detected' if leak_details else 'no_significant_leaks',
            'total_leaks': len(leak_details),
            'leaks': sorted(leak_details, key=lambda x: x['size_mb'], reverse=True)
        }
    
    async def optimize_for_batch_processing(self):
        """Optimize memory settings for batch processing"""
        # Increase GC thresholds for better batch performance
        gc.set_threshold(2000, 20, 20)
        
        # Pre-allocate memory if possible
        try:
            import resource
            # Increase memory limits if running as privileged user
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            if hard != resource.RLIM_INFINITY and soft < hard:
                resource.setrlimit(resource.RLIMIT_AS, (hard, hard))
        except Exception:
            pass
        
        logger.info("Optimized memory settings for batch processing")
    
    async def restore_default_settings(self):
        """Restore default memory settings"""
        gc.set_threshold(700, 10, 10)
        logger.info("Restored default memory settings")
    
    async def shutdown(self):
        """Shutdown memory manager"""
        self._shutdown_event.set()
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        # Final cleanup
        if self.tracemalloc_enabled:
            tracemalloc.stop()
        
        # Clear all data structures
        self.metrics_history.clear()
        self.object_counts.clear()
        self.object_sizes.clear()
        self.potential_leaks.clear()
        self.bounded_collections.clear()
        self.cleanup_tasks.clear()
        
        logger.info("Memory manager shutdown complete")


# Global memory manager instance
_memory_manager: Optional[MemoryManager] = None


async def get_memory_manager(
    gc_strategy: GCStrategy = GCStrategy.ADAPTIVE,
    memory_threshold_mb: int = 2048
) -> MemoryManager:
    """Get or create global memory manager"""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager(
            gc_strategy=gc_strategy,
            memory_threshold_mb=memory_threshold_mb
        )
        await _memory_manager.initialize()
    return _memory_manager


# Memory management decorators and utilities

def memory_efficient(func):
    """Decorator for memory-efficient function execution"""
    async def async_wrapper(*args, **kwargs):
        memory_manager = await get_memory_manager()
        
        # Take memory snapshot before
        initial_metrics = await memory_manager.collect_metrics()
        
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            # Cleanup after execution
            gc.collect()
            
            # Check for memory growth
            final_metrics = await memory_manager.collect_metrics()
            growth_mb = final_metrics.process_memory_mb - initial_metrics.process_memory_mb
            
            if growth_mb > 50:  # Significant memory growth
                logger.warning(
                    f"Function {func.__name__} increased memory by {growth_mb:.1f}MB"
                )
                await memory_manager.standard_cleanup()
    
    def sync_wrapper(*args, **kwargs):
        # Simplified sync version
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            gc.collect()
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


class BoundedDict(dict):
    """Dictionary with automatic size limiting to prevent memory leaks"""
    
    def __init__(self, max_size: int = 10000):
        super().__init__()
        self.max_size = max_size
        self.access_order = deque()
    
    def __setitem__(self, key, value):
        # Add to access order
        if key not in self:
            self.access_order.append(key)
        
        # Remove oldest if over limit
        while len(self) >= self.max_size:
            oldest_key = self.access_order.popleft()
            if oldest_key in self:
                super().__delitem__(oldest_key)
        
        super().__setitem__(key, value)
    
    def __getitem__(self, key):
        # Move to end of access order
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
        return super().__getitem__(key)


class BoundedList(list):
    """List with automatic size limiting"""
    
    def __init__(self, max_size: int = 10000):
        super().__init__()
        self.max_size = max_size
    
    def append(self, item):
        super().append(item)
        if len(self) > self.max_size:
            self.pop(0)
    
    def extend(self, items):
        super().extend(items)
        while len(self) > self.max_size:
            self.pop(0)