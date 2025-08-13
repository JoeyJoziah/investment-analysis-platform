"""
Comprehensive Performance Monitoring and Profiling System
Advanced performance analysis, monitoring, and optimization recommendations
"""

import asyncio
import time
import tracemalloc
import cProfile
import pstats
import io
import threading
import psutil
import gc
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import numpy as np
from collections import deque, defaultdict
import json
import os
import sys
import weakref
from contextlib import contextmanager
import functools

from backend.utils.memory_manager import get_memory_manager
from backend.utils.dynamic_resource_manager import get_resource_manager

logger = logging.getLogger(__name__)


class ProfilerType(Enum):
    """Types of profilers"""
    MEMORY = "memory"
    CPU = "cpu"
    IO = "io"
    NETWORK = "network"
    DATABASE = "database"
    CUSTOM = "custom"


class MetricType(Enum):
    """Types of performance metrics"""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    RESOURCE_USAGE = "resource_usage"
    QUEUE_SIZE = "queue_size"
    CACHE_HIT_RATE = "cache_hit_rate"


@dataclass
class PerformanceMetric:
    """Performance metric data point"""
    metric_type: MetricType
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'metric_type': self.metric_type.value,
            'name': self.name,
            'value': self.value,
            'timestamp': self.timestamp.isoformat(),
            'tags': self.tags,
            'unit': self.unit
        }


@dataclass
class ProfileReport:
    """Profiling report"""
    profile_type: ProfilerType
    duration_seconds: float
    start_time: datetime
    end_time: datetime
    metrics: List[PerformanceMetric]
    summary: Dict[str, Any]
    recommendations: List[str]
    raw_data: Optional[Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'profile_type': self.profile_type.value,
            'duration_seconds': self.duration_seconds,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'metrics': [m.to_dict() for m in self.metrics],
            'summary': self.summary,
            'recommendations': self.recommendations
        }


class PerformanceProfiler:
    """
    Comprehensive performance profiling system
    """
    
    def __init__(
        self,
        enable_memory_profiling: bool = True,
        enable_cpu_profiling: bool = True,
        enable_io_profiling: bool = True,
        sampling_interval: float = 0.1,
        max_samples: int = 10000
    ):
        self.enable_memory_profiling = enable_memory_profiling
        self.enable_cpu_profiling = enable_cpu_profiling
        self.enable_io_profiling = enable_io_profiling
        self.sampling_interval = sampling_interval
        self.max_samples = max_samples
        
        # Profiling data storage
        self.metrics_storage: deque = deque(maxlen=max_samples)
        self.active_profilers: Dict[str, Any] = {}
        self.profiling_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Performance baselines
        self.baselines: Dict[str, Dict[str, float]] = {}
        
        # Function call tracking
        self.function_stats: defaultdict = defaultdict(lambda: {
            'call_count': 0,
            'total_time': 0,
            'avg_time': 0,
            'max_time': 0,
            'min_time': float('inf'),
            'memory_usage': []
        })
        
        # Hot spots detection
        self.hot_spots: List[Dict[str, Any]] = []
        
        # Anomaly detection
        self.anomalies: deque = deque(maxlen=1000)
        
        # Background monitoring
        self._monitoring_active = False
        self._monitor_task: Optional[asyncio.Task] = None
        
        # Resource managers
        self._memory_manager = None
        self._resource_manager = None
    
    async def initialize(self):
        """Initialize the performance profiler"""
        # Initialize resource managers
        self._memory_manager = await get_memory_manager()
        self._resource_manager = await get_resource_manager()
        
        # Start memory profiling if enabled
        if self.enable_memory_profiling:
            tracemalloc.start(25)  # Keep 25 frames
        
        # Establish performance baselines
        await self._establish_baselines()
        
        # Start background monitoring
        await self._start_monitoring()
        
        logger.info("Performance profiler initialized")
    
    async def _establish_baselines(self):
        """Establish performance baselines"""
        logger.info("Establishing performance baselines...")
        
        # CPU baseline
        cpu_samples = []
        for _ in range(10):
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_samples.append(cpu_percent)
        
        # Memory baseline
        memory = psutil.virtual_memory()
        
        # I/O baseline
        io_start = psutil.disk_io_counters()
        await asyncio.sleep(1)
        io_end = psutil.disk_io_counters()
        
        if io_start and io_end:
            read_rate = (io_end.read_bytes - io_start.read_bytes) / (1024 * 1024)
            write_rate = (io_end.write_bytes - io_start.write_bytes) / (1024 * 1024)
        else:
            read_rate = write_rate = 0
        
        # Network baseline
        net_start = psutil.net_io_counters()
        await asyncio.sleep(1)
        net_end = psutil.net_io_counters()
        
        if net_start and net_end:
            net_send_rate = (net_end.bytes_sent - net_start.bytes_sent) / (1024 * 1024)
            net_recv_rate = (net_end.bytes_recv - net_start.bytes_recv) / (1024 * 1024)
        else:
            net_send_rate = net_recv_rate = 0
        
        self.baselines = {
            'cpu': {
                'avg_percent': np.mean(cpu_samples),
                'std_percent': np.std(cpu_samples)
            },
            'memory': {
                'baseline_percent': memory.percent,
                'baseline_used_gb': memory.used / (1024**3)
            },
            'io': {
                'baseline_read_mb_s': read_rate,
                'baseline_write_mb_s': write_rate
            },
            'network': {
                'baseline_send_mb_s': net_send_rate,
                'baseline_recv_mb_s': net_recv_rate
            }
        }
        
        logger.info(f"Baselines established: {self.baselines}")
    
    async def _start_monitoring(self):
        """Start background performance monitoring"""
        if not self._monitoring_active:
            self._monitoring_active = True
            self._monitor_task = asyncio.create_task(self._monitoring_loop())
    
    async def _monitoring_loop(self):
        """Background monitoring loop"""
        while self._monitoring_active:
            try:
                # Collect current metrics
                await self._collect_monitoring_metrics()
                
                # Detect anomalies
                await self._detect_anomalies()
                
                # Update hot spots
                await self._update_hot_spots()
                
                # Sleep for next interval
                await asyncio.sleep(self.sampling_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(1)
    
    async def _collect_monitoring_metrics(self):
        """Collect performance metrics"""
        timestamp = datetime.utcnow()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=None)
        self._add_metric(MetricType.RESOURCE_USAGE, "cpu_percent", cpu_percent, timestamp, unit="%")
        
        # Memory metrics
        memory = psutil.virtual_memory()
        self._add_metric(MetricType.RESOURCE_USAGE, "memory_percent", memory.percent, timestamp, unit="%")
        self._add_metric(MetricType.RESOURCE_USAGE, "memory_used_gb", memory.used / (1024**3), timestamp, unit="GB")
        
        # Process-specific metrics
        try:
            process = psutil.Process()
            process_memory = process.memory_info()
            self._add_metric(MetricType.RESOURCE_USAGE, "process_memory_rss_mb", 
                           process_memory.rss / (1024*1024), timestamp, unit="MB")
            self._add_metric(MetricType.RESOURCE_USAGE, "process_cpu_percent", 
                           process.cpu_percent(), timestamp, unit="%")
            self._add_metric(MetricType.RESOURCE_USAGE, "process_threads", 
                           process.num_threads(), timestamp, unit="count")
        except Exception as e:
            logger.warning(f"Error collecting process metrics: {e}")
        
        # GC metrics
        gc_stats = gc.get_stats()
        for i, stats in enumerate(gc_stats):
            self._add_metric(MetricType.RESOURCE_USAGE, f"gc_generation_{i}_collections", 
                           stats['collections'], timestamp, unit="count")
    
    def _add_metric(self, metric_type: MetricType, name: str, value: float, 
                   timestamp: datetime, tags: Optional[Dict[str, str]] = None, unit: str = ""):
        """Add a performance metric"""
        metric = PerformanceMetric(
            metric_type=metric_type,
            name=name,
            value=value,
            timestamp=timestamp,
            tags=tags or {},
            unit=unit
        )
        self.metrics_storage.append(metric)
    
    async def _detect_anomalies(self):
        """Detect performance anomalies"""
        if len(self.metrics_storage) < 100:
            return  # Need more data
        
        # Get recent metrics
        recent_metrics = list(self.metrics_storage)[-100:]
        
        # Group by metric name
        metric_groups = defaultdict(list)
        for metric in recent_metrics:
            metric_groups[metric.name].append(metric.value)
        
        # Check each metric for anomalies
        for metric_name, values in metric_groups.items():
            if len(values) < 50:
                continue
            
            # Calculate statistics
            recent_values = values[-10:]  # Last 10 values
            historical_values = values[-50:-10]  # Previous 40 values
            
            if len(historical_values) < 20:
                continue
            
            # Anomaly detection using z-score
            hist_mean = np.mean(historical_values)
            hist_std = np.std(historical_values)
            
            if hist_std == 0:
                continue
            
            recent_mean = np.mean(recent_values)
            z_score = abs(recent_mean - hist_mean) / hist_std
            
            # Detect anomaly (z-score > 3)
            if z_score > 3:
                anomaly = {
                    'timestamp': datetime.utcnow(),
                    'metric_name': metric_name,
                    'z_score': z_score,
                    'recent_mean': recent_mean,
                    'historical_mean': hist_mean,
                    'historical_std': hist_std,
                    'severity': 'high' if z_score > 5 else 'medium'
                }
                
                self.anomalies.append(anomaly)
                logger.warning(f"Performance anomaly detected: {anomaly}")
    
    async def _update_hot_spots(self):
        """Update performance hot spots"""
        # Clear old hot spots
        self.hot_spots.clear()
        
        # Analyze function stats
        for func_name, stats in self.function_stats.items():
            if stats['call_count'] > 10:  # Only consider functions called multiple times
                hot_spot = {
                    'function_name': func_name,
                    'call_count': stats['call_count'],
                    'total_time': stats['total_time'],
                    'avg_time': stats['avg_time'],
                    'max_time': stats['max_time'],
                    'efficiency_score': stats['total_time'] / stats['call_count'] if stats['call_count'] > 0 else 0
                }
                self.hot_spots.append(hot_spot)
        
        # Sort by efficiency score (higher is worse)
        self.hot_spots.sort(key=lambda x: x['efficiency_score'], reverse=True)
        
        # Keep only top 20 hot spots
        self.hot_spots = self.hot_spots[:20]
    
    @contextmanager
    def profile_memory(self, session_name: str):
        """Context manager for memory profiling"""
        if not self.enable_memory_profiling:
            yield
            return
        
        # Take initial snapshot
        if tracemalloc.is_tracing():
            snapshot_start = tracemalloc.take_snapshot()
        else:
            tracemalloc.start(25)
            snapshot_start = None
        
        start_time = datetime.utcnow()
        
        try:
            yield
        finally:
            end_time = datetime.utcnow()
            
            if tracemalloc.is_tracing():
                snapshot_end = tracemalloc.take_snapshot()
                
                if snapshot_start:
                    # Compare snapshots
                    top_stats = snapshot_end.compare_to(snapshot_start, 'lineno')
                else:
                    top_stats = snapshot_end.statistics('lineno')
                
                # Analyze memory usage
                total_memory = sum(stat.size for stat in top_stats)
                
                # Store profiling session
                self.profiling_sessions[session_name] = {
                    'type': 'memory',
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': (end_time - start_time).total_seconds(),
                    'total_memory_bytes': total_memory,
                    'top_stats': top_stats[:10]  # Top 10 memory consumers
                }
                
                logger.info(f"Memory profiling completed for {session_name}: {total_memory / (1024*1024):.2f} MB")
    
    @contextmanager
    def profile_cpu(self, session_name: str):
        """Context manager for CPU profiling"""
        if not self.enable_cpu_profiling:
            yield
            return
        
        profiler = cProfile.Profile()
        start_time = datetime.utcnow()
        
        profiler.enable()
        
        try:
            yield
        finally:
            profiler.disable()
            end_time = datetime.utcnow()
            
            # Analyze profiling data
            stats_stream = io.StringIO()
            stats = pstats.Stats(profiler, stream=stats_stream)
            stats.sort_stats('cumulative')
            stats.print_stats(20)  # Top 20 functions
            
            # Store profiling session
            self.profiling_sessions[session_name] = {
                'type': 'cpu',
                'start_time': start_time,
                'end_time': end_time,
                'duration': (end_time - start_time).total_seconds(),
                'stats_output': stats_stream.getvalue(),
                'function_count': len(stats.stats)
            }
            
            logger.info(f"CPU profiling completed for {session_name}")
    
    def profile_function(self, func_name: Optional[str] = None):
        """Decorator for profiling individual functions"""
        def decorator(func):
            name = func_name or f"{func.__module__}.{func.__name__}"
            
            if asyncio.iscoroutinefunction(func):
                @functools.wraps(func)
                async def async_wrapper(*args, **kwargs):
                    start_time = time.perf_counter()
                    start_memory = psutil.Process().memory_info().rss
                    
                    try:
                        result = await func(*args, **kwargs)
                        return result
                    except Exception as e:
                        # Track errors
                        self._add_metric(MetricType.ERROR_RATE, f"{name}_errors", 1, 
                                       datetime.utcnow(), tags={'function': name})
                        raise
                    finally:
                        end_time = time.perf_counter()
                        end_memory = psutil.Process().memory_info().rss
                        
                        execution_time = end_time - start_time
                        memory_delta = end_memory - start_memory
                        
                        # Update function stats
                        stats = self.function_stats[name]
                        stats['call_count'] += 1
                        stats['total_time'] += execution_time
                        stats['avg_time'] = stats['total_time'] / stats['call_count']
                        stats['max_time'] = max(stats['max_time'], execution_time)
                        stats['min_time'] = min(stats['min_time'], execution_time)
                        stats['memory_usage'].append(memory_delta)
                        
                        # Keep only recent memory usage data
                        if len(stats['memory_usage']) > 100:
                            stats['memory_usage'] = stats['memory_usage'][-100:]
                        
                        # Add metrics
                        timestamp = datetime.utcnow()
                        self._add_metric(MetricType.LATENCY, f"{name}_duration", execution_time, 
                                       timestamp, tags={'function': name}, unit='seconds')
                        self._add_metric(MetricType.RESOURCE_USAGE, f"{name}_memory_delta", memory_delta, 
                                       timestamp, tags={'function': name}, unit='bytes')
                
                return async_wrapper
            else:
                @functools.wraps(func)
                def sync_wrapper(*args, **kwargs):
                    start_time = time.perf_counter()
                    start_memory = psutil.Process().memory_info().rss
                    
                    try:
                        result = func(*args, **kwargs)
                        return result
                    except Exception as e:
                        # Track errors
                        self._add_metric(MetricType.ERROR_RATE, f"{name}_errors", 1, 
                                       datetime.utcnow(), tags={'function': name})
                        raise
                    finally:
                        end_time = time.perf_counter()
                        end_memory = psutil.Process().memory_info().rss
                        
                        execution_time = end_time - start_time
                        memory_delta = end_memory - start_memory
                        
                        # Update function stats
                        stats = self.function_stats[name]
                        stats['call_count'] += 1
                        stats['total_time'] += execution_time
                        stats['avg_time'] = stats['total_time'] / stats['call_count']
                        stats['max_time'] = max(stats['max_time'], execution_time)
                        stats['min_time'] = min(stats['min_time'], execution_time)
                        stats['memory_usage'].append(memory_delta)
                        
                        # Keep only recent memory usage data
                        if len(stats['memory_usage']) > 100:
                            stats['memory_usage'] = stats['memory_usage'][-100:]
                        
                        # Add metrics
                        timestamp = datetime.utcnow()
                        self._add_metric(MetricType.LATENCY, f"{name}_duration", execution_time, 
                                       timestamp, tags={'function': name}, unit='seconds')
                        self._add_metric(MetricType.RESOURCE_USAGE, f"{name}_memory_delta", memory_delta, 
                                       timestamp, tags={'function': name}, unit='bytes')
                
                return sync_wrapper
        
        return decorator
    
    def generate_performance_report(self, time_range_minutes: int = 60) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=time_range_minutes)
        
        # Filter metrics by time range
        relevant_metrics = [
            metric for metric in self.metrics_storage
            if start_time <= metric.timestamp <= end_time
        ]
        
        if not relevant_metrics:
            return {'status': 'no_data', 'message': 'No metrics available for the specified time range'}
        
        # Group metrics by type and name
        metric_groups = defaultdict(list)
        for metric in relevant_metrics:
            key = f"{metric.metric_type.value}_{metric.name}"
            metric_groups[key].append(metric.value)
        
        # Calculate statistics for each metric
        metric_stats = {}
        for key, values in metric_groups.items():
            if values:
                metric_stats[key] = {
                    'count': len(values),
                    'min': min(values),
                    'max': max(values),
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'p50': np.percentile(values, 50),
                    'p95': np.percentile(values, 95),
                    'p99': np.percentile(values, 99)
                }
        
        # Generate recommendations
        recommendations = self._generate_recommendations(metric_stats)
        
        # Compile report
        report = {
            'report_generated': datetime.utcnow().isoformat(),
            'time_range_minutes': time_range_minutes,
            'total_metrics': len(relevant_metrics),
            'metric_statistics': metric_stats,
            'function_statistics': dict(self.function_stats),
            'hot_spots': self.hot_spots,
            'recent_anomalies': [
                {
                    'timestamp': anomaly['timestamp'].isoformat(),
                    'metric_name': anomaly['metric_name'],
                    'z_score': anomaly['z_score'],
                    'severity': anomaly['severity']
                }
                for anomaly in list(self.anomalies)[-10:]  # Last 10 anomalies
            ],
            'baselines': self.baselines,
            'recommendations': recommendations,
            'profiling_sessions': {
                name: {
                    **session,
                    'start_time': session['start_time'].isoformat(),
                    'end_time': session['end_time'].isoformat(),
                    'top_stats': [str(stat) for stat in session.get('top_stats', [])]
                }
                for name, session in self.profiling_sessions.items()
            }
        }
        
        return report
    
    def _generate_recommendations(self, metric_stats: Dict[str, Dict[str, float]]) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        # Check CPU usage
        cpu_key = 'resource_usage_cpu_percent'
        if cpu_key in metric_stats:
            cpu_stats = metric_stats[cpu_key]
            if cpu_stats['mean'] > 80:
                recommendations.append(
                    f"High CPU usage detected (avg: {cpu_stats['mean']:.1f}%). "
                    "Consider optimizing CPU-intensive operations or scaling horizontally."
                )
            elif cpu_stats['p95'] > 90:
                recommendations.append(
                    f"CPU spikes detected (95th percentile: {cpu_stats['p95']:.1f}%). "
                    "Investigate peak load handling and consider load balancing."
                )
        
        # Check memory usage
        memory_key = 'resource_usage_memory_percent'
        if memory_key in metric_stats:
            memory_stats = metric_stats[memory_key]
            if memory_stats['mean'] > 85:
                recommendations.append(
                    f"High memory usage detected (avg: {memory_stats['mean']:.1f}%). "
                    "Consider memory optimization or increasing available memory."
                )
            elif memory_stats['std'] > 10:
                recommendations.append(
                    f"High memory usage volatility detected (std: {memory_stats['std']:.1f}%). "
                    "This may indicate memory leaks or inefficient garbage collection."
                )
        
        # Check process memory
        process_memory_key = 'resource_usage_process_memory_rss_mb'
        if process_memory_key in metric_stats:
            process_memory_stats = metric_stats[process_memory_key]
            if process_memory_stats['max'] > 2000:  # > 2GB
                recommendations.append(
                    f"High process memory usage detected (max: {process_memory_stats['max']:.0f}MB). "
                    "Consider memory profiling and optimization."
                )
        
        # Check function performance
        slow_functions = [
            name for name, stats in self.function_stats.items()
            if stats['avg_time'] > 1.0  # Functions taking more than 1 second on average
        ]
        
        if slow_functions:
            recommendations.append(
                f"Slow functions detected: {', '.join(slow_functions[:5])}. "
                "Consider optimizing these functions for better performance."
            )
        
        # Check for hot spots
        if len(self.hot_spots) > 0:
            top_hot_spot = self.hot_spots[0]
            recommendations.append(
                f"Performance hot spot detected: {top_hot_spot['function_name']} "
                f"(avg time: {top_hot_spot['avg_time']:.3f}s, calls: {top_hot_spot['call_count']}). "
                "Focus optimization efforts on this function."
            )
        
        # Check for anomalies
        recent_anomalies = [a for a in self.anomalies if 
                           (datetime.utcnow() - a['timestamp']).total_seconds() < 3600]  # Last hour
        
        if len(recent_anomalies) > 5:
            recommendations.append(
                f"Multiple performance anomalies detected ({len(recent_anomalies)} in last hour). "
                "System may be under unusual load or experiencing issues."
            )
        
        # GC recommendations
        for generation in range(3):
            gc_key = f'resource_usage_gc_generation_{generation}_collections'
            if gc_key in metric_stats:
                gc_stats = metric_stats[gc_key]
                if gc_stats['max'] - gc_stats['min'] > 100:  # High GC activity
                    recommendations.append(
                        f"High garbage collection activity detected for generation {generation}. "
                        "Consider memory usage optimization and GC tuning."
                    )
                    break
        
        if not recommendations:
            recommendations.append("No significant performance issues detected. System is performing well.")
        
        return recommendations
    
    def export_metrics(self, format: str = 'json', time_range_minutes: int = 60) -> str:
        """Export metrics in various formats"""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=time_range_minutes)
        
        # Filter metrics by time range
        relevant_metrics = [
            metric for metric in self.metrics_storage
            if start_time <= metric.timestamp <= end_time
        ]
        
        if format.lower() == 'json':
            data = {
                'export_time': datetime.utcnow().isoformat(),
                'time_range_minutes': time_range_minutes,
                'metrics': [metric.to_dict() for metric in relevant_metrics]
            }
            return json.dumps(data, indent=2)
        
        elif format.lower() == 'csv':
            import csv
            output = io.StringIO()
            
            if relevant_metrics:
                writer = csv.DictWriter(output, fieldnames=[
                    'timestamp', 'metric_type', 'name', 'value', 'unit', 'tags'
                ])
                writer.writeheader()
                
                for metric in relevant_metrics:
                    writer.writerow({
                        'timestamp': metric.timestamp.isoformat(),
                        'metric_type': metric.metric_type.value,
                        'name': metric.name,
                        'value': metric.value,
                        'unit': metric.unit,
                        'tags': json.dumps(metric.tags)
                    })
            
            return output.getvalue()
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    async def shutdown(self):
        """Shutdown the performance profiler"""
        self._monitoring_active = False
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        # Stop tracemalloc if it's running
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        
        # Clear data structures
        self.metrics_storage.clear()
        self.active_profilers.clear()
        self.profiling_sessions.clear()
        self.function_stats.clear()
        self.hot_spots.clear()
        self.anomalies.clear()
        
        logger.info("Performance profiler shutdown complete")


# Global performance profiler instance
_performance_profiler: Optional[PerformanceProfiler] = None


async def get_performance_profiler(
    enable_memory_profiling: bool = True,
    enable_cpu_profiling: bool = True,
    enable_io_profiling: bool = True
) -> PerformanceProfiler:
    """Get or create global performance profiler"""
    global _performance_profiler
    if _performance_profiler is None:
        _performance_profiler = PerformanceProfiler(
            enable_memory_profiling=enable_memory_profiling,
            enable_cpu_profiling=enable_cpu_profiling,
            enable_io_profiling=enable_io_profiling
        )
        await _performance_profiler.initialize()
    return _performance_profiler


# Convenience decorators
def profile_performance(func_name: Optional[str] = None):
    """Convenience decorator for performance profiling"""
    async def get_profiler():
        return await get_performance_profiler()
    
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                profiler = await get_profiler()
                decorated_func = profiler.profile_function(func_name)(func)
                return await decorated_func(*args, **kwargs)
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                # For sync functions, we need to handle this differently
                # This is a simplified version - in practice, you might want to use a different approach
                return func(*args, **kwargs)
            return sync_wrapper
    
    return decorator