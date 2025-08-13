"""
Dynamic Resource Allocation and Performance Management System
Automatically adjusts system resources based on workload and performance metrics
"""

import asyncio
import psutil
import time
import threading
import multiprocessing
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import numpy as np
from collections import deque, defaultdict
import json
import os

from backend.utils.memory_manager import get_memory_manager, MemoryPressureLevel
from backend.config.settings import settings

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Types of system resources"""
    CPU = "cpu"
    MEMORY = "memory"
    IO = "io"
    NETWORK = "network"
    DATABASE = "database"
    CACHE = "cache"


class WorkloadType(Enum):
    """Types of workloads"""
    CPU_INTENSIVE = "cpu_intensive"
    MEMORY_INTENSIVE = "memory_intensive"
    IO_INTENSIVE = "io_intensive"
    NETWORK_INTENSIVE = "network_intensive"
    BALANCED = "balanced"
    BATCH_PROCESSING = "batch_processing"
    REAL_TIME = "real_time"


class ResourcePressureLevel(Enum):
    """Resource pressure levels"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class ResourceMetrics:
    """Comprehensive resource metrics"""
    timestamp: datetime
    
    # CPU metrics
    cpu_percent: float
    cpu_count_logical: int
    cpu_count_physical: int
    cpu_freq_current: float
    cpu_temp: Optional[float] = None
    
    # Memory metrics
    memory_total_gb: float
    memory_used_gb: float
    memory_available_gb: float
    memory_percent: float
    swap_total_gb: float
    swap_used_gb: float
    
    # Disk I/O metrics
    disk_read_mb_s: float
    disk_write_mb_s: float
    disk_usage_percent: float
    disk_iops: float
    
    # Network metrics
    network_sent_mb_s: float
    network_recv_mb_s: float
    network_connections: int
    
    # Process metrics
    process_count: int
    thread_count: int
    file_descriptor_count: int
    
    # Custom application metrics
    active_requests: int = 0
    queue_size: int = 0
    cache_hit_rate: float = 0.0
    database_connections: int = 0
    
    @property
    def overall_pressure_level(self) -> ResourcePressureLevel:
        """Calculate overall resource pressure level"""
        pressure_scores = []
        
        # CPU pressure
        if self.cpu_percent > 90:
            pressure_scores.append(5)  # Emergency
        elif self.cpu_percent > 80:
            pressure_scores.append(4)  # Critical
        elif self.cpu_percent > 70:
            pressure_scores.append(3)  # High
        elif self.cpu_percent > 50:
            pressure_scores.append(2)  # Moderate
        else:
            pressure_scores.append(1)  # Low
        
        # Memory pressure
        if self.memory_percent > 95:
            pressure_scores.append(5)
        elif self.memory_percent > 85:
            pressure_scores.append(4)
        elif self.memory_percent > 75:
            pressure_scores.append(3)
        elif self.memory_percent > 60:
            pressure_scores.append(2)
        else:
            pressure_scores.append(1)
        
        # I/O pressure (simplified)
        io_pressure = min(self.disk_read_mb_s + self.disk_write_mb_s, 1000) / 1000
        if io_pressure > 0.9:
            pressure_scores.append(4)
        elif io_pressure > 0.7:
            pressure_scores.append(3)
        elif io_pressure > 0.5:
            pressure_scores.append(2)
        else:
            pressure_scores.append(1)
        
        # Calculate average pressure
        avg_pressure = np.mean(pressure_scores)
        
        if avg_pressure >= 4.5:
            return ResourcePressureLevel.EMERGENCY
        elif avg_pressure >= 3.5:
            return ResourcePressureLevel.CRITICAL
        elif avg_pressure >= 2.5:
            return ResourcePressureLevel.HIGH
        elif avg_pressure >= 1.5:
            return ResourcePressureLevel.MODERATE
        else:
            return ResourcePressureLevel.LOW


@dataclass
class ResourceAllocation:
    """Resource allocation configuration"""
    cpu_workers: int
    memory_limit_mb: int
    io_concurrent_limit: int
    network_concurrent_limit: int
    database_pool_size: int
    cache_size_mb: int
    batch_size: int
    queue_size_limit: int
    
    # Performance targets
    target_cpu_percent: float = 70.0
    target_memory_percent: float = 75.0
    target_response_time_ms: float = 1000.0


@dataclass
class WorkloadProfile:
    """Workload characterization profile"""
    workload_type: WorkloadType
    intensity_level: float  # 0.0 to 1.0
    duration_estimate_s: Optional[float] = None
    resource_requirements: Dict[ResourceType, float] = field(default_factory=dict)
    priority: int = 5  # 1 = highest, 10 = lowest
    
    # SLA requirements
    max_latency_ms: Optional[float] = None
    min_throughput_rps: Optional[float] = None
    availability_requirement: float = 0.99  # 99% availability


class DynamicResourceManager:
    """
    Dynamic resource allocation and performance management system
    """
    
    def __init__(
        self,
        monitoring_interval_s: int = 30,
        enable_auto_scaling: bool = True,
        enable_predictive_scaling: bool = True,
        enable_resource_optimization: bool = True
    ):
        self.monitoring_interval_s = monitoring_interval_s
        self.enable_auto_scaling = enable_auto_scaling
        self.enable_predictive_scaling = enable_predictive_scaling
        self.enable_resource_optimization = enable_resource_optimization
        
        # Metrics storage
        self.metrics_history: deque = deque(maxlen=1000)
        self.resource_allocations: Dict[str, ResourceAllocation] = {}
        self.workload_profiles: Dict[str, WorkloadProfile] = {}
        
        # Performance tracking
        self.performance_history: deque = deque(maxlen=500)
        self.optimization_history: deque = deque(maxlen=100)
        
        # Resource limits and targets
        self.system_limits = self._detect_system_limits()
        self.current_allocation = self._calculate_initial_allocation()
        
        # Predictive models (simplified)
        self.cpu_trend: deque = deque(maxlen=50)
        self.memory_trend: deque = deque(maxlen=50)
        self.workload_patterns: Dict[str, List[float]] = defaultdict(list)
        
        # Control flags
        self._monitoring_active = False
        self._shutdown_event = threading.Event()
        self._monitor_task: Optional[asyncio.Task] = None
        
        # Memory manager integration
        self._memory_manager = None
        
        # Callbacks for resource changes
        self._resource_change_callbacks: List[Callable] = []
    
    def _detect_system_limits(self) -> Dict[str, Any]:
        """Detect system resource limits"""
        try:
            # CPU limits
            cpu_count = multiprocessing.cpu_count()
            cpu_logical = psutil.cpu_count(logical=True)
            cpu_physical = psutil.cpu_count(logical=False)
            
            # Memory limits
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Disk limits (primary disk)
            disk_usage = psutil.disk_usage('/')
            
            # Network limits (estimated)
            network_limit = 1000  # 1 Gbps default assumption
            
            return {
                'cpu': {
                    'logical_cores': cpu_logical,
                    'physical_cores': cpu_physical,
                    'max_workers': min(cpu_logical * 2, 32),  # Conservative limit
                    'safe_utilization': 80.0  # Max 80% CPU usage
                },
                'memory': {
                    'total_gb': memory.total / (1024**3),
                    'safe_limit_gb': memory.total * 0.8 / (1024**3),  # 80% of total
                    'swap_gb': swap.total / (1024**3)
                },
                'disk': {
                    'total_gb': disk_usage.total / (1024**3),
                    'safe_limit_percent': 90.0
                },
                'network': {
                    'estimated_limit_mbps': network_limit,
                    'safe_limit_mbps': network_limit * 0.8
                }
            }
        except Exception as e:
            logger.error(f"Error detecting system limits: {e}")
            # Return conservative defaults
            return {
                'cpu': {'logical_cores': 4, 'physical_cores': 2, 'max_workers': 8, 'safe_utilization': 70.0},
                'memory': {'total_gb': 8, 'safe_limit_gb': 6, 'swap_gb': 2},
                'disk': {'total_gb': 100, 'safe_limit_percent': 85.0},
                'network': {'estimated_limit_mbps': 100, 'safe_limit_mbps': 80}
            }
    
    def _calculate_initial_allocation(self) -> ResourceAllocation:
        """Calculate initial resource allocation based on system limits"""
        limits = self.system_limits
        
        return ResourceAllocation(
            cpu_workers=max(4, limits['cpu']['logical_cores']),
            memory_limit_mb=int(limits['memory']['safe_limit_gb'] * 1024 * 0.6),  # 60% of safe limit
            io_concurrent_limit=20,
            network_concurrent_limit=50,
            database_pool_size=20,
            cache_size_mb=int(limits['memory']['safe_limit_gb'] * 1024 * 0.2),  # 20% for cache
            batch_size=100,
            queue_size_limit=1000,
            target_cpu_percent=70.0,
            target_memory_percent=70.0,
            target_response_time_ms=1000.0
        )
    
    async def initialize(self):
        """Initialize the dynamic resource manager"""
        # Get memory manager
        self._memory_manager = await get_memory_manager()
        
        # Start monitoring
        if not self._monitoring_active:
            self._monitoring_active = True
            self._monitor_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Dynamic resource manager initialized")
    
    async def _monitoring_loop(self):
        """Main monitoring and optimization loop"""
        while self._monitoring_active and not self._shutdown_event.is_set():
            try:
                # Collect current metrics
                metrics = await self._collect_comprehensive_metrics()
                self.metrics_history.append(metrics)
                
                # Update trends
                self._update_trends(metrics)
                
                # Check if optimization is needed
                if self._should_optimize(metrics):
                    await self._optimize_resources(metrics)
                
                # Predictive scaling
                if self.enable_predictive_scaling:
                    await self._predictive_scaling()
                
                # Performance analysis
                await self._analyze_performance()
                
                # Sleep until next monitoring cycle
                await asyncio.sleep(self.monitoring_interval_s)
                
            except Exception as e:
                logger.error(f"Error in resource monitoring loop: {e}")
                await asyncio.sleep(self.monitoring_interval_s)
    
    async def _collect_comprehensive_metrics(self) -> ResourceMetrics:
        """Collect comprehensive system and application metrics"""
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count_logical = psutil.cpu_count(logical=True)
        cpu_count_physical = psutil.cpu_count(logical=False)
        
        try:
            cpu_freq = psutil.cpu_freq()
            cpu_freq_current = cpu_freq.current if cpu_freq else 0.0
        except:
            cpu_freq_current = 0.0
        
        # Temperature (if available)
        cpu_temp = None
        try:
            temps = psutil.sensors_temperatures()
            if temps and 'coretemp' in temps:
                cpu_temp = temps['coretemp'][0].current
        except:
            pass
        
        # Memory metrics
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # Disk I/O metrics
        disk_io_start = psutil.disk_io_counters()
        await asyncio.sleep(0.1)  # Small interval for rate calculation
        disk_io_end = psutil.disk_io_counters()
        
        if disk_io_start and disk_io_end:
            disk_read_mb_s = (disk_io_end.read_bytes - disk_io_start.read_bytes) / (1024 * 1024) / 0.1
            disk_write_mb_s = (disk_io_end.write_bytes - disk_io_start.write_bytes) / (1024 * 1024) / 0.1
            disk_iops = (disk_io_end.read_count + disk_io_end.write_count - 
                        disk_io_start.read_count - disk_io_start.write_count) / 0.1
        else:
            disk_read_mb_s = disk_write_mb_s = disk_iops = 0.0
        
        # Disk usage
        disk_usage = psutil.disk_usage('/')
        disk_usage_percent = (disk_usage.used / disk_usage.total) * 100
        
        # Network metrics
        network_io_start = psutil.net_io_counters()
        await asyncio.sleep(0.1)
        network_io_end = psutil.net_io_counters()
        
        if network_io_start and network_io_end:
            network_sent_mb_s = (network_io_end.bytes_sent - network_io_start.bytes_sent) / (1024 * 1024) / 0.1
            network_recv_mb_s = (network_io_end.bytes_recv - network_io_start.bytes_recv) / (1024 * 1024) / 0.1
        else:
            network_sent_mb_s = network_recv_mb_s = 0.0
        
        # Network connections
        try:
            network_connections = len(psutil.net_connections())
        except:
            network_connections = 0
        
        # Process metrics
        try:
            process_count = len(psutil.pids())
            current_process = psutil.Process()
            thread_count = current_process.num_threads()
            
            # File descriptors (Unix only)
            try:
                file_descriptor_count = current_process.num_fds()
            except:
                file_descriptor_count = 0
        except:
            process_count = thread_count = file_descriptor_count = 0
        
        return ResourceMetrics(
            timestamp=datetime.utcnow(),
            cpu_percent=cpu_percent,
            cpu_count_logical=cpu_count_logical,
            cpu_count_physical=cpu_count_physical,
            cpu_freq_current=cpu_freq_current,
            cpu_temp=cpu_temp,
            memory_total_gb=memory.total / (1024**3),
            memory_used_gb=memory.used / (1024**3),
            memory_available_gb=memory.available / (1024**3),
            memory_percent=memory.percent,
            swap_total_gb=swap.total / (1024**3),
            swap_used_gb=swap.used / (1024**3),
            disk_read_mb_s=disk_read_mb_s,
            disk_write_mb_s=disk_write_mb_s,
            disk_usage_percent=disk_usage_percent,
            disk_iops=disk_iops,
            network_sent_mb_s=network_sent_mb_s,
            network_recv_mb_s=network_recv_mb_s,
            network_connections=network_connections,
            process_count=process_count,
            thread_count=thread_count,
            file_descriptor_count=file_descriptor_count
        )
    
    def _update_trends(self, metrics: ResourceMetrics):
        """Update resource usage trends"""
        self.cpu_trend.append(metrics.cpu_percent)
        self.memory_trend.append(metrics.memory_percent)
    
    def _should_optimize(self, metrics: ResourceMetrics) -> bool:
        """Determine if resource optimization is needed"""
        if not self.enable_auto_scaling:
            return False
        
        pressure_level = metrics.overall_pressure_level
        
        # Optimize if pressure is high or critical
        if pressure_level in [ResourcePressureLevel.HIGH, ResourcePressureLevel.CRITICAL, ResourcePressureLevel.EMERGENCY]:
            return True
        
        # Optimize if we're significantly under-utilizing resources
        if (metrics.cpu_percent < 30 and metrics.memory_percent < 40 and 
            len(self.metrics_history) > 10):
            # Check if this is a sustained pattern
            recent_cpu = [m.cpu_percent for m in list(self.metrics_history)[-10:]]
            recent_memory = [m.memory_percent for m in list(self.metrics_history)[-10:]]
            
            if np.mean(recent_cpu) < 30 and np.mean(recent_memory) < 40:
                return True
        
        return False
    
    async def _optimize_resources(self, metrics: ResourceMetrics):
        """Optimize resource allocation based on current metrics"""
        logger.info(f"Optimizing resources - Pressure level: {metrics.overall_pressure_level.value}")
        
        optimization_start = time.time()
        
        # Create new allocation based on current metrics
        new_allocation = self._calculate_optimal_allocation(metrics)
        
        # Apply changes if significant
        changes_made = await self._apply_allocation_changes(new_allocation, metrics)
        
        # Record optimization
        optimization_time = time.time() - optimization_start
        
        self.optimization_history.append({
            'timestamp': datetime.utcnow(),
            'metrics': metrics,
            'old_allocation': self.current_allocation,
            'new_allocation': new_allocation,
            'changes_made': changes_made,
            'optimization_time_ms': optimization_time * 1000
        })
        
        if changes_made:
            self.current_allocation = new_allocation
            logger.info(f"Resource optimization completed in {optimization_time:.3f}s")
        else:
            logger.debug("No significant resource changes needed")
    
    def _calculate_optimal_allocation(self, metrics: ResourceMetrics) -> ResourceAllocation:
        """Calculate optimal resource allocation"""
        current = self.current_allocation
        limits = self.system_limits
        pressure = metrics.overall_pressure_level
        
        # Start with current allocation
        new_cpu_workers = current.cpu_workers
        new_memory_limit = current.memory_limit_mb
        new_io_limit = current.io_concurrent_limit
        new_network_limit = current.network_concurrent_limit
        new_batch_size = current.batch_size
        
        # Adjust based on pressure level
        if pressure == ResourcePressureLevel.EMERGENCY:
            # Severe resource pressure - reduce everything
            new_cpu_workers = max(2, int(current.cpu_workers * 0.5))
            new_memory_limit = int(current.memory_limit_mb * 0.6)
            new_io_limit = max(5, int(current.io_concurrent_limit * 0.5))
            new_network_limit = max(10, int(current.network_concurrent_limit * 0.5))
            new_batch_size = max(10, int(current.batch_size * 0.5))
            
        elif pressure == ResourcePressureLevel.CRITICAL:
            # High pressure - reduce moderately
            new_cpu_workers = max(2, int(current.cpu_workers * 0.7))
            new_memory_limit = int(current.memory_limit_mb * 0.8)
            new_io_limit = max(5, int(current.io_concurrent_limit * 0.7))
            new_network_limit = max(10, int(current.network_concurrent_limit * 0.7))
            new_batch_size = max(10, int(current.batch_size * 0.8))
            
        elif pressure == ResourcePressureLevel.HIGH:
            # Moderate pressure - reduce slightly
            new_cpu_workers = max(2, int(current.cpu_workers * 0.9))
            new_memory_limit = int(current.memory_limit_mb * 0.9)
            new_io_limit = max(5, int(current.io_concurrent_limit * 0.9))
            new_network_limit = max(10, int(current.network_concurrent_limit * 0.9))
            
        elif pressure == ResourcePressureLevel.LOW:
            # Low pressure - can increase if under-utilized
            if metrics.cpu_percent < 40:
                new_cpu_workers = min(limits['cpu']['max_workers'], int(current.cpu_workers * 1.2))
            
            if metrics.memory_percent < 50:
                max_memory = int(limits['memory']['safe_limit_gb'] * 1024)
                new_memory_limit = min(max_memory, int(current.memory_limit_mb * 1.1))
            
            # Increase concurrency limits if not saturated
            new_io_limit = min(50, int(current.io_concurrent_limit * 1.1))
            new_network_limit = min(100, int(current.network_concurrent_limit * 1.1))
            new_batch_size = min(500, int(current.batch_size * 1.1))
        
        # Apply absolute limits
        new_cpu_workers = max(1, min(limits['cpu']['max_workers'], new_cpu_workers))
        max_memory_mb = int(limits['memory']['safe_limit_gb'] * 1024)
        new_memory_limit = max(512, min(max_memory_mb, new_memory_limit))
        
        return ResourceAllocation(
            cpu_workers=new_cpu_workers,
            memory_limit_mb=new_memory_limit,
            io_concurrent_limit=new_io_limit,
            network_concurrent_limit=new_network_limit,
            database_pool_size=current.database_pool_size,  # Keep stable for now
            cache_size_mb=current.cache_size_mb,  # Keep stable for now
            batch_size=new_batch_size,
            queue_size_limit=current.queue_size_limit,  # Keep stable for now
            target_cpu_percent=current.target_cpu_percent,
            target_memory_percent=current.target_memory_percent,
            target_response_time_ms=current.target_response_time_ms
        )
    
    async def _apply_allocation_changes(
        self,
        new_allocation: ResourceAllocation,
        metrics: ResourceMetrics
    ) -> bool:
        """Apply resource allocation changes"""
        current = self.current_allocation
        changes_made = False
        
        # Check for significant changes (> 10% difference)
        def is_significant_change(old_val, new_val, threshold=0.1):
            return abs(new_val - old_val) / old_val > threshold if old_val > 0 else new_val > 0
        
        # CPU workers change
        if new_allocation.cpu_workers != current.cpu_workers:
            logger.info(f"Adjusting CPU workers: {current.cpu_workers} -> {new_allocation.cpu_workers}")
            await self._notify_resource_change('cpu_workers', current.cpu_workers, new_allocation.cpu_workers)
            changes_made = True
        
        # Memory limit change
        if is_significant_change(current.memory_limit_mb, new_allocation.memory_limit_mb):
            logger.info(f"Adjusting memory limit: {current.memory_limit_mb}MB -> {new_allocation.memory_limit_mb}MB")
            
            # Apply memory management optimization
            if self._memory_manager:
                if new_allocation.memory_limit_mb < current.memory_limit_mb:
                    # Reducing memory limit - trigger cleanup
                    await self._memory_manager.aggressive_cleanup()
                else:
                    # Increasing memory limit - optimize for performance
                    await self._memory_manager.optimize_for_batch_processing()
            
            await self._notify_resource_change('memory_limit', current.memory_limit_mb, new_allocation.memory_limit_mb)
            changes_made = True
        
        # Concurrency limits
        if new_allocation.io_concurrent_limit != current.io_concurrent_limit:
            logger.info(f"Adjusting I/O concurrency: {current.io_concurrent_limit} -> {new_allocation.io_concurrent_limit}")
            await self._notify_resource_change('io_concurrency', current.io_concurrent_limit, new_allocation.io_concurrent_limit)
            changes_made = True
        
        if new_allocation.network_concurrent_limit != current.network_concurrent_limit:
            logger.info(f"Adjusting network concurrency: {current.network_concurrent_limit} -> {new_allocation.network_concurrent_limit}")
            await self._notify_resource_change('network_concurrency', current.network_concurrent_limit, new_allocation.network_concurrent_limit)
            changes_made = True
        
        # Batch size
        if is_significant_change(current.batch_size, new_allocation.batch_size):
            logger.info(f"Adjusting batch size: {current.batch_size} -> {new_allocation.batch_size}")
            await self._notify_resource_change('batch_size', current.batch_size, new_allocation.batch_size)
            changes_made = True
        
        return changes_made
    
    async def _notify_resource_change(self, resource_type: str, old_value: Any, new_value: Any):
        """Notify registered callbacks about resource changes"""
        for callback in self._resource_change_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(resource_type, old_value, new_value)
                else:
                    callback(resource_type, old_value, new_value)
            except Exception as e:
                logger.error(f"Error in resource change callback: {e}")
    
    async def _predictive_scaling(self):
        """Perform predictive scaling based on trends"""
        if len(self.cpu_trend) < 20 or len(self.memory_trend) < 20:
            return  # Not enough data
        
        # Simple trend analysis
        cpu_trend = np.array(self.cpu_trend)
        memory_trend = np.array(self.memory_trend)
        
        # Calculate trends (simple linear regression)
        cpu_slope = np.polyfit(range(len(cpu_trend)), cpu_trend, 1)[0]
        memory_slope = np.polyfit(range(len(memory_trend)), memory_trend, 1)[0]
        
        # Predict future values (next 10 monitoring intervals)
        cpu_prediction = cpu_trend[-1] + cpu_slope * 10
        memory_prediction = memory_trend[-1] + memory_slope * 10
        
        # Take action if predictions indicate resource pressure
        if cpu_prediction > 85 or memory_prediction > 85:
            logger.info(f"Predictive scaling triggered - CPU: {cpu_prediction:.1f}%, Memory: {memory_prediction:.1f}%")
            
            # Create a synthetic metrics object for optimization
            current_metrics = self.metrics_history[-1] if self.metrics_history else None
            if current_metrics:
                # Adjust current metrics with predictions
                predicted_metrics = ResourceMetrics(
                    timestamp=datetime.utcnow(),
                    cpu_percent=min(100, cpu_prediction),
                    memory_percent=min(100, memory_prediction),
                    cpu_count_logical=current_metrics.cpu_count_logical,
                    cpu_count_physical=current_metrics.cpu_count_physical,
                    cpu_freq_current=current_metrics.cpu_freq_current,
                    memory_total_gb=current_metrics.memory_total_gb,
                    memory_used_gb=current_metrics.memory_used_gb,
                    memory_available_gb=current_metrics.memory_available_gb,
                    swap_total_gb=current_metrics.swap_total_gb,
                    swap_used_gb=current_metrics.swap_used_gb,
                    disk_read_mb_s=current_metrics.disk_read_mb_s,
                    disk_write_mb_s=current_metrics.disk_write_mb_s,
                    disk_usage_percent=current_metrics.disk_usage_percent,
                    disk_iops=current_metrics.disk_iops,
                    network_sent_mb_s=current_metrics.network_sent_mb_s,
                    network_recv_mb_s=current_metrics.network_recv_mb_s,
                    network_connections=current_metrics.network_connections,
                    process_count=current_metrics.process_count,
                    thread_count=current_metrics.thread_count,
                    file_descriptor_count=current_metrics.file_descriptor_count
                )
                
                await self._optimize_resources(predicted_metrics)
    
    async def _analyze_performance(self):
        """Analyze performance metrics and record insights"""
        if len(self.metrics_history) < 10:
            return
        
        recent_metrics = list(self.metrics_history)[-10:]
        
        # Calculate performance indicators
        avg_cpu = np.mean([m.cpu_percent for m in recent_metrics])
        avg_memory = np.mean([m.memory_percent for m in recent_metrics])
        avg_disk_io = np.mean([m.disk_read_mb_s + m.disk_write_mb_s for m in recent_metrics])
        avg_network_io = np.mean([m.network_sent_mb_s + m.network_recv_mb_s for m in recent_metrics])
        
        # CPU efficiency
        cpu_efficiency = self.current_allocation.cpu_workers / max(avg_cpu / 100, 0.1)
        
        # Memory efficiency
        memory_efficiency = (self.current_allocation.memory_limit_mb / 1024) / max(avg_memory / 100 * self.system_limits['memory']['total_gb'], 0.1)
        
        performance_data = {
            'timestamp': datetime.utcnow(),
            'avg_cpu_percent': avg_cpu,
            'avg_memory_percent': avg_memory,
            'avg_disk_io_mb_s': avg_disk_io,
            'avg_network_io_mb_s': avg_network_io,
            'cpu_efficiency': cpu_efficiency,
            'memory_efficiency': memory_efficiency,
            'current_allocation': self.current_allocation
        }
        
        self.performance_history.append(performance_data)
    
    def register_workload(self, workload_id: str, profile: WorkloadProfile):
        """Register a workload profile for optimization"""
        self.workload_profiles[workload_id] = profile
        logger.info(f"Registered workload: {workload_id} ({profile.workload_type.value})")
    
    def register_resource_change_callback(self, callback: Callable):
        """Register callback for resource changes"""
        self._resource_change_callbacks.append(callback)
    
    def get_current_allocation(self) -> ResourceAllocation:
        """Get current resource allocation"""
        return self.current_allocation
    
    def get_system_limits(self) -> Dict[str, Any]:
        """Get detected system limits"""
        return self.system_limits
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        if not self.metrics_history:
            return {'status': 'no_data'}
        
        current_metrics = self.metrics_history[-1]
        recent_metrics = list(self.metrics_history)[-10:] if len(self.metrics_history) >= 10 else list(self.metrics_history)
        
        stats = {
            'current': {
                'timestamp': current_metrics.timestamp.isoformat(),
                'cpu_percent': current_metrics.cpu_percent,
                'memory_percent': current_metrics.memory_percent,
                'disk_io_mb_s': current_metrics.disk_read_mb_s + current_metrics.disk_write_mb_s,
                'network_io_mb_s': current_metrics.network_sent_mb_s + current_metrics.network_recv_mb_s,
                'pressure_level': current_metrics.overall_pressure_level.value
            },
            'recent_averages': {
                'cpu_percent': np.mean([m.cpu_percent for m in recent_metrics]),
                'memory_percent': np.mean([m.memory_percent for m in recent_metrics]),
                'disk_io_mb_s': np.mean([m.disk_read_mb_s + m.disk_write_mb_s for m in recent_metrics]),
                'network_io_mb_s': np.mean([m.network_sent_mb_s + m.network_recv_mb_s for m in recent_metrics])
            },
            'system_limits': self.system_limits,
            'current_allocation': {
                'cpu_workers': self.current_allocation.cpu_workers,
                'memory_limit_mb': self.current_allocation.memory_limit_mb,
                'io_concurrent_limit': self.current_allocation.io_concurrent_limit,
                'network_concurrent_limit': self.current_allocation.network_concurrent_limit,
                'batch_size': self.current_allocation.batch_size
            },
            'optimization_count': len(self.optimization_history),
            'workload_profiles_count': len(self.workload_profiles)
        }
        
        # Add performance trends if available
        if len(self.cpu_trend) > 10:
            cpu_trend_array = np.array(self.cpu_trend)
            memory_trend_array = np.array(self.memory_trend)
            
            stats['trends'] = {
                'cpu_slope': float(np.polyfit(range(len(cpu_trend_array)), cpu_trend_array, 1)[0]),
                'memory_slope': float(np.polyfit(range(len(memory_trend_array)), memory_trend_array, 1)[0]),
                'cpu_volatility': float(np.std(cpu_trend_array)),
                'memory_volatility': float(np.std(memory_trend_array))
            }
        
        return stats
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get resource optimization history"""
        return [
            {
                'timestamp': opt['timestamp'].isoformat(),
                'pressure_level': opt['metrics'].overall_pressure_level.value,
                'cpu_change': opt['new_allocation'].cpu_workers - opt['old_allocation'].cpu_workers,
                'memory_change_mb': opt['new_allocation'].memory_limit_mb - opt['old_allocation'].memory_limit_mb,
                'optimization_time_ms': opt['optimization_time_ms'],
                'changes_made': opt['changes_made']
            }
            for opt in self.optimization_history
        ]
    
    async def force_optimization(self):
        """Force immediate resource optimization"""
        if self.metrics_history:
            current_metrics = self.metrics_history[-1]
            await self._optimize_resources(current_metrics)
        else:
            logger.warning("No metrics available for forced optimization")
    
    async def shutdown(self):
        """Shutdown the resource manager"""
        self._monitoring_active = False
        self._shutdown_event.set()
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        # Clear data structures
        self.metrics_history.clear()
        self.performance_history.clear()
        self.optimization_history.clear()
        self.workload_profiles.clear()
        self.resource_allocations.clear()
        self._resource_change_callbacks.clear()
        
        logger.info("Dynamic resource manager shutdown complete")


# Global resource manager instance
_resource_manager: Optional[DynamicResourceManager] = None


async def get_resource_manager(
    monitoring_interval_s: int = 30,
    enable_auto_scaling: bool = True,
    enable_predictive_scaling: bool = True
) -> DynamicResourceManager:
    """Get or create global resource manager"""
    global _resource_manager
    if _resource_manager is None:
        _resource_manager = DynamicResourceManager(
            monitoring_interval_s=monitoring_interval_s,
            enable_auto_scaling=enable_auto_scaling,
            enable_predictive_scaling=enable_predictive_scaling
        )
        await _resource_manager.initialize()
    return _resource_manager