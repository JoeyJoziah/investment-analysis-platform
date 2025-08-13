"""
Service Health Management and Monitoring System
Comprehensive health monitoring with dependency tracking, bulkhead patterns, and automatic recovery
"""

import asyncio
import time
import json
import psutil
import socket
import httpx
from typing import Any, Dict, List, Optional, Callable, Set, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
from datetime import datetime, timedelta
from collections import deque, defaultdict
import threading
import logging
from pathlib import Path
import redis
import aiofiles
from contextlib import asynccontextmanager

from .enhanced_error_handling import with_error_handling, ErrorSeverity
from .advanced_circuit_breaker import EnhancedCircuitBreaker, AdaptiveThresholds
from .exceptions import *

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Service health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"  
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class DependencyType(Enum):
    """Types of service dependencies"""
    DATABASE = "database"
    CACHE = "cache"
    EXTERNAL_API = "external_api"
    INTERNAL_SERVICE = "internal_service"
    MESSAGE_QUEUE = "message_queue"
    FILE_SYSTEM = "file_system"
    NETWORK = "network"


class RecoveryAction(Enum):
    """Automated recovery actions"""
    RESTART_SERVICE = "restart_service"
    CLEAR_CACHE = "clear_cache"
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    CIRCUIT_BREAK = "circuit_break"
    FALLBACK_MODE = "fallback_mode"
    ALERT_OPERATOR = "alert_operator"
    GRACEFUL_SHUTDOWN = "graceful_shutdown"


@dataclass
class HealthMetrics:
    """Comprehensive health metrics for a service"""
    service_name: str
    status: HealthStatus
    timestamp: datetime
    response_time_ms: float
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    active_connections: int
    error_rate: float
    throughput_per_second: float
    uptime_seconds: float
    dependency_status: Dict[str, HealthStatus]
    custom_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DependencyConfig:
    """Configuration for service dependency monitoring"""
    name: str
    dependency_type: DependencyType
    endpoint: str
    timeout_seconds: float
    check_interval_seconds: int
    critical: bool
    health_check_func: Optional[Callable] = None
    recovery_actions: List[RecoveryAction] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ServiceConfig:
    """Configuration for service health monitoring"""
    name: str
    check_interval_seconds: int
    dependencies: List[DependencyConfig]
    health_check_func: Optional[Callable] = None
    recovery_actions: List[RecoveryAction] = field(default_factory=list)
    resource_thresholds: Dict[str, float] = field(default_factory=dict)
    custom_checks: List[Callable] = field(default_factory=list)
    bulkhead_enabled: bool = True
    circuit_breaker_config: Optional[Dict] = None


class ResourceMonitor:
    """System resource monitoring"""
    
    def __init__(self, check_interval: int = 30):
        self.check_interval = check_interval
        self.metrics_history: deque = deque(maxlen=1440)  # 12 hours at 30s intervals
        self.thresholds = {
            'cpu_warning': 70.0,
            'cpu_critical': 85.0,
            'memory_warning': 75.0,
            'memory_critical': 90.0,
            'disk_warning': 80.0,
            'disk_critical': 95.0
        }
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
    
    async def start_monitoring(self):
        """Start resource monitoring"""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Resource monitoring started")
    
    async def stop_monitoring(self):
        """Stop resource monitoring"""
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Resource monitoring stopped")
    
    async def _monitor_loop(self):
        """Main monitoring loop"""
        while self._monitoring:
            try:
                metrics = await self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Check for threshold violations
                await self._check_thresholds(metrics)
                
                await asyncio.sleep(self.check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def _collect_metrics(self) -> Dict[str, Any]:
        """Collect system resource metrics"""
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
        
        # Memory metrics
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_io = psutil.disk_io_counters()
        
        # Network metrics
        net_io = psutil.net_io_counters()
        
        # Process metrics
        process = psutil.Process()
        process_memory = process.memory_info()
        
        return {
            'timestamp': datetime.now(),
            'cpu': {
                'percent': cpu_percent,
                'count': cpu_count,
                'load_avg_1m': load_avg[0],
                'load_avg_5m': load_avg[1],
                'load_avg_15m': load_avg[2]
            },
            'memory': {
                'total_gb': memory.total / (1024**3),
                'available_gb': memory.available / (1024**3),
                'used_gb': memory.used / (1024**3),
                'percent': memory.percent,
                'swap_percent': swap.percent
            },
            'disk': {
                'total_gb': disk.total / (1024**3),
                'free_gb': disk.free / (1024**3),
                'used_gb': disk.used / (1024**3),
                'percent': (disk.used / disk.total) * 100,
                'read_mb_s': disk_io.read_bytes / (1024**2) if disk_io else 0,
                'write_mb_s': disk_io.write_bytes / (1024**2) if disk_io else 0
            },
            'network': {
                'bytes_sent_mb': net_io.bytes_sent / (1024**2) if net_io else 0,
                'bytes_recv_mb': net_io.bytes_recv / (1024**2) if net_io else 0,
                'packets_sent': net_io.packets_sent if net_io else 0,
                'packets_recv': net_io.packets_recv if net_io else 0
            },
            'process': {
                'memory_mb': process_memory.rss / (1024**2),
                'memory_percent': process.memory_percent(),
                'cpu_percent': process.cpu_percent(),
                'num_threads': process.num_threads(),
                'num_fds': process.num_fds() if hasattr(process, 'num_fds') else 0
            }
        }
    
    async def _check_thresholds(self, metrics: Dict[str, Any]):
        """Check resource thresholds and log warnings"""
        cpu_percent = metrics['cpu']['percent']
        memory_percent = metrics['memory']['percent']
        disk_percent = metrics['disk']['percent']
        
        if cpu_percent > self.thresholds['cpu_critical']:
            logger.critical(f"CPU usage critical: {cpu_percent:.1f}%")
        elif cpu_percent > self.thresholds['cpu_warning']:
            logger.warning(f"CPU usage high: {cpu_percent:.1f}%")
        
        if memory_percent > self.thresholds['memory_critical']:
            logger.critical(f"Memory usage critical: {memory_percent:.1f}%")
        elif memory_percent > self.thresholds['memory_warning']:
            logger.warning(f"Memory usage high: {memory_percent:.1f}%")
        
        if disk_percent > self.thresholds['disk_critical']:
            logger.critical(f"Disk usage critical: {disk_percent:.1f}%")
        elif disk_percent > self.thresholds['disk_warning']:
            logger.warning(f"Disk usage high: {disk_percent:.1f}%")
    
    def get_current_metrics(self) -> Optional[Dict[str, Any]]:
        """Get the most recent metrics"""
        if not self.metrics_history:
            return None
        return self.metrics_history[-1]
    
    def get_average_metrics(self, minutes: int = 5) -> Optional[Dict[str, float]]:
        """Get average metrics over specified time period"""
        if not self.metrics_history:
            return None
        
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent_metrics = [
            m for m in self.metrics_history
            if m['timestamp'] > cutoff_time
        ]
        
        if not recent_metrics:
            return None
        
        # Calculate averages
        avg_metrics = {
            'cpu_percent': sum(m['cpu']['percent'] for m in recent_metrics) / len(recent_metrics),
            'memory_percent': sum(m['memory']['percent'] for m in recent_metrics) / len(recent_metrics),
            'disk_percent': sum(m['disk']['percent'] for m in recent_metrics) / len(recent_metrics),
            'load_avg_1m': sum(m['cpu']['load_avg_1m'] for m in recent_metrics) / len(recent_metrics),
            'memory_used_gb': sum(m['memory']['used_gb'] for m in recent_metrics) / len(recent_metrics),
        }
        
        return avg_metrics


class DependencyHealthChecker:
    """Health checking for service dependencies"""
    
    def __init__(self, dependency_config: DependencyConfig):
        self.config = dependency_config
        self.last_check_time: Optional[datetime] = None
        self.last_status: HealthStatus = HealthStatus.UNKNOWN
        self.response_times: deque = deque(maxlen=100)
        self.error_count = 0
        self.consecutive_failures = 0
        self.uptime_start = datetime.now()
        
        # Circuit breaker for dependency
        if self.config.critical:
            self.circuit_breaker = EnhancedCircuitBreaker(
                name=f"dependency_{self.config.name}",
                base_thresholds=AdaptiveThresholds(
                    failure_threshold=3,
                    recovery_timeout=30,
                    success_threshold=2,
                    rate_limit_threshold=2,
                    timeout_threshold=3,
                    error_rate_threshold=0.5
                )
            )
        else:
            self.circuit_breaker = None
    
    async def check_health(self) -> HealthStatus:
        """Perform health check for this dependency"""
        start_time = time.time()
        
        try:
            if self.config.health_check_func:
                # Use custom health check function
                if self.circuit_breaker:
                    result = await self.circuit_breaker.call(self.config.health_check_func)
                else:
                    result = await self._safe_call(self.config.health_check_func)
                
                status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
            else:
                # Use default health checks based on dependency type
                status = await self._default_health_check()
            
            # Record successful check
            response_time = (time.time() - start_time) * 1000
            self.response_times.append(response_time)
            self.consecutive_failures = 0
            
            if status == HealthStatus.HEALTHY and self.last_status != HealthStatus.HEALTHY:
                logger.info(f"Dependency {self.config.name} recovered")
            
            self.last_status = status
            self.last_check_time = datetime.now()
            
            return status
            
        except Exception as e:
            # Record failed check
            response_time = (time.time() - start_time) * 1000
            self.response_times.append(response_time)
            self.error_count += 1
            self.consecutive_failures += 1
            
            logger.error(f"Health check failed for {self.config.name}: {e}")
            
            # Determine status based on failure pattern
            if self.consecutive_failures >= 3:
                status = HealthStatus.CRITICAL if self.config.critical else HealthStatus.UNHEALTHY
            else:
                status = HealthStatus.DEGRADED
            
            self.last_status = status
            self.last_check_time = datetime.now()
            
            return status
    
    async def _safe_call(self, func: Callable) -> Any:
        """Safely call function (sync or async)"""
        if asyncio.iscoroutinefunction(func):
            return await func()
        else:
            return func()
    
    async def _default_health_check(self) -> HealthStatus:
        """Default health checks based on dependency type"""
        dep_type = self.config.dependency_type
        endpoint = self.config.endpoint
        timeout = self.config.timeout_seconds
        
        if dep_type == DependencyType.DATABASE:
            return await self._check_database_health(endpoint, timeout)
        elif dep_type == DependencyType.CACHE:
            return await self._check_cache_health(endpoint, timeout)
        elif dep_type == DependencyType.EXTERNAL_API:
            return await self._check_http_health(endpoint, timeout)
        elif dep_type == DependencyType.INTERNAL_SERVICE:
            return await self._check_service_health(endpoint, timeout)
        elif dep_type == DependencyType.MESSAGE_QUEUE:
            return await self._check_message_queue_health(endpoint, timeout)
        elif dep_type == DependencyType.FILE_SYSTEM:
            return await self._check_file_system_health(endpoint)
        elif dep_type == DependencyType.NETWORK:
            return await self._check_network_health(endpoint, timeout)
        else:
            return HealthStatus.UNKNOWN
    
    async def _check_database_health(self, connection_string: str, timeout: float) -> HealthStatus:
        """Check database health"""
        try:
            # This would integrate with your actual database connection
            # For now, we'll simulate a basic connection check
            import asyncpg
            
            conn = await asyncio.wait_for(
                asyncpg.connect(connection_string),
                timeout=timeout
            )
            
            # Simple query to verify functionality
            result = await conn.fetchval("SELECT 1")
            await conn.close()
            
            return HealthStatus.HEALTHY if result == 1 else HealthStatus.UNHEALTHY
            
        except asyncio.TimeoutError:
            return HealthStatus.DEGRADED
        except Exception:
            return HealthStatus.UNHEALTHY
    
    async def _check_cache_health(self, connection_string: str, timeout: float) -> HealthStatus:
        """Check cache (Redis) health"""
        try:
            redis_client = redis.from_url(connection_string, socket_timeout=timeout)
            
            # Simple ping test
            result = redis_client.ping()
            redis_client.close()
            
            return HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
            
        except redis.TimeoutError:
            return HealthStatus.DEGRADED
        except Exception:
            return HealthStatus.UNHEALTHY
    
    async def _check_http_health(self, url: str, timeout: float) -> HealthStatus:
        """Check HTTP endpoint health"""
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(url)
                
                if response.status_code == 200:
                    return HealthStatus.HEALTHY
                elif 400 <= response.status_code < 500:
                    return HealthStatus.DEGRADED
                else:
                    return HealthStatus.UNHEALTHY
                    
        except httpx.TimeoutException:
            return HealthStatus.DEGRADED
        except Exception:
            return HealthStatus.UNHEALTHY
    
    async def _check_service_health(self, endpoint: str, timeout: float) -> HealthStatus:
        """Check internal service health"""
        # For internal services, we might check a health endpoint
        health_url = f"{endpoint}/health" if not endpoint.endswith('/health') else endpoint
        return await self._check_http_health(health_url, timeout)
    
    async def _check_message_queue_health(self, connection_string: str, timeout: float) -> HealthStatus:
        """Check message queue health"""
        try:
            # This would integrate with your message queue (Kafka, RabbitMQ, etc.)
            # For now, we'll simulate a basic connection check
            
            # Parse connection details and attempt connection
            # This is a placeholder implementation
            await asyncio.sleep(0.1)  # Simulate connection check
            return HealthStatus.HEALTHY
            
        except Exception:
            return HealthStatus.UNHEALTHY
    
    async def _check_file_system_health(self, path: str) -> HealthStatus:
        """Check file system health"""
        try:
            path_obj = Path(path)
            
            # Check if path exists and is writable
            if not path_obj.exists():
                path_obj.mkdir(parents=True, exist_ok=True)
            
            # Test write access
            test_file = path_obj / f".health_check_{int(time.time())}"
            test_file.write_text("health_check")
            test_file.unlink()
            
            return HealthStatus.HEALTHY
            
        except Exception:
            return HealthStatus.UNHEALTHY
    
    async def _check_network_health(self, host: str, timeout: float) -> HealthStatus:
        """Check network connectivity"""
        try:
            # Parse host and port
            if ':' in host:
                hostname, port_str = host.rsplit(':', 1)
                port = int(port_str)
            else:
                hostname = host
                port = 80
            
            # Test TCP connection
            _, writer = await asyncio.wait_for(
                asyncio.open_connection(hostname, port),
                timeout=timeout
            )
            writer.close()
            await writer.wait_closed()
            
            return HealthStatus.HEALTHY
            
        except asyncio.TimeoutError:
            return HealthStatus.DEGRADED
        except Exception:
            return HealthStatus.UNHEALTHY
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get dependency health metrics"""
        avg_response_time = (
            sum(self.response_times) / len(self.response_times)
            if self.response_times else 0
        )
        
        uptime = (datetime.now() - self.uptime_start).total_seconds()
        
        return {
            'name': self.config.name,
            'type': self.config.dependency_type.value,
            'status': self.last_status.value,
            'critical': self.config.critical,
            'last_check': self.last_check_time.isoformat() if self.last_check_time else None,
            'avg_response_time_ms': round(avg_response_time, 2),
            'error_count': self.error_count,
            'consecutive_failures': self.consecutive_failures,
            'uptime_seconds': round(uptime, 1),
            'circuit_breaker': (
                self.circuit_breaker.get_comprehensive_metrics()
                if self.circuit_breaker else None
            )
        }


class ServiceHealthManager:
    """
    Comprehensive service health management with dependency tracking,
    bulkhead patterns, and automatic recovery
    """
    
    def __init__(self, service_config: ServiceConfig):
        self.config = service_config
        self.resource_monitor = ResourceMonitor()
        
        # Health checkers for dependencies
        self.dependency_checkers = {
            dep.name: DependencyHealthChecker(dep)
            for dep in service_config.dependencies
        }
        
        # Service state
        self.service_status = HealthStatus.UNKNOWN
        self.service_start_time = datetime.now()
        self.last_health_check = None
        
        # Metrics collection
        self.health_history: deque = deque(maxlen=1000)
        self.recovery_actions_taken: List[Dict] = []
        
        # Bulkhead isolation
        self.isolation_groups: Dict[str, Set[str]] = {}
        self.isolated_services: Set[str] = set()
        
        # Circuit breakers for critical paths
        self.circuit_breakers: Dict[str, EnhancedCircuitBreaker] = {}
        
        # Monitoring control
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
        
        # Thread safety
        self._lock = asyncio.Lock()
    
    async def start_monitoring(self):
        """Start comprehensive health monitoring"""
        if self._monitoring:
            logger.warning("Health monitoring already running")
            return
        
        self._monitoring = True
        
        # Start resource monitoring
        await self.resource_monitor.start_monitoring()
        
        # Start health checking loop
        self._monitor_task = asyncio.create_task(self._health_monitor_loop())
        
        logger.info(f"Started health monitoring for service: {self.config.name}")
    
    async def stop_monitoring(self):
        """Stop health monitoring gracefully"""
        if not self._monitoring:
            return
        
        self._monitoring = False
        
        # Stop monitoring task
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        # Stop resource monitoring
        await self.resource_monitor.stop_monitoring()
        
        logger.info(f"Stopped health monitoring for service: {self.config.name}")
    
    async def _health_monitor_loop(self):
        """Main health monitoring loop"""
        while self._monitoring:
            try:
                # Perform comprehensive health check
                health_metrics = await self._perform_health_check()
                
                # Store metrics
                async with self._lock:
                    self.health_history.append(health_metrics)
                
                # Check for recovery actions needed
                await self._evaluate_recovery_actions(health_metrics)
                
                # Save health status
                await self._save_health_status(health_metrics)
                
                await asyncio.sleep(self.config.check_interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(self.config.check_interval_seconds)
    
    async def _perform_health_check(self) -> HealthMetrics:
        """Perform comprehensive health check"""
        check_start_time = time.time()
        
        # Get resource metrics
        resource_metrics = self.resource_monitor.get_current_metrics()
        
        # Check all dependencies
        dependency_statuses = {}
        dependency_check_tasks = []
        
        for name, checker in self.dependency_checkers.items():
            task = asyncio.create_task(checker.check_health())
            dependency_check_tasks.append((name, task))
        
        # Wait for all dependency checks with timeout
        for name, task in dependency_check_tasks:
            try:
                status = await asyncio.wait_for(task, timeout=30)
                dependency_statuses[name] = status
            except asyncio.TimeoutError:
                dependency_statuses[name] = HealthStatus.DEGRADED
                logger.warning(f"Dependency check timeout: {name}")
            except Exception as e:
                dependency_statuses[name] = HealthStatus.UNHEALTHY
                logger.error(f"Dependency check error for {name}: {e}")
        
        # Run custom health checks
        custom_check_results = []
        for check_func in self.config.custom_checks:
            try:
                result = await self._safe_call(check_func)
                custom_check_results.append(result)
            except Exception as e:
                logger.error(f"Custom health check failed: {e}")
                custom_check_results.append(False)
        
        # Determine overall service status
        overall_status = await self._calculate_overall_status(
            dependency_statuses,
            custom_check_results,
            resource_metrics
        )
        
        # Calculate metrics
        response_time = (time.time() - check_start_time) * 1000
        uptime = (datetime.now() - self.service_start_time).total_seconds()
        
        # Get resource usage
        cpu_usage = resource_metrics['cpu']['percent'] if resource_metrics else 0
        memory_usage = resource_metrics['memory']['percent'] if resource_metrics else 0
        disk_usage = resource_metrics['disk']['percent'] if resource_metrics else 0
        
        network_io = {
            'bytes_sent_mb': resource_metrics['network']['bytes_sent_mb'] if resource_metrics else 0,
            'bytes_recv_mb': resource_metrics['network']['bytes_recv_mb'] if resource_metrics else 0
        }
        
        # Calculate error rate and throughput (simplified)
        error_rate = self._calculate_error_rate()
        throughput = self._calculate_throughput()
        
        # Get active connections (simplified)
        active_connections = resource_metrics['process']['num_fds'] if resource_metrics else 0
        
        health_metrics = HealthMetrics(
            service_name=self.config.name,
            status=overall_status,
            timestamp=datetime.now(),
            response_time_ms=response_time,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            disk_usage=disk_usage,
            network_io=network_io,
            active_connections=active_connections,
            error_rate=error_rate,
            throughput_per_second=throughput,
            uptime_seconds=uptime,
            dependency_status=dependency_statuses,
            custom_metrics={
                'custom_checks_passed': sum(custom_check_results),
                'total_custom_checks': len(custom_check_results),
                'resource_score': self._calculate_resource_score(resource_metrics)
            }
        )
        
        self.service_status = overall_status
        self.last_health_check = datetime.now()
        
        return health_metrics
    
    async def _safe_call(self, func: Callable) -> Any:
        """Safely call function (sync or async)"""
        if asyncio.iscoroutinefunction(func):
            return await func()
        else:
            return func()
    
    async def _calculate_overall_status(
        self,
        dependency_statuses: Dict[str, HealthStatus],
        custom_check_results: List[bool],
        resource_metrics: Optional[Dict[str, Any]]
    ) -> HealthStatus:
        """Calculate overall service health status"""
        
        # Check critical dependencies first
        critical_deps = [
            dep for dep in self.config.dependencies if dep.critical
        ]
        
        for dep in critical_deps:
            status = dependency_statuses.get(dep.name, HealthStatus.UNKNOWN)
            if status in [HealthStatus.CRITICAL, HealthStatus.UNHEALTHY]:
                return HealthStatus.CRITICAL
        
        # Check resource thresholds
        if resource_metrics:
            thresholds = self.config.resource_thresholds
            
            cpu_critical = thresholds.get('cpu_critical', 90)
            memory_critical = thresholds.get('memory_critical', 95)
            disk_critical = thresholds.get('disk_critical', 95)
            
            if (resource_metrics['cpu']['percent'] > cpu_critical or
                resource_metrics['memory']['percent'] > memory_critical or
                resource_metrics['disk']['percent'] > disk_critical):
                return HealthStatus.CRITICAL
        
        # Check custom health checks
        if custom_check_results and not all(custom_check_results):
            failed_checks = len(custom_check_results) - sum(custom_check_results)
            if failed_checks > len(custom_check_results) / 2:
                return HealthStatus.UNHEALTHY
        
        # Check non-critical dependencies
        unhealthy_deps = [
            name for name, status in dependency_statuses.items()
            if status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]
        ]
        
        if len(unhealthy_deps) > len(dependency_statuses) / 2:
            return HealthStatus.DEGRADED
        elif unhealthy_deps:
            return HealthStatus.DEGRADED
        
        return HealthStatus.HEALTHY
    
    def _calculate_error_rate(self) -> float:
        """Calculate service error rate"""
        # This would integrate with your error tracking system
        # For now, return a placeholder calculation
        recent_metrics = list(self.health_history)[-10:]
        if not recent_metrics:
            return 0.0
        
        error_count = sum(1 for m in recent_metrics if m.status != HealthStatus.HEALTHY)
        return error_count / len(recent_metrics)
    
    def _calculate_throughput(self) -> float:
        """Calculate service throughput"""
        # This would integrate with your metrics system
        # For now, return a placeholder calculation
        return 0.0
    
    def _calculate_resource_score(self, resource_metrics: Optional[Dict[str, Any]]) -> float:
        """Calculate resource utilization score (0-1, higher is better)"""
        if not resource_metrics:
            return 1.0
        
        # Inverse scoring - lower utilization is better
        cpu_score = max(0, 1 - resource_metrics['cpu']['percent'] / 100)
        memory_score = max(0, 1 - resource_metrics['memory']['percent'] / 100)
        disk_score = max(0, 1 - resource_metrics['disk']['percent'] / 100)
        
        return (cpu_score + memory_score + disk_score) / 3
    
    async def _evaluate_recovery_actions(self, health_metrics: HealthMetrics):
        """Evaluate and execute recovery actions if needed"""
        if health_metrics.status in [HealthStatus.CRITICAL, HealthStatus.UNHEALTHY]:
            
            # Determine appropriate recovery actions
            recovery_actions = self._determine_recovery_actions(health_metrics)
            
            for action in recovery_actions:
                try:
                    success = await self._execute_recovery_action(action, health_metrics)
                    
                    self.recovery_actions_taken.append({
                        'timestamp': datetime.now().isoformat(),
                        'action': action.value,
                        'trigger_status': health_metrics.status.value,
                        'success': success
                    })
                    
                except Exception as e:
                    logger.error(f"Recovery action {action.value} failed: {e}")
    
    def _determine_recovery_actions(self, health_metrics: HealthMetrics) -> List[RecoveryAction]:
        """Determine appropriate recovery actions based on health status"""
        actions = []
        
        # High CPU usage
        if health_metrics.cpu_usage > 85:
            actions.append(RecoveryAction.SCALE_UP)
        
        # High memory usage
        if health_metrics.memory_usage > 90:
            actions.extend([RecoveryAction.CLEAR_CACHE, RecoveryAction.SCALE_UP])
        
        # Dependency failures
        critical_deps_failed = [
            name for name, status in health_metrics.dependency_status.items()
            if status == HealthStatus.CRITICAL and 
            any(dep.critical for dep in self.config.dependencies if dep.name == name)
        ]
        
        if critical_deps_failed:
            actions.extend([RecoveryAction.CIRCUIT_BREAK, RecoveryAction.FALLBACK_MODE])
        
        # High error rate
        if health_metrics.error_rate > 0.5:
            actions.append(RecoveryAction.RESTART_SERVICE)
        
        # Critical status
        if health_metrics.status == HealthStatus.CRITICAL:
            actions.append(RecoveryAction.ALERT_OPERATOR)
        
        return list(set(actions))  # Remove duplicates
    
    async def _execute_recovery_action(
        self,
        action: RecoveryAction,
        health_metrics: HealthMetrics
    ) -> bool:
        """Execute a specific recovery action"""
        logger.warning(f"Executing recovery action: {action.value}")
        
        try:
            if action == RecoveryAction.CLEAR_CACHE:
                return await self._clear_cache()
            
            elif action == RecoveryAction.CIRCUIT_BREAK:
                return await self._activate_circuit_breakers()
            
            elif action == RecoveryAction.FALLBACK_MODE:
                return await self._activate_fallback_mode()
            
            elif action == RecoveryAction.SCALE_UP:
                return await self._request_scale_up()
            
            elif action == RecoveryAction.RESTART_SERVICE:
                return await self._request_service_restart()
            
            elif action == RecoveryAction.ALERT_OPERATOR:
                return await self._send_operator_alert(health_metrics)
            
            elif action == RecoveryAction.GRACEFUL_SHUTDOWN:
                return await self._initiate_graceful_shutdown()
            
            else:
                logger.warning(f"Unknown recovery action: {action.value}")
                return False
                
        except Exception as e:
            logger.error(f"Recovery action {action.value} execution failed: {e}")
            return False
    
    async def _clear_cache(self) -> bool:
        """Clear application caches"""
        # This would integrate with your caching system
        logger.info("Clearing application caches")
        return True
    
    async def _activate_circuit_breakers(self) -> bool:
        """Activate circuit breakers for failing dependencies"""
        for name, checker in self.dependency_checkers.items():
            if checker.circuit_breaker and checker.last_status == HealthStatus.CRITICAL:
                checker.circuit_breaker.force_open("Health check failure")
        
        logger.info("Activated circuit breakers for failing dependencies")
        return True
    
    async def _activate_fallback_mode(self) -> bool:
        """Activate service fallback mode"""
        logger.info("Activating service fallback mode")
        # This would reduce service functionality to essential features only
        return True
    
    async def _request_scale_up(self) -> bool:
        """Request service scaling up"""
        logger.info("Requesting service scale up")
        # This would integrate with your orchestration system (Kubernetes, etc.)
        return True
    
    async def _request_service_restart(self) -> bool:
        """Request service restart"""
        logger.critical("Requesting service restart due to health issues")
        # This would integrate with your process manager or orchestration system
        return True
    
    async def _send_operator_alert(self, health_metrics: HealthMetrics) -> bool:
        """Send alert to operations team"""
        alert_data = {
            'service': self.config.name,
            'status': health_metrics.status.value,
            'timestamp': health_metrics.timestamp.isoformat(),
            'cpu_usage': health_metrics.cpu_usage,
            'memory_usage': health_metrics.memory_usage,
            'error_rate': health_metrics.error_rate,
            'failed_dependencies': [
                name for name, status in health_metrics.dependency_status.items()
                if status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]
            ]
        }
        
        logger.critical(f"OPERATOR ALERT: {json.dumps(alert_data, indent=2)}")
        
        # This would integrate with your alerting system (PagerDuty, Slack, etc.)
        return True
    
    async def _initiate_graceful_shutdown(self) -> bool:
        """Initiate graceful service shutdown"""
        logger.critical("Initiating graceful service shutdown")
        # This would gracefully shut down the service
        return True
    
    async def _save_health_status(self, health_metrics: HealthMetrics):
        """Save health status to persistent storage"""
        try:
            status_dir = Path("data/health_status")
            status_dir.mkdir(parents=True, exist_ok=True)
            
            status_file = status_dir / f"{self.config.name}_status.json"
            
            status_data = {
                'service_name': health_metrics.service_name,
                'status': health_metrics.status.value,
                'timestamp': health_metrics.timestamp.isoformat(),
                'metrics': asdict(health_metrics),
                'dependency_metrics': {
                    name: checker.get_metrics()
                    for name, checker in self.dependency_checkers.items()
                },
                'recent_recovery_actions': self.recovery_actions_taken[-10:]
            }
            
            async with aiofiles.open(status_file, 'w') as f:
                await f.write(json.dumps(status_data, indent=2, default=str))
                
        except Exception as e:
            logger.error(f"Failed to save health status: {e}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current comprehensive health status"""
        current_metrics = self.health_history[-1] if self.health_history else None
        
        return {
            'service_name': self.config.name,
            'overall_status': self.service_status.value,
            'last_check': self.last_health_check.isoformat() if self.last_health_check else None,
            'uptime_seconds': (datetime.now() - self.service_start_time).total_seconds(),
            'current_metrics': asdict(current_metrics) if current_metrics else None,
            'dependency_health': {
                name: checker.get_metrics()
                for name, checker in self.dependency_checkers.items()
            },
            'resource_metrics': self.resource_monitor.get_current_metrics(),
            'recent_recovery_actions': self.recovery_actions_taken[-5:],
            'monitoring_active': self._monitoring
        }
    
    def get_health_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Get health trends over specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        relevant_metrics = [
            m for m in self.health_history
            if m.timestamp > cutoff_time
        ]
        
        if not relevant_metrics:
            return {'message': f'No health data for the last {hours} hours'}
        
        # Calculate trends
        status_distribution = defaultdict(int)
        avg_response_time = 0
        avg_cpu_usage = 0
        avg_memory_usage = 0
        avg_error_rate = 0
        
        for metrics in relevant_metrics:
            status_distribution[metrics.status.value] += 1
            avg_response_time += metrics.response_time_ms
            avg_cpu_usage += metrics.cpu_usage
            avg_memory_usage += metrics.memory_usage
            avg_error_rate += metrics.error_rate
        
        count = len(relevant_metrics)
        
        return {
            'time_period_hours': hours,
            'total_checks': count,
            'status_distribution': dict(status_distribution),
            'averages': {
                'response_time_ms': round(avg_response_time / count, 2),
                'cpu_usage_percent': round(avg_cpu_usage / count, 2),
                'memory_usage_percent': round(avg_memory_usage / count, 2),
                'error_rate': round(avg_error_rate / count, 3)
            },
            'health_score': round(
                status_distribution[HealthStatus.HEALTHY.value] / count, 3
            ),
            'recovery_actions_count': len([
                action for action in self.recovery_actions_taken
                if datetime.fromisoformat(action['timestamp']) > cutoff_time
            ])
        }


# Bulkhead Pattern Implementation
class BulkheadManager:
    """Implements bulkhead isolation patterns to prevent cascading failures"""
    
    def __init__(self):
        self.isolation_groups: Dict[str, Dict] = {}
        self.resource_pools: Dict[str, Dict] = {}
        self.isolated_services: Set[str] = set()
        self._lock = asyncio.Lock()
    
    async def create_isolation_group(
        self,
        group_name: str,
        services: List[str],
        max_concurrent_requests: int = 100,
        timeout_seconds: int = 30
    ):
        """Create isolation group with resource limits"""
        async with self._lock:
            self.isolation_groups[group_name] = {
                'services': set(services),
                'max_concurrent': max_concurrent_requests,
                'timeout': timeout_seconds,
                'current_requests': 0,
                'semaphore': asyncio.Semaphore(max_concurrent_requests),
                'created_at': datetime.now()
            }
            
            logger.info(f"Created isolation group '{group_name}' with services: {services}")
    
    @asynccontextmanager
    async def isolate_request(self, service_name: str):
        """Context manager for isolated request execution"""
        group_name = None
        
        # Find which isolation group this service belongs to
        async with self._lock:
            for name, group in self.isolation_groups.items():
                if service_name in group['services']:
                    group_name = name
                    break
        
        if not group_name:
            # No isolation group, execute normally
            yield
            return
        
        group = self.isolation_groups[group_name]
        
        try:
            # Acquire semaphore with timeout
            await asyncio.wait_for(
                group['semaphore'].acquire(),
                timeout=group['timeout']
            )
            
            async with self._lock:
                group['current_requests'] += 1
            
            yield
            
        except asyncio.TimeoutError:
            logger.warning(f"Request to {service_name} timed out due to bulkhead limits")
            raise ResourceExhaustedException(f"Service {service_name} request queue full")
        
        finally:
            if group_name:
                group['semaphore'].release()
                async with self._lock:
                    group['current_requests'] -= 1
    
    async def isolate_service(self, service_name: str, reason: str = "Health check failure"):
        """Temporarily isolate a failing service"""
        async with self._lock:
            self.isolated_services.add(service_name)
            logger.warning(f"Service {service_name} isolated: {reason}")
    
    async def restore_service(self, service_name: str):
        """Restore service from isolation"""
        async with self._lock:
            if service_name in self.isolated_services:
                self.isolated_services.remove(service_name)
                logger.info(f"Service {service_name} restored from isolation")
    
    def get_isolation_status(self) -> Dict[str, Any]:
        """Get current isolation status"""
        return {
            'isolation_groups': {
                name: {
                    'services': list(group['services']),
                    'max_concurrent': group['max_concurrent'],
                    'current_requests': group['current_requests'],
                    'timeout_seconds': group['timeout'],
                    'created_at': group['created_at'].isoformat()
                }
                for name, group in self.isolation_groups.items()
            },
            'isolated_services': list(self.isolated_services),
            'total_groups': len(self.isolation_groups)
        }


# Global instances
bulkhead_manager = BulkheadManager()


class ResourceExhaustedException(InvestmentAnalysisException):
    """Exception when resources are exhausted"""
    pass