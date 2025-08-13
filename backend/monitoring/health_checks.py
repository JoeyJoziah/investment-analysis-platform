"""
Comprehensive Health Check and SLA Monitoring System
Advanced health checks with dependency monitoring and SLA tracking.
"""

import asyncio
import logging
import time
import json
from typing import Dict, Any, Optional, List, Callable, Union
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import statistics

import aiohttp
import asyncpg
from prometheus_client import Gauge, Counter, Histogram

from backend.config.settings import settings
from backend.config.monitoring_config import monitoring_config
from backend.utils.structured_logging import get_structured_logger

logger = get_structured_logger(__name__)


class HealthStatus(Enum):
    """Health check status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


class ServiceType(Enum):
    """Service types for classification."""
    DATABASE = "database"
    CACHE = "cache"
    API = "api"
    EXTERNAL_API = "external_api"
    MESSAGE_QUEUE = "message_queue"
    FILE_SYSTEM = "file_system"
    BUSINESS_LOGIC = "business_logic"


# Health Check Metrics
health_check_status = Gauge(
    'health_check_status',
    'Health check status (1=healthy, 0.5=degraded, 0=unhealthy)',
    ['service', 'check_type']
)

health_check_duration = Histogram(
    'health_check_duration_seconds',
    'Health check execution time',
    ['service', 'check_type']
)

health_check_failures = Counter(
    'health_check_failures_total',
    'Health check failure count',
    ['service', 'failure_type']
)

sla_compliance = Gauge(
    'sla_compliance_percent',
    'SLA compliance percentage',
    ['service', 'sla_type', 'time_window']
)

service_availability = Gauge(
    'service_availability_percent',
    'Service availability percentage',
    ['service', 'time_window']
)

response_time_sla = Gauge(
    'response_time_sla_compliance_percent',
    'Response time SLA compliance',
    ['service', 'percentile', 'time_window']
)

service_dependency_health = Gauge(
    'service_dependency_health',
    'Service dependency health score',
    ['service', 'dependency']
)


@dataclass
class HealthCheckResult:
    """Health check result data structure."""
    service: str
    check_type: str
    status: HealthStatus
    message: str
    duration: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_details: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'service': self.service,
            'check_type': self.check_type,
            'status': self.status.value,
            'message': self.message,
            'duration': self.duration,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata,
            'error_details': self.error_details
        }


@dataclass
class SLATarget:
    """SLA target definition."""
    name: str
    service: str
    target_type: str  # 'availability', 'response_time', 'error_rate'
    threshold: float
    time_window: str  # '1h', '24h', '30d'
    measurement_interval: int = 60  # seconds


@dataclass
class ServiceHealth:
    """Overall service health tracking."""
    service_name: str
    service_type: ServiceType
    overall_status: HealthStatus = HealthStatus.HEALTHY
    last_check: Optional[datetime] = None
    consecutive_failures: int = 0
    uptime_percentage: float = 100.0
    avg_response_time: float = 0.0
    error_rate: float = 0.0
    dependencies: List[str] = field(default_factory=list)
    health_history: deque = field(default_factory=lambda: deque(maxlen=1000))


class HealthChecker:
    """Individual health check implementation."""
    
    def __init__(
        self,
        name: str,
        service: str,
        check_func: Callable,
        service_type: ServiceType,
        interval: int = 60,
        timeout: int = 30,
        retries: int = 2,
        critical: bool = False
    ):
        self.name = name
        self.service = service
        self.check_func = check_func
        self.service_type = service_type
        self.interval = interval
        self.timeout = timeout
        self.retries = retries
        self.critical = critical
        self.last_result: Optional[HealthCheckResult] = None
    
    async def execute(self) -> HealthCheckResult:
        """Execute health check with retries and timeout."""
        start_time = time.time()
        
        for attempt in range(self.retries + 1):
            try:
                # Execute with timeout
                result = await asyncio.wait_for(
                    self.check_func(),
                    timeout=self.timeout
                )
                
                duration = time.time() - start_time
                
                # Create result
                if isinstance(result, dict):
                    status = HealthStatus(result.get('status', HealthStatus.HEALTHY.value))
                    message = result.get('message', 'Health check passed')
                    metadata = result.get('metadata', {})
                    error_details = result.get('error_details')
                else:
                    status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
                    message = 'Health check passed' if result else 'Health check failed'
                    metadata = {}
                    error_details = None
                
                self.last_result = HealthCheckResult(
                    service=self.service,
                    check_type=self.name,
                    status=status,
                    message=message,
                    duration=duration,
                    timestamp=datetime.now(),
                    metadata=metadata,
                    error_details=error_details
                )
                
                # Record metrics
                status_value = {
                    HealthStatus.HEALTHY: 1.0,
                    HealthStatus.DEGRADED: 0.5,
                    HealthStatus.UNHEALTHY: 0.0,
                    HealthStatus.CRITICAL: 0.0
                }.get(status, 0.0)
                
                health_check_status.labels(
                    service=self.service,
                    check_type=self.name
                ).set(status_value)
                
                health_check_duration.labels(
                    service=self.service,
                    check_type=self.name
                ).observe(duration)
                
                if status != HealthStatus.HEALTHY:
                    health_check_failures.labels(
                        service=self.service,
                        failure_type=status.value
                    ).inc()
                
                return self.last_result
            
            except asyncio.TimeoutError:
                if attempt < self.retries:
                    continue
                error_msg = f"Health check timeout after {self.timeout}s"
                logger.warning(f"Health check timeout: {self.service}.{self.name}")
            
            except Exception as e:
                if attempt < self.retries:
                    continue
                error_msg = f"Health check error: {str(e)}"
                logger.error(f"Health check error: {self.service}.{self.name}: {e}")
        
        # All retries failed
        duration = time.time() - start_time
        self.last_result = HealthCheckResult(
            service=self.service,
            check_type=self.name,
            status=HealthStatus.CRITICAL if self.critical else HealthStatus.UNHEALTHY,
            message=error_msg,
            duration=duration,
            timestamp=datetime.now(),
            error_details=str(e) if 'e' in locals() else None
        )
        
        # Record failure metrics
        health_check_status.labels(
            service=self.service,
            check_type=self.name
        ).set(0.0)
        
        health_check_failures.labels(
            service=self.service,
            failure_type='timeout' if 'timeout' in error_msg else 'error'
        ).inc()
        
        return self.last_result


class SLAMonitor:
    """SLA monitoring and compliance tracking."""
    
    def __init__(self):
        self.sla_targets: Dict[str, SLATarget] = {}
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.compliance_cache: Dict[str, Dict] = {}
        self._monitoring_task: Optional[asyncio.Task] = None
    
    async def start_monitoring(self):
        """Start SLA monitoring."""
        if not self._monitoring_task:
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            logger.info("Started SLA monitoring")
    
    async def stop_monitoring(self):
        """Stop SLA monitoring."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            logger.info("Stopped SLA monitoring")
    
    def add_sla_target(self, sla_target: SLATarget):
        """Add SLA target for monitoring."""
        key = f"{sla_target.service}:{sla_target.name}"
        self.sla_targets[key] = sla_target
        logger.info(f"Added SLA target: {key}")
    
    async def _monitoring_loop(self):
        """Background SLA monitoring loop."""
        while True:
            try:
                await self._calculate_sla_compliance()
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error in SLA monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def _calculate_sla_compliance(self):
        """Calculate SLA compliance for all targets."""
        for key, target in self.sla_targets.items():
            try:
                compliance = await self._calculate_single_sla(target)
                
                # Update metrics
                sla_compliance.labels(
                    service=target.service,
                    sla_type=target.target_type,
                    time_window=target.time_window
                ).set(compliance)
                
                # Cache result
                self.compliance_cache[key] = {
                    'compliance': compliance,
                    'timestamp': datetime.now(),
                    'target': target.threshold
                }
            
            except Exception as e:
                logger.error(f"Error calculating SLA compliance for {key}: {e}")
    
    async def _calculate_single_sla(self, target: SLATarget) -> float:
        """Calculate compliance for single SLA target."""
        if target.target_type == 'availability':
            return await self._calculate_availability_sla(target)
        elif target.target_type == 'response_time':
            return await self._calculate_response_time_sla(target)
        elif target.target_type == 'error_rate':
            return await self._calculate_error_rate_sla(target)
        else:
            logger.warning(f"Unknown SLA target type: {target.target_type}")
            return 0.0
    
    async def _calculate_availability_sla(self, target: SLATarget) -> float:
        """Calculate availability SLA compliance."""
        try:
            from backend.monitoring.metrics_collector import metrics_collector
            from prometheus_client import REGISTRY
            
            # Query service availability from health check metrics
            service_key = f"{target.service}:availability"
            if service_key in self.metrics_history:
                recent_checks = list(self.metrics_history[service_key])
                if recent_checks:
                    successful_checks = sum(1 for check in recent_checks if check >= 0.5)
                    availability = (successful_checks / len(recent_checks)) * 100
                    
                    service_availability.labels(
                        service=target.service,
                        time_window=target.time_window
                    ).set(availability)
                    
                    return min(100.0, (availability / target.threshold) * 100)
            
            return 0.0
        
        except Exception as e:
            logger.error(f"Error calculating availability SLA: {e}")
            return 0.0
    
    async def _calculate_response_time_sla(self, target: SLATarget) -> float:
        """Calculate response time SLA compliance."""
        try:
            service_key = f"{target.service}:response_time"
            if service_key in self.metrics_history:
                response_times = list(self.metrics_history[service_key])
                if response_times:
                    p95_response_time = statistics.quantiles(response_times, n=20)[18]  # 95th percentile
                    
                    response_time_sla.labels(
                        service=target.service,
                        percentile="p95",
                        time_window=target.time_window
                    ).set(100.0 if p95_response_time <= target.threshold else 0.0)
                    
                    compliant_responses = sum(1 for rt in response_times if rt <= target.threshold)
                    compliance = (compliant_responses / len(response_times)) * 100
                    
                    return compliance
            
            return 100.0
        
        except Exception as e:
            logger.error(f"Error calculating response time SLA: {e}")
            return 0.0
    
    async def _calculate_error_rate_sla(self, target: SLATarget) -> float:
        """Calculate error rate SLA compliance."""
        try:
            error_key = f"{target.service}:errors"
            success_key = f"{target.service}:success"
            
            errors = list(self.metrics_history.get(error_key, []))
            successes = list(self.metrics_history.get(success_key, []))
            
            if errors or successes:
                total_requests = len(errors) + len(successes)
                error_rate = (len(errors) / total_requests) * 100 if total_requests > 0 else 0.0
                
                compliance = 100.0 if error_rate <= target.threshold else 0.0
                return compliance
            
            return 100.0
        
        except Exception as e:
            logger.error(f"Error calculating error rate SLA: {e}")
            return 100.0
    
    def record_metric(self, service: str, metric_type: str, value: float):
        """Record metric for SLA calculation."""
        key = f"{service}:{metric_type}"
        self.metrics_history[key].append(value)
    
    def get_sla_summary(self) -> Dict[str, Any]:
        """Get SLA compliance summary."""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "sla_targets": len(self.sla_targets),
            "compliance_summary": {}
        }
        
        for key, cached in self.compliance_cache.items():
            summary["compliance_summary"][key] = {
                "compliance_percent": cached["compliance"],
                "target_threshold": cached["target"],
                "last_calculated": cached["timestamp"].isoformat()
            }
        
        return summary


class HealthMonitoringSystem:
    """
    Comprehensive health monitoring system with dependency tracking.
    """
    
    def __init__(self):
        self.health_checkers: Dict[str, HealthChecker] = {}
        self.service_health: Dict[str, ServiceHealth] = {}
        self.sla_monitor = SLAMonitor()
        self.dependency_graph: Dict[str, List[str]] = {}
        
        # Background tasks
        self._health_check_tasks: Dict[str, asyncio.Task] = {}
        self._overall_monitor_task: Optional[asyncio.Task] = None
        
        # Setup default health checks
        self._setup_default_health_checks()
        self._setup_default_sla_targets()
    
    def _setup_default_health_checks(self):
        """Setup default health checks for core services."""
        # Database health check
        self.add_health_check(
            name="connectivity",
            service="database",
            check_func=self._check_database_health,
            service_type=ServiceType.DATABASE,
            interval=30,
            critical=True
        )
        
        # Redis health check
        self.add_health_check(
            name="connectivity",
            service="redis",
            check_func=self._check_redis_health,
            service_type=ServiceType.CACHE,
            interval=30,
            critical=True
        )
        
        # API health check
        self.add_health_check(
            name="responsiveness",
            service="api",
            check_func=self._check_api_health,
            service_type=ServiceType.API,
            interval=60
        )
        
        # External API health checks
        for provider in ['alpha_vantage', 'finnhub', 'polygon']:
            self.add_health_check(
                name="connectivity",
                service=f"external_api_{provider}",
                check_func=lambda p=provider: self._check_external_api_health(p),
                service_type=ServiceType.EXTERNAL_API,
                interval=300  # 5 minutes
            )
        
        # Business logic health checks
        self.add_health_check(
            name="recommendation_engine",
            service="business_logic",
            check_func=self._check_recommendation_engine_health,
            service_type=ServiceType.BUSINESS_LOGIC,
            interval=120
        )
        
        self.add_health_check(
            name="stock_processing",
            service="business_logic",
            check_func=self._check_stock_processing_health,
            service_type=ServiceType.BUSINESS_LOGIC,
            interval=180
        )
    
    def _setup_default_sla_targets(self):
        """Setup default SLA targets."""
        # API response time SLA
        self.sla_monitor.add_sla_target(SLATarget(
            name="response_time_p95",
            service="api",
            target_type="response_time",
            threshold=2.0,  # 2 seconds
            time_window="24h"
        ))
        
        # Database availability SLA
        self.sla_monitor.add_sla_target(SLATarget(
            name="availability",
            service="database",
            target_type="availability",
            threshold=99.9,  # 99.9% uptime
            time_window="30d"
        ))
        
        # API error rate SLA
        self.sla_monitor.add_sla_target(SLATarget(
            name="error_rate",
            service="api",
            target_type="error_rate",
            threshold=1.0,  # Max 1% error rate
            time_window="24h"
        ))
    
    def add_health_check(
        self,
        name: str,
        service: str,
        check_func: Callable,
        service_type: ServiceType,
        interval: int = 60,
        timeout: int = 30,
        retries: int = 2,
        critical: bool = False
    ):
        """Add health check to monitoring."""
        key = f"{service}:{name}"
        
        checker = HealthChecker(
            name=name,
            service=service,
            check_func=check_func,
            service_type=service_type,
            interval=interval,
            timeout=timeout,
            retries=retries,
            critical=critical
        )
        
        self.health_checkers[key] = checker
        
        # Initialize service health if not exists
        if service not in self.service_health:
            self.service_health[service] = ServiceHealth(
                service_name=service,
                service_type=service_type
            )
        
        logger.info(f"Added health check: {key}")
    
    async def start_monitoring(self):
        """Start all health monitoring."""
        # Start individual health check tasks
        for key, checker in self.health_checkers.items():
            if key not in self._health_check_tasks:
                task = asyncio.create_task(self._health_check_loop(checker))
                self._health_check_tasks[key] = task
        
        # Start overall monitoring
        if not self._overall_monitor_task:
            self._overall_monitor_task = asyncio.create_task(self._overall_monitoring_loop())
        
        # Start SLA monitoring
        await self.sla_monitor.start_monitoring()
        
        logger.info("Started comprehensive health monitoring")
    
    async def stop_monitoring(self):
        """Stop all health monitoring."""
        # Stop health check tasks
        for task in self._health_check_tasks.values():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        self._health_check_tasks.clear()
        
        # Stop overall monitoring
        if self._overall_monitor_task:
            self._overall_monitor_task.cancel()
            try:
                await self._overall_monitor_task
            except asyncio.CancelledError:
                pass
        
        # Stop SLA monitoring
        await self.sla_monitor.stop_monitoring()
        
        logger.info("Stopped health monitoring")
    
    async def _health_check_loop(self, checker: HealthChecker):
        """Background loop for individual health check."""
        while True:
            try:
                result = await checker.execute()
                
                # Update service health
                self._update_service_health(result)
                
                # Record SLA metrics
                self.sla_monitor.record_metric(
                    checker.service, 
                    "availability", 
                    1.0 if result.status == HealthStatus.HEALTHY else 0.0
                )
                self.sla_monitor.record_metric(
                    checker.service,
                    "response_time",
                    result.duration
                )
                
                await asyncio.sleep(checker.interval)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop for {checker.service}:{checker.name}: {e}")
                await asyncio.sleep(checker.interval)
    
    def _update_service_health(self, result: HealthCheckResult):
        """Update overall service health based on check result."""
        service = result.service
        if service in self.service_health:
            health = self.service_health[service]
            
            # Update basic metrics
            health.last_check = result.timestamp
            health.health_history.append(result)
            
            # Update failure count
            if result.status == HealthStatus.HEALTHY:
                health.consecutive_failures = 0
            else:
                health.consecutive_failures += 1
            
            # Calculate overall status
            if health.consecutive_failures >= 3:
                health.overall_status = HealthStatus.CRITICAL
            elif health.consecutive_failures >= 2:
                health.overall_status = HealthStatus.UNHEALTHY
            elif health.consecutive_failures >= 1:
                health.overall_status = HealthStatus.DEGRADED
            else:
                health.overall_status = HealthStatus.HEALTHY
            
            # Update uptime percentage (last 100 checks)
            recent_checks = list(health.health_history)[-100:]
            if recent_checks:
                healthy_checks = sum(1 for check in recent_checks 
                                   if check.status == HealthStatus.HEALTHY)
                health.uptime_percentage = (healthy_checks / len(recent_checks)) * 100
            
            # Update average response time
            if recent_checks:
                health.avg_response_time = sum(check.duration for check in recent_checks) / len(recent_checks)
    
    async def _overall_monitoring_loop(self):
        """Background loop for overall system monitoring."""
        while True:
            try:
                await self._update_dependency_health()
                await self._check_system_wide_health()
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error in overall monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def _update_dependency_health(self):
        """Update service dependency health scores."""
        for service, dependencies in self.dependency_graph.items():
            for dependency in dependencies:
                if dependency in self.service_health:
                    dep_health = self.service_health[dependency]
                    
                    # Calculate health score (0-1)
                    status_scores = {
                        HealthStatus.HEALTHY: 1.0,
                        HealthStatus.DEGRADED: 0.7,
                        HealthStatus.UNHEALTHY: 0.3,
                        HealthStatus.CRITICAL: 0.0
                    }
                    
                    score = status_scores.get(dep_health.overall_status, 0.0)
                    
                    service_dependency_health.labels(
                        service=service,
                        dependency=dependency
                    ).set(score)
    
    async def _check_system_wide_health(self):
        """Check overall system health and trigger alerts if needed."""
        unhealthy_services = []
        critical_services = []
        
        for service, health in self.service_health.items():
            if health.overall_status == HealthStatus.CRITICAL:
                critical_services.append(service)
            elif health.overall_status == HealthStatus.UNHEALTHY:
                unhealthy_services.append(service)
        
        # Trigger alerts for critical services
        if critical_services:
            from backend.monitoring.alerting_system import alert_manager, AlertSeverity
            
            await alert_manager.create_alert(
                title=f"Critical Services: {', '.join(critical_services)}",
                description=f"The following services are in critical state: {', '.join(critical_services)}",
                severity=AlertSeverity.CRITICAL,
                source="health_monitor",
                alert_type="service_critical",
                metadata={
                    "critical_services": critical_services,
                    "unhealthy_services": unhealthy_services
                }
            )
    
    # Health check implementations
    
    async def _check_database_health(self) -> Dict[str, Any]:
        """Check database connectivity and performance."""
        try:
            from backend.utils.async_database import async_db_manager
            
            start_time = time.time()
            
            async with async_db_manager.get_session() as db:
                # Simple connectivity test
                result = await db.execute("SELECT 1 as health_check")
                
                # Check connection pool
                if async_db_manager._engine and async_db_manager._engine.pool:
                    pool = async_db_manager._engine.pool
                    pool_utilization = (pool.checked_out() / pool.size()) * 100 if pool.size() > 0 else 0
                else:
                    pool_utilization = 0
                
                duration = time.time() - start_time
                
                # Determine status based on performance
                if duration > 2.0:
                    status = HealthStatus.DEGRADED
                    message = f"Database slow response: {duration:.2f}s"
                elif pool_utilization > 80:
                    status = HealthStatus.DEGRADED
                    message = f"High pool utilization: {pool_utilization:.1f}%"
                else:
                    status = HealthStatus.HEALTHY
                    message = "Database healthy"
                
                return {
                    'status': status.value,
                    'message': message,
                    'metadata': {
                        'response_time': duration,
                        'pool_utilization': pool_utilization,
                        'pool_size': pool.size() if async_db_manager._engine and async_db_manager._engine.pool else 0
                    }
                }
        
        except Exception as e:
            return {
                'status': HealthStatus.UNHEALTHY.value,
                'message': f"Database health check failed: {str(e)}",
                'error_details': str(e)
            }
    
    async def _check_redis_health(self) -> Dict[str, Any]:
        """Check Redis connectivity and performance."""
        try:
            from backend.utils.redis_resilience import resilient_redis
            
            start_time = time.time()
            
            # Test basic connectivity
            await resilient_redis.ping()
            
            # Get Redis info
            info = await resilient_redis.info()
            memory_usage = info.get('used_memory', 0)
            max_memory = info.get('maxmemory', 0)
            
            duration = time.time() - start_time
            
            # Calculate memory utilization
            memory_utilization = (memory_usage / max_memory * 100) if max_memory > 0 else 0
            
            # Determine status
            if duration > 1.0:
                status = HealthStatus.DEGRADED
                message = f"Redis slow response: {duration:.2f}s"
            elif memory_utilization > 90:
                status = HealthStatus.DEGRADED
                message = f"High memory usage: {memory_utilization:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = "Redis healthy"
            
            return {
                'status': status.value,
                'message': message,
                'metadata': {
                    'response_time': duration,
                    'memory_utilization': memory_utilization,
                    'connected_clients': info.get('connected_clients', 0)
                }
            }
        
        except Exception as e:
            return {
                'status': HealthStatus.UNHEALTHY.value,
                'message': f"Redis health check failed: {str(e)}",
                'error_details': str(e)
            }
    
    async def _check_api_health(self) -> Dict[str, Any]:
        """Check API responsiveness."""
        try:
            start_time = time.time()
            
            # Test internal API endpoint
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get('http://localhost:8000/api/health') as response:
                    duration = time.time() - start_time
                    
                    if response.status == 200:
                        if duration > 3.0:
                            status = HealthStatus.DEGRADED
                            message = f"API slow response: {duration:.2f}s"
                        else:
                            status = HealthStatus.HEALTHY
                            message = "API healthy"
                    else:
                        status = HealthStatus.UNHEALTHY
                        message = f"API returned status {response.status}"
                    
                    return {
                        'status': status.value,
                        'message': message,
                        'metadata': {
                            'response_time': duration,
                            'status_code': response.status
                        }
                    }
        
        except Exception as e:
            return {
                'status': HealthStatus.UNHEALTHY.value,
                'message': f"API health check failed: {str(e)}",
                'error_details': str(e)
            }
    
    async def _check_external_api_health(self, provider: str) -> Dict[str, Any]:
        """Check external API connectivity."""
        try:
            from backend.data_ingestion.base_client import BaseAPIClient
            
            # Get appropriate client
            if provider == 'alpha_vantage':
                from backend.data_ingestion.alpha_vantage_client import AlphaVantageClient
                client = AlphaVantageClient()
            elif provider == 'finnhub':
                from backend.data_ingestion.finnhub_client import FinnhubClient
                client = FinnhubClient()
            elif provider == 'polygon':
                from backend.data_ingestion.polygon_client import PolygonClient
                client = PolygonClient()
            else:
                raise ValueError(f"Unknown provider: {provider}")
            
            start_time = time.time()
            
            # Test simple API call
            health_result = await client.health_check()
            duration = time.time() - start_time
            
            if health_result:
                status = HealthStatus.HEALTHY
                message = f"{provider} API healthy"
            else:
                status = HealthStatus.DEGRADED
                message = f"{provider} API degraded"
            
            return {
                'status': status.value,
                'message': message,
                'metadata': {
                    'response_time': duration,
                    'provider': provider
                }
            }
        
        except Exception as e:
            return {
                'status': HealthStatus.UNHEALTHY.value,
                'message': f"{provider} API health check failed: {str(e)}",
                'error_details': str(e)
            }
    
    async def _check_recommendation_engine_health(self) -> Dict[str, Any]:
        """Check recommendation engine health."""
        try:
            from backend.analytics.recommendation_engine import RecommendationEngine
            
            engine = RecommendationEngine()
            start_time = time.time()
            
            # Test recommendation generation with a sample stock
            test_result = await engine.quick_health_check()
            duration = time.time() - start_time
            
            if test_result:
                if duration > 10.0:
                    status = HealthStatus.DEGRADED
                    message = f"Recommendation engine slow: {duration:.2f}s"
                else:
                    status = HealthStatus.HEALTHY
                    message = "Recommendation engine healthy"
            else:
                status = HealthStatus.UNHEALTHY
                message = "Recommendation engine failed health check"
            
            return {
                'status': status.value,
                'message': message,
                'metadata': {
                    'response_time': duration,
                    'test_passed': test_result
                }
            }
        
        except Exception as e:
            return {
                'status': HealthStatus.UNHEALTHY.value,
                'message': f"Recommendation engine health check failed: {str(e)}",
                'error_details': str(e)
            }
    
    async def _check_stock_processing_health(self) -> Dict[str, Any]:
        """Check stock processing pipeline health."""
        try:
            from backend.repositories.stock_repository import StockRepository
            
            stock_repo = StockRepository()
            start_time = time.time()
            
            # Check recent processing activity
            processing_stats = await stock_repo.get_processing_health_stats()
            duration = time.time() - start_time
            
            if processing_stats:
                recent_activity = processing_stats.get('recent_activity', 0)
                error_rate = processing_stats.get('error_rate', 0)
                
                if error_rate > 10:  # > 10% error rate
                    status = HealthStatus.DEGRADED
                    message = f"High stock processing error rate: {error_rate:.1f}%"
                elif recent_activity < 50:  # < 50 stocks processed recently
                    status = HealthStatus.DEGRADED
                    message = f"Low stock processing activity: {recent_activity} stocks"
                else:
                    status = HealthStatus.HEALTHY
                    message = "Stock processing healthy"
                
                return {
                    'status': status.value,
                    'message': message,
                    'metadata': {
                        'response_time': duration,
                        'recent_activity': recent_activity,
                        'error_rate': error_rate
                    }
                }
            else:
                return {
                    'status': HealthStatus.DEGRADED.value,
                    'message': "No recent stock processing activity"
                }
        
        except Exception as e:
            return {
                'status': HealthStatus.UNHEALTHY.value,
                'message': f"Stock processing health check failed: {str(e)}",
                'error_details': str(e)
            }
    
    # Public API methods
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        overall_status = HealthStatus.HEALTHY
        service_statuses = {}
        
        for service, health in self.service_health.items():
            service_statuses[service] = {
                'status': health.overall_status.value,
                'uptime_percentage': health.uptime_percentage,
                'avg_response_time': health.avg_response_time,
                'consecutive_failures': health.consecutive_failures,
                'last_check': health.last_check.isoformat() if health.last_check else None
            }
            
            # Update overall status (worst case)
            if health.overall_status.value == HealthStatus.CRITICAL.value:
                overall_status = HealthStatus.CRITICAL
            elif health.overall_status.value == HealthStatus.UNHEALTHY.value and overall_status != HealthStatus.CRITICAL:
                overall_status = HealthStatus.UNHEALTHY
            elif health.overall_status.value == HealthStatus.DEGRADED.value and overall_status == HealthStatus.HEALTHY:
                overall_status = HealthStatus.DEGRADED
        
        return {
            'timestamp': datetime.now().isoformat(),
            'overall_status': overall_status.value,
            'services': service_statuses,
            'sla_summary': self.sla_monitor.get_sla_summary()
        }
    
    async def get_detailed_health(self, service: Optional[str] = None) -> Dict[str, Any]:
        """Get detailed health information."""
        if service:
            # Return detailed info for specific service
            if service in self.service_health:
                health = self.service_health[service]
                recent_checks = list(health.health_history)[-10:]
                
                return {
                    'service': service,
                    'status': health.overall_status.value,
                    'metrics': {
                        'uptime_percentage': health.uptime_percentage,
                        'avg_response_time': health.avg_response_time,
                        'error_rate': health.error_rate,
                        'consecutive_failures': health.consecutive_failures
                    },
                    'recent_checks': [check.to_dict() for check in recent_checks],
                    'dependencies': health.dependencies
                }
            else:
                return {'error': f'Service {service} not found'}
        else:
            # Return detailed info for all services
            detailed_info = {}
            for svc in self.service_health.keys():
                detailed_info[svc] = await self.get_detailed_health(svc)
            
            return detailed_info


# Global health monitoring system
health_monitor = HealthMonitoringSystem()


# Setup function
async def setup_health_monitoring():
    """Setup health monitoring system."""
    await health_monitor.start_monitoring()
    logger.info("Health monitoring system setup completed")


# Convenience functions
async def get_system_health():
    """Get current system health status."""
    return await health_monitor.get_health_status()


async def get_service_health(service: str):
    """Get specific service health status."""
    return await health_monitor.get_detailed_health(service)