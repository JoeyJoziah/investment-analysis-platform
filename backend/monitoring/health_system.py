"""
Production Health Check and Graceful Shutdown System
Comprehensive health monitoring with dependency tracking and graceful degradation
"""

import asyncio
import logging
import signal
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import psutil
import redis
import sqlalchemy
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import HTTPException
import httpx

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health check status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class HealthCheck:
    """Individual health check configuration"""
    name: str
    check_func: Callable
    timeout: int = 30
    interval: int = 60
    retries: int = 3
    critical: bool = False
    dependencies: List[str] = field(default_factory=list)
    
    # Runtime state
    last_check: Optional[datetime] = None
    last_status: HealthStatus = HealthStatus.HEALTHY
    consecutive_failures: int = 0
    last_error: Optional[str] = None


@dataclass
class ServiceHealth:
    """Overall service health information"""
    service_name: str
    status: HealthStatus
    checks: Dict[str, Dict[str, Any]]
    uptime: float
    timestamp: datetime
    version: str
    environment: str


class HealthMonitor:
    """
    Production-ready health monitoring system
    Provides comprehensive health checks with dependency tracking
    """
    
    def __init__(self, service_name: str = "investment-platform", version: str = "1.0.0"):
        self.service_name = service_name
        self.version = version
        self.startup_time = time.time()
        
        # Health checks registry
        self.health_checks: Dict[str, HealthCheck] = {}
        
        # Shutdown handling
        self.shutdown_event = asyncio.Event()
        self.shutdown_timeout = 30  # seconds
        self.cleanup_tasks: List[Callable] = []
        
        # Health check results cache
        self.health_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = 10  # seconds
        
        # Initialize core health checks
        self._setup_core_health_checks()
        self._setup_shutdown_handlers()
        
        # Background health monitoring
        self._monitor_task = None
        self.start_monitoring()
    
    def _setup_core_health_checks(self):
        """Setup essential health checks"""
        
        # System resource checks
        self.register_health_check(
            "system_memory",
            self._check_system_memory,
            timeout=5,
            interval=30,
            critical=True
        )
        
        self.register_health_check(
            "system_cpu",
            self._check_system_cpu,
            timeout=5,
            interval=30
        )
        
        self.register_health_check(
            "disk_space",
            self._check_disk_space,
            timeout=5,
            interval=60,
            critical=True
        )
        
        # Application checks
        self.register_health_check(
            "application_startup",
            self._check_application_startup,
            timeout=10,
            interval=120
        )
    
    def _setup_shutdown_handlers(self):
        """Setup graceful shutdown signal handlers"""
        def signal_handler(signum, frame):
            logger.info(f"Received shutdown signal {signum}")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    
    def register_health_check(self, name: str, check_func: Callable, **kwargs):
        """Register a new health check"""
        health_check = HealthCheck(name=name, check_func=check_func, **kwargs)
        self.health_checks[name] = health_check
        logger.info(f"Registered health check: {name}")
    
    def register_cleanup_task(self, cleanup_func: Callable):
        """Register cleanup task for graceful shutdown"""
        self.cleanup_tasks.append(cleanup_func)
    
    async def _check_system_memory(self) -> Dict[str, Any]:
        """Check system memory usage"""
        memory = psutil.virtual_memory()
        
        status = HealthStatus.HEALTHY
        if memory.percent > 90:
            status = HealthStatus.CRITICAL
        elif memory.percent > 80:
            status = HealthStatus.DEGRADED
        elif memory.percent > 70:
            status = HealthStatus.UNHEALTHY
        
        return {
            "status": status,
            "memory_percent": memory.percent,
            "memory_available": memory.available,
            "memory_total": memory.total,
            "details": f"Memory usage: {memory.percent:.1f}%"
        }
    
    async def _check_system_cpu(self) -> Dict[str, Any]:
        """Check CPU usage"""
        cpu_percent = psutil.cpu_percent(interval=1)
        load_avg = psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else None
        
        status = HealthStatus.HEALTHY
        if cpu_percent > 95:
            status = HealthStatus.CRITICAL
        elif cpu_percent > 85:
            status = HealthStatus.DEGRADED
        elif cpu_percent > 75:
            status = HealthStatus.UNHEALTHY
        
        result = {
            "status": status,
            "cpu_percent": cpu_percent,
            "details": f"CPU usage: {cpu_percent:.1f}%"
        }
        
        if load_avg is not None:
            result["load_average"] = load_avg
        
        return result
    
    async def _check_disk_space(self) -> Dict[str, Any]:
        """Check available disk space"""
        disk = psutil.disk_usage('/')
        
        usage_percent = (disk.used / disk.total) * 100
        
        status = HealthStatus.HEALTHY
        if usage_percent > 95:
            status = HealthStatus.CRITICAL
        elif usage_percent > 90:
            status = HealthStatus.DEGRADED
        elif usage_percent > 85:
            status = HealthStatus.UNHEALTHY
        
        return {
            "status": status,
            "disk_usage_percent": usage_percent,
            "disk_free": disk.free,
            "disk_total": disk.total,
            "details": f"Disk usage: {usage_percent:.1f}%"
        }
    
    async def _check_application_startup(self) -> Dict[str, Any]:
        """Check if application has started successfully"""
        uptime = time.time() - self.startup_time
        
        status = HealthStatus.HEALTHY
        if uptime < 30:  # Still starting up
            status = HealthStatus.DEGRADED
        
        return {
            "status": status,
            "uptime_seconds": uptime,
            "startup_time": datetime.fromtimestamp(self.startup_time).isoformat(),
            "details": f"Application uptime: {uptime:.1f} seconds"
        }
    
    async def check_database(self, db_session: AsyncSession) -> Dict[str, Any]:
        """Check database connectivity and performance"""
        try:
            start_time = time.time()
            
            # Simple query to test connectivity
            result = await db_session.execute(sqlalchemy.text("SELECT 1"))
            query_time = time.time() - start_time
            
            # Check connection pool status
            pool_info = {
                "size": db_session.bind.pool.size(),
                "checked_in": db_session.bind.pool.checkedin(),
                "checked_out": db_session.bind.pool.checkedout(),
                "overflow": db_session.bind.pool.overflow(),
                "invalid": db_session.bind.pool.invalid()
            }
            
            status = HealthStatus.HEALTHY
            if query_time > 5.0:
                status = HealthStatus.CRITICAL
            elif query_time > 2.0:
                status = HealthStatus.DEGRADED
            elif query_time > 1.0:
                status = HealthStatus.UNHEALTHY
            
            return {
                "status": status,
                "query_time": query_time,
                "pool_info": pool_info,
                "details": f"Database response time: {query_time:.3f}s"
            }
            
        except Exception as e:
            return {
                "status": HealthStatus.CRITICAL,
                "error": str(e),
                "details": f"Database connection failed: {str(e)}"
            }
    
    async def check_redis(self, redis_client: redis.Redis) -> Dict[str, Any]:
        """Check Redis connectivity and performance"""
        try:
            start_time = time.time()
            
            # Test basic operations
            await redis_client.ping()
            await redis_client.set("health_check", "ok", ex=60)
            result = await redis_client.get("health_check")
            
            response_time = time.time() - start_time
            
            # Get Redis info
            info = await redis_client.info()
            memory_usage = info.get('used_memory', 0)
            max_memory = info.get('maxmemory', 0)
            
            status = HealthStatus.HEALTHY
            if response_time > 1.0:
                status = HealthStatus.CRITICAL
            elif response_time > 0.5:
                status = HealthStatus.DEGRADED
            elif response_time > 0.1:
                status = HealthStatus.UNHEALTHY
            
            # Check memory usage if maxmemory is set
            memory_percent = 0
            if max_memory > 0:
                memory_percent = (memory_usage / max_memory) * 100
                if memory_percent > 90:
                    status = max(status, HealthStatus.CRITICAL)
                elif memory_percent > 80:
                    status = max(status, HealthStatus.DEGRADED)
            
            return {
                "status": status,
                "response_time": response_time,
                "memory_usage": memory_usage,
                "memory_percent": memory_percent,
                "connected_clients": info.get('connected_clients', 0),
                "details": f"Redis response time: {response_time:.3f}s"
            }
            
        except Exception as e:
            return {
                "status": HealthStatus.CRITICAL,
                "error": str(e),
                "details": f"Redis connection failed: {str(e)}"
            }
    
    async def check_external_api(self, api_name: str, url: str, 
                                timeout: int = 10) -> Dict[str, Any]:
        """Check external API connectivity"""
        try:
            start_time = time.time()
            
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(url)
            
            response_time = time.time() - start_time
            
            status = HealthStatus.HEALTHY
            if response.status_code >= 500:
                status = HealthStatus.CRITICAL
            elif response.status_code >= 400:
                status = HealthStatus.UNHEALTHY
            elif response_time > 10.0:
                status = HealthStatus.DEGRADED
            
            return {
                "status": status,
                "response_time": response_time,
                "status_code": response.status_code,
                "details": f"{api_name} response: {response.status_code} in {response_time:.3f}s"
            }
            
        except Exception as e:
            return {
                "status": HealthStatus.CRITICAL,
                "error": str(e),
                "details": f"{api_name} connection failed: {str(e)}"
            }
    
    async def _run_health_check(self, check: HealthCheck) -> Dict[str, Any]:
        """Run a single health check with timeout and retry logic"""
        for attempt in range(check.retries):
            try:
                # Run health check with timeout
                result = await asyncio.wait_for(
                    check.check_func(),
                    timeout=check.timeout
                )
                
                # Update check state
                check.last_check = datetime.utcnow()
                check.last_status = result["status"]
                
                if result["status"] in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]:
                    check.consecutive_failures = 0
                    check.last_error = None
                    return result
                else:
                    check.consecutive_failures += 1
                
                return result
                
            except asyncio.TimeoutError:
                error_msg = f"Health check '{check.name}' timed out after {check.timeout}s"
                logger.warning(error_msg)
                
                if attempt == check.retries - 1:  # Last attempt
                    check.consecutive_failures += 1
                    check.last_error = error_msg
                    return {
                        "status": HealthStatus.CRITICAL,
                        "error": error_msg,
                        "details": f"Timeout after {check.timeout}s (attempt {attempt + 1}/{check.retries})"
                    }
                    
                await asyncio.sleep(min(2 ** attempt, 10))  # Exponential backoff
                
            except Exception as e:
                error_msg = f"Health check '{check.name}' failed: {str(e)}"
                logger.warning(error_msg)
                
                if attempt == check.retries - 1:  # Last attempt
                    check.consecutive_failures += 1
                    check.last_error = error_msg
                    return {
                        "status": HealthStatus.CRITICAL,
                        "error": error_msg,
                        "details": f"Exception: {str(e)} (attempt {attempt + 1}/{check.retries})"
                    }
                
                await asyncio.sleep(min(2 ** attempt, 10))  # Exponential backoff
    
    async def run_all_health_checks(self) -> Dict[str, Dict[str, Any]]:
        """Run all registered health checks"""
        results = {}
        
        # Check cache first
        now = datetime.utcnow()
        if hasattr(self, '_last_full_check'):
            if (now - self._last_full_check).total_seconds() < self.cache_ttl:
                return self.health_cache.copy()
        
        # Run health checks concurrently
        tasks = []
        for name, check in self.health_checks.items():
            tasks.append(self._run_health_check(check))
        
        check_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, (name, check) in enumerate(self.health_checks.items()):
            result = check_results[i]
            
            if isinstance(result, Exception):
                results[name] = {
                    "status": HealthStatus.CRITICAL,
                    "error": str(result),
                    "details": f"Health check failed with exception: {str(result)}"
                }
            else:
                results[name] = result
        
        # Cache results
        self.health_cache = results.copy()
        self._last_full_check = now
        
        return results
    
    def get_overall_health_status(self, check_results: Dict[str, Dict[str, Any]]) -> HealthStatus:
        """Determine overall health status from individual check results"""
        if not check_results:
            return HealthStatus.UNHEALTHY
        
        critical_failed = False
        any_critical = False
        any_unhealthy = False
        any_degraded = False
        
        for name, result in check_results.items():
            status = result.get("status", HealthStatus.CRITICAL)
            check = self.health_checks.get(name)
            
            if status == HealthStatus.CRITICAL:
                if check and check.critical:
                    critical_failed = True
                any_critical = True
            elif status == HealthStatus.UNHEALTHY:
                any_unhealthy = True
            elif status == HealthStatus.DEGRADED:
                any_degraded = True
        
        # Return most severe status
        if critical_failed:
            return HealthStatus.CRITICAL
        elif any_critical:
            return HealthStatus.UNHEALTHY
        elif any_unhealthy:
            return HealthStatus.UNHEALTHY
        elif any_degraded:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY
    
    async def get_health_report(self) -> ServiceHealth:
        """Get comprehensive health report"""
        check_results = await self.run_all_health_checks()
        overall_status = self.get_overall_health_status(check_results)
        uptime = time.time() - self.startup_time
        
        return ServiceHealth(
            service_name=self.service_name,
            status=overall_status,
            checks=check_results,
            uptime=uptime,
            timestamp=datetime.utcnow(),
            version=self.version,
            environment=os.getenv("ENVIRONMENT", "unknown")
        )
    
    def start_monitoring(self):
        """Start background health monitoring"""
        if self._monitor_task is None:
            self._monitor_task = asyncio.create_task(self._health_monitor_loop())
            logger.info("Started health monitoring")
    
    async def _health_monitor_loop(self):
        """Background health monitoring loop"""
        while not self.shutdown_event.is_set():
            try:
                # Run health checks
                await self.run_all_health_checks()
                
                # Check if any critical health checks are failing
                critical_failures = []
                for name, check in self.health_checks.items():
                    if (check.critical and 
                        check.last_status in [HealthStatus.CRITICAL, HealthStatus.UNHEALTHY] and
                        check.consecutive_failures >= 3):
                        critical_failures.append(name)
                
                if critical_failures:
                    logger.critical(f"Critical health check failures: {', '.join(critical_failures)}")
                
                # Wait before next check
                await asyncio.sleep(min(check.interval for check in self.health_checks.values()))
                
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    async def shutdown(self):
        """Graceful shutdown procedure"""
        logger.info("Initiating graceful shutdown...")
        
        # Set shutdown event
        self.shutdown_event.set()
        
        # Stop health monitoring
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        # Run cleanup tasks
        cleanup_tasks = []
        for cleanup_func in self.cleanup_tasks:
            try:
                if asyncio.iscoroutinefunction(cleanup_func):
                    cleanup_tasks.append(cleanup_func())
                else:
                    cleanup_func()
            except Exception as e:
                logger.error(f"Cleanup task error: {e}")
        
        if cleanup_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*cleanup_tasks, return_exceptions=True),
                    timeout=self.shutdown_timeout
                )
            except asyncio.TimeoutError:
                logger.warning(f"Cleanup tasks timed out after {self.shutdown_timeout}s")
        
        logger.info("Graceful shutdown completed")


# Global health monitor instance
health_monitor = None

def get_health_monitor() -> HealthMonitor:
    """Get global health monitor instance"""
    global health_monitor
    if health_monitor is None:
        health_monitor = HealthMonitor()
    return health_monitor


# FastAPI integration
async def health_endpoint():
    """FastAPI health check endpoint"""
    monitor = get_health_monitor()
    health_report = await monitor.get_health_report()
    
    # Return appropriate HTTP status based on health
    status_code_map = {
        HealthStatus.HEALTHY: 200,
        HealthStatus.DEGRADED: 200,
        HealthStatus.UNHEALTHY: 503,
        HealthStatus.CRITICAL: 503
    }
    
    status_code = status_code_map.get(health_report.status, 503)
    
    if status_code != 200:
        raise HTTPException(
            status_code=status_code,
            detail=f"Service is {health_report.status.value}"
        )
    
    return {
        "status": health_report.status.value,
        "service": health_report.service_name,
        "version": health_report.version,
        "uptime": health_report.uptime,
        "timestamp": health_report.timestamp.isoformat(),
        "checks": health_report.checks
    }


async def readiness_endpoint():
    """FastAPI readiness check endpoint (simpler check for K8s)"""
    monitor = get_health_monitor()
    
    # Only check critical systems for readiness
    critical_checks = {}
    for name, check in monitor.health_checks.items():
        if check.critical:
            result = await monitor._run_health_check(check)
            critical_checks[name] = result
    
    overall_status = monitor.get_overall_health_status(critical_checks)
    
    if overall_status in [HealthStatus.CRITICAL, HealthStatus.UNHEALTHY]:
        raise HTTPException(
            status_code=503,
            detail="Service not ready"
        )
    
    return {"status": "ready"}


async def liveness_endpoint():
    """FastAPI liveness check endpoint (basic health check)"""
    monitor = get_health_monitor()
    
    # Simple check - just verify the service is running
    uptime = time.time() - monitor.startup_time
    
    if uptime < 5:  # Give service time to start
        raise HTTPException(status_code=503, detail="Service starting")
    
    return {
        "status": "alive",
        "uptime": uptime
    }


import os