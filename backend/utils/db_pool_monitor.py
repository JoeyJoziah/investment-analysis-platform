"""
Database Connection Pool Monitoring and Validation
"""

import logging
import time
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from threading import Thread, Lock
import psutil
from sqlalchemy import text, create_engine, pool
from sqlalchemy.pool import QueuePool, NullPool, StaticPool
from backend.config.settings import settings

logger = logging.getLogger(__name__)

class DatabasePoolMonitor:
    """Monitor and validate database connection pool health"""
    
    def __init__(self, engine):
        self.engine = engine
        self.metrics = {
            "total_connections": 0,
            "active_connections": 0,
            "idle_connections": 0,
            "overflow_connections": 0,
            "connection_wait_time": [],
            "connection_errors": 0,
            "last_error": None,
            "pool_exhausted_count": 0,
            "slow_queries": 0,
            "deadlocks": 0
        }
        self.lock = Lock()
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self, interval: int = 30):
        """Start background monitoring thread"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = Thread(target=self._monitor_loop, args=(interval,))
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            logger.info(f"Database pool monitoring started with {interval}s interval")
    
    def stop_monitoring(self):
        """Stop monitoring thread"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
            logger.info("Database pool monitoring stopped")
    
    def _monitor_loop(self, interval: int):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                self.collect_metrics()
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Error in pool monitoring loop: {e}")
                time.sleep(interval)
    
    def collect_metrics(self) -> Dict[str, Any]:
        """Collect current pool metrics"""
        with self.lock:
            try:
                pool = self.engine.pool
                
                # Basic pool stats
                self.metrics["total_connections"] = pool.size() if hasattr(pool, 'size') else 0
                self.metrics["active_connections"] = pool.checkedout() if hasattr(pool, 'checkedout') else 0
                self.metrics["idle_connections"] = pool.checkedin() if hasattr(pool, 'checkedin') else 0
                self.metrics["overflow_connections"] = pool.overflow() if hasattr(pool, 'overflow') else 0
                
                # Calculate pool utilization
                if self.metrics["total_connections"] > 0:
                    self.metrics["pool_utilization"] = (
                        self.metrics["active_connections"] / self.metrics["total_connections"]
                    ) * 100
                else:
                    self.metrics["pool_utilization"] = 0
                
                # Test connection health
                self._test_connection_health()
                
                # Check for connection leaks
                self._check_connection_leaks()
                
                # Monitor slow queries
                self._monitor_slow_queries()
                
                self.metrics["last_check"] = datetime.utcnow().isoformat()
                
            except Exception as e:
                logger.error(f"Error collecting pool metrics: {e}")
                self.metrics["connection_errors"] += 1
                self.metrics["last_error"] = str(e)
        
        return self.metrics.copy()
    
    def _test_connection_health(self):
        """Test if connections are healthy"""
        try:
            start_time = time.time()
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                result.fetchone()
            
            connection_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Track connection wait times
            self.metrics["connection_wait_time"].append(connection_time)
            if len(self.metrics["connection_wait_time"]) > 100:
                self.metrics["connection_wait_time"] = self.metrics["connection_wait_time"][-100:]
            
            # Calculate average wait time
            if self.metrics["connection_wait_time"]:
                self.metrics["avg_connection_time_ms"] = sum(self.metrics["connection_wait_time"]) / len(self.metrics["connection_wait_time"])
            
            # Flag slow connections
            if connection_time > 1000:  # More than 1 second
                logger.warning(f"Slow database connection: {connection_time:.2f}ms")
                self.metrics["slow_connections"] = self.metrics.get("slow_connections", 0) + 1
                
        except Exception as e:
            logger.error(f"Connection health check failed: {e}")
            self.metrics["connection_errors"] += 1
            self.metrics["last_error"] = str(e)
    
    def _check_connection_leaks(self):
        """Check for potential connection leaks"""
        try:
            # Get process info
            process = psutil.Process()
            connections = process.connections(kind='tcp')
            
            # Count database connections
            db_connections = 0
            for conn in connections:
                if conn.status == 'ESTABLISHED':
                    # Check if it's a database connection (default PostgreSQL port)
                    if conn.raddr and conn.raddr.port == 5432:
                        db_connections += 1
            
            self.metrics["system_db_connections"] = db_connections
            
            # Check for leak indicators
            pool = self.engine.pool
            expected_connections = pool.size() + pool.overflow() if hasattr(pool, 'overflow') else pool.size()
            
            if db_connections > expected_connections * 1.5:
                logger.warning(f"Potential connection leak detected: {db_connections} system connections vs {expected_connections} expected")
                self.metrics["potential_leak"] = True
            else:
                self.metrics["potential_leak"] = False
                
        except Exception as e:
            logger.debug(f"Could not check for connection leaks: {e}")
    
    def _monitor_slow_queries(self):
        """Monitor for slow queries"""
        try:
            with self.engine.connect() as conn:
                # Check PostgreSQL slow query stats if available
                result = conn.execute(text("""
                    SELECT COUNT(*) as slow_count
                    FROM pg_stat_activity
                    WHERE state = 'active'
                    AND query_start < NOW() - INTERVAL '5 seconds'
                    AND query NOT LIKE '%pg_stat_activity%'
                """))
                slow_count = result.scalar()
                
                if slow_count:
                    self.metrics["slow_queries"] = slow_count
                    logger.warning(f"Found {slow_count} slow queries running")
                    
        except Exception as e:
            logger.debug(f"Could not monitor slow queries: {e}")
    
    def validate_pool_configuration(self) -> Dict[str, Any]:
        """Validate pool configuration against best practices"""
        validation_results = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "recommendations": []
        }
        
        pool = self.engine.pool
        
        # Check pool size
        if hasattr(pool, 'size'):
            pool_size = pool.size()
            if pool_size < 5:
                validation_results["warnings"].append(f"Pool size ({pool_size}) is very small, may cause connection starvation")
            elif pool_size > 100:
                validation_results["warnings"].append(f"Pool size ({pool_size}) is very large, may waste resources")
            
            # Recommended pool size based on CPU cores
            cpu_count = psutil.cpu_count()
            recommended_pool_size = cpu_count * 4
            if abs(pool_size - recommended_pool_size) > recommended_pool_size * 0.5:
                validation_results["recommendations"].append(
                    f"Consider setting pool size to {recommended_pool_size} (4x CPU cores)"
                )
        
        # Check overflow
        if hasattr(pool, 'max_overflow'):
            max_overflow = pool.max_overflow
            if max_overflow < 0:
                validation_results["recommendations"].append("Consider enabling overflow for handling traffic spikes")
            elif max_overflow > pool.size() * 2:
                validation_results["warnings"].append("Max overflow is very high, may cause resource issues")
        
        # Check timeout settings
        if hasattr(pool, 'timeout'):
            timeout = pool.timeout
            if timeout < 10:
                validation_results["warnings"].append(f"Pool timeout ({timeout}s) is very short, may cause unnecessary errors")
            elif timeout > 60:
                validation_results["warnings"].append(f"Pool timeout ({timeout}s) is very long, may hide connection issues")
        
        # Check recycle time
        if hasattr(pool, '_recycle'):
            recycle = pool._recycle
            if recycle and recycle < 300:
                validation_results["warnings"].append("Connection recycle time is very short, may cause overhead")
            elif not recycle or recycle > 7200:
                validation_results["recommendations"].append("Consider setting connection recycle to 1-2 hours")
        
        # Check for NullPool (not recommended for production)
        if isinstance(pool, NullPool):
            validation_results["errors"].append("NullPool is not recommended for production use")
            validation_results["valid"] = False
        
        # Check current utilization
        metrics = self.collect_metrics()
        if metrics.get("pool_utilization", 0) > 80:
            validation_results["warnings"].append(f"High pool utilization ({metrics['pool_utilization']:.1f}%)")
        
        if metrics.get("connection_errors", 0) > 10:
            validation_results["errors"].append(f"High number of connection errors ({metrics['connection_errors']})")
            validation_results["valid"] = False
        
        return validation_results
    
    def get_pool_recommendations(self) -> Dict[str, Any]:
        """Get recommendations for pool optimization"""
        metrics = self.collect_metrics()
        recommendations = {
            "current_config": {},
            "recommended_config": {},
            "reasoning": []
        }
        
        pool = self.engine.pool
        
        # Current configuration
        if hasattr(pool, 'size'):
            recommendations["current_config"]["pool_size"] = pool.size()
        if hasattr(pool, 'max_overflow'):
            recommendations["current_config"]["max_overflow"] = pool.max_overflow
        if hasattr(pool, 'timeout'):
            recommendations["current_config"]["timeout"] = pool.timeout
        if hasattr(pool, '_recycle'):
            recommendations["current_config"]["recycle"] = pool._recycle
        
        # Calculate recommendations
        cpu_count = psutil.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Pool size recommendation
        if metrics.get("pool_utilization", 0) > 70:
            recommended_size = min(pool.size() * 1.5, cpu_count * 6)
            recommendations["recommended_config"]["pool_size"] = int(recommended_size)
            recommendations["reasoning"].append("High utilization suggests increasing pool size")
        else:
            recommendations["recommended_config"]["pool_size"] = cpu_count * 4
        
        # Overflow recommendation
        recommendations["recommended_config"]["max_overflow"] = int(recommendations["recommended_config"]["pool_size"] * 0.5)
        recommendations["reasoning"].append("Overflow set to 50% of pool size for burst handling")
        
        # Timeout recommendation
        avg_conn_time = metrics.get("avg_connection_time_ms", 100)
        if avg_conn_time > 500:
            recommendations["recommended_config"]["timeout"] = 30
            recommendations["reasoning"].append("High connection time suggests longer timeout")
        else:
            recommendations["recommended_config"]["timeout"] = 20
        
        # Recycle recommendation
        recommendations["recommended_config"]["recycle"] = 3600
        recommendations["reasoning"].append("1-hour recycle prevents stale connections")
        
        # Pre-ping recommendation
        recommendations["recommended_config"]["pre_ping"] = True
        recommendations["reasoning"].append("Pre-ping ensures connection validity before use")
        
        return recommendations
    
    def reset_metrics(self):
        """Reset all collected metrics"""
        with self.lock:
            self.metrics = {
                "total_connections": 0,
                "active_connections": 0,
                "idle_connections": 0,
                "overflow_connections": 0,
                "connection_wait_time": [],
                "connection_errors": 0,
                "last_error": None,
                "pool_exhausted_count": 0,
                "slow_queries": 0,
                "deadlocks": 0
            }
            logger.info("Pool metrics reset")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status of the connection pool"""
        metrics = self.collect_metrics()
        validation = self.validate_pool_configuration()
        
        health = {
            "status": "healthy",
            "metrics": metrics,
            "validation": validation,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Determine overall health
        if not validation["valid"] or metrics.get("connection_errors", 0) > 10:
            health["status"] = "unhealthy"
        elif validation["warnings"] or metrics.get("pool_utilization", 0) > 80:
            health["status"] = "degraded"
        
        return health

# Global pool monitor instance
_pool_monitor: Optional[DatabasePoolMonitor] = None

def get_pool_monitor(engine) -> DatabasePoolMonitor:
    """Get or create the global pool monitor instance"""
    global _pool_monitor
    if _pool_monitor is None:
        _pool_monitor = DatabasePoolMonitor(engine)
        _pool_monitor.start_monitoring()
    return _pool_monitor

def validate_and_optimize_pool(engine) -> Dict[str, Any]:
    """Validate and optimize database connection pool"""
    monitor = get_pool_monitor(engine)
    
    # Get current status
    health = monitor.get_health_status()
    recommendations = monitor.get_pool_recommendations()
    
    result = {
        "health": health,
        "recommendations": recommendations,
        "action_required": health["status"] != "healthy"
    }
    
    # Log warnings if needed
    if health["status"] == "unhealthy":
        logger.error(f"Database pool is unhealthy: {health['validation']['errors']}")
    elif health["status"] == "degraded":
        logger.warning(f"Database pool is degraded: {health['validation']['warnings']}")
    
    return result