"""
Comprehensive Monitoring and Metrics Collection
Provides centralized metrics collection with Prometheus integration.
"""

import time
import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import psutil
import gc

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Summary,
    Info,
    generate_latest,
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    push_to_gateway
)
from prometheus_client.core import GaugeMetricFamily, CounterMetricFamily
from fastapi import FastAPI, Response
import aiohttp

from backend.config.settings import settings

logger = logging.getLogger(__name__)

# Create custom registry
registry = CollectorRegistry()

# System Metrics
system_info = Info('system', 'System information', registry=registry)
cpu_usage = Gauge('cpu_usage_percent', 'CPU usage percentage', registry=registry)
memory_usage = Gauge('memory_usage_bytes', 'Memory usage in bytes', registry=registry)
memory_percent = Gauge('memory_usage_percent', 'Memory usage percentage', registry=registry)
disk_usage = Gauge('disk_usage_percent', 'Disk usage percentage', registry=registry)
open_connections = Gauge('open_connections', 'Number of open network connections', registry=registry)

# Application Metrics
app_info = Info('application', 'Application information', registry=registry)
uptime_seconds = Gauge('app_uptime_seconds', 'Application uptime in seconds', registry=registry)
active_tasks = Gauge('active_tasks', 'Number of active async tasks', registry=registry)
gc_collections = Counter('gc_collections_total', 'Total garbage collections', ['generation'], registry=registry)

# API Metrics
api_requests = Counter(
    'api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status', 'version'],
    registry=registry
)
api_latency = Histogram(
    'api_latency_seconds',
    'API request latency',
    ['method', 'endpoint', 'version'],
    registry=registry
)
api_errors = Counter(
    'api_errors_total',
    'Total API errors',
    ['method', 'endpoint', 'error_type'],
    registry=registry
)
concurrent_requests = Gauge('concurrent_requests', 'Current concurrent requests', registry=registry)

# Database Metrics
db_connections = Gauge('database_connections', 'Active database connections', ['database', 'state'], registry=registry)
db_query_duration = Histogram(
    'database_query_duration_seconds',
    'Database query duration',
    ['operation', 'table'],
    registry=registry
)
db_errors = Counter(
    'database_errors_total',
    'Database errors',
    ['operation', 'error_type'],
    registry=registry
)
db_pool_size = Gauge('database_pool_size', 'Database connection pool size', ['pool'], registry=registry)
db_pool_overflow = Gauge('database_pool_overflow', 'Database pool overflow', ['pool'], registry=registry)

# Cache Metrics
cache_hits = Counter('cache_hits_total', 'Cache hits', ['cache_type'], registry=registry)
cache_misses = Counter('cache_misses_total', 'Cache misses', ['cache_type'], registry=registry)
cache_evictions = Counter('cache_evictions_total', 'Cache evictions', ['cache_type'], registry=registry)
cache_size = Gauge('cache_size_entries', 'Cache size in entries', ['cache_type'], registry=registry)
cache_memory = Gauge('cache_memory_bytes', 'Cache memory usage', ['cache_type'], registry=registry)

# Stock Processing Metrics
stocks_processed = Counter(
    'stocks_processed_total',
    'Total stocks processed',
    ['tier', 'data_type'],
    registry=registry
)
stock_processing_duration = Histogram(
    'stock_processing_duration_seconds',
    'Stock processing duration',
    ['tier'],
    registry=registry
)
stock_errors = Counter(
    'stock_processing_errors_total',
    'Stock processing errors',
    ['tier', 'error_type'],
    registry=registry
)
stocks_in_tier = Gauge('stocks_in_tier', 'Number of stocks in each tier', ['tier'], registry=registry)

# Cost Metrics
api_calls_remaining = Gauge(
    'api_calls_remaining',
    'API calls remaining in quota',
    ['provider', 'limit_type'],
    registry=registry
)
estimated_monthly_cost = Gauge('estimated_monthly_cost_usd', 'Estimated monthly cost in USD', registry=registry)
daily_api_calls = Counter(
    'daily_api_calls_total',
    'Daily API calls',
    ['provider'],
    registry=registry
)
cost_saving_mode = Gauge('cost_saving_mode_active', 'Cost saving mode active (1) or not (0)', registry=registry)

# Redis Metrics
redis_connections = Gauge('redis_connections', 'Redis connections', ['state'], registry=registry)
redis_memory = Gauge('redis_memory_bytes', 'Redis memory usage', registry=registry)
redis_commands = Counter('redis_commands_total', 'Redis commands executed', ['command'], registry=registry)
redis_errors = Counter('redis_errors_total', 'Redis errors', ['error_type'], registry=registry)
circuit_breaker_state = Gauge(
    'circuit_breaker_state',
    'Circuit breaker state (0=closed, 1=open, 2=half-open)',
    ['service'],
    registry=registry
)

# ML Model Metrics
model_predictions = Counter(
    'model_predictions_total',
    'Model predictions',
    ['model_name', 'model_version'],
    registry=registry
)
model_inference_time = Histogram(
    'model_inference_seconds',
    'Model inference time',
    ['model_name'],
    registry=registry
)
model_accuracy = Gauge('model_accuracy', 'Model accuracy score', ['model_name'], registry=registry)
model_training_time = Gauge('model_training_seconds', 'Model training time', ['model_name'], registry=registry)

# Financial Performance Metrics
portfolio_value = Gauge('portfolio_value_usd', 'Portfolio value in USD', ['user_id'], registry=registry)
portfolio_return = Gauge('portfolio_return_percent', 'Portfolio return percentage', ['user_id', 'period'], registry=registry)
recommendation_accuracy = Gauge(
    'recommendation_accuracy_percent',
    'Recommendation accuracy percentage',
    ['model', 'time_horizon'],
    registry=registry
)
trade_success_rate = Gauge(
    'trade_success_rate_percent',
    'Trade success rate percentage',
    ['strategy', 'tier'],
    registry=registry
)
alpha_generated = Gauge('alpha_generated_percent', 'Alpha generated by strategy', ['strategy'], registry=registry)
sharpe_ratio = Gauge('sharpe_ratio', 'Sharpe ratio', ['strategy'], registry=registry)
max_drawdown = Gauge('max_drawdown_percent', 'Maximum drawdown percentage', ['strategy'], registry=registry)

# Data Quality Metrics
data_quality_score = Gauge('data_quality_score', 'Data quality score 0-100', ['data_type'], registry=registry)
data_staleness = Gauge('data_staleness_seconds', 'Data staleness in seconds', ['data_type', 'provider'], registry=registry)
data_missing_percent = Gauge('data_missing_percent', 'Missing data percentage', ['data_type'], registry=registry)
data_anomalies = Counter('data_anomalies_total', 'Data anomalies detected', ['anomaly_type'], registry=registry)

# Resource Utilization Metrics
thread_pool_active = Gauge('thread_pool_active', 'Active threads in pool', ['pool_name'], registry=registry)
thread_pool_queued = Gauge('thread_pool_queued', 'Queued tasks in thread pool', ['pool_name'], registry=registry)
file_descriptors = Gauge('file_descriptors_open', 'Open file descriptors', registry=registry)
network_bytes_sent = Counter('network_bytes_sent_total', 'Network bytes sent', ['interface'], registry=registry)
network_bytes_received = Counter('network_bytes_received_total', 'Network bytes received', ['interface'], registry=registry)

# Business Logic Metrics
stocks_analyzed_per_hour = Gauge('stocks_analyzed_per_hour', 'Stocks analyzed per hour', ['tier'], registry=registry)
recommendations_generated = Counter('recommendations_generated_total', 'Recommendations generated', ['type'], registry=registry)
user_sessions = Gauge('user_sessions_active', 'Active user sessions', registry=registry)
api_quota_usage = Gauge('api_quota_usage_percent', 'API quota usage percentage', ['provider'], registry=registry)

# Alerting and SLA Metrics  
sla_compliance = Gauge('sla_compliance_percent', 'SLA compliance percentage', ['service'], registry=registry)
alert_notifications_sent = Counter('alert_notifications_total', 'Alert notifications sent', ['channel', 'severity'], registry=registry)
incident_duration = Histogram('incident_duration_seconds', 'Incident duration', ['severity'], registry=registry)
mttr = Gauge('mttr_minutes', 'Mean Time To Recovery in minutes', ['service'], registry=registry)
mtbf = Gauge('mtbf_hours', 'Mean Time Between Failures in hours', ['service'], registry=registry)


class MetricsCollector:
    """
    Centralized metrics collection and monitoring.
    """
    
    def __init__(self):
        """Initialize metrics collector."""
        self.start_time = time.time()
        self._collection_interval = 10  # seconds
        self._collection_task: Optional[asyncio.Task] = None
        
        # Initialize system info
        system_info.info({
            'platform': psutil.os.name,
            'python_version': psutil.sys.version,
            'cpu_count': str(psutil.cpu_count()),
            'total_memory': str(psutil.virtual_memory().total)
        })
        
        # Initialize app info
        app_info.info({
            'version': settings.VERSION,
            'environment': settings.ENVIRONMENT,
            'debug': str(settings.DEBUG)
        })
    
    async def start_collection(self) -> None:
        """Start background metrics collection."""
        if not self._collection_task:
            self._collection_task = asyncio.create_task(self._collect_metrics_loop())
            logger.info("Started metrics collection")
    
    async def stop_collection(self) -> None:
        """Stop metrics collection."""
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
            logger.info("Stopped metrics collection")
    
    async def _collect_metrics_loop(self) -> None:
        """Background loop for collecting metrics."""
        while True:
            try:
                await self.collect_system_metrics()
                await self.collect_application_metrics()
                await self.collect_database_metrics()
                await self.collect_cache_metrics()
                await self.collect_cost_metrics()
                await self.collect_redis_metrics()
                await self.collect_financial_metrics()
                await self.collect_data_quality_metrics()
                await self.collect_business_metrics()
                await asyncio.sleep(self._collection_interval)
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
                await asyncio.sleep(self._collection_interval)
    
    async def collect_system_metrics(self) -> None:
        """Collect system-level metrics."""
        try:
            # CPU metrics
            cpu_usage.set(psutil.cpu_percent(interval=1))
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_usage.set(memory.used)
            memory_percent.set(memory.percent)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_usage.set(disk.percent)
            
            # Network connections
            connections = len(psutil.net_connections())
            open_connections.set(connections)
            
            # File descriptors
            try:
                process = psutil.Process()
                file_descriptors.set(process.num_fds())
            except (psutil.AccessDenied, AttributeError):
                pass
            
            # Network I/O
            net_io = psutil.net_io_counters()
            if net_io:
                network_bytes_sent.labels(interface='total')._value._value = net_io.bytes_sent
                network_bytes_received.labels(interface='total')._value._value = net_io.bytes_recv
            
            # Thread pool metrics
            import concurrent.futures
            import threading
            thread_count = threading.active_count()
            thread_pool_active.labels(pool_name='main').set(thread_count)
            
            # Uptime
            uptime = time.time() - self.start_time
            uptime_seconds.set(uptime)
            
            # Garbage collection
            for i, count in enumerate(gc.get_count()):
                gc_collections.labels(generation=str(i)).inc(count)
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    async def collect_application_metrics(self) -> None:
        """Collect application-level metrics."""
        try:
            # Active async tasks
            tasks = len(asyncio.all_tasks())
            active_tasks.set(tasks)
            
        except Exception as e:
            logger.error(f"Error collecting application metrics: {e}")
    
    async def collect_database_metrics(self) -> None:
        """Collect database metrics."""
        try:
            from backend.utils.async_database import async_db_manager
            
            if async_db_manager._engine:
                pool = async_db_manager._engine.pool
                if pool:
                    db_pool_size.labels(pool='main').set(pool.size())
                    db_pool_overflow.labels(pool='main').set(pool.overflow())
                    db_connections.labels(database='main', state='active').set(pool.checked_out())
                    db_connections.labels(database='main', state='idle').set(pool.size() - pool.checked_out())
            
            # Read replica metrics
            from backend.utils.db_read_replicas import replica_manager
            
            if replica_manager._initialized:
                metrics = replica_manager.get_metrics()
                if metrics:
                    summary = metrics.get('summary', {})
                    db_connections.labels(database='replicas', state='healthy').set(
                        summary.get('healthy_replicas', 0)
                    )
                    db_connections.labels(database='replicas', state='unhealthy').set(
                        summary.get('unhealthy_replicas', 0)
                    )
            
        except Exception as e:
            logger.error(f"Error collecting database metrics: {e}")
    
    async def collect_cache_metrics(self) -> None:
        """Collect cache metrics."""
        try:
            from backend.utils.query_cache import query_cache
            from backend.utils.bounded_cache import bounded_cache, fallback_cache
            
            # Query cache metrics
            if query_cache:
                stats = query_cache.get_stats()
                cache_hits.labels(cache_type='query').inc(stats.get('hits', 0))
                cache_misses.labels(cache_type='query').inc(stats.get('misses', 0))
                cache_evictions.labels(cache_type='query').inc(stats.get('evictions', 0))
            
            # Bounded cache metrics
            if bounded_cache:
                metrics = bounded_cache.get_metrics()
                cache_size.labels(cache_type='bounded').set(metrics.get('current_size', 0))
                cache_memory.labels(cache_type='bounded').set(metrics.get('size_bytes', 0))
            
            # Fallback cache metrics
            if fallback_cache:
                metrics = fallback_cache.get_metrics()
                cache_size.labels(cache_type='fallback').set(metrics.get('current_size', 0))
            
        except Exception as e:
            logger.error(f"Error collecting cache metrics: {e}")
    
    async def collect_cost_metrics(self) -> None:
        """Collect cost and API usage metrics."""
        try:
            from backend.utils.persistent_cost_monitor import PersistentCostMonitor
            
            monitor = PersistentCostMonitor()
            
            # Get usage report
            report = await monitor.get_usage_report()
            
            if report:
                # Monthly cost
                monthly = report.get('monthly', {})
                estimated_monthly_cost.set(monthly.get('total_cost', 0))
                
                # Cost saving mode
                in_emergency = await monitor.is_in_emergency_mode()
                cost_saving_mode.set(1 if in_emergency else 0)
                
                # API limits
                limits = report.get('limits_remaining', {})
                for provider, remaining in limits.items():
                    if isinstance(remaining, dict):
                        for limit_type, value in remaining.items():
                            api_calls_remaining.labels(
                                provider=provider,
                                limit_type=limit_type
                            ).set(value)
            
        except Exception as e:
            logger.error(f"Error collecting cost metrics: {e}")
    
    async def collect_redis_metrics(self) -> None:
        """Collect Redis metrics."""
        try:
            from backend.utils.redis_resilience import resilient_redis
            
            if resilient_redis:
                health = resilient_redis.get_health_status()
                
                # Circuit breaker state
                circuit_state = health.get('circuit_state', 'closed')
                state_map = {'closed': 0, 'open': 1, 'half_open': 2}
                circuit_breaker_state.labels(service='redis').set(
                    state_map.get(circuit_state, 0)
                )
                
                # Connection metrics
                conn_metrics = health.get('connection_metrics', {})
                redis_connections.labels(state='active').set(
                    conn_metrics.get('connections', 0) - conn_metrics.get('reconnections', 0)
                )
                redis_connections.labels(state='failed').set(
                    conn_metrics.get('failures', 0)
                )
            
        except Exception as e:
            logger.error(f"Error collecting Redis metrics: {e}")
    
    async def collect_financial_metrics(self) -> None:
        """Collect financial performance metrics."""
        try:
            from backend.analytics.recommendation_engine import RecommendationEngine
            from backend.repositories.portfolio_repository import PortfolioRepository
            
            # Portfolio performance metrics
            portfolio_repo = PortfolioRepository()
            portfolios = await portfolio_repo.get_active_portfolios(limit=100)
            
            for portfolio in portfolios:
                # Portfolio value
                total_value = await portfolio_repo.get_portfolio_value(portfolio.id)
                if total_value:
                    portfolio_value.labels(user_id=str(portfolio.user_id)).set(total_value)
                
                # Portfolio returns
                returns = await portfolio_repo.get_portfolio_returns(portfolio.id)
                if returns:
                    for period, return_pct in returns.items():
                        portfolio_return.labels(
                            user_id=str(portfolio.user_id),
                            period=period
                        ).set(return_pct)
            
            # Recommendation accuracy tracking
            engine = RecommendationEngine()
            accuracy_metrics = await engine.get_accuracy_metrics()
            
            if accuracy_metrics:
                for model_name, metrics in accuracy_metrics.items():
                    for horizon, accuracy in metrics.items():
                        recommendation_accuracy.labels(
                            model=model_name,
                            time_horizon=horizon
                        ).set(accuracy * 100)  # Convert to percentage
            
        except Exception as e:
            logger.error(f"Error collecting financial metrics: {e}")
    
    async def collect_data_quality_metrics(self) -> None:
        """Collect data quality metrics."""
        try:
            from backend.utils.data_quality import DataQualityChecker
            from backend.monitoring.data_quality_metrics import get_quality_summary
            
            # Data quality scores by type
            quality_summary = get_quality_summary()
            
            if quality_summary:
                for data_type, metrics in quality_summary.items():
                    data_quality_score.labels(data_type=data_type).set(
                        metrics.get('quality_score', 0)
                    )
                    data_missing_percent.labels(data_type=data_type).set(
                        metrics.get('missing_percent', 0)
                    )
                    
                    # Data staleness
                    if 'staleness' in metrics:
                        for provider, staleness_sec in metrics['staleness'].items():
                            data_staleness.labels(
                                data_type=data_type,
                                provider=provider
                            ).set(staleness_sec)
            
            # Anomaly detection results
            checker = DataQualityChecker()
            anomalies = await checker.get_recent_anomalies(hours=1)
            
            for anomaly in anomalies:
                data_anomalies.labels(anomaly_type=anomaly.get('type', 'unknown')).inc()
            
        except Exception as e:
            logger.error(f"Error collecting data quality metrics: {e}")
    
    async def collect_business_metrics(self) -> None:
        """Collect business logic metrics."""
        try:
            from backend.utils.persistent_cost_monitor import PersistentCostMonitor
            from backend.repositories.stock_repository import StockRepository
            
            # Stock processing rates
            stock_repo = StockRepository()
            processing_stats = await stock_repo.get_processing_stats()
            
            if processing_stats:
                for tier, stats in processing_stats.items():
                    stocks_analyzed_per_hour.labels(tier=tier).set(
                        stats.get('per_hour', 0)
                    )
            
            # API quota usage
            cost_monitor = PersistentCostMonitor()
            usage_report = await cost_monitor.get_usage_report()
            
            if usage_report and 'api_usage' in usage_report:
                for provider, usage in usage_report['api_usage'].items():
                    quota_pct = usage.get('quota_usage_percent', 0)
                    api_quota_usage.labels(provider=provider).set(quota_pct)
            
            # User sessions (approximate from active connections)
            try:
                from backend.utils.async_database import async_db_manager
                if async_db_manager._engine:
                    pool = async_db_manager._engine.pool
                    active_connections = pool.checked_out() if pool else 0
                    # Rough estimate: 1 session per 2 connections on average
                    estimated_sessions = max(1, active_connections // 2)
                    user_sessions.set(estimated_sessions)
            except Exception:
                pass
            
        except Exception as e:
            logger.error(f"Error collecting business metrics: {e}")
    
    def record_api_request(
        self,
        method: str,
        endpoint: str,
        status: int,
        duration: float,
        version: str = "v3"
    ) -> None:
        """Record API request metrics."""
        api_requests.labels(
            method=method,
            endpoint=endpoint,
            status=str(status),
            version=version
        ).inc()
        
        api_latency.labels(
            method=method,
            endpoint=endpoint,
            version=version
        ).observe(duration)
        
        if status >= 400:
            error_type = 'client_error' if status < 500 else 'server_error'
            api_errors.labels(
                method=method,
                endpoint=endpoint,
                error_type=error_type
            ).inc()
    
    def record_stock_processing(
        self,
        tier: str,
        data_type: str,
        duration: float,
        success: bool = True
    ) -> None:
        """Record stock processing metrics."""
        stocks_processed.labels(tier=tier, data_type=data_type).inc()
        stock_processing_duration.labels(tier=tier).observe(duration)
        
        if not success:
            stock_errors.labels(tier=tier, error_type='processing_failed').inc()
    
    def record_model_prediction(
        self,
        model_name: str,
        model_version: str,
        inference_time: float
    ) -> None:
        """Record ML model prediction metrics."""
        model_predictions.labels(
            model_name=model_name,
            model_version=model_version
        ).inc()
        
        model_inference_time.labels(model_name=model_name).observe(inference_time)
    
    def record_financial_performance(
        self,
        strategy: str,
        alpha: float,
        sharpe: float,
        max_dd: float,
        success_rate: float,
        tier: str = "all"
    ) -> None:
        """Record financial performance metrics."""
        alpha_generated.labels(strategy=strategy).set(alpha)
        sharpe_ratio.labels(strategy=strategy).set(sharpe)
        max_drawdown.labels(strategy=strategy).set(max_dd)
        trade_success_rate.labels(strategy=strategy, tier=tier).set(success_rate)
    
    def record_recommendation_generated(self, recommendation_type: str) -> None:
        """Record recommendation generation."""
        recommendations_generated.labels(type=recommendation_type).inc()
    
    def record_data_quality(self, data_type: str, quality_score: float) -> None:
        """Record data quality score."""
        data_quality_score.labels(data_type=data_type).set(quality_score)
    
    def record_alert_sent(self, channel: str, severity: str) -> None:
        """Record alert notification sent."""
        alert_notifications_sent.labels(channel=channel, severity=severity).inc()
    
    def record_sla_compliance(self, service: str, compliance_pct: float) -> None:
        """Record SLA compliance percentage."""
        sla_compliance.labels(service=service).set(compliance_pct)
    
    def record_incident(self, severity: str, duration_seconds: float) -> None:
        """Record incident metrics."""
        incident_duration.labels(severity=severity).observe(duration_seconds)
    
    def update_mttr_mtbf(self, service: str, mttr_minutes: float, mtbf_hours: float) -> None:
        """Update MTTR and MTBF metrics."""
        mttr.labels(service=service).set(mttr_minutes)
        mtbf.labels(service=service).set(mtbf_hours)
    
    def get_metrics(self) -> bytes:
        """Get metrics in Prometheus format."""
        return generate_latest(registry)


# Global metrics collector
metrics_collector = MetricsCollector()


# FastAPI integration
def setup_metrics_endpoint(app: FastAPI):
    """Setup /metrics endpoint for Prometheus scraping."""
    
    @app.get("/metrics")
    async def get_metrics():
        """Prometheus metrics endpoint."""
        metrics = metrics_collector.get_metrics()
        return Response(content=metrics, media_type=CONTENT_TYPE_LATEST)
    
    @app.on_event("startup")
    async def startup_metrics():
        """Start metrics collection on app startup."""
        await metrics_collector.start_collection()
    
    @app.on_event("shutdown")
    async def shutdown_metrics():
        """Stop metrics collection on app shutdown."""
        await metrics_collector.stop_collection()


# Middleware for request metrics
@asynccontextmanager
async def track_request(method: str, endpoint: str, version: str = "v3"):
    """Context manager for tracking request metrics."""
    start_time = time.time()
    concurrent_requests.inc()
    
    try:
        yield
        status = 200
    except Exception as e:
        status = 500
        raise
    finally:
        duration = time.time() - start_time
        concurrent_requests.dec()
        metrics_collector.record_api_request(method, endpoint, status, duration, version)