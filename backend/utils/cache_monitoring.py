"""
Cache Monitoring and Metrics Collection System

This module provides comprehensive monitoring for the multi-layer caching system,
including performance metrics, cost tracking, and alerting capabilities.
"""

import asyncio
import json
import logging
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import time

from prometheus_client import Counter, Histogram, Gauge, Info
import redis.asyncio as redis
from sqlalchemy import text

from backend.utils.comprehensive_cache import get_cache_manager
from backend.utils.intelligent_cache_policies import get_policy_manager
from backend.utils.database_query_cache import get_query_cache_manager
from backend.config.database import get_async_db_session
from backend.config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class CacheMetricsSnapshot:
    """Snapshot of cache metrics at a point in time"""
    timestamp: datetime
    l1_hits: int
    l1_misses: int
    l2_hits: int
    l2_misses: int
    l3_hits: int
    l3_misses: int
    total_requests: int
    hit_ratio: float
    api_calls_saved: int
    estimated_cost_savings: float
    storage_bytes: int
    active_warming_tasks: int


@dataclass
class ApiUsageMetrics:
    """API usage metrics for cost tracking"""
    provider: str
    daily_calls: int
    monthly_calls: int
    daily_limit: int
    monthly_limit: Optional[int]
    cost_per_call: float
    estimated_daily_cost: float
    estimated_monthly_cost: float
    calls_remaining: int


# Prometheus metrics
cache_requests_total = Counter(
    'cache_requests_total',
    'Total cache requests',
    ['layer', 'data_type', 'result']
)

cache_hit_ratio = Gauge(
    'cache_hit_ratio',
    'Cache hit ratio by layer',
    ['layer', 'data_type']
)

cache_response_time = Histogram(
    'cache_response_time_seconds',
    'Cache response time',
    ['layer', 'data_type']
)

cache_storage_bytes = Gauge(
    'cache_storage_bytes_total',
    'Total bytes stored in cache',
    ['layer']
)

api_calls_saved = Counter(
    'api_calls_saved_total',
    'Total API calls saved by caching',
    ['provider', 'data_type']
)

api_usage = Gauge(
    'api_usage_current',
    'Current API usage',
    ['provider', 'period']
)

cost_savings = Gauge(
    'cost_savings_usd',
    'Estimated cost savings in USD',
    ['provider', 'period']
)


class CacheMonitor:
    """
    Comprehensive cache monitoring with real-time metrics and alerting
    """
    
    def __init__(self):
        self.metrics_history: deque = deque(maxlen=1440)  # 24 hours of minute-by-minute data
        self.api_usage_tracker: Dict[str, Dict] = defaultdict(dict)
        self.alert_thresholds = {
            'low_hit_ratio': 0.7,      # Alert if hit ratio < 70%
            'high_api_usage': 0.8,     # Alert if API usage > 80% of limit
            'high_response_time': 1.0,  # Alert if response time > 1 second
            'low_storage_efficiency': 0.5  # Alert if storage efficiency < 50%
        }
        self.alert_cooldown = {}  # Prevent spam alerts
        self.monitoring_active = False
    
    async def start_monitoring(self):
        """Start the monitoring system"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        logger.info("Starting cache monitoring system")
        
        # Start monitoring tasks
        asyncio.create_task(self._collect_metrics_loop())
        asyncio.create_task(self._update_prometheus_metrics_loop())
        asyncio.create_task(self._check_alerts_loop())
        asyncio.create_task(self._cleanup_metrics_loop())
        
        logger.info("Cache monitoring system started successfully")
    
    async def stop_monitoring(self):
        """Stop the monitoring system"""
        self.monitoring_active = False
        logger.info("Cache monitoring system stopped")
    
    async def get_current_metrics(self) -> Dict[str, Any]:
        """Get current cache metrics"""
        cache_manager = await get_cache_manager()
        policy_manager = get_policy_manager()
        query_cache = get_query_cache_manager()
        
        # Get metrics from all cache layers
        cache_metrics = await cache_manager.get_metrics()
        query_metrics = query_cache.get_query_statistics()
        
        # Get API usage metrics
        api_metrics = await self._collect_api_usage_metrics()
        
        # Get Redis-specific metrics
        redis_metrics = await self._collect_redis_metrics()
        
        # Calculate efficiency metrics
        efficiency_metrics = self._calculate_efficiency_metrics(cache_metrics, api_metrics)
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'cache_performance': cache_metrics,
            'query_cache_performance': query_metrics,
            'api_usage': api_metrics,
            'redis_metrics': redis_metrics,
            'efficiency': efficiency_metrics,
            'monitoring_status': {
                'active': self.monitoring_active,
                'metrics_history_size': len(self.metrics_history),
                'last_collection': self.metrics_history[-1]['timestamp'] if self.metrics_history else None
            }
        }
    
    async def get_historical_metrics(
        self,
        hours_back: int = 24,
        granularity_minutes: int = 60
    ) -> List[Dict[str, Any]]:
        """Get historical metrics with specified granularity"""
        if not self.metrics_history:
            return []
        
        # Filter by time range
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
        filtered_metrics = [
            m for m in self.metrics_history 
            if m['timestamp'] > cutoff_time
        ]
        
        # Aggregate by granularity
        if granularity_minutes <= 1:
            return filtered_metrics
        
        aggregated = []
        bucket_size = granularity_minutes
        
        for i in range(0, len(filtered_metrics), bucket_size):
            bucket = filtered_metrics[i:i + bucket_size]
            if bucket:
                # Calculate averages for the bucket
                avg_metrics = self._aggregate_metrics_bucket(bucket)
                aggregated.append(avg_metrics)
        
        return aggregated
    
    async def get_cost_analysis(self) -> Dict[str, Any]:
        """Get comprehensive cost analysis"""
        api_metrics = await self._collect_api_usage_metrics()
        
        total_daily_cost = sum(m['estimated_daily_cost'] for m in api_metrics)
        total_monthly_cost = sum(m['estimated_monthly_cost'] for m in api_metrics)
        
        # Calculate savings from caching
        cache_manager = await get_cache_manager()
        cache_metrics = await cache_manager.get_metrics()
        api_calls_saved = cache_metrics['cache_metrics']['api_calls_saved']
        estimated_savings = cache_metrics['cache_metrics']['estimated_cost_savings']
        
        # Cost breakdown by provider
        cost_breakdown = {}
        for metrics in api_metrics:
            cost_breakdown[metrics['provider']] = {
                'daily_cost': metrics['estimated_daily_cost'],
                'monthly_cost': metrics['estimated_monthly_cost'],
                'usage_percentage': (metrics['daily_calls'] / metrics['daily_limit']) * 100 if metrics['daily_limit'] > 0 else 0
            }
        
        # Budget utilization
        monthly_budget = 50.0  # $50/month budget
        budget_utilization = (total_monthly_cost / monthly_budget) * 100
        
        return {
            'current_costs': {
                'daily_cost': total_daily_cost,
                'monthly_cost': total_monthly_cost,
                'budget_utilization_percent': budget_utilization
            },
            'savings': {
                'api_calls_saved': api_calls_saved,
                'estimated_savings': estimated_savings
            },
            'cost_breakdown': cost_breakdown,
            'budget_status': {
                'monthly_budget': monthly_budget,
                'remaining_budget': max(0, monthly_budget - total_monthly_cost),
                'on_track': total_monthly_cost <= monthly_budget
            },
            'projections': {
                'end_of_month_cost': total_monthly_cost,
                'days_remaining_in_budget': max(0, (monthly_budget - total_monthly_cost) / (total_daily_cost or 1))
            }
        }
    
    async def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        current_metrics = await self.get_current_metrics()
        historical_metrics = await self.get_historical_metrics(hours_back=24)
        cost_analysis = await self.get_cost_analysis()
        
        # Calculate performance trends
        trends = self._calculate_performance_trends(historical_metrics)
        
        # Identify top performing cache keys
        top_performers = await self._get_top_performing_cache_keys()
        
        # Generate recommendations
        recommendations = await self._generate_optimization_recommendations(current_metrics)
        
        return {
            'summary': {
                'overall_hit_ratio': current_metrics['cache_performance']['cache_metrics']['hit_ratio'],
                'total_requests': current_metrics['cache_performance']['cache_metrics']['total_requests'],
                'api_calls_saved': current_metrics['cache_performance']['cache_metrics']['api_calls_saved'],
                'cost_savings': current_metrics['cache_performance']['cache_metrics']['estimated_cost_savings']
            },
            'current_metrics': current_metrics,
            'trends': trends,
            'cost_analysis': cost_analysis,
            'top_performers': top_performers,
            'recommendations': recommendations,
            'alerts': await self._get_active_alerts()
        }
    
    # Private methods
    
    async def _collect_metrics_loop(self):
        """Continuously collect metrics"""
        while self.monitoring_active:
            try:
                metrics = await self.get_current_metrics()
                
                # Create snapshot
                cache_perf = metrics['cache_performance']['cache_metrics']
                snapshot = CacheMetricsSnapshot(
                    timestamp=datetime.utcnow(),
                    l1_hits=cache_perf['l1_hits'],
                    l1_misses=cache_perf['l1_misses'],
                    l2_hits=cache_perf['l2_hits'],
                    l2_misses=cache_perf['l2_misses'],
                    l3_hits=cache_perf['l3_hits'],
                    l3_misses=cache_perf['l3_misses'],
                    total_requests=cache_perf['total_requests'],
                    hit_ratio=cache_perf['hit_ratio'],
                    api_calls_saved=cache_perf['api_calls_saved'],
                    estimated_cost_savings=cache_perf['estimated_cost_savings'],
                    storage_bytes=metrics['cache_performance']['storage_bytes'],
                    active_warming_tasks=metrics['cache_performance']['active_warming_tasks']
                )
                
                # Add to history
                self.metrics_history.append(asdict(snapshot))
                
                # Store in database for long-term analysis
                await self._store_metrics_snapshot(snapshot)
                
                await asyncio.sleep(60)  # Collect every minute
                
            except Exception as e:
                logger.error(f"Error collecting cache metrics: {e}")
                await asyncio.sleep(60)
    
    async def _update_prometheus_metrics_loop(self):
        """Update Prometheus metrics"""
        while self.monitoring_active:
            try:
                if not self.metrics_history:
                    await asyncio.sleep(30)
                    continue
                
                latest = self.metrics_history[-1]
                
                # Update cache hit ratios
                total_requests = latest['total_requests']
                if total_requests > 0:
                    l1_ratio = latest['l1_hits'] / (latest['l1_hits'] + latest['l1_misses']) if (latest['l1_hits'] + latest['l1_misses']) > 0 else 0
                    l2_ratio = latest['l2_hits'] / (latest['l2_hits'] + latest['l2_misses']) if (latest['l2_hits'] + latest['l2_misses']) > 0 else 0
                    l3_ratio = latest['l3_hits'] / (latest['l3_hits'] + latest['l3_misses']) if (latest['l3_hits'] + latest['l3_misses']) > 0 else 0
                    
                    cache_hit_ratio.labels(layer='l1', data_type='all').set(l1_ratio)
                    cache_hit_ratio.labels(layer='l2', data_type='all').set(l2_ratio)
                    cache_hit_ratio.labels(layer='l3', data_type='all').set(l3_ratio)
                
                # Update storage metrics
                cache_storage_bytes.labels(layer='total').set(latest['storage_bytes'])
                
                # Update API usage
                api_metrics = await self._collect_api_usage_metrics()
                for metrics in api_metrics:
                    api_usage.labels(provider=metrics['provider'], period='daily').set(metrics['daily_calls'])
                    api_usage.labels(provider=metrics['provider'], period='monthly').set(metrics['monthly_calls'])
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logger.error(f"Error updating Prometheus metrics: {e}")
                await asyncio.sleep(30)
    
    async def _check_alerts_loop(self):
        """Check for alert conditions"""
        while self.monitoring_active:
            try:
                if not self.metrics_history:
                    await asyncio.sleep(300)  # Check every 5 minutes
                    continue
                
                latest = self.metrics_history[-1]
                alerts = []
                
                # Check hit ratio
                if latest['hit_ratio'] < self.alert_thresholds['low_hit_ratio']:
                    alerts.append({
                        'type': 'low_hit_ratio',
                        'severity': 'warning',
                        'message': f"Cache hit ratio is {latest['hit_ratio']:.2%}, below threshold of {self.alert_thresholds['low_hit_ratio']:.2%}",
                        'timestamp': datetime.utcnow()
                    })
                
                # Check API usage
                api_metrics = await self._collect_api_usage_metrics()
                for metrics in api_metrics:
                    usage_ratio = metrics['daily_calls'] / metrics['daily_limit'] if metrics['daily_limit'] > 0 else 0
                    if usage_ratio > self.alert_thresholds['high_api_usage']:
                        alerts.append({
                            'type': 'high_api_usage',
                            'severity': 'critical',
                            'message': f"{metrics['provider']} API usage is at {usage_ratio:.2%} of daily limit",
                            'timestamp': datetime.utcnow(),
                            'provider': metrics['provider']
                        })
                
                # Process alerts (with cooldown to prevent spam)
                for alert in alerts:
                    await self._process_alert(alert)
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error checking alerts: {e}")
                await asyncio.sleep(300)
    
    async def _cleanup_metrics_loop(self):
        """Clean up old metrics data"""
        while self.monitoring_active:
            try:
                # Clean up database metrics older than 30 days
                async with get_async_db_session() as db:
                    cutoff_date = datetime.utcnow() - timedelta(days=30)
                    await db.execute(
                        text("DELETE FROM cache_metrics WHERE created_at < :cutoff"),
                        {"cutoff": cutoff_date}
                    )
                    await db.commit()
                
                # Sleep for 24 hours
                await asyncio.sleep(86400)
                
            except Exception as e:
                logger.error(f"Error cleaning up metrics: {e}")
                await asyncio.sleep(86400)
    
    async def _collect_api_usage_metrics(self) -> List[ApiUsageMetrics]:
        """Collect API usage metrics from various providers"""
        policy_manager = get_policy_manager()
        
        providers_info = {
            'alpha_vantage': {'daily_limit': 25, 'cost_per_call': 0.10},
            'finnhub': {'daily_limit': 60 * 24, 'cost_per_call': 0.02},  # 60 per minute
            'polygon': {'daily_limit': 5 * 60 * 24, 'cost_per_call': 0.01},  # 5 per minute
            'newsapi': {'daily_limit': 1000, 'cost_per_call': 0.05}
        }
        
        metrics = []
        for provider, info in providers_info.items():
            daily_calls = 0
            monthly_calls = 0
            
            # Get usage from policy manager
            today = datetime.utcnow().strftime('%Y%m%d')
            if provider in policy_manager.api_usage_tracker:
                daily_calls = policy_manager.api_usage_tracker[provider].get(today, 0)
                
                # Calculate monthly calls (rough estimate)
                monthly_calls = sum(
                    policy_manager.api_usage_tracker[provider].values()
                )
            
            remaining_calls = max(0, info['daily_limit'] - daily_calls)
            
            metrics.append(ApiUsageMetrics(
                provider=provider,
                daily_calls=daily_calls,
                monthly_calls=monthly_calls,
                daily_limit=info['daily_limit'],
                monthly_limit=info['daily_limit'] * 30,  # Rough estimate
                cost_per_call=info['cost_per_call'],
                estimated_daily_cost=daily_calls * info['cost_per_call'],
                estimated_monthly_cost=monthly_calls * info['cost_per_call'],
                calls_remaining=remaining_calls
            ))
        
        return metrics
    
    async def _collect_redis_metrics(self) -> Dict[str, Any]:
        """Collect Redis-specific metrics"""
        try:
            cache_manager = await get_cache_manager()
            redis_client = cache_manager.redis_client
            
            if not redis_client:
                return {}
            
            info = await redis_client.info()
            
            return {
                'connected_clients': info.get('connected_clients', 0),
                'used_memory': info.get('used_memory', 0),
                'used_memory_human': info.get('used_memory_human', '0B'),
                'used_memory_peak': info.get('used_memory_peak', 0),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0),
                'total_commands_processed': info.get('total_commands_processed', 0),
                'evicted_keys': info.get('evicted_keys', 0),
                'expired_keys': info.get('expired_keys', 0)
            }
        
        except Exception as e:
            logger.warning(f"Could not collect Redis metrics: {e}")
            return {}
    
    def _calculate_efficiency_metrics(
        self,
        cache_metrics: Dict,
        api_metrics: List[ApiUsageMetrics]
    ) -> Dict[str, Any]:
        """Calculate cache efficiency metrics"""
        
        cache_perf = cache_metrics['cache_metrics']
        total_requests = cache_perf['total_requests']
        
        if total_requests == 0:
            return {
                'request_efficiency': 0.0,
                'cost_efficiency': 0.0,
                'storage_efficiency': 0.0
            }
        
        # Request efficiency (hit ratio weighted by layer speed)
        l1_weight, l2_weight, l3_weight = 1.0, 0.8, 0.6
        weighted_hits = (
            cache_perf['l1_hits'] * l1_weight +
            cache_perf['l2_hits'] * l2_weight +
            cache_perf['l3_hits'] * l3_weight
        )
        request_efficiency = weighted_hits / total_requests
        
        # Cost efficiency (savings vs potential costs)
        total_api_calls = sum(m.daily_calls for m in api_metrics)
        total_potential_cost = sum(m.estimated_daily_cost for m in api_metrics) + cache_perf['estimated_cost_savings']
        cost_efficiency = cache_perf['estimated_cost_savings'] / total_potential_cost if total_potential_cost > 0 else 0
        
        # Storage efficiency (hits per byte stored)
        storage_bytes = cache_metrics.get('storage_bytes', 1)
        storage_efficiency = (cache_perf['l1_hits'] + cache_perf['l2_hits'] + cache_perf['l3_hits']) / (storage_bytes / 1024)  # Hits per KB
        
        return {
            'request_efficiency': request_efficiency,
            'cost_efficiency': cost_efficiency,
            'storage_efficiency': storage_efficiency
        }
    
    def _calculate_performance_trends(self, historical_metrics: List[Dict]) -> Dict[str, Any]:
        """Calculate performance trends from historical data"""
        if len(historical_metrics) < 2:
            return {'trend_available': False}
        
        # Get first and last data points
        first = historical_metrics[0]
        last = historical_metrics[-1]
        
        # Calculate trends
        hit_ratio_trend = last['hit_ratio'] - first['hit_ratio']
        api_calls_trend = last['api_calls_saved'] - first['api_calls_saved']
        storage_trend = last['storage_bytes'] - first['storage_bytes']
        
        return {
            'trend_available': True,
            'time_span_hours': len(historical_metrics) / 60,  # Assuming minute granularity
            'hit_ratio_change': hit_ratio_trend,
            'api_calls_saved_change': api_calls_trend,
            'storage_change_bytes': storage_trend,
            'improvement_indicators': {
                'hit_ratio_improving': hit_ratio_trend > 0,
                'api_savings_improving': api_calls_trend > 0,
                'storage_efficient': storage_trend < api_calls_trend * 1000  # Less storage per additional API call saved
            }
        }
    
    async def _get_top_performing_cache_keys(self, limit: int = 10) -> List[Dict]:
        """Get top performing cache keys"""
        try:
            async with get_async_db_session() as db:
                result = await db.execute(
                    text("""
                        SELECT cache_key, access_count, data_size, created_at
                        FROM cache_storage
                        ORDER BY access_count DESC
                        LIMIT :limit
                    """),
                    {"limit": limit}
                )
                rows = result.fetchall()
                
                return [
                    {
                        'cache_key': row[0][:50] + '...' if len(row[0]) > 50 else row[0],
                        'access_count': row[1],
                        'data_size': row[2],
                        'efficiency': row[1] / max(row[2], 1),  # Accesses per byte
                        'created_at': row[3].isoformat()
                    }
                    for row in rows
                ]
        except Exception as e:
            logger.warning(f"Could not get top performing cache keys: {e}")
            return []
    
    async def _generate_optimization_recommendations(self, current_metrics: Dict) -> List[Dict]:
        """Generate optimization recommendations based on current metrics"""
        recommendations = []
        
        cache_perf = current_metrics['cache_performance']['cache_metrics']
        
        # Low hit ratio recommendation
        if cache_perf['hit_ratio'] < 0.7:
            recommendations.append({
                'type': 'performance',
                'priority': 'high',
                'title': 'Low Cache Hit Ratio',
                'description': f"Current hit ratio is {cache_perf['hit_ratio']:.2%}. Consider increasing TTL values or improving cache warming strategies.",
                'action': 'Review and optimize TTL policies'
            })
        
        # High API usage recommendation
        api_metrics = current_metrics['api_usage']
        for api in api_metrics:
            if api.daily_calls / api.daily_limit > 0.8:
                recommendations.append({
                    'type': 'cost',
                    'priority': 'critical',
                    'title': f'High {api.provider} API Usage',
                    'description': f"Using {api.daily_calls}/{api.daily_limit} daily calls ({api.daily_calls/api.daily_limit:.1%})",
                    'action': 'Increase caching or reduce API calls'
                })
        
        # Storage optimization recommendation
        storage_mb = cache_perf.get('storage_bytes', 0) / (1024 * 1024)
        if storage_mb > 100:  # More than 100MB
            recommendations.append({
                'type': 'storage',
                'priority': 'medium',
                'title': 'High Cache Storage Usage',
                'description': f"Cache using {storage_mb:.1f}MB of storage. Consider data compression or cleanup.",
                'action': 'Review cache cleanup policies'
            })
        
        return recommendations
    
    async def _get_active_alerts(self) -> List[Dict]:
        """Get currently active alerts"""
        # This would typically come from an alerting system
        # For now, return empty list
        return []
    
    def _aggregate_metrics_bucket(self, metrics_bucket: List[Dict]) -> Dict[str, Any]:
        """Aggregate a bucket of metrics into averages"""
        if not metrics_bucket:
            return {}
        
        # Calculate averages for numeric fields
        numeric_fields = [
            'l1_hits', 'l1_misses', 'l2_hits', 'l2_misses', 'l3_hits', 'l3_misses',
            'total_requests', 'hit_ratio', 'api_calls_saved', 'estimated_cost_savings',
            'storage_bytes', 'active_warming_tasks'
        ]
        
        aggregated = {'timestamp': metrics_bucket[-1]['timestamp']}  # Use last timestamp
        
        for field in numeric_fields:
            values = [m.get(field, 0) for m in metrics_bucket]
            aggregated[field] = sum(values) / len(values)
        
        return aggregated
    
    async def _store_metrics_snapshot(self, snapshot: CacheMetricsSnapshot):
        """Store metrics snapshot in database for long-term analysis"""
        try:
            async with get_async_db_session() as db:
                # Create table if it doesn't exist (should be done in migrations)
                await db.execute(text("""
                    CREATE TABLE IF NOT EXISTS cache_metrics (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMPTZ NOT NULL,
                        metrics_data JSONB NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """))
                
                # Insert snapshot
                await db.execute(
                    text("""
                        INSERT INTO cache_metrics (timestamp, metrics_data)
                        VALUES (:timestamp, :data)
                    """),
                    {
                        "timestamp": snapshot.timestamp,
                        "data": json.dumps(asdict(snapshot))
                    }
                )
                await db.commit()
                
        except Exception as e:
            logger.debug(f"Could not store metrics snapshot: {e}")
    
    async def _process_alert(self, alert: Dict):
        """Process an alert with cooldown logic"""
        alert_key = f"{alert['type']}_{alert.get('provider', '')}"
        current_time = time.time()
        
        # Check cooldown (prevent spam)
        if alert_key in self.alert_cooldown:
            if current_time - self.alert_cooldown[alert_key] < 1800:  # 30 minute cooldown
                return
        
        # Log alert
        if alert['severity'] == 'critical':
            logger.critical(f"CACHE ALERT: {alert['message']}")
        else:
            logger.warning(f"Cache Alert: {alert['message']}")
        
        # Set cooldown
        self.alert_cooldown[alert_key] = current_time
        
        # Here you would typically send alerts to external systems
        # (email, Slack, PagerDuty, etc.)


# Global monitor instance
_cache_monitor: Optional[CacheMonitor] = None


async def get_cache_monitor() -> CacheMonitor:
    """Get global cache monitor instance"""
    global _cache_monitor
    
    if _cache_monitor is None:
        _cache_monitor = CacheMonitor()
        await _cache_monitor.start_monitoring()
    
    return _cache_monitor


async def initialize_cache_monitoring():
    """Initialize cache monitoring system"""
    try:
        monitor = await get_cache_monitor()
        logger.info("Cache monitoring system initialized successfully")
        return monitor
    except Exception as e:
        logger.error(f"Failed to initialize cache monitoring: {e}")
        raise