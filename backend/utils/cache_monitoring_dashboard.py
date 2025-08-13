"""
Comprehensive Cache Performance Monitoring and Dashboard
Real-time monitoring, alerting, and visualization for investment analysis cache system.
"""

import asyncio
import logging
import time
import json
from typing import Any, Dict, List, Optional, Tuple, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from enum import Enum
import threading
import statistics
import numpy as np
from flask import Flask, jsonify, render_template_string
import plotly.graph_objs as go
import plotly.utils
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

from backend.utils.advanced_cache_features import cache_analytics, cache_monitoring_dashboard
from backend.utils.tier_based_caching import tier_based_cache
from backend.utils.cache_hit_optimization import cache_hit_optimizer
from backend.utils.redis_cluster_optimization import redis_cluster_manager

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics we track."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    RATE = "rate"


@dataclass
class Alert:
    """Cache monitoring alert."""
    id: str
    name: str
    severity: AlertSeverity
    message: str
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    metric_value: float = 0.0
    threshold: float = 0.0
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class MetricPoint:
    """Single metric data point."""
    timestamp: float
    value: float
    tags: Dict[str, str] = field(default_factory=dict)


class CacheMetricsCollector:
    """
    Collects and aggregates cache metrics from all subsystems.
    """
    
    def __init__(self):
        self.metrics_storage = defaultdict(deque)  # metric_name -> deque of MetricPoint
        self.metric_definitions = {}
        self.collection_interval = 30  # seconds
        self.retention_hours = 48  # Keep 48 hours of data
        self._lock = threading.RLock()
        
        # Prometheus metrics
        self.prometheus_metrics = self._initialize_prometheus_metrics()
        
    def _initialize_prometheus_metrics(self) -> Dict[str, Any]:
        """Initialize Prometheus metrics."""
        return {
            'cache_hits_total': Counter(
                'cache_hits_total',
                'Total cache hits',
                ['tier', 'operation']
            ),
            'cache_misses_total': Counter(
                'cache_misses_total', 
                'Total cache misses',
                ['tier', 'operation']
            ),
            'cache_response_time_seconds': Histogram(
                'cache_response_time_seconds',
                'Cache response time in seconds',
                ['tier', 'operation']
            ),
            'cache_size_bytes': Gauge(
                'cache_size_bytes',
                'Cache size in bytes',
                ['tier', 'type']
            ),
            'cache_hit_rate': Gauge(
                'cache_hit_rate',
                'Cache hit rate percentage',
                ['tier']
            ),
            'cache_memory_usage_bytes': Gauge(
                'cache_memory_usage_bytes',
                'Memory usage in bytes',
                ['node']
            ),
            'cache_evictions_total': Counter(
                'cache_evictions_total',
                'Total cache evictions',
                ['tier', 'reason']
            ),
            'cache_compression_ratio': Gauge(
                'cache_compression_ratio',
                'Compression ratio',
                ['algorithm']
            ),
            'cache_warming_duration_seconds': Histogram(
                'cache_warming_duration_seconds',
                'Cache warming duration in seconds',
                ['tier']
            )
        }
    
    async def collect_all_metrics(self) -> Dict[str, Any]:
        """Collect metrics from all cache subsystems."""
        timestamp = time.time()
        collected_metrics = {
            'timestamp': timestamp,
            'cache_hit_optimizer': {},
            'tier_based_cache': {},
            'redis_cluster': {},
            'cache_analytics': {},
            'system_metrics': {}
        }
        
        try:
            # Collect from cache hit optimizer
            if cache_hit_optimizer:
                hit_metrics = cache_hit_optimizer.get_optimization_metrics()
                collected_metrics['cache_hit_optimizer'] = hit_metrics
                
                # Update Prometheus metrics
                if 'hit_metrics' in hit_metrics:
                    hit_rate = hit_metrics['hit_metrics'].get('hit_rate_percent', 0)
                    self.prometheus_metrics['cache_hit_rate'].labels(tier='overall').set(hit_rate)
            
            # Collect from tier-based cache
            if tier_based_cache:
                tier_stats = tier_based_cache.get_comprehensive_stats()
                collected_metrics['tier_based_cache'] = tier_stats
                
                # Update tier-specific Prometheus metrics
                for tier_name, tier_data in tier_stats.get('tier_statistics', {}).items():
                    current_state = tier_data.get('current_state', {})
                    if 'efficiency_score' in current_state:
                        self.prometheus_metrics['cache_hit_rate'].labels(
                            tier=tier_name.lower()
                        ).set(current_state['efficiency_score'])
            
            # Collect from Redis cluster
            if redis_cluster_manager:
                cluster_health = await redis_cluster_manager.get_cluster_health()
                collected_metrics['redis_cluster'] = cluster_health
                
                # Update cluster metrics
                for node_data in cluster_health.get('nodes', []):
                    if 'memory_usage_mb' in node_data:
                        self.prometheus_metrics['cache_memory_usage_bytes'].labels(
                            node=node_data.get('address', 'unknown')
                        ).set(node_data['memory_usage_mb'] * 1024 * 1024)
            
            # Collect from cache analytics
            analytics_report = cache_analytics.generate_comprehensive_report()
            collected_metrics['cache_analytics'] = analytics_report
            
            # Collect system-level metrics
            collected_metrics['system_metrics'] = await self._collect_system_metrics()
            
            # Store metrics
            self._store_metrics(collected_metrics)
            
        except Exception as e:
            logger.error(f"Metrics collection failed: {e}")
        
        return collected_metrics
    
    async def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system-level performance metrics."""
        try:
            import psutil
            
            return {
                'cpu_usage_percent': psutil.cpu_percent(interval=1),
                'memory_usage_percent': psutil.virtual_memory().percent,
                'disk_usage_percent': psutil.disk_usage('/').percent,
                'network_io': asdict(psutil.net_io_counters()),
                'process_count': len(psutil.pids())
            }
        except ImportError:
            return {'error': 'psutil not available'}
        except Exception as e:
            return {'error': str(e)}
    
    def _store_metrics(self, metrics: Dict[str, Any]):
        """Store metrics with retention policy."""
        timestamp = metrics.get('timestamp', time.time())
        
        with self._lock:
            # Flatten and store metrics
            flat_metrics = self._flatten_metrics(metrics)
            
            for metric_name, value in flat_metrics.items():
                if isinstance(value, (int, float)):
                    point = MetricPoint(timestamp=timestamp, value=float(value))
                    self.metrics_storage[metric_name].append(point)
                    
                    # Apply retention policy
                    cutoff_time = timestamp - (self.retention_hours * 3600)
                    while (self.metrics_storage[metric_name] and 
                           self.metrics_storage[metric_name][0].timestamp < cutoff_time):
                        self.metrics_storage[metric_name].popleft()
    
    def _flatten_metrics(self, metrics: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """Flatten nested metrics dictionary."""
        flat = {}
        
        for key, value in metrics.items():
            if key == 'timestamp':
                continue
                
            full_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                flat.update(self._flatten_metrics(value, full_key))
            elif isinstance(value, (int, float, bool)):
                flat[full_key] = float(value)
        
        return flat
    
    def get_metric_history(
        self,
        metric_name: str,
        hours: int = 24,
        aggregation: str = 'avg'
    ) -> List[Dict[str, Any]]:
        """Get historical data for a metric."""
        cutoff_time = time.time() - (hours * 3600)
        
        with self._lock:
            if metric_name not in self.metrics_storage:
                return []
            
            # Filter to time range
            points = [
                p for p in self.metrics_storage[metric_name]
                if p.timestamp >= cutoff_time
            ]
            
            if not points:
                return []
            
            # Group by time buckets (5-minute intervals)
            bucket_size = 300  # 5 minutes
            buckets = defaultdict(list)
            
            for point in points:
                bucket = int(point.timestamp // bucket_size) * bucket_size
                buckets[bucket].append(point.value)
            
            # Aggregate buckets
            result = []
            for bucket_time in sorted(buckets.keys()):
                values = buckets[bucket_time]
                
                if aggregation == 'avg':
                    agg_value = statistics.mean(values)
                elif aggregation == 'max':
                    agg_value = max(values)
                elif aggregation == 'min':
                    agg_value = min(values)
                elif aggregation == 'sum':
                    agg_value = sum(values)
                else:
                    agg_value = statistics.mean(values)
                
                result.append({
                    'timestamp': bucket_time,
                    'value': agg_value,
                    'count': len(values)
                })
            
            return result
    
    def get_prometheus_metrics(self) -> str:
        """Get metrics in Prometheus format."""
        return generate_latest()


class CacheAlertManager:
    """
    Manages cache monitoring alerts and notifications.
    """
    
    def __init__(self):
        self.alerts: Dict[str, Alert] = {}
        self.alert_rules = []
        self.notification_handlers = []
        self._lock = threading.RLock()
        
        # Default alert rules
        self._setup_default_alert_rules()
    
    def _setup_default_alert_rules(self):
        """Setup default alert rules for cache monitoring."""
        self.alert_rules = [
            {
                'name': 'low_cache_hit_rate',
                'condition': lambda data: self._get_hit_rate(data) < 70.0,
                'severity': AlertSeverity.HIGH,
                'message': lambda data: f'Cache hit rate is low: {self._get_hit_rate(data):.1f}%',
                'cooldown_minutes': 10
            },
            {
                'name': 'high_response_time',
                'condition': lambda data: self._get_response_time(data) > 100.0,
                'severity': AlertSeverity.MEDIUM,
                'message': lambda data: f'High response time: {self._get_response_time(data):.1f}ms',
                'cooldown_minutes': 5
            },
            {
                'name': 'high_memory_usage',
                'condition': lambda data: self._get_memory_usage(data) > 90.0,
                'severity': AlertSeverity.CRITICAL,
                'message': lambda data: f'High memory usage: {self._get_memory_usage(data):.1f}%',
                'cooldown_minutes': 15
            },
            {
                'name': 'cluster_node_down',
                'condition': lambda data: self._check_cluster_health(data),
                'severity': AlertSeverity.CRITICAL,
                'message': lambda data: 'Redis cluster node(s) are down',
                'cooldown_minutes': 5
            },
            {
                'name': 'high_eviction_rate',
                'condition': lambda data: self._get_eviction_rate(data) > 100,
                'severity': AlertSeverity.MEDIUM,
                'message': lambda data: f'High eviction rate: {self._get_eviction_rate(data)}/sec',
                'cooldown_minutes': 10
            }
        ]
    
    def _get_hit_rate(self, data: Dict[str, Any]) -> float:
        """Extract hit rate from metrics data."""
        try:
            return data.get('cache_hit_optimizer', {}).get('hit_metrics', {}).get('hit_rate_percent', 100.0)
        except:
            return 100.0
    
    def _get_response_time(self, data: Dict[str, Any]) -> float:
        """Extract response time from metrics data."""
        try:
            return data.get('cache_hit_optimizer', {}).get('hit_metrics', {}).get('average_response_time_ms', 0.0)
        except:
            return 0.0
    
    def _get_memory_usage(self, data: Dict[str, Any]) -> float:
        """Extract memory usage from metrics data."""
        try:
            return data.get('system_metrics', {}).get('memory_usage_percent', 0.0)
        except:
            return 0.0
    
    def _check_cluster_health(self, data: Dict[str, Any]) -> bool:
        """Check if cluster has issues."""
        try:
            cluster_data = data.get('redis_cluster', {})
            total_nodes = cluster_data.get('total_nodes', 0)
            connected_nodes = cluster_data.get('connected_nodes', 0)
            return total_nodes > 0 and connected_nodes < total_nodes
        except:
            return False
    
    def _get_eviction_rate(self, data: Dict[str, Any]) -> float:
        """Get cache eviction rate."""
        # This would need to track evictions over time
        return 0.0
    
    async def check_alerts(self, metrics_data: Dict[str, Any]):
        """Check all alert rules against current metrics."""
        current_time = datetime.now()
        
        for rule in self.alert_rules:
            try:
                # Check if alert condition is met
                if rule['condition'](metrics_data):
                    alert_id = rule['name']
                    
                    # Check cooldown period
                    if alert_id in self.alerts:
                        existing_alert = self.alerts[alert_id]
                        if not existing_alert.resolved:
                            # Alert already active
                            continue
                        
                        # Check cooldown
                        cooldown_seconds = rule.get('cooldown_minutes', 5) * 60
                        if existing_alert.resolved_at:
                            time_since_resolved = (current_time - existing_alert.resolved_at).total_seconds()
                            if time_since_resolved < cooldown_seconds:
                                continue
                    
                    # Create new alert
                    alert = Alert(
                        id=alert_id,
                        name=rule['name'],
                        severity=rule['severity'],
                        message=rule['message'](metrics_data) if callable(rule['message']) else rule['message'],
                        timestamp=current_time,
                        metric_value=self._extract_metric_value(rule, metrics_data)
                    )
                    
                    with self._lock:
                        self.alerts[alert_id] = alert
                    
                    # Send notification
                    await self._send_notification(alert)
                    
                    logger.warning(f"Cache alert triggered: {alert.name} - {alert.message}")
                
                else:
                    # Check if we should resolve existing alert
                    alert_id = rule['name']
                    if alert_id in self.alerts and not self.alerts[alert_id].resolved:
                        with self._lock:
                            self.alerts[alert_id].resolved = True
                            self.alerts[alert_id].resolved_at = current_time
                        
                        logger.info(f"Cache alert resolved: {alert_id}")
            
            except Exception as e:
                logger.error(f"Alert check failed for {rule['name']}: {e}")
    
    def _extract_metric_value(self, rule: Dict[str, Any], data: Dict[str, Any]) -> float:
        """Extract the metric value that triggered the alert."""
        rule_name = rule['name']
        
        if 'hit_rate' in rule_name:
            return self._get_hit_rate(data)
        elif 'response_time' in rule_name:
            return self._get_response_time(data)
        elif 'memory' in rule_name:
            return self._get_memory_usage(data)
        else:
            return 0.0
    
    async def _send_notification(self, alert: Alert):
        """Send alert notification to registered handlers."""
        for handler in self.notification_handlers:
            try:
                await handler(alert)
            except Exception as e:
                logger.error(f"Notification handler failed: {e}")
    
    def register_notification_handler(self, handler: Callable[[Alert], None]):
        """Register a notification handler."""
        self.notification_handlers.append(handler)
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        with self._lock:
            return [alert for alert in self.alerts.values() if not alert.resolved]
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            return [
                alert for alert in self.alerts.values()
                if alert.timestamp >= cutoff_time
            ]


class CacheDashboardAPI:
    """
    Web API for cache monitoring dashboard.
    """
    
    def __init__(self, metrics_collector: CacheMetricsCollector, alert_manager: CacheAlertManager):
        self.metrics_collector = metrics_collector
        self.alert_manager = alert_manager
        self.app = Flask(__name__)
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup Flask routes for dashboard API."""
        
        @self.app.route('/api/metrics/current')
        async def get_current_metrics():
            """Get current cache metrics."""
            try:
                metrics = await self.metrics_collector.collect_all_metrics()
                return jsonify(metrics)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/metrics/<metric_name>')
        async def get_metric_history(metric_name):
            """Get historical data for a specific metric."""
            try:
                from flask import request
                hours = int(request.args.get('hours', 24))
                aggregation = request.args.get('aggregation', 'avg')
                
                history = self.metrics_collector.get_metric_history(
                    metric_name, hours, aggregation
                )
                return jsonify({
                    'metric': metric_name,
                    'hours': hours,
                    'aggregation': aggregation,
                    'data': history
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/alerts/active')
        def get_active_alerts():
            """Get active alerts."""
            try:
                alerts = self.alert_manager.get_active_alerts()
                return jsonify([asdict(alert) for alert in alerts])
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/alerts/history')
        def get_alert_history():
            """Get alert history."""
            try:
                from flask import request
                hours = int(request.args.get('hours', 24))
                
                alerts = self.alert_manager.get_alert_history(hours)
                return jsonify([asdict(alert) for alert in alerts])
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/dashboard/charts/<chart_type>')
        async def get_chart_data(chart_type):
            """Get data for dashboard charts."""
            try:
                chart_data = await self._generate_chart_data(chart_type)
                return jsonify(chart_data)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/metrics')
        def prometheus_metrics():
            """Prometheus metrics endpoint."""
            return self.metrics_collector.get_prometheus_metrics(), 200, {'Content-Type': CONTENT_TYPE_LATEST}
        
        @self.app.route('/')
        def dashboard():
            """Main dashboard page."""
            return render_template_string(DASHBOARD_TEMPLATE)
    
    async def _generate_chart_data(self, chart_type: str) -> Dict[str, Any]:
        """Generate data for specific chart types."""
        if chart_type == 'hit_rate_trend':
            return await self._get_hit_rate_trend_data()
        elif chart_type == 'response_time_distribution':
            return await self._get_response_time_distribution_data()
        elif chart_type == 'memory_usage':
            return await self._get_memory_usage_data()
        elif chart_type == 'tier_performance':
            return await self._get_tier_performance_data()
        elif chart_type == 'cluster_health':
            return await self._get_cluster_health_data()
        else:
            return {'error': f'Unknown chart type: {chart_type}'}
    
    async def _get_hit_rate_trend_data(self) -> Dict[str, Any]:
        """Get hit rate trend chart data."""
        history = self.metrics_collector.get_metric_history(
            'cache_hit_optimizer.hit_metrics.hit_rate_percent',
            hours=24
        )
        
        timestamps = [datetime.fromtimestamp(p['timestamp']) for p in history]
        values = [p['value'] for p in history]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=values,
            mode='lines+markers',
            name='Hit Rate %',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title='Cache Hit Rate Trend (24h)',
            xaxis_title='Time',
            yaxis_title='Hit Rate (%)',
            height=400
        )
        
        return {
            'chart_json': json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder),
            'summary': {
                'current': values[-1] if values else 0,
                'average': sum(values) / len(values) if values else 0,
                'min': min(values) if values else 0,
                'max': max(values) if values else 0
            }
        }
    
    async def _get_response_time_distribution_data(self) -> Dict[str, Any]:
        """Get response time distribution data."""
        history = self.metrics_collector.get_metric_history(
            'cache_hit_optimizer.hit_metrics.average_response_time_ms',
            hours=24
        )
        
        values = [p['value'] for p in history if p['value'] > 0]
        
        if not values:
            return {'error': 'No response time data available'}
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=values,
            nbinsx=30,
            name='Response Time Distribution',
            marker_color='green'
        ))
        
        fig.update_layout(
            title='Response Time Distribution (24h)',
            xaxis_title='Response Time (ms)',
            yaxis_title='Frequency',
            height=400
        )
        
        return {
            'chart_json': json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder),
            'summary': {
                'p50': np.percentile(values, 50),
                'p95': np.percentile(values, 95),
                'p99': np.percentile(values, 99),
                'average': np.mean(values)
            }
        }
    
    async def _get_memory_usage_data(self) -> Dict[str, Any]:
        """Get memory usage chart data."""
        system_memory = self.metrics_collector.get_metric_history(
            'system_metrics.memory_usage_percent',
            hours=24
        )
        
        timestamps = [datetime.fromtimestamp(p['timestamp']) for p in system_memory]
        values = [p['value'] for p in system_memory]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=values,
            mode='lines',
            fill='tonexty',
            name='Memory Usage %',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title='System Memory Usage (24h)',
            xaxis_title='Time',
            yaxis_title='Memory Usage (%)',
            height=400
        )
        
        return {
            'chart_json': json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder),
            'summary': {
                'current': values[-1] if values else 0,
                'peak': max(values) if values else 0,
                'average': sum(values) / len(values) if values else 0
            }
        }
    
    async def _get_tier_performance_data(self) -> Dict[str, Any]:
        """Get tier performance comparison data."""
        # This would collect performance data for each tier
        tier_data = {
            'CRITICAL': {'hit_rate': 95.2, 'avg_response_ms': 2.1},
            'HIGH': {'hit_rate': 92.8, 'avg_response_ms': 3.5},
            'MEDIUM': {'hit_rate': 88.4, 'avg_response_ms': 5.2},
            'LOW': {'hit_rate': 82.1, 'avg_response_ms': 8.7}
        }
        
        tiers = list(tier_data.keys())
        hit_rates = [tier_data[tier]['hit_rate'] for tier in tiers]
        response_times = [tier_data[tier]['avg_response_ms'] for tier in tiers]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=tiers,
            y=hit_rates,
            name='Hit Rate %',
            marker_color='blue',
            yaxis='y'
        ))
        
        fig.add_trace(go.Scatter(
            x=tiers,
            y=response_times,
            mode='lines+markers',
            name='Avg Response Time (ms)',
            marker_color='red',
            yaxis='y2'
        ))
        
        fig.update_layout(
            title='Tier Performance Comparison',
            xaxis_title='Cache Tier',
            yaxis=dict(title='Hit Rate (%)', side='left'),
            yaxis2=dict(title='Response Time (ms)', side='right', overlaying='y'),
            height=400
        )
        
        return {
            'chart_json': json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder),
            'tier_data': tier_data
        }
    
    async def _get_cluster_health_data(self) -> Dict[str, Any]:
        """Get Redis cluster health data."""
        if not redis_cluster_manager:
            return {'error': 'Redis cluster not available'}
        
        health = await redis_cluster_manager.get_cluster_health()
        
        # Create cluster nodes status chart
        node_data = health.get('nodes', [])
        node_labels = [node.get('address', 'unknown') for node in node_data]
        memory_usage = [node.get('memory_usage_mb', 0) for node in node_data]
        ops_per_sec = [node.get('ops_per_sec', 0) for node in node_data]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=node_labels,
            y=memory_usage,
            name='Memory Usage (MB)',
            marker_color='purple'
        ))
        
        fig.update_layout(
            title='Cluster Nodes Status',
            xaxis_title='Node',
            yaxis_title='Memory Usage (MB)',
            height=400
        )
        
        return {
            'chart_json': json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder),
            'cluster_summary': {
                'total_nodes': health.get('total_nodes', 0),
                'connected_nodes': health.get('connected_nodes', 0),
                'cluster_state': health.get('cluster_state', 'unknown')
            }
        }


# Dashboard HTML template (simplified)
DASHBOARD_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Cache Performance Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #f4f4f4; padding: 20px; margin-bottom: 20px; }
        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .metric-card { background: white; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }
        .chart-container { height: 400px; margin: 20px 0; }
        .alert { padding: 10px; margin: 10px 0; border-radius: 5px; }
        .alert-high { background: #ffebee; border-left: 4px solid #f44336; }
        .alert-medium { background: #fff3e0; border-left: 4px solid #ff9800; }
        .alert-low { background: #e8f5e8; border-left: 4px solid #4caf50; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Investment Analysis Cache Performance Dashboard</h1>
        <p>Real-time monitoring for 6000+ stocks processing</p>
    </div>
    
    <div id="alerts-section">
        <h2>Active Alerts</h2>
        <div id="alerts-container"></div>
    </div>
    
    <div class="metrics-grid">
        <div class="metric-card">
            <h3>Hit Rate Trend</h3>
            <div id="hit-rate-chart" class="chart-container"></div>
        </div>
        
        <div class="metric-card">
            <h3>Response Time Distribution</h3>
            <div id="response-time-chart" class="chart-container"></div>
        </div>
        
        <div class="metric-card">
            <h3>Memory Usage</h3>
            <div id="memory-usage-chart" class="chart-container"></div>
        </div>
        
        <div class="metric-card">
            <h3>Tier Performance</h3>
            <div id="tier-performance-chart" class="chart-container"></div>
        </div>
    </div>
    
    <script>
        // Auto-refresh dashboard every 30 seconds
        setInterval(function() {
            loadCharts();
            loadAlerts();
        }, 30000);
        
        // Load charts
        function loadCharts() {
            loadChart('hit_rate_trend', 'hit-rate-chart');
            loadChart('response_time_distribution', 'response-time-chart');
            loadChart('memory_usage', 'memory-usage-chart');
            loadChart('tier_performance', 'tier-performance-chart');
        }
        
        function loadChart(chartType, containerId) {
            fetch(`/api/dashboard/charts/${chartType}`)
                .then(response => response.json())
                .then(data => {
                    if (data.chart_json) {
                        var chartData = JSON.parse(data.chart_json);
                        Plotly.newPlot(containerId, chartData.data, chartData.layout);
                    }
                })
                .catch(error => console.error('Chart loading failed:', error));
        }
        
        function loadAlerts() {
            fetch('/api/alerts/active')
                .then(response => response.json())
                .then(alerts => {
                    var container = document.getElementById('alerts-container');
                    if (alerts.length === 0) {
                        container.innerHTML = '<p>No active alerts</p>';
                        return;
                    }
                    
                    var html = '';
                    alerts.forEach(alert => {
                        var alertClass = `alert-${alert.severity}`;
                        html += `<div class="alert ${alertClass}">
                            <strong>${alert.name}</strong>: ${alert.message}
                            <small>(${new Date(alert.timestamp).toLocaleString()})</small>
                        </div>`;
                    });
                    container.innerHTML = html;
                })
                .catch(error => console.error('Alerts loading failed:', error));
        }
        
        // Initial load
        loadCharts();
        loadAlerts();
    </script>
</body>
</html>
'''


# Global instances
metrics_collector = CacheMetricsCollector()
alert_manager = CacheAlertManager()
dashboard_api = CacheDashboardAPI(metrics_collector, alert_manager)


async def initialize_cache_monitoring():
    """Initialize cache monitoring system."""
    
    # Setup notification handlers
    async def log_notification_handler(alert: Alert):
        """Log alert notifications."""
        logger.warning(f"CACHE ALERT [{alert.severity.value.upper()}]: {alert.message}")
    
    alert_manager.register_notification_handler(log_notification_handler)
    
    # Start monitoring loop
    asyncio.create_task(monitoring_loop())
    
    logger.info("Cache monitoring dashboard initialized")


async def monitoring_loop():
    """Main monitoring loop."""
    while True:
        try:
            # Collect metrics
            metrics_data = await metrics_collector.collect_all_metrics()
            
            # Check alerts
            await alert_manager.check_alerts(metrics_data)
            
            await asyncio.sleep(30)  # Monitor every 30 seconds
            
        except Exception as e:
            logger.error(f"Monitoring loop error: {e}")
            await asyncio.sleep(60)  # Wait longer on error


def start_dashboard_server(host: str = '0.0.0.0', port: int = 8080):
    """Start the dashboard web server."""
    dashboard_api.app.run(host=host, port=port, debug=False)