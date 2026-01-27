"""
AlertManager Webhook Integration for Business Metrics

Provides the ability to send alerts from business metrics to various
webhook endpoints including:
- Prometheus AlertManager
- PagerDuty
- Slack
- Generic webhooks

Includes rate limiting to prevent alert storms and configurable thresholds.
"""

import asyncio
import hashlib
import logging
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

import aiohttp
from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels compatible with AlertManager."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertState(Enum):
    """Alert state for AlertManager."""
    FIRING = "firing"
    RESOLVED = "resolved"


# Metrics for alert tracking
alerts_sent_total = Counter(
    'business_alerts_sent_total',
    'Total business alerts sent',
    ['severity', 'alert_type', 'target']
)

alerts_suppressed_total = Counter(
    'business_alerts_suppressed_total',
    'Total business alerts suppressed by rate limiting',
    ['alert_type', 'reason']
)

alert_send_latency = Histogram(
    'business_alert_send_latency_seconds',
    'Latency of sending business alerts',
    ['target'],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

active_alerts_gauge = Gauge(
    'business_active_alerts_count',
    'Number of currently active business alerts',
    ['severity']
)


@dataclass
class AlertThresholds:
    """Configurable thresholds for business metric alerts."""
    # API error rate thresholds
    api_error_rate_warning: float = 0.01  # 1%
    api_error_rate_critical: float = 0.05  # 5%

    # Response time thresholds (seconds)
    response_time_p99_warning: float = 1.5
    response_time_p99_critical: float = 2.0

    # Cache hit rate thresholds
    cache_hit_rate_warning: float = 0.85  # 85%
    cache_hit_rate_critical: float = 0.80  # 80%

    # Memory usage thresholds
    memory_usage_warning: float = 0.80  # 80%
    memory_usage_critical: float = 0.85  # 85%

    # Budget thresholds (percentage of monthly budget)
    budget_usage_warning: float = 0.70  # 70%
    budget_usage_critical: float = 0.85  # 85%

    # Data pipeline thresholds
    pipeline_success_rate_warning: float = 0.95  # 95%
    pipeline_success_rate_critical: float = 0.90  # 90%

    # ML model accuracy thresholds
    ml_accuracy_warning: float = 0.75  # 75%
    ml_accuracy_critical: float = 0.65  # 65%

    @classmethod
    def from_env(cls) -> 'AlertThresholds':
        """Load thresholds from environment variables."""
        return cls(
            api_error_rate_warning=float(os.getenv('ALERT_API_ERROR_RATE_WARNING', '0.01')),
            api_error_rate_critical=float(os.getenv('ALERT_API_ERROR_RATE_CRITICAL', '0.05')),
            response_time_p99_warning=float(os.getenv('ALERT_RESPONSE_TIME_WARNING', '1.5')),
            response_time_p99_critical=float(os.getenv('ALERT_RESPONSE_TIME_CRITICAL', '2.0')),
            cache_hit_rate_warning=float(os.getenv('ALERT_CACHE_HIT_RATE_WARNING', '0.85')),
            cache_hit_rate_critical=float(os.getenv('ALERT_CACHE_HIT_RATE_CRITICAL', '0.80')),
            memory_usage_warning=float(os.getenv('ALERT_MEMORY_USAGE_WARNING', '0.80')),
            memory_usage_critical=float(os.getenv('ALERT_MEMORY_USAGE_CRITICAL', '0.85')),
            budget_usage_warning=float(os.getenv('ALERT_BUDGET_USAGE_WARNING', '0.70')),
            budget_usage_critical=float(os.getenv('ALERT_BUDGET_USAGE_CRITICAL', '0.85')),
            pipeline_success_rate_warning=float(os.getenv('ALERT_PIPELINE_SUCCESS_WARNING', '0.95')),
            pipeline_success_rate_critical=float(os.getenv('ALERT_PIPELINE_SUCCESS_CRITICAL', '0.90')),
            ml_accuracy_warning=float(os.getenv('ALERT_ML_ACCURACY_WARNING', '0.75')),
            ml_accuracy_critical=float(os.getenv('ALERT_ML_ACCURACY_CRITICAL', '0.65')),
        )


@dataclass
class AlertCooldownConfig:
    """Configuration for alert cooldown periods to prevent alert storms."""
    # Default cooldown periods in seconds
    default_cooldown: int = 300  # 5 minutes

    # Per-alert-type cooldowns
    alert_cooldowns: Dict[str, int] = field(default_factory=lambda: {
        'high_api_error_rate': 300,       # 5 minutes
        'slow_response_time': 300,        # 5 minutes
        'low_cache_hit_rate': 600,        # 10 minutes
        'high_memory_usage': 300,         # 5 minutes
        'budget_exceeded': 1800,          # 30 minutes
        'pipeline_failure': 300,          # 5 minutes
        'ml_model_degradation': 600,      # 10 minutes
        'job_failure': 180,               # 3 minutes
    })

    # Maximum alerts per hour per type
    max_alerts_per_hour: int = 10


@dataclass
class BusinessAlert:
    """Represents a business metric alert."""
    alert_name: str
    severity: AlertSeverity
    summary: str
    description: str
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    metric_value: Optional[float] = None
    threshold_value: Optional[float] = None
    state: AlertState = AlertState.FIRING
    started_at: datetime = field(default_factory=datetime.utcnow)
    fingerprint: Optional[str] = None

    def __post_init__(self):
        if not self.fingerprint:
            self.fingerprint = self._generate_fingerprint()

    def _generate_fingerprint(self) -> str:
        """Generate unique fingerprint for deduplication."""
        content = f"{self.alert_name}:{sorted(self.labels.items())}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def to_alertmanager_format(self) -> Dict[str, Any]:
        """Convert to Prometheus AlertManager API format."""
        alert_labels = {
            'alertname': self.alert_name,
            'severity': self.severity.value,
            **self.labels
        }

        alert_annotations = {
            'summary': self.summary,
            'description': self.description,
            **self.annotations
        }

        if self.metric_value is not None:
            alert_annotations['metric_value'] = str(self.metric_value)
        if self.threshold_value is not None:
            alert_annotations['threshold_value'] = str(self.threshold_value)

        return {
            'labels': alert_labels,
            'annotations': alert_annotations,
            'startsAt': self.started_at.isoformat() + 'Z',
            'generatorURL': os.getenv('ALERT_GENERATOR_URL', 'http://investment-platform/metrics'),
        }

    def to_pagerduty_format(self) -> Dict[str, Any]:
        """Convert to PagerDuty Events API v2 format."""
        severity_map = {
            AlertSeverity.INFO: 'info',
            AlertSeverity.WARNING: 'warning',
            AlertSeverity.CRITICAL: 'critical',
        }

        return {
            'routing_key': os.getenv('PAGERDUTY_ROUTING_KEY', ''),
            'event_action': 'trigger' if self.state == AlertState.FIRING else 'resolve',
            'dedup_key': self.fingerprint,
            'payload': {
                'summary': self.summary,
                'source': os.getenv('SERVICE_NAME', 'investment-analysis-platform'),
                'severity': severity_map.get(self.severity, 'warning'),
                'timestamp': self.started_at.isoformat() + 'Z',
                'component': self.labels.get('component', 'business-metrics'),
                'group': self.labels.get('group', 'metrics'),
                'class': self.alert_name,
                'custom_details': {
                    'description': self.description,
                    'metric_value': self.metric_value,
                    'threshold_value': self.threshold_value,
                    **self.labels
                }
            },
            'links': [
                {
                    'href': os.getenv('GRAFANA_DASHBOARD_URL', 'http://grafana:3000'),
                    'text': 'Grafana Dashboard'
                }
            ]
        }

    def to_slack_format(self) -> Dict[str, Any]:
        """Convert to Slack webhook format."""
        severity_colors = {
            AlertSeverity.INFO: '#17a2b8',
            AlertSeverity.WARNING: '#ffc107',
            AlertSeverity.CRITICAL: '#dc3545',
        }

        severity_emojis = {
            AlertSeverity.INFO: ':information_source:',
            AlertSeverity.WARNING: ':warning:',
            AlertSeverity.CRITICAL: ':rotating_light:',
        }

        color = severity_colors.get(self.severity, '#6c757d')
        emoji = severity_emojis.get(self.severity, ':bell:')

        fields = [
            {'title': 'Severity', 'value': self.severity.value.upper(), 'short': True},
            {'title': 'Alert', 'value': self.alert_name, 'short': True},
        ]

        if self.metric_value is not None:
            fields.append({'title': 'Current Value', 'value': f'{self.metric_value:.4f}', 'short': True})
        if self.threshold_value is not None:
            fields.append({'title': 'Threshold', 'value': f'{self.threshold_value:.4f}', 'short': True})

        for key, value in self.labels.items():
            if len(fields) < 10:
                fields.append({'title': key, 'value': str(value), 'short': True})

        return {
            'username': os.getenv('SLACK_BOT_NAME', 'Investment Platform Alerts'),
            'icon_emoji': ':chart_with_upwards_trend:',
            'attachments': [
                {
                    'color': color,
                    'title': f'{emoji} {self.summary}',
                    'text': self.description,
                    'fields': fields,
                    'footer': 'Investment Analysis Platform',
                    'ts': int(self.started_at.timestamp())
                }
            ]
        }


class AlertRateLimiter:
    """Rate limiter to prevent alert storms."""

    def __init__(self, config: AlertCooldownConfig):
        self.config = config
        self._last_alert_times: Dict[str, datetime] = {}
        self._alert_counts: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self._suppressed_alerts: Set[str] = set()

    def should_allow_alert(self, alert: BusinessAlert) -> bool:
        """Check if alert should be allowed based on rate limiting rules."""
        alert_key = f"{alert.alert_name}:{alert.fingerprint}"
        now = datetime.utcnow()

        # Check cooldown period
        cooldown_seconds = self.config.alert_cooldowns.get(
            alert.alert_name,
            self.config.default_cooldown
        )

        last_time = self._last_alert_times.get(alert_key)
        if last_time:
            time_since_last = (now - last_time).total_seconds()
            if time_since_last < cooldown_seconds:
                logger.debug(
                    f"Alert {alert.alert_name} in cooldown "
                    f"({cooldown_seconds - time_since_last:.0f}s remaining)"
                )
                alerts_suppressed_total.labels(
                    alert_type=alert.alert_name,
                    reason='cooldown'
                ).inc()
                return False

        # Check hourly rate limit
        hour_ago = now - timedelta(hours=1)
        alert_times = self._alert_counts[alert.alert_name]

        # Clean old entries
        while alert_times and alert_times[0] < hour_ago:
            alert_times.popleft()

        if len(alert_times) >= self.config.max_alerts_per_hour:
            logger.warning(
                f"Alert {alert.alert_name} rate limited "
                f"({len(alert_times)}/{self.config.max_alerts_per_hour} per hour)"
            )
            alerts_suppressed_total.labels(
                alert_type=alert.alert_name,
                reason='rate_limit'
            ).inc()
            return False

        return True

    def record_alert(self, alert: BusinessAlert):
        """Record that an alert was sent."""
        alert_key = f"{alert.alert_name}:{alert.fingerprint}"
        now = datetime.utcnow()

        self._last_alert_times[alert_key] = now
        self._alert_counts[alert.alert_name].append(now)

    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        now = datetime.utcnow()
        hour_ago = now - timedelta(hours=1)

        hourly_counts = {}
        for alert_name, times in self._alert_counts.items():
            count = sum(1 for t in times if t > hour_ago)
            hourly_counts[alert_name] = count

        return {
            'hourly_counts': hourly_counts,
            'cooldown_active': len([
                k for k, v in self._last_alert_times.items()
                if (now - v).total_seconds() < self.config.default_cooldown
            ]),
            'max_per_hour': self.config.max_alerts_per_hour
        }


class AlertManagerWebhook:
    """
    Sends alerts to AlertManager and other webhook endpoints.

    Supports:
    - Prometheus AlertManager API
    - PagerDuty Events API v2
    - Slack Incoming Webhooks
    - Generic HTTP webhooks
    """

    def __init__(
        self,
        alertmanager_url: Optional[str] = None,
        pagerduty_url: Optional[str] = None,
        slack_webhook_url: Optional[str] = None,
        generic_webhook_url: Optional[str] = None,
        thresholds: Optional[AlertThresholds] = None,
        cooldown_config: Optional[AlertCooldownConfig] = None,
    ):
        self.alertmanager_url = alertmanager_url or os.getenv(
            'ALERTMANAGER_URL',
            'http://alertmanager:9093'
        )
        self.pagerduty_url = pagerduty_url or os.getenv(
            'PAGERDUTY_URL',
            'https://events.pagerduty.com/v2/enqueue'
        )
        self.slack_webhook_url = slack_webhook_url or os.getenv('SLACK_WEBHOOK_URL')
        self.generic_webhook_url = generic_webhook_url or os.getenv('GENERIC_WEBHOOK_URL')

        self.thresholds = thresholds or AlertThresholds.from_env()
        self.cooldown_config = cooldown_config or AlertCooldownConfig()
        self.rate_limiter = AlertRateLimiter(self.cooldown_config)

        self._active_alerts: Dict[str, BusinessAlert] = {}
        self._http_timeout = aiohttp.ClientTimeout(total=30)

        # Track enabled targets
        self._enabled_targets: Set[str] = set()
        if self.alertmanager_url:
            self._enabled_targets.add('alertmanager')
        if self.slack_webhook_url:
            self._enabled_targets.add('slack')
        if os.getenv('PAGERDUTY_ROUTING_KEY'):
            self._enabled_targets.add('pagerduty')
        if self.generic_webhook_url:
            self._enabled_targets.add('generic')

        logger.info(f"AlertManager webhook initialized with targets: {self._enabled_targets}")

    async def send_alert(self, alert: BusinessAlert) -> bool:
        """
        Send alert to all configured webhook endpoints.

        Returns True if at least one target received the alert successfully.
        """
        # Check rate limiting
        if not self.rate_limiter.should_allow_alert(alert):
            return False

        # Track active alert
        self._active_alerts[alert.fingerprint] = alert
        self._update_active_alerts_metric()

        tasks = []

        if 'alertmanager' in self._enabled_targets:
            tasks.append(self._send_to_alertmanager(alert))

        if 'slack' in self._enabled_targets:
            tasks.append(self._send_to_slack(alert))

        if 'pagerduty' in self._enabled_targets and alert.severity == AlertSeverity.CRITICAL:
            tasks.append(self._send_to_pagerduty(alert))

        if 'generic' in self._enabled_targets:
            tasks.append(self._send_to_generic_webhook(alert))

        if not tasks:
            logger.warning("No webhook targets configured")
            return False

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Record if any succeeded
        success = any(r is True for r in results if not isinstance(r, Exception))

        if success:
            self.rate_limiter.record_alert(alert)
            logger.info(f"Alert sent successfully: {alert.alert_name}")
        else:
            logger.error(f"Failed to send alert to any target: {alert.alert_name}")

        return success

    async def resolve_alert(self, fingerprint: str) -> bool:
        """Resolve an active alert."""
        if fingerprint not in self._active_alerts:
            logger.warning(f"No active alert with fingerprint: {fingerprint}")
            return False

        alert = self._active_alerts[fingerprint]
        alert.state = AlertState.RESOLVED

        tasks = []

        if 'alertmanager' in self._enabled_targets:
            tasks.append(self._send_to_alertmanager(alert))

        if 'pagerduty' in self._enabled_targets:
            tasks.append(self._send_to_pagerduty(alert))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        del self._active_alerts[fingerprint]
        self._update_active_alerts_metric()

        logger.info(f"Alert resolved: {alert.alert_name}")
        return True

    async def _send_to_alertmanager(self, alert: BusinessAlert) -> bool:
        """Send alert to Prometheus AlertManager."""
        start_time = time.time()
        try:
            url = f"{self.alertmanager_url}/api/v2/alerts"
            payload = [alert.to_alertmanager_format()]

            async with aiohttp.ClientSession(timeout=self._http_timeout) as session:
                async with session.post(url, json=payload) as response:
                    duration = time.time() - start_time
                    alert_send_latency.labels(target='alertmanager').observe(duration)

                    if response.status in (200, 202):
                        alerts_sent_total.labels(
                            severity=alert.severity.value,
                            alert_type=alert.alert_name,
                            target='alertmanager'
                        ).inc()
                        return True
                    else:
                        body = await response.text()
                        logger.error(
                            f"AlertManager returned {response.status}: {body}"
                        )
                        return False

        except asyncio.TimeoutError:
            logger.error("AlertManager request timed out")
            return False
        except Exception as e:
            logger.error(f"Failed to send to AlertManager: {e}")
            return False

    async def _send_to_pagerduty(self, alert: BusinessAlert) -> bool:
        """Send alert to PagerDuty."""
        start_time = time.time()
        try:
            payload = alert.to_pagerduty_format()

            if not payload.get('routing_key'):
                logger.warning("PagerDuty routing key not configured")
                return False

            async with aiohttp.ClientSession(timeout=self._http_timeout) as session:
                async with session.post(self.pagerduty_url, json=payload) as response:
                    duration = time.time() - start_time
                    alert_send_latency.labels(target='pagerduty').observe(duration)

                    if response.status == 202:
                        alerts_sent_total.labels(
                            severity=alert.severity.value,
                            alert_type=alert.alert_name,
                            target='pagerduty'
                        ).inc()
                        return True
                    else:
                        body = await response.text()
                        logger.error(
                            f"PagerDuty returned {response.status}: {body}"
                        )
                        return False

        except asyncio.TimeoutError:
            logger.error("PagerDuty request timed out")
            return False
        except Exception as e:
            logger.error(f"Failed to send to PagerDuty: {e}")
            return False

    async def _send_to_slack(self, alert: BusinessAlert) -> bool:
        """Send alert to Slack webhook."""
        start_time = time.time()
        try:
            if not self.slack_webhook_url:
                return False

            payload = alert.to_slack_format()

            async with aiohttp.ClientSession(timeout=self._http_timeout) as session:
                async with session.post(self.slack_webhook_url, json=payload) as response:
                    duration = time.time() - start_time
                    alert_send_latency.labels(target='slack').observe(duration)

                    if response.status == 200:
                        alerts_sent_total.labels(
                            severity=alert.severity.value,
                            alert_type=alert.alert_name,
                            target='slack'
                        ).inc()
                        return True
                    else:
                        body = await response.text()
                        logger.error(
                            f"Slack returned {response.status}: {body}"
                        )
                        return False

        except asyncio.TimeoutError:
            logger.error("Slack request timed out")
            return False
        except Exception as e:
            logger.error(f"Failed to send to Slack: {e}")
            return False

    async def _send_to_generic_webhook(self, alert: BusinessAlert) -> bool:
        """Send alert to generic webhook endpoint."""
        start_time = time.time()
        try:
            if not self.generic_webhook_url:
                return False

            payload = {
                'event': 'business_alert',
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'alert': {
                    'name': alert.alert_name,
                    'severity': alert.severity.value,
                    'state': alert.state.value,
                    'summary': alert.summary,
                    'description': alert.description,
                    'labels': alert.labels,
                    'annotations': alert.annotations,
                    'metric_value': alert.metric_value,
                    'threshold_value': alert.threshold_value,
                    'fingerprint': alert.fingerprint,
                    'started_at': alert.started_at.isoformat() + 'Z',
                },
                'source': os.getenv('SERVICE_NAME', 'investment-analysis-platform')
            }

            headers = {'Content-Type': 'application/json'}

            # Add auth if configured
            auth_token = os.getenv('GENERIC_WEBHOOK_AUTH_TOKEN')
            if auth_token:
                headers['Authorization'] = f'Bearer {auth_token}'

            async with aiohttp.ClientSession(timeout=self._http_timeout) as session:
                async with session.post(
                    self.generic_webhook_url,
                    json=payload,
                    headers=headers
                ) as response:
                    duration = time.time() - start_time
                    alert_send_latency.labels(target='generic').observe(duration)

                    if 200 <= response.status < 300:
                        alerts_sent_total.labels(
                            severity=alert.severity.value,
                            alert_type=alert.alert_name,
                            target='generic'
                        ).inc()
                        return True
                    else:
                        body = await response.text()
                        logger.error(
                            f"Generic webhook returned {response.status}: {body}"
                        )
                        return False

        except asyncio.TimeoutError:
            logger.error("Generic webhook request timed out")
            return False
        except Exception as e:
            logger.error(f"Failed to send to generic webhook: {e}")
            return False

    def _update_active_alerts_metric(self):
        """Update the active alerts gauge metric."""
        severity_counts = defaultdict(int)
        for alert in self._active_alerts.values():
            severity_counts[alert.severity.value] += 1

        for severity in AlertSeverity:
            active_alerts_gauge.labels(severity=severity.value).set(
                severity_counts.get(severity.value, 0)
            )

    # Convenience methods for creating specific alert types

    async def alert_high_api_error_rate(
        self,
        error_rate: float,
        endpoint: Optional[str] = None
    ) -> bool:
        """Create and send alert for high API error rate."""
        if error_rate >= self.thresholds.api_error_rate_critical:
            severity = AlertSeverity.CRITICAL
        elif error_rate >= self.thresholds.api_error_rate_warning:
            severity = AlertSeverity.WARNING
        else:
            return False

        threshold = (
            self.thresholds.api_error_rate_critical
            if severity == AlertSeverity.CRITICAL
            else self.thresholds.api_error_rate_warning
        )

        labels = {'component': 'api', 'metric': 'error_rate'}
        if endpoint:
            labels['endpoint'] = endpoint

        alert = BusinessAlert(
            alert_name='high_api_error_rate',
            severity=severity,
            summary=f'API error rate at {error_rate*100:.2f}%',
            description=(
                f'API error rate has exceeded threshold. '
                f'Current: {error_rate*100:.2f}%, Threshold: {threshold*100:.2f}%'
            ),
            labels=labels,
            metric_value=error_rate,
            threshold_value=threshold,
        )

        return await self.send_alert(alert)

    async def alert_slow_response_time(
        self,
        p99_latency: float,
        endpoint: Optional[str] = None
    ) -> bool:
        """Create and send alert for slow response times."""
        if p99_latency >= self.thresholds.response_time_p99_critical:
            severity = AlertSeverity.CRITICAL
        elif p99_latency >= self.thresholds.response_time_p99_warning:
            severity = AlertSeverity.WARNING
        else:
            return False

        threshold = (
            self.thresholds.response_time_p99_critical
            if severity == AlertSeverity.CRITICAL
            else self.thresholds.response_time_p99_warning
        )

        labels = {'component': 'api', 'metric': 'latency_p99'}
        if endpoint:
            labels['endpoint'] = endpoint

        alert = BusinessAlert(
            alert_name='slow_response_time',
            severity=severity,
            summary=f'P99 response time at {p99_latency:.2f}s',
            description=(
                f'API response time (P99) has exceeded threshold. '
                f'Current: {p99_latency:.2f}s, Threshold: {threshold:.2f}s'
            ),
            labels=labels,
            metric_value=p99_latency,
            threshold_value=threshold,
        )

        return await self.send_alert(alert)

    async def alert_low_cache_hit_rate(
        self,
        hit_rate: float,
        cache_name: Optional[str] = None
    ) -> bool:
        """Create and send alert for low cache hit rate."""
        if hit_rate <= self.thresholds.cache_hit_rate_critical:
            severity = AlertSeverity.CRITICAL
        elif hit_rate <= self.thresholds.cache_hit_rate_warning:
            severity = AlertSeverity.WARNING
        else:
            return False

        threshold = (
            self.thresholds.cache_hit_rate_critical
            if severity == AlertSeverity.CRITICAL
            else self.thresholds.cache_hit_rate_warning
        )

        labels = {'component': 'cache', 'metric': 'hit_rate'}
        if cache_name:
            labels['cache_name'] = cache_name

        alert = BusinessAlert(
            alert_name='low_cache_hit_rate',
            severity=severity,
            summary=f'Cache hit rate at {hit_rate*100:.1f}%',
            description=(
                f'Cache hit rate has dropped below threshold. '
                f'Current: {hit_rate*100:.1f}%, Threshold: {threshold*100:.1f}%'
            ),
            labels=labels,
            metric_value=hit_rate,
            threshold_value=threshold,
        )

        return await self.send_alert(alert)

    async def alert_high_memory_usage(
        self,
        memory_percent: float,
        process_name: Optional[str] = None
    ) -> bool:
        """Create and send alert for high memory usage."""
        if memory_percent >= self.thresholds.memory_usage_critical:
            severity = AlertSeverity.CRITICAL
        elif memory_percent >= self.thresholds.memory_usage_warning:
            severity = AlertSeverity.WARNING
        else:
            return False

        threshold = (
            self.thresholds.memory_usage_critical
            if severity == AlertSeverity.CRITICAL
            else self.thresholds.memory_usage_warning
        )

        labels = {'component': 'system', 'metric': 'memory_usage'}
        if process_name:
            labels['process'] = process_name

        alert = BusinessAlert(
            alert_name='high_memory_usage',
            severity=severity,
            summary=f'Memory usage at {memory_percent*100:.1f}%',
            description=(
                f'Memory usage has exceeded threshold. '
                f'Current: {memory_percent*100:.1f}%, Threshold: {threshold*100:.1f}%'
            ),
            labels=labels,
            metric_value=memory_percent,
            threshold_value=threshold,
        )

        return await self.send_alert(alert)

    async def alert_budget_exceeded(
        self,
        current_spend: float,
        budget: float
    ) -> bool:
        """Create and send alert for budget threshold exceeded."""
        usage_percent = current_spend / budget if budget > 0 else 0

        if usage_percent >= self.thresholds.budget_usage_critical:
            severity = AlertSeverity.CRITICAL
        elif usage_percent >= self.thresholds.budget_usage_warning:
            severity = AlertSeverity.WARNING
        else:
            return False

        threshold = (
            self.thresholds.budget_usage_critical
            if severity == AlertSeverity.CRITICAL
            else self.thresholds.budget_usage_warning
        )

        alert = BusinessAlert(
            alert_name='budget_exceeded',
            severity=severity,
            summary=f'Budget usage at {usage_percent*100:.1f}%',
            description=(
                f'Monthly budget usage has exceeded threshold. '
                f'Current spend: ${current_spend:.2f}, Budget: ${budget:.2f} '
                f'({usage_percent*100:.1f}% used)'
            ),
            labels={
                'component': 'cost',
                'metric': 'budget_usage',
            },
            annotations={
                'current_spend': f'${current_spend:.2f}',
                'monthly_budget': f'${budget:.2f}',
            },
            metric_value=usage_percent,
            threshold_value=threshold,
        )

        return await self.send_alert(alert)

    async def alert_pipeline_failure(
        self,
        pipeline_name: str,
        success_rate: float,
        error_message: Optional[str] = None
    ) -> bool:
        """Create and send alert for pipeline failures."""
        if success_rate <= self.thresholds.pipeline_success_rate_critical:
            severity = AlertSeverity.CRITICAL
        elif success_rate <= self.thresholds.pipeline_success_rate_warning:
            severity = AlertSeverity.WARNING
        else:
            return False

        threshold = (
            self.thresholds.pipeline_success_rate_critical
            if severity == AlertSeverity.CRITICAL
            else self.thresholds.pipeline_success_rate_warning
        )

        description = (
            f'Data pipeline "{pipeline_name}" success rate has dropped. '
            f'Current: {success_rate*100:.1f}%, Threshold: {threshold*100:.1f}%'
        )
        if error_message:
            description += f'\nLast error: {error_message}'

        alert = BusinessAlert(
            alert_name='pipeline_failure',
            severity=severity,
            summary=f'Pipeline {pipeline_name} success rate: {success_rate*100:.1f}%',
            description=description,
            labels={
                'component': 'pipeline',
                'pipeline_name': pipeline_name,
                'metric': 'success_rate',
            },
            metric_value=success_rate,
            threshold_value=threshold,
        )

        return await self.send_alert(alert)

    async def alert_ml_model_degradation(
        self,
        model_name: str,
        accuracy: float,
        prediction_type: Optional[str] = None
    ) -> bool:
        """Create and send alert for ML model accuracy degradation."""
        if accuracy <= self.thresholds.ml_accuracy_critical:
            severity = AlertSeverity.CRITICAL
        elif accuracy <= self.thresholds.ml_accuracy_warning:
            severity = AlertSeverity.WARNING
        else:
            return False

        threshold = (
            self.thresholds.ml_accuracy_critical
            if severity == AlertSeverity.CRITICAL
            else self.thresholds.ml_accuracy_warning
        )

        labels = {
            'component': 'ml',
            'model_name': model_name,
            'metric': 'accuracy',
        }
        if prediction_type:
            labels['prediction_type'] = prediction_type

        alert = BusinessAlert(
            alert_name='ml_model_degradation',
            severity=severity,
            summary=f'ML model {model_name} accuracy: {accuracy*100:.1f}%',
            description=(
                f'ML model "{model_name}" accuracy has dropped below threshold. '
                f'Current: {accuracy*100:.1f}%, Threshold: {threshold*100:.1f}%'
            ),
            labels=labels,
            metric_value=accuracy,
            threshold_value=threshold,
        )

        return await self.send_alert(alert)

    async def alert_job_failure(
        self,
        job_name: str,
        error_message: str,
        job_type: Optional[str] = None
    ) -> bool:
        """Create and send alert for failed jobs/tasks."""
        labels = {
            'component': 'jobs',
            'job_name': job_name,
        }
        if job_type:
            labels['job_type'] = job_type

        alert = BusinessAlert(
            alert_name='job_failure',
            severity=AlertSeverity.CRITICAL,
            summary=f'Job {job_name} failed',
            description=f'Job "{job_name}" has failed. Error: {error_message}',
            labels=labels,
            annotations={
                'error_message': error_message[:500],  # Truncate long errors
            },
        )

        return await self.send_alert(alert)

    def get_active_alerts(self) -> List[BusinessAlert]:
        """Get list of currently active alerts."""
        return list(self._active_alerts.values())

    def get_stats(self) -> Dict[str, Any]:
        """Get webhook and alerting statistics."""
        return {
            'enabled_targets': list(self._enabled_targets),
            'active_alerts': len(self._active_alerts),
            'rate_limiter': self.rate_limiter.get_stats(),
            'thresholds': {
                'api_error_rate_warning': self.thresholds.api_error_rate_warning,
                'api_error_rate_critical': self.thresholds.api_error_rate_critical,
                'response_time_warning': self.thresholds.response_time_p99_warning,
                'response_time_critical': self.thresholds.response_time_p99_critical,
                'cache_hit_rate_warning': self.thresholds.cache_hit_rate_warning,
                'cache_hit_rate_critical': self.thresholds.cache_hit_rate_critical,
                'memory_usage_warning': self.thresholds.memory_usage_warning,
                'memory_usage_critical': self.thresholds.memory_usage_critical,
                'budget_usage_warning': self.thresholds.budget_usage_warning,
                'budget_usage_critical': self.thresholds.budget_usage_critical,
            }
        }


# Global webhook instance
alert_webhook = AlertManagerWebhook()


# Convenience functions for direct use
async def send_api_error_alert(error_rate: float, endpoint: Optional[str] = None) -> bool:
    """Convenience function to send API error rate alert."""
    return await alert_webhook.alert_high_api_error_rate(error_rate, endpoint)


async def send_latency_alert(p99_latency: float, endpoint: Optional[str] = None) -> bool:
    """Convenience function to send latency alert."""
    return await alert_webhook.alert_slow_response_time(p99_latency, endpoint)


async def send_cache_alert(hit_rate: float, cache_name: Optional[str] = None) -> bool:
    """Convenience function to send cache hit rate alert."""
    return await alert_webhook.alert_low_cache_hit_rate(hit_rate, cache_name)


async def send_memory_alert(memory_percent: float, process: Optional[str] = None) -> bool:
    """Convenience function to send memory usage alert."""
    return await alert_webhook.alert_high_memory_usage(memory_percent, process)


async def send_budget_alert(current_spend: float, budget: float) -> bool:
    """Convenience function to send budget alert."""
    return await alert_webhook.alert_budget_exceeded(current_spend, budget)


async def send_pipeline_alert(
    pipeline_name: str,
    success_rate: float,
    error: Optional[str] = None
) -> bool:
    """Convenience function to send pipeline failure alert."""
    return await alert_webhook.alert_pipeline_failure(pipeline_name, success_rate, error)


async def send_ml_alert(
    model_name: str,
    accuracy: float,
    prediction_type: Optional[str] = None
) -> bool:
    """Convenience function to send ML model alert."""
    return await alert_webhook.alert_ml_model_degradation(model_name, accuracy, prediction_type)


async def send_job_alert(
    job_name: str,
    error_message: str,
    job_type: Optional[str] = None
) -> bool:
    """Convenience function to send job failure alert."""
    return await alert_webhook.alert_job_failure(job_name, error_message, job_type)
