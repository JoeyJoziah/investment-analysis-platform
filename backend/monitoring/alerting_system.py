"""
Comprehensive Alerting System
Multi-channel alerting with intelligent routing, escalation, and deduplication.
"""

import asyncio
import logging
import time
import hashlib
import json
from typing import Dict, Any, Optional, List, Set, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import aiohttp
import asyncpg
from prometheus_client import Counter, Gauge, Histogram

from backend.config.monitoring_config import monitoring_config
from backend.utils.structured_logging import get_structured_logger

logger = get_structured_logger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertStatus(Enum):
    """Alert status."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged" 
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class NotificationChannel(Enum):
    """Notification channels."""
    EMAIL = "email"
    SLACK = "slack"
    PAGERDUTY = "pagerduty"
    WEBHOOK = "webhook"
    SMS = "sms"


# Alerting Metrics
alerts_generated = Counter(
    'alerts_generated_total',
    'Total alerts generated',
    ['severity', 'source', 'alert_type']
)

alerts_sent = Counter(
    'alerts_sent_total',
    'Total alert notifications sent',
    ['channel', 'severity', 'success']
)

alert_escalations = Counter(
    'alert_escalations_total',
    'Total alert escalations',
    ['from_severity', 'to_severity', 'reason']
)

alert_suppression_rate = Gauge(
    'alert_suppression_rate_percent',
    'Alert suppression rate percentage',
    ['suppression_type']
)

alert_resolution_time = Histogram(
    'alert_resolution_time_seconds',
    'Alert resolution time in seconds',
    ['severity', 'alert_type']
)

notification_latency = Histogram(
    'notification_latency_seconds',
    'Notification delivery latency',
    ['channel']
)

alert_accuracy_rate = Gauge(
    'alert_accuracy_rate_percent',
    'Alert accuracy rate (true positive rate)',
    ['alert_type', 'severity']
)


@dataclass
class Alert:
    """Alert data structure."""
    id: str
    title: str
    description: str
    severity: AlertSeverity
    source: str
    alert_type: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: AlertStatus = AlertStatus.ACTIVE
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    escalation_count: int = 0
    fingerprint: Optional[str] = None
    
    def __post_init__(self):
        if not self.fingerprint:
            self.fingerprint = self._generate_fingerprint()
    
    def _generate_fingerprint(self) -> str:
        """Generate unique fingerprint for deduplication."""
        content = f"{self.source}:{self.alert_type}:{self.title}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'severity': self.severity.value,
            'source': self.source,
            'alert_type': self.alert_type,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata,
            'status': self.status.value,
            'acknowledged_by': self.acknowledged_by,
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'escalation_count': self.escalation_count,
            'fingerprint': self.fingerprint
        }


@dataclass
class NotificationConfig:
    """Notification channel configuration."""
    channel: NotificationChannel
    enabled: bool
    config: Dict[str, Any]
    severity_filter: Set[AlertSeverity]
    rate_limit: Optional[int] = None  # Max notifications per hour
    escalation_delay: Optional[int] = None  # Minutes before escalation


@dataclass
class AlertRule:
    """Alert rule definition."""
    name: str
    condition: str
    severity: AlertSeverity
    description: str
    source: str
    alert_type: str
    enabled: bool = True
    cooldown_minutes: int = 15
    threshold_config: Dict[str, Any] = field(default_factory=dict)
    metadata_template: Dict[str, Any] = field(default_factory=dict)


class NotificationHandler:
    """Base notification handler."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rate_limiter: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
    
    async def send_notification(self, alert: Alert) -> bool:
        """Send notification. Must be implemented by subclasses."""
        raise NotImplementedError
    
    def is_rate_limited(self, rate_limit: Optional[int]) -> bool:
        """Check if notifications are rate limited."""
        if not rate_limit:
            return False
        
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        
        # Clean old entries
        while self.rate_limiter['notifications'] and self.rate_limiter['notifications'][0] < hour_ago:
            self.rate_limiter['notifications'].popleft()
        
        return len(self.rate_limiter['notifications']) >= rate_limit
    
    def record_notification(self):
        """Record notification for rate limiting."""
        self.rate_limiter['notifications'].append(datetime.now())


class EmailHandler(NotificationHandler):
    """Email notification handler."""
    
    async def send_notification(self, alert: Alert) -> bool:
        """Send email notification."""
        try:
            smtp_host = self.config.get('smtp_host')
            smtp_port = self.config.get('smtp_port', 587)
            from_address = self.config.get('from_address')
            to_addresses = self.config.get('to_addresses', [])
            username = self.config.get('username')
            password = self.config.get('password')
            
            if not all([smtp_host, from_address, to_addresses]):
                logger.error("Email configuration incomplete")
                return False
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = from_address
            msg['To'] = ', '.join(to_addresses)
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            
            # Email body
            body = self._format_email_body(alert)
            msg.attach(MIMEText(body, 'html'))
            
            # Send email
            with smtplib.SMTP(smtp_host, smtp_port) as server:
                server.starttls()
                if username and password:
                    server.login(username, password)
                text = msg.as_string()
                server.sendmail(from_address, to_addresses, text)
            
            self.record_notification()
            return True
        
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            return False
    
    def _format_email_body(self, alert: Alert) -> str:
        """Format email body."""
        severity_colors = {
            AlertSeverity.INFO: "#17a2b8",
            AlertSeverity.WARNING: "#ffc107", 
            AlertSeverity.CRITICAL: "#dc3545",
            AlertSeverity.EMERGENCY: "#6f42c1"
        }
        
        color = severity_colors.get(alert.severity, "#6c757d")
        
        return f"""
        <html>
        <body>
            <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
                <div style="background-color: {color}; color: white; padding: 20px; border-radius: 5px 5px 0 0;">
                    <h1 style="margin: 0; font-size: 24px;">{alert.severity.value.upper()} Alert</h1>
                    <p style="margin: 5px 0 0 0; font-size: 16px;">{alert.title}</p>
                </div>
                
                <div style="background-color: #f8f9fa; padding: 20px; border: 1px solid #dee2e6;">
                    <h2 style="color: #495057; font-size: 18px;">Alert Details</h2>
                    
                    <table style="width: 100%; border-collapse: collapse;">
                        <tr>
                            <td style="padding: 8px; font-weight: bold; color: #495057;">Source:</td>
                            <td style="padding: 8px; color: #212529;">{alert.source}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px; font-weight: bold; color: #495057;">Type:</td>
                            <td style="padding: 8px; color: #212529;">{alert.alert_type}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px; font-weight: bold; color: #495057;">Time:</td>
                            <td style="padding: 8px; color: #212529;">{alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px; font-weight: bold; color: #495057;">Alert ID:</td>
                            <td style="padding: 8px; color: #212529;">{alert.id}</td>
                        </tr>
                    </table>
                    
                    <h3 style="color: #495057; font-size: 16px; margin-top: 20px;">Description</h3>
                    <p style="color: #212529; background-color: white; padding: 15px; border-left: 4px solid {color}; margin: 0;">
                        {alert.description}
                    </p>
                    
                    {self._format_metadata(alert.metadata) if alert.metadata else ''}
                </div>
                
                <div style="background-color: #e9ecef; padding: 15px; border-radius: 0 0 5px 5px; font-size: 12px; color: #6c757d;">
                    <p style="margin: 0;">Investment Analysis Application Monitoring System</p>
                    <p style="margin: 5px 0 0 0;">This is an automated alert. Please do not reply to this email.</p>
                </div>
            </div>
        </body>
        </html>
        """
    
    def _format_metadata(self, metadata: Dict[str, Any]) -> str:
        """Format metadata for email."""
        if not metadata:
            return ""
        
        rows = []
        for key, value in metadata.items():
            if isinstance(value, (dict, list)):
                value = json.dumps(value, indent=2)
            rows.append(f"<tr><td style='padding: 5px; font-weight: bold;'>{key}:</td><td style='padding: 5px;'>{value}</td></tr>")
        
        return f"""
        <h3 style="color: #495057; font-size: 16px; margin-top: 20px;">Additional Information</h3>
        <table style="width: 100%; background-color: white; border-collapse: collapse;">
            {''.join(rows)}
        </table>
        """


class SlackHandler(NotificationHandler):
    """Slack notification handler."""
    
    async def send_notification(self, alert: Alert) -> bool:
        """Send Slack notification."""
        try:
            webhook_url = self.config.get('webhook_url')
            if not webhook_url:
                logger.error("Slack webhook URL not configured")
                return False
            
            # Severity colors
            colors = {
                AlertSeverity.INFO: "#17a2b8",
                AlertSeverity.WARNING: "#ffc107",
                AlertSeverity.CRITICAL: "#dc3545", 
                AlertSeverity.EMERGENCY: "#6f42c1"
            }
            
            severity_emojis = {
                AlertSeverity.INFO: ":information_source:",
                AlertSeverity.WARNING: ":warning:",
                AlertSeverity.CRITICAL: ":rotating_light:",
                AlertSeverity.EMERGENCY: ":fire:"
            }
            
            # Build Slack message
            payload = {
                "username": self.config.get('username', 'Investment Analysis Bot'),
                "channel": self.config.get('channel', '#alerts'),
                "icon_emoji": ":chart_with_upwards_trend:",
                "attachments": [
                    {
                        "color": colors.get(alert.severity, "#6c757d"),
                        "title": f"{severity_emojis.get(alert.severity, ':bell:')} {alert.severity.value.upper()} Alert",
                        "title_link": self._get_alert_url(alert.id),
                        "text": alert.title,
                        "fields": [
                            {
                                "title": "Description",
                                "value": alert.description[:500] + ("..." if len(alert.description) > 500 else ""),
                                "short": False
                            },
                            {
                                "title": "Source",
                                "value": alert.source,
                                "short": True
                            },
                            {
                                "title": "Type", 
                                "value": alert.alert_type,
                                "short": True
                            },
                            {
                                "title": "Time",
                                "value": alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC'),
                                "short": True
                            },
                            {
                                "title": "Alert ID",
                                "value": f"`{alert.id}`",
                                "short": True
                            }
                        ],
                        "footer": "Investment Analysis Monitoring",
                        "ts": int(alert.timestamp.timestamp())
                    }
                ]
            }
            
            # Add metadata fields if present
            if alert.metadata:
                for key, value in alert.metadata.items():
                    if len(payload["attachments"][0]["fields"]) < 10:  # Slack limit
                        payload["attachments"][0]["fields"].append({
                            "title": key,
                            "value": str(value)[:100],
                            "short": True
                        })
            
            # Send to Slack
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload) as response:
                    if response.status == 200:
                        self.record_notification()
                        return True
                    else:
                        logger.error(f"Slack notification failed: {response.status}")
                        return False
        
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
            return False
    
    def _get_alert_url(self, alert_id: str) -> str:
        """Get URL for alert details."""
        base_url = self.config.get('alert_base_url', 'http://localhost:3000')
        return f"{base_url}/alerts/{alert_id}"


class PagerDutyHandler(NotificationHandler):
    """PagerDuty notification handler."""
    
    async def send_notification(self, alert: Alert) -> bool:
        """Send PagerDuty notification."""
        try:
            integration_key = self.config.get('integration_key')
            if not integration_key:
                logger.error("PagerDuty integration key not configured")
                return False
            
            # Map severity to PagerDuty severity
            severity_map = {
                AlertSeverity.INFO: "info",
                AlertSeverity.WARNING: "warning",
                AlertSeverity.CRITICAL: "error",
                AlertSeverity.EMERGENCY: "critical"
            }
            
            payload = {
                "routing_key": integration_key,
                "event_action": "trigger",
                "dedup_key": alert.fingerprint,
                "payload": {
                    "summary": alert.title,
                    "source": alert.source,
                    "severity": severity_map.get(alert.severity, "error"),
                    "timestamp": alert.timestamp.isoformat(),
                    "component": alert.alert_type,
                    "group": alert.source,
                    "class": alert.alert_type,
                    "custom_details": {
                        "description": alert.description,
                        "alert_id": alert.id,
                        "metadata": alert.metadata
                    }
                },
                "links": [
                    {
                        "href": self._get_alert_url(alert.id),
                        "text": "View Alert Details"
                    }
                ]
            }
            
            # Send to PagerDuty
            url = "https://events.pagerduty.com/v2/enqueue"
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 202:
                        self.record_notification()
                        return True
                    else:
                        logger.error(f"PagerDuty notification failed: {response.status}")
                        return False
        
        except Exception as e:
            logger.error(f"Failed to send PagerDuty notification: {e}")
            return False
    
    def _get_alert_url(self, alert_id: str) -> str:
        """Get URL for alert details."""
        base_url = self.config.get('alert_base_url', 'http://localhost:3000')
        return f"{base_url}/alerts/{alert_id}"


class WebhookHandler(NotificationHandler):
    """Generic webhook notification handler."""
    
    async def send_notification(self, alert: Alert) -> bool:
        """Send webhook notification."""
        try:
            webhook_url = self.config.get('url')
            if not webhook_url:
                logger.error("Webhook URL not configured")
                return False
            
            payload = {
                "event": "alert",
                "alert": alert.to_dict(),
                "timestamp": datetime.now().isoformat(),
                "source": "investment_analysis_monitoring"
            }
            
            # Add custom headers if configured
            headers = self.config.get('headers', {})
            headers.setdefault('Content-Type', 'application/json')
            
            # Send webhook
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(webhook_url, json=payload, headers=headers) as response:
                    if 200 <= response.status < 300:
                        self.record_notification()
                        return True
                    else:
                        logger.error(f"Webhook notification failed: {response.status}")
                        return False
        
        except Exception as e:
            logger.error(f"Failed to send webhook notification: {e}")
            return False


class AlertManager:
    """
    Comprehensive alert management system.
    """
    
    def __init__(self):
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=10000)
        self.suppression_rules: Dict[str, Dict] = {}
        self.notification_handlers: Dict[NotificationChannel, NotificationHandler] = {}
        self.alert_rules: Dict[str, AlertRule] = {}
        
        # Deduplication and grouping
        self.alert_groups: Dict[str, List[str]] = defaultdict(list)
        self.suppressed_fingerprints: Set[str] = set()
        self.maintenance_mode: bool = False
        self.maintenance_end_time: Optional[datetime] = None
        
        # Rate limiting and cooldowns
        self.alert_cooldowns: Dict[str, datetime] = {}
        self.escalation_timers: Dict[str, asyncio.Task] = {}
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._escalation_task: Optional[asyncio.Task] = None
        
        self._setup_notification_handlers()
    
    def _setup_notification_handlers(self):
        """Setup notification handlers based on configuration."""
        config = monitoring_config.alerting.notification_channels
        
        # Email handler
        if config['email']['enabled']:
            self.notification_handlers[NotificationChannel.EMAIL] = EmailHandler(config['email'])
        
        # Slack handler  
        if config['slack']['enabled']:
            self.notification_handlers[NotificationChannel.SLACK] = SlackHandler(config['slack'])
        
        # PagerDuty handler
        if config['pagerduty']['enabled']:
            self.notification_handlers[NotificationChannel.PAGERDUTY] = PagerDutyHandler(config['pagerduty'])
    
    async def start(self):
        """Start alert manager background tasks."""
        if not self._cleanup_task:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        if not self._escalation_task:
            self._escalation_task = asyncio.create_task(self._escalation_loop())
        
        logger.info("Alert manager started")
    
    async def stop(self):
        """Stop alert manager background tasks."""
        tasks = [self._cleanup_task, self._escalation_task]
        for task in tasks:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Cancel escalation timers
        for timer in self.escalation_timers.values():
            timer.cancel()
        
        logger.info("Alert manager stopped")
    
    async def create_alert(
        self,
        title: str,
        description: str,
        severity: AlertSeverity,
        source: str,
        alert_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Alert]:
        """Create and process a new alert."""
        try:
            # Generate alert ID
            alert_id = f"alert_{int(time.time() * 1000)}"
            
            # Create alert
            alert = Alert(
                id=alert_id,
                title=title,
                description=description,
                severity=severity,
                source=source,
                alert_type=alert_type,
                timestamp=datetime.now(),
                metadata=metadata or {}
            )
            
            # Check suppression and deduplication
            if await self._should_suppress_alert(alert):
                logger.debug(f"Alert suppressed: {alert.fingerprint}")
                return None
            
            # Check cooldown
            cooldown_key = f"{source}:{alert_type}"
            if cooldown_key in self.alert_cooldowns:
                cooldown_end = self.alert_cooldowns[cooldown_key]
                if datetime.now() < cooldown_end:
                    logger.debug(f"Alert in cooldown: {cooldown_key}")
                    return None
            
            # Check for existing alert with same fingerprint
            existing_alert = self._find_existing_alert(alert.fingerprint)
            if existing_alert:
                # Update existing alert instead of creating new one
                existing_alert.timestamp = alert.timestamp
                existing_alert.metadata.update(alert.metadata)
                alert = existing_alert
            else:
                # Add to active alerts
                self.active_alerts[alert.id] = alert
            
            # Record metrics
            alerts_generated.labels(
                severity=severity.value,
                source=source,
                alert_type=alert_type
            ).inc()
            
            # Send notifications
            await self._send_notifications(alert)
            
            # Set cooldown
            rule = self.alert_rules.get(f"{source}:{alert_type}")
            cooldown_minutes = rule.cooldown_minutes if rule else 15
            self.alert_cooldowns[cooldown_key] = datetime.now() + timedelta(minutes=cooldown_minutes)
            
            # Schedule escalation if needed
            await self._schedule_escalation(alert)
            
            # Add to history
            self.alert_history.append(alert)
            
            logger.info(
                f"Alert created: {alert.id}",
                extra={
                    "alert_id": alert.id,
                    "severity": severity.value,
                    "source": source,
                    "type": alert_type,
                    "fingerprint": alert.fingerprint
                }
            )
            
            return alert
        
        except Exception as e:
            logger.error(f"Error creating alert: {e}")
            return None
    
    def _find_existing_alert(self, fingerprint: str) -> Optional[Alert]:
        """Find existing alert with same fingerprint."""
        for alert in self.active_alerts.values():
            if alert.fingerprint == fingerprint and alert.status == AlertStatus.ACTIVE:
                return alert
        return None
    
    async def _should_suppress_alert(self, alert: Alert) -> bool:
        """Check if alert should be suppressed."""
        # Maintenance mode check
        if self.maintenance_mode:
            if not self.maintenance_end_time or datetime.now() < self.maintenance_end_time:
                return True
        
        # Fingerprint suppression
        if alert.fingerprint in self.suppressed_fingerprints:
            return True
        
        # Custom suppression rules
        for rule_name, rule_config in self.suppression_rules.items():
            if await self._matches_suppression_rule(alert, rule_config):
                logger.debug(f"Alert suppressed by rule: {rule_name}")
                return True
        
        return False
    
    async def _matches_suppression_rule(self, alert: Alert, rule: Dict[str, Any]) -> bool:
        """Check if alert matches suppression rule."""
        try:
            # Source matching
            if 'sources' in rule and alert.source not in rule['sources']:
                return False
            
            # Severity matching
            if 'severities' in rule:
                severity_names = [s.value for s in rule['severities']]
                if alert.severity.value not in severity_names:
                    return False
            
            # Type matching
            if 'alert_types' in rule and alert.alert_type not in rule['alert_types']:
                return False
            
            # Time window matching
            if 'time_windows' in rule:
                current_time = datetime.now().time()
                for window in rule['time_windows']:
                    start_time = datetime.strptime(window['start'], '%H:%M').time()
                    end_time = datetime.strptime(window['end'], '%H:%M').time()
                    if start_time <= current_time <= end_time:
                        return True
                return False
            
            return True
        
        except Exception as e:
            logger.error(f"Error checking suppression rule: {e}")
            return False
    
    async def _send_notifications(self, alert: Alert):
        """Send alert notifications to configured channels."""
        tasks = []
        
        for channel, handler in self.notification_handlers.items():
            # Check if channel should receive this severity
            if self._should_notify_channel(channel, alert.severity):
                task = asyncio.create_task(self._send_notification_with_metrics(channel, handler, alert))
                tasks.append(task)
        
        # Send all notifications concurrently
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    def _should_notify_channel(self, channel: NotificationChannel, severity: AlertSeverity) -> bool:
        """Check if channel should receive notifications for this severity."""
        config = monitoring_config.alerting.notification_channels
        
        severity_filters = {
            NotificationChannel.EMAIL: [AlertSeverity.WARNING, AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY],
            NotificationChannel.SLACK: [AlertSeverity.WARNING, AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY],
            NotificationChannel.PAGERDUTY: [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]
        }
        
        return severity in severity_filters.get(channel, [AlertSeverity.INFO])
    
    async def _send_notification_with_metrics(
        self, 
        channel: NotificationChannel, 
        handler: NotificationHandler, 
        alert: Alert
    ):
        """Send notification and record metrics."""
        start_time = time.time()
        success = False
        
        try:
            success = await handler.send_notification(alert)
        except Exception as e:
            logger.error(f"Notification handler error ({channel.value}): {e}")
        finally:
            # Record metrics
            duration = time.time() - start_time
            notification_latency.labels(channel=channel.value).observe(duration)
            
            alerts_sent.labels(
                channel=channel.value,
                severity=alert.severity.value,
                success=str(success).lower()
            ).inc()
            
            if success:
                logger.info(f"Alert notification sent via {channel.value}: {alert.id}")
            else:
                logger.error(f"Failed to send alert notification via {channel.value}: {alert.id}")
    
    async def _schedule_escalation(self, alert: Alert):
        """Schedule alert escalation if needed."""
        if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]:
            escalation_delay = 30  # 30 minutes for critical alerts
            
            async def escalate():
                await asyncio.sleep(escalation_delay * 60)
                if alert.id in self.active_alerts and alert.status == AlertStatus.ACTIVE:
                    await self._escalate_alert(alert)
            
            self.escalation_timers[alert.id] = asyncio.create_task(escalate())
    
    async def _escalate_alert(self, alert: Alert):
        """Escalate alert to higher severity."""
        try:
            old_severity = alert.severity
            
            # Escalate severity
            if alert.severity == AlertSeverity.WARNING:
                alert.severity = AlertSeverity.CRITICAL
            elif alert.severity == AlertSeverity.CRITICAL:
                alert.severity = AlertSeverity.EMERGENCY
            
            alert.escalation_count += 1
            
            # Record escalation
            alert_escalations.labels(
                from_severity=old_severity.value,
                to_severity=alert.severity.value,
                reason="timeout"
            ).inc()
            
            # Send notifications for escalated alert
            await self._send_notifications(alert)
            
            logger.warning(
                f"Alert escalated: {alert.id} from {old_severity.value} to {alert.severity.value}",
                extra={"alert_id": alert.id, "escalation_count": alert.escalation_count}
            )
            
            # Schedule next escalation if still critical
            if alert.severity == AlertSeverity.CRITICAL:
                await self._schedule_escalation(alert)
        
        except Exception as e:
            logger.error(f"Error escalating alert {alert.id}: {e}")
    
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert."""
        try:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_by = acknowledged_by
                alert.acknowledged_at = datetime.now()
                
                # Cancel escalation timer
                if alert_id in self.escalation_timers:
                    self.escalation_timers[alert_id].cancel()
                    del self.escalation_timers[alert_id]
                
                logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
                return True
        
        except Exception as e:
            logger.error(f"Error acknowledging alert {alert_id}: {e}")
        
        return False
    
    async def resolve_alert(self, alert_id: str, resolved_by: str = "system") -> bool:
        """Resolve an alert."""
        try:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = datetime.now()
                
                # Calculate resolution time
                resolution_time = (alert.resolved_at - alert.timestamp).total_seconds()
                alert_resolution_time.labels(
                    severity=alert.severity.value,
                    alert_type=alert.alert_type
                ).observe(resolution_time)
                
                # Cancel escalation timer
                if alert_id in self.escalation_timers:
                    self.escalation_timers[alert_id].cancel()
                    del self.escalation_timers[alert_id]
                
                # Remove from active alerts
                del self.active_alerts[alert_id]
                
                logger.info(f"Alert resolved: {alert_id} by {resolved_by}")
                return True
        
        except Exception as e:
            logger.error(f"Error resolving alert {alert_id}: {e}")
        
        return False
    
    def enable_maintenance_mode(self, duration_minutes: int = 60):
        """Enable maintenance mode to suppress alerts."""
        self.maintenance_mode = True
        self.maintenance_end_time = datetime.now() + timedelta(minutes=duration_minutes)
        logger.info(f"Maintenance mode enabled for {duration_minutes} minutes")
    
    def disable_maintenance_mode(self):
        """Disable maintenance mode."""
        self.maintenance_mode = False
        self.maintenance_end_time = None
        logger.info("Maintenance mode disabled")
    
    async def _cleanup_loop(self):
        """Background cleanup of old alerts and expired data."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Clean expired cooldowns
                now = datetime.now()
                expired_cooldowns = [
                    key for key, expiry in self.alert_cooldowns.items()
                    if now > expiry
                ]
                for key in expired_cooldowns:
                    del self.alert_cooldowns[key]
                
                # Clean old alert history
                cutoff_time = now - timedelta(days=30)
                while self.alert_history and self.alert_history[0].timestamp < cutoff_time:
                    self.alert_history.popleft()
                
                # Update suppression metrics
                total_fingerprints = len(self.suppressed_fingerprints)
                if total_fingerprints > 0:
                    suppression_rate = (len(self.suppressed_fingerprints) / 
                                      (len(self.active_alerts) + len(self.suppressed_fingerprints))) * 100
                    alert_suppression_rate.labels(suppression_type="fingerprint").set(suppression_rate)
                
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    async def _escalation_loop(self):
        """Background escalation processing."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Check for alerts that need escalation
                now = datetime.now()
                for alert in list(self.active_alerts.values()):
                    if (alert.status == AlertStatus.ACTIVE and
                        alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY] and
                        alert.id not in self.escalation_timers):
                        
                        # Check if alert is old enough for escalation
                        age_minutes = (now - alert.timestamp).total_seconds() / 60
                        if age_minutes > 30:  # 30 minutes threshold
                            await self._escalate_alert(alert)
                
            except Exception as e:
                logger.error(f"Error in escalation loop: {e}")
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get comprehensive alert summary."""
        try:
            severity_counts = defaultdict(int)
            status_counts = defaultdict(int)
            
            for alert in self.active_alerts.values():
                severity_counts[alert.severity.value] += 1
                status_counts[alert.status.value] += 1
            
            return {
                "timestamp": datetime.now().isoformat(),
                "active_alerts": len(self.active_alerts),
                "maintenance_mode": self.maintenance_mode,
                "maintenance_end": self.maintenance_end_time.isoformat() if self.maintenance_end_time else None,
                "severity_breakdown": dict(severity_counts),
                "status_breakdown": dict(status_counts),
                "suppressed_fingerprints": len(self.suppressed_fingerprints),
                "escalation_timers": len(self.escalation_timers),
                "cooldowns_active": len(self.alert_cooldowns),
                "history_size": len(self.alert_history)
            }
        
        except Exception as e:
            logger.error(f"Error generating alert summary: {e}")
            return {"error": str(e)}


# Global alert manager
alert_manager = AlertManager()


# Convenience functions for creating alerts
async def create_budget_alert(current_cost: float, monthly_budget: float):
    """Create budget-related alert."""
    percentage = (current_cost / monthly_budget) * 100
    
    if percentage >= monitoring_config.alerting.thresholds["budget_critical"]:
        severity = AlertSeverity.CRITICAL
        title = f"Monthly Budget Critical: {percentage:.1f}% used"
    elif percentage >= monitoring_config.alerting.thresholds["budget_warning"]:
        severity = AlertSeverity.WARNING
        title = f"Monthly Budget Warning: {percentage:.1f}% used"
    else:
        return
    
    await alert_manager.create_alert(
        title=title,
        description=f"Current monthly cost is ${current_cost:.2f} out of ${monthly_budget:.2f} budget ({percentage:.1f}%)",
        severity=severity,
        source="cost_monitor",
        alert_type="budget_exceeded",
        metadata={
            "current_cost": current_cost,
            "monthly_budget": monthly_budget,
            "percentage_used": percentage
        }
    )


async def create_data_quality_alert(data_type: str, quality_score: float):
    """Create data quality alert."""
    if quality_score < monitoring_config.alerting.thresholds["data_quality_critical"]:
        severity = AlertSeverity.CRITICAL
        title = f"Data Quality Critical: {data_type} score {quality_score:.1f}"
    elif quality_score < monitoring_config.alerting.thresholds["data_quality_warning"]:
        severity = AlertSeverity.WARNING
        title = f"Data Quality Warning: {data_type} score {quality_score:.1f}"
    else:
        return
    
    await alert_manager.create_alert(
        title=title,
        description=f"Data quality score for {data_type} has dropped to {quality_score:.1f}",
        severity=severity,
        source="data_quality",
        alert_type="quality_degradation",
        metadata={
            "data_type": data_type,
            "quality_score": quality_score
        }
    )


async def create_api_performance_alert(endpoint: str, latency: float):
    """Create API performance alert."""
    if latency >= monitoring_config.alerting.thresholds["api_latency_critical"]:
        severity = AlertSeverity.CRITICAL
        title = f"API Performance Critical: {endpoint} latency {latency:.2f}s"
    elif latency >= monitoring_config.alerting.thresholds["api_latency_warning"]:
        severity = AlertSeverity.WARNING
        title = f"API Performance Warning: {endpoint} latency {latency:.2f}s"
    else:
        return
    
    await alert_manager.create_alert(
        title=title,
        description=f"API endpoint {endpoint} is experiencing high latency of {latency:.2f} seconds",
        severity=severity,
        source="api_monitor",
        alert_type="high_latency",
        metadata={
            "endpoint": endpoint,
            "latency_seconds": latency
        }
    )


# Setup function
async def setup_alerting_system():
    """Setup alerting system."""
    await alert_manager.start()
    logger.info("Alerting system setup completed")