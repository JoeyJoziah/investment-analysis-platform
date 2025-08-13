"""
Advanced Log Analysis and Management System
Provides log aggregation, pattern detection, and automated analysis.
"""

import asyncio
import logging
import json
import re
import gzip
from typing import Dict, Any, Optional, List, Tuple, Set
from datetime import datetime, timedelta
from collections import defaultdict, deque, Counter
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import statistics

from prometheus_client import Counter as PrometheusCounter, Gauge, Histogram
import aiofiles
import aiofiles.os
from elasticsearch import AsyncElasticsearch

from backend.config.monitoring_config import monitoring_config
from backend.utils.structured_logging import get_structured_logger

logger = get_structured_logger(__name__)


class LogLevel(Enum):
    """Log levels for categorization."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AlertCategory(Enum):
    """Log-based alert categories."""
    ERROR_SPIKE = "error_spike"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    SECURITY_INCIDENT = "security_incident"
    BUSINESS_ANOMALY = "business_anomaly"
    SYSTEM_FAILURE = "system_failure"


# Log Analysis Metrics
log_entries_processed = PrometheusCounter(
    'log_entries_processed_total',
    'Total log entries processed',
    ['level', 'service', 'category']
)

log_patterns_detected = PrometheusCounter(
    'log_patterns_detected_total',
    'Log patterns detected',
    ['pattern_type', 'severity']
)

log_anomalies_detected = PrometheusCounter(
    'log_anomalies_detected_total',
    'Log anomalies detected',
    ['anomaly_type', 'service']
)

log_processing_latency = Histogram(
    'log_processing_latency_seconds',
    'Log processing latency',
    ['processor_type']
)

error_rate_by_service = Gauge(
    'log_error_rate_by_service',
    'Error rate by service from logs',
    ['service', 'time_window']
)

suspicious_activity_score = Gauge(
    'suspicious_activity_score',
    'Suspicious activity score from logs',
    ['activity_type', 'source_ip']
)


@dataclass
class LogEntry:
    """Structured log entry representation."""
    timestamp: datetime
    level: LogLevel
    service: str
    message: str
    correlation_id: Optional[str] = None
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_file: Optional[str] = None
    line_number: Optional[int] = None
    exception_info: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_json(cls, json_str: str) -> 'LogEntry':
        """Create LogEntry from JSON string."""
        try:
            data = json.loads(json_str)
            
            # Parse timestamp
            timestamp_str = data.get('timestamp', data.get('@timestamp'))
            if timestamp_str:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            else:
                timestamp = datetime.now()
            
            # Parse level
            level_str = data.get('level', data.get('levelname', 'INFO')).upper()
            try:
                level = LogLevel(level_str)
            except ValueError:
                level = LogLevel.INFO
            
            return cls(
                timestamp=timestamp,
                level=level,
                service=data.get('service', 'unknown'),
                message=data.get('message', data.get('msg', '')),
                correlation_id=data.get('correlation_id'),
                request_id=data.get('request_id'),
                user_id=data.get('user_id'),
                metadata={k: v for k, v in data.items() if k not in [
                    'timestamp', '@timestamp', 'level', 'levelname', 'service',
                    'message', 'msg', 'correlation_id', 'request_id', 'user_id'
                ]},
                source_file=data.get('filename'),
                line_number=data.get('lineno'),
                exception_info=data.get('error', data.get('exception'))
            )
        
        except Exception as e:
            logger.error(f"Failed to parse log entry: {e}")
            return cls(
                timestamp=datetime.now(),
                level=LogLevel.ERROR,
                service='log_parser',
                message=f"Failed to parse log: {json_str[:100]}..."
            )


@dataclass
class LogPattern:
    """Detected log pattern."""
    pattern_id: str
    pattern_regex: str
    category: str
    severity: str
    description: str
    occurrences: int = 0
    last_seen: Optional[datetime] = None
    services_affected: Set[str] = field(default_factory=set)
    sample_messages: List[str] = field(default_factory=list)


@dataclass
class LogAnomaly:
    """Detected log anomaly."""
    anomaly_id: str
    anomaly_type: AlertCategory
    service: str
    description: str
    score: float  # 0-100 severity score
    timestamp: datetime
    affected_period: Tuple[datetime, datetime]
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    correlation_ids: Set[str] = field(default_factory=set)


class LogPatternDetector:
    """Detects patterns in log messages."""
    
    def __init__(self):
        self.patterns: Dict[str, LogPattern] = {}
        self._setup_default_patterns()
    
    def _setup_default_patterns(self):
        """Setup default log patterns to detect."""
        patterns = [
            # Error patterns
            LogPattern(
                pattern_id="sql_error",
                pattern_regex=r"(?i)(sql|database|postgresql|mysql).*?(error|exception|failed)",
                category="database",
                severity="high",
                description="Database/SQL errors"
            ),
            LogPattern(
                pattern_id="api_timeout",
                pattern_regex=r"(?i)(timeout|timed out|request timeout)",
                category="performance",
                severity="medium",
                description="API timeouts"
            ),
            LogPattern(
                pattern_id="authentication_failure",
                pattern_regex=r"(?i)(authentication|auth|login).*?(failed|error|denied|invalid)",
                category="security",
                severity="high",
                description="Authentication failures"
            ),
            LogPattern(
                pattern_id="rate_limit_exceeded",
                pattern_regex=r"(?i)(rate limit|quota|throttle).*?(exceeded|reached|limit)",
                category="quota",
                severity="medium",
                description="Rate limit violations"
            ),
            LogPattern(
                pattern_id="memory_error",
                pattern_regex=r"(?i)(out of memory|oom|memory error|malloc)",
                category="system",
                severity="critical",
                description="Memory-related errors"
            ),
            LogPattern(
                pattern_id="disk_full",
                pattern_regex=r"(?i)(disk full|no space|disk space)",
                category="system",
                severity="critical",
                description="Disk space issues"
            ),
            LogPattern(
                pattern_id="ssl_certificate",
                pattern_regex=r"(?i)(ssl|tls|certificate).*?(error|expired|invalid|failed)",
                category="security",
                severity="high",
                description="SSL/TLS certificate issues"
            ),
            LogPattern(
                pattern_id="external_api_error",
                pattern_regex=r"(?i)(alpha.vantage|finnhub|polygon).*?(error|failed|timeout)",
                category="external_api",
                severity="medium",
                description="External API failures"
            )
        ]
        
        for pattern in patterns:
            self.patterns[pattern.pattern_id] = pattern
    
    def detect_patterns(self, log_entry: LogEntry) -> List[str]:
        """Detect patterns in log entry."""
        detected = []
        
        for pattern_id, pattern in self.patterns.items():
            if re.search(pattern.pattern_regex, log_entry.message):
                pattern.occurrences += 1
                pattern.last_seen = log_entry.timestamp
                pattern.services_affected.add(log_entry.service)
                
                # Keep sample messages (max 5)
                if len(pattern.sample_messages) < 5:
                    pattern.sample_messages.append(log_entry.message)
                
                detected.append(pattern_id)
                
                # Record metrics
                log_patterns_detected.labels(
                    pattern_type=pattern_id,
                    severity=pattern.severity
                ).inc()
        
        return detected


class LogAnomalyDetector:
    """Detects anomalies in log patterns."""
    
    def __init__(self):
        self.time_windows = {
            '5m': timedelta(minutes=5),
            '15m': timedelta(minutes=15),
            '1h': timedelta(hours=1),
            '1d': timedelta(days=1)
        }
        
        self.baseline_metrics: Dict[str, Dict[str, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=1000))
        )
        
        self.anomaly_thresholds = {
            'error_rate_spike': 5.0,  # 5x normal rate
            'performance_degradation': 2.0,  # 2x normal latency
            'volume_spike': 10.0,  # 10x normal volume
            'new_error_pattern': 1,  # Any new error pattern
        }
    
    def update_baseline(self, service: str, metric: str, value: float):
        """Update baseline metrics for anomaly detection."""
        self.baseline_metrics[service][metric].append({
            'value': value,
            'timestamp': datetime.now()
        })
    
    def detect_anomalies(
        self, 
        log_entries: List[LogEntry], 
        time_window: str = '15m'
    ) -> List[LogAnomaly]:
        """Detect anomalies in log entries."""
        anomalies = []
        window = self.time_windows.get(time_window, timedelta(minutes=15))
        cutoff_time = datetime.now() - window
        
        # Filter entries to time window
        recent_entries = [
            entry for entry in log_entries 
            if entry.timestamp > cutoff_time
        ]
        
        if not recent_entries:
            return anomalies
        
        # Group by service
        service_entries = defaultdict(list)
        for entry in recent_entries:
            service_entries[entry.service].append(entry)
        
        for service, entries in service_entries.items():
            # Error rate anomaly detection
            error_anomaly = self._detect_error_rate_anomaly(service, entries, window)
            if error_anomaly:
                anomalies.append(error_anomaly)
            
            # Performance anomaly detection
            perf_anomaly = self._detect_performance_anomaly(service, entries, window)
            if perf_anomaly:
                anomalies.append(perf_anomaly)
            
            # Volume anomaly detection
            volume_anomaly = self._detect_volume_anomaly(service, entries, window)
            if volume_anomaly:
                anomalies.append(volume_anomaly)
            
            # Security anomaly detection
            security_anomaly = self._detect_security_anomaly(service, entries, window)
            if security_anomaly:
                anomalies.append(security_anomaly)
        
        return anomalies
    
    def _detect_error_rate_anomaly(
        self, 
        service: str, 
        entries: List[LogEntry], 
        window: timedelta
    ) -> Optional[LogAnomaly]:
        """Detect error rate spikes."""
        if len(entries) < 10:  # Need minimum entries
            return None
        
        error_count = sum(1 for entry in entries if entry.level in [LogLevel.ERROR, LogLevel.CRITICAL])
        total_count = len(entries)
        current_error_rate = (error_count / total_count) * 100
        
        # Get baseline error rate
        baseline_rates = [
            m['value'] for m in self.baseline_metrics[service]['error_rate']
            if datetime.now() - m['timestamp'] < timedelta(hours=24)
        ]
        
        if len(baseline_rates) < 10:
            # Not enough baseline data
            self.update_baseline(service, 'error_rate', current_error_rate)
            return None
        
        baseline_avg = statistics.mean(baseline_rates)
        baseline_std = statistics.stdev(baseline_rates) if len(baseline_rates) > 1 else 0
        
        # Check for spike
        threshold = baseline_avg + (3 * baseline_std) if baseline_std > 0 else baseline_avg * 2
        
        if current_error_rate > max(threshold, baseline_avg * self.anomaly_thresholds['error_rate_spike']):
            return LogAnomaly(
                anomaly_id=f"error_spike_{service}_{int(datetime.now().timestamp())}",
                anomaly_type=AlertCategory.ERROR_SPIKE,
                service=service,
                description=f"Error rate spike detected: {current_error_rate:.1f}% (baseline: {baseline_avg:.1f}%)",
                score=min(100.0, (current_error_rate / baseline_avg) * 20),
                timestamp=datetime.now(),
                affected_period=(datetime.now() - window, datetime.now()),
                evidence=[
                    {
                        'current_error_rate': current_error_rate,
                        'baseline_error_rate': baseline_avg,
                        'spike_factor': current_error_rate / baseline_avg if baseline_avg > 0 else float('inf'),
                        'error_count': error_count,
                        'total_entries': total_count
                    }
                ]
            )
        
        # Update baseline
        self.update_baseline(service, 'error_rate', current_error_rate)
        return None
    
    def _detect_performance_anomaly(
        self, 
        service: str, 
        entries: List[LogEntry], 
        window: timedelta
    ) -> Optional[LogAnomaly]:
        """Detect performance degradation."""
        # Extract duration information from log entries
        durations = []
        for entry in entries:
            if 'duration_ms' in entry.metadata:
                try:
                    duration = float(entry.metadata['duration_ms'])
                    durations.append(duration)
                except (ValueError, TypeError):
                    continue
        
        if len(durations) < 10:
            return None
        
        current_avg_duration = statistics.mean(durations)
        current_p95_duration = statistics.quantiles(durations, n=20)[18]  # 95th percentile
        
        # Get baseline performance
        baseline_durations = [
            m['value'] for m in self.baseline_metrics[service]['avg_duration']
            if datetime.now() - m['timestamp'] < timedelta(hours=24)
        ]
        
        if len(baseline_durations) < 10:
            self.update_baseline(service, 'avg_duration', current_avg_duration)
            self.update_baseline(service, 'p95_duration', current_p95_duration)
            return None
        
        baseline_avg = statistics.mean(baseline_durations)
        
        # Check for performance degradation
        if current_avg_duration > baseline_avg * self.anomaly_thresholds['performance_degradation']:
            return LogAnomaly(
                anomaly_id=f"perf_degradation_{service}_{int(datetime.now().timestamp())}",
                anomaly_type=AlertCategory.PERFORMANCE_DEGRADATION,
                service=service,
                description=f"Performance degradation detected: {current_avg_duration:.1f}ms avg (baseline: {baseline_avg:.1f}ms)",
                score=min(100.0, (current_avg_duration / baseline_avg) * 30),
                timestamp=datetime.now(),
                affected_period=(datetime.now() - window, datetime.now()),
                evidence=[
                    {
                        'current_avg_duration': current_avg_duration,
                        'current_p95_duration': current_p95_duration,
                        'baseline_avg_duration': baseline_avg,
                        'degradation_factor': current_avg_duration / baseline_avg,
                        'sample_count': len(durations)
                    }
                ]
            )
        
        # Update baseline
        self.update_baseline(service, 'avg_duration', current_avg_duration)
        self.update_baseline(service, 'p95_duration', current_p95_duration)
        return None
    
    def _detect_volume_anomaly(
        self, 
        service: str, 
        entries: List[LogEntry], 
        window: timedelta
    ) -> Optional[LogAnomaly]:
        """Detect unusual log volume."""
        current_volume = len(entries)
        
        # Get baseline volume
        baseline_volumes = [
            m['value'] for m in self.baseline_metrics[service]['log_volume']
            if datetime.now() - m['timestamp'] < timedelta(hours=24)
        ]
        
        if len(baseline_volumes) < 10:
            self.update_baseline(service, 'log_volume', current_volume)
            return None
        
        baseline_avg = statistics.mean(baseline_volumes)
        baseline_std = statistics.stdev(baseline_volumes) if len(baseline_volumes) > 1 else 0
        
        # Check for volume spike
        threshold = baseline_avg + (3 * baseline_std) if baseline_std > 0 else baseline_avg * 2
        
        if current_volume > max(threshold, baseline_avg * self.anomaly_thresholds['volume_spike']):
            return LogAnomaly(
                anomaly_id=f"volume_spike_{service}_{int(datetime.now().timestamp())}",
                anomaly_type=AlertCategory.BUSINESS_ANOMALY,
                service=service,
                description=f"Log volume spike detected: {current_volume} entries (baseline: {baseline_avg:.0f})",
                score=min(100.0, (current_volume / baseline_avg) * 10),
                timestamp=datetime.now(),
                affected_period=(datetime.now() - window, datetime.now()),
                evidence=[
                    {
                        'current_volume': current_volume,
                        'baseline_volume': baseline_avg,
                        'spike_factor': current_volume / baseline_avg if baseline_avg > 0 else float('inf')
                    }
                ]
            )
        
        # Update baseline
        self.update_baseline(service, 'log_volume', current_volume)
        return None
    
    def _detect_security_anomaly(
        self, 
        service: str, 
        entries: List[LogEntry], 
        window: timedelta
    ) -> Optional[LogAnomaly]:
        """Detect security-related anomalies."""
        security_indicators = [
            'authentication', 'auth', 'login', 'access denied',
            'unauthorized', 'forbidden', 'security', 'attack',
            'intrusion', 'malicious', 'suspicious'
        ]
        
        security_events = []
        ip_addresses = Counter()
        user_ids = Counter()
        
        for entry in entries:
            # Check for security-related keywords
            if any(indicator in entry.message.lower() for indicator in security_indicators):
                security_events.append(entry)
                
                # Extract IP addresses
                if 'client_ip' in entry.metadata:
                    ip_addresses[entry.metadata['client_ip']] += 1
                
                # Track user activity
                if entry.user_id:
                    user_ids[entry.user_id] += 1
        
        if not security_events:
            return None
        
        # Check for suspicious patterns
        suspicious_ips = [ip for ip, count in ip_addresses.items() if count > 50]  # > 50 security events
        suspicious_users = [user for user, count in user_ids.items() if count > 20]  # > 20 security events
        
        if suspicious_ips or suspicious_users:
            return LogAnomaly(
                anomaly_id=f"security_anomaly_{service}_{int(datetime.now().timestamp())}",
                anomaly_type=AlertCategory.SECURITY_INCIDENT,
                service=service,
                description=f"Security anomaly detected: {len(suspicious_ips)} suspicious IPs, {len(suspicious_users)} suspicious users",
                score=min(100.0, len(security_events) * 2),
                timestamp=datetime.now(),
                affected_period=(datetime.now() - window, datetime.now()),
                evidence=[
                    {
                        'security_events_count': len(security_events),
                        'suspicious_ips': suspicious_ips,
                        'suspicious_users': suspicious_users,
                        'top_ip_requests': dict(ip_addresses.most_common(5)),
                        'top_user_requests': dict(user_ids.most_common(5))
                    }
                ]
            )
        
        return None


class LogAggregator:
    """Aggregates logs from multiple sources."""
    
    def __init__(self):
        self.elasticsearch_client: Optional[AsyncElasticsearch] = None
        self.log_buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.processed_files: Set[str] = set()
        self._setup_elasticsearch()
    
    def _setup_elasticsearch(self):
        """Setup Elasticsearch client if configured."""
        es_config = monitoring_config.logging.log_aggregation_endpoint
        if es_config:
            try:
                self.elasticsearch_client = AsyncElasticsearch([es_config])
                logger.info("Elasticsearch client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Elasticsearch: {e}")
    
    async def aggregate_log_files(self, log_directory: str) -> List[LogEntry]:
        """Aggregate log entries from files."""
        log_entries = []
        
        try:
            async for file_path in aiofiles.os.listdir(log_directory):
                full_path = f"{log_directory}/{file_path}"
                
                if file_path.endswith('.log') or file_path.endswith('.log.gz'):
                    entries = await self._process_log_file(full_path)
                    log_entries.extend(entries)
        
        except Exception as e:
            logger.error(f"Error aggregating log files: {e}")
        
        return log_entries
    
    async def _process_log_file(self, file_path: str) -> List[LogEntry]:
        """Process individual log file."""
        entries = []
        
        try:
            # Check if file already processed
            file_stat = await aiofiles.os.stat(file_path)
            file_key = f"{file_path}:{file_stat.st_mtime}"
            
            if file_key in self.processed_files:
                return entries
            
            # Open file (handle gzip)
            if file_path.endswith('.gz'):
                async with aiofiles.open(file_path, 'rb') as f:
                    content = await f.read()
                    text_content = gzip.decompress(content).decode('utf-8')
                    lines = text_content.split('\n')
            else:
                async with aiofiles.open(file_path, 'r') as f:
                    content = await f.read()
                    lines = content.split('\n')
            
            # Process each line
            for line_num, line in enumerate(lines, 1):
                if line.strip():
                    try:
                        entry = LogEntry.from_json(line)
                        entry.source_file = file_path
                        entry.line_number = line_num
                        entries.append(entry)
                    except Exception as e:
                        logger.debug(f"Failed to parse log line {line_num} in {file_path}: {e}")
            
            # Mark file as processed
            self.processed_files.add(file_key)
            
            logger.debug(f"Processed {len(entries)} log entries from {file_path}")
        
        except Exception as e:
            logger.error(f"Error processing log file {file_path}: {e}")
        
        return entries
    
    async def send_to_elasticsearch(self, log_entries: List[LogEntry]):
        """Send log entries to Elasticsearch."""
        if not self.elasticsearch_client or not log_entries:
            return
        
        try:
            actions = []
            for entry in log_entries:
                action = {
                    "_index": f"investment-analysis-logs-{entry.timestamp.strftime('%Y.%m')}",
                    "_source": {
                        "timestamp": entry.timestamp.isoformat(),
                        "level": entry.level.value,
                        "service": entry.service,
                        "message": entry.message,
                        "correlation_id": entry.correlation_id,
                        "request_id": entry.request_id,
                        "user_id": entry.user_id,
                        "metadata": entry.metadata,
                        "source_file": entry.source_file,
                        "line_number": entry.line_number,
                        "exception_info": entry.exception_info
                    }
                }
                actions.append(action)
            
            # Bulk index
            from elasticsearch.helpers import async_bulk
            await async_bulk(self.elasticsearch_client, actions)
            
            logger.info(f"Sent {len(actions)} log entries to Elasticsearch")
        
        except Exception as e:
            logger.error(f"Error sending logs to Elasticsearch: {e}")


class LogAnalysisSystem:
    """
    Comprehensive log analysis system.
    """
    
    def __init__(self):
        self.pattern_detector = LogPatternDetector()
        self.anomaly_detector = LogAnomalyDetector()
        self.log_aggregator = LogAggregator()
        
        self.recent_logs: deque = deque(maxlen=50000)  # Keep recent logs in memory
        self.analysis_history: deque = deque(maxlen=1000)
        
        self._analysis_task: Optional[asyncio.Task] = None
        self._analysis_interval = 300  # 5 minutes
    
    async def start_analysis(self):
        """Start log analysis system."""
        if not self._analysis_task:
            self._analysis_task = asyncio.create_task(self._analysis_loop())
            logger.info("Started log analysis system")
    
    async def stop_analysis(self):
        """Stop log analysis system."""
        if self._analysis_task:
            self._analysis_task.cancel()
            try:
                await self._analysis_task
            except asyncio.CancelledError:
                pass
            logger.info("Stopped log analysis system")
    
    async def _analysis_loop(self):
        """Background log analysis loop."""
        while True:
            try:
                start_time = asyncio.get_event_loop().time()
                
                # Aggregate logs from files
                log_directory = monitoring_config.logging.log_file_path.rsplit('/', 1)[0] if monitoring_config.logging.log_file_path else '/app/logs'
                log_entries = await self.log_aggregator.aggregate_log_files(log_directory)
                
                if log_entries:
                    # Add to recent logs
                    self.recent_logs.extend(log_entries)
                    
                    # Process logs
                    await self._process_log_entries(log_entries)
                    
                    # Send to Elasticsearch if configured
                    await self.log_aggregator.send_to_elasticsearch(log_entries)
                
                # Record processing metrics
                processing_time = asyncio.get_event_loop().time() - start_time
                log_processing_latency.labels(
                    processor_type='full_analysis'
                ).observe(processing_time)
                
                await asyncio.sleep(self._analysis_interval)
            
            except Exception as e:
                logger.error(f"Error in log analysis loop: {e}")
                await asyncio.sleep(self._analysis_interval)
    
    async def _process_log_entries(self, log_entries: List[LogEntry]):
        """Process log entries for patterns and anomalies."""
        try:
            # Pattern detection
            for entry in log_entries:
                detected_patterns = self.pattern_detector.detect_patterns(entry)
                
                # Record metrics
                log_entries_processed.labels(
                    level=entry.level.value,
                    service=entry.service,
                    category='pattern_analysis'
                ).inc()
            
            # Anomaly detection
            anomalies = self.anomaly_detector.detect_anomalies(log_entries)
            
            if anomalies:
                await self._handle_anomalies(anomalies)
            
            # Calculate service error rates
            await self._calculate_error_rates(log_entries)
            
            # Store analysis results
            analysis_result = {
                'timestamp': datetime.now().isoformat(),
                'entries_processed': len(log_entries),
                'patterns_detected': len(self.pattern_detector.patterns),
                'anomalies_detected': len(anomalies)
            }
            self.analysis_history.append(analysis_result)
        
        except Exception as e:
            logger.error(f"Error processing log entries: {e}")
    
    async def _handle_anomalies(self, anomalies: List[LogAnomaly]):
        """Handle detected anomalies."""
        for anomaly in anomalies:
            # Record metrics
            log_anomalies_detected.labels(
                anomaly_type=anomaly.anomaly_type.value,
                service=anomaly.service
            ).inc()
            
            # Create alert if severity is high enough
            if anomaly.score >= 70:
                await self._create_alert_for_anomaly(anomaly)
            
            logger.warning(
                f"Log anomaly detected: {anomaly.description}",
                extra={
                    'anomaly_id': anomaly.anomaly_id,
                    'service': anomaly.service,
                    'score': anomaly.score,
                    'evidence': anomaly.evidence
                }
            )
    
    async def _create_alert_for_anomaly(self, anomaly: LogAnomaly):
        """Create alert for detected anomaly."""
        try:
            from backend.monitoring.alerting_system import alert_manager, AlertSeverity
            
            # Map anomaly score to alert severity
            if anomaly.score >= 90:
                severity = AlertSeverity.CRITICAL
            elif anomaly.score >= 70:
                severity = AlertSeverity.WARNING
            else:
                severity = AlertSeverity.INFO
            
            await alert_manager.create_alert(
                title=f"Log Anomaly: {anomaly.description}",
                description=f"Detected {anomaly.anomaly_type.value} in {anomaly.service} service",
                severity=severity,
                source="log_analysis",
                alert_type=anomaly.anomaly_type.value,
                metadata={
                    'anomaly_id': anomaly.anomaly_id,
                    'service': anomaly.service,
                    'score': anomaly.score,
                    'affected_period': {
                        'start': anomaly.affected_period[0].isoformat(),
                        'end': anomaly.affected_period[1].isoformat()
                    },
                    'evidence': anomaly.evidence
                }
            )
        
        except Exception as e:
            logger.error(f"Error creating alert for anomaly {anomaly.anomaly_id}: {e}")
    
    async def _calculate_error_rates(self, log_entries: List[LogEntry]):
        """Calculate error rates by service."""
        service_stats = defaultdict(lambda: {'total': 0, 'errors': 0})
        
        for entry in log_entries:
            service_stats[entry.service]['total'] += 1
            if entry.level in [LogLevel.ERROR, LogLevel.CRITICAL]:
                service_stats[entry.service]['errors'] += 1
        
        for service, stats in service_stats.items():
            if stats['total'] > 0:
                error_rate = (stats['errors'] / stats['total']) * 100
                error_rate_by_service.labels(
                    service=service,
                    time_window='5m'
                ).set(error_rate)
    
    # Public API methods
    
    async def search_logs(
        self,
        query: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        services: Optional[List[str]] = None,
        log_levels: Optional[List[LogLevel]] = None,
        limit: int = 100
    ) -> List[LogEntry]:
        """Search logs with filters."""
        try:
            # Filter recent logs in memory
            filtered_logs = []
            
            for entry in list(self.recent_logs):
                # Time filter
                if start_time and entry.timestamp < start_time:
                    continue
                if end_time and entry.timestamp > end_time:
                    continue
                
                # Service filter
                if services and entry.service not in services:
                    continue
                
                # Level filter
                if log_levels and entry.level not in log_levels:
                    continue
                
                # Text search
                if query.lower() in entry.message.lower():
                    filtered_logs.append(entry)
                
                if len(filtered_logs) >= limit:
                    break
            
            return filtered_logs[-limit:]  # Return most recent
        
        except Exception as e:
            logger.error(f"Error searching logs: {e}")
            return []
    
    def get_log_patterns(self) -> Dict[str, LogPattern]:
        """Get detected log patterns."""
        return dict(self.pattern_detector.patterns)
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get log analysis summary."""
        recent_analysis = list(self.analysis_history)[-10:] if self.analysis_history else []
        
        total_patterns = len(self.pattern_detector.patterns)
        active_patterns = sum(
            1 for pattern in self.pattern_detector.patterns.values()
            if pattern.last_seen and (datetime.now() - pattern.last_seen).seconds < 3600
        )
        
        return {
            'timestamp': datetime.now().isoformat(),
            'recent_logs_count': len(self.recent_logs),
            'total_patterns': total_patterns,
            'active_patterns': active_patterns,
            'recent_analysis': recent_analysis,
            'processing_status': 'running' if self._analysis_task else 'stopped'
        }


# Global log analysis system
log_analysis_system = LogAnalysisSystem()


# Setup function
async def setup_log_analysis():
    """Setup log analysis system."""
    await log_analysis_system.start_analysis()
    logger.info("Log analysis system setup completed")


# Convenience functions
async def search_logs(query: str, **filters):
    """Search logs with query and filters."""
    return await log_analysis_system.search_logs(query, **filters)


def get_log_patterns():
    """Get current log patterns."""
    return log_analysis_system.get_log_patterns()


def get_analysis_summary():
    """Get log analysis summary."""
    return log_analysis_system.get_analysis_summary()