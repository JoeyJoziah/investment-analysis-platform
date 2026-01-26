"""
Enhanced Logging and Debugging Capabilities
Comprehensive structured logging with correlation IDs, real-time monitoring, and intelligent analysis
"""

import asyncio
import json
import time
import uuid
import inspect
import traceback
import sys
import os
import re
from typing import Any, Dict, List, Optional, Callable, Union, TextIO
from dataclasses import dataclass, asdict, field
from enum import Enum
from datetime import datetime, timedelta
from collections import deque, defaultdict, Counter
import threading
import logging
import logging.handlers
from pathlib import Path
import aiofiles
from functools import wraps
import hashlib
import gzip
import json_logging

# Elasticsearch is optional - removed to save $15-20/month
# Using file-based logging and PostgreSQL full-text search instead
try:
    import elasticsearch
    from elasticsearch.helpers import bulk
    ELASTICSEARCH_AVAILABLE = True
except ImportError:
    ELASTICSEARCH_AVAILABLE = False
    elasticsearch = None
    bulk = None

from .exceptions import *

# Configure JSON logging
json_logging.init_non_web()

logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Enhanced log levels"""
    TRACE = 5
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50
    AUDIT = 60


class LogContext(Enum):
    """Log context categories"""
    REQUEST = "request"
    BUSINESS = "business"
    SECURITY = "security"
    PERFORMANCE = "performance"
    ERROR = "error"
    AUDIT = "audit"
    SYSTEM = "system"
    INTEGRATION = "integration"


@dataclass
class CorrelationContext:
    """Request/operation correlation context"""
    correlation_id: str
    parent_id: Optional[str]
    operation_name: str
    user_id: Optional[str]
    session_id: Optional[str]
    request_id: Optional[str]
    trace_depth: int
    start_time: datetime
    context_data: Dict[str, Any] = field(default_factory=dict)
    
    def create_child(self, operation_name: str) -> 'CorrelationContext':
        """Create child correlation context"""
        return CorrelationContext(
            correlation_id=self.correlation_id,
            parent_id=f"{self.correlation_id}:{self.trace_depth}",
            operation_name=operation_name,
            user_id=self.user_id,
            session_id=self.session_id,
            request_id=self.request_id,
            trace_depth=self.trace_depth + 1,
            start_time=datetime.now(),
            context_data=self.context_data.copy()
        )


@dataclass
class LogEvent:
    """Structured log event"""
    timestamp: datetime
    level: LogLevel
    context: LogContext
    correlation_id: str
    message: str
    logger_name: str
    module: str
    function: str
    line_number: int
    thread_id: str
    process_id: int
    user_id: Optional[str]
    session_id: Optional[str]
    request_id: Optional[str]
    duration_ms: Optional[float]
    error_type: Optional[str]
    error_message: Optional[str]
    stack_trace: Optional[str]
    custom_fields: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['level'] = self.level.name
        data['context'] = self.context.value
        return data


class CorrelationManager:
    """Manages correlation contexts across request/operation boundaries"""
    
    def __init__(self):
        self._contexts: Dict[str, CorrelationContext] = {}
        self._thread_contexts: Dict[str, str] = {}  # thread_id -> correlation_id
        self._lock = threading.RLock()
    
    def create_context(
        self,
        operation_name: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
        parent_correlation_id: Optional[str] = None,
        context_data: Dict[str, Any] = None
    ) -> CorrelationContext:
        """Create new correlation context"""
        
        correlation_id = str(uuid.uuid4())
        
        context = CorrelationContext(
            correlation_id=correlation_id,
            parent_id=parent_correlation_id,
            operation_name=operation_name,
            user_id=user_id,
            session_id=session_id,
            request_id=request_id,
            trace_depth=0,
            start_time=datetime.now(),
            context_data=context_data or {}
        )
        
        with self._lock:
            self._contexts[correlation_id] = context
            self._thread_contexts[str(threading.get_ident())] = correlation_id
        
        return context
    
    def get_current_context(self) -> Optional[CorrelationContext]:
        """Get correlation context for current thread"""
        thread_id = str(threading.get_ident())
        
        with self._lock:
            correlation_id = self._thread_contexts.get(thread_id)
            if correlation_id:
                return self._contexts.get(correlation_id)
        
        return None
    
    def set_context(self, context: CorrelationContext):
        """Set correlation context for current thread"""
        thread_id = str(threading.get_ident())
        
        with self._lock:
            self._contexts[context.correlation_id] = context
            self._thread_contexts[thread_id] = context.correlation_id
    
    def clear_context(self):
        """Clear correlation context for current thread"""
        thread_id = str(threading.get_ident())
        
        with self._lock:
            if thread_id in self._thread_contexts:
                correlation_id = self._thread_contexts[thread_id]
                del self._thread_contexts[thread_id]
                
                # Clean up context if no threads reference it
                if correlation_id not in self._thread_contexts.values():
                    self._contexts.pop(correlation_id, None)
    
    def add_context_data(self, key: str, value: Any):
        """Add data to current correlation context"""
        context = self.get_current_context()
        if context:
            context.context_data[key] = value
    
    def get_context_data(self, key: str) -> Any:
        """Get data from current correlation context"""
        context = self.get_current_context()
        if context:
            return context.context_data.get(key)
        return None


class LogPattern:
    """Defines patterns for log analysis and alerting"""
    
    def __init__(
        self,
        pattern_id: str,
        name: str,
        description: str,
        pattern_regex: str,
        severity: LogLevel,
        time_window_minutes: int = 5,
        occurrence_threshold: int = 1,
        alert_enabled: bool = True
    ):
        self.pattern_id = pattern_id
        self.name = name
        self.description = description
        self.pattern_regex = re.compile(pattern_regex)
        self.severity = severity
        self.time_window_minutes = time_window_minutes
        self.occurrence_threshold = occurrence_threshold
        self.alert_enabled = alert_enabled
        
        # Tracking
        self.matches: deque = deque(maxlen=10000)
        self.last_alert: Optional[datetime] = None
        self.alert_cooldown_minutes = 30


class LogAnalyzer:
    """Analyzes log patterns and generates alerts"""
    
    def __init__(self):
        self.patterns: Dict[str, LogPattern] = {}
        self.error_signatures: Dict[str, Dict] = {}
        self.performance_metrics: deque = deque(maxlen=10000)
        self.alert_callbacks: List[Callable] = []
        
        # Load default patterns
        self._load_default_patterns()
    
    def _load_default_patterns(self):
        """Load default log analysis patterns"""
        
        # Critical error patterns
        self.register_pattern(LogPattern(
            pattern_id="auth_failure_burst",
            name="Authentication Failure Burst",
            description="Multiple authentication failures in short time",
            pattern_regex=r"authentication.*fail",
            severity=LogLevel.CRITICAL,
            time_window_minutes=5,
            occurrence_threshold=10
        ))
        
        self.register_pattern(LogPattern(
            pattern_id="database_error_spike",
            name="Database Error Spike",
            description="High frequency of database errors",
            pattern_regex=r"database.*error|connection.*lost|query.*timeout",
            severity=LogLevel.ERROR,
            time_window_minutes=3,
            occurrence_threshold=5
        ))
        
        self.register_pattern(LogPattern(
            pattern_id="api_rate_limit",
            name="API Rate Limit Exceeded",
            description="External API rate limits being exceeded",
            pattern_regex=r"rate.limit|too.many.requests|quota.exceeded",
            severity=LogLevel.WARNING,
            time_window_minutes=5,
            occurrence_threshold=3
        ))
        
        self.register_pattern(LogPattern(
            pattern_id="memory_pressure",
            name="Memory Pressure",
            description="System experiencing memory pressure",
            pattern_regex=r"out.of.memory|memory.*exhausted|gc.*pressure",
            severity=LogLevel.CRITICAL,
            time_window_minutes=2,
            occurrence_threshold=1
        ))
        
        self.register_pattern(LogPattern(
            pattern_id="slow_response",
            name="Slow Response Times",
            description="Response times exceeding thresholds",
            pattern_regex=r"slow.*response|timeout|response.*time.*\d{4,}ms",
            severity=LogLevel.WARNING,
            time_window_minutes=10,
            occurrence_threshold=20
        ))
    
    def register_pattern(self, pattern: LogPattern):
        """Register log pattern for analysis"""
        self.patterns[pattern.pattern_id] = pattern
        logger.info(f"Registered log pattern: {pattern.pattern_id}")
    
    def analyze_log_event(self, log_event: LogEvent):
        """Analyze log event against patterns"""
        message = log_event.message.lower()
        
        # Check all patterns
        for pattern in self.patterns.values():
            if pattern.pattern_regex.search(message):
                self._record_pattern_match(pattern, log_event)
        
        # Analyze error signatures
        if log_event.level in [LogLevel.ERROR, LogLevel.CRITICAL]:
            self._analyze_error_signature(log_event)
        
        # Track performance metrics
        if log_event.duration_ms is not None:
            self._record_performance_metric(log_event)
    
    def _record_pattern_match(self, pattern: LogPattern, log_event: LogEvent):
        """Record pattern match and check thresholds"""
        pattern.matches.append({
            'timestamp': log_event.timestamp,
            'correlation_id': log_event.correlation_id,
            'message': log_event.message,
            'level': log_event.level
        })
        
        # Check if threshold exceeded within time window
        cutoff_time = datetime.now() - timedelta(minutes=pattern.time_window_minutes)
        recent_matches = [
            m for m in pattern.matches
            if m['timestamp'] > cutoff_time
        ]
        
        if len(recent_matches) >= pattern.occurrence_threshold:
            self._trigger_pattern_alert(pattern, recent_matches)
    
    def _trigger_pattern_alert(self, pattern: LogPattern, matches: List[Dict]):
        """Trigger alert for pattern match"""
        # Check cooldown
        if (pattern.last_alert and
            datetime.now() - pattern.last_alert < timedelta(minutes=pattern.alert_cooldown_minutes)):
            return
        
        if not pattern.alert_enabled:
            return
        
        alert_data = {
            'alert_id': str(uuid.uuid4()),
            'pattern_id': pattern.pattern_id,
            'pattern_name': pattern.name,
            'severity': pattern.severity.name,
            'description': pattern.description,
            'match_count': len(matches),
            'time_window_minutes': pattern.time_window_minutes,
            'first_occurrence': min(m['timestamp'] for m in matches).isoformat(),
            'last_occurrence': max(m['timestamp'] for m in matches).isoformat(),
            'sample_messages': [m['message'] for m in matches[:5]],
            'correlation_ids': list(set(m['correlation_id'] for m in matches)),
            'timestamp': datetime.now().isoformat()
        }
        
        # Send to alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert_data)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
        
        pattern.last_alert = datetime.now()
        
        logger.critical(f"LOG PATTERN ALERT: {pattern.name} - {len(matches)} occurrences")
    
    def _analyze_error_signature(self, log_event: LogEvent):
        """Analyze error for signature patterns"""
        if not log_event.error_message:
            return
        
        # Create error signature
        signature_parts = [
            log_event.error_type or "unknown",
            log_event.module,
            log_event.function
        ]
        
        # Normalize error message (remove variable parts)
        normalized_message = self._normalize_error_message(log_event.error_message)
        signature_parts.append(normalized_message)
        
        signature_hash = hashlib.md5("|".join(signature_parts).encode()).hexdigest()
        
        # Track signature
        if signature_hash not in self.error_signatures:
            self.error_signatures[signature_hash] = {
                'first_seen': log_event.timestamp,
                'last_seen': log_event.timestamp,
                'count': 1,
                'error_type': log_event.error_type,
                'module': log_event.module,
                'function': log_event.function,
                'normalized_message': normalized_message,
                'recent_correlations': [log_event.correlation_id]
            }
        else:
            signature = self.error_signatures[signature_hash]
            signature['last_seen'] = log_event.timestamp
            signature['count'] += 1
            signature['recent_correlations'].append(log_event.correlation_id)
            
            # Keep only recent correlations
            if len(signature['recent_correlations']) > 100:
                signature['recent_correlations'] = signature['recent_correlations'][-100:]
    
    def _normalize_error_message(self, message: str) -> str:
        """Normalize error message by removing variable parts"""
        # Remove timestamps
        message = re.sub(r'\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}.*?(?=\s|$)', '<TIMESTAMP>', message)
        
        # Remove IDs and UUIDs
        message = re.sub(r'\b[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}\b', '<UUID>', message)
        message = re.sub(r'\b\d{6,}\b', '<ID>', message)
        
        # Remove file paths
        message = re.sub(r'/[^\s]+', '<PATH>', message)
        
        # Remove numbers
        message = re.sub(r'\b\d+\.\d+\b', '<NUMBER>', message)
        message = re.sub(r'\b\d+\b', '<NUMBER>', message)
        
        return message.strip()
    
    def _record_performance_metric(self, log_event: LogEvent):
        """Record performance metric"""
        self.performance_metrics.append({
            'timestamp': log_event.timestamp,
            'operation': f"{log_event.module}.{log_event.function}",
            'duration_ms': log_event.duration_ms,
            'correlation_id': log_event.correlation_id
        })
    
    def register_alert_callback(self, callback: Callable):
        """Register callback for alerts"""
        self.alert_callbacks.append(callback)
    
    def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get error summary for specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_errors = {
            sig_hash: sig for sig_hash, sig in self.error_signatures.items()
            if sig['last_seen'] > cutoff_time
        }
        
        if not recent_errors:
            return {'message': f'No errors in the last {hours} hours'}
        
        # Top errors by frequency
        top_errors = sorted(
            recent_errors.items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )[:10]
        
        # New errors (first seen in time window)
        new_errors = [
            (sig_hash, sig) for sig_hash, sig in recent_errors.items()
            if sig['first_seen'] > cutoff_time
        ]
        
        return {
            'time_period_hours': hours,
            'total_error_signatures': len(recent_errors),
            'new_error_signatures': len(new_errors),
            'top_errors': [
                {
                    'signature_hash': sig_hash,
                    'count': sig['count'],
                    'error_type': sig['error_type'],
                    'module': sig['module'],
                    'function': sig['function'],
                    'normalized_message': sig['normalized_message'],
                    'first_seen': sig['first_seen'].isoformat(),
                    'last_seen': sig['last_seen'].isoformat()
                }
                for sig_hash, sig in top_errors
            ],
            'new_errors': [
                {
                    'signature_hash': sig_hash,
                    'error_type': sig['error_type'],
                    'module': sig['module'],
                    'function': sig['function'],
                    'normalized_message': sig['normalized_message'],
                    'first_seen': sig['first_seen'].isoformat()
                }
                for sig_hash, sig in new_errors[:5]
            ]
        }


class ElasticsearchHandler(logging.Handler):
    """Custom log handler for Elasticsearch (optional - disabled by default to save costs)"""

    def __init__(self, elasticsearch_hosts: List[str], index_prefix: str = "logs"):
        super().__init__()
        if not ELASTICSEARCH_AVAILABLE:
            raise ImportError("Elasticsearch library not available. Install with: pip install elasticsearch")
        self.es_client = elasticsearch.Elasticsearch(elasticsearch_hosts)
        self.index_prefix = index_prefix
        self.buffer: deque = deque(maxlen=1000)
        self.buffer_lock = threading.Lock()

        # Start background flush task
        self.flush_interval = 5  # seconds
        self.flush_task = threading.Thread(target=self._flush_loop, daemon=True)
        self.flush_task.start()
    
    def emit(self, record: logging.LogRecord):
        """Emit log record to buffer"""
        try:
            log_entry = self._format_record(record)
            
            with self.buffer_lock:
                self.buffer.append(log_entry)
            
        except Exception as e:
            self.handleError(record)
    
    def _format_record(self, record: logging.LogRecord) -> Dict[str, Any]:
        """Format log record for Elasticsearch"""
        # Get correlation context
        correlation_manager = get_correlation_manager()
        context = correlation_manager.get_current_context()
        
        log_entry = {
            '@timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread': record.thread,
            'process': record.process,
        }
        
        # Add correlation data if available
        if context:
            log_entry.update({
                'correlation_id': context.correlation_id,
                'operation': context.operation_name,
                'user_id': context.user_id,
                'session_id': context.session_id,
                'request_id': context.request_id,
                'trace_depth': context.trace_depth
            })
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': self.formatter.formatException(record.exc_info) if self.formatter else None
            }
        
        # Add custom fields from record
        for key, value in record.__dict__.items():
            if key.startswith('custom_') and key not in log_entry:
                log_entry[key] = value
        
        return log_entry
    
    def _flush_loop(self):
        """Background task to flush logs to Elasticsearch"""
        while True:
            try:
                time.sleep(self.flush_interval)
                self._flush_buffer()
            except Exception as e:
                print(f"Elasticsearch flush error: {e}", file=sys.stderr)
    
    def _flush_buffer(self):
        """Flush buffered logs to Elasticsearch"""
        if not self.buffer:
            return
        
        with self.buffer_lock:
            logs_to_flush = list(self.buffer)
            self.buffer.clear()
        
        if not logs_to_flush:
            return
        
        # Prepare bulk index operations
        today = datetime.now().strftime('%Y.%m.%d')
        index_name = f"{self.index_prefix}-{today}"
        
        actions = []
        for log_entry in logs_to_flush:
            actions.append({
                '_index': index_name,
                '_source': log_entry
            })
        
        try:
            # Bulk index to Elasticsearch
            bulk(self.es_client, actions)
        except Exception as e:
            print(f"Elasticsearch bulk index error: {e}", file=sys.stderr)
            
            # Put logs back in buffer on failure
            with self.buffer_lock:
                self.buffer.extendleft(reversed(logs_to_flush))


class StructuredLogger:
    """Enhanced structured logger with correlation support"""
    
    def __init__(self, name: str, correlation_manager: CorrelationManager, log_analyzer: LogAnalyzer):
        self.name = name
        self.correlation_manager = correlation_manager
        self.log_analyzer = log_analyzer
        self._logger = logging.getLogger(name)
        
        # Performance tracking
        self._operation_start_times: Dict[str, float] = {}
    
    def _create_log_event(
        self,
        level: LogLevel,
        message: str,
        context: LogContext = LogContext.SYSTEM,
        duration_ms: Optional[float] = None,
        error: Optional[Exception] = None,
        **kwargs
    ) -> LogEvent:
        """Create structured log event"""
        
        # Get caller information
        frame = inspect.currentframe()
        caller_frame = frame.f_back.f_back  # Skip this method and the public method
        
        module = caller_frame.f_globals.get('__name__', 'unknown')
        function = caller_frame.f_code.co_name
        line_number = caller_frame.f_lineno
        
        # Get correlation context
        correlation_context = self.correlation_manager.get_current_context()
        
        # Extract error information
        error_type = None
        error_message = None
        stack_trace = None
        
        if error:
            error_type = type(error).__name__
            error_message = str(error)
            stack_trace = ''.join(traceback.format_exception(type(error), error, error.__traceback__))
        
        log_event = LogEvent(
            timestamp=datetime.now(),
            level=level,
            context=context,
            correlation_id=correlation_context.correlation_id if correlation_context else 'no-correlation',
            message=message,
            logger_name=self.name,
            module=module,
            function=function,
            line_number=line_number,
            thread_id=str(threading.get_ident()),
            process_id=os.getpid(),
            user_id=correlation_context.user_id if correlation_context else None,
            session_id=correlation_context.session_id if correlation_context else None,
            request_id=correlation_context.request_id if correlation_context else None,
            duration_ms=duration_ms,
            error_type=error_type,
            error_message=error_message,
            stack_trace=stack_trace,
            custom_fields=kwargs.get('custom_fields', {}),
            tags=kwargs.get('tags', [])
        )
        
        return log_event
    
    def _emit_log_event(self, log_event: LogEvent):
        """Emit log event to logging system and analyzer"""
        
        # Create logging record
        record = logging.LogRecord(
            name=self.name,
            level=log_event.level.value,
            pathname="",
            lineno=log_event.line_number,
            msg=log_event.message,
            args=(),
            exc_info=None
        )
        
        # Add custom fields to record
        for key, value in log_event.custom_fields.items():
            setattr(record, f'custom_{key}', value)
        
        # Add correlation fields
        record.custom_correlation_id = log_event.correlation_id
        record.custom_user_id = log_event.user_id
        record.custom_duration_ms = log_event.duration_ms
        
        # Emit to logging system
        self._logger.handle(record)
        
        # Send to analyzer
        self.log_analyzer.analyze_log_event(log_event)
    
    def trace(self, message: str, **kwargs):
        """Log trace message"""
        log_event = self._create_log_event(LogLevel.TRACE, message, **kwargs)
        self._emit_log_event(log_event)
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        log_event = self._create_log_event(LogLevel.DEBUG, message, **kwargs)
        self._emit_log_event(log_event)
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        log_event = self._create_log_event(LogLevel.INFO, message, **kwargs)
        self._emit_log_event(log_event)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        log_event = self._create_log_event(LogLevel.WARNING, message, **kwargs)
        self._emit_log_event(log_event)
    
    def error(self, message: str, error: Optional[Exception] = None, **kwargs):
        """Log error message"""
        log_event = self._create_log_event(
            LogLevel.ERROR, message, context=LogContext.ERROR, error=error, **kwargs
        )
        self._emit_log_event(log_event)
    
    def critical(self, message: str, error: Optional[Exception] = None, **kwargs):
        """Log critical message"""
        log_event = self._create_log_event(
            LogLevel.CRITICAL, message, context=LogContext.ERROR, error=error, **kwargs
        )
        self._emit_log_event(log_event)
    
    def audit(self, message: str, **kwargs):
        """Log audit message"""
        log_event = self._create_log_event(
            LogLevel.AUDIT, message, context=LogContext.AUDIT, **kwargs
        )
        self._emit_log_event(log_event)
    
    def business(self, message: str, **kwargs):
        """Log business event"""
        log_event = self._create_log_event(
            LogLevel.INFO, message, context=LogContext.BUSINESS, **kwargs
        )
        self._emit_log_event(log_event)
    
    def security(self, message: str, **kwargs):
        """Log security event"""
        log_event = self._create_log_event(
            LogLevel.WARNING, message, context=LogContext.SECURITY, **kwargs
        )
        self._emit_log_event(log_event)
    
    def performance(self, message: str, duration_ms: float, **kwargs):
        """Log performance event"""
        log_event = self._create_log_event(
            LogLevel.INFO, message, context=LogContext.PERFORMANCE, 
            duration_ms=duration_ms, **kwargs
        )
        self._emit_log_event(log_event)
    
    def start_operation(self, operation_name: str) -> str:
        """Start timing an operation"""
        operation_id = str(uuid.uuid4())
        self._operation_start_times[operation_id] = time.time()
        
        self.info(f"Started operation: {operation_name}", 
                  custom_fields={'operation_id': operation_id, 'operation_name': operation_name},
                  tags=['operation_start'])
        
        return operation_id
    
    def end_operation(self, operation_id: str, operation_name: str, success: bool = True, **kwargs):
        """End timing an operation"""
        start_time = self._operation_start_times.pop(operation_id, None)
        
        if start_time:
            duration_ms = (time.time() - start_time) * 1000
            
            status = "completed" if success else "failed"
            message = f"Operation {status}: {operation_name} ({duration_ms:.1f}ms)"
            
            if success:
                self.performance(message, duration_ms, 
                               custom_fields={'operation_id': operation_id, 'operation_name': operation_name},
                               tags=['operation_end', 'success'], **kwargs)
            else:
                self.error(message, 
                          custom_fields={'operation_id': operation_id, 'operation_name': operation_name},
                          tags=['operation_end', 'failure'], **kwargs)
        else:
            self.warning(f"End operation called without start: {operation_name}")


class LoggingSystem:
    """
    Complete enhanced logging system with structured logging,
    correlation tracking, pattern analysis, and real-time monitoring
    """
    
    def __init__(
        self,
        log_level: LogLevel = LogLevel.INFO,
        elasticsearch_hosts: Optional[List[str]] = None,
        log_directory: str = "logs",
        enable_file_rotation: bool = True,
        enable_console_output: bool = True
    ):
        self.log_level = log_level
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(parents=True, exist_ok=True)
        
        # Core components
        self.correlation_manager = CorrelationManager()
        self.log_analyzer = LogAnalyzer()
        
        # Loggers registry
        self._loggers: Dict[str, StructuredLogger] = {}
        
        # Configure root logging
        self._configure_root_logging(
            elasticsearch_hosts, enable_file_rotation, enable_console_output
        )
        
        # Performance monitoring
        self._request_metrics: deque = deque(maxlen=100000)
        self._operation_metrics: deque = deque(maxlen=100000)
        
        # Real-time monitoring
        self._monitoring_enabled = False
        self._monitoring_task: Optional[asyncio.Task] = None
        
        # Register default alert callbacks
        self.log_analyzer.register_alert_callback(self._default_alert_handler)
    
    def _configure_root_logging(
        self,
        elasticsearch_hosts: Optional[List[str]],
        enable_file_rotation: bool,
        enable_console_output: bool
    ):
        """Configure root logging with handlers"""
        
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level.value)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # JSON formatter
        formatter = logging.Formatter(
            '{"timestamp":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s",'
            '"message":"%(message)s","module":"%(module)s","function":"%(funcName)s",'
            '"line":%(lineno)d,"thread":"%(thread)d","process":"%(process)d"'
            '%(custom_fields)s}'
        )
        
        class JSONFormatter(logging.Formatter):
            def format(self, record):
                # Add custom fields
                custom_fields = []
                for key, value in record.__dict__.items():
                    if key.startswith('custom_') and value is not None:
                        field_name = key[7:]  # Remove 'custom_' prefix
                        custom_fields.append(f'"{field_name}":"{value}"')
                
                record.custom_fields = ',' + ','.join(custom_fields) if custom_fields else ''
                
                return super().format(record)
        
        json_formatter = JSONFormatter(
            '{"timestamp":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s",'
            '"message":"%(message)s","module":"%(module)s","function":"%(funcName)s",'
            '"line":%(lineno)d,"thread":"%(thread)d","process":"%(process)d"'
            '%(custom_fields)s}'
        )
        
        # Console handler
        if enable_console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.log_level.value)
            console_handler.setFormatter(json_formatter)
            root_logger.addHandler(console_handler)
        
        # File handler with rotation
        if enable_file_rotation:
            file_handler = logging.handlers.RotatingFileHandler(
                self.log_directory / "application.log",
                maxBytes=100 * 1024 * 1024,  # 100MB
                backupCount=10
            )
            file_handler.setLevel(self.log_level.value)
            file_handler.setFormatter(json_formatter)
            root_logger.addHandler(file_handler)
        
        # Elasticsearch handler (optional - disabled by default to save $15-20/month)
        if elasticsearch_hosts and ELASTICSEARCH_AVAILABLE:
            try:
                es_handler = ElasticsearchHandler(elasticsearch_hosts, "investment-app-logs")
                es_handler.setLevel(LogLevel.INFO.value)
                root_logger.addHandler(es_handler)
            except Exception as e:
                print(f"Elasticsearch logging disabled (optional): {e}", file=sys.stderr)
    
    def get_logger(self, name: str) -> StructuredLogger:
        """Get or create structured logger"""
        if name not in self._loggers:
            self._loggers[name] = StructuredLogger(
                name, self.correlation_manager, self.log_analyzer
            )
        
        return self._loggers[name]
    
    def create_correlation_context(
        self,
        operation_name: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
        context_data: Dict[str, Any] = None
    ) -> CorrelationContext:
        """Create correlation context for request/operation tracking"""
        return self.correlation_manager.create_context(
            operation_name, user_id, session_id, request_id, context_data=context_data
        )
    
    def get_current_context(self) -> Optional[CorrelationContext]:
        """Get current correlation context"""
        return self.correlation_manager.get_current_context()
    
    def _default_alert_handler(self, alert_data: Dict[str, Any]):
        """Default alert handler - logs critical alerts"""
        alert_logger = self.get_logger("alerts")
        alert_logger.critical(
            f"LOG ALERT: {alert_data['pattern_name']} - {alert_data['match_count']} occurrences",
            custom_fields=alert_data,
            tags=['alert', 'pattern_match']
        )
    
    def register_alert_callback(self, callback: Callable):
        """Register custom alert callback"""
        self.log_analyzer.register_alert_callback(callback)
    
    def start_real_time_monitoring(self):
        """Start real-time log monitoring"""
        if self._monitoring_enabled:
            return
        
        self._monitoring_enabled = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Real-time log monitoring started")
    
    async def stop_real_time_monitoring(self):
        """Stop real-time log monitoring"""
        if not self._monitoring_enabled:
            return
        
        self._monitoring_enabled = False
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Real-time log monitoring stopped")
    
    async def _monitoring_loop(self):
        """Real-time monitoring loop"""
        while self._monitoring_enabled:
            try:
                # Generate monitoring reports
                await self._generate_monitoring_report()
                
                await asyncio.sleep(60)  # Monitor every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(60)
    
    async def _generate_monitoring_report(self):
        """Generate real-time monitoring report"""
        # This could generate periodic reports on log patterns, error rates, etc.
        pass
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get logging system metrics"""
        return {
            'active_loggers': len(self._loggers),
            'active_correlations': len(self.correlation_manager._contexts),
            'registered_patterns': len(self.log_analyzer.patterns),
            'error_signatures': len(self.log_analyzer.error_signatures),
            'monitoring_enabled': self._monitoring_enabled,
            'log_level': self.log_level.name,
            'log_directory': str(self.log_directory)
        }
    
    def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get error analysis summary"""
        return self.log_analyzer.get_error_summary(hours)


# Global logging system instance
_logging_system: Optional[LoggingSystem] = None
_correlation_manager: Optional[CorrelationManager] = None


def initialize_logging_system(
    log_level: LogLevel = LogLevel.INFO,
    elasticsearch_hosts: Optional[List[str]] = None,
    log_directory: str = "logs",
    enable_file_rotation: bool = True,
    enable_console_output: bool = True
) -> LoggingSystem:
    """Initialize the enhanced logging system"""
    global _logging_system, _correlation_manager
    
    _logging_system = LoggingSystem(
        log_level=log_level,
        elasticsearch_hosts=elasticsearch_hosts,
        log_directory=log_directory,
        enable_file_rotation=enable_file_rotation,
        enable_console_output=enable_console_output
    )
    
    _correlation_manager = _logging_system.correlation_manager
    
    return _logging_system


def get_logger(name: str) -> StructuredLogger:
    """Get structured logger instance"""
    if not _logging_system:
        raise RuntimeError("Logging system not initialized. Call initialize_logging_system() first.")
    
    return _logging_system.get_logger(name)


def get_correlation_manager() -> CorrelationManager:
    """Get correlation manager instance"""
    if not _correlation_manager:
        raise RuntimeError("Logging system not initialized. Call initialize_logging_system() first.")
    
    return _correlation_manager


def get_correlation_logger(correlation_id: str) -> StructuredLogger:
    """Get logger with specific correlation context"""
    logger = get_logger("correlation")
    
    # This is a simplified version - in practice you'd want to set the context properly
    return logger


# Decorators for automatic logging
def log_function_calls(
    log_args: bool = True,
    log_result: bool = False,
    log_duration: bool = True,
    logger_name: Optional[str] = None
):
    """Decorator to automatically log function calls"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            func_logger = get_logger(logger_name or func.__module__)
            operation_id = func_logger.start_operation(func.__name__)
            
            try:
                if log_args:
                    func_logger.debug(
                        f"Calling {func.__name__}",
                        custom_fields={
                            'function_args': str(args) if args else None,
                            'function_kwargs': str(kwargs) if kwargs else None
                        },
                        tags=['function_call']
                    )
                
                result = await func(*args, **kwargs)
                
                if log_result:
                    func_logger.debug(
                        f"Function {func.__name__} returned",
                        custom_fields={'result': str(result)[:1000]},  # Truncate long results
                        tags=['function_return']
                    )
                
                func_logger.end_operation(operation_id, func.__name__, success=True)
                return result
                
            except Exception as e:
                func_logger.end_operation(operation_id, func.__name__, success=False, error=e)
                func_logger.error(f"Function {func.__name__} failed", error=e)
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            func_logger = get_logger(logger_name or func.__module__)
            operation_id = func_logger.start_operation(func.__name__)
            
            try:
                if log_args:
                    func_logger.debug(
                        f"Calling {func.__name__}",
                        custom_fields={
                            'function_args': str(args) if args else None,
                            'function_kwargs': str(kwargs) if kwargs else None
                        },
                        tags=['function_call']
                    )
                
                result = func(*args, **kwargs)
                
                if log_result:
                    func_logger.debug(
                        f"Function {func.__name__} returned",
                        custom_fields={'result': str(result)[:1000]},
                        tags=['function_return']
                    )
                
                func_logger.end_operation(operation_id, func.__name__, success=True)
                return result
                
            except Exception as e:
                func_logger.end_operation(operation_id, func.__name__, success=False, error=e)
                func_logger.error(f"Function {func.__name__} failed", error=e)
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Context manager for correlation tracking
class correlation_context:
    """Context manager for correlation tracking"""
    
    def __init__(
        self,
        operation_name: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
        context_data: Dict[str, Any] = None
    ):
        self.operation_name = operation_name
        self.user_id = user_id
        self.session_id = session_id
        self.request_id = request_id
        self.context_data = context_data
        self.context: Optional[CorrelationContext] = None
    
    def __enter__(self):
        correlation_manager = get_correlation_manager()
        self.context = correlation_manager.create_context(
            self.operation_name,
            self.user_id,
            self.session_id,
            self.request_id,
            self.context_data
        )
        return self.context
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        correlation_manager = get_correlation_manager()
        correlation_manager.clear_context()


# Example usage functions
def example_alert_handler(alert_data: Dict[str, Any]):
    """Example custom alert handler"""
    print(f"CUSTOM ALERT: {alert_data['pattern_name']} - {alert_data['match_count']} matches")
    
    # Could send to Slack, PagerDuty, email, etc.
    # integrate_with_slack(alert_data)
    # integrate_with_pagerduty(alert_data)


def setup_application_logging():
    """Setup logging for the investment analysis application"""

    # Initialize logging system
    # Note: Elasticsearch is disabled by default to save $15-20/month
    # Using file-based logging instead - can be re-enabled if needed
    logging_system = initialize_logging_system(
        log_level=LogLevel.DEBUG,
        elasticsearch_hosts=None,  # Elasticsearch disabled to save costs
        log_directory='logs',
        enable_file_rotation=True,
        enable_console_output=True
    )
    
    # Register custom alert handler
    logging_system.register_alert_callback(example_alert_handler)
    
    # Start real-time monitoring
    asyncio.create_task(logging_system.start_real_time_monitoring())
    
    return logging_system