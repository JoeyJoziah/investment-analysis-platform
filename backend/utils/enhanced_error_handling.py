"""
Enhanced Error Handling and Classification System
Comprehensive error categorization, correlation, and intelligent response strategies
"""

import asyncio
import time
import uuid
import json
import traceback
from typing import Any, Dict, List, Optional, Type, Callable, Union
from enum import Enum, IntEnum
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import threading
import logging
import hashlib
import re
from pathlib import Path
from functools import wraps

from .exceptions import *
from .structured_logging import StructuredLogger, get_correlation_id

logger = logging.getLogger(__name__)


class ErrorSeverity(IntEnum):
    """Error severity levels for prioritized handling"""
    CRITICAL = 5    # System-threatening errors requiring immediate action
    HIGH = 4        # Service-impacting errors requiring prompt attention
    MEDIUM = 3      # Degraded performance but functional
    LOW = 2         # Minor issues with workarounds available
    INFO = 1        # Informational, no action required


class ErrorCategory(Enum):
    """Primary error categories for classification"""
    TRANSIENT = "transient"           # Temporary issues that may resolve
    PERMANENT = "permanent"           # Persistent issues requiring intervention
    RATE_LIMIT = "rate_limit"        # API rate limiting issues
    NETWORK = "network"              # Network connectivity issues
    AUTHENTICATION = "authentication" # Auth/authorization failures
    DATA_QUALITY = "data_quality"    # Data validation/quality issues
    CONFIGURATION = "configuration"   # Config or setup issues
    DEPENDENCY = "dependency"        # External service failures
    RESOURCE = "resource"            # Resource exhaustion (memory, disk, etc.)
    BUSINESS_LOGIC = "business_logic" # Application logic errors


class ErrorPattern(Enum):
    """Common error patterns for intelligent handling"""
    SPIKE = "spike"                  # Sudden increase in errors
    SUSTAINED = "sustained"          # Continuous high error rate
    INTERMITTENT = "intermittent"    # Periodic error bursts
    CASCADING = "cascading"          # Failures spreading across services
    THRESHOLD_BREACH = "threshold_breach"  # SLA/threshold violations


class RecoveryStrategy(Enum):
    """Recovery strategies for different error types"""
    RETRY_EXPONENTIAL = "retry_exponential"
    RETRY_LINEAR = "retry_linear"
    CIRCUIT_BREAK = "circuit_break"
    FALLBACK = "fallback"
    GRACEFUL_DEGRADE = "graceful_degrade"
    MANUAL_INTERVENTION = "manual_intervention"
    AUTO_SCALE = "auto_scale"
    CACHE_FALLBACK = "cache_fallback"


@dataclass
class ErrorContext:
    """Rich error context for analysis and correlation"""
    error_id: str
    correlation_id: str
    timestamp: datetime
    severity: ErrorSeverity
    category: ErrorCategory
    pattern: Optional[ErrorPattern]
    service: str
    operation: str
    user_id: Optional[str]
    request_id: Optional[str]
    error_type: str
    error_message: str
    stack_trace: str
    environment: Dict[str, Any]
    metadata: Dict[str, Any]
    suggested_actions: List[str]
    recovery_strategy: Optional[RecoveryStrategy]
    cost_impact: Optional[float]
    business_impact: str


@dataclass
class ErrorSignature:
    """Unique error signature for correlation"""
    signature_hash: str
    error_type: str
    normalized_message: str
    service: str
    operation: str
    first_seen: datetime
    last_seen: datetime
    occurrence_count: int
    avg_frequency_per_hour: float


class ErrorClassifier:
    """Intelligent error classification and categorization"""
    
    def __init__(self):
        self.classification_rules: Dict[str, Dict] = self._load_classification_rules()
        self.learned_patterns: Dict[str, ErrorSignature] = {}
        self._lock = threading.RLock()
    
    def classify_error(
        self,
        error: Exception,
        context: Dict[str, Any] = None
    ) -> ErrorContext:
        """
        Classify error with comprehensive analysis
        
        Args:
            error: The exception to classify
            context: Additional context information
            
        Returns:
            ErrorContext with full classification details
        """
        context = context or {}
        error_id = str(uuid.uuid4())
        correlation_id = context.get('correlation_id', str(uuid.uuid4()))
        
        # Basic classification
        error_type = type(error).__name__
        error_message = str(error)
        severity = self._determine_severity(error, context)
        category = self._determine_category(error, context)
        
        # Pattern recognition
        signature = self._create_error_signature(error, context)
        pattern = self._detect_pattern(signature)
        
        # Recovery strategy
        recovery_strategy = self._suggest_recovery_strategy(error, category, severity)
        
        # Cost and business impact analysis
        cost_impact = self._estimate_cost_impact(error, context)
        business_impact = self._assess_business_impact(error, context, severity)
        
        # Suggested actions
        suggested_actions = self._generate_suggested_actions(
            error, category, severity, recovery_strategy
        )
        
        # Environment context
        environment = {
            'service': context.get('service', 'unknown'),
            'operation': context.get('operation', 'unknown'),
            'node_id': context.get('node_id'),
            'deployment_version': context.get('version'),
            'resource_usage': context.get('resource_usage', {}),
            'active_users': context.get('active_users', 0)
        }
        
        error_context = ErrorContext(
            error_id=error_id,
            correlation_id=correlation_id,
            timestamp=datetime.now(),
            severity=severity,
            category=category,
            pattern=pattern,
            service=environment['service'],
            operation=environment['operation'],
            user_id=context.get('user_id'),
            request_id=context.get('request_id'),
            error_type=error_type,
            error_message=error_message,
            stack_trace=traceback.format_exc(),
            environment=environment,
            metadata=context.get('metadata', {}),
            suggested_actions=suggested_actions,
            recovery_strategy=recovery_strategy,
            cost_impact=cost_impact,
            business_impact=business_impact
        )
        
        # Update learned patterns
        self._update_learned_patterns(signature)
        
        return error_context
    
    def _determine_severity(self, error: Exception, context: Dict[str, Any]) -> ErrorSeverity:
        """Determine error severity based on type and context"""
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        # Critical errors
        if any(keyword in error_message for keyword in [
            'out of memory', 'disk full', 'database connection lost',
            'security breach', 'data corruption', 'system failure'
        ]):
            return ErrorSeverity.CRITICAL
        
        # High severity errors
        if any(keyword in error_message for keyword in [
            'authentication failed', 'unauthorized access', 'quota exceeded',
            'service unavailable', 'timeout', 'connection refused'
        ]):
            return ErrorSeverity.HIGH
        
        # Check for specific exception types
        severity_map = {
            'AuthenticationException': ErrorSeverity.HIGH,
            'DatabaseException': ErrorSeverity.HIGH,
            'ExternalAPIException': ErrorSeverity.MEDIUM,
            'RateLimitException': ErrorSeverity.MEDIUM,
            'ValidationException': ErrorSeverity.LOW,
            'NotFoundException': ErrorSeverity.LOW
        }
        
        if error_type in severity_map:
            return severity_map[error_type]
        
        # Context-based severity
        if context.get('critical_path', False):
            return ErrorSeverity.HIGH
        
        return ErrorSeverity.MEDIUM
    
    def _determine_category(self, error: Exception, context: Dict[str, Any]) -> ErrorCategory:
        """Determine error category for targeted handling"""
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        # Network-related errors
        if any(keyword in error_message for keyword in [
            'connection', 'network', 'timeout', 'unreachable', 'dns'
        ]):
            return ErrorCategory.NETWORK
        
        # Authentication/Authorization
        if any(keyword in error_message for keyword in [
            'authentication', 'authorization', 'unauthorized', 'forbidden', 'token'
        ]):
            return ErrorCategory.AUTHENTICATION
        
        # Rate limiting
        if any(keyword in error_message for keyword in [
            'rate limit', 'too many requests', 'quota', 'throttle'
        ]):
            return ErrorCategory.RATE_LIMIT
        
        # Data quality issues
        if any(keyword in error_message for keyword in [
            'validation', 'invalid data', 'corrupt', 'format error', 'parse error'
        ]):
            return ErrorCategory.DATA_QUALITY
        
        # Resource exhaustion
        if any(keyword in error_message for keyword in [
            'memory', 'disk', 'cpu', 'resource', 'capacity', 'limit'
        ]):
            return ErrorCategory.RESOURCE
        
        # Check specific exception types
        category_map = {
            'ExternalAPIException': ErrorCategory.DEPENDENCY,
            'ConfigurationException': ErrorCategory.CONFIGURATION,
            'ValidationException': ErrorCategory.DATA_QUALITY,
            'RateLimitException': ErrorCategory.RATE_LIMIT,
            'AuthenticationException': ErrorCategory.AUTHENTICATION,
            'DatabaseException': ErrorCategory.DEPENDENCY
        }
        
        if error_type in category_map:
            return category_map[error_type]
        
        # Default classification logic
        if hasattr(error, 'status_code'):
            status_code = getattr(error, 'status_code')
            if 400 <= status_code < 500:
                return ErrorCategory.BUSINESS_LOGIC
            elif 500 <= status_code < 600:
                return ErrorCategory.DEPENDENCY
        
        return ErrorCategory.TRANSIENT
    
    def _create_error_signature(self, error: Exception, context: Dict[str, Any]) -> ErrorSignature:
        """Create unique error signature for pattern recognition"""
        error_type = type(error).__name__
        
        # Normalize error message (remove variable parts)
        normalized_message = self._normalize_error_message(str(error))
        
        # Create signature hash
        signature_parts = [
            error_type,
            normalized_message,
            context.get('service', 'unknown'),
            context.get('operation', 'unknown')
        ]
        signature_hash = hashlib.md5('|'.join(signature_parts).encode()).hexdigest()
        
        now = datetime.now()
        existing_signature = self.learned_patterns.get(signature_hash)
        
        if existing_signature:
            existing_signature.last_seen = now
            existing_signature.occurrence_count += 1
            # Update frequency calculation
            hours_since_first = max(1, (now - existing_signature.first_seen).total_seconds() / 3600)
            existing_signature.avg_frequency_per_hour = existing_signature.occurrence_count / hours_since_first
            return existing_signature
        else:
            return ErrorSignature(
                signature_hash=signature_hash,
                error_type=error_type,
                normalized_message=normalized_message,
                service=context.get('service', 'unknown'),
                operation=context.get('operation', 'unknown'),
                first_seen=now,
                last_seen=now,
                occurrence_count=1,
                avg_frequency_per_hour=1.0
            )
    
    def _normalize_error_message(self, message: str) -> str:
        """Normalize error message by removing variable parts"""
        # Remove timestamps
        message = re.sub(r'\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}.*?(?=\s|$)', '<TIMESTAMP>', message)
        
        # Remove IDs and UUIDs
        message = re.sub(r'\b[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}\b', '<UUID>', message)
        message = re.sub(r'\b\d{6,}\b', '<ID>', message)
        
        # Remove IP addresses
        message = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '<IP>', message)
        
        # Remove file paths
        message = re.sub(r'/[^\s]+', '<PATH>', message)
        
        # Remove numbers that might be variable
        message = re.sub(r'\b\d+\.\d+\b', '<NUMBER>', message)
        
        return message.strip()
    
    def _detect_pattern(self, signature: ErrorSignature) -> Optional[ErrorPattern]:
        """Detect error patterns for proactive handling"""
        if signature.occurrence_count == 1:
            return None
        
        # Spike detection (sudden increase)
        if signature.avg_frequency_per_hour > 10 and signature.occurrence_count > 5:
            recent_window = datetime.now() - timedelta(minutes=15)
            if signature.last_seen > recent_window:
                return ErrorPattern.SPIKE
        
        # Sustained errors
        if signature.avg_frequency_per_hour > 1 and signature.occurrence_count > 10:
            return ErrorPattern.SUSTAINED
        
        # Intermittent errors
        hours_active = max(1, (signature.last_seen - signature.first_seen).total_seconds() / 3600)
        if hours_active > 2 and signature.avg_frequency_per_hour < 1:
            return ErrorPattern.INTERMITTENT
        
        return None
    
    def _suggest_recovery_strategy(
        self,
        error: Exception,
        category: ErrorCategory,
        severity: ErrorSeverity
    ) -> Optional[RecoveryStrategy]:
        """Suggest appropriate recovery strategy"""
        strategy_map = {
            ErrorCategory.TRANSIENT: RecoveryStrategy.RETRY_EXPONENTIAL,
            ErrorCategory.RATE_LIMIT: RecoveryStrategy.CIRCUIT_BREAK,
            ErrorCategory.NETWORK: RecoveryStrategy.RETRY_EXPONENTIAL,
            ErrorCategory.DEPENDENCY: RecoveryStrategy.FALLBACK,
            ErrorCategory.RESOURCE: RecoveryStrategy.AUTO_SCALE,
            ErrorCategory.AUTHENTICATION: RecoveryStrategy.MANUAL_INTERVENTION,
            ErrorCategory.DATA_QUALITY: RecoveryStrategy.GRACEFUL_DEGRADE,
            ErrorCategory.CONFIGURATION: RecoveryStrategy.MANUAL_INTERVENTION,
            ErrorCategory.BUSINESS_LOGIC: RecoveryStrategy.GRACEFUL_DEGRADE
        }
        
        base_strategy = strategy_map.get(category, RecoveryStrategy.RETRY_EXPONENTIAL)
        
        # Adjust based on severity
        if severity == ErrorSeverity.CRITICAL:
            if base_strategy in [RecoveryStrategy.RETRY_EXPONENTIAL, RecoveryStrategy.RETRY_LINEAR]:
                return RecoveryStrategy.MANUAL_INTERVENTION
        
        return base_strategy
    
    def _estimate_cost_impact(self, error: Exception, context: Dict[str, Any]) -> Optional[float]:
        """Estimate financial cost impact of error"""
        error_type = type(error).__name__
        service = context.get('service', '')
        
        # Base cost estimates per error type
        cost_estimates = {
            'ExternalAPIException': 0.001,  # API call cost
            'DatabaseException': 0.01,     # Database operation cost
            'RateLimitException': 0.0,     # No direct cost but opportunity cost
            'AuthenticationException': 0.005,  # Security overhead
        }
        
        base_cost = cost_estimates.get(error_type, 0.001)
        
        # Scale based on service criticality
        service_multiplier = {
            'recommendation_engine': 5.0,
            'data_ingestion': 3.0,
            'analysis': 2.0,
            'api': 1.5,
        }.get(service, 1.0)
        
        return base_cost * service_multiplier
    
    def _assess_business_impact(
        self,
        error: Exception,
        context: Dict[str, Any],
        severity: ErrorSeverity
    ) -> str:
        """Assess business impact of error"""
        service = context.get('service', '')
        operation = context.get('operation', '')
        
        if severity == ErrorSeverity.CRITICAL:
            return "Service disruption - immediate revenue impact"
        elif severity == ErrorSeverity.HIGH:
            if 'recommendation' in service:
                return "Recommendation quality degraded - potential user dissatisfaction"
            elif 'data_ingestion' in service:
                return "Data freshness impacted - analysis accuracy reduced"
            else:
                return "Service functionality reduced - user experience impacted"
        elif severity == ErrorSeverity.MEDIUM:
            return "Performance degradation - minimal user impact"
        else:
            return "Minor issue - no significant business impact"
    
    def _generate_suggested_actions(
        self,
        error: Exception,
        category: ErrorCategory,
        severity: ErrorSeverity,
        recovery_strategy: Optional[RecoveryStrategy]
    ) -> List[str]:
        """Generate actionable suggestions for error resolution"""
        actions = []
        
        # Immediate actions based on severity
        if severity == ErrorSeverity.CRITICAL:
            actions.append("Alert on-call engineer immediately")
            actions.append("Initiate incident response procedure")
            actions.append("Consider activating disaster recovery")
        elif severity == ErrorSeverity.HIGH:
            actions.append("Notify development team")
            actions.append("Monitor error frequency")
        
        # Category-specific actions
        category_actions = {
            ErrorCategory.RATE_LIMIT: [
                "Implement exponential backoff",
                "Switch to alternative API provider",
                "Enable cached data fallback"
            ],
            ErrorCategory.NETWORK: [
                "Check network connectivity",
                "Verify DNS resolution",
                "Review firewall configurations"
            ],
            ErrorCategory.AUTHENTICATION: [
                "Verify API credentials",
                "Check token expiration",
                "Review authentication configuration"
            ],
            ErrorCategory.DATA_QUALITY: [
                "Enable data validation",
                "Review data source quality",
                "Implement data cleansing"
            ],
            ErrorCategory.RESOURCE: [
                "Monitor resource usage",
                "Consider horizontal scaling",
                "Review resource allocation"
            ]
        }
        
        actions.extend(category_actions.get(category, []))
        
        # Recovery strategy actions
        if recovery_strategy == RecoveryStrategy.CIRCUIT_BREAK:
            actions.append("Enable circuit breaker protection")
        elif recovery_strategy == RecoveryStrategy.FALLBACK:
            actions.append("Activate fallback service")
        elif recovery_strategy == RecoveryStrategy.CACHE_FALLBACK:
            actions.append("Switch to cached data")
        
        return actions
    
    def _update_learned_patterns(self, signature: ErrorSignature):
        """Update learned patterns for improved classification"""
        with self._lock:
            self.learned_patterns[signature.signature_hash] = signature
    
    def _load_classification_rules(self) -> Dict[str, Dict]:
        """Load classification rules from configuration"""
        try:
            rules_file = Path("config/error_classification_rules.json")
            if rules_file.exists():
                with open(rules_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load classification rules: {e}")
        
        # Default rules
        return {
            "severity_keywords": {
                "critical": ["memory", "disk", "corruption", "security", "breach"],
                "high": ["timeout", "unavailable", "unauthorized", "quota"],
                "medium": ["validation", "format", "parse"],
                "low": ["not found", "invalid input"]
            },
            "category_patterns": {
                "network": ["connection", "timeout", "unreachable"],
                "authentication": ["unauthorized", "token", "credential"],
                "rate_limit": ["rate limit", "throttle", "quota"],
                "data_quality": ["validation", "format", "corrupt"]
            }
        }


class ErrorCorrelationEngine:
    """Correlate errors across services to identify root causes"""
    
    def __init__(self, time_window_minutes: int = 10, correlation_threshold: float = 0.7):
        self.time_window_minutes = time_window_minutes
        self.correlation_threshold = correlation_threshold
        self.error_timeline: deque = deque(maxlen=10000)
        self.correlation_cache: Dict[str, List[str]] = {}
        self._lock = threading.RLock()
    
    def add_error_context(self, error_context: ErrorContext):
        """Add error context to correlation timeline"""
        with self._lock:
            self.error_timeline.append(error_context)
            self._cleanup_old_errors()
            
            # Perform real-time correlation
            correlations = self._find_correlations(error_context)
            if correlations:
                self._handle_correlations(error_context, correlations)
    
    def _cleanup_old_errors(self):
        """Remove errors outside the correlation time window"""
        cutoff_time = datetime.now() - timedelta(minutes=self.time_window_minutes)
        while self.error_timeline and self.error_timeline[0].timestamp < cutoff_time:
            self.error_timeline.popleft()
    
    def _find_correlations(self, error_context: ErrorContext) -> List[ErrorContext]:
        """Find correlated errors within time window"""
        correlations = []
        
        for other_error in self.error_timeline:
            if other_error.error_id == error_context.error_id:
                continue
            
            correlation_score = self._calculate_correlation_score(error_context, other_error)
            if correlation_score >= self.correlation_threshold:
                correlations.append(other_error)
        
        return correlations
    
    def _calculate_correlation_score(self, error1: ErrorContext, error2: ErrorContext) -> float:
        """Calculate correlation score between two errors"""
        score = 0.0
        
        # Time proximity (closer in time = higher correlation)
        time_diff = abs((error1.timestamp - error2.timestamp).total_seconds())
        if time_diff < 60:  # Within 1 minute
            score += 0.4
        elif time_diff < 300:  # Within 5 minutes
            score += 0.2
        
        # Same service or related services
        if error1.service == error2.service:
            score += 0.3
        
        # Same error category
        if error1.category == error2.category:
            score += 0.2
        
        # Same correlation ID (part of same request flow)
        if error1.correlation_id == error2.correlation_id:
            score += 0.5
        
        # Same user (user-specific issue)
        if error1.user_id and error1.user_id == error2.user_id:
            score += 0.1
        
        return min(score, 1.0)
    
    def _handle_correlations(self, error_context: ErrorContext, correlations: List[ErrorContext]):
        """Handle discovered correlations"""
        correlation_key = f"{error_context.service}_{error_context.category.value}"
        
        with self._lock:
            if correlation_key not in self.correlation_cache:
                self.correlation_cache[correlation_key] = []
            
            # Add correlation IDs
            correlation_ids = [c.error_id for c in correlations]
            self.correlation_cache[correlation_key].extend(correlation_ids)
            
            # Log correlation discovery
            logger.warning(
                f"Error correlation detected: {error_context.error_id} "
                f"correlates with {len(correlations)} other errors"
            )
            
            # Check for cascade patterns
            if len(correlations) > 5:
                logger.critical(
                    f"Potential cascading failure detected in {error_context.service}"
                )
    
    def get_root_cause_analysis(self, error_id: str) -> Dict[str, Any]:
        """Perform root cause analysis for an error"""
        target_error = None
        for error in self.error_timeline:
            if error.error_id == error_id:
                target_error = error
                break
        
        if not target_error:
            return {"error": "Error not found in correlation timeline"}
        
        correlations = self._find_correlations(target_error)
        
        # Analyze correlation patterns
        service_breakdown = defaultdict(int)
        category_breakdown = defaultdict(int)
        timeline = []
        
        for corr in correlations:
            service_breakdown[corr.service] += 1
            category_breakdown[corr.category.value] += 1
            timeline.append({
                'timestamp': corr.timestamp.isoformat(),
                'service': corr.service,
                'error_type': corr.error_type,
                'message': corr.error_message
            })
        
        # Sort timeline chronologically
        timeline.sort(key=lambda x: x['timestamp'])
        
        # Determine likely root cause
        root_cause_service = max(service_breakdown.items(), key=lambda x: x[1])[0] if service_breakdown else "unknown"
        
        return {
            'target_error_id': error_id,
            'correlation_count': len(correlations),
            'likely_root_cause_service': root_cause_service,
            'affected_services': dict(service_breakdown),
            'error_categories': dict(category_breakdown),
            'correlation_timeline': timeline,
            'analysis_timestamp': datetime.now().isoformat()
        }


class ErrorHandlingManager:
    """Central manager for enhanced error handling across the application"""
    
    def __init__(self):
        self.classifier = ErrorClassifier()
        self.correlator = ErrorCorrelationEngine()
        self.error_history: deque = deque(maxlen=50000)
        self.active_incidents: Dict[str, Dict] = {}
        self._lock = threading.RLock()
    
    async def handle_error(
        self,
        error: Exception,
        context: Dict[str, Any] = None,
        should_raise: bool = True
    ) -> ErrorContext:
        """
        Comprehensive error handling with classification and correlation
        
        Args:
            error: The exception to handle
            context: Additional context information
            should_raise: Whether to re-raise the error after handling
            
        Returns:
            ErrorContext with full analysis
        """
        # Classify the error
        error_context = self.classifier.classify_error(error, context)
        
        # Add to correlation engine
        self.correlator.add_error_context(error_context)
        
        # Store in history
        with self._lock:
            self.error_history.append(error_context)
        
        # Log with structured logging
        correlation_logger = StructuredLogger(f"{__name__}.{error_context.correlation_id}")
        log_data = {
            'error_id': error_context.error_id,
            'severity': error_context.severity.name,
            'category': error_context.category.value,
            'service': error_context.service,
            'operation': error_context.operation,
            'recovery_strategy': error_context.recovery_strategy.value if error_context.recovery_strategy else None,
            'cost_impact': error_context.cost_impact,
            'suggested_actions': error_context.suggested_actions
        }
        
        if error_context.severity >= ErrorSeverity.HIGH:
            correlation_logger.error(f"High severity error: {error_context.error_message}", extra=log_data)
        elif error_context.severity == ErrorSeverity.MEDIUM:
            correlation_logger.warning(f"Medium severity error: {error_context.error_message}", extra=log_data)
        else:
            correlation_logger.info(f"Low severity error: {error_context.error_message}", extra=log_data)
        
        # Handle incident creation/escalation
        await self._handle_incident_management(error_context)
        
        # Execute recovery strategy if applicable
        if error_context.recovery_strategy:
            await self._execute_recovery_strategy(error_context)
        
        if should_raise:
            raise error
        
        return error_context
    
    async def _handle_incident_management(self, error_context: ErrorContext):
        """Handle incident creation and escalation"""
        if error_context.severity >= ErrorSeverity.HIGH:
            incident_key = f"{error_context.service}_{error_context.category.value}"
            
            with self._lock:
                if incident_key not in self.active_incidents:
                    # Create new incident
                    self.active_incidents[incident_key] = {
                        'incident_id': str(uuid.uuid4()),
                        'created_at': error_context.timestamp,
                        'service': error_context.service,
                        'category': error_context.category.value,
                        'severity': error_context.severity.name,
                        'error_count': 1,
                        'last_error': error_context.timestamp,
                        'status': 'active'
                    }
                    logger.critical(f"New incident created: {incident_key}")
                else:
                    # Update existing incident
                    incident = self.active_incidents[incident_key]
                    incident['error_count'] += 1
                    incident['last_error'] = error_context.timestamp
                    
                    # Escalate if error count is high
                    if incident['error_count'] > 10:
                        logger.critical(f"Incident escalation: {incident_key} has {incident['error_count']} errors")
    
    async def _execute_recovery_strategy(self, error_context: ErrorContext):
        """Execute appropriate recovery strategy"""
        strategy = error_context.recovery_strategy
        
        if strategy == RecoveryStrategy.CACHE_FALLBACK:
            logger.info(f"Attempting cache fallback for {error_context.service}")
            # Implementation would integrate with cache system
            
        elif strategy == RecoveryStrategy.GRACEFUL_DEGRADE:
            logger.info(f"Initiating graceful degradation for {error_context.service}")
            # Implementation would reduce service functionality
            
        elif strategy == RecoveryStrategy.AUTO_SCALE:
            logger.info(f"Triggering auto-scale for {error_context.service}")
            # Implementation would integrate with scaling system
    
    def get_error_analytics(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Get comprehensive error analytics"""
        cutoff = datetime.now() - timedelta(hours=time_window_hours)
        recent_errors = [e for e in self.error_history if e.timestamp > cutoff]
        
        if not recent_errors:
            return {'message': 'No errors in specified time window'}
        
        # Aggregate statistics
        severity_breakdown = defaultdict(int)
        category_breakdown = defaultdict(int)
        service_breakdown = defaultdict(int)
        cost_impact_total = 0.0
        
        for error in recent_errors:
            severity_breakdown[error.severity.name] += 1
            category_breakdown[error.category.value] += 1
            service_breakdown[error.service] += 1
            if error.cost_impact:
                cost_impact_total += error.cost_impact
        
        # Top error patterns
        top_services = sorted(service_breakdown.items(), key=lambda x: x[1], reverse=True)[:5]
        top_categories = sorted(category_breakdown.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'time_window_hours': time_window_hours,
            'total_errors': len(recent_errors),
            'severity_breakdown': dict(severity_breakdown),
            'category_breakdown': dict(category_breakdown),
            'top_affected_services': top_services,
            'top_error_categories': top_categories,
            'estimated_cost_impact': round(cost_impact_total, 4),
            'active_incidents': len(self.active_incidents),
            'analysis_timestamp': datetime.now().isoformat()
        }


# Global error handling manager
error_handler = ErrorHandlingManager()


# Utility functions for API error handling
async def handle_api_error(error: Exception, operation: str, context: Dict[str, Any] = None):
    """
    Handle API errors with comprehensive logging and analysis
    
    Args:
        error: The exception that occurred
        operation: Description of the operation that failed
        context: Additional context information
    """
    if context is None:
        context = {}
    
    context.update({
        'operation': operation,
        'service': 'api'
    })
    
    await error_handler.handle_error(error, context, should_raise=False)


def validate_stock_symbol(symbol: str) -> bool:
    """
    Validate stock symbol format
    
    Args:
        symbol: Stock symbol to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not symbol or not isinstance(symbol, str):
        return False
    
    # Basic validation: alphanumeric, 1-5 characters, uppercase
    symbol = symbol.strip().upper()
    return (
        len(symbol) >= 1 and 
        len(symbol) <= 5 and 
        symbol.isalnum() and 
        symbol.isalpha()  # Only letters for basic validation
    )


# Decorator for automatic error handling
def with_error_handling(
    service: str = None,
    operation: str = None,
    critical_path: bool = False,
    should_raise: bool = True
):
    """
    Decorator to automatically handle errors with comprehensive analysis
    
    Args:
        service: Service name for context
        operation: Operation name for context  
        critical_path: Whether this is a critical code path
        should_raise: Whether to re-raise errors after handling
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            context = {
                'service': service or func.__module__.split('.')[-1],
                'operation': operation or func.__name__,
                'critical_path': critical_path,
                'correlation_id': kwargs.pop('correlation_id', str(uuid.uuid4()))
            }
            
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                await error_handler.handle_error(e, context, should_raise)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            context = {
                'service': service or func.__module__.split('.')[-1],
                'operation': operation or func.__name__,
                'critical_path': critical_path,
                'correlation_id': kwargs.pop('correlation_id', str(uuid.uuid4()))
            }
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Run async handler in sync context
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    loop.run_until_complete(error_handler.handle_error(e, context, should_raise))
                except RuntimeError:
                    # Create new event loop if none exists
                    asyncio.set_event_loop(asyncio.new_event_loop())
                    loop = asyncio.get_event_loop()
                    loop.run_until_complete(error_handler.handle_error(e, context, should_raise))
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator