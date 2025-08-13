"""
Advanced Circuit Breaker System with Adaptive Thresholds
Enhanced error handling and resilience features for the investment analysis application
"""

import asyncio
import time
import math
import statistics
from typing import Callable, Optional, Any, Dict, Union, List, Type
from functools import wraps
from enum import Enum
from datetime import datetime, timedelta
import logging
from collections import deque, defaultdict
import threading
import json
from dataclasses import dataclass, asdict
from pathlib import Path

from .exceptions import CircuitBreakerError, ExternalAPIException

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Enhanced circuit breaker states"""
    CLOSED = "closed"           # Normal operation
    OPEN = "open"              # Failing fast
    HALF_OPEN = "half_open"    # Testing recovery
    FORCE_OPEN = "force_open"  # Manually opened


class FailureType(Enum):
    """Types of failures for categorization"""
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    SERVER_ERROR = "server_error"
    NETWORK_ERROR = "network_error"
    AUTHENTICATION = "authentication"
    QUOTA_EXCEEDED = "quota_exceeded"
    DATA_QUALITY = "data_quality"
    UNKNOWN = "unknown"


@dataclass
class FailureMetrics:
    """Detailed failure metrics for analysis"""
    timestamp: datetime
    failure_type: FailureType
    response_time: float
    status_code: Optional[int]
    error_message: str
    retry_count: int = 0
    severity: int = 1  # 1-5 scale


@dataclass
class AdaptiveThresholds:
    """Adaptive thresholds based on historical data"""
    failure_threshold: int
    recovery_timeout: int
    success_threshold: int
    rate_limit_threshold: int
    timeout_threshold: int
    error_rate_threshold: float


class ProviderReliabilityTracker:
    """Tracks reliability metrics for adaptive threshold adjustment"""
    
    def __init__(self, provider_name: str, window_hours: int = 24):
        self.provider_name = provider_name
        self.window_hours = window_hours
        self.metrics: deque = deque(maxlen=1000)
        self.reliability_score = 1.0
        self.last_update = datetime.now()
        
    def add_metric(self, success: bool, response_time: float, failure_type: Optional[FailureType] = None):
        """Add performance metric"""
        metric = {
            'timestamp': datetime.now(),
            'success': success,
            'response_time': response_time,
            'failure_type': failure_type.value if failure_type else None
        }
        self.metrics.append(metric)
        self._update_reliability_score()
    
    def _update_reliability_score(self):
        """Update reliability score based on recent metrics"""
        if not self.metrics:
            return
            
        cutoff = datetime.now() - timedelta(hours=self.window_hours)
        recent_metrics = [m for m in self.metrics if m['timestamp'] > cutoff]
        
        if not recent_metrics:
            return
            
        success_rate = sum(1 for m in recent_metrics if m['success']) / len(recent_metrics)
        avg_response_time = statistics.mean([m['response_time'] for m in recent_metrics])
        
        # Calculate reliability score (0.0 to 1.0)
        time_penalty = min(avg_response_time / 5.0, 0.3)  # Cap at 30% penalty
        self.reliability_score = max(0.1, success_rate - time_penalty)
        self.last_update = datetime.now()
    
    def get_adaptive_thresholds(self, base_thresholds: AdaptiveThresholds) -> AdaptiveThresholds:
        """Get adaptive thresholds based on reliability"""
        reliability_factor = self.reliability_score
        
        # Adjust thresholds based on reliability
        if reliability_factor > 0.9:  # High reliability
            multiplier = 1.5
        elif reliability_factor > 0.7:  # Medium reliability  
            multiplier = 1.0
        else:  # Low reliability
            multiplier = 0.7
            
        return AdaptiveThresholds(
            failure_threshold=max(1, int(base_thresholds.failure_threshold * multiplier)),
            recovery_timeout=max(10, int(base_thresholds.recovery_timeout / multiplier)),
            success_threshold=base_thresholds.success_threshold,
            rate_limit_threshold=max(1, int(base_thresholds.rate_limit_threshold * multiplier)),
            timeout_threshold=max(1, int(base_thresholds.timeout_threshold * multiplier)),
            error_rate_threshold=max(0.1, base_thresholds.error_rate_threshold * multiplier)
        )


class EnhancedCircuitBreaker:
    """
    Advanced circuit breaker with adaptive thresholds and intelligent failure handling
    """
    
    def __init__(
        self,
        name: str,
        base_thresholds: AdaptiveThresholds,
        provider_name: Optional[str] = None,
        fallback_func: Optional[Callable] = None,
        on_state_change: Optional[Callable] = None,
        persistent_state: bool = True,
        max_jitter: float = 0.1
    ):
        """
        Initialize enhanced circuit breaker
        
        Args:
            name: Circuit breaker identifier
            base_thresholds: Base threshold configuration
            provider_name: External provider name for reliability tracking
            fallback_func: Fallback function when circuit is open
            on_state_change: Callback for state changes
            persistent_state: Whether to persist state to disk
            max_jitter: Maximum jitter for retry timing (0.0-1.0)
        """
        self.name = name
        self.base_thresholds = base_thresholds
        self.provider_name = provider_name or name
        self.fallback_func = fallback_func
        self.on_state_change = on_state_change
        self.persistent_state = persistent_state
        self.max_jitter = max_jitter
        
        # State management
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
        self._last_attempt_time = None
        self._consecutive_failures = 0
        
        # Enhanced metrics
        self._call_count = 0
        self._failure_history: deque[FailureMetrics] = deque(maxlen=1000)
        self._response_times: deque[float] = deque(maxlen=1000)
        self._failure_types: defaultdict[FailureType, int] = defaultdict(int)
        
        # Reliability tracking
        self.reliability_tracker = ProviderReliabilityTracker(self.provider_name)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Load persistent state
        if self.persistent_state:
            self._load_state()
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state with auto-update"""
        with self._lock:
            self._update_state()
            return self._state
    
    @property
    def current_thresholds(self) -> AdaptiveThresholds:
        """Get current adaptive thresholds"""
        return self.reliability_tracker.get_adaptive_thresholds(self.base_thresholds)
    
    def _update_state(self):
        """Update circuit state based on current conditions and adaptive thresholds"""
        thresholds = self.current_thresholds
        
        if self._state == CircuitState.OPEN:
            if self._should_attempt_recovery():
                logger.info(f"{self.name}: Attempting recovery, moving to HALF_OPEN")
                self._state = CircuitState.HALF_OPEN
                self._success_count = 0
                if self.on_state_change:
                    self.on_state_change(self._state)
        
        elif self._state == CircuitState.CLOSED:
            # Check if we should open based on failure rate
            if self._should_open_circuit():
                self._open_circuit("Failure threshold exceeded")
    
    def _should_attempt_recovery(self) -> bool:
        """Determine if circuit should attempt recovery with jitter"""
        if not self._last_failure_time:
            return True
            
        thresholds = self.current_thresholds
        base_timeout = thresholds.recovery_timeout
        
        # Add exponential backoff based on consecutive failures
        backoff_multiplier = min(2 ** min(self._consecutive_failures, 5), 60)
        timeout_with_backoff = base_timeout * backoff_multiplier
        
        # Add jitter to prevent thundering herd
        jitter = 1 + (hash(self.name) % 100 / 100 * self.max_jitter)
        final_timeout = timeout_with_backoff * jitter
        
        return time.time() - self._last_failure_time > final_timeout
    
    def _should_open_circuit(self) -> bool:
        """Determine if circuit should open based on adaptive criteria"""
        thresholds = self.current_thresholds
        
        # Check failure count
        if self._failure_count >= thresholds.failure_threshold:
            return True
            
        # Check error rate over recent time window
        recent_failures = self._get_recent_failures(minutes=5)
        if len(recent_failures) >= 5:  # Minimum sample size
            error_rate = len(recent_failures) / max(self._call_count, 1)
            if error_rate > thresholds.error_rate_threshold:
                return True
        
        # Check specific failure type thresholds
        for failure_type, count in self._failure_types.items():
            if failure_type == FailureType.RATE_LIMIT and count >= thresholds.rate_limit_threshold:
                return True
            elif failure_type == FailureType.TIMEOUT and count >= thresholds.timeout_threshold:
                return True
                
        return False
    
    def _get_recent_failures(self, minutes: int = 5) -> List[FailureMetrics]:
        """Get recent failures within time window"""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        return [f for f in self._failure_history if f.timestamp > cutoff]
    
    def _open_circuit(self, reason: str):
        """Open the circuit with logging and callbacks"""
        self._state = CircuitState.OPEN
        self._consecutive_failures += 1
        logger.error(f"{self.name}: Circuit opened - {reason}")
        
        if self.on_state_change:
            self.on_state_change(self._state)
            
        if self.persistent_state:
            self._save_state()
    
    def _record_success(self, response_time: float):
        """Record successful call with metrics"""
        with self._lock:
            self._failure_count = 0
            self._consecutive_failures = 0
            self._failure_types.clear()
            self._call_count += 1
            self._response_times.append(response_time)
            
            # Update reliability tracker
            self.reliability_tracker.add_metric(True, response_time)
            
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                thresholds = self.current_thresholds
                if self._success_count >= thresholds.success_threshold:
                    self._state = CircuitState.CLOSED
                    logger.info(f"{self.name}: Circuit closed after successful recovery")
                    if self.on_state_change:
                        self.on_state_change(self._state)
            
            if self.persistent_state:
                self._save_state()
    
    def _record_failure(
        self, 
        failure_type: FailureType,
        response_time: float,
        error_message: str,
        status_code: Optional[int] = None,
        severity: int = 1
    ):
        """Record failed call with detailed metrics"""
        with self._lock:
            self._failure_count += 1
            self._call_count += 1
            self._last_failure_time = time.time()
            self._failure_types[failure_type] += 1
            
            # Create detailed failure metric
            failure_metric = FailureMetrics(
                timestamp=datetime.now(),
                failure_type=failure_type,
                response_time=response_time,
                status_code=status_code,
                error_message=error_message,
                severity=severity
            )
            self._failure_history.append(failure_metric)
            
            # Update reliability tracker
            self.reliability_tracker.add_metric(False, response_time, failure_type)
            
            if self._state == CircuitState.HALF_OPEN:
                self._open_circuit("Recovery attempt failed")
            elif self._should_open_circuit():
                self._open_circuit(f"Failure threshold exceeded: {failure_type.value}")
            
            if self.persistent_state:
                self._save_state()
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function through circuit breaker with enhanced error handling
        """
        if self.state == CircuitState.OPEN or self.state == CircuitState.FORCE_OPEN:
            if self.fallback_func:
                logger.info(f"{self.name}: Circuit open, using fallback")
                return await self._safe_call(self.fallback_func, *args, **kwargs)
            else:
                raise CircuitBreakerError(f"{self.name}: Circuit is open", self.name)
        
        start_time = time.time()
        try:
            result = await self._safe_call(func, *args, **kwargs)
            response_time = time.time() - start_time
            self._record_success(response_time)
            return result
            
        except Exception as e:
            response_time = time.time() - start_time
            failure_type = self._classify_error(e)
            status_code = getattr(e, 'status_code', None)
            severity = self._calculate_severity(failure_type, e)
            
            self._record_failure(
                failure_type=failure_type,
                response_time=response_time,
                error_message=str(e),
                status_code=status_code,
                severity=severity
            )
            
            # Try fallback if available and not already in fallback
            if self.fallback_func and self.state == CircuitState.OPEN:
                logger.warning(f"{self.name}: Primary failed, using fallback")
                try:
                    return await self._safe_call(self.fallback_func, *args, **kwargs)
                except Exception as fallback_error:
                    logger.error(f"{self.name}: Fallback also failed: {fallback_error}")
            
            raise e
    
    async def _safe_call(self, func: Callable, *args, **kwargs) -> Any:
        """Safely call function (sync or async)"""
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    
    def _classify_error(self, error: Exception) -> FailureType:
        """Classify error type for targeted handling"""
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()
        
        if 'timeout' in error_str or 'timeout' in error_type:
            return FailureType.TIMEOUT
        elif 'rate limit' in error_str or 'too many requests' in error_str:
            return FailureType.RATE_LIMIT
        elif 'quota' in error_str or 'limit exceeded' in error_str:
            return FailureType.QUOTA_EXCEEDED
        elif 'unauthorized' in error_str or 'authentication' in error_str:
            return FailureType.AUTHENTICATION
        elif 'network' in error_str or 'connection' in error_str:
            return FailureType.NETWORK_ERROR
        elif hasattr(error, 'status_code') and 500 <= getattr(error, 'status_code', 0) < 600:
            return FailureType.SERVER_ERROR
        else:
            return FailureType.UNKNOWN
    
    def _calculate_severity(self, failure_type: FailureType, error: Exception) -> int:
        """Calculate failure severity (1-5 scale)"""
        severity_map = {
            FailureType.RATE_LIMIT: 2,
            FailureType.QUOTA_EXCEEDED: 3,
            FailureType.TIMEOUT: 3,
            FailureType.NETWORK_ERROR: 4,
            FailureType.SERVER_ERROR: 4,
            FailureType.AUTHENTICATION: 5,
            FailureType.DATA_QUALITY: 2,
            FailureType.UNKNOWN: 3
        }
        return severity_map.get(failure_type, 3)
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive circuit breaker metrics"""
        with self._lock:
            recent_failures = self._get_recent_failures(minutes=5)
            recent_error_rate = len(recent_failures) / max(self._call_count, 1) if self._call_count > 0 else 0
            
            avg_response_time = statistics.mean(self._response_times) if self._response_times else 0
            p95_response_time = (
                statistics.quantiles(self._response_times, n=20)[18] 
                if len(self._response_times) >= 20 else avg_response_time
            )
            
            failure_breakdown = {ft.value: count for ft, count in self._failure_types.items()}
            
            thresholds = self.current_thresholds
            
            return {
                'name': self.name,
                'state': self._state.value,
                'provider': self.provider_name,
                'reliability_score': round(self.reliability_tracker.reliability_score, 3),
                'thresholds': asdict(thresholds),
                'metrics': {
                    'total_calls': self._call_count,
                    'failure_count': self._failure_count,
                    'success_count': self._success_count,
                    'consecutive_failures': self._consecutive_failures,
                    'recent_error_rate_5min': round(recent_error_rate, 3),
                    'avg_response_time_ms': round(avg_response_time * 1000, 2),
                    'p95_response_time_ms': round(p95_response_time * 1000, 2),
                    'failure_breakdown': failure_breakdown,
                    'last_failure': self._last_failure_time
                },
                'recent_failures': [
                    {
                        'timestamp': f.timestamp.isoformat(),
                        'type': f.failure_type.value,
                        'message': f.error_message,
                        'severity': f.severity
                    }
                    for f in recent_failures[-10:]  # Last 10 failures
                ]
            }
    
    def force_open(self, reason: str = "Manual intervention"):
        """Manually force circuit open"""
        with self._lock:
            self._state = CircuitState.FORCE_OPEN
            logger.warning(f"{self.name}: Circuit force opened - {reason}")
            if self.on_state_change:
                self.on_state_change(self._state)
    
    def force_close(self, reason: str = "Manual intervention"):
        """Manually force circuit closed"""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._consecutive_failures = 0
            self._failure_types.clear()
            logger.info(f"{self.name}: Circuit force closed - {reason}")
            if self.on_state_change:
                self.on_state_change(self._state)
    
    def reset(self):
        """Reset circuit breaker to initial state"""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._consecutive_failures = 0
            self._last_failure_time = None
            self._failure_types.clear()
            logger.info(f"{self.name}: Circuit breaker reset")
    
    def _save_state(self):
        """Save circuit breaker state to disk"""
        if not self.persistent_state:
            return
            
        try:
            state_dir = Path("data/circuit_breaker_states")
            state_dir.mkdir(parents=True, exist_ok=True)
            
            state_data = {
                'state': self._state.value,
                'failure_count': self._failure_count,
                'consecutive_failures': self._consecutive_failures,
                'last_failure_time': self._last_failure_time,
                'reliability_score': self.reliability_tracker.reliability_score,
                'updated_at': datetime.now().isoformat()
            }
            
            state_file = state_dir / f"{self.name}.json"
            with open(state_file, 'w') as f:
                json.dump(state_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save circuit breaker state: {e}")
    
    def _load_state(self):
        """Load circuit breaker state from disk"""
        try:
            state_file = Path(f"data/circuit_breaker_states/{self.name}.json")
            if state_file.exists():
                with open(state_file, 'r') as f:
                    state_data = json.load(f)
                
                # Only restore non-persistent failures (not force states)
                if state_data.get('state') in ['closed', 'open', 'half_open']:
                    self._state = CircuitState(state_data['state'])
                    self._failure_count = state_data.get('failure_count', 0)
                    self._consecutive_failures = state_data.get('consecutive_failures', 0)
                    self._last_failure_time = state_data.get('last_failure_time')
                    self.reliability_tracker.reliability_score = state_data.get('reliability_score', 1.0)
                    
                    logger.info(f"{self.name}: Loaded previous state: {self._state.value}")
                
        except Exception as e:
            logger.error(f"Failed to load circuit breaker state: {e}")


class CascadingFailurePreventor:
    """Prevents cascading failures by isolating services and managing load"""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, EnhancedCircuitBreaker] = {}
        self.service_dependencies: Dict[str, List[str]] = {}
        self.isolation_groups: Dict[str, List[str]] = {}
        self._lock = threading.RLock()
    
    def register_circuit_breaker(self, name: str, breaker: EnhancedCircuitBreaker):
        """Register circuit breaker for monitoring"""
        with self._lock:
            self.circuit_breakers[name] = breaker
            breaker.on_state_change = self._on_circuit_state_change
    
    def add_dependency(self, service: str, dependency: str):
        """Add service dependency relationship"""
        with self._lock:
            if service not in self.service_dependencies:
                self.service_dependencies[service] = []
            self.service_dependencies[service].append(dependency)
    
    def add_isolation_group(self, group_name: str, services: List[str]):
        """Add isolation group to prevent cascading failures"""
        with self._lock:
            self.isolation_groups[group_name] = services
    
    def _on_circuit_state_change(self, state: CircuitState):
        """Handle circuit state changes to prevent cascades"""
        if state == CircuitState.OPEN:
            self._check_cascade_risk()
    
    def _check_cascade_risk(self):
        """Check for cascade failure risk and take preventive action"""
        with self._lock:
            open_circuits = [
                name for name, breaker in self.circuit_breakers.items()
                if breaker.state == CircuitState.OPEN
            ]
            
            # Check if isolation is needed
            for group_name, services in self.isolation_groups.items():
                open_in_group = [s for s in services if s in open_circuits]
                if len(open_in_group) / len(services) > 0.5:  # More than 50% open
                    logger.critical(f"Isolation group {group_name} at risk, implementing isolation")
                    self._implement_isolation(group_name, services)
    
    def _implement_isolation(self, group_name: str, services: List[str]):
        """Implement service isolation to prevent cascades"""
        for service in services:
            if service in self.circuit_breakers:
                # Temporarily increase recovery timeout to reduce load
                breaker = self.circuit_breakers[service]
                current_thresholds = breaker.current_thresholds
                breaker.base_thresholds.recovery_timeout *= 2
                logger.warning(f"Isolated {service} with increased recovery timeout")