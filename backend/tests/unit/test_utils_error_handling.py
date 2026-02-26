"""
Unit tests for backend/utils/enhanced_error_handling.py

Pure unit tests covering:
- ErrorSeverity, ErrorCategory, ErrorPattern, RecoveryStrategy enums
- ErrorClassifier: severity determination, category determination,
  error signature creation, pattern detection, recovery strategy,
  cost impact estimation, business impact, suggested actions,
  message normalization, learned patterns
- ErrorCorrelationEngine: add_error, cleanup, correlation scoring,
  root cause analysis
- ErrorHandlingManager: handle_error, incident management, analytics
- validate_stock_symbol utility
- with_error_handling decorator

No Redis, no database, no network.
"""

import os
os.environ["TESTING"] = "True"
os.environ["DEBUG"] = "True"
os.environ["DATABASE_URL"] = "sqlite:///:memory:"

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, AsyncMock
from collections import deque

from backend.utils.enhanced_error_handling import (
    ErrorSeverity,
    ErrorCategory,
    ErrorPattern,
    RecoveryStrategy,
    ErrorContext,
    ErrorSignature,
    ErrorClassifier,
    ErrorCorrelationEngine,
    ErrorHandlingManager,
    validate_stock_symbol,
    with_error_handling,
)
from backend.utils.exceptions import (
    ValidationException,
    AuthenticationException,
    ExternalAPIException,
    RateLimitException,
    NotFoundException,
    DatabaseException,
    ConfigurationException,
    APIException,
)


# ===========================================================================
# Enum Tests
# ===========================================================================

class TestErrorSeverity:
    """Tests for ErrorSeverity enum values and ordering."""

    def test_critical_value(self):
        assert ErrorSeverity.CRITICAL == 5

    def test_high_value(self):
        assert ErrorSeverity.HIGH == 4

    def test_medium_value(self):
        assert ErrorSeverity.MEDIUM == 3

    def test_low_value(self):
        assert ErrorSeverity.LOW == 2

    def test_info_value(self):
        assert ErrorSeverity.INFO == 1

    def test_critical_greater_than_high(self):
        assert ErrorSeverity.CRITICAL > ErrorSeverity.HIGH

    def test_ordering_full(self):
        assert ErrorSeverity.INFO < ErrorSeverity.LOW < ErrorSeverity.MEDIUM < ErrorSeverity.HIGH < ErrorSeverity.CRITICAL


class TestErrorCategory:
    """Tests for ErrorCategory enum values."""

    def test_transient_value(self):
        assert ErrorCategory.TRANSIENT.value == "transient"

    def test_rate_limit_value(self):
        assert ErrorCategory.RATE_LIMIT.value == "rate_limit"

    def test_network_value(self):
        assert ErrorCategory.NETWORK.value == "network"

    def test_authentication_value(self):
        assert ErrorCategory.AUTHENTICATION.value == "authentication"

    def test_all_categories_unique(self):
        values = [c.value for c in ErrorCategory]
        assert len(values) == len(set(values))


class TestRecoveryStrategy:
    """Tests for RecoveryStrategy enum values."""

    def test_retry_exponential(self):
        assert RecoveryStrategy.RETRY_EXPONENTIAL.value == "retry_exponential"

    def test_circuit_break(self):
        assert RecoveryStrategy.CIRCUIT_BREAK.value == "circuit_break"

    def test_fallback(self):
        assert RecoveryStrategy.FALLBACK.value == "fallback"

    def test_manual_intervention(self):
        assert RecoveryStrategy.MANUAL_INTERVENTION.value == "manual_intervention"


# ===========================================================================
# ErrorClassifier - Severity Tests
# ===========================================================================

class TestErrorClassifierSeverity:
    """Tests for ErrorClassifier._determine_severity()"""

    def setup_method(self):
        self.classifier = ErrorClassifier()

    def test_out_of_memory_is_critical(self):
        error = RuntimeError("out of memory on node 3")
        severity = self.classifier._determine_severity(error, {})
        assert severity == ErrorSeverity.CRITICAL

    def test_disk_full_is_critical(self):
        error = OSError("disk full, cannot write")
        severity = self.classifier._determine_severity(error, {})
        assert severity == ErrorSeverity.CRITICAL

    def test_database_connection_lost_is_critical(self):
        error = RuntimeError("database connection lost")
        severity = self.classifier._determine_severity(error, {})
        assert severity == ErrorSeverity.CRITICAL

    def test_security_breach_is_critical(self):
        error = RuntimeError("security breach detected")
        severity = self.classifier._determine_severity(error, {})
        assert severity == ErrorSeverity.CRITICAL

    def test_data_corruption_is_critical(self):
        error = RuntimeError("data corruption in table users")
        severity = self.classifier._determine_severity(error, {})
        assert severity == ErrorSeverity.CRITICAL

    def test_timeout_is_high(self):
        error = TimeoutError("timeout waiting for response")
        severity = self.classifier._determine_severity(error, {})
        assert severity == ErrorSeverity.HIGH

    def test_authentication_failed_is_high(self):
        error = RuntimeError("authentication failed for user")
        severity = self.classifier._determine_severity(error, {})
        assert severity == ErrorSeverity.HIGH

    def test_service_unavailable_is_high(self):
        error = RuntimeError("service unavailable")
        severity = self.classifier._determine_severity(error, {})
        assert severity == ErrorSeverity.HIGH

    def test_connection_refused_is_high(self):
        error = ConnectionRefusedError("connection refused on port 5432")
        severity = self.classifier._determine_severity(error, {})
        assert severity == ErrorSeverity.HIGH

    def test_authentication_exception_type_is_high(self):
        error = AuthenticationException("bad token")
        severity = self.classifier._determine_severity(error, {})
        assert severity == ErrorSeverity.HIGH

    def test_database_exception_type_is_high(self):
        error = DatabaseException("query failed")
        severity = self.classifier._determine_severity(error, {})
        assert severity == ErrorSeverity.HIGH

    def test_external_api_exception_is_medium(self):
        error = ExternalAPIException("alpha_vantage", "API error")
        severity = self.classifier._determine_severity(error, {})
        assert severity == ErrorSeverity.MEDIUM

    def test_rate_limit_exception_is_medium(self):
        error = RateLimitException("rate limit exceeded")
        severity = self.classifier._determine_severity(error, {})
        assert severity == ErrorSeverity.MEDIUM

    def test_validation_exception_is_low(self):
        error = ValidationException("invalid email format")
        severity = self.classifier._determine_severity(error, {})
        assert severity == ErrorSeverity.LOW

    def test_not_found_exception_is_low(self):
        error = NotFoundException("Stock", "ZZZZZ")
        severity = self.classifier._determine_severity(error, {})
        assert severity == ErrorSeverity.LOW

    def test_critical_path_context_escalates_to_high(self):
        error = RuntimeError("something generic went wrong")
        severity = self.classifier._determine_severity(error, {"critical_path": True})
        assert severity == ErrorSeverity.HIGH

    def test_generic_error_defaults_to_medium(self):
        error = RuntimeError("some generic error")
        severity = self.classifier._determine_severity(error, {})
        assert severity == ErrorSeverity.MEDIUM


# ===========================================================================
# ErrorClassifier - Category Tests
# ===========================================================================

class TestErrorClassifierCategory:
    """Tests for ErrorClassifier._determine_category()"""

    def setup_method(self):
        self.classifier = ErrorClassifier()

    def test_connection_error_is_network(self):
        error = ConnectionError("connection reset by peer")
        category = self.classifier._determine_category(error, {})
        assert category == ErrorCategory.NETWORK

    def test_timeout_is_network(self):
        error = TimeoutError("timeout connecting to host")
        category = self.classifier._determine_category(error, {})
        assert category == ErrorCategory.NETWORK

    def test_dns_error_is_network(self):
        error = RuntimeError("dns resolution failed")
        category = self.classifier._determine_category(error, {})
        assert category == ErrorCategory.NETWORK

    def test_unauthorized_is_authentication(self):
        error = RuntimeError("unauthorized access attempt")
        category = self.classifier._determine_category(error, {})
        assert category == ErrorCategory.AUTHENTICATION

    def test_token_expired_is_authentication(self):
        error = RuntimeError("token has expired")
        category = self.classifier._determine_category(error, {})
        assert category == ErrorCategory.AUTHENTICATION

    def test_rate_limit_keyword_is_rate_limit(self):
        error = RuntimeError("rate limit exceeded, retry after 60s")
        category = self.classifier._determine_category(error, {})
        assert category == ErrorCategory.RATE_LIMIT

    def test_too_many_requests_is_rate_limit(self):
        error = RuntimeError("too many requests from this IP")
        category = self.classifier._determine_category(error, {})
        assert category == ErrorCategory.RATE_LIMIT

    def test_validation_error_is_data_quality(self):
        error = RuntimeError("validation error on field email")
        category = self.classifier._determine_category(error, {})
        assert category == ErrorCategory.DATA_QUALITY

    def test_memory_error_is_resource(self):
        error = MemoryError("memory allocation failed")
        category = self.classifier._determine_category(error, {})
        assert category == ErrorCategory.RESOURCE

    def test_external_api_exception_type_is_dependency(self):
        error = ExternalAPIException("alpha_vantage", "server error")
        category = self.classifier._determine_category(error, {})
        assert category == ErrorCategory.DEPENDENCY

    def test_configuration_exception_type_is_configuration(self):
        error = ConfigurationException("missing API key")
        category = self.classifier._determine_category(error, {})
        assert category == ErrorCategory.CONFIGURATION

    def test_validation_exception_type_is_data_quality(self):
        error = ValidationException("invalid input")
        category = self.classifier._determine_category(error, {})
        assert category == ErrorCategory.DATA_QUALITY

    def test_rate_limit_exception_type_is_rate_limit(self):
        error = RateLimitException()
        category = self.classifier._determine_category(error, {})
        assert category == ErrorCategory.RATE_LIMIT

    def test_authentication_exception_type_is_authentication(self):
        error = AuthenticationException()
        category = self.classifier._determine_category(error, {})
        assert category == ErrorCategory.AUTHENTICATION

    def test_database_exception_type_is_dependency(self):
        error = DatabaseException("query execution failed")
        category = self.classifier._determine_category(error, {})
        assert category == ErrorCategory.DEPENDENCY

    def test_status_code_4xx_is_business_logic(self):
        error = APIException("bad request", status_code=400)
        category = self.classifier._determine_category(error, {})
        assert category == ErrorCategory.BUSINESS_LOGIC

    def test_status_code_5xx_is_dependency(self):
        error = APIException("internal error", status_code=500)
        category = self.classifier._determine_category(error, {})
        assert category == ErrorCategory.DEPENDENCY

    def test_generic_error_defaults_to_transient(self):
        error = RuntimeError("something unexpected happened")
        category = self.classifier._determine_category(error, {})
        assert category == ErrorCategory.TRANSIENT


# ===========================================================================
# ErrorClassifier - Message Normalization Tests
# ===========================================================================

class TestErrorClassifierNormalization:
    """Tests for ErrorClassifier._normalize_error_message()"""

    def setup_method(self):
        self.classifier = ErrorClassifier()

    def test_uuid_replaced(self):
        msg = "Error for user 550e8400-e29b-41d4-a716-446655440000"
        result = self.classifier._normalize_error_message(msg)
        assert "<UUID>" in result
        assert "550e8400" not in result

    def test_long_numbers_replaced(self):
        msg = "Record ID 1234567 not found"
        result = self.classifier._normalize_error_message(msg)
        assert "<ID>" in result

    def test_ip_address_replaced(self):
        msg = "Connection refused from 192.168.1.100"
        result = self.classifier._normalize_error_message(msg)
        assert "<IP>" in result
        assert "192.168.1.100" not in result

    def test_file_path_replaced(self):
        msg = "Error reading /var/log/app.log"
        result = self.classifier._normalize_error_message(msg)
        assert "<PATH>" in result

    def test_decimal_numbers_replaced(self):
        msg = "Threshold exceeded: 99.5 percent"
        result = self.classifier._normalize_error_message(msg)
        assert "<NUMBER>" in result

    def test_timestamp_replaced(self):
        msg = "Error at 2025-06-15T12:30:00Z in module"
        result = self.classifier._normalize_error_message(msg)
        assert "<TIMESTAMP>" in result


# ===========================================================================
# ErrorClassifier - Recovery Strategy Tests
# ===========================================================================

class TestErrorClassifierRecoveryStrategy:
    """Tests for ErrorClassifier._suggest_recovery_strategy()"""

    def setup_method(self):
        self.classifier = ErrorClassifier()

    def test_transient_gets_retry_exponential(self):
        error = RuntimeError("temporary glitch")
        strategy = self.classifier._suggest_recovery_strategy(
            error, ErrorCategory.TRANSIENT, ErrorSeverity.MEDIUM
        )
        assert strategy == RecoveryStrategy.RETRY_EXPONENTIAL

    def test_rate_limit_gets_circuit_break(self):
        error = RateLimitException()
        strategy = self.classifier._suggest_recovery_strategy(
            error, ErrorCategory.RATE_LIMIT, ErrorSeverity.MEDIUM
        )
        assert strategy == RecoveryStrategy.CIRCUIT_BREAK

    def test_network_gets_retry_exponential(self):
        error = ConnectionError("conn refused")
        strategy = self.classifier._suggest_recovery_strategy(
            error, ErrorCategory.NETWORK, ErrorSeverity.MEDIUM
        )
        assert strategy == RecoveryStrategy.RETRY_EXPONENTIAL

    def test_dependency_gets_fallback(self):
        error = ExternalAPIException("api", "down")
        strategy = self.classifier._suggest_recovery_strategy(
            error, ErrorCategory.DEPENDENCY, ErrorSeverity.MEDIUM
        )
        assert strategy == RecoveryStrategy.FALLBACK

    def test_resource_gets_auto_scale(self):
        error = MemoryError("oom")
        strategy = self.classifier._suggest_recovery_strategy(
            error, ErrorCategory.RESOURCE, ErrorSeverity.MEDIUM
        )
        assert strategy == RecoveryStrategy.AUTO_SCALE

    def test_authentication_gets_manual_intervention(self):
        error = AuthenticationException()
        strategy = self.classifier._suggest_recovery_strategy(
            error, ErrorCategory.AUTHENTICATION, ErrorSeverity.MEDIUM
        )
        assert strategy == RecoveryStrategy.MANUAL_INTERVENTION

    def test_data_quality_gets_graceful_degrade(self):
        error = ValidationException("bad data")
        strategy = self.classifier._suggest_recovery_strategy(
            error, ErrorCategory.DATA_QUALITY, ErrorSeverity.MEDIUM
        )
        assert strategy == RecoveryStrategy.GRACEFUL_DEGRADE

    def test_critical_severity_overrides_retry_to_manual(self):
        error = RuntimeError("critical failure")
        strategy = self.classifier._suggest_recovery_strategy(
            error, ErrorCategory.TRANSIENT, ErrorSeverity.CRITICAL
        )
        assert strategy == RecoveryStrategy.MANUAL_INTERVENTION

    def test_critical_severity_does_not_override_circuit_break(self):
        error = RateLimitException()
        strategy = self.classifier._suggest_recovery_strategy(
            error, ErrorCategory.RATE_LIMIT, ErrorSeverity.CRITICAL
        )
        assert strategy == RecoveryStrategy.CIRCUIT_BREAK

    def test_business_logic_gets_graceful_degrade(self):
        error = RuntimeError("invalid operation")
        strategy = self.classifier._suggest_recovery_strategy(
            error, ErrorCategory.BUSINESS_LOGIC, ErrorSeverity.LOW
        )
        assert strategy == RecoveryStrategy.GRACEFUL_DEGRADE


# ===========================================================================
# ErrorClassifier - Cost and Business Impact Tests
# ===========================================================================

class TestErrorClassifierImpact:
    """Tests for cost impact and business impact analysis."""

    def setup_method(self):
        self.classifier = ErrorClassifier()

    def test_cost_impact_external_api(self):
        error = ExternalAPIException("alpha_vantage", "error")
        cost = self.classifier._estimate_cost_impact(error, {"service": "api"})
        assert cost == pytest.approx(0.001 * 1.5)

    def test_cost_impact_database_exception(self):
        error = DatabaseException("query failed")
        cost = self.classifier._estimate_cost_impact(error, {})
        assert cost == pytest.approx(0.01)

    def test_cost_impact_recommendation_engine_multiplier(self):
        error = ExternalAPIException("api", "down")
        cost = self.classifier._estimate_cost_impact(
            error, {"service": "recommendation_engine"}
        )
        assert cost == pytest.approx(0.001 * 5.0)

    def test_cost_impact_generic_error(self):
        error = RuntimeError("generic")
        cost = self.classifier._estimate_cost_impact(error, {})
        assert cost == pytest.approx(0.001)

    def test_business_impact_critical(self):
        impact = self.classifier._assess_business_impact(
            RuntimeError("fail"), {}, ErrorSeverity.CRITICAL
        )
        assert "disruption" in impact.lower()

    def test_business_impact_high_recommendation(self):
        impact = self.classifier._assess_business_impact(
            RuntimeError("fail"),
            {"service": "recommendation_engine"},
            ErrorSeverity.HIGH,
        )
        assert "recommendation" in impact.lower()

    def test_business_impact_high_data_ingestion(self):
        impact = self.classifier._assess_business_impact(
            RuntimeError("fail"),
            {"service": "data_ingestion"},
            ErrorSeverity.HIGH,
        )
        assert "data freshness" in impact.lower()

    def test_business_impact_high_generic(self):
        impact = self.classifier._assess_business_impact(
            RuntimeError("fail"), {"service": "api"}, ErrorSeverity.HIGH
        )
        assert "reduced" in impact.lower()

    def test_business_impact_medium(self):
        impact = self.classifier._assess_business_impact(
            RuntimeError("fail"), {}, ErrorSeverity.MEDIUM
        )
        assert "degradation" in impact.lower()

    def test_business_impact_low(self):
        impact = self.classifier._assess_business_impact(
            RuntimeError("fail"), {}, ErrorSeverity.LOW
        )
        assert "minor" in impact.lower()


# ===========================================================================
# ErrorClassifier - Suggested Actions Tests
# ===========================================================================

class TestErrorClassifierSuggestedActions:
    """Tests for ErrorClassifier._generate_suggested_actions()"""

    def setup_method(self):
        self.classifier = ErrorClassifier()

    def test_critical_severity_includes_alert(self):
        actions = self.classifier._generate_suggested_actions(
            RuntimeError("bad"), ErrorCategory.TRANSIENT,
            ErrorSeverity.CRITICAL, RecoveryStrategy.MANUAL_INTERVENTION
        )
        assert any("alert" in a.lower() for a in actions)
        assert any("incident" in a.lower() for a in actions)

    def test_high_severity_includes_notify(self):
        actions = self.classifier._generate_suggested_actions(
            RuntimeError("bad"), ErrorCategory.TRANSIENT,
            ErrorSeverity.HIGH, RecoveryStrategy.RETRY_EXPONENTIAL
        )
        assert any("notify" in a.lower() for a in actions)

    def test_rate_limit_category_actions(self):
        actions = self.classifier._generate_suggested_actions(
            RuntimeError("throttled"), ErrorCategory.RATE_LIMIT,
            ErrorSeverity.MEDIUM, RecoveryStrategy.CIRCUIT_BREAK
        )
        assert any("backoff" in a.lower() for a in actions)
        assert any("circuit breaker" in a.lower() for a in actions)

    def test_network_category_actions(self):
        actions = self.classifier._generate_suggested_actions(
            RuntimeError("timeout"), ErrorCategory.NETWORK,
            ErrorSeverity.MEDIUM, None
        )
        assert any("connectivity" in a.lower() for a in actions)

    def test_authentication_category_actions(self):
        actions = self.classifier._generate_suggested_actions(
            RuntimeError("auth"), ErrorCategory.AUTHENTICATION,
            ErrorSeverity.MEDIUM, None
        )
        assert any("credentials" in a.lower() or "token" in a.lower() for a in actions)

    def test_fallback_recovery_adds_action(self):
        actions = self.classifier._generate_suggested_actions(
            RuntimeError("err"), ErrorCategory.DEPENDENCY,
            ErrorSeverity.MEDIUM, RecoveryStrategy.FALLBACK
        )
        assert any("fallback" in a.lower() for a in actions)


# ===========================================================================
# ErrorClassifier - Pattern Detection Tests
# ===========================================================================

class TestErrorClassifierPatternDetection:
    """Tests for ErrorClassifier._detect_pattern()"""

    def setup_method(self):
        self.classifier = ErrorClassifier()

    def test_single_occurrence_returns_none(self):
        sig = ErrorSignature(
            signature_hash="abc",
            error_type="RuntimeError",
            normalized_message="test",
            service="api",
            operation="get",
            first_seen=datetime.now(),
            last_seen=datetime.now(),
            occurrence_count=1,
            avg_frequency_per_hour=1.0,
        )
        pattern = self.classifier._detect_pattern(sig)
        assert pattern is None

    def test_spike_detection(self):
        now = datetime.now()
        sig = ErrorSignature(
            signature_hash="abc",
            error_type="RuntimeError",
            normalized_message="test",
            service="api",
            operation="get",
            first_seen=now - timedelta(minutes=5),
            last_seen=now,
            occurrence_count=20,
            avg_frequency_per_hour=15.0,
        )
        pattern = self.classifier._detect_pattern(sig)
        assert pattern == ErrorPattern.SPIKE

    def test_sustained_detection(self):
        now = datetime.now()
        sig = ErrorSignature(
            signature_hash="abc",
            error_type="RuntimeError",
            normalized_message="test",
            service="api",
            operation="get",
            first_seen=now - timedelta(hours=5),
            last_seen=now,
            occurrence_count=15,
            avg_frequency_per_hour=3.0,
        )
        pattern = self.classifier._detect_pattern(sig)
        assert pattern == ErrorPattern.SUSTAINED

    def test_intermittent_detection(self):
        now = datetime.now()
        sig = ErrorSignature(
            signature_hash="abc",
            error_type="RuntimeError",
            normalized_message="test",
            service="api",
            operation="get",
            first_seen=now - timedelta(hours=5),
            last_seen=now,
            occurrence_count=3,
            avg_frequency_per_hour=0.5,
        )
        pattern = self.classifier._detect_pattern(sig)
        assert pattern == ErrorPattern.INTERMITTENT


# ===========================================================================
# ErrorClassifier - Full Classification Tests
# ===========================================================================

class TestErrorClassifierClassifyError:
    """Tests for the full classify_error() pipeline."""

    def setup_method(self):
        self.classifier = ErrorClassifier()

    def test_classify_returns_error_context(self):
        error = RuntimeError("connection refused")
        ctx = self.classifier.classify_error(error, {"service": "api", "operation": "fetch"})
        assert isinstance(ctx, ErrorContext)
        assert ctx.error_type == "RuntimeError"
        assert ctx.error_message == "connection refused"
        assert ctx.service == "api"
        assert ctx.operation == "fetch"

    def test_classify_generates_error_id(self):
        error = RuntimeError("some error")
        ctx = self.classifier.classify_error(error)
        assert ctx.error_id is not None
        assert len(ctx.error_id) > 0

    def test_classify_uses_provided_correlation_id(self):
        error = RuntimeError("err")
        ctx = self.classifier.classify_error(
            error, {"correlation_id": "my-corr-123"}
        )
        assert ctx.correlation_id == "my-corr-123"

    def test_classify_updates_learned_patterns(self):
        error = RuntimeError("test error")
        self.classifier.classify_error(error, {"service": "api", "operation": "get"})
        assert len(self.classifier.learned_patterns) == 1

    def test_classify_same_error_increments_occurrence(self):
        error = RuntimeError("test error")
        self.classifier.classify_error(error, {"service": "api", "operation": "get"})
        self.classifier.classify_error(error, {"service": "api", "operation": "get"})
        # Both create same signature hash
        sigs = list(self.classifier.learned_patterns.values())
        assert sigs[0].occurrence_count == 2

    def test_classify_includes_recovery_strategy(self):
        error = ConnectionError("timeout connecting")
        ctx = self.classifier.classify_error(error)
        assert ctx.recovery_strategy is not None

    def test_classify_includes_suggested_actions(self):
        error = RuntimeError("out of memory critical failure")
        ctx = self.classifier.classify_error(error)
        assert len(ctx.suggested_actions) > 0


# ===========================================================================
# ErrorCorrelationEngine Tests
# ===========================================================================

class TestErrorCorrelationEngine:
    """Tests for ErrorCorrelationEngine."""

    def _make_context(self, **overrides):
        """Helper to create an ErrorContext with defaults."""
        defaults = {
            "error_id": "err-1",
            "correlation_id": "corr-1",
            "timestamp": datetime.now(),
            "severity": ErrorSeverity.MEDIUM,
            "category": ErrorCategory.TRANSIENT,
            "pattern": None,
            "service": "api",
            "operation": "get",
            "user_id": None,
            "request_id": None,
            "error_type": "RuntimeError",
            "error_message": "test error",
            "stack_trace": "",
            "environment": {},
            "metadata": {},
            "suggested_actions": [],
            "recovery_strategy": None,
            "cost_impact": 0.001,
            "business_impact": "minor",
        }
        defaults.update(overrides)
        return ErrorContext(**defaults)

    def test_add_error_context_stores_in_timeline(self):
        engine = ErrorCorrelationEngine()
        ctx = self._make_context()
        engine.add_error_context(ctx)
        assert len(engine.error_timeline) == 1

    def test_cleanup_old_errors(self):
        engine = ErrorCorrelationEngine(time_window_minutes=5)
        old_ctx = self._make_context(
            error_id="old",
            timestamp=datetime.now() - timedelta(minutes=10),
        )
        engine.error_timeline.append(old_ctx)
        engine._cleanup_old_errors()
        assert len(engine.error_timeline) == 0

    def test_correlation_score_same_service_and_category(self):
        engine = ErrorCorrelationEngine()
        now = datetime.now()
        ctx1 = self._make_context(
            error_id="e1", correlation_id="c1", timestamp=now,
            service="api", category=ErrorCategory.NETWORK
        )
        ctx2 = self._make_context(
            error_id="e2", correlation_id="c2", timestamp=now,
            service="api", category=ErrorCategory.NETWORK
        )
        score = engine._calculate_correlation_score(ctx1, ctx2)
        # Same time (<60s): 0.4, same service: 0.3, same category: 0.2 = 0.9
        assert score == pytest.approx(0.9)

    def test_correlation_score_different_everything(self):
        engine = ErrorCorrelationEngine()
        now = datetime.now()
        ctx1 = self._make_context(
            error_id="e1", correlation_id="c1",
            timestamp=now, service="api",
            category=ErrorCategory.NETWORK
        )
        ctx2 = self._make_context(
            error_id="e2", correlation_id="c2",
            timestamp=now - timedelta(minutes=20), service="db",
            category=ErrorCategory.AUTHENTICATION
        )
        score = engine._calculate_correlation_score(ctx1, ctx2)
        assert score == 0.0

    def test_correlation_score_same_correlation_id(self):
        engine = ErrorCorrelationEngine()
        now = datetime.now()
        ctx1 = self._make_context(
            error_id="e1", correlation_id="shared",
            timestamp=now, service="svc1",
            category=ErrorCategory.NETWORK
        )
        ctx2 = self._make_context(
            error_id="e2", correlation_id="shared",
            timestamp=now, service="svc2",
            category=ErrorCategory.DEPENDENCY
        )
        score = engine._calculate_correlation_score(ctx1, ctx2)
        # Same time: 0.4, different service: 0, different cat: 0, same corr_id: 0.5 = 0.9
        assert score == pytest.approx(0.9)

    def test_correlation_score_capped_at_1(self):
        engine = ErrorCorrelationEngine()
        now = datetime.now()
        ctx1 = self._make_context(
            error_id="e1", correlation_id="shared",
            timestamp=now, service="api",
            category=ErrorCategory.NETWORK, user_id="user-1"
        )
        ctx2 = self._make_context(
            error_id="e2", correlation_id="shared",
            timestamp=now, service="api",
            category=ErrorCategory.NETWORK, user_id="user-1"
        )
        score = engine._calculate_correlation_score(ctx1, ctx2)
        assert score <= 1.0

    def test_root_cause_analysis_error_not_found(self):
        engine = ErrorCorrelationEngine()
        result = engine.get_root_cause_analysis("nonexistent")
        assert "error" in result

    def test_root_cause_analysis_with_correlated_errors(self):
        engine = ErrorCorrelationEngine(correlation_threshold=0.3)
        now = datetime.now()
        ctx1 = self._make_context(
            error_id="e1", timestamp=now, service="api",
            category=ErrorCategory.NETWORK
        )
        ctx2 = self._make_context(
            error_id="e2", timestamp=now, service="api",
            category=ErrorCategory.NETWORK
        )
        engine.error_timeline.append(ctx1)
        engine.error_timeline.append(ctx2)

        result = engine.get_root_cause_analysis("e1")
        assert result["target_error_id"] == "e1"
        assert result["correlation_count"] >= 1


# ===========================================================================
# ErrorHandlingManager Tests
# ===========================================================================

class TestErrorHandlingManager:
    """Tests for ErrorHandlingManager."""

    def test_initial_state(self):
        manager = ErrorHandlingManager()
        assert len(manager.error_history) == 0
        assert len(manager.active_incidents) == 0

    @pytest.mark.asyncio
    async def test_handle_error_classifies_and_stores(self):
        manager = ErrorHandlingManager()
        error = RuntimeError("test error")
        with patch.object(manager, "_handle_incident_management", new_callable=AsyncMock):
            with patch.object(manager, "_execute_recovery_strategy", new_callable=AsyncMock):
                ctx = await manager.handle_error(
                    error, {"service": "api", "operation": "test"},
                    should_raise=False
                )
        assert isinstance(ctx, ErrorContext)
        assert len(manager.error_history) == 1

    @pytest.mark.asyncio
    async def test_handle_error_reraises_when_should_raise_true(self):
        manager = ErrorHandlingManager()
        error = RuntimeError("fatal error")
        with patch.object(manager, "_handle_incident_management", new_callable=AsyncMock):
            with patch.object(manager, "_execute_recovery_strategy", new_callable=AsyncMock):
                with pytest.raises(RuntimeError, match="fatal error"):
                    await manager.handle_error(
                        error, {"service": "api"}, should_raise=True
                    )

    @pytest.mark.asyncio
    async def test_incident_created_for_high_severity(self):
        manager = ErrorHandlingManager()
        error = RuntimeError("authentication failed for service")
        ctx = manager.classifier.classify_error(
            error, {"service": "auth", "operation": "login"}
        )
        assert ctx.severity >= ErrorSeverity.HIGH
        await manager._handle_incident_management(ctx)
        assert len(manager.active_incidents) == 1

    @pytest.mark.asyncio
    async def test_incident_escalated_on_repeated_errors(self):
        manager = ErrorHandlingManager()
        for i in range(12):
            error = RuntimeError("authentication failed repeatedly")
            ctx = manager.classifier.classify_error(
                error, {"service": "auth", "operation": "login"}
            )
            await manager._handle_incident_management(ctx)

        incident_key = "auth_authentication"
        assert incident_key in manager.active_incidents
        assert manager.active_incidents[incident_key]["error_count"] == 12

    def test_error_analytics_empty(self):
        manager = ErrorHandlingManager()
        result = manager.get_error_analytics()
        assert result["message"] == "No errors in specified time window"

    def test_error_analytics_with_data(self):
        manager = ErrorHandlingManager()
        classifier = ErrorClassifier()

        # Add some errors to history
        for msg in ["timeout connecting", "validation error on field", "out of memory"]:
            error = RuntimeError(msg)
            ctx = classifier.classify_error(error, {"service": "api", "operation": "get"})
            manager.error_history.append(ctx)

        result = manager.get_error_analytics(time_window_hours=1)
        assert result["total_errors"] == 3
        assert "severity_breakdown" in result
        assert "category_breakdown" in result
        assert "estimated_cost_impact" in result


# ===========================================================================
# validate_stock_symbol Tests
# ===========================================================================

class TestValidateStockSymbol:
    """Tests for validate_stock_symbol utility."""

    def test_valid_symbols(self):
        assert validate_stock_symbol("AAPL") is True
        assert validate_stock_symbol("MSFT") is True
        assert validate_stock_symbol("A") is True
        assert validate_stock_symbol("GOOGL") is True

    def test_lowercase_is_valid_after_upper(self):
        assert validate_stock_symbol("aapl") is True

    def test_empty_string_is_invalid(self):
        assert validate_stock_symbol("") is False

    def test_none_is_invalid(self):
        assert validate_stock_symbol(None) is False

    def test_non_string_is_invalid(self):
        assert validate_stock_symbol(123) is False

    def test_too_long_is_invalid(self):
        assert validate_stock_symbol("ABCDEF") is False

    def test_digits_only_is_invalid(self):
        assert validate_stock_symbol("12345") is False

    def test_mixed_alpha_numeric_is_invalid(self):
        assert validate_stock_symbol("AAP1") is False

    def test_special_chars_invalid(self):
        assert validate_stock_symbol("AA.L") is False
        assert validate_stock_symbol("A-B") is False

    def test_whitespace_trimmed(self):
        assert validate_stock_symbol("  AAPL  ") is True


# ===========================================================================
# with_error_handling Decorator Tests
# ===========================================================================

class TestWithErrorHandlingDecorator:
    """Tests for with_error_handling() decorator."""

    @pytest.mark.asyncio
    async def test_async_function_success_passthrough(self):
        @with_error_handling(service="test", operation="add", should_raise=False)
        async def add(a, b):
            return a + b

        result = await add(1, 2)
        assert result == 3

    @pytest.mark.asyncio
    async def test_async_function_error_raised_when_should_raise(self):
        @with_error_handling(service="test", operation="fail", should_raise=True)
        async def failing_func():
            raise ValueError("broken")

        with pytest.raises(ValueError, match="broken"):
            await failing_func()

    @pytest.mark.asyncio
    async def test_async_function_error_swallowed_when_not_should_raise(self):
        @with_error_handling(service="test", operation="fail", should_raise=False)
        async def failing_func():
            raise ValueError("broken")

        # Should not raise; returns None
        result = await failing_func()
        assert result is None

    def test_sync_function_decorated_correctly(self):
        @with_error_handling(service="test", should_raise=False)
        def sync_add(a, b):
            return a + b

        # The decorator should detect it's a sync function
        assert not asyncio.iscoroutinefunction(sync_add)

    @pytest.mark.asyncio
    async def test_decorator_preserves_function_name(self):
        @with_error_handling(service="test")
        async def my_special_function():
            pass

        assert my_special_function.__name__ == "my_special_function"

    @pytest.mark.asyncio
    async def test_decorator_default_service_from_module(self):
        """Decorator uses module name when service not specified."""
        @with_error_handling(should_raise=False)
        async def some_func():
            raise RuntimeError("err")

        # Should not crash - service derived from module
        result = await some_func()
        assert result is None


# ===========================================================================
# ErrorClassifier - Error Signature Tests
# ===========================================================================

class TestErrorSignatureCreation:
    """Tests for error signature creation and tracking."""

    def setup_method(self):
        self.classifier = ErrorClassifier()

    def test_signature_hash_is_deterministic(self):
        error = RuntimeError("connection timeout")
        ctx = {"service": "api", "operation": "fetch"}
        sig1 = self.classifier._create_error_signature(error, ctx)
        # Reset learned patterns to get fresh signature
        self.classifier.learned_patterns.clear()
        sig2 = self.classifier._create_error_signature(error, ctx)
        assert sig1.signature_hash == sig2.signature_hash

    def test_different_errors_different_signatures(self):
        err1 = RuntimeError("error A")
        err2 = ValueError("error B")
        ctx = {"service": "api", "operation": "get"}
        sig1 = self.classifier._create_error_signature(err1, ctx)
        sig2 = self.classifier._create_error_signature(err2, ctx)
        assert sig1.signature_hash != sig2.signature_hash

    def test_existing_signature_updates_count(self):
        error = RuntimeError("repeated error")
        ctx = {"service": "api", "operation": "get"}
        sig1 = self.classifier._create_error_signature(error, ctx)
        self.classifier._update_learned_patterns(sig1)

        sig2 = self.classifier._create_error_signature(error, ctx)
        assert sig2.occurrence_count == 2

    def test_signature_frequency_updates(self):
        error = RuntimeError("repeated error")
        ctx = {"service": "api", "operation": "get"}
        sig = self.classifier._create_error_signature(error, ctx)
        self.classifier._update_learned_patterns(sig)
        assert sig.avg_frequency_per_hour >= 0
