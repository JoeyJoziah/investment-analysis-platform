"""
Unit tests for monitoring layer modules with zero prior test coverage.

Tests cover:
- health_checks.py: HealthStatus, ServiceType, HealthCheckResult, SLATarget,
  ServiceHealth, HealthChecker, SLAMonitor, HealthMonitoringSystem
- metrics_collector.py: MetricsCollector record_* methods, metric calculations
- sla_tracker.py: SLATracker measurement recording, status calculation,
  violation detection, severity classification, credit calculation
"""

import asyncio
import importlib
import sys
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock, PropertyMock
from dataclasses import dataclass

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Stub heavy dependencies before importing monitoring modules.
# These modules import from backend.config, backend.utils, prometheus_client etc.
# We stub at sys.modules level to avoid import-time failures.
# ---------------------------------------------------------------------------

# Stub backend.config.settings
_mock_settings = MagicMock()
_mock_settings.VERSION = "1.0.0"
_mock_settings.ENVIRONMENT = "test"
_mock_settings.DEBUG = True
sys.modules.setdefault("backend.config.settings", MagicMock(settings=_mock_settings))

# Stub backend.config.monitoring_config
_mock_mon_config = MagicMock()
_mock_mon_config.enable_monitoring = True
_mock_mon_config.metrics_collection_interval = 10
_mock_mon_config.health_check_interval = 30
_mock_mon_config.enable_cost_monitoring = False
_mock_mon_config.enable_compliance_monitoring = False
_mock_mon_config.monthly_budget = 100.0
_mock_mon_config.emergency_mode_threshold = 0.9
_mock_mon_config.logging = MagicMock(service_name="test-service")
sys.modules.setdefault("backend.config.monitoring_config", MagicMock(
    monitoring_config=_mock_mon_config,
    initialize_monitoring=MagicMock(return_value=_mock_mon_config),
))

# Stub backend.utils.structured_logging
_mock_structured_logging = MagicMock()
_mock_structured_logging.get_structured_logger = MagicMock(return_value=MagicMock())
sys.modules.setdefault("backend.utils.structured_logging", _mock_structured_logging)

# Stub asyncpg (used by health_checks.py)
sys.modules.setdefault("asyncpg", MagicMock())

# Stub aiohttp
sys.modules.setdefault("aiohttp", MagicMock())

# Stub backend.utils.cache (used by sla_tracker)
_mock_cache = MagicMock()
_mock_cache.get_redis = AsyncMock(return_value=MagicMock(
    set=AsyncMock(),
    get=AsyncMock(return_value=None),
))
sys.modules.setdefault("backend.utils.cache", _mock_cache)

# Stub backend.utils.enhanced_cost_monitor (used by sla_tracker)
from enum import Enum as _Enum

class _StockPriority(_Enum):
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    MINIMAL = 5

_mock_cost_monitor = MagicMock()
_mock_cost_monitor.StockPriority = _StockPriority
sys.modules.setdefault("backend.utils.enhanced_cost_monitor", _mock_cost_monitor)

# ---------------------------------------------------------------------------
# Now import the actual monitoring modules via importlib
# ---------------------------------------------------------------------------

_mon_dir = Path(__file__).resolve().parents[2] / "monitoring"

# --- health_checks.py ---
_hc_spec = importlib.util.spec_from_file_location("health_checks", _mon_dir / "health_checks.py")
_hc = importlib.util.module_from_spec(_hc_spec)
_hc_spec.loader.exec_module(_hc)

HealthStatus = _hc.HealthStatus
ServiceType = _hc.ServiceType
HealthCheckResult = _hc.HealthCheckResult
SLATarget_HC = _hc.SLATarget  # Health-checks SLATarget (different from sla_tracker's)
ServiceHealth = _hc.ServiceHealth
HealthChecker = _hc.HealthChecker
SLAMonitor_HC = _hc.SLAMonitor
HealthMonitoringSystem = _hc.HealthMonitoringSystem

# --- sla_tracker.py ---
_st_spec = importlib.util.spec_from_file_location("sla_tracker", _mon_dir / "sla_tracker.py")
_st = importlib.util.module_from_spec(_st_spec)
_st_spec.loader.exec_module(_st)

SLAMetric = _st.SLAMetric
SLATarget_ST = _st.SLATarget
SLAMeasurement = _st.SLAMeasurement
SLATracker = _st.SLATracker

# --- metrics_collector.py ---
# This module imports from prometheus_client and FastAPI.
# prometheus_client is installed, but we need to handle backend.config.settings
# which we already stubbed above.
_mc_spec = importlib.util.spec_from_file_location("metrics_collector", _mon_dir / "metrics_collector.py")
_mc = importlib.util.module_from_spec(_mc_spec)
_mc_spec.loader.exec_module(_mc)

MetricsCollector = _mc.MetricsCollector


# ==========================================================================
# health_checks.py
# ==========================================================================


class TestHealthStatus:
    def test_enum_values(self):
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"
        assert HealthStatus.CRITICAL.value == "critical"

    def test_all_members(self):
        assert len(HealthStatus) == 4


class TestServiceType:
    def test_enum_values(self):
        assert ServiceType.DATABASE.value == "database"
        assert ServiceType.CACHE.value == "cache"
        assert ServiceType.API.value == "api"

    def test_all_members(self):
        assert len(ServiceType) == 7


class TestHealthCheckResult:
    def test_construction(self):
        result = HealthCheckResult(
            service="database",
            check_type="connectivity",
            status=HealthStatus.HEALTHY,
            message="OK",
            duration=0.05,
            timestamp=datetime.now(),
        )
        assert result.service == "database"
        assert result.status == HealthStatus.HEALTHY

    def test_to_dict(self):
        now = datetime.now()
        result = HealthCheckResult(
            service="redis",
            check_type="ping",
            status=HealthStatus.DEGRADED,
            message="Slow",
            duration=1.5,
            timestamp=now,
            metadata={"latency": 1.5},
            error_details="timeout",
        )
        d = result.to_dict()
        assert d["service"] == "redis"
        assert d["status"] == "degraded"
        assert d["duration"] == 1.5
        assert d["metadata"]["latency"] == 1.5
        assert d["error_details"] == "timeout"
        assert d["timestamp"] == now.isoformat()

    def test_default_metadata(self):
        result = HealthCheckResult(
            service="api", check_type="test", status=HealthStatus.HEALTHY,
            message="OK", duration=0.01, timestamp=datetime.now(),
        )
        assert result.metadata == {}
        assert result.error_details is None


class TestSLATargetHC:
    def test_construction(self):
        target = SLATarget_HC(
            name="response_time",
            service="api",
            target_type="response_time",
            threshold=2.0,
            time_window="24h",
        )
        assert target.threshold == 2.0
        assert target.measurement_interval == 60  # default


class TestServiceHealth:
    def test_default_values(self):
        sh = ServiceHealth(
            service_name="database",
            service_type=ServiceType.DATABASE,
        )
        assert sh.overall_status == HealthStatus.HEALTHY
        assert sh.consecutive_failures == 0
        assert sh.uptime_percentage == 100.0
        assert sh.avg_response_time == 0.0
        assert sh.error_rate == 0.0
        assert sh.dependencies == []


class TestHealthChecker:
    async def test_execute_success_bool(self):
        async def check():
            return True

        checker = HealthChecker(
            name="test",
            service="test_service",
            check_func=check,
            service_type=ServiceType.API,
            timeout=5,
            retries=0,
        )
        result = await checker.execute()
        assert result.status == HealthStatus.HEALTHY
        assert "passed" in result.message

    async def test_execute_failure_bool(self):
        async def check():
            return False

        checker = HealthChecker(
            name="test",
            service="test_service",
            check_func=check,
            service_type=ServiceType.API,
            timeout=5,
            retries=0,
        )
        result = await checker.execute()
        assert result.status == HealthStatus.UNHEALTHY

    async def test_execute_dict_result(self):
        async def check():
            return {
                "status": "degraded",
                "message": "High latency",
                "metadata": {"latency_ms": 500},
            }

        checker = HealthChecker(
            name="latency",
            service="api",
            check_func=check,
            service_type=ServiceType.API,
            timeout=5,
            retries=0,
        )
        result = await checker.execute()
        assert result.status == HealthStatus.DEGRADED
        assert result.message == "High latency"
        assert result.metadata["latency_ms"] == 500

    async def test_execute_timeout(self):
        async def slow_check():
            await asyncio.sleep(10)
            return True

        checker = HealthChecker(
            name="timeout_test",
            service="slow_service",
            check_func=slow_check,
            service_type=ServiceType.API,
            timeout=0.01,
            retries=0,
            critical=True,
        )
        result = await checker.execute()
        assert result.status == HealthStatus.CRITICAL

    async def test_execute_error_with_retries(self):
        call_count = 0

        async def failing_check():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("DB down")

        checker = HealthChecker(
            name="retry_test",
            service="database",
            check_func=failing_check,
            service_type=ServiceType.DATABASE,
            timeout=5,
            retries=2,
            critical=False,
        )
        result = await checker.execute()
        assert result.status == HealthStatus.UNHEALTHY
        assert call_count == 3  # initial + 2 retries


class TestSLAMonitorHC:
    def test_add_sla_target(self):
        monitor = SLAMonitor_HC()
        target = SLATarget_HC(
            name="availability",
            service="database",
            target_type="availability",
            threshold=99.9,
            time_window="30d",
        )
        monitor.add_sla_target(target)
        assert "database:availability" in monitor.sla_targets

    def test_record_metric(self):
        monitor = SLAMonitor_HC()
        monitor.record_metric("api", "response_time", 0.5)
        monitor.record_metric("api", "response_time", 0.3)
        assert len(monitor.metrics_history["api:response_time"]) == 2

    def test_get_sla_summary_empty(self):
        monitor = SLAMonitor_HC()
        summary = monitor.get_sla_summary()
        assert summary["sla_targets"] == 0
        assert summary["compliance_summary"] == {}

    def test_get_sla_summary_with_cache(self):
        monitor = SLAMonitor_HC()
        monitor.compliance_cache["api:response_time"] = {
            "compliance": 98.5,
            "target": 99.0,
            "timestamp": datetime.now(),
        }
        summary = monitor.get_sla_summary()
        assert "api:response_time" in summary["compliance_summary"]
        assert summary["compliance_summary"]["api:response_time"]["compliance_percent"] == 98.5


class TestHealthMonitoringSystem:
    def test_init_creates_default_health_checks(self):
        hms = HealthMonitoringSystem()
        # Should have default health checks registered
        assert len(hms.health_checkers) > 0
        # Should include database and redis checks
        assert "database:connectivity" in hms.health_checkers
        assert "redis:connectivity" in hms.health_checkers

    def test_init_creates_sla_targets(self):
        hms = HealthMonitoringSystem()
        assert len(hms.sla_monitor.sla_targets) > 0

    def test_add_health_check(self):
        hms = HealthMonitoringSystem()
        initial_count = len(hms.health_checkers)

        async def custom_check():
            return True

        hms.add_health_check(
            name="custom",
            service="my_service",
            check_func=custom_check,
            service_type=ServiceType.BUSINESS_LOGIC,
        )
        assert len(hms.health_checkers) == initial_count + 1
        assert "my_service:custom" in hms.health_checkers

    def test_update_service_health_healthy(self):
        hms = HealthMonitoringSystem()
        # Manually set up a service
        hms.service_health["test_svc"] = ServiceHealth(
            service_name="test_svc",
            service_type=ServiceType.API,
            consecutive_failures=2,
        )
        result = HealthCheckResult(
            service="test_svc",
            check_type="test",
            status=HealthStatus.HEALTHY,
            message="OK",
            duration=0.01,
            timestamp=datetime.now(),
        )
        hms._update_service_health(result)
        assert hms.service_health["test_svc"].consecutive_failures == 0
        assert hms.service_health["test_svc"].overall_status == HealthStatus.HEALTHY

    def test_update_service_health_failures(self):
        hms = HealthMonitoringSystem()
        hms.service_health["test_svc"] = ServiceHealth(
            service_name="test_svc",
            service_type=ServiceType.API,
        )
        # Simulate 3 consecutive failures
        for i in range(3):
            result = HealthCheckResult(
                service="test_svc",
                check_type="test",
                status=HealthStatus.UNHEALTHY,
                message="Failed",
                duration=0.5,
                timestamp=datetime.now(),
            )
            hms._update_service_health(result)

        assert hms.service_health["test_svc"].consecutive_failures == 3
        assert hms.service_health["test_svc"].overall_status == HealthStatus.CRITICAL

    def test_update_service_health_degraded(self):
        hms = HealthMonitoringSystem()
        hms.service_health["test_svc"] = ServiceHealth(
            service_name="test_svc",
            service_type=ServiceType.API,
        )
        result = HealthCheckResult(
            service="test_svc",
            check_type="test",
            status=HealthStatus.DEGRADED,
            message="Slow",
            duration=2.0,
            timestamp=datetime.now(),
        )
        hms._update_service_health(result)
        assert hms.service_health["test_svc"].consecutive_failures == 1
        assert hms.service_health["test_svc"].overall_status == HealthStatus.DEGRADED

    async def test_get_health_status(self):
        hms = HealthMonitoringSystem()
        status = await hms.get_health_status()
        assert "overall_status" in status
        assert "services" in status
        assert "timestamp" in status

    async def test_get_detailed_health_unknown_service(self):
        hms = HealthMonitoringSystem()
        result = await hms.get_detailed_health("nonexistent")
        assert "error" in result

    async def test_get_detailed_health_known_service(self):
        hms = HealthMonitoringSystem()
        hms.service_health["database"] = ServiceHealth(
            service_name="database",
            service_type=ServiceType.DATABASE,
        )
        result = await hms.get_detailed_health("database")
        assert result["service"] == "database"
        assert "metrics" in result


# ==========================================================================
# metrics_collector.py
# ==========================================================================


class TestMetricsCollector:
    @pytest.fixture
    def collector(self):
        return MetricsCollector()

    def test_record_api_request(self, collector):
        # Should not raise
        collector.record_api_request("GET", "/api/stocks", 200, 0.05)

    def test_record_api_request_error(self, collector):
        collector.record_api_request("POST", "/api/orders", 500, 1.5)

    def test_record_api_request_client_error(self, collector):
        collector.record_api_request("GET", "/api/invalid", 404, 0.02)

    def test_record_stock_processing_success(self, collector):
        collector.record_stock_processing("tier1", "price_data", 0.5, success=True)

    def test_record_stock_processing_failure(self, collector):
        collector.record_stock_processing("tier2", "sentiment", 2.0, success=False)

    def test_record_model_prediction(self, collector):
        collector.record_model_prediction("lstm", "v2.1", 0.15)

    def test_record_financial_performance(self, collector):
        collector.record_financial_performance(
            strategy="momentum",
            alpha=2.5,
            sharpe=1.8,
            max_dd=-10.0,
            success_rate=65.0,
            tier="tier1",
        )

    def test_record_recommendation_generated(self, collector):
        collector.record_recommendation_generated("buy")

    def test_record_data_quality(self, collector):
        collector.record_data_quality("price_data", 95.0)

    def test_record_alert_sent(self, collector):
        collector.record_alert_sent("slack", "warning")

    def test_record_sla_compliance(self, collector):
        collector.record_sla_compliance("api", 99.5)

    def test_record_incident(self, collector):
        collector.record_incident("high", 300.0)

    def test_update_mttr_mtbf(self, collector):
        collector.update_mttr_mtbf("database", mttr_minutes=15.0, mtbf_hours=720.0)

    def test_get_metrics_returns_bytes(self, collector):
        result = collector.get_metrics()
        assert isinstance(result, bytes)

    def test_start_time_set(self, collector):
        assert collector.start_time > 0
        assert collector._collection_interval == 10


# ==========================================================================
# sla_tracker.py
# ==========================================================================


class TestSLAMetric:
    def test_enum_values(self):
        assert SLAMetric.DATA_FRESHNESS.value == "data_freshness"
        assert SLAMetric.UPDATE_FREQUENCY.value == "update_frequency"
        assert SLAMetric.API_LATENCY.value == "api_latency"
        assert SLAMetric.AVAILABILITY.value == "availability"
        assert SLAMetric.ERROR_RATE.value == "error_rate"

    def test_all_members(self):
        assert len(SLAMetric) == 7


class TestSLATargetST:
    def test_construction(self):
        target = SLATarget_ST(
            metric=SLAMetric.DATA_FRESHNESS,
            target_value=5,
            unit="minutes",
            measurement_window=timedelta(hours=1),
            critical_threshold=15,
            warning_threshold=10,
        )
        assert target.target_value == 5
        assert target.critical_threshold == 15


class TestSLAMeasurement:
    def test_default_values(self):
        m = SLAMeasurement()
        assert m.value == 0.0
        assert m.meets_sla is True
        assert m.warning is False
        assert m.metadata == {}
        assert m.ticker is None

    def test_custom_values(self):
        m = SLAMeasurement(
            metric=SLAMetric.API_LATENCY,
            tier=_StockPriority.HIGH,
            ticker="AAPL",
            value=250.0,
            unit="milliseconds",
            meets_sla=True,
        )
        assert m.ticker == "AAPL"
        assert m.value == 250.0


class TestSLATracker:
    @pytest.fixture
    def tracker(self):
        t = SLATracker()
        # Provide a mock Redis so methods don't fail
        t.redis = MagicMock()
        t.redis.set = AsyncMock()
        t.redis.get = AsyncMock(return_value=None)
        return t

    def test_define_sla_targets(self, tracker):
        targets = tracker.sla_targets
        # Should have targets for all tiers
        assert _StockPriority.CRITICAL in targets
        assert _StockPriority.HIGH in targets
        assert _StockPriority.MEDIUM in targets
        assert _StockPriority.LOW in targets
        assert _StockPriority.MINIMAL in targets

    def test_critical_tier_has_most_metrics(self, tracker):
        critical = tracker.sla_targets[_StockPriority.CRITICAL]
        minimal = tracker.sla_targets[_StockPriority.MINIMAL]
        assert len(critical) > len(minimal)

    def test_critical_freshness_target(self, tracker):
        target = tracker.sla_targets[_StockPriority.CRITICAL][SLAMetric.DATA_FRESHNESS]
        assert target.target_value == 5
        assert target.unit == "minutes"
        assert target.critical_threshold == 15

    async def test_record_measurement_meets_sla(self, tracker):
        m = await tracker.record_measurement(
            metric=SLAMetric.DATA_FRESHNESS,
            tier=_StockPriority.CRITICAL,
            value=3.0,  # 3 minutes, well within 15-minute threshold
            ticker="AAPL",
        )
        assert m.meets_sla is True
        assert m.warning is False

    async def test_record_measurement_warning(self, tracker):
        m = await tracker.record_measurement(
            metric=SLAMetric.DATA_FRESHNESS,
            tier=_StockPriority.CRITICAL,
            value=12.0,  # Between warning (10) and critical (15)
            ticker="MSFT",
        )
        assert m.meets_sla is True
        assert m.warning is True

    async def test_record_measurement_violation(self, tracker):
        m = await tracker.record_measurement(
            metric=SLAMetric.DATA_FRESHNESS,
            tier=_StockPriority.CRITICAL,
            value=20.0,  # Exceeds critical threshold of 15
            ticker="GOOG",
        )
        assert m.meets_sla is False

    async def test_record_measurement_higher_is_better(self, tracker):
        # DATA_COMPLETENESS: higher is better
        m = await tracker.record_measurement(
            metric=SLAMetric.DATA_COMPLETENESS,
            tier=_StockPriority.CRITICAL,
            value=99.95,  # Above critical threshold of 95
        )
        assert m.meets_sla is True

    async def test_record_measurement_higher_is_better_violation(self, tracker):
        m = await tracker.record_measurement(
            metric=SLAMetric.DATA_COMPLETENESS,
            tier=_StockPriority.CRITICAL,
            value=90.0,  # Below critical threshold of 95
        )
        assert m.meets_sla is False

    async def test_record_measurement_stores_in_measurements(self, tracker):
        await tracker.record_measurement(
            metric=SLAMetric.API_LATENCY,
            tier=_StockPriority.HIGH,
            value=150.0,
        )
        key = f"{_StockPriority.HIGH.value}:{SLAMetric.API_LATENCY.value}"
        assert len(tracker.measurements[key]) == 1

    def test_calculate_severity_low(self, tracker):
        target = SLATarget_ST(
            metric=SLAMetric.DATA_FRESHNESS,
            target_value=5,
            unit="minutes",
            measurement_window=timedelta(hours=1),
            critical_threshold=15,
            warning_threshold=10,
        )
        severity = tracker._calculate_severity(7.0, target)
        assert severity == "low"

    def test_calculate_severity_medium(self, tracker):
        target = SLATarget_ST(
            metric=SLAMetric.DATA_FRESHNESS,
            target_value=5,
            unit="minutes",
            measurement_window=timedelta(hours=1),
            critical_threshold=15,
            warning_threshold=10,
        )
        # deviation = (8 - 5)/5 = 0.6 -> medium
        severity = tracker._calculate_severity(8.0, target)
        assert severity == "medium"

    def test_calculate_severity_high(self, tracker):
        target = SLATarget_ST(
            metric=SLAMetric.DATA_FRESHNESS,
            target_value=5,
            unit="minutes",
            measurement_window=timedelta(hours=1),
            critical_threshold=15,
            warning_threshold=10,
        )
        # deviation = (12 - 5)/5 = 1.4 -> high
        severity = tracker._calculate_severity(12.0, target)
        assert severity == "high"

    def test_calculate_severity_critical(self, tracker):
        target = SLATarget_ST(
            metric=SLAMetric.DATA_FRESHNESS,
            target_value=5,
            unit="minutes",
            measurement_window=timedelta(hours=1),
            critical_threshold=15,
            warning_threshold=10,
        )
        # deviation = (20 - 5)/5 = 3.0 -> critical
        severity = tracker._calculate_severity(20.0, target)
        assert severity == "critical"

    def test_calculate_severity_higher_is_better(self, tracker):
        target = SLATarget_ST(
            metric=SLAMetric.DATA_COMPLETENESS,
            target_value=99.9,
            unit="percent",
            measurement_window=timedelta(hours=24),
            critical_threshold=95,
            warning_threshold=98,
        )
        # deviation = (99.9 - 50)/99.9 = ~0.4996 -> abs < 0.5 -> "low"
        severity = tracker._calculate_severity(50.0, target)
        assert severity == "low"

    def test_calculate_severity_higher_is_better_critical(self, tracker):
        target = SLATarget_ST(
            metric=SLAMetric.DATA_COMPLETENESS,
            target_value=99.9,
            unit="percent",
            measurement_window=timedelta(hours=24),
            critical_threshold=95,
            warning_threshold=98,
        )
        # deviation = (99.9 - (-110))/99.9 = 209.9/99.9 = ~2.1 -> critical
        severity = tracker._calculate_severity(-110.0, target)
        assert severity == "critical"

    async def test_get_sla_status_empty(self, tracker):
        status = await tracker.get_sla_status()
        assert "overall_compliance" in status
        assert "tiers" in status

    async def test_get_sla_status_with_measurements(self, tracker):
        # Record several measurements
        for val in [3.0, 5.0, 8.0, 12.0, 20.0]:
            await tracker.record_measurement(
                metric=SLAMetric.DATA_FRESHNESS,
                tier=_StockPriority.CRITICAL,
                value=val,
            )
        status = await tracker.get_sla_status(
            tier=_StockPriority.CRITICAL,
            metric=SLAMetric.DATA_FRESHNESS,
        )
        tiers = status["tiers"]
        assert _StockPriority.CRITICAL.value in tiers

    async def test_get_tier_performance(self, tracker):
        for val in [100, 200, 300, 150, 250]:
            await tracker.record_measurement(
                metric=SLAMetric.API_LATENCY,
                tier=_StockPriority.CRITICAL,
                value=val,
            )
        perf = await tracker.get_tier_performance(_StockPriority.CRITICAL, hours=24)
        assert perf["tier"] == _StockPriority.CRITICAL.value
        assert "metrics" in perf

    async def test_get_violation_report(self, tracker):
        # Record a violation
        await tracker.record_measurement(
            metric=SLAMetric.DATA_FRESHNESS,
            tier=_StockPriority.CRITICAL,
            value=25.0,
            ticker="TSLA",
        )
        report = await tracker.get_violation_report()
        assert "total_violations" in report
        assert report["total_violations"] >= 1

    async def test_calculate_sla_credits_no_violations(self, tracker):
        credits = await tracker.calculate_sla_credits(
            _StockPriority.CRITICAL,
            timedelta(days=30),
        )
        assert credits["violations"] == 0
        assert credits["credit_amount"] == credits["base_value"]
        assert credits["credit_percentage"] == 100.0

    async def test_calculate_sla_credits_with_violations(self, tracker):
        # Generate violations
        for _ in range(5):
            await tracker.record_measurement(
                metric=SLAMetric.DATA_FRESHNESS,
                tier=_StockPriority.CRITICAL,
                value=25.0,
            )
        credits = await tracker.calculate_sla_credits(
            _StockPriority.CRITICAL,
            timedelta(days=30),
        )
        assert credits["violations"] >= 5
        assert credits["penalty"] > 0
        assert credits["credit_amount"] < credits["base_value"]

    async def test_detect_violation_patterns_empty(self, tracker):
        patterns = await tracker._detect_violation_patterns(
            datetime.now(timezone.utc) - timedelta(days=7),
            datetime.now(timezone.utc),
        )
        assert isinstance(patterns, list)
