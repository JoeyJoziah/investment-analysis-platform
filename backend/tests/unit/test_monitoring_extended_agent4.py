"""
Unit tests for monitoring modules: health_system.py, log_analysis.py, real_time_alerts.py

Tests cover:
- health_system.py: HealthStatus, HealthCheck, HealthMonitor, ServiceHealth,
  dependency tracking, graceful shutdown, signal handling, cache, overall status
- log_analysis.py: LogLevel, AlertCategory, LogEntry, LogPatternDetector,
  LogAnomalyDetector, LogAggregator, LogAnalysisSystem
- real_time_alerts.py: AlertSeverity, AlertCategory, AlertChannel, AlertRule,
  Alert, RealTimeAlertManager (trigger, cooldown, dedup, escalation, stats)
"""

import asyncio
import importlib
import importlib.util
import sys
import time
import json
import hashlib
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
from dataclasses import dataclass

import pytest

# ---------------------------------------------------------------------------
# Stub heavy dependencies BEFORE importing the monitoring modules.
# ---------------------------------------------------------------------------

# Prometheus stub
class _FakeMetric:
    def __init__(self, *args, **kwargs):
        pass
    def labels(self, *args, **kwargs):
        return self
    def inc(self, *args, **kwargs):
        pass
    def dec(self, *args, **kwargs):
        pass
    def set(self, *args, **kwargs):
        pass
    def observe(self, *args, **kwargs):
        pass
    def time(self):
        return MagicMock(__enter__=MagicMock(), __exit__=MagicMock())
    def info(self, *args, **kwargs):
        pass

_prom_mock = MagicMock()
_prom_mock.Counter = _FakeMetric
_prom_mock.Gauge = _FakeMetric
_prom_mock.Histogram = _FakeMetric
_prom_mock.Summary = _FakeMetric
_prom_mock.Info = _FakeMetric
_prom_mock.Enum = _FakeMetric
_prom_mock.CollectorRegistry = MagicMock
_prom_mock.generate_latest = MagicMock(return_value=b"")
_orig_prom4 = sys.modules.get("prometheus_client")
sys.modules["prometheus_client"] = _prom_mock

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
_mock_mon_config.logging = MagicMock(
    service_name="test-service",
    log_file_path="/tmp/test/app.log",
    log_aggregation_endpoint=None,
)
sys.modules.setdefault(
    "backend.config.monitoring_config",
    MagicMock(
        monitoring_config=_mock_mon_config,
        initialize_monitoring=MagicMock(return_value=_mock_mon_config),
    ),
)

# Stub backend.utils.structured_logging
_mock_structured_logging = MagicMock()
_mock_structured_logging.get_structured_logger = MagicMock(return_value=MagicMock())
sys.modules.setdefault("backend.utils.structured_logging", _mock_structured_logging)

# Stub backend.utils.cache (with CacheManager)
_mock_cache_mod = MagicMock()
_mock_cache_mod.CacheManager = MagicMock
_mock_cache_mod.get_redis = AsyncMock(return_value=MagicMock(
    set=AsyncMock(), get=AsyncMock(return_value=None),
))
sys.modules.setdefault("backend.utils.cache", _mock_cache_mod)

# Stub backend.utils.cost_monitor (with CostMonitor)
_mock_cost_monitor_mod = MagicMock()
_mock_cost_monitor_mod.CostMonitor = MagicMock
sys.modules.setdefault("backend.utils.cost_monitor", _mock_cost_monitor_mod)

# Stub backend.utils.database (with SessionLocal)
_mock_db_mod = MagicMock()
_mock_db_mod.SessionLocal = MagicMock
sys.modules.setdefault("backend.utils.database", _mock_db_mod)

# Stub asyncpg, aiohttp, aiofiles, aiofiles.os
sys.modules.setdefault("asyncpg", MagicMock())
sys.modules.setdefault("aiohttp", MagicMock())
_aiofiles_mock = MagicMock()
_aiofiles_os_mock = MagicMock()
sys.modules.setdefault("aiofiles", _aiofiles_mock)
sys.modules.setdefault("aiofiles.os", _aiofiles_os_mock)

# Stub redis, psutil, httpx, docker
sys.modules.setdefault("redis", MagicMock())
sys.modules.setdefault("psutil", MagicMock())
sys.modules.setdefault("httpx", MagicMock())
sys.modules.setdefault("docker", MagicMock())

# Stub smtplib
sys.modules.setdefault("smtplib", MagicMock())

# Stub pandas, numpy (numpy is real, but pandas may not be installed)
sys.modules.setdefault("pandas", MagicMock())

# Stub email.mime.text (real_time_alerts.py has typo: MimeText instead of MIMEText)
# Must force-set (not setdefault) because these are real stdlib modules that
# don't have the MimeText/MimeMultipart typo names.
import email.mime.text as _orig_mime_text
_orig_mime_text.MimeText = MagicMock  # Add the typo name
import email.mime.multipart as _orig_mime_mp
_orig_mime_mp.MimeMultipart = MagicMock  # Add the typo name

# Stub backend.monitoring.alerting_system (used by log_analysis _create_alert_for_anomaly)
_mock_alerting = MagicMock()
_mock_alerting.alert_manager = MagicMock()
_mock_alerting.alert_manager.create_alert = AsyncMock()
_saved_backend_mods_a4 = {}
_backend_stubs_a4 = {
    "backend.monitoring.alerting_system": _mock_alerting,
}
for _modname, _stub in _backend_stubs_a4.items():
    _saved_backend_mods_a4[_modname] = sys.modules.get(_modname)
    sys.modules[_modname] = _stub

# ---------------------------------------------------------------------------
# Import the monitoring modules via importlib
# ---------------------------------------------------------------------------
_mon_dir = Path(__file__).resolve().parents[2] / "monitoring"

# --- health_system.py ---
# Patch signal.signal and asyncio.create_task at import time
with patch("signal.signal"), patch("asyncio.create_task", return_value=MagicMock()):
    _hs_spec = importlib.util.spec_from_file_location("health_system", _mon_dir / "health_system.py")
    _hs = importlib.util.module_from_spec(_hs_spec)
    _hs_spec.loader.exec_module(_hs)

HealthStatus_HS = _hs.HealthStatus
HealthCheck_HS = _hs.HealthCheck
ServiceHealth_HS = _hs.ServiceHealth
HealthMonitor = _hs.HealthMonitor

# --- log_analysis.py ---
_la_spec = importlib.util.spec_from_file_location("log_analysis", _mon_dir / "log_analysis.py")
_la = importlib.util.module_from_spec(_la_spec)
_la_spec.loader.exec_module(_la)

LogLevel = _la.LogLevel
AlertCategory_LA = _la.AlertCategory
LogEntry = _la.LogEntry
LogPattern = _la.LogPattern
LogAnomaly = _la.LogAnomaly
LogPatternDetector = _la.LogPatternDetector
LogAnomalyDetector = _la.LogAnomalyDetector
LogAggregator = _la.LogAggregator
LogAnalysisSystem = _la.LogAnalysisSystem

# --- real_time_alerts.py ---
_rta_spec = importlib.util.spec_from_file_location("real_time_alerts", _mon_dir / "real_time_alerts.py")
_rta = importlib.util.module_from_spec(_rta_spec)
_rta_spec.loader.exec_module(_rta)

AlertSeverity = _rta.AlertSeverity
AlertCategory_RTA = _rta.AlertCategory
AlertChannel = _rta.AlertChannel
AlertRule = _rta.AlertRule
Alert_RTA = _rta.Alert
RealTimeAlertManager = _rta.RealTimeAlertManager

# Restore prometheus_client so other test files can use the real one.
if _orig_prom4 is not None:
    sys.modules["prometheus_client"] = _orig_prom4
else:
    sys.modules.pop("prometheus_client", None)

# Restore all backend.* modules we temporarily stubbed.
for _modname, _orig_mod in _saved_backend_mods_a4.items():
    if _orig_mod is not None:
        sys.modules[_modname] = _orig_mod
    else:
        sys.modules.pop(_modname, None)


# ==========================================================================
# health_system.py
# ==========================================================================

class TestHealthStatusHS:
    def test_enum_values(self):
        assert HealthStatus_HS.HEALTHY.value == "healthy"
        assert HealthStatus_HS.DEGRADED.value == "degraded"
        assert HealthStatus_HS.UNHEALTHY.value == "unhealthy"
        assert HealthStatus_HS.CRITICAL.value == "critical"

    def test_member_count(self):
        assert len(HealthStatus_HS) == 4


class TestHealthCheckHS:
    def test_defaults(self):
        hc = HealthCheck_HS(name="db", check_func=lambda: True)
        assert hc.timeout == 30
        assert hc.interval == 60
        assert hc.retries == 3
        assert hc.critical is False
        assert hc.dependencies == []
        assert hc.last_check is None
        assert hc.last_status == HealthStatus_HS.HEALTHY
        assert hc.consecutive_failures == 0
        assert hc.last_error is None

    def test_custom_values(self):
        hc = HealthCheck_HS(
            name="redis",
            check_func=lambda: True,
            timeout=10,
            interval=15,
            retries=5,
            critical=True,
            dependencies=["db"],
        )
        assert hc.timeout == 10
        assert hc.critical is True
        assert hc.dependencies == ["db"]

    def test_state_tracking(self):
        hc = HealthCheck_HS(name="test", check_func=lambda: True)
        hc.consecutive_failures = 5
        hc.last_error = "connection refused"
        hc.last_status = HealthStatus_HS.CRITICAL
        assert hc.consecutive_failures == 5
        assert hc.last_error == "connection refused"
        assert hc.last_status == HealthStatus_HS.CRITICAL


class TestServiceHealthHS:
    def test_construction(self):
        now = datetime.now(timezone.utc)
        sh = ServiceHealth_HS(
            service_name="investment-platform",
            status=HealthStatus_HS.HEALTHY,
            checks={"db": {"status": HealthStatus_HS.HEALTHY}},
            uptime=120.5,
            timestamp=now,
            version="1.0.0",
            environment="test",
        )
        assert sh.service_name == "investment-platform"
        assert sh.status == HealthStatus_HS.HEALTHY
        assert sh.uptime == 120.5
        assert sh.version == "1.0.0"


class TestHealthMonitor:
    @pytest.fixture
    def monitor(self):
        with patch("signal.signal"), patch("asyncio.create_task", return_value=MagicMock()):
            m = HealthMonitor(service_name="test-svc", version="2.0.0")
        return m

    def test_init_registers_core_checks(self, monitor):
        assert "system_memory" in monitor.health_checks
        assert "system_cpu" in monitor.health_checks
        assert "disk_space" in monitor.health_checks
        assert "application_startup" in monitor.health_checks

    def test_register_health_check(self, monitor):
        initial = len(monitor.health_checks)
        monitor.register_health_check("custom", lambda: True, timeout=5)
        assert len(monitor.health_checks) == initial + 1
        assert "custom" in monitor.health_checks

    def test_register_cleanup_task(self, monitor):
        func = MagicMock()
        monitor.register_cleanup_task(func)
        assert func in monitor.cleanup_tasks

    def test_get_overall_health_status_empty(self, monitor):
        result = monitor.get_overall_health_status({})
        assert result == HealthStatus_HS.UNHEALTHY

    def test_get_overall_health_status_all_healthy(self, monitor):
        checks = {
            "a": {"status": HealthStatus_HS.HEALTHY},
            "b": {"status": HealthStatus_HS.HEALTHY},
        }
        result = monitor.get_overall_health_status(checks)
        assert result == HealthStatus_HS.HEALTHY

    def test_get_overall_health_status_degraded(self, monitor):
        checks = {
            "a": {"status": HealthStatus_HS.HEALTHY},
            "b": {"status": HealthStatus_HS.DEGRADED},
        }
        result = monitor.get_overall_health_status(checks)
        assert result == HealthStatus_HS.DEGRADED

    def test_get_overall_health_status_critical_non_critical_check(self, monitor):
        """A CRITICAL status on a non-critical check yields UNHEALTHY (not CRITICAL)."""
        # Register a non-critical check
        monitor.register_health_check("non_crit", lambda: True, critical=False)
        checks = {
            "non_crit": {"status": HealthStatus_HS.CRITICAL},
        }
        result = monitor.get_overall_health_status(checks)
        assert result == HealthStatus_HS.UNHEALTHY

    def test_get_overall_health_status_critical_critical_check(self, monitor):
        """A CRITICAL status on a critical check yields CRITICAL."""
        checks = {
            "system_memory": {"status": HealthStatus_HS.CRITICAL},
        }
        result = monitor.get_overall_health_status(checks)
        assert result == HealthStatus_HS.CRITICAL

    @pytest.mark.asyncio
    async def test_run_health_check_success(self, monitor):
        async def good_check():
            return {"status": HealthStatus_HS.HEALTHY, "details": "OK"}

        hc = HealthCheck_HS(name="good", check_func=good_check, retries=1, timeout=5)
        result = await monitor._run_health_check(hc)
        assert result["status"] == HealthStatus_HS.HEALTHY
        assert hc.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_run_health_check_timeout(self, monitor):
        async def slow_check():
            await asyncio.sleep(10)
            return {"status": HealthStatus_HS.HEALTHY}

        hc = HealthCheck_HS(name="slow", check_func=slow_check, retries=1, timeout=0.01)
        result = await monitor._run_health_check(hc)
        assert result["status"] == HealthStatus_HS.CRITICAL
        assert hc.consecutive_failures == 1

    @pytest.mark.asyncio
    async def test_run_health_check_exception_with_retries(self, monitor):
        call_count = 0

        async def failing_check():
            nonlocal call_count
            call_count += 1
            raise RuntimeError("boom")

        hc = HealthCheck_HS(name="fail", check_func=failing_check, retries=2, timeout=5)
        result = await monitor._run_health_check(hc)
        assert result["status"] == HealthStatus_HS.CRITICAL
        assert call_count == 2  # retries=2 means 2 attempts total (range(retries))

    @pytest.mark.asyncio
    async def test_shutdown(self, monitor):
        # Create a real asyncio.Task that we can cancel and await
        async def noop():
            await asyncio.sleep(3600)

        loop = asyncio.get_event_loop()
        task = loop.create_task(noop())
        monitor._monitor_task = task

        cleanup_called = False

        def sync_cleanup():
            nonlocal cleanup_called
            cleanup_called = True

        monitor.register_cleanup_task(sync_cleanup)
        await monitor.shutdown()
        assert monitor.shutdown_event.is_set()
        assert cleanup_called
        assert task.cancelled()

    def test_is_in_cooldown_false(self, monitor):
        """_is_in_cooldown returns False when key not in cooldown_cache."""
        # HealthMonitor doesn't have _is_in_cooldown but RealTimeAlertManager does
        # Testing via cache_ttl instead
        assert monitor.cache_ttl == 10


# ==========================================================================
# log_analysis.py
# ==========================================================================

class TestLogLevel:
    def test_enum_values(self):
        assert LogLevel.DEBUG.value == "DEBUG"
        assert LogLevel.INFO.value == "INFO"
        assert LogLevel.WARNING.value == "WARNING"
        assert LogLevel.ERROR.value == "ERROR"
        assert LogLevel.CRITICAL.value == "CRITICAL"

    def test_member_count(self):
        assert len(LogLevel) == 5


class TestAlertCategoryLA:
    def test_enum_values(self):
        assert AlertCategory_LA.ERROR_SPIKE.value == "error_spike"
        assert AlertCategory_LA.PERFORMANCE_DEGRADATION.value == "performance_degradation"
        assert AlertCategory_LA.SECURITY_INCIDENT.value == "security_incident"
        assert AlertCategory_LA.BUSINESS_ANOMALY.value == "business_anomaly"
        assert AlertCategory_LA.SYSTEM_FAILURE.value == "system_failure"


class TestLogEntry:
    def test_defaults(self):
        entry = LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            service="api",
            message="Hello",
        )
        assert entry.correlation_id is None
        assert entry.request_id is None
        assert entry.user_id is None
        assert entry.metadata == {}
        assert entry.source_file is None
        assert entry.line_number is None
        assert entry.exception_info is None

    def test_from_json_valid(self):
        data = {
            "timestamp": "2025-06-15T10:30:00Z",
            "level": "ERROR",
            "service": "stocks",
            "message": "Database connection failed",
            "correlation_id": "abc-123",
            "user_id": "user42",
        }
        entry = LogEntry.from_json(json.dumps(data))
        assert entry.level == LogLevel.ERROR
        assert entry.service == "stocks"
        assert "Database connection failed" in entry.message
        assert entry.correlation_id == "abc-123"
        assert entry.user_id == "user42"

    def test_from_json_missing_level(self):
        data = {"timestamp": "2025-01-01T00:00:00Z", "service": "test", "message": "hi"}
        entry = LogEntry.from_json(json.dumps(data))
        assert entry.level == LogLevel.INFO

    def test_from_json_invalid_level(self):
        data = {"level": "TRACE", "service": "test", "message": "hi"}
        entry = LogEntry.from_json(json.dumps(data))
        assert entry.level == LogLevel.INFO  # falls back to INFO

    def test_from_json_invalid_json(self):
        entry = LogEntry.from_json("not valid json {{{")
        assert entry.level == LogLevel.ERROR
        assert entry.service == "log_parser"

    def test_from_json_alternate_keys(self):
        """Test @timestamp and msg keys."""
        data = {"@timestamp": "2025-06-15T12:00:00Z", "levelname": "WARNING", "msg": "slow query"}
        entry = LogEntry.from_json(json.dumps(data))
        assert entry.level == LogLevel.WARNING
        assert entry.message == "slow query"


class TestLogPatternDetector:
    @pytest.fixture
    def detector(self):
        return LogPatternDetector()

    def test_default_patterns_loaded(self, detector):
        assert len(detector.patterns) > 0
        assert "sql_error" in detector.patterns
        assert "api_timeout" in detector.patterns
        assert "authentication_failure" in detector.patterns
        assert "memory_error" in detector.patterns

    def test_detect_sql_error(self, detector):
        entry = LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.ERROR,
            service="api",
            message="PostgreSQL database error: connection refused",
        )
        detected = detector.detect_patterns(entry)
        assert "sql_error" in detected
        assert detector.patterns["sql_error"].occurrences >= 1

    def test_detect_timeout(self, detector):
        entry = LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.WARNING,
            service="api",
            message="Request timed out after 30s",
        )
        detected = detector.detect_patterns(entry)
        assert "api_timeout" in detected

    def test_detect_auth_failure(self, detector):
        entry = LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.WARNING,
            service="auth",
            message="Authentication failed for user admin",
        )
        detected = detector.detect_patterns(entry)
        assert "authentication_failure" in detected

    def test_detect_memory_error(self, detector):
        entry = LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.CRITICAL,
            service="worker",
            message="Out of memory: malloc failed",
        )
        detected = detector.detect_patterns(entry)
        assert "memory_error" in detected

    def test_no_pattern_match(self, detector):
        entry = LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            service="api",
            message="User logged in successfully",
        )
        detected = detector.detect_patterns(entry)
        assert len(detected) == 0

    def test_sample_messages_capped(self, detector):
        for i in range(10):
            entry = LogEntry(
                timestamp=datetime.now(),
                level=LogLevel.ERROR,
                service="db",
                message=f"SQL database error number {i}",
            )
            detector.detect_patterns(entry)
        assert len(detector.patterns["sql_error"].sample_messages) <= 5

    def test_services_affected_tracked(self, detector):
        for svc in ["api", "worker", "scheduler"]:
            entry = LogEntry(
                timestamp=datetime.now(),
                level=LogLevel.ERROR,
                service=svc,
                message="Database connection error",
            )
            detector.detect_patterns(entry)
        assert "api" in detector.patterns["sql_error"].services_affected
        assert "worker" in detector.patterns["sql_error"].services_affected


class TestLogAnomalyDetector:
    @pytest.fixture
    def detector(self):
        return LogAnomalyDetector()

    def test_update_baseline(self, detector):
        detector.update_baseline("api", "error_rate", 5.0)
        assert len(detector.baseline_metrics["api"]["error_rate"]) == 1

    def test_detect_anomalies_empty(self, detector):
        anomalies = detector.detect_anomalies([], time_window="5m")
        assert anomalies == []

    def test_detect_anomalies_few_entries(self, detector):
        """Under 10 entries should not trigger error rate anomaly."""
        entries = [
            LogEntry(timestamp=datetime.now(), level=LogLevel.ERROR, service="api", message="err")
            for _ in range(5)
        ]
        anomalies = detector.detect_anomalies(entries, time_window="5m")
        # Should not detect error_rate anomaly (< 10 entries)
        error_anomalies = [a for a in anomalies if a.anomaly_type == AlertCategory_LA.ERROR_SPIKE]
        assert len(error_anomalies) == 0

    def test_detect_security_anomaly(self, detector):
        """50+ security events from one IP should trigger a security anomaly."""
        entries = []
        for i in range(60):
            entry = LogEntry(
                timestamp=datetime.now(),
                level=LogLevel.WARNING,
                service="auth",
                message="Authentication failed from suspicious IP",
                metadata={"client_ip": "10.0.0.1"},
            )
            entries.append(entry)
        anomalies = detector.detect_anomalies(entries, time_window="15m")
        security = [a for a in anomalies if a.anomaly_type == AlertCategory_LA.SECURITY_INCIDENT]
        assert len(security) == 1
        assert security[0].service == "auth"

    def test_no_security_anomaly_below_threshold(self, detector):
        """Fewer than 50 security events from one IP should not trigger."""
        entries = []
        for i in range(30):
            entry = LogEntry(
                timestamp=datetime.now(),
                level=LogLevel.WARNING,
                service="auth",
                message="Login failed",
                metadata={"client_ip": "10.0.0.1"},
            )
            entries.append(entry)
        anomalies = detector.detect_anomalies(entries, time_window="15m")
        security = [a for a in anomalies if a.anomaly_type == AlertCategory_LA.SECURITY_INCIDENT]
        assert len(security) == 0


class TestLogAnalysisSystem:
    @pytest.fixture
    def system(self):
        return LogAnalysisSystem()

    def test_init(self, system):
        assert system.pattern_detector is not None
        assert system.anomaly_detector is not None
        assert system.log_aggregator is not None
        assert len(system.recent_logs) == 0

    def test_get_analysis_summary_empty(self, system):
        summary = system.get_analysis_summary()
        assert summary["recent_logs_count"] == 0
        assert summary["processing_status"] == "stopped"

    def test_get_log_patterns(self, system):
        patterns = system.get_log_patterns()
        assert isinstance(patterns, dict)
        assert "sql_error" in patterns

    @pytest.mark.asyncio
    async def test_search_logs_empty(self, system):
        results = await system.search_logs("error")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_logs_with_entries(self, system):
        now = datetime.now()
        for i in range(5):
            entry = LogEntry(
                timestamp=now,
                level=LogLevel.ERROR,
                service="api",
                message=f"Database error occurred on query {i}",
            )
            system.recent_logs.append(entry)
        # Add a non-matching entry
        system.recent_logs.append(LogEntry(
            timestamp=now, level=LogLevel.INFO, service="api", message="All good"
        ))

        results = await system.search_logs("database error")
        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_search_logs_service_filter(self, system):
        now = datetime.now()
        system.recent_logs.append(LogEntry(
            timestamp=now, level=LogLevel.ERROR, service="api", message="error xyz"
        ))
        system.recent_logs.append(LogEntry(
            timestamp=now, level=LogLevel.ERROR, service="worker", message="error xyz"
        ))
        results = await system.search_logs("error", services=["api"])
        assert len(results) == 1
        assert results[0].service == "api"

    @pytest.mark.asyncio
    async def test_search_logs_level_filter(self, system):
        now = datetime.now()
        system.recent_logs.append(LogEntry(
            timestamp=now, level=LogLevel.ERROR, service="api", message="fail xyz"
        ))
        system.recent_logs.append(LogEntry(
            timestamp=now, level=LogLevel.INFO, service="api", message="fail xyz"
        ))
        results = await system.search_logs("fail", log_levels=[LogLevel.ERROR])
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_search_logs_limit(self, system):
        now = datetime.now()
        for i in range(20):
            system.recent_logs.append(LogEntry(
                timestamp=now, level=LogLevel.INFO, service="api", message="event log"
            ))
        results = await system.search_logs("event", limit=5)
        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_process_log_entries_pattern_detection(self, system):
        entries = [
            LogEntry(
                timestamp=datetime.now(),
                level=LogLevel.ERROR,
                service="db",
                message="SQL database error: constraint violation",
            ),
        ]
        await system._process_log_entries(entries)
        assert system.pattern_detector.patterns["sql_error"].occurrences >= 1

    @pytest.mark.asyncio
    async def test_calculate_error_rates(self, system):
        entries = [
            LogEntry(timestamp=datetime.now(), level=LogLevel.INFO, service="api", message="ok"),
            LogEntry(timestamp=datetime.now(), level=LogLevel.ERROR, service="api", message="fail"),
            LogEntry(timestamp=datetime.now(), level=LogLevel.CRITICAL, service="api", message="crash"),
        ]
        # Should not raise
        await system._calculate_error_rates(entries)


# ==========================================================================
# real_time_alerts.py
# ==========================================================================

class TestAlertSeverity:
    def test_enum_values(self):
        assert AlertSeverity.CRITICAL.value == "critical"
        assert AlertSeverity.HIGH.value == "high"
        assert AlertSeverity.MEDIUM.value == "medium"
        assert AlertSeverity.LOW.value == "low"
        assert AlertSeverity.INFO.value == "info"

    def test_member_count(self):
        assert len(AlertSeverity) == 5


class TestAlertCategoryRTA:
    def test_enum_values(self):
        assert AlertCategory_RTA.MARKET_MOVE.value == "market_move"
        assert AlertCategory_RTA.EARNINGS.value == "earnings"
        assert AlertCategory_RTA.SYSTEM.value == "system"
        assert AlertCategory_RTA.COST.value == "cost"

    def test_member_count(self):
        assert len(AlertCategory_RTA) == 11


class TestAlertChannel:
    def test_enum_values(self):
        assert AlertChannel.EMAIL.value == "email"
        assert AlertChannel.SMS.value == "sms"
        assert AlertChannel.WEBHOOK.value == "webhook"
        assert AlertChannel.IN_APP.value == "in_app"
        assert AlertChannel.SLACK.value == "slack"
        assert AlertChannel.DISCORD.value == "discord"


class TestAlertRule:
    def test_defaults(self):
        rule = AlertRule(
            rule_id="r1",
            name="Price Drop",
            category=AlertCategory_RTA.MARKET_MOVE,
            severity=AlertSeverity.HIGH,
            condition="price_change < -5",
        )
        assert rule.enabled is True
        assert rule.channels == [AlertChannel.EMAIL]
        assert rule.cooldown_minutes == 60
        assert rule.symbols is None
        assert rule.threshold is None
        assert rule.escalation_rules is None
        assert rule.custom_message_template is None


class TestAlertRTA:
    def test_construction(self):
        now = datetime.now()
        alert = Alert_RTA(
            alert_id="a1",
            rule_id="r1",
            symbol="AAPL",
            category=AlertCategory_RTA.MARKET_MOVE,
            severity=AlertSeverity.HIGH,
            title="AAPL dropped 5%",
            message="Alert details",
            data={"change": -5.2},
            created_at=now,
            triggered_value=-5.2,
        )
        assert alert.alert_id == "a1"
        assert alert.symbol == "AAPL"
        assert alert.acknowledged is False
        assert alert.escalated is False
        assert alert.channels == []
        assert len(alert.delivered_channels) == 0


class TestRealTimeAlertManager:
    @pytest.fixture
    def manager(self):
        config = {
            "email": {"enabled": False},
            "sms": {"enabled": False},
            "webhook": {},
            "slack": {},
        }
        mgr = RealTimeAlertManager(config)
        # Replace CacheManager with an async-capable mock
        mgr.cache = MagicMock()
        mgr.cache.set = AsyncMock()
        mgr.cache.lpush = AsyncMock()
        return mgr

    def test_init_delivery_stats(self, manager):
        for channel in AlertChannel:
            assert channel in manager.delivery_stats
            assert manager.delivery_stats[channel]["sent"] == 0

    def test_validate_alert_rule_valid(self, manager):
        rule = AlertRule(
            rule_id="r1",
            name="Test",
            category=AlertCategory_RTA.SYSTEM,
            severity=AlertSeverity.LOW,
            condition="value > 10",
        )
        assert manager._validate_alert_rule(rule) is True

    def test_validate_alert_rule_no_id(self, manager):
        rule = AlertRule(
            rule_id="",
            name="Test",
            category=AlertCategory_RTA.SYSTEM,
            severity=AlertSeverity.LOW,
            condition="value > 10",
        )
        assert manager._validate_alert_rule(rule) is False

    def test_validate_condition_safe(self, manager):
        assert manager._validate_condition("price > 100") is True

    def test_validate_condition_dangerous(self, manager):
        assert manager._validate_condition("import os; os.system('rm')") is False
        assert manager._validate_condition("eval('bad')") is False
        assert manager._validate_condition("exec('danger')") is False
        assert manager._validate_condition("obj.__dict__") is False

    @pytest.mark.asyncio
    async def test_add_alert_rule(self, manager):
        rule = AlertRule(
            rule_id="r1",
            name="Price Alert",
            category=AlertCategory_RTA.MARKET_MOVE,
            severity=AlertSeverity.HIGH,
            condition="price_change > 5",
        )
        result = await manager.add_alert_rule(rule)
        assert result is True
        assert "r1" in manager.alert_rules

    @pytest.mark.asyncio
    async def test_add_invalid_rule(self, manager):
        rule = AlertRule(
            rule_id="",
            name="",
            category=AlertCategory_RTA.SYSTEM,
            severity=AlertSeverity.LOW,
            condition="",
        )
        result = await manager.add_alert_rule(rule)
        assert result is False

    def test_is_in_cooldown_no_entry(self, manager):
        assert manager._is_in_cooldown("key", 60) is False

    def test_is_in_cooldown_active(self, manager):
        manager.cooldown_cache["key"] = datetime.now()
        assert manager._is_in_cooldown("key", 60) is True

    def test_is_in_cooldown_expired(self, manager):
        manager.cooldown_cache["key"] = datetime.now() - timedelta(minutes=120)
        assert manager._is_in_cooldown("key", 60) is False

    def test_generate_alert_id(self, manager):
        aid = manager._generate_alert_id("rule1", "AAPL", 150.0)
        assert isinstance(aid, str)
        assert len(aid) == 12

    def test_generate_alert_id_deterministic_same_day(self, manager):
        aid1 = manager._generate_alert_id("rule1", "AAPL", 150.0)
        aid2 = manager._generate_alert_id("rule1", "AAPL", 150.0)
        assert aid1 == aid2

    def test_generate_correlation_id(self, manager):
        rule = AlertRule(
            rule_id="r1",
            name="Test",
            category=AlertCategory_RTA.MARKET_MOVE,
            severity=AlertSeverity.HIGH,
            condition="x > 0",
        )
        cid = manager._generate_correlation_id(rule, "AAPL", 100.0)
        assert isinstance(cid, str)
        assert len(cid) == 8

    def test_generate_default_message(self, manager):
        rule = AlertRule(
            rule_id="r1",
            name="Price Drop",
            category=AlertCategory_RTA.MARKET_MOVE,
            severity=AlertSeverity.HIGH,
            condition="x < -5",
        )
        title, message = manager._generate_default_message(rule, "AAPL", -5.2)
        assert "HIGH" in title
        assert "Price Drop" in title
        assert "AAPL" in title
        assert "Symbol: AAPL" in message
        assert "Triggered Value: -5.2" in message

    def test_generate_default_message_no_symbol(self, manager):
        rule = AlertRule(
            rule_id="r1",
            name="System Alert",
            category=AlertCategory_RTA.SYSTEM,
            severity=AlertSeverity.CRITICAL,
            condition="cpu > 90",
        )
        title, message = manager._generate_default_message(rule, None, None)
        assert "System Alert" in title
        assert "Symbol" not in message

    def test_generate_alert_message_custom_template(self, manager):
        rule = AlertRule(
            rule_id="r1",
            name="Custom Alert",
            category=AlertCategory_RTA.MARKET_MOVE,
            severity=AlertSeverity.MEDIUM,
            condition="x > 0",
            custom_message_template="Alert for {symbol}: value={value}",
        )
        title, message = manager._generate_alert_message(rule, "TSLA", 42.0, None)
        assert title == "Custom Alert"
        assert "TSLA" in message
        assert "42.0" in message

    @pytest.mark.asyncio
    async def test_trigger_alert_unknown_rule(self, manager):
        result = await manager.trigger_alert("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_trigger_alert_disabled_rule(self, manager):
        rule = AlertRule(
            rule_id="r1",
            name="Disabled",
            category=AlertCategory_RTA.SYSTEM,
            severity=AlertSeverity.LOW,
            condition="x > 0",
            enabled=False,
        )
        manager.alert_rules["r1"] = rule
        result = await manager.trigger_alert("r1")
        assert result is None

    @pytest.mark.asyncio
    async def test_trigger_alert_success(self, manager):
        rule = AlertRule(
            rule_id="r1",
            name="Test Alert",
            category=AlertCategory_RTA.SYSTEM,
            severity=AlertSeverity.LOW,
            condition="x > 0",
            channels=[],  # no channels = no delivery
            cooldown_minutes=0,
        )
        manager.alert_rules["r1"] = rule
        alert_id = await manager.trigger_alert("r1", symbol="AAPL", triggered_value=10.0)
        assert alert_id is not None
        assert alert_id in manager.active_alerts
        assert len(manager.alert_history) == 1

    @pytest.mark.asyncio
    async def test_trigger_alert_cooldown(self, manager):
        rule = AlertRule(
            rule_id="r1",
            name="Cooldown Test",
            category=AlertCategory_RTA.SYSTEM,
            severity=AlertSeverity.LOW,
            condition="x > 0",
            channels=[],
            cooldown_minutes=60,
        )
        manager.alert_rules["r1"] = rule
        first = await manager.trigger_alert("r1", symbol="AAPL")
        assert first is not None
        second = await manager.trigger_alert("r1", symbol="AAPL")
        # Second should be suppressed by cooldown
        assert second is None

    @pytest.mark.asyncio
    async def test_trigger_alert_dedup(self, manager):
        rule = AlertRule(
            rule_id="r1",
            name="Dedup Test",
            category=AlertCategory_RTA.SYSTEM,
            severity=AlertSeverity.LOW,
            condition="x > 0",
            channels=[],
            cooldown_minutes=0,
        )
        manager.alert_rules["r1"] = rule
        first = await manager.trigger_alert("r1", symbol="AAPL", triggered_value=10.0)
        # Clear cooldown so we test dedup path, not cooldown
        manager.cooldown_cache.clear()
        second = await manager.trigger_alert("r1", symbol="AAPL", triggered_value=10.0)
        # Same alert_id (deterministic) should be suppressed as duplicate
        assert second is None

    @pytest.mark.asyncio
    async def test_acknowledge_alert(self, manager):
        alert = Alert_RTA(
            alert_id="a1", rule_id="r1", symbol="AAPL",
            category=AlertCategory_RTA.SYSTEM, severity=AlertSeverity.LOW,
            title="Test", message="msg", data={}, created_at=datetime.now(),
        )
        manager.active_alerts["a1"] = alert
        result = await manager.acknowledge_alert("a1", "admin")
        assert result is True
        assert alert.acknowledged is True
        assert alert.acknowledged_by == "admin"
        assert alert.acknowledged_at is not None

    @pytest.mark.asyncio
    async def test_acknowledge_alert_not_found(self, manager):
        result = await manager.acknowledge_alert("nonexistent", "admin")
        assert result is False

    def test_get_active_alerts_empty(self, manager):
        alerts = manager.get_active_alerts()
        assert alerts == []

    def test_get_active_alerts_filtered(self, manager):
        a1 = Alert_RTA(
            alert_id="a1", rule_id="r1", symbol="AAPL",
            category=AlertCategory_RTA.MARKET_MOVE, severity=AlertSeverity.HIGH,
            title="T1", message="m", data={}, created_at=datetime.now(),
        )
        a2 = Alert_RTA(
            alert_id="a2", rule_id="r2", symbol="TSLA",
            category=AlertCategory_RTA.SYSTEM, severity=AlertSeverity.LOW,
            title="T2", message="m", data={}, created_at=datetime.now(),
        )
        manager.active_alerts["a1"] = a1
        manager.active_alerts["a2"] = a2

        by_severity = manager.get_active_alerts(severity=AlertSeverity.HIGH)
        assert len(by_severity) == 1
        assert by_severity[0].alert_id == "a1"

        by_category = manager.get_active_alerts(category=AlertCategory_RTA.SYSTEM)
        assert len(by_category) == 1

        by_symbol = manager.get_active_alerts(symbol="TSLA")
        assert len(by_symbol) == 1

    def test_get_alert_statistics_empty(self, manager):
        stats = manager.get_alert_statistics()
        assert stats["total_alerts"] == 0
        assert stats["acknowledged_rate"] == 0
        assert stats["escalation_rate"] == 0

    def test_get_alert_statistics_with_history(self, manager):
        now = datetime.now()
        a1 = Alert_RTA(
            alert_id="a1", rule_id="r1", symbol="AAPL",
            category=AlertCategory_RTA.MARKET_MOVE, severity=AlertSeverity.HIGH,
            title="T", message="m", data={}, created_at=now,
            acknowledged=True,
        )
        a2 = Alert_RTA(
            alert_id="a2", rule_id="r2", symbol="TSLA",
            category=AlertCategory_RTA.SYSTEM, severity=AlertSeverity.LOW,
            title="T", message="m", data={}, created_at=now,
            escalated=True,
        )
        manager.alert_history = [a1, a2]
        stats = manager.get_alert_statistics(days=7)
        assert stats["total_alerts"] == 2
        assert stats["by_severity"]["high"] == 1
        assert stats["by_severity"]["low"] == 1
        assert stats["acknowledged_rate"] == 0.5
        assert stats["escalation_rate"] == 0.5

    def test_generate_webhook_signature(self, manager):
        manager.webhook_config = {"secret": "test-secret"}
        payload = {"alert_id": "a1", "title": "test"}
        sig = manager._generate_webhook_signature(payload)
        assert isinstance(sig, str)
        assert len(sig) == 64  # SHA-256 hex

    def test_create_email_html(self, manager):
        alert = Alert_RTA(
            alert_id="a1", rule_id="r1", symbol="AAPL",
            category=AlertCategory_RTA.MARKET_MOVE, severity=AlertSeverity.CRITICAL,
            title="Price Alert", message="AAPL dropped",
            data={"change": -5}, created_at=datetime.now(),
            triggered_value=-5.0,
        )
        html = manager._create_email_html(alert)
        assert "Price Alert" in html
        assert "AAPL" in html
        assert "#dc3545" in html  # CRITICAL color

    @pytest.mark.asyncio
    async def test_check_escalation_no_rule(self, manager):
        alert = Alert_RTA(
            alert_id="a1", rule_id="missing",
            symbol=None, category=AlertCategory_RTA.SYSTEM,
            severity=AlertSeverity.LOW, title="T", message="m",
            data={}, created_at=datetime.now(),
        )
        # Should not raise
        await manager._check_escalation(alert)

    @pytest.mark.asyncio
    async def test_check_escalation_occurrence_count(self, manager):
        rule = AlertRule(
            rule_id="r1", name="Test",
            category=AlertCategory_RTA.SYSTEM,
            severity=AlertSeverity.HIGH,
            condition="x > 0",
            escalation_rules={"occurrence_count": 2, "channels": []},
        )
        manager.alert_rules["r1"] = rule

        correlation_id = "corr-1"
        # Create two correlated alerts
        a1 = Alert_RTA(
            alert_id="a1", rule_id="r1", symbol=None,
            category=AlertCategory_RTA.SYSTEM, severity=AlertSeverity.HIGH,
            title="T", message="m", data={}, created_at=datetime.now(),
            correlation_id=correlation_id,
        )
        a2 = Alert_RTA(
            alert_id="a2", rule_id="r1", symbol=None,
            category=AlertCategory_RTA.SYSTEM, severity=AlertSeverity.HIGH,
            title="T", message="m", data={}, created_at=datetime.now(),
            correlation_id=correlation_id,
        )
        manager.active_alerts["a1"] = a1
        manager.active_alerts["a2"] = a2

        await manager._check_escalation(a2)
        assert a2.escalated is True
        # Escalated alert should be in active_alerts
        assert f"{a2.alert_id}_ESC" in manager.active_alerts
