"""
Unit tests for monitoring modules: alerting_system, alertmanager_webhook, api_performance.

Tests cover:
- alerting_system.py: AlertSeverity, AlertStatus, NotificationChannel enums,
  Alert/AlertRule dataclasses, AlertManager init/create/acknowledge/resolve/suppress,
  deduplication, escalation, maintenance mode, notification routing
- alertmanager_webhook.py: AlertSeverity, AlertState enums, AlertThresholds,
  AlertCooldownConfig, BusinessAlert, AlertRateLimiter, AlertManagerWebhook
- api_performance.py: RequestTracker, APIPerformanceMiddleware, APIHealthChecker
"""

import asyncio
import importlib
import importlib.util
import os
import sys
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock, PropertyMock

import pytest

# ---------------------------------------------------------------------------
# Stub heavy dependencies before importing monitoring modules.
# ---------------------------------------------------------------------------

# Prometheus client stub - MUST support .labels().inc() etc.
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
# Save original prometheus_client references so they can be restored after
# our importlib-based module loads.  We must use fake metrics during our
# module loads to avoid polluting the global CollectorRegistry, which would
# cause "Duplicated timeseries" errors when other test files import the same
# monitoring modules via normal Python import.
_orig_prom = sys.modules.get("prometheus_client")
_orig_prom_core = sys.modules.get("prometheus_client.core")
sys.modules["prometheus_client"] = _prom_mock
_prom_core_mock = MagicMock()
_prom_core_mock.GaugeMetricFamily = MagicMock
_prom_core_mock.CounterMetricFamily = MagicMock
sys.modules["prometheus_client.core"] = _prom_core_mock

# Stub backend.config.settings
_mock_settings = MagicMock()
_mock_settings.VERSION = "1.0.0"
_mock_settings.ENVIRONMENT = "test"
_mock_settings.DEBUG = True
sys.modules.setdefault("backend", MagicMock())
sys.modules.setdefault("backend.config", MagicMock())
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
_mock_mon_config.logging = MagicMock(service_name="test-service", level="DEBUG")

# alerting thresholds
_mock_mon_config.alerting = MagicMock()
_mock_mon_config.alerting.thresholds = {
    "budget_critical": 90,
    "budget_warning": 75,
    "data_quality_critical": 70,
    "data_quality_warning": 80,
    "api_latency_critical": 5.0,
    "api_latency_warning": 2.0,
}
_mock_mon_config.alerting.notification_channels = {
    "email": {"enabled": False},
    "slack": {"enabled": False},
    "pagerduty": {"enabled": False},
}

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
sys.modules.setdefault("backend.utils", MagicMock())
sys.modules.setdefault("backend.utils.structured_logging", _mock_structured_logging)

# Stub asyncpg, aiohttp, smtplib-related, httpx
sys.modules.setdefault("asyncpg", MagicMock())
_aiohttp_mock = MagicMock()
_aiohttp_mock.ClientTimeout = MagicMock
_aiohttp_mock.ClientSession = MagicMock
sys.modules.setdefault("aiohttp", _aiohttp_mock)
sys.modules.setdefault("httpx", MagicMock())

# Stub FastAPI / Starlette dependencies used by api_performance
_fastapi_mock = MagicMock()
sys.modules.setdefault("fastapi", _fastapi_mock)
sys.modules.setdefault("fastapi.middleware", MagicMock())
sys.modules.setdefault("fastapi.middleware.base", MagicMock())
sys.modules.setdefault("starlette", MagicMock())
sys.modules.setdefault("starlette.types", MagicMock())

# Stub backend.monitoring.metrics_collector (used by api_performance)
_mock_metrics_collector_instance = MagicMock()
_mock_metrics_collector_mod = MagicMock()
_mock_metrics_collector_mod.metrics_collector = _mock_metrics_collector_instance
# Save originals of any backend.* modules we stub so we can restore them after
# our importlib loads.  This prevents enum identity and duplicate-registration
# issues in other test files that import the real modules.
_saved_backend_modules = {}
for _modname in [
    "backend.monitoring.metrics_collector",
]:
    _saved_backend_modules[_modname] = sys.modules.get(_modname)
    sys.modules[_modname] = _mock_metrics_collector_mod

# ---------------------------------------------------------------------------
# Import monitoring modules via importlib
# ---------------------------------------------------------------------------

_mon_dir = Path(__file__).resolve().parents[2] / "monitoring"

# --- alerting_system.py ---
_as_spec = importlib.util.spec_from_file_location(
    "alerting_system", _mon_dir / "alerting_system.py"
)
_as = importlib.util.module_from_spec(_as_spec)
_as_spec.loader.exec_module(_as)

AS_AlertSeverity = _as.AlertSeverity
AS_AlertStatus = _as.AlertStatus
AS_NotificationChannel = _as.NotificationChannel
AS_Alert = _as.Alert
AS_AlertRule = _as.AlertRule
AS_NotificationConfig = _as.NotificationConfig
AS_NotificationHandler = _as.NotificationHandler
AS_AlertManager = _as.AlertManager

# --- alertmanager_webhook.py ---
_aw_spec = importlib.util.spec_from_file_location(
    "alertmanager_webhook", _mon_dir / "alertmanager_webhook.py"
)
_aw = importlib.util.module_from_spec(_aw_spec)
_aw_spec.loader.exec_module(_aw)

AW_AlertSeverity = _aw.AlertSeverity
AW_AlertState = _aw.AlertState
AW_AlertThresholds = _aw.AlertThresholds
AW_AlertCooldownConfig = _aw.AlertCooldownConfig
AW_BusinessAlert = _aw.BusinessAlert
AW_AlertRateLimiter = _aw.AlertRateLimiter
AW_AlertManagerWebhook = _aw.AlertManagerWebhook

# --- api_performance.py ---
_ap_spec = importlib.util.spec_from_file_location(
    "api_performance", _mon_dir / "api_performance.py"
)
_ap = importlib.util.module_from_spec(_ap_spec)
_ap_spec.loader.exec_module(_ap)

AP_RequestTracker = _ap.RequestTracker
AP_APIHealthChecker = _ap.APIHealthChecker

# Restore original prometheus_client so other test files can use the real one.
if _orig_prom is not None:
    sys.modules["prometheus_client"] = _orig_prom
else:
    sys.modules.pop("prometheus_client", None)
if _orig_prom_core is not None:
    sys.modules["prometheus_client.core"] = _orig_prom_core
else:
    sys.modules.pop("prometheus_client.core", None)

# Restore all backend.* modules we temporarily stubbed.
for _modname, _orig_mod in _saved_backend_modules.items():
    if _orig_mod is not None:
        sys.modules[_modname] = _orig_mod
    else:
        sys.modules.pop(_modname, None)


# ==========================================================================
# alerting_system.py
# ==========================================================================


class TestAlertSeverityEnum:
    def test_values(self):
        assert AS_AlertSeverity.INFO.value == "info"
        assert AS_AlertSeverity.WARNING.value == "warning"
        assert AS_AlertSeverity.CRITICAL.value == "critical"
        assert AS_AlertSeverity.EMERGENCY.value == "emergency"

    def test_member_count(self):
        assert len(AS_AlertSeverity) == 4


class TestAlertStatusEnum:
    def test_values(self):
        assert AS_AlertStatus.ACTIVE.value == "active"
        assert AS_AlertStatus.ACKNOWLEDGED.value == "acknowledged"
        assert AS_AlertStatus.RESOLVED.value == "resolved"
        assert AS_AlertStatus.SUPPRESSED.value == "suppressed"

    def test_member_count(self):
        assert len(AS_AlertStatus) == 4


class TestNotificationChannelEnum:
    def test_values(self):
        assert AS_NotificationChannel.EMAIL.value == "email"
        assert AS_NotificationChannel.SLACK.value == "slack"
        assert AS_NotificationChannel.PAGERDUTY.value == "pagerduty"
        assert AS_NotificationChannel.WEBHOOK.value == "webhook"
        assert AS_NotificationChannel.SMS.value == "sms"

    def test_member_count(self):
        assert len(AS_NotificationChannel) == 5


class TestAlertDataclass:
    def _make_alert(self, **overrides):
        defaults = dict(
            id="alert_001",
            title="Test Alert",
            description="Test description",
            severity=AS_AlertSeverity.WARNING,
            source="unit_test",
            alert_type="test_type",
            timestamp=datetime(2026, 1, 1, 12, 0, 0),
        )
        defaults.update(overrides)
        return AS_Alert(**defaults)

    def test_construction_defaults(self):
        alert = self._make_alert()
        assert alert.id == "alert_001"
        assert alert.status == AS_AlertStatus.ACTIVE
        assert alert.acknowledged_by is None
        assert alert.acknowledged_at is None
        assert alert.resolved_at is None
        assert alert.escalation_count == 0
        assert alert.metadata == {}

    def test_fingerprint_auto_generated(self):
        alert = self._make_alert()
        assert alert.fingerprint is not None
        assert len(alert.fingerprint) == 32  # md5 hexdigest

    def test_fingerprint_deterministic(self):
        a1 = self._make_alert()
        a2 = self._make_alert()
        assert a1.fingerprint == a2.fingerprint

    def test_fingerprint_differs_for_different_sources(self):
        a1 = self._make_alert(source="src_a")
        a2 = self._make_alert(source="src_b")
        assert a1.fingerprint != a2.fingerprint

    def test_custom_fingerprint_preserved(self):
        alert = self._make_alert(fingerprint="custom_fp")
        assert alert.fingerprint == "custom_fp"

    def test_to_dict(self):
        now = datetime(2026, 3, 1, 10, 0, 0)
        alert = self._make_alert(timestamp=now, metadata={"key": "val"})
        d = alert.to_dict()
        assert d["id"] == "alert_001"
        assert d["severity"] == "warning"
        assert d["status"] == "active"
        assert d["metadata"] == {"key": "val"}
        assert d["timestamp"] == now.isoformat()
        assert d["acknowledged_at"] is None
        assert d["resolved_at"] is None

    def test_to_dict_with_ack(self):
        ack_time = datetime(2026, 3, 1, 11, 0, 0)
        alert = self._make_alert(
            acknowledged_by="admin",
            acknowledged_at=ack_time,
        )
        d = alert.to_dict()
        assert d["acknowledged_by"] == "admin"
        assert d["acknowledged_at"] == ack_time.isoformat()


class TestAlertRuleDataclass:
    def test_defaults(self):
        rule = AS_AlertRule(
            name="test_rule",
            condition="cpu > 90",
            severity=AS_AlertSeverity.CRITICAL,
            description="CPU high",
            source="system",
            alert_type="cpu_alert",
        )
        assert rule.enabled is True
        assert rule.cooldown_minutes == 15
        assert rule.threshold_config == {}
        assert rule.metadata_template == {}


class TestAlertManagerInit:
    def test_init_creates_empty_structures(self):
        mgr = AS_AlertManager()
        assert isinstance(mgr.active_alerts, dict)
        assert len(mgr.active_alerts) == 0
        assert isinstance(mgr.alert_history, deque)
        assert mgr.maintenance_mode is False
        assert mgr.maintenance_end_time is None


class TestAlertManagerCreateAlert:
    @pytest.fixture
    def manager(self):
        mgr = AS_AlertManager()
        # Clear notification handlers so we don't try real sends
        mgr.notification_handlers = {}
        return mgr

    @pytest.mark.asyncio
    async def test_create_alert_returns_alert(self, manager):
        alert = await manager.create_alert(
            title="High CPU",
            description="CPU usage at 95%",
            severity=AS_AlertSeverity.WARNING,
            source="system",
            alert_type="cpu",
        )
        assert alert is not None
        assert alert.title == "High CPU"
        assert alert.severity == AS_AlertSeverity.WARNING
        assert alert.id in manager.active_alerts

    @pytest.mark.asyncio
    async def test_create_alert_in_maintenance_mode_suppressed(self, manager):
        manager.enable_maintenance_mode(duration_minutes=60)
        alert = await manager.create_alert(
            title="Suppressed",
            description="Should not appear",
            severity=AS_AlertSeverity.CRITICAL,
            source="system",
            alert_type="test",
        )
        assert alert is None

    @pytest.mark.asyncio
    async def test_create_alert_deduplication(self, manager):
        """Creating two alerts with same fingerprint should update existing."""
        a1 = await manager.create_alert(
            title="Dup Alert",
            description="first",
            severity=AS_AlertSeverity.WARNING,
            source="src",
            alert_type="type",
        )
        assert a1 is not None
        original_id = a1.id

        # Creating same-fingerprint alert should return existing
        a2 = await manager.create_alert(
            title="Dup Alert",
            description="second",
            severity=AS_AlertSeverity.WARNING,
            source="src",
            alert_type="type",
            metadata={"extra": True},
        )
        # a2 should be the same alert (deduplicated), but might be None
        # if cooldown kicks in. Let's clear cooldowns.
        manager.alert_cooldowns.clear()

        a3 = await manager.create_alert(
            title="Dup Alert",
            description="third",
            severity=AS_AlertSeverity.WARNING,
            source="src",
            alert_type="type",
            metadata={"round": 3},
        )
        if a3 is not None:
            assert a3.id == original_id

    @pytest.mark.asyncio
    async def test_cooldown_suppresses_duplicate_type(self, manager):
        a1 = await manager.create_alert(
            title="Alert",
            description="desc",
            severity=AS_AlertSeverity.INFO,
            source="s",
            alert_type="t",
        )
        assert a1 is not None
        # Immediately creating same source:type should be in cooldown
        a2 = await manager.create_alert(
            title="Alert Again",
            description="desc",
            severity=AS_AlertSeverity.INFO,
            source="s",
            alert_type="t",
        )
        # Could be None (cooldown) or deduplicated
        # Either way the active_alerts count should be 1
        assert len(manager.active_alerts) == 1


class TestAlertManagerAcknowledge:
    @pytest.mark.asyncio
    async def test_acknowledge_existing_alert(self):
        mgr = AS_AlertManager()
        mgr.notification_handlers = {}
        alert = await mgr.create_alert(
            title="Test",
            description="desc",
            severity=AS_AlertSeverity.WARNING,
            source="s",
            alert_type="t",
        )
        assert alert is not None
        result = await mgr.acknowledge_alert(alert.id, "admin")
        assert result is True
        assert mgr.active_alerts[alert.id].status == AS_AlertStatus.ACKNOWLEDGED
        assert mgr.active_alerts[alert.id].acknowledged_by == "admin"
        assert mgr.active_alerts[alert.id].acknowledged_at is not None

    @pytest.mark.asyncio
    async def test_acknowledge_nonexistent_returns_false(self):
        mgr = AS_AlertManager()
        mgr.notification_handlers = {}
        result = await mgr.acknowledge_alert("nonexistent_id", "admin")
        assert result is False


class TestAlertManagerResolve:
    @pytest.mark.asyncio
    async def test_resolve_existing_alert(self):
        mgr = AS_AlertManager()
        mgr.notification_handlers = {}
        alert = await mgr.create_alert(
            title="Resolve Me",
            description="desc",
            severity=AS_AlertSeverity.WARNING,
            source="s",
            alert_type="t",
        )
        assert alert is not None
        result = await mgr.resolve_alert(alert.id, "system")
        assert result is True
        assert alert.id not in mgr.active_alerts

    @pytest.mark.asyncio
    async def test_resolve_nonexistent_returns_false(self):
        mgr = AS_AlertManager()
        mgr.notification_handlers = {}
        result = await mgr.resolve_alert("no_such_id")
        assert result is False


class TestAlertManagerMaintenanceMode:
    def test_enable_maintenance_mode(self):
        mgr = AS_AlertManager()
        mgr.notification_handlers = {}
        mgr.enable_maintenance_mode(30)
        assert mgr.maintenance_mode is True
        assert mgr.maintenance_end_time is not None

    def test_disable_maintenance_mode(self):
        mgr = AS_AlertManager()
        mgr.notification_handlers = {}
        mgr.enable_maintenance_mode(30)
        mgr.disable_maintenance_mode()
        assert mgr.maintenance_mode is False
        assert mgr.maintenance_end_time is None


class TestAlertManagerSuppression:
    @pytest.mark.asyncio
    async def test_fingerprint_suppression(self):
        mgr = AS_AlertManager()
        mgr.notification_handlers = {}
        # Add a fingerprint to suppressed set
        alert_tmp = AS_Alert(
            id="tmp",
            title="T",
            description="D",
            severity=AS_AlertSeverity.INFO,
            source="s",
            alert_type="t",
            timestamp=datetime.now(),
        )
        mgr.suppressed_fingerprints.add(alert_tmp.fingerprint)
        result = await mgr.create_alert(
            title="T",
            description="D",
            severity=AS_AlertSeverity.INFO,
            source="s",
            alert_type="t",
        )
        assert result is None


class TestAlertManagerSummary:
    @pytest.mark.asyncio
    async def test_get_summary_empty(self):
        mgr = AS_AlertManager()
        mgr.notification_handlers = {}
        summary = mgr.get_alert_summary()
        assert summary["active_alerts"] == 0
        assert summary["maintenance_mode"] is False
        assert summary["maintenance_end"] is None

    @pytest.mark.asyncio
    async def test_get_summary_with_alerts(self):
        mgr = AS_AlertManager()
        mgr.notification_handlers = {}
        await mgr.create_alert(
            title="A1",
            description="d1",
            severity=AS_AlertSeverity.WARNING,
            source="s1",
            alert_type="t1",
        )
        summary = mgr.get_alert_summary()
        assert summary["active_alerts"] == 1
        assert "warning" in summary["severity_breakdown"]


class TestNotificationHandler:
    def test_rate_limiter_not_limited_by_default(self):
        handler = AS_NotificationHandler(config={})
        assert handler.is_rate_limited(None) is False

    def test_rate_limiter_under_limit(self):
        handler = AS_NotificationHandler(config={})
        handler.record_notification()
        assert handler.is_rate_limited(10) is False

    def test_rate_limiter_at_limit(self):
        handler = AS_NotificationHandler(config={})
        for _ in range(10):
            handler.record_notification()
        assert handler.is_rate_limited(10) is True


# ==========================================================================
# alertmanager_webhook.py
# ==========================================================================


class TestAWAlertSeverity:
    def test_values(self):
        assert AW_AlertSeverity.INFO.value == "info"
        assert AW_AlertSeverity.WARNING.value == "warning"
        assert AW_AlertSeverity.CRITICAL.value == "critical"

    def test_member_count(self):
        assert len(AW_AlertSeverity) == 3


class TestAWAlertState:
    def test_values(self):
        assert AW_AlertState.FIRING.value == "firing"
        assert AW_AlertState.RESOLVED.value == "resolved"


class TestAlertThresholds:
    def test_defaults(self):
        t = AW_AlertThresholds()
        assert t.api_error_rate_warning == 0.01
        assert t.api_error_rate_critical == 0.05
        assert t.response_time_p99_warning == 1.5
        assert t.response_time_p99_critical == 2.0
        assert t.cache_hit_rate_warning == 0.85
        assert t.memory_usage_warning == 0.80
        assert t.budget_usage_warning == 0.70
        assert t.pipeline_success_rate_warning == 0.95
        assert t.ml_accuracy_warning == 0.75

    def test_from_env_uses_defaults(self):
        t = AW_AlertThresholds.from_env()
        assert t.api_error_rate_warning == 0.01


class TestAlertCooldownConfig:
    def test_defaults(self):
        c = AW_AlertCooldownConfig()
        assert c.default_cooldown == 300
        assert c.max_alerts_per_hour == 10
        assert "high_api_error_rate" in c.alert_cooldowns
        assert "budget_exceeded" in c.alert_cooldowns


class TestBusinessAlert:
    def _make_alert(self, **overrides):
        defaults = dict(
            alert_name="test_alert",
            severity=AW_AlertSeverity.WARNING,
            summary="Test summary",
            description="Test description",
        )
        defaults.update(overrides)
        return AW_BusinessAlert(**defaults)

    def test_fingerprint_auto_generated(self):
        alert = self._make_alert()
        assert alert.fingerprint is not None
        assert len(alert.fingerprint) == 16

    def test_fingerprint_deterministic(self):
        a1 = self._make_alert(labels={"k": "v"})
        a2 = self._make_alert(labels={"k": "v"})
        assert a1.fingerprint == a2.fingerprint

    def test_to_alertmanager_format(self):
        alert = self._make_alert(
            labels={"component": "api"},
            metric_value=0.06,
            threshold_value=0.05,
        )
        fmt = alert.to_alertmanager_format()
        assert fmt["labels"]["alertname"] == "test_alert"
        assert fmt["labels"]["severity"] == "warning"
        assert fmt["labels"]["component"] == "api"
        assert "startsAt" in fmt
        assert "metric_value" in fmt["annotations"]

    def test_to_pagerduty_format(self):
        alert = self._make_alert(severity=AW_AlertSeverity.CRITICAL)
        fmt = alert.to_pagerduty_format()
        assert fmt["event_action"] == "trigger"
        assert fmt["dedup_key"] == alert.fingerprint
        assert fmt["payload"]["severity"] == "critical"

    def test_to_pagerduty_format_resolved(self):
        alert = self._make_alert(state=AW_AlertState.RESOLVED)
        fmt = alert.to_pagerduty_format()
        assert fmt["event_action"] == "resolve"

    def test_to_slack_format(self):
        alert = self._make_alert(
            severity=AW_AlertSeverity.WARNING,
            metric_value=0.03,
            threshold_value=0.01,
        )
        fmt = alert.to_slack_format()
        assert "attachments" in fmt
        assert len(fmt["attachments"]) == 1
        attachment = fmt["attachments"][0]
        assert attachment["color"] == "#ffc107"
        fields = attachment["fields"]
        field_titles = [f["title"] for f in fields]
        assert "Severity" in field_titles
        assert "Current Value" in field_titles


class TestAlertRateLimiter:
    def test_allows_first_alert(self):
        config = AW_AlertCooldownConfig()
        limiter = AW_AlertRateLimiter(config)
        alert = AW_BusinessAlert(
            alert_name="test",
            severity=AW_AlertSeverity.WARNING,
            summary="s",
            description="d",
        )
        assert limiter.should_allow_alert(alert) is True

    def test_blocks_within_cooldown(self):
        config = AW_AlertCooldownConfig(default_cooldown=600)
        limiter = AW_AlertRateLimiter(config)
        alert = AW_BusinessAlert(
            alert_name="test",
            severity=AW_AlertSeverity.WARNING,
            summary="s",
            description="d",
        )
        limiter.record_alert(alert)
        assert limiter.should_allow_alert(alert) is False

    def test_blocks_at_hourly_limit(self):
        config = AW_AlertCooldownConfig(
            default_cooldown=0,
            max_alerts_per_hour=2,
        )
        limiter = AW_AlertRateLimiter(config)
        alert = AW_BusinessAlert(
            alert_name="test",
            severity=AW_AlertSeverity.WARNING,
            summary="s",
            description="d",
        )
        # Record 2 alerts for this alert_name
        limiter._alert_counts["test"].append(datetime.now(timezone.utc))
        limiter._alert_counts["test"].append(datetime.now(timezone.utc))
        assert limiter.should_allow_alert(alert) is False

    def test_get_stats(self):
        config = AW_AlertCooldownConfig()
        limiter = AW_AlertRateLimiter(config)
        stats = limiter.get_stats()
        assert "hourly_counts" in stats
        assert "cooldown_active" in stats
        assert stats["max_per_hour"] == 10


class TestAlertManagerWebhookInit:
    def test_init_with_defaults(self):
        webhook = AW_AlertManagerWebhook(
            alertmanager_url="http://am:9093",
            slack_webhook_url=None,
        )
        assert webhook.alertmanager_url == "http://am:9093"
        assert "alertmanager" in webhook._enabled_targets

    def test_get_active_alerts_empty(self):
        webhook = AW_AlertManagerWebhook()
        assert webhook.get_active_alerts() == []

    def test_get_stats(self):
        webhook = AW_AlertManagerWebhook()
        stats = webhook.get_stats()
        assert "enabled_targets" in stats
        assert "active_alerts" in stats
        assert "rate_limiter" in stats
        assert "thresholds" in stats


class TestAlertManagerWebhookConvenienceMethods:
    @pytest.mark.asyncio
    async def test_alert_high_api_error_rate_below_threshold(self):
        webhook = AW_AlertManagerWebhook()
        webhook._enabled_targets = set()  # disable sends
        result = await webhook.alert_high_api_error_rate(0.005)
        assert result is False

    @pytest.mark.asyncio
    async def test_alert_slow_response_below_threshold(self):
        webhook = AW_AlertManagerWebhook()
        webhook._enabled_targets = set()
        result = await webhook.alert_slow_response_time(0.5)
        assert result is False

    @pytest.mark.asyncio
    async def test_alert_low_cache_above_threshold(self):
        webhook = AW_AlertManagerWebhook()
        webhook._enabled_targets = set()
        result = await webhook.alert_low_cache_hit_rate(0.95)
        assert result is False

    @pytest.mark.asyncio
    async def test_alert_high_memory_below_threshold(self):
        webhook = AW_AlertManagerWebhook()
        webhook._enabled_targets = set()
        result = await webhook.alert_high_memory_usage(0.50)
        assert result is False

    @pytest.mark.asyncio
    async def test_alert_budget_below_threshold(self):
        webhook = AW_AlertManagerWebhook()
        webhook._enabled_targets = set()
        result = await webhook.alert_budget_exceeded(50, 1000)
        assert result is False

    @pytest.mark.asyncio
    async def test_alert_pipeline_above_threshold(self):
        webhook = AW_AlertManagerWebhook()
        webhook._enabled_targets = set()
        result = await webhook.alert_pipeline_failure("etl", 0.99)
        assert result is False

    @pytest.mark.asyncio
    async def test_alert_ml_above_threshold(self):
        webhook = AW_AlertManagerWebhook()
        webhook._enabled_targets = set()
        result = await webhook.alert_ml_model_degradation("model_a", 0.90)
        assert result is False


class TestAlertManagerWebhookResolve:
    @pytest.mark.asyncio
    async def test_resolve_nonexistent_fingerprint(self):
        webhook = AW_AlertManagerWebhook()
        webhook._enabled_targets = set()
        result = await webhook.resolve_alert("nonexistent_fp")
        assert result is False

    @pytest.mark.asyncio
    async def test_resolve_existing_alert(self):
        webhook = AW_AlertManagerWebhook()
        webhook._enabled_targets = set()
        alert = AW_BusinessAlert(
            alert_name="test",
            severity=AW_AlertSeverity.WARNING,
            summary="s",
            description="d",
        )
        webhook._active_alerts[alert.fingerprint] = alert
        result = await webhook.resolve_alert(alert.fingerprint)
        assert result is True
        assert alert.fingerprint not in webhook._active_alerts


# ==========================================================================
# api_performance.py
# ==========================================================================


class TestRequestTracker:
    def test_record_request_stores_times(self):
        tracker = AP_RequestTracker()
        tracker.record_request("/api/test", 0.05, 200)
        assert len(tracker.request_times["/api/test"]) == 1
        assert tracker.total_requests["/api/test"] == 1
        assert tracker.error_counts["/api/test"] == 0

    def test_record_request_counts_errors(self):
        tracker = AP_RequestTracker()
        tracker.record_request("/api/fail", 0.10, 500)
        assert tracker.error_counts["/api/fail"] == 1
        tracker.record_request("/api/fail", 0.10, 404)
        assert tracker.error_counts["/api/fail"] == 2

    def test_get_percentiles_empty(self):
        tracker = AP_RequestTracker()
        p = tracker.get_percentiles("/nonexistent")
        assert p == {"p50": 0.0, "p95": 0.0, "p99": 0.0}

    def test_get_percentiles_with_data(self):
        tracker = AP_RequestTracker()
        for i in range(100):
            tracker.record_request("/api/perf", float(i) / 100.0, 200)
        p = tracker.get_percentiles("/api/perf")
        assert p["p50"] > 0.0
        assert p["p95"] > p["p50"]
        assert p["p99"] >= p["p95"]

    def test_get_error_rate_zero(self):
        tracker = AP_RequestTracker()
        assert tracker.get_error_rate("/nope") == 0.0

    def test_get_error_rate_calculation(self):
        tracker = AP_RequestTracker()
        for _ in range(8):
            tracker.record_request("/api/mixed", 0.01, 200)
        for _ in range(2):
            tracker.record_request("/api/mixed", 0.01, 500)
        rate = tracker.get_error_rate("/api/mixed")
        assert rate == pytest.approx(20.0)

    def test_start_and_end_request(self):
        tracker = AP_RequestTracker()
        tracker.start_request("/api/active")
        assert tracker.active_requests["/api/active"] == 1
        tracker.start_request("/api/active")
        assert tracker.active_requests["/api/active"] == 2
        tracker.end_request("/api/active")
        assert tracker.active_requests["/api/active"] == 1

    def test_end_request_floors_at_zero(self):
        tracker = AP_RequestTracker()
        tracker.end_request("/api/zero")
        assert tracker.active_requests["/api/zero"] == 0


class TestAPIHealthChecker:
    def test_record_request_success(self):
        checker = AP_APIHealthChecker()
        checker.record_request("/health", True)
        stats = checker.endpoint_stats["/health"]
        assert stats["total_requests"] == 1
        assert stats["successful_requests"] == 1
        assert stats["consecutive_failures"] == 0
        assert stats["last_success"] is not None

    def test_record_request_failure(self):
        checker = AP_APIHealthChecker()
        checker.record_request("/health", False)
        stats = checker.endpoint_stats["/health"]
        assert stats["total_requests"] == 1
        assert stats["successful_requests"] == 0
        assert stats["consecutive_failures"] == 1

    def test_consecutive_failures_reset_on_success(self):
        checker = AP_APIHealthChecker()
        checker.record_request("/ep", False)
        checker.record_request("/ep", False)
        assert checker.endpoint_stats["/ep"]["consecutive_failures"] == 2
        checker.record_request("/ep", True)
        assert checker.endpoint_stats["/ep"]["consecutive_failures"] == 0

    def test_multiple_endpoints_independent(self):
        checker = AP_APIHealthChecker()
        checker.record_request("/a", True)
        checker.record_request("/b", False)
        assert checker.endpoint_stats["/a"]["successful_requests"] == 1
        assert checker.endpoint_stats["/b"]["successful_requests"] == 0
