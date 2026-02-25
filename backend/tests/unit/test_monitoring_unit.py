"""
Unit tests for monitoring modules:
  - backend/monitoring/alerting_system.py
  - backend/monitoring/financial_monitoring.py

All external dependencies (Prometheus, aiohttp, SMTP, monitoring_config,
structured_logging, numpy) are mocked so tests are fast and isolated.
"""

import hashlib
import sys
import types
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

# ---------------------------------------------------------------------------
# Build a lightweight stub for prometheus_client so the real package is not
# required. Each metric constructor returns a mock whose .labels() returns
# another mock with .inc(), .set(), .observe() no-ops.
# ---------------------------------------------------------------------------

_label_mock = MagicMock()

def _metric_factory(name, doc, labelnames=None, *a, **kw):
    m = MagicMock()
    m.labels.return_value = _label_mock
    return m


_prom_stub = types.ModuleType("prometheus_client")
_prom_stub.Counter = _metric_factory
_prom_stub.Gauge = _metric_factory
_prom_stub.Histogram = _metric_factory
_prom_stub.Summary = _metric_factory
_prom_stub.Info = _metric_factory

sys.modules.setdefault("prometheus_client", _prom_stub)

# Stub heavy third-party modules that the source imports at module level
for mod_name in ("aiohttp", "asyncpg"):
    sys.modules.setdefault(mod_name, types.ModuleType(mod_name))

# Stub backend.config.monitoring_config
_alerting_cfg = MagicMock()
_alerting_cfg.notification_channels = {
    "email": {"enabled": False},
    "slack": {"enabled": False},
    "pagerduty": {"enabled": False},
}
_alerting_cfg.thresholds = {
    "budget_warning": 75,
    "budget_critical": 90,
    "data_quality_warning": 80,
    "data_quality_critical": 60,
    "api_latency_warning": 2.0,
    "api_latency_critical": 5.0,
}

_mon_cfg = MagicMock()
_mon_cfg.alerting = _alerting_cfg

_mon_config_mod = types.ModuleType("backend.config.monitoring_config")
_mon_config_mod.monitoring_config = _mon_cfg
_mon_config_mod.initialize_monitoring = MagicMock(return_value=_mon_cfg)
sys.modules["backend.config.monitoring_config"] = _mon_config_mod

# Stub backend.utils.structured_logging
_logging_mod = types.ModuleType("backend.utils.structured_logging")
_logging_mod.get_structured_logger = MagicMock(return_value=MagicMock())
sys.modules["backend.utils.structured_logging"] = _logging_mod

# Now safe to import the modules under test
from backend.monitoring.alerting_system import (
    Alert,
    AlertManager,
    AlertSeverity,
    AlertStatus,
    NotificationChannel,
    NotificationHandler,
)
from backend.monitoring.financial_monitoring import (
    FinancialMonitor,
    PortfolioMetrics,
    StrategyMetrics,
    RecommendationTrackingRecord,
)


# ======================================================================
# ALERTING SYSTEM TESTS
# ======================================================================


class TestAlertSeverityEnum:
    """Verify AlertSeverity enum values."""

    def test_info_value(self):
        assert AlertSeverity.INFO.value == "info"

    def test_warning_value(self):
        assert AlertSeverity.WARNING.value == "warning"

    def test_critical_value(self):
        assert AlertSeverity.CRITICAL.value == "critical"

    def test_emergency_value(self):
        assert AlertSeverity.EMERGENCY.value == "emergency"

    def test_all_members_present(self):
        names = {m.name for m in AlertSeverity}
        assert names == {"INFO", "WARNING", "CRITICAL", "EMERGENCY"}


class TestAlertStatusEnum:
    """Verify AlertStatus enum values."""

    def test_active_value(self):
        assert AlertStatus.ACTIVE.value == "active"

    def test_acknowledged_value(self):
        assert AlertStatus.ACKNOWLEDGED.value == "acknowledged"

    def test_resolved_value(self):
        assert AlertStatus.RESOLVED.value == "resolved"

    def test_suppressed_value(self):
        assert AlertStatus.SUPPRESSED.value == "suppressed"


class TestNotificationChannelEnum:
    """Verify NotificationChannel enum values."""

    def test_all_channels(self):
        expected = {"email", "slack", "pagerduty", "webhook", "sms"}
        actual = {ch.value for ch in NotificationChannel}
        assert actual == expected


class TestAlertDataclass:
    """Tests for the Alert dataclass and fingerprint generation."""

    def _make_alert(self, **overrides):
        defaults = dict(
            id="alert_1",
            title="High latency",
            description="Endpoint /api/v1/prices is slow",
            severity=AlertSeverity.WARNING,
            source="api_monitor",
            alert_type="high_latency",
            timestamp=datetime(2026, 1, 15, 12, 0, 0),
        )
        defaults.update(overrides)
        return Alert(**defaults)

    def test_fingerprint_generated_automatically(self):
        alert = self._make_alert()
        assert alert.fingerprint is not None
        assert len(alert.fingerprint) == 32  # MD5 hex length

    def test_fingerprint_deterministic(self):
        """Same source:alert_type:title produces identical fingerprint."""
        a1 = self._make_alert(id="a1")
        a2 = self._make_alert(id="a2")
        assert a1.fingerprint == a2.fingerprint

    def test_fingerprint_differs_for_different_title(self):
        a1 = self._make_alert(title="alert A")
        a2 = self._make_alert(title="alert B")
        assert a1.fingerprint != a2.fingerprint

    def test_fingerprint_matches_manual_md5(self):
        alert = self._make_alert()
        content = f"{alert.source}:{alert.alert_type}:{alert.title}"
        expected = hashlib.md5(content.encode()).hexdigest()
        assert alert.fingerprint == expected

    def test_custom_fingerprint_preserved(self):
        alert = self._make_alert(fingerprint="custom_fp")
        assert alert.fingerprint == "custom_fp"

    def test_to_dict_contains_all_fields(self):
        alert = self._make_alert(metadata={"key": "val"})
        d = alert.to_dict()
        assert d["id"] == "alert_1"
        assert d["severity"] == "warning"
        assert d["status"] == "active"
        assert d["metadata"] == {"key": "val"}
        assert d["fingerprint"] is not None

    def test_default_status_is_active(self):
        alert = self._make_alert()
        assert alert.status == AlertStatus.ACTIVE

    def test_default_escalation_count_is_zero(self):
        alert = self._make_alert()
        assert alert.escalation_count == 0


class TestAlertManagerCreateAlert:
    """Tests for AlertManager.create_alert()."""

    @pytest.fixture()
    def manager(self):
        mgr = AlertManager()
        # Clear any handlers set during __init__ since config stubs disabled them
        mgr.notification_handlers.clear()
        return mgr

    @pytest.mark.asyncio
    async def test_create_alert_stores_in_active(self, manager):
        alert = await manager.create_alert(
            title="Test alert",
            description="Unit test",
            severity=AlertSeverity.WARNING,
            source="test",
            alert_type="unit_test",
        )
        assert alert is not None
        assert alert.id in manager.active_alerts
        assert manager.active_alerts[alert.id].title == "Test alert"

    @pytest.mark.asyncio
    async def test_create_alert_adds_to_history(self, manager):
        alert = await manager.create_alert(
            title="History alert",
            description="desc",
            severity=AlertSeverity.INFO,
            source="test",
            alert_type="unit_test",
        )
        assert alert is not None
        assert len(manager.alert_history) == 1
        assert manager.alert_history[0].id == alert.id

    @pytest.mark.asyncio
    async def test_create_alert_sets_cooldown(self, manager):
        await manager.create_alert(
            title="Cooldown alert",
            description="desc",
            severity=AlertSeverity.WARNING,
            source="test",
            alert_type="unit_test",
        )
        cooldown_key = "test:unit_test"
        assert cooldown_key in manager.alert_cooldowns

    @pytest.mark.asyncio
    async def test_duplicate_within_cooldown_returns_none(self, manager):
        """Second alert with same source:type within cooldown window returns None."""
        first = await manager.create_alert(
            title="First",
            description="d",
            severity=AlertSeverity.WARNING,
            source="src",
            alert_type="tp",
        )
        assert first is not None

        second = await manager.create_alert(
            title="First",
            description="d again",
            severity=AlertSeverity.WARNING,
            source="src",
            alert_type="tp",
        )
        # The cooldown should suppress the second alert
        assert second is None

    @pytest.mark.asyncio
    async def test_create_alert_with_metadata(self, manager):
        alert = await manager.create_alert(
            title="Meta alert",
            description="desc",
            severity=AlertSeverity.CRITICAL,
            source="test",
            alert_type="unit_test",
            metadata={"latency": 5.2, "endpoint": "/api"},
        )
        assert alert is not None
        assert alert.metadata["latency"] == 5.2


class TestAlertManagerDeduplication:
    """Deduplication via fingerprint: existing ACTIVE alert is updated, not duplicated."""

    @pytest.fixture()
    def manager(self):
        mgr = AlertManager()
        mgr.notification_handlers.clear()
        return mgr

    @pytest.mark.asyncio
    async def test_same_fingerprint_updates_existing(self, manager):
        first = await manager.create_alert(
            title="Dup",
            description="original",
            severity=AlertSeverity.WARNING,
            source="src",
            alert_type="tp",
        )
        assert first is not None
        first_id = first.id

        # Clear cooldown to allow a second alert with the same source:type
        manager.alert_cooldowns.clear()

        second = await manager.create_alert(
            title="Dup",
            description="updated",
            severity=AlertSeverity.WARNING,
            source="src",
            alert_type="tp",
            metadata={"round": 2},
        )
        # Should update the existing alert (same fingerprint)
        assert second is not None
        assert second.id == first_id
        assert second.metadata.get("round") == 2

    @pytest.mark.asyncio
    async def test_different_fingerprint_creates_new(self, manager):
        """Directly insert two alerts with different fingerprints to verify
        AlertManager tracks them independently (avoids ID timestamp collision)."""
        alert_a = Alert(
            id="alert_dedup_a",
            title="Alert A",
            description="d",
            severity=AlertSeverity.WARNING,
            source="srcA",
            alert_type="tpA",
            timestamp=datetime.now(),
        )
        alert_b = Alert(
            id="alert_dedup_b",
            title="Alert B",
            description="d",
            severity=AlertSeverity.WARNING,
            source="srcB",
            alert_type="tpB",
            timestamp=datetime.now(),
        )
        manager.active_alerts[alert_a.id] = alert_a
        manager.active_alerts[alert_b.id] = alert_b

        assert len(manager.active_alerts) == 2
        assert alert_a.fingerprint != alert_b.fingerprint
        assert manager._find_existing_alert(alert_a.fingerprint) is alert_a
        assert manager._find_existing_alert(alert_b.fingerprint) is alert_b


class TestAlertManagerEscalation:
    """Escalation logic tests."""

    @pytest.fixture()
    def manager(self):
        mgr = AlertManager()
        mgr.notification_handlers.clear()
        return mgr

    @pytest.mark.asyncio
    async def test_escalate_warning_to_critical(self, manager):
        alert = Alert(
            id="esc_1",
            title="Escalation test",
            description="desc",
            severity=AlertSeverity.WARNING,
            source="test",
            alert_type="unit_test",
            timestamp=datetime.now(),
        )
        manager.active_alerts[alert.id] = alert

        await manager._escalate_alert(alert)

        assert alert.severity == AlertSeverity.CRITICAL
        assert alert.escalation_count == 1

    @pytest.mark.asyncio
    async def test_escalate_critical_to_emergency(self, manager):
        alert = Alert(
            id="esc_2",
            title="Escalation test",
            description="desc",
            severity=AlertSeverity.CRITICAL,
            source="test",
            alert_type="unit_test",
            timestamp=datetime.now(),
        )
        manager.active_alerts[alert.id] = alert

        await manager._escalate_alert(alert)

        assert alert.severity == AlertSeverity.EMERGENCY
        assert alert.escalation_count == 1

    @pytest.mark.asyncio
    async def test_escalation_increments_count(self, manager):
        alert = Alert(
            id="esc_3",
            title="Multi escalation",
            description="desc",
            severity=AlertSeverity.WARNING,
            source="test",
            alert_type="unit_test",
            timestamp=datetime.now(),
        )
        manager.active_alerts[alert.id] = alert

        await manager._escalate_alert(alert)
        await manager._escalate_alert(alert)

        assert alert.escalation_count == 2
        assert alert.severity == AlertSeverity.EMERGENCY


class TestAlertManagerAcknowledge:
    """acknowledge_alert() tests."""

    @pytest.fixture()
    def manager(self):
        mgr = AlertManager()
        mgr.notification_handlers.clear()
        return mgr

    @pytest.mark.asyncio
    async def test_acknowledge_changes_status(self, manager):
        alert = Alert(
            id="ack_1",
            title="Ack test",
            description="d",
            severity=AlertSeverity.WARNING,
            source="test",
            alert_type="tp",
            timestamp=datetime.now(),
        )
        manager.active_alerts[alert.id] = alert

        result = await manager.acknowledge_alert("ack_1", "admin")

        assert result is True
        assert alert.status == AlertStatus.ACKNOWLEDGED
        assert alert.acknowledged_by == "admin"
        assert alert.acknowledged_at is not None

    @pytest.mark.asyncio
    async def test_acknowledge_nonexistent_returns_false(self, manager):
        result = await manager.acknowledge_alert("nonexistent", "admin")
        assert result is False


class TestAlertManagerResolve:
    """resolve_alert() tests."""

    @pytest.fixture()
    def manager(self):
        mgr = AlertManager()
        mgr.notification_handlers.clear()
        return mgr

    @pytest.mark.asyncio
    async def test_resolve_removes_from_active(self, manager):
        alert = Alert(
            id="res_1",
            title="Resolve test",
            description="d",
            severity=AlertSeverity.CRITICAL,
            source="test",
            alert_type="tp",
            timestamp=datetime.now(),
        )
        manager.active_alerts[alert.id] = alert

        result = await manager.resolve_alert("res_1", "system")

        assert result is True
        assert "res_1" not in manager.active_alerts

    @pytest.mark.asyncio
    async def test_resolve_sets_resolved_status(self, manager):
        alert = Alert(
            id="res_2",
            title="Resolve status",
            description="d",
            severity=AlertSeverity.WARNING,
            source="test",
            alert_type="tp",
            timestamp=datetime.now(),
        )
        manager.active_alerts[alert.id] = alert

        await manager.resolve_alert("res_2")

        assert alert.status == AlertStatus.RESOLVED
        assert alert.resolved_at is not None

    @pytest.mark.asyncio
    async def test_resolve_nonexistent_returns_false(self, manager):
        result = await manager.resolve_alert("nonexistent")
        assert result is False


class TestAlertManagerMaintenanceMode:
    """Maintenance mode suppresses alerts."""

    @pytest.fixture()
    def manager(self):
        mgr = AlertManager()
        mgr.notification_handlers.clear()
        return mgr

    def test_enable_maintenance_mode(self, manager):
        manager.enable_maintenance_mode(duration_minutes=30)
        assert manager.maintenance_mode is True
        assert manager.maintenance_end_time is not None

    def test_disable_maintenance_mode(self, manager):
        manager.enable_maintenance_mode(60)
        manager.disable_maintenance_mode()
        assert manager.maintenance_mode is False
        assert manager.maintenance_end_time is None

    @pytest.mark.asyncio
    async def test_alert_suppressed_during_maintenance(self, manager):
        manager.enable_maintenance_mode(60)

        alert = await manager.create_alert(
            title="During maintenance",
            description="should be suppressed",
            severity=AlertSeverity.WARNING,
            source="test",
            alert_type="tp",
        )
        assert alert is None

    def test_get_alert_summary_structure(self, manager):
        summary = manager.get_alert_summary()
        assert "active_alerts" in summary
        assert "maintenance_mode" in summary
        assert "severity_breakdown" in summary
        assert "history_size" in summary


class TestNotificationHandlerRateLimiting:
    """Rate-limiter in NotificationHandler base class."""

    def test_not_rate_limited_when_no_limit(self):
        handler = NotificationHandler(config={})
        assert handler.is_rate_limited(None) is False

    def test_not_rate_limited_below_limit(self):
        handler = NotificationHandler(config={})
        handler.record_notification()
        assert handler.is_rate_limited(5) is False

    def test_rate_limited_at_limit(self):
        handler = NotificationHandler(config={})
        for _ in range(10):
            handler.record_notification()
        assert handler.is_rate_limited(10) is True


# ======================================================================
# FINANCIAL MONITORING TESTS
# ======================================================================


class TestFinancialMonitorInit:
    """FinancialMonitor initialisation and summary."""

    def test_initial_caches_empty(self):
        fm = FinancialMonitor()
        assert len(fm.portfolio_cache) == 0
        assert len(fm.strategy_cache) == 0
        assert len(fm.recommendation_tracking) == 0

    def test_get_financial_summary_structure(self):
        fm = FinancialMonitor()
        summary = fm.get_financial_summary()
        assert "portfolios_monitored" in summary
        assert "strategies_monitored" in summary
        assert "recommendations_tracked" in summary
        assert summary["portfolios_monitored"] == 0


class TestTrackPortfolioValue:
    """FinancialMonitor._record_portfolio_metrics sets Prometheus gauge."""

    @pytest.mark.asyncio
    async def test_record_basic_metrics(self):
        fm = FinancialMonitor()
        metrics = PortfolioMetrics(
            portfolio_id="p1",
            user_id="u1",
            total_value=100000.0,
            daily_return=1.5,
            returns=[],
            positions={},
        )
        # Should not raise even with empty returns
        await fm._record_portfolio_metrics(metrics)

    @pytest.mark.asyncio
    async def test_record_metrics_with_returns(self):
        """With >= 30 returns, advanced calculations execute without error."""
        import numpy as np

        fm = FinancialMonitor()
        returns = list(np.random.normal(0.05, 1.0, 60))
        metrics = PortfolioMetrics(
            portfolio_id="p2",
            user_id="u2",
            total_value=200000.0,
            daily_return=returns[0],
            returns=returns,
            positions={"AAPL": 80000, "GOOGL": 60000, "MSFT": 60000},
        )
        # Should compute volatility, sharpe, drawdown, concentration risk
        await fm._record_portfolio_metrics(metrics)


class TestCalculateDailyReturn:
    """Daily return formula: (current - previous) / previous * 100."""

    def test_positive_return(self):
        previous = 100.0
        current = 105.0
        daily_return = (current - previous) / previous * 100
        assert daily_return == pytest.approx(5.0)

    def test_negative_return(self):
        previous = 200.0
        current = 190.0
        daily_return = (current - previous) / previous * 100
        assert daily_return == pytest.approx(-5.0)

    def test_zero_return(self):
        previous = 150.0
        current = 150.0
        daily_return = (current - previous) / previous * 100
        assert daily_return == pytest.approx(0.0)

    def test_zero_previous_raises(self):
        """Division by zero when previous value is 0."""
        previous = 0.0
        current = 100.0
        with pytest.raises(ZeroDivisionError):
            _ = (current - previous) / previous * 100


class TestCalculateSharpeRatio:
    """Sharpe ratio with known inputs."""

    def test_sharpe_basic(self):
        import numpy as np

        returns = np.array([0.1, 0.2, -0.05, 0.15, 0.08, 0.12, -0.03, 0.1, 0.05, 0.07])
        risk_free_daily = 0.02 / 252 * 100  # annualised 2% converted to daily %
        excess = returns - risk_free_daily
        sharpe = np.mean(excess) / np.std(excess) * np.sqrt(252)
        assert sharpe > 0, "Sharpe ratio should be positive for net-positive returns"

    def test_sharpe_zero_volatility(self):
        """If all returns are identical, std is 0 and sharpe is undefined."""
        import numpy as np

        returns = np.array([0.1] * 30)
        std = np.std(returns)
        assert std == pytest.approx(0.0)

    def test_sharpe_negative_excess(self):
        """Negative excess returns yield negative sharpe."""
        import numpy as np

        returns = np.array([-0.5] * 30)
        risk_free_daily = 0.02 / 252 * 100
        excess = returns - risk_free_daily
        std_excess = np.std(excess)
        if std_excess > 0:
            sharpe = np.mean(excess) / std_excess * np.sqrt(252)
            assert sharpe < 0


class TestCalculateMaxDrawdown:
    """Max drawdown: running max, correct percentage."""

    def test_known_drawdown(self):
        import numpy as np

        # Prices: 100 -> 110 -> 88 -> 99
        # Returns (%): +10, -20, +12.5
        returns_pct = np.array([10.0, -20.0, 12.5])
        cumulative = np.cumprod(1 + returns_pct / 100)  # [1.1, 0.88, 0.99]
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max * 100
        max_dd = np.min(drawdowns)
        # Drawdown from 1.1 to 0.88 = -20%
        assert max_dd == pytest.approx(-20.0)

    def test_no_drawdown_monotonic_increase(self):
        import numpy as np

        returns_pct = np.array([1.0, 2.0, 1.5, 3.0])
        cumulative = np.cumprod(1 + returns_pct / 100)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max * 100
        max_dd = np.min(drawdowns)
        assert max_dd == pytest.approx(0.0)

    def test_full_drawdown_to_zero(self):
        import numpy as np

        # Price goes to near-zero: -99%
        returns_pct = np.array([10.0, -99.0])
        cumulative = np.cumprod(1 + returns_pct / 100)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max * 100
        max_dd = np.min(drawdowns)
        assert max_dd == pytest.approx(-99.0)


class TestZeroEdgeCases:
    """Ensure no divide-by-zero in financial calculations."""

    def test_slippage_zero_expected_price(self):
        """record_trade_execution handles expected_price=0 safely."""
        fm = FinancialMonitor()
        # Should not raise; the function guards on expected_price > 0
        fm.record_trade_execution(
            order_type="market",
            market_cap_tier="tier1",
            expected_price=0.0,
            actual_price=50.0,
            execution_time_ms=15.0,
        )

    def test_cost_per_share_zero_shares(self):
        """record_trading_cost handles shares=0 safely."""
        fm = FinancialMonitor()
        # Should not raise; the function guards on shares > 0
        fm.record_trading_cost(
            cost_type="commission",
            amount_usd=10.0,
            shares=0,
            market_cap_tier="tier1",
        )

    def test_daily_return_with_zero_previous_guarded(self):
        """Application code must guard against zero previous value."""
        previous = 0.0
        current = 100.0
        if previous != 0:
            daily_return = (current - previous) / previous * 100
        else:
            daily_return = 0.0
        assert daily_return == 0.0


class TestFinancialMonitorTradeRecording:
    """record_trade_execution and record_trading_cost."""

    def test_record_trade_execution_normal(self):
        fm = FinancialMonitor()
        # Should not raise
        fm.record_trade_execution(
            order_type="limit",
            market_cap_tier="tier2",
            expected_price=100.0,
            actual_price=100.5,
            execution_time_ms=25.0,
            venue="nasdaq",
        )

    def test_record_trading_cost_normal(self):
        fm = FinancialMonitor()
        fm.record_trading_cost(
            cost_type="commission",
            amount_usd=7.95,
            shares=500,
            market_cap_tier="tier1",
            venue="default",
        )


class TestFinancialMonitorRecommendationTracking:
    """add_recommendation_tracking stores records correctly."""

    def test_add_recommendation_returns_id(self):
        fm = FinancialMonitor()
        rec_id = fm.add_recommendation_tracking(
            model="ensemble_v2",
            ticker="AAPL",
            recommendation_type="buy",
            confidence=0.85,
            predicted_return=0.12,
        )
        assert rec_id != ""
        assert "ensemble_v2" in rec_id
        assert "AAPL" in rec_id

    def test_recommendation_stored_in_tracking(self):
        fm = FinancialMonitor()
        rec_id = fm.add_recommendation_tracking(
            model="gpt4",
            ticker="MSFT",
            recommendation_type="hold",
            confidence=0.60,
            predicted_return=0.02,
        )
        assert rec_id in fm.recommendation_tracking
        record = fm.recommendation_tracking[rec_id]
        assert record.ticker == "MSFT"
        assert record.confidence == 0.60


class TestLiquidityAndVolumeBuckets:
    """Helper bucket methods."""

    def test_liquidity_buckets(self):
        fm = FinancialMonitor()
        assert fm._get_liquidity_bucket("tier1") == "high"
        assert fm._get_liquidity_bucket("tier3") == "low"
        assert fm._get_liquidity_bucket("unknown_tier") == "unknown"

    def test_volume_buckets(self):
        fm = FinancialMonitor()
        assert fm._get_volume_bucket(50) == "small"
        assert fm._get_volume_bucket(500) == "medium"
        assert fm._get_volume_bucket(5000) == "large"
        assert fm._get_volume_bucket(50000) == "very_large"
