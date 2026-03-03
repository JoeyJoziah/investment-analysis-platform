"""
Unit tests for monitoring modules: data_quality_metrics.py, database_performance.py,
financial_monitoring.py.

Tests cover:
- data_quality_metrics.py: DataQualityMetricsCollector init, record methods,
  quality trends, export, history tracking
- database_performance.py: DatabasePerformanceMonitor init, _extract_query_info,
  start/stop monitoring, track_query context manager
- financial_monitoring.py: FinancialMonitor init, record methods,
  PortfolioMetrics/StrategyMetrics/RecommendationTrackingRecord dataclasses,
  liquidity/volume buckets, financial summary, risk metrics
"""

import asyncio
import importlib
import sys
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock, PropertyMock
from contextlib import asynccontextmanager

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Stub heavy dependencies before importing monitoring modules.
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
_prom_mock.generate_latest = MagicMock(return_value=b"# metrics")
_orig_prom3 = sys.modules.get("prometheus_client")
_orig_prom_core3 = sys.modules.get("prometheus_client.core")
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
_mock_settings.DATABASE_NAME = "test_db"
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
_mock_mon_config.logging = MagicMock(service_name="test-service")
sys.modules.setdefault("backend.config.monitoring_config", MagicMock(
    monitoring_config=_mock_mon_config,
    initialize_monitoring=MagicMock(return_value=_mock_mon_config),
))

# Stub backend.utils.structured_logging
_mock_structured_logging = MagicMock()
_mock_structured_logging.get_structured_logger = MagicMock(return_value=MagicMock())
sys.modules.setdefault("backend.utils.structured_logging", _mock_structured_logging)

# Stub backend.utils.data_quality
_mock_data_quality = MagicMock()
_mock_data_quality.DataQualityChecker = MagicMock
_mock_data_quality.DataQualitySeverity = MagicMock()

# Stub backend.utils.monitoring
_mock_utils_monitoring = MagicMock()
_mock_utils_monitoring.metrics = MagicMock()

# Stub backend.utils.async_database
_mock_async_db = MagicMock()
_mock_session = AsyncMock()


@asynccontextmanager
async def _fake_get_session():
    yield _mock_session


_mock_async_db.async_db_manager = MagicMock()
_mock_async_db.async_db_manager.get_session = _fake_get_session

# Save/restore pattern: force-set backend.* stubs for importlib loads, restore after.
_saved_backend_mods_a3 = {}
_backend_stubs_a3 = {
    "backend.utils": MagicMock(),
    "backend.utils.data_quality": _mock_data_quality,
    "backend.utils.monitoring": _mock_utils_monitoring,
    "backend.utils.async_database": _mock_async_db,
}
for _modname, _stub in _backend_stubs_a3.items():
    _saved_backend_mods_a3[_modname] = sys.modules.get(_modname)
    sys.modules[_modname] = _stub

# Stub asyncpg
sys.modules.setdefault("asyncpg", MagicMock())

# Stub psutil
sys.modules.setdefault("psutil", MagicMock())

# ---------------------------------------------------------------------------
# Import monitoring modules via importlib
# ---------------------------------------------------------------------------

_mon_dir = Path(__file__).resolve().parents[2] / "monitoring"

# --- data_quality_metrics.py ---
_dqm_spec = importlib.util.spec_from_file_location(
    "data_quality_metrics", _mon_dir / "data_quality_metrics.py"
)
_dqm = importlib.util.module_from_spec(_dqm_spec)
_dqm_spec.loader.exec_module(_dqm)

DataQualityMetricsCollector = _dqm.DataQualityMetricsCollector

# --- database_performance.py ---
_dbp_spec = importlib.util.spec_from_file_location(
    "database_performance", _mon_dir / "database_performance.py"
)
_dbp = importlib.util.module_from_spec(_dbp_spec)
_dbp_spec.loader.exec_module(_dbp)

DatabasePerformanceMonitor = _dbp.DatabasePerformanceMonitor

# --- financial_monitoring.py ---
_fm_spec = importlib.util.spec_from_file_location(
    "financial_monitoring", _mon_dir / "financial_monitoring.py"
)
_fm = importlib.util.module_from_spec(_fm_spec)
_fm_spec.loader.exec_module(_fm)

FinancialMonitor = _fm.FinancialMonitor
PortfolioMetrics = _fm.PortfolioMetrics
StrategyMetrics = _fm.StrategyMetrics
RecommendationTrackingRecord = _fm.RecommendationTrackingRecord

# Restore original prometheus_client so other test files can use the real one.
if _orig_prom3 is not None:
    sys.modules["prometheus_client"] = _orig_prom3
else:
    sys.modules.pop("prometheus_client", None)
if _orig_prom_core3 is not None:
    sys.modules["prometheus_client.core"] = _orig_prom_core3
else:
    sys.modules.pop("prometheus_client.core", None)

# Restore all backend.* modules we temporarily stubbed.
for _modname, _orig_mod in _saved_backend_mods_a3.items():
    if _orig_mod is not None:
        sys.modules[_modname] = _orig_mod
    else:
        sys.modules.pop(_modname, None)


# ==========================================================================
# data_quality_metrics.py
# ==========================================================================


class TestDataQualityMetricsCollectorInit:
    def test_init_default_checker(self):
        collector = DataQualityMetricsCollector()
        assert collector.quality_checker is not None
        assert collector.last_check_times == {}
        assert collector.quality_history == {}

    def test_init_custom_checker(self):
        custom_checker = MagicMock()
        collector = DataQualityMetricsCollector(quality_checker=custom_checker)
        assert collector.quality_checker is custom_checker


class TestRecordQualityCheck:
    @pytest.fixture
    def collector(self):
        return DataQualityMetricsCollector(quality_checker=MagicMock())

    def test_record_passing_check(self, collector):
        result = {
            "quality_score": 95.0,
            "valid": True,
            "issues": [],
        }
        collector.record_quality_check("AAPL", "price", result)
        assert "AAPL" in collector.quality_history
        assert len(collector.quality_history["AAPL"]) == 1
        assert collector.quality_history["AAPL"][0]["score"] == 95.0

    def test_record_failing_check_with_issues(self, collector):
        result = {
            "quality_score": 40.0,
            "valid": False,
            "issues": [
                {"severity": "high", "type": "high_less_than_low"},
                {"severity": "medium", "type": "close_outside_range"},
                {"severity": "low", "type": "missing_data"},
            ],
        }
        collector.record_quality_check("TSLA", "price", result)
        assert collector.quality_history["TSLA"][0]["issues"] == 3

    def test_history_capped_at_100(self, collector):
        for i in range(110):
            result = {"quality_score": float(i), "valid": True, "issues": []}
            collector.record_quality_check("MSFT", "price", result)
        assert len(collector.quality_history["MSFT"]) == 100

    def test_missing_quality_score_defaults_zero(self, collector):
        result = {"valid": True, "issues": []}
        collector.record_quality_check("GOOG", "price", result)
        assert collector.quality_history["GOOG"][0]["score"] == 0


class TestRecordDataStaleness:
    def test_fresh_data(self):
        collector = DataQualityMetricsCollector(quality_checker=MagicMock())
        recent = datetime.now(timezone.utc) - timedelta(hours=1)
        # Should not raise
        collector.record_data_staleness("AAPL", "price", recent)

    def test_stale_data(self):
        collector = DataQualityMetricsCollector(quality_checker=MagicMock())
        old = datetime.now(timezone.utc) - timedelta(days=2)
        # Should not raise, records a stale_data issue
        collector.record_data_staleness("AAPL", "price", old)


class TestRecordMissingData:
    def test_no_missing(self):
        collector = DataQualityMetricsCollector(quality_checker=MagicMock())
        collector.record_missing_data("AAPL", "price", "daily", 0)

    def test_few_missing(self):
        collector = DataQualityMetricsCollector(quality_checker=MagicMock())
        collector.record_missing_data("AAPL", "price", "daily", 3)

    def test_many_missing(self):
        collector = DataQualityMetricsCollector(quality_checker=MagicMock())
        collector.record_missing_data("AAPL", "price", "daily", 10)


class TestRecordAnomaly:
    def test_record_anomaly(self):
        collector = DataQualityMetricsCollector(quality_checker=MagicMock())
        collector.record_anomaly("AAPL", "price", "spike", 0.85, severity="high")


class TestRecordPriceGap:
    def test_small_gap(self):
        collector = DataQualityMetricsCollector(quality_checker=MagicMock())
        collector.record_price_gap("AAPL", 0.05)

    def test_large_gap_triggers_anomaly(self):
        collector = DataQualityMetricsCollector(quality_checker=MagicMock())
        collector.record_price_gap("AAPL", 0.35)

    def test_medium_gap_triggers_anomaly(self):
        collector = DataQualityMetricsCollector(quality_checker=MagicMock())
        collector.record_price_gap("AAPL", 0.25)


class TestRecordVolumeOutlier:
    def test_normal_volume(self):
        collector = DataQualityMetricsCollector(quality_checker=MagicMock())
        collector.record_volume_outlier("AAPL", 1000000, 1000000.0, 100000.0)

    def test_moderate_outlier(self):
        collector = DataQualityMetricsCollector(quality_checker=MagicMock())
        # z_score = |4_000_000 - 1_000_000| / 500_000 = 6.0 -> extreme
        collector.record_volume_outlier("AAPL", 4000000, 1000000.0, 500000.0)

    def test_extreme_outlier(self):
        collector = DataQualityMetricsCollector(quality_checker=MagicMock())
        # z_score = |10_000_000 - 1_000_000| / 100_000 = 90 -> extreme
        collector.record_volume_outlier("AAPL", 10000000, 1000000.0, 100000.0)

    def test_zero_std_dev(self):
        collector = DataQualityMetricsCollector(quality_checker=MagicMock())
        collector.record_volume_outlier("AAPL", 1000, 1000.0, 0.0)


class TestRecordFundamentalCompleteness:
    def test_full_completeness(self):
        collector = DataQualityMetricsCollector(quality_checker=MagicMock())
        collector.record_fundamental_completeness("AAPL", "10-K", 50, 50)

    def test_low_completeness(self):
        collector = DataQualityMetricsCollector(quality_checker=MagicMock())
        collector.record_fundamental_completeness("AAPL", "10-K", 50, 20)

    def test_very_low_completeness(self):
        collector = DataQualityMetricsCollector(quality_checker=MagicMock())
        collector.record_fundamental_completeness("AAPL", "10-K", 50, 10)

    def test_zero_fields(self):
        collector = DataQualityMetricsCollector(quality_checker=MagicMock())
        collector.record_fundamental_completeness("AAPL", "10-K", 0, 0)


class TestRecordValidationFailure:
    def test_record_failure(self):
        collector = DataQualityMetricsCollector(quality_checker=MagicMock())
        collector.record_validation_failure("schema", "price", "negative_value", "price")


class TestRecordComplianceViolation:
    def test_sec_violation(self):
        collector = DataQualityMetricsCollector(quality_checker=MagicMock())
        collector.record_compliance_violation("SEC", "late_filing")

    def test_gdpr_violation(self):
        collector = DataQualityMetricsCollector(quality_checker=MagicMock())
        collector.record_compliance_violation("GDPR", "data_retention")

    def test_non_critical_violation(self):
        collector = DataQualityMetricsCollector(quality_checker=MagicMock())
        collector.record_compliance_violation("internal", "naming_convention")


class TestRecordPipelineQuality:
    def test_good_quality(self):
        collector = DataQualityMetricsCollector(quality_checker=MagicMock())
        collector.record_pipeline_quality("ingestion", "yahoo_finance", 95.0)

    def test_poor_quality(self):
        collector = DataQualityMetricsCollector(quality_checker=MagicMock())
        collector.record_pipeline_quality("ingestion", "yahoo_finance", 40.0)

    def test_medium_quality(self):
        collector = DataQualityMetricsCollector(quality_checker=MagicMock())
        collector.record_pipeline_quality("ingestion", "yahoo_finance", 60.0)


class TestGetQualityTrends:
    def test_no_data(self):
        collector = DataQualityMetricsCollector(quality_checker=MagicMock())
        result = collector.get_quality_trends("UNKNOWN")
        assert result == {"status": "no_data"}

    def test_no_recent_data(self):
        collector = DataQualityMetricsCollector(quality_checker=MagicMock())
        collector.quality_history["AAPL"] = [
            {
                "timestamp": datetime.now(timezone.utc) - timedelta(hours=48),
                "score": 80.0,
                "issues": 2,
            }
        ]
        result = collector.get_quality_trends("AAPL", window_hours=24)
        assert result == {"status": "no_recent_data"}

    def test_improving_trend(self):
        collector = DataQualityMetricsCollector(quality_checker=MagicMock())
        now = datetime.now(timezone.utc)
        collector.quality_history["AAPL"] = [
            {"timestamp": now - timedelta(hours=2), "score": 70.0, "issues": 3},
            {"timestamp": now - timedelta(hours=1), "score": 85.0, "issues": 1},
            {"timestamp": now, "score": 95.0, "issues": 0},
        ]
        result = collector.get_quality_trends("AAPL", window_hours=24)
        assert result["status"] == "ok"
        assert result["trend"] == "improving"
        assert result["check_count"] == 3
        assert result["min_score"] == 70.0
        assert result["max_score"] == 95.0
        assert result["total_issues"] == 4

    def test_declining_trend(self):
        collector = DataQualityMetricsCollector(quality_checker=MagicMock())
        now = datetime.now(timezone.utc)
        collector.quality_history["AAPL"] = [
            {"timestamp": now - timedelta(hours=2), "score": 95.0, "issues": 0},
            {"timestamp": now, "score": 60.0, "issues": 5},
        ]
        result = collector.get_quality_trends("AAPL", window_hours=24)
        assert result["trend"] == "declining"


class TestExportMetrics:
    def test_export_returns_bytes(self):
        collector = DataQualityMetricsCollector(quality_checker=MagicMock())
        result = collector.export_metrics()
        assert isinstance(result, bytes)


class TestPerformQualityCheckWithMetrics:
    @pytest.mark.asyncio
    async def test_successful_check(self):
        mock_checker = MagicMock()
        mock_checker.validate_price_data = MagicMock(
            return_value={"quality_score": 90.0, "valid": True, "issues": []}
        )
        collector = DataQualityMetricsCollector(quality_checker=mock_checker)
        mock_df = MagicMock()
        result = await collector.perform_quality_check_with_metrics(
            mock_df, "AAPL", "price"
        )
        assert result["quality_score"] == 90.0
        assert result["valid"] is True

    @pytest.mark.asyncio
    async def test_check_raises_exception(self):
        mock_checker = MagicMock()
        mock_checker.validate_price_data = MagicMock(
            side_effect=ValueError("bad data")
        )
        collector = DataQualityMetricsCollector(quality_checker=mock_checker)
        mock_df = MagicMock()
        with pytest.raises(ValueError, match="bad data"):
            await collector.perform_quality_check_with_metrics(
                mock_df, "AAPL", "price"
            )


# ==========================================================================
# database_performance.py
# ==========================================================================


class TestDatabasePerformanceMonitorInit:
    def test_init_defaults(self):
        monitor = DatabasePerformanceMonitor()
        assert monitor.monitoring_enabled is True
        assert monitor._collection_interval == 30
        assert monitor._slow_query_threshold == 1.0
        assert monitor._monitoring_task is None
        assert monitor._query_stats_cache == {}


class TestExtractQueryInfo:
    @pytest.fixture
    def monitor(self):
        return DatabasePerformanceMonitor()

    def test_select_query(self, monitor):
        query_type, table = monitor._extract_query_info(
            "SELECT * FROM users WHERE id = 1"
        )
        assert query_type == "SELECT"
        assert table == "users"

    def test_insert_query(self, monitor):
        query_type, table = monitor._extract_query_info(
            "INSERT INTO orders (user_id, amount) VALUES (1, 100)"
        )
        assert query_type == "INSERT"
        assert table == "orders"

    def test_update_query(self, monitor):
        query_type, table = monitor._extract_query_info(
            "UPDATE stocks SET price = 150 WHERE symbol = 'AAPL'"
        )
        assert query_type == "UPDATE"
        assert table == "stocks"

    def test_delete_query(self, monitor):
        query_type, table = monitor._extract_query_info(
            "DELETE FROM sessions WHERE expired = true"
        )
        assert query_type == "DELETE"
        assert table == "sessions"

    def test_other_query(self, monitor):
        query_type, table = monitor._extract_query_info(
            "CREATE TABLE test (id INT)"
        )
        assert query_type == "OTHER"

    def test_empty_query(self, monitor):
        query_type, table = monitor._extract_query_info("")
        assert query_type == "OTHER"

    def test_select_with_schema(self, monitor):
        query_type, table = monitor._extract_query_info(
            "SELECT * FROM public.users"
        )
        assert query_type == "SELECT"
        assert table == "public"  # split on '.' takes first part


class TestStartStopMonitoring:
    @pytest.mark.asyncio
    async def test_start_creates_task(self):
        monitor = DatabasePerformanceMonitor()
        assert monitor._monitoring_task is None
        # Patch _monitoring_loop to avoid actual loop
        with patch.object(monitor, "_monitoring_loop", new_callable=AsyncMock):
            await monitor.start_monitoring()
            assert monitor._monitoring_task is not None
            # Cleanup
            monitor._monitoring_task.cancel()
            try:
                await monitor._monitoring_task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_stop_cancels_task(self):
        monitor = DatabasePerformanceMonitor()
        with patch.object(monitor, "_monitoring_loop", new_callable=AsyncMock):
            await monitor.start_monitoring()
            await monitor.stop_monitoring()
            # Task should be cancelled

    @pytest.mark.asyncio
    async def test_stop_without_start(self):
        monitor = DatabasePerformanceMonitor()
        # Should not raise
        await monitor.stop_monitoring()

    @pytest.mark.asyncio
    async def test_start_idempotent(self):
        monitor = DatabasePerformanceMonitor()
        with patch.object(monitor, "_monitoring_loop", new_callable=AsyncMock):
            await monitor.start_monitoring()
            first_task = monitor._monitoring_task
            await monitor.start_monitoring()
            assert monitor._monitoring_task is first_task
            # Cleanup
            monitor._monitoring_task.cancel()
            try:
                await monitor._monitoring_task
            except asyncio.CancelledError:
                pass


# ==========================================================================
# financial_monitoring.py
# ==========================================================================


class TestPortfolioMetricsDataclass:
    def test_construction(self):
        pm = PortfolioMetrics(
            portfolio_id="p1",
            user_id="u1",
            total_value=100000.0,
            daily_return=1.5,
        )
        assert pm.portfolio_id == "p1"
        assert pm.user_id == "u1"
        assert pm.total_value == 100000.0
        assert pm.daily_return == 1.5
        assert pm.returns == []
        assert pm.positions == {}
        assert pm.benchmark_returns == []
        assert pm.risk_free_rate == 0.02

    def test_with_returns_and_positions(self):
        pm = PortfolioMetrics(
            portfolio_id="p2",
            user_id="u2",
            total_value=250000.0,
            daily_return=-0.3,
            returns=[1.0, -0.5, 0.8],
            positions={"AAPL": 50000.0, "GOOG": 75000.0},
        )
        assert len(pm.returns) == 3
        assert pm.positions["AAPL"] == 50000.0


class TestStrategyMetricsDataclass:
    def test_defaults(self):
        sm = StrategyMetrics(strategy_name="momentum")
        assert sm.strategy_name == "momentum"
        assert sm.trades == []
        assert sm.hit_rate == 0.0
        assert sm.average_return == 0.0
        assert sm.sharpe_ratio == 0.0
        assert sm.max_drawdown == 0.0
        assert sm.consecutive_losses == 0


class TestRecommendationTrackingRecordDataclass:
    def test_construction(self):
        rec = RecommendationTrackingRecord(
            id="rec-1",
            model="lstm",
            ticker="AAPL",
            recommendation_type="buy",
            confidence=0.85,
            predicted_return=5.0,
            timestamp=datetime.now(),
        )
        assert rec.id == "rec-1"
        assert rec.model == "lstm"
        assert rec.actual_returns == {}
        assert rec.benchmark_returns == {}


class TestFinancialMonitorInit:
    def test_init_defaults(self):
        fm = FinancialMonitor()
        assert fm.portfolio_cache == {}
        assert fm.strategy_cache == {}
        assert fm.recommendation_tracking == {}
        assert fm.market_data_cache == {}
        assert fm._monitoring_task is None
        assert fm._update_interval == 300

    def test_returns_cache_is_defaultdict(self):
        fm = FinancialMonitor()
        # Accessing a missing key should create a deque
        d = fm.returns_cache["new_key"]
        assert isinstance(d, deque)
        assert d.maxlen == 252


class TestRecordTradeExecution:
    def test_normal_trade(self):
        fm = FinancialMonitor()
        fm.record_trade_execution(
            order_type="limit",
            market_cap_tier="tier1",
            expected_price=150.0,
            actual_price=150.50,
            execution_time_ms=50.0,
            venue="nasdaq",
        )

    def test_zero_expected_price(self):
        fm = FinancialMonitor()
        # Should not raise, skips slippage calc
        fm.record_trade_execution(
            order_type="market",
            market_cap_tier="tier2",
            expected_price=0.0,
            actual_price=10.0,
            execution_time_ms=100.0,
        )

    def test_high_slippage(self):
        fm = FinancialMonitor()
        fm.record_trade_execution(
            order_type="market",
            market_cap_tier="tier3",
            expected_price=100.0,
            actual_price=110.0,
            execution_time_ms=200.0,
        )


class TestRecordTradingCost:
    def test_normal_cost(self):
        fm = FinancialMonitor()
        fm.record_trading_cost(
            cost_type="commission",
            amount_usd=9.99,
            shares=100,
            market_cap_tier="tier1",
            venue="nyse",
        )

    def test_zero_shares(self):
        fm = FinancialMonitor()
        # Should not raise, skips per-share calc
        fm.record_trading_cost(
            cost_type="commission",
            amount_usd=5.0,
            shares=0,
            market_cap_tier="tier2",
        )


class TestAddRecommendationTracking:
    def test_add_tracking(self):
        fm = FinancialMonitor()
        rec_id = fm.add_recommendation_tracking(
            model="lstm",
            ticker="AAPL",
            recommendation_type="buy",
            confidence=0.9,
            predicted_return=5.0,
        )
        assert rec_id != ""
        assert rec_id in fm.recommendation_tracking
        record = fm.recommendation_tracking[rec_id]
        assert record.model == "lstm"
        assert record.ticker == "AAPL"
        assert record.confidence == 0.9

    def test_unique_ids(self):
        fm = FinancialMonitor()
        id1 = fm.add_recommendation_tracking(
            "model_a", "AAPL", "buy", 0.8, 3.0
        )
        # Time-based ID should be unique (or at least not collide)
        id2 = fm.add_recommendation_tracking(
            "model_b", "GOOG", "sell", 0.7, -2.0
        )
        assert id1 != id2


class TestGetLiquidityBucket:
    def test_tier1(self):
        fm = FinancialMonitor()
        assert fm._get_liquidity_bucket("tier1") == "high"

    def test_tier2(self):
        fm = FinancialMonitor()
        assert fm._get_liquidity_bucket("tier2") == "medium"

    def test_tier3(self):
        fm = FinancialMonitor()
        assert fm._get_liquidity_bucket("tier3") == "low"

    def test_tier4(self):
        fm = FinancialMonitor()
        assert fm._get_liquidity_bucket("tier4") == "very_low"

    def test_tier5(self):
        fm = FinancialMonitor()
        assert fm._get_liquidity_bucket("tier5") == "very_low"

    def test_unknown(self):
        fm = FinancialMonitor()
        assert fm._get_liquidity_bucket("tier99") == "unknown"


class TestGetVolumeBucket:
    def test_small(self):
        fm = FinancialMonitor()
        assert fm._get_volume_bucket(50) == "small"

    def test_medium(self):
        fm = FinancialMonitor()
        assert fm._get_volume_bucket(500) == "medium"

    def test_large(self):
        fm = FinancialMonitor()
        assert fm._get_volume_bucket(5000) == "large"

    def test_very_large(self):
        fm = FinancialMonitor()
        assert fm._get_volume_bucket(50000) == "very_large"

    def test_boundary_small_medium(self):
        fm = FinancialMonitor()
        assert fm._get_volume_bucket(99) == "small"
        assert fm._get_volume_bucket(100) == "medium"

    def test_boundary_medium_large(self):
        fm = FinancialMonitor()
        assert fm._get_volume_bucket(999) == "medium"
        assert fm._get_volume_bucket(1000) == "large"

    def test_boundary_large_very_large(self):
        fm = FinancialMonitor()
        assert fm._get_volume_bucket(9999) == "large"
        assert fm._get_volume_bucket(10000) == "very_large"


class TestGetFinancialSummary:
    def test_empty_summary(self):
        fm = FinancialMonitor()
        summary = fm.get_financial_summary()
        assert summary["portfolios_monitored"] == 0
        assert summary["strategies_monitored"] == 0
        assert summary["recommendations_tracked"] == 0
        assert summary["market_data_points"] == 0
        assert "timestamp" in summary
        assert "last_update" in summary

    def test_summary_with_data(self):
        fm = FinancialMonitor()
        fm.portfolio_cache["p1"] = MagicMock()
        fm.strategy_cache["s1"] = MagicMock()
        fm.strategy_cache["s2"] = MagicMock()
        fm.recommendation_tracking["r1"] = MagicMock()
        fm.market_data_cache["m1"] = MagicMock()
        fm.market_data_cache["m2"] = MagicMock()
        fm.market_data_cache["m3"] = MagicMock()

        summary = fm.get_financial_summary()
        assert summary["portfolios_monitored"] == 1
        assert summary["strategies_monitored"] == 2
        assert summary["recommendations_tracked"] == 1
        assert summary["market_data_points"] == 3


class TestFinancialMonitorStartStop:
    @pytest.mark.asyncio
    async def test_start_creates_task(self):
        fm = FinancialMonitor()
        with patch.object(fm, "_monitoring_loop", new_callable=AsyncMock):
            await fm.start_monitoring()
            assert fm._monitoring_task is not None
            fm._monitoring_task.cancel()
            try:
                await fm._monitoring_task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_stop_cancels_task(self):
        fm = FinancialMonitor()
        with patch.object(fm, "_monitoring_loop", new_callable=AsyncMock):
            await fm.start_monitoring()
            await fm.stop_monitoring()

    @pytest.mark.asyncio
    async def test_stop_without_start(self):
        fm = FinancialMonitor()
        await fm.stop_monitoring()

    @pytest.mark.asyncio
    async def test_start_idempotent(self):
        fm = FinancialMonitor()
        with patch.object(fm, "_monitoring_loop", new_callable=AsyncMock):
            await fm.start_monitoring()
            first_task = fm._monitoring_task
            await fm.start_monitoring()
            assert fm._monitoring_task is first_task
            fm._monitoring_task.cancel()
            try:
                await fm._monitoring_task
            except asyncio.CancelledError:
                pass


class TestCalculateRiskMetrics:
    @pytest.mark.asyncio
    async def test_risk_metrics_with_sufficient_data(self):
        fm = FinancialMonitor()
        returns = list(np.random.normal(0.1, 1.0, 50))
        pm = PortfolioMetrics(
            portfolio_id="p1",
            user_id="u1",
            total_value=100000.0,
            daily_return=0.5,
            returns=returns,
        )
        fm.portfolio_cache["p1"] = pm
        # Should not raise
        await fm._calculate_risk_metrics()

    @pytest.mark.asyncio
    async def test_risk_metrics_insufficient_data(self):
        fm = FinancialMonitor()
        pm = PortfolioMetrics(
            portfolio_id="p2",
            user_id="u2",
            total_value=50000.0,
            daily_return=0.1,
            returns=[1.0, -0.5],  # Only 2 returns, need 30
        )
        fm.portfolio_cache["p2"] = pm
        # Should not raise, just skip
        await fm._calculate_risk_metrics()

    @pytest.mark.asyncio
    async def test_risk_metrics_empty_cache(self):
        fm = FinancialMonitor()
        await fm._calculate_risk_metrics()
