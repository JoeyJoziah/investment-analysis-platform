"""
Unit tests for monitoring modules: application_monitoring.py, auto_scaler.py,
data_quality_dashboard.py.

Uses importlib.util.spec_from_file_location with full dependency stubbing
to avoid import-time failures from prometheus_client, docker, psutil, etc.
"""

import asyncio
import importlib
import importlib.util
import sys
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
from dataclasses import dataclass
from enum import Enum as _Enum

import pytest

# ---------------------------------------------------------------------------
# Stub heavy dependencies BEFORE importing monitoring modules
# ---------------------------------------------------------------------------

# --- Prometheus stub ---
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
sys.modules.setdefault("prometheus_client", _prom_mock)

# --- backend.config.settings ---
_mock_settings = MagicMock()
_mock_settings.VERSION = "1.0.0"
_mock_settings.ENVIRONMENT = "test"
_mock_settings.DEBUG = True
sys.modules.setdefault("backend.config.settings", MagicMock(settings=_mock_settings))

# --- backend.config.monitoring_config ---
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

# --- backend.utils.structured_logging ---
_mock_structured_logging = MagicMock()
_mock_structured_logging.get_structured_logger = MagicMock(return_value=MagicMock())
sys.modules.setdefault("backend.utils.structured_logging", _mock_structured_logging)

# --- docker ---
sys.modules.setdefault("docker", MagicMock())

# --- psutil ---
_mock_psutil = MagicMock()
_mock_psutil.disk_usage = MagicMock(return_value=MagicMock(percent=45.0))
sys.modules.setdefault("psutil", _mock_psutil)

# --- httpx ---
sys.modules.setdefault("httpx", MagicMock())

# --- sqlalchemy (for auto_scaler) ---
sys.modules.setdefault("sqlalchemy", MagicMock())
sys.modules.setdefault("sqlalchemy.ext", MagicMock())
sys.modules.setdefault("sqlalchemy.ext.asyncio", MagicMock())
sys.modules.setdefault("sqlalchemy.orm", MagicMock())

# --- pandas ---
sys.modules.setdefault("pandas", MagicMock())

# --- numpy (provide a real-enough stub for data_quality_dashboard) ---
import numpy as np  # numpy is installed

# --- scipy ---
_mock_scipy = MagicMock()
_mock_scipy.stats = MagicMock()
sys.modules.setdefault("scipy", _mock_scipy)
sys.modules.setdefault("scipy.stats", _mock_scipy.stats)

# --- backend.utils.cache ---
_mock_cache_mod = MagicMock()
_mock_cache_manager = MagicMock()
_mock_cache_manager.get = AsyncMock(return_value=None)
_mock_cache_manager.set = AsyncMock()
_mock_cache_mod.CacheManager = MagicMock(return_value=_mock_cache_manager)
_mock_cache_mod.get_redis = AsyncMock(return_value=MagicMock(
    set=AsyncMock(), get=AsyncMock(return_value=None),
))
sys.modules.setdefault("backend.utils.cache", _mock_cache_mod)

# --- backend.utils.database ---
_mock_database_mod = MagicMock()
sys.modules.setdefault("backend.utils.database", _mock_database_mod)

# --- backend.monitoring.real_time_alerts ---
class _AlertSeverity(_Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

class _AlertCategory(_Enum):
    SYSTEM = "system"
    DATA_QUALITY = "data_quality"
    PERFORMANCE = "performance"

_mock_rta = MagicMock()
_mock_rta.AlertSeverity = _AlertSeverity
_mock_rta.AlertCategory = _AlertCategory
_mock_rta.RealTimeAlertManager = MagicMock
# Save originals and force-set backend.* stubs (restored after importlib loads)
_saved_backend_mods_a2 = {}
_backend_stubs_a2 = {
    "backend.monitoring.real_time_alerts": _mock_rta,
    "backend.utils.enhanced_cost_monitor": MagicMock(),
    "backend.monitoring.metrics_collector": MagicMock(),
}

# --- backend.utils.enhanced_cost_monitor ---
class _StockPriority(_Enum):
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    MINIMAL = 5

_backend_stubs_a2["backend.utils.enhanced_cost_monitor"].StockPriority = _StockPriority
_backend_stubs_a2["backend.monitoring.metrics_collector"].model_inference_time = _FakeMetric()

for _modname, _stub in _backend_stubs_a2.items():
    _saved_backend_mods_a2[_modname] = sys.modules.get(_modname)
    sys.modules[_modname] = _stub

# --- asyncpg / aiohttp ---
sys.modules.setdefault("asyncpg", MagicMock())
sys.modules.setdefault("aiohttp", MagicMock())

# ---------------------------------------------------------------------------
# Import modules via importlib
# ---------------------------------------------------------------------------

_mon_dir = Path(__file__).resolve().parents[2] / "monitoring"

# --- application_monitoring.py ---
_am_spec = importlib.util.spec_from_file_location(
    "application_monitoring", _mon_dir / "application_monitoring.py"
)
_am = importlib.util.module_from_spec(_am_spec)
_am_spec.loader.exec_module(_am)

ApplicationMonitor = _am.ApplicationMonitor
StockProcessingMetrics = _am.StockProcessingMetrics
RecommendationMetrics = _am.RecommendationMetrics
StockProcessingContext = _am.StockProcessingContext
RecommendationContext = _am.RecommendationContext
AnalysisContext = _am.AnalysisContext
ModelOperationContext = _am.ModelOperationContext

# --- auto_scaler.py ---
# auto_scaler calls asyncio.create_task in __init__.start_monitoring()
# We patch asyncio.create_task to avoid event-loop errors at import time.
with patch("asyncio.create_task", return_value=MagicMock()):
    _as_spec = importlib.util.spec_from_file_location(
        "auto_scaler", _mon_dir / "auto_scaler.py"
    )
    _as_mod = importlib.util.module_from_spec(_as_spec)
    _as_spec.loader.exec_module(_as_mod)

ScalingAction = _as_mod.ScalingAction
ScalingRule = _as_mod.ScalingRule
ResourceMetrics = _as_mod.ResourceMetrics
CostOptimizedAutoScaler = _as_mod.CostOptimizedAutoScaler

# --- data_quality_dashboard.py ---
_dq_spec = importlib.util.spec_from_file_location(
    "data_quality_dashboard", _mon_dir / "data_quality_dashboard.py"
)
_dq = importlib.util.module_from_spec(_dq_spec)
_dq_spec.loader.exec_module(_dq)

DataQualityStatus = _dq.DataQualityStatus
DataSourceType = _dq.DataSourceType
DataQualityMetric = _dq.DataQualityMetric
DataSourceQuality = _dq.DataSourceQuality
DataQualityMonitor = _dq.DataQualityMonitor

# Restore all backend.* modules we temporarily stubbed.
for _modname, _orig_mod in _saved_backend_mods_a2.items():
    if _orig_mod is not None:
        sys.modules[_modname] = _orig_mod
    else:
        sys.modules.pop(_modname, None)


# ==========================================================================
# application_monitoring.py
# ==========================================================================


class TestStockProcessingMetrics:
    def test_construction(self):
        m = StockProcessingMetrics(
            ticker="AAPL",
            tier="tier1",
            start_time=datetime.now(),
            stages_completed=["fetch", "transform"],
            api_calls_made=5,
            cache_hits=3,
            cache_misses=2,
            data_quality_scores={"price": 0.95},
            errors=[],
        )
        assert m.ticker == "AAPL"
        assert m.tier == "tier1"
        assert m.api_calls_made == 5
        assert m.cache_hits == 3
        assert m.cache_misses == 2
        assert len(m.stages_completed) == 2

    def test_with_errors(self):
        m = StockProcessingMetrics(
            ticker="MSFT",
            tier="tier2",
            start_time=datetime.now(),
            stages_completed=["fetch"],
            api_calls_made=1,
            cache_hits=0,
            cache_misses=1,
            data_quality_scores={},
            errors=[{"type": "timeout", "stage": "fetch", "source": "api"}],
        )
        assert len(m.errors) == 1
        assert m.errors[0]["type"] == "timeout"


class TestRecommendationMetrics:
    def test_construction(self):
        m = RecommendationMetrics(
            model_name="lstm",
            generation_time=1.5,
            input_features=42,
            confidence_score=0.87,
            recommendation_type="buy",
            complexity_score=0.65,
        )
        assert m.model_name == "lstm"
        assert m.confidence_score == 0.87
        assert m.recommendation_type == "buy"


class TestApplicationMonitor:
    @pytest.fixture
    def monitor(self):
        return ApplicationMonitor()

    def test_init_defaults(self, monitor):
        assert monitor.stock_processing_sessions == {}
        assert isinstance(monitor.recommendation_history, deque)
        assert monitor.active_processing_count == 0
        assert monitor._monitoring_task is None

    def test_record_stock_processing_complete(self, monitor):
        metrics = StockProcessingMetrics(
            ticker="GOOG",
            tier="tier1",
            start_time=datetime.now() - timedelta(seconds=10),
            stages_completed=["fetch", "transform", "store"],
            api_calls_made=3,
            cache_hits=2,
            cache_misses=1,
            data_quality_scores={"price": 0.98, "volume": 0.95},
            errors=[],
        )
        # Should not raise
        monitor.record_stock_processing_complete(metrics)

    def test_record_stock_processing_with_errors(self, monitor):
        metrics = StockProcessingMetrics(
            ticker="TSLA",
            tier="tier2",
            start_time=datetime.now() - timedelta(seconds=5),
            stages_completed=["fetch"],
            api_calls_made=1,
            cache_hits=0,
            cache_misses=1,
            data_quality_scores={},
            errors=[
                {"type": "rate_limit", "stage": "fetch", "source": "alphavantage"},
                {"type": "parse_error", "stage": "transform", "source": "system"},
            ],
        )
        monitor.record_stock_processing_complete(metrics)

    def test_record_recommendation_metrics(self, monitor):
        metrics = RecommendationMetrics(
            model_name="gradient_boost",
            generation_time=0.8,
            input_features=50,
            confidence_score=0.92,
            recommendation_type="sell",
            complexity_score=0.75,
        )
        monitor.record_recommendation_metrics(metrics)
        assert len(monitor.recommendation_history) == 1
        assert monitor.recommendation_history[0]["model"] == "gradient_boost"
        assert monitor.recommendation_history[0]["confidence"] == 0.92

    def test_record_recommendation_low_complexity(self, monitor):
        metrics = RecommendationMetrics(
            model_name="simple_avg",
            generation_time=0.1,
            input_features=5,
            confidence_score=0.55,
            recommendation_type="hold",
            complexity_score=0.2,
        )
        monitor.record_recommendation_metrics(metrics)

    def test_record_data_validation_failure(self, monitor):
        # Should not raise
        monitor.record_data_validation_failure("range_check", "price_data", "polygon")

    def test_get_monitoring_summary_empty(self, monitor):
        summary = monitor.get_monitoring_summary()
        assert summary["active_processing"] == 0
        assert summary["recent_recommendations"] == 0
        assert summary["average_confidence"] == 0

    def test_get_monitoring_summary_with_recommendations(self, monitor):
        for i in range(5):
            metrics = RecommendationMetrics(
                model_name="test_model",
                generation_time=0.5,
                input_features=10,
                confidence_score=0.7 + i * 0.05,
                recommendation_type="buy",
                complexity_score=0.5,
            )
            monitor.record_recommendation_metrics(metrics)

        summary = monitor.get_monitoring_summary()
        assert summary["recent_recommendations"] == 5
        assert summary["average_confidence"] > 0

    def test_track_stock_processing_returns_context(self, monitor):
        ctx = monitor.track_stock_processing("AAPL", "tier1")
        assert isinstance(ctx, StockProcessingContext)

    def test_track_recommendation_generation_returns_context(self, monitor):
        ctx = monitor.track_recommendation_generation("lstm")
        assert isinstance(ctx, RecommendationContext)

    def test_track_analysis_returns_context(self, monitor):
        ctx = monitor.track_analysis_operation("technical_rsi", "high")
        assert isinstance(ctx, AnalysisContext)

    def test_track_model_operation_returns_context(self, monitor):
        ctx = monitor.track_model_operation("xgboost", "training")
        assert isinstance(ctx, ModelOperationContext)

    @pytest.mark.asyncio
    async def test_stock_processing_context(self, monitor):
        ctx = monitor.track_stock_processing("NVDA", "tier1")
        async with ctx as metrics:
            metrics.stages_completed.append("fetch")
            metrics.api_calls_made = 2
            metrics.data_quality_scores["price"] = 0.97
        assert "NVDA" not in monitor.stock_processing_sessions
        assert monitor.active_processing_count == 0

    @pytest.mark.asyncio
    async def test_stock_processing_context_increments_count(self, monitor):
        ctx = monitor.track_stock_processing("AMD", "tier2")
        async with ctx as metrics:
            assert monitor.active_processing_count == 1
        assert monitor.active_processing_count == 0

    @pytest.mark.asyncio
    async def test_analysis_context_technical(self, monitor):
        ctx = AnalysisContext(monitor, "technical_macd", "medium")
        async with ctx:
            pass  # duration recorded in __aexit__

    @pytest.mark.asyncio
    async def test_analysis_context_fundamental(self, monitor):
        ctx = AnalysisContext(monitor, "fundamental_eps", "high")
        async with ctx:
            pass

    @pytest.mark.asyncio
    async def test_analysis_context_sentiment(self, monitor):
        ctx = AnalysisContext(monitor, "sentiment_twitter", "low")
        async with ctx:
            pass

    @pytest.mark.asyncio
    async def test_model_operation_context_training(self, monitor):
        ctx = ModelOperationContext(monitor, "lstm_v2", "training")
        async with ctx:
            pass

    @pytest.mark.asyncio
    async def test_model_operation_context_inference(self, monitor):
        ctx = ModelOperationContext(monitor, "lstm_v2", "inference")
        async with ctx:
            pass


# ==========================================================================
# auto_scaler.py
# ==========================================================================


class TestScalingAction:
    def test_enum_values(self):
        assert ScalingAction.SCALE_UP.value == "scale_up"
        assert ScalingAction.SCALE_DOWN.value == "scale_down"
        assert ScalingAction.MAINTAIN.value == "maintain"

    def test_all_members(self):
        assert len(ScalingAction) == 3


class TestScalingRule:
    def test_defaults(self):
        rule = ScalingRule(service_name="test_service")
        assert rule.min_replicas == 1
        assert rule.max_replicas == 3
        assert rule.target_cpu_percent == 70.0
        assert rule.target_memory_percent == 80.0
        assert rule.target_response_time_ms == 1000.0
        assert rule.scale_up_threshold == 0.8
        assert rule.scale_down_threshold == 0.3
        assert rule.cooldown_seconds == 300
        assert rule.cost_weight == 1.0
        assert rule.weekend_scale_factor == 0.5
        assert rule.current_replicas == 1
        assert rule.consecutive_violations == 0
        assert rule.last_scale_action is None

    def test_custom_values(self):
        rule = ScalingRule(
            service_name="custom",
            min_replicas=2,
            max_replicas=10,
            target_cpu_percent=50.0,
            target_memory_percent=60.0,
            cooldown_seconds=120,
            cost_weight=5.0,
        )
        assert rule.min_replicas == 2
        assert rule.max_replicas == 10
        assert rule.target_cpu_percent == 50.0
        assert rule.cost_weight == 5.0

    def test_peak_hours_default(self):
        rule = ScalingRule(service_name="svc")
        assert rule.peak_hours == list(range(9, 17))


class TestResourceMetrics:
    def test_construction(self):
        m = ResourceMetrics(
            cpu_percent=55.0,
            memory_percent=70.0,
            disk_percent=45.0,
            response_time_ms=250.0,
            error_rate_percent=0.5,
            active_connections=42,
            queue_length=10,
            timestamp=datetime.now(timezone.utc),
        )
        assert m.cpu_percent == 55.0
        assert m.memory_percent == 70.0
        assert m.queue_length == 10


class TestCostOptimizedAutoScaler:
    @pytest.fixture
    def scaler(self):
        mock_docker = MagicMock()
        mock_db = MagicMock()
        with patch("asyncio.create_task", return_value=MagicMock()):
            s = CostOptimizedAutoScaler(mock_docker, mock_db)
        return s

    def test_init_creates_default_rules(self, scaler):
        assert "backend" in scaler.scaling_rules
        assert "frontend" in scaler.scaling_rules
        assert "worker" in scaler.scaling_rules

    def test_backend_rule_config(self, scaler):
        rule = scaler.scaling_rules["backend"]
        assert rule.service_name == "investment_api_prod"
        assert rule.max_replicas == 4
        assert rule.cost_weight == 3.0

    def test_frontend_rule_config(self, scaler):
        rule = scaler.scaling_rules["frontend"]
        assert rule.service_name == "investment_web_prod"
        assert rule.max_replicas == 3
        assert rule.cooldown_seconds == 120

    def test_worker_rule_config(self, scaler):
        rule = scaler.scaling_rules["worker"]
        assert rule.service_name == "investment_worker_prod"
        assert rule.target_cpu_percent == 80.0

    def test_daily_budget(self, scaler):
        assert scaler.daily_budget_usd == pytest.approx(1.67)

    def test_cost_per_replica(self, scaler):
        assert scaler.cost_per_replica_hour["backend"] == 0.02
        assert scaler.cost_per_replica_hour["database"] == 0.05
        assert scaler.cost_per_replica_hour["redis"] == 0.01

    def test_calculate_daily_cost_projection(self, scaler):
        cost = scaler._calculate_daily_cost_projection()
        # Should include all 3 services at 1 replica each + database + redis fixed
        assert cost > 0

    def test_scaling_decision_maintain(self, scaler):
        rule = ScalingRule(service_name="test_svc", current_replicas=1)
        metrics = ResourceMetrics(
            cpu_percent=30.0,
            memory_percent=40.0,
            disk_percent=30.0,
            response_time_ms=200.0,
            error_rate_percent=0.0,
            active_connections=5,
            queue_length=0,
            timestamp=datetime.now(timezone.utc),
        )
        action, target = scaler._calculate_scaling_decision(rule, metrics)
        assert action == ScalingAction.MAINTAIN

    def test_scaling_decision_scale_up_high_cpu(self, scaler):
        rule = ScalingRule(
            service_name="test_svc",
            current_replicas=1,
            max_replicas=3,
            target_cpu_percent=70.0,
            scale_up_threshold=0.8,
        )
        metrics = ResourceMetrics(
            cpu_percent=90.0,  # 90/70 = 1.28 > 0.8 threshold
            memory_percent=40.0,
            disk_percent=30.0,
            response_time_ms=200.0,
            error_rate_percent=0.0,
            active_connections=5,
            queue_length=0,
            timestamp=datetime.now(timezone.utc),
        )
        # Force peak hour for reliable test
        with patch.object(scaler, "_is_peak_hour", return_value=True):
            with patch.object(scaler, "_calculate_daily_cost_projection", return_value=0.5):
                action, target = scaler._calculate_scaling_decision(rule, metrics)
        assert action == ScalingAction.SCALE_UP
        assert target == 2

    def test_scaling_decision_scale_down_low_usage(self, scaler):
        rule = ScalingRule(
            service_name="test_svc",
            current_replicas=3,
            min_replicas=1,
            target_cpu_percent=70.0,
            target_memory_percent=80.0,
            target_response_time_ms=1000.0,
            scale_down_threshold=0.3,
        )
        metrics = ResourceMetrics(
            cpu_percent=10.0,  # 10/70 = 0.14 < 0.3
            memory_percent=15.0,  # 15/80 = 0.19 < 0.3
            disk_percent=20.0,
            response_time_ms=100.0,  # 100/1000 = 0.1 < 0.3
            error_rate_percent=0.0,
            active_connections=1,
            queue_length=0,
            timestamp=datetime.now(timezone.utc),
        )
        with patch.object(scaler, "_is_peak_hour", return_value=True):
            action, target = scaler._calculate_scaling_decision(rule, metrics)
        assert action == ScalingAction.SCALE_DOWN
        assert target == 2

    def test_scaling_at_max_replicas_stays_maintain(self, scaler):
        rule = ScalingRule(
            service_name="test_svc",
            current_replicas=3,
            max_replicas=3,
            target_cpu_percent=70.0,
            scale_up_threshold=0.8,
        )
        metrics = ResourceMetrics(
            cpu_percent=95.0,
            memory_percent=90.0,
            disk_percent=50.0,
            response_time_ms=2000.0,
            error_rate_percent=1.0,
            active_connections=100,
            queue_length=0,
            timestamp=datetime.now(timezone.utc),
        )
        with patch.object(scaler, "_is_peak_hour", return_value=True):
            with patch.object(scaler, "_calculate_daily_cost_projection", return_value=0.5):
                action, target = scaler._calculate_scaling_decision(rule, metrics)
        assert action == ScalingAction.MAINTAIN
        assert target == 3

    def test_scaling_at_min_replicas_stays_maintain(self, scaler):
        rule = ScalingRule(
            service_name="test_svc",
            current_replicas=1,
            min_replicas=1,
            target_cpu_percent=70.0,
            scale_down_threshold=0.3,
        )
        metrics = ResourceMetrics(
            cpu_percent=5.0,
            memory_percent=10.0,
            disk_percent=20.0,
            response_time_ms=50.0,
            error_rate_percent=0.0,
            active_connections=0,
            queue_length=0,
            timestamp=datetime.now(timezone.utc),
        )
        with patch.object(scaler, "_is_peak_hour", return_value=True):
            action, target = scaler._calculate_scaling_decision(rule, metrics)
        assert action == ScalingAction.MAINTAIN
        assert target == 1

    def test_cost_constraint_prevents_scale_up(self, scaler):
        rule = ScalingRule(
            service_name="test_svc",
            current_replicas=1,
            max_replicas=5,
            target_cpu_percent=70.0,
            scale_up_threshold=0.8,
        )
        metrics = ResourceMetrics(
            cpu_percent=90.0,
            memory_percent=85.0,
            disk_percent=50.0,
            response_time_ms=1500.0,
            error_rate_percent=0.5,
            active_connections=50,
            queue_length=0,
            timestamp=datetime.now(timezone.utc),
        )
        # Over budget - cost_constraint_factor = 0.1, prevents scale_up
        with patch.object(scaler, "_is_peak_hour", return_value=True):
            with patch.object(scaler, "_calculate_daily_cost_projection", return_value=2.0):
                action, target = scaler._calculate_scaling_decision(rule, metrics)
        assert action == ScalingAction.MAINTAIN

    def test_worker_queue_score(self, scaler):
        rule = ScalingRule(
            service_name="investment_worker_prod",
            current_replicas=1,
            max_replicas=5,
            target_cpu_percent=80.0,
            scale_up_threshold=0.8,
        )
        metrics = ResourceMetrics(
            cpu_percent=50.0,
            memory_percent=50.0,
            disk_percent=30.0,
            response_time_ms=100.0,
            error_rate_percent=0.0,
            active_connections=0,
            queue_length=50,  # 50 tasks, ideal 5 replicas, score = 5/1 = 5.0
            timestamp=datetime.now(timezone.utc),
        )
        with patch.object(scaler, "_is_peak_hour", return_value=True):
            with patch.object(scaler, "_calculate_daily_cost_projection", return_value=0.5):
                action, target = scaler._calculate_scaling_decision(rule, metrics)
        assert action == ScalingAction.SCALE_UP

    @pytest.mark.asyncio
    async def test_execute_scaling_maintain(self, scaler):
        rule = ScalingRule(service_name="test_svc", current_replicas=2)
        result = await scaler._execute_scaling_action(
            rule, ScalingAction.MAINTAIN, 2
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_execute_scaling_cooldown(self, scaler):
        rule = ScalingRule(
            service_name="test_svc",
            current_replicas=1,
            cooldown_seconds=300,
            last_scale_action=datetime.now(timezone.utc),  # just now
        )
        result = await scaler._execute_scaling_action(
            rule, ScalingAction.SCALE_UP, 2
        )
        assert result is False  # in cooldown

    def test_get_scaling_status(self, scaler):
        status = scaler.get_scaling_status()
        assert "daily_cost_projection" in status
        assert "daily_budget" in status
        assert "budget_utilization_percent" in status
        assert "services" in status
        assert "backend" in status["services"]
        assert "frontend" in status["services"]
        assert "worker" in status["services"]

    def test_is_peak_hour_returns_bool(self, scaler):
        # _is_peak_hour uses datetime.now() internally; just verify it returns bool
        result = scaler._is_peak_hour()
        assert isinstance(result, bool)

    def test_is_peak_hour_weekday_business(self, scaler):
        # Patch the module-level datetime used by auto_scaler
        mock_now = MagicMock()
        mock_now.hour = 10
        mock_now.weekday.return_value = 2  # Wednesday
        with patch.object(_as_mod, "datetime") as mock_dt:
            mock_dt.now.return_value = mock_now
            assert scaler._is_peak_hour() is True

    def test_is_peak_hour_weekend(self, scaler):
        mock_now = MagicMock()
        mock_now.hour = 10
        mock_now.weekday.return_value = 5  # Saturday
        with patch.object(_as_mod, "datetime") as mock_dt:
            mock_dt.now.return_value = mock_now
            assert scaler._is_peak_hour() is False

    def test_is_peak_hour_off_hours(self, scaler):
        mock_now = MagicMock()
        mock_now.hour = 22  # 10 PM
        mock_now.weekday.return_value = 1  # Tuesday
        with patch.object(_as_mod, "datetime") as mock_dt:
            mock_dt.now.return_value = mock_now
            assert scaler._is_peak_hour() is False

    def test_stop_monitoring(self, scaler):
        scaler._monitor_task = MagicMock()
        scaler.stop_monitoring()
        assert scaler._monitor_task is None


# ==========================================================================
# data_quality_dashboard.py
# ==========================================================================


class TestDataQualityStatus:
    def test_enum_values(self):
        assert DataQualityStatus.EXCELLENT.value == "excellent"
        assert DataQualityStatus.GOOD.value == "good"
        assert DataQualityStatus.FAIR.value == "fair"
        assert DataQualityStatus.POOR.value == "poor"
        assert DataQualityStatus.CRITICAL.value == "critical"

    def test_all_members(self):
        assert len(DataQualityStatus) == 5


class TestDataSourceType:
    def test_enum_values(self):
        assert DataSourceType.PRICE_DATA.value == "price_data"
        assert DataSourceType.VOLUME_DATA.value == "volume_data"
        assert DataSourceType.FUNDAMENTAL_DATA.value == "fundamental_data"
        assert DataSourceType.NEWS_DATA.value == "news_data"
        assert DataSourceType.SOCIAL_DATA.value == "social_data"
        assert DataSourceType.OPTIONS_DATA.value == "options_data"
        assert DataSourceType.INSIDER_DATA.value == "insider_data"
        assert DataSourceType.ECONOMIC_DATA.value == "economic_data"

    def test_all_members(self):
        assert len(DataSourceType) == 8


class TestDataQualityMetric:
    def test_construction(self):
        m = DataQualityMetric(
            metric_name="freshness",
            current_value=0.95,
            target_value=1.0,
            threshold_warning=0.8,
            threshold_critical=0.5,
            status=DataQualityStatus.EXCELLENT,
            last_updated=datetime.now(),
        )
        assert m.metric_name == "freshness"
        assert m.current_value == 0.95
        assert m.trend_7d is None
        assert m.trend_30d is None

    def test_with_trends(self):
        m = DataQualityMetric(
            metric_name="completeness",
            current_value=0.88,
            target_value=1.0,
            threshold_warning=0.9,
            threshold_critical=0.7,
            status=DataQualityStatus.GOOD,
            last_updated=datetime.now(),
            trend_7d=2.5,
            trend_30d=-1.0,
        )
        assert m.trend_7d == 2.5
        assert m.trend_30d == -1.0


class TestDataQualityMonitor:
    @pytest.fixture
    def monitor(self):
        config = {
            "quality_thresholds": {
                "excellent": 0.95,
                "good": 0.85,
                "fair": 0.70,
                "poor": 0.50,
            },
            "data_sources": {
                "prices": {
                    "type": "price_data",
                    "table_name": "stock_prices",
                    "timestamp_column": "created_at",
                    "expected_frequency_hours": 1,
                    "required_columns": ["close", "volume"],
                },
            },
        }
        return DataQualityMonitor(config)

    def test_init(self, monitor):
        assert monitor.thresholds["excellent"] == 0.95
        assert "prices" in monitor.data_sources
        assert monitor.quality_cache == {}
        assert monitor.quality_history == {}

    def test_calculate_status_excellent(self, monitor):
        assert monitor._calculate_status(0.98) == DataQualityStatus.EXCELLENT

    def test_calculate_status_good(self, monitor):
        assert monitor._calculate_status(0.90) == DataQualityStatus.GOOD

    def test_calculate_status_fair(self, monitor):
        assert monitor._calculate_status(0.75) == DataQualityStatus.FAIR

    def test_calculate_status_poor(self, monitor):
        assert monitor._calculate_status(0.55) == DataQualityStatus.POOR

    def test_calculate_status_critical(self, monitor):
        assert monitor._calculate_status(0.40) == DataQualityStatus.CRITICAL

    def test_calculate_status_boundary_excellent(self, monitor):
        assert monitor._calculate_status(0.95) == DataQualityStatus.EXCELLENT

    def test_calculate_status_boundary_good(self, monitor):
        assert monitor._calculate_status(0.85) == DataQualityStatus.GOOD

    def test_identify_issues_critical(self, monitor):
        metrics = {
            "freshness": DataQualityMetric(
                metric_name="freshness",
                current_value=0.30,
                target_value=1.0,
                threshold_warning=0.8,
                threshold_critical=0.5,
                status=DataQualityStatus.CRITICAL,
                last_updated=datetime.now(),
            ),
        }
        issues = monitor._identify_issues(metrics)
        assert len(issues) == 1
        assert "Critical freshness" in issues[0]

    def test_identify_issues_poor(self, monitor):
        metrics = {
            "completeness": DataQualityMetric(
                metric_name="completeness",
                current_value=0.55,
                target_value=1.0,
                threshold_warning=0.9,
                threshold_critical=0.7,
                status=DataQualityStatus.POOR,
                last_updated=datetime.now(),
            ),
        }
        issues = monitor._identify_issues(metrics)
        assert len(issues) == 1
        assert "Poor completeness" in issues[0]

    def test_identify_issues_warning(self, monitor):
        metrics = {
            "accuracy": DataQualityMetric(
                metric_name="accuracy",
                current_value=0.88,
                target_value=1.0,
                threshold_warning=0.95,
                threshold_critical=0.85,
                status=DataQualityStatus.GOOD,
                last_updated=datetime.now(),
            ),
        }
        issues = monitor._identify_issues(metrics)
        assert len(issues) == 1
        assert "Accuracy below warning threshold" in issues[0]

    def test_identify_issues_none(self, monitor):
        metrics = {
            "freshness": DataQualityMetric(
                metric_name="freshness",
                current_value=0.99,
                target_value=1.0,
                threshold_warning=0.8,
                threshold_critical=0.5,
                status=DataQualityStatus.EXCELLENT,
                last_updated=datetime.now(),
            ),
        }
        issues = monitor._identify_issues(metrics)
        assert len(issues) == 0

    def test_generate_recommendations_freshness(self, monitor):
        metrics = {
            "freshness": DataQualityMetric(
                metric_name="freshness",
                current_value=0.60,
                target_value=1.0,
                threshold_warning=0.8,
                threshold_critical=0.5,
                status=DataQualityStatus.POOR,
                last_updated=datetime.now(),
            ),
        }
        recs = monitor._generate_recommendations(metrics, DataSourceType.PRICE_DATA)
        assert any("data collection frequency" in r or "data pipeline" in r for r in recs)

    def test_generate_recommendations_completeness(self, monitor):
        metrics = {
            "completeness": DataQualityMetric(
                metric_name="completeness",
                current_value=0.70,
                target_value=1.0,
                threshold_warning=0.9,
                threshold_critical=0.7,
                status=DataQualityStatus.FAIR,
                last_updated=datetime.now(),
            ),
        }
        recs = monitor._generate_recommendations(metrics, DataSourceType.VOLUME_DATA)
        assert any("missing values" in r for r in recs)

    def test_generate_recommendations_price_data_redundancy(self, monitor):
        metrics = {
            "accuracy": DataQualityMetric(
                metric_name="accuracy",
                current_value=0.80,
                target_value=1.0,
                threshold_warning=0.95,
                threshold_critical=0.85,
                status=DataQualityStatus.FAIR,
                last_updated=datetime.now(),
            ),
        }
        recs = monitor._generate_recommendations(metrics, DataSourceType.PRICE_DATA)
        assert any("multiple price data providers" in r for r in recs)

    def test_generate_recommendations_news_freshness(self, monitor):
        metrics = {
            "freshness": DataQualityMetric(
                metric_name="freshness",
                current_value=0.70,
                target_value=1.0,
                threshold_warning=0.8,
                threshold_critical=0.5,
                status=DataQualityStatus.FAIR,
                last_updated=datetime.now(),
            ),
        }
        recs = monitor._generate_recommendations(metrics, DataSourceType.NEWS_DATA)
        assert any("news scraping frequency" in r for r in recs)

    def test_update_quality_history_new_source(self, monitor):
        monitor._update_quality_history("prices", 0.92)
        assert "prices" in monitor.quality_history
        assert len(monitor.quality_history["prices"]) == 1
        assert monitor.quality_history["prices"][0][1] == 0.92

    def test_update_quality_history_appends(self, monitor):
        monitor._update_quality_history("prices", 0.90)
        monitor._update_quality_history("prices", 0.92)
        monitor._update_quality_history("prices", 0.95)
        assert len(monitor.quality_history["prices"]) == 3

    def test_update_quality_history_prunes_old(self, monitor):
        old_time = datetime.now() - timedelta(days=35)
        monitor.quality_history["prices"] = [(old_time, 0.80)]
        monitor._update_quality_history("prices", 0.95)
        # Old entry should be pruned
        assert len(monitor.quality_history["prices"]) == 1
        assert monitor.quality_history["prices"][0][1] == 0.95

    def test_calculate_quality_trends_empty(self, monitor):
        trends = monitor._calculate_quality_trends()
        assert trends == {}

    def test_calculate_quality_trends_insufficient_data(self, monitor):
        monitor.quality_history["prices"] = [(datetime.now(), 0.90)]
        trends = monitor._calculate_quality_trends()
        assert "prices" not in trends  # needs >= 2 data points

    def test_calculate_quality_trends_with_data(self, monitor):
        now = datetime.now()
        monitor.quality_history["prices"] = [
            (now - timedelta(days=5), 0.80),
            (now - timedelta(days=3), 0.85),
            (now - timedelta(days=1), 0.90),
            (now, 0.95),
        ]
        trends = monitor._calculate_quality_trends()
        assert "prices" in trends
        assert "trend_7d" in trends["prices"]
        assert "trend_30d" in trends["prices"]
        assert trends["prices"]["current_score"] == 0.95
        assert trends["prices"]["data_points"] == 4

    def test_calculate_overall_health_excellent(self, monitor):
        assert monitor._calculate_overall_health([0.98, 0.96, 0.97]) == "excellent"

    def test_calculate_overall_health_good(self, monitor):
        assert monitor._calculate_overall_health([0.90, 0.88, 0.92]) == "good"

    def test_calculate_overall_health_fair(self, monitor):
        assert monitor._calculate_overall_health([0.75, 0.78, 0.72]) == "fair"

    def test_calculate_overall_health_poor(self, monitor):
        assert monitor._calculate_overall_health([0.40, 0.45, 0.35]) == "poor"

    def test_calculate_overall_health_empty(self, monitor):
        assert monitor._calculate_overall_health([]) == "unknown"

    @pytest.mark.asyncio
    async def test_check_quality_alerts_no_manager(self, monitor):
        quality = DataSourceQuality(
            source_name="prices",
            source_type=DataSourceType.PRICE_DATA,
            overall_score=0.30,
            status=DataQualityStatus.CRITICAL,
            metrics={},
            last_update=datetime.now(),
            issues=["Critical issue"],
            recommendations=[],
        )
        # Should not raise even without alert_manager
        await monitor._check_quality_alerts(quality)

    @pytest.mark.asyncio
    async def test_check_quality_alerts_critical(self, monitor):
        mock_alert_mgr = MagicMock()
        mock_alert_mgr.trigger_alert = AsyncMock()
        monitor.alert_manager = mock_alert_mgr

        quality = DataSourceQuality(
            source_name="prices",
            source_type=DataSourceType.PRICE_DATA,
            overall_score=0.30,
            status=DataQualityStatus.CRITICAL,
            metrics={},
            last_update=datetime.now(),
            issues=["Data stale"],
            recommendations=[],
        )
        await monitor._check_quality_alerts(quality)
        mock_alert_mgr.trigger_alert.assert_called()

    @pytest.mark.asyncio
    async def test_check_quality_alerts_poor(self, monitor):
        mock_alert_mgr = MagicMock()
        mock_alert_mgr.trigger_alert = AsyncMock()
        monitor.alert_manager = mock_alert_mgr

        quality = DataSourceQuality(
            source_name="prices",
            source_type=DataSourceType.PRICE_DATA,
            overall_score=0.55,
            status=DataQualityStatus.POOR,
            metrics={},
            last_update=datetime.now(),
            issues=["Low quality"],
            recommendations=[],
        )
        await monitor._check_quality_alerts(quality)
        mock_alert_mgr.trigger_alert.assert_called()

    @pytest.mark.asyncio
    async def test_check_quality_alerts_metric_below_critical(self, monitor):
        mock_alert_mgr = MagicMock()
        mock_alert_mgr.trigger_alert = AsyncMock()
        monitor.alert_manager = mock_alert_mgr

        quality = DataSourceQuality(
            source_name="prices",
            source_type=DataSourceType.PRICE_DATA,
            overall_score=0.90,
            status=DataQualityStatus.GOOD,
            metrics={
                "freshness": DataQualityMetric(
                    metric_name="freshness",
                    current_value=0.40,  # below threshold_critical of 0.5
                    target_value=1.0,
                    threshold_warning=0.8,
                    threshold_critical=0.5,
                    status=DataQualityStatus.CRITICAL,
                    last_updated=datetime.now(),
                ),
            },
            last_update=datetime.now(),
            issues=[],
            recommendations=[],
        )
        await monitor._check_quality_alerts(quality)
        mock_alert_mgr.trigger_alert.assert_called()
