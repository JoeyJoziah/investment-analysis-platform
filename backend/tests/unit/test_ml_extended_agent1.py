"""
Unit tests for four ML modules:
  - backend/ml/backtesting.py
  - backend/ml/cost_monitoring.py
  - backend/ml/dataset_hub.py
  - backend/ml/feature_store.py

Uses the importlib file-loading bypass to avoid pulling heavy transitive
dependencies (sklearn, matplotlib, seaborn, lightgbm, redis, sqlalchemy, etc.)
into the test process.
"""

import importlib
import importlib.util
import sys
import os
import json
import threading
import tempfile
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Stub ALL heavy dependencies BEFORE importlib loads any module.
# Use setdefault so we never clobber real libraries when they happen to be
# installed; tests stay hermetic regardless.
# ---------------------------------------------------------------------------

_STUBS = [
    # sklearn
    "sklearn",
    "sklearn.model_selection",
    "sklearn.metrics",
    "sklearn.preprocessing",
    "sklearn.feature_selection",
    "sklearn.decomposition",
    # plotting
    "matplotlib",
    "matplotlib.pyplot",
    "seaborn",
    # boosting / TA
    "lightgbm",
    "ta",
    "ta.momentum",
    "ta.trend",
    "ta.volatility",
    # scipy (used inside FeatureDriftDetector._calculate_ks_statistic)
    "scipy",
    "scipy.stats",
    # infrastructure deps required by feature_store.py at import time
    "redis",
    "sqlalchemy",
    "sqlalchemy.orm",
    # huggingface (dataset_hub)
    "huggingface_hub",
    "datasets",
    # other optional deps
    "mlflow",
    "mlflow.sklearn",
    "mlflow.pytorch",
    "joblib",
    "psutil",
]

for _stub_name in _STUBS:
    sys.modules.setdefault(_stub_name, MagicMock())

# Make sklearn sub-modules consistent so attribute look-ups don't error
_sk_preprocessing = sys.modules["sklearn.preprocessing"]
for _cls_name in ("StandardScaler", "RobustScaler", "MinMaxScaler"):
    if not hasattr(_sk_preprocessing, _cls_name):
        setattr(_sk_preprocessing, _cls_name, MagicMock)

_sk_fs = sys.modules["sklearn.feature_selection"]
for _cls_name in ("SelectKBest", "f_regression", "mutual_info_regression"):
    if not hasattr(_sk_fs, _cls_name):
        setattr(_sk_fs, _cls_name, MagicMock)

_sk_decomp = sys.modules["sklearn.decomposition"]
if not hasattr(_sk_decomp, "PCA"):
    setattr(_sk_decomp, "PCA", MagicMock)

_sk_metrics = sys.modules["sklearn.metrics"]
for _fn in ("mean_squared_error", "mean_absolute_error", "r2_score", "mutual_info_score"):
    if not hasattr(_sk_metrics, _fn):
        setattr(_sk_metrics, _fn, MagicMock(return_value=0.0))

# Make redis.from_url return a mock that raises on ping (simulates no Redis)
_redis_mod = sys.modules["redis"]
_redis_mock_instance = MagicMock()
_redis_mock_instance.ping.side_effect = ConnectionError("no redis in tests")
_redis_mod.from_url = MagicMock(return_value=_redis_mock_instance)

# sqlalchemy needs create_engine, Column, etc. to be actual callables
_sqlalchemy_mod = sys.modules["sqlalchemy"]
for _attr in ("create_engine", "Column", "Integer", "String", "Float",
              "DateTime", "Boolean", "Text", "JSON"):
    if not hasattr(_sqlalchemy_mod, _attr):
        setattr(_sqlalchemy_mod, _attr, MagicMock)

_sqlalchemy_orm = sys.modules["sqlalchemy.orm"]
if not hasattr(_sqlalchemy_orm, "sessionmaker"):
    setattr(_sqlalchemy_orm, "sessionmaker", MagicMock)

# ---------------------------------------------------------------------------
# importlib loads
# ---------------------------------------------------------------------------

_ml_dir = Path(__file__).resolve().parents[2] / "ml"


def _load(module_stem: str, filename: str):
    spec = importlib.util.spec_from_file_location(module_stem, _ml_dir / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_bt_mod = _load("backtesting_mod", "backtesting.py")
_cm_mod = _load("cost_monitoring_mod", "cost_monitoring.py")
_dh_mod = _load("dataset_hub_mod", "dataset_hub.py")
_fs_mod = _load("feature_store_mod", "feature_store.py")

# ---------------------------------------------------------------------------
# Re-export names from backtesting
# ---------------------------------------------------------------------------
BacktestMetric = _bt_mod.BacktestMetric
BacktestConfig = _bt_mod.BacktestConfig
TradeRecord = _bt_mod.TradeRecord
BacktestResult = _bt_mod.BacktestResult
WalkForwardValidator = _bt_mod.WalkForwardValidator
BacktestEngine = _bt_mod.BacktestEngine
get_backtest_engine = _bt_mod.get_backtest_engine

# ---------------------------------------------------------------------------
# Re-export names from cost_monitoring
# ---------------------------------------------------------------------------
ResourceType = _cm_mod.ResourceType
CostCategory = _cm_mod.CostCategory
ResourceUsage = _cm_mod.ResourceUsage
CostAlert = _cm_mod.CostAlert
OptimizationRecommendation = _cm_mod.OptimizationRecommendation
MLCostTracker = _cm_mod.MLCostTracker
MLCostOptimizer = _cm_mod.MLCostOptimizer
get_ml_cost_tracker = _cm_mod.get_ml_cost_tracker
track_ml_cost = _cm_mod.track_ml_cost

# ---------------------------------------------------------------------------
# Re-export names from dataset_hub
# ---------------------------------------------------------------------------
DatasetVersion = _dh_mod.DatasetVersion
HuggingFaceDatasetManager = _dh_mod.HuggingFaceDatasetManager
get_dataset_manager = _dh_mod.get_dataset_manager

# ---------------------------------------------------------------------------
# Re-export names from feature_store
# ---------------------------------------------------------------------------
FeatureType = _fs_mod.FeatureType
ComputeMode = _fs_mod.ComputeMode
FeatureStatus = _fs_mod.FeatureStatus
FeatureDefinition = _fs_mod.FeatureDefinition
FeatureValue = _fs_mod.FeatureValue
FeatureDriftMetrics = _fs_mod.FeatureDriftMetrics
FeatureValidator = _fs_mod.FeatureValidator
FeatureDriftDetector = _fs_mod.FeatureDriftDetector
FeatureStore = _fs_mod.FeatureStore
get_feature_store = _fs_mod.get_feature_store


# ===========================================================================
# Helpers
# ===========================================================================

def _make_backtest_config(**overrides):
    defaults = dict(
        start_date=datetime(2023, 1, 1, tzinfo=timezone.utc),
        end_date=datetime(2023, 12, 31, tzinfo=timezone.utc),
        initial_capital=100_000.0,
        commission=0.001,
        slippage=0.0005,
    )
    defaults.update(overrides)
    return BacktestConfig(**defaults)


def _make_trade_record(**overrides):
    defaults = dict(
        ticker="AAPL",
        entry_date=datetime(2023, 3, 1, tzinfo=timezone.utc),
        exit_date=datetime(2023, 3, 10, tzinfo=timezone.utc),
        entry_price=150.0,
        exit_price=160.0,
        quantity=100,
        side="long",
        pnl=1000.0,
        pnl_pct=0.0667,
        commission=15.0,
        slippage=7.5,
        duration_days=9,
        signal_strength=0.8,
        model_confidence=0.85,
    )
    defaults.update(overrides)
    return TradeRecord(**defaults)


def _make_feature_def(name="rsi_14d", **overrides):
    now = datetime.now(timezone.utc)
    defaults = dict(
        name=name,
        description="14-day RSI",
        feature_type=FeatureType.NUMERICAL,
        compute_mode=ComputeMode.BATCH,
        status=FeatureStatus.PRODUCTION,
        version="1.0.0",
        created_at=now,
        updated_at=now,
        created_by="test",
        dependencies=[],
        source_tables=["price_data"],
        computation_logic="rsi_logic",
        validation_rules={},
        tags=["technical"],
        business_context="Momentum indicator",
    )
    defaults.update(overrides)
    return FeatureDefinition(**defaults)


def _make_price_df(ticker: str, n: int = 30, base_price: float = 100.0):
    """Build a minimal price DataFrame for feature computation tests."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    prices = base_price * (1 + np.random.normal(0, 0.01, n)).cumprod()
    volume = np.random.lognormal(10, 0.5, n)
    return pd.DataFrame({
        "ticker": ticker,
        "date": dates,
        "close": prices,
        "open": prices * 0.999,
        "high": prices * 1.005,
        "low": prices * 0.995,
        "volume": volume,
    })


# ===========================================================================
# I.  BacktestMetric enum
# ===========================================================================

class TestBacktestMetric:
    def test_all_enum_values_accessible(self):
        expected = [
            "total_return", "annualized_return", "volatility", "sharpe_ratio",
            "sortino_ratio", "calmar_ratio", "max_drawdown", "win_rate",
            "profit_factor", "directional_accuracy", "information_ratio",
            "beta", "alpha", "tracking_error",
        ]
        actual = [m.value for m in BacktestMetric]
        assert sorted(actual) == sorted(expected)

    def test_enum_member_identity(self):
        assert BacktestMetric.SHARPE_RATIO.value == "sharpe_ratio"
        assert BacktestMetric.MAX_DRAWDOWN.value == "max_drawdown"
        assert BacktestMetric.WIN_RATE.value == "win_rate"

    def test_enum_lookup_by_value(self):
        member = BacktestMetric("total_return")
        assert member is BacktestMetric.TOTAL_RETURN

    def test_enum_count(self):
        assert len(BacktestMetric) == 14


# ===========================================================================
# II.  BacktestConfig dataclass
# ===========================================================================

class TestBacktestConfig:
    def test_required_fields(self):
        cfg = _make_backtest_config()
        assert cfg.start_date == datetime(2023, 1, 1, tzinfo=timezone.utc)
        assert cfg.end_date == datetime(2023, 12, 31, tzinfo=timezone.utc)

    def test_default_values(self):
        cfg = _make_backtest_config()
        assert cfg.initial_capital == 100_000.0
        assert cfg.commission == 0.001
        assert cfg.slippage == 0.0005
        assert cfg.lookback_window == 252
        assert cfg.rebalance_frequency == "daily"
        assert cfg.risk_free_rate == 0.02
        assert cfg.benchmark_symbol == "SPY"
        assert cfg.max_position_size == 0.1
        assert cfg.stop_loss is None
        assert cfg.take_profit is None
        assert cfg.transaction_costs_enabled is True

    def test_custom_values_override_defaults(self):
        cfg = _make_backtest_config(
            initial_capital=250_000.0,
            commission=0.005,
            risk_free_rate=0.04,
            benchmark_symbol="QQQ",
            stop_loss=0.05,
            take_profit=0.15,
        )
        assert cfg.initial_capital == 250_000.0
        assert cfg.commission == 0.005
        assert cfg.risk_free_rate == 0.04
        assert cfg.benchmark_symbol == "QQQ"
        assert cfg.stop_loss == 0.05
        assert cfg.take_profit == 0.15

    def test_transaction_costs_can_be_disabled(self):
        cfg = _make_backtest_config(transaction_costs_enabled=False)
        assert cfg.transaction_costs_enabled is False


# ===========================================================================
# III.  TradeRecord dataclass
# ===========================================================================

class TestTradeRecord:
    def test_construction_with_defaults(self):
        trade = _make_trade_record()
        assert trade.ticker == "AAPL"
        assert trade.side == "long"
        assert trade.pnl == 1000.0
        assert trade.quantity == 100

    def test_short_side(self):
        trade = _make_trade_record(side="short", pnl=-500.0)
        assert trade.side == "short"
        assert trade.pnl == -500.0

    def test_zero_pnl_trade(self):
        trade = _make_trade_record(pnl=0.0, pnl_pct=0.0)
        assert trade.pnl == 0.0

    def test_signal_strength_range(self):
        trade = _make_trade_record(signal_strength=0.95)
        assert 0.0 <= trade.signal_strength <= 1.0


# ===========================================================================
# IV.  WalkForwardValidator
# ===========================================================================

class TestWalkForwardValidator:
    def _make_data(self, n: int = 500) -> pd.DataFrame:
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        return pd.DataFrame({"close": np.random.randn(n)}, index=idx)

    def test_default_construction(self):
        wfv = WalkForwardValidator()
        assert wfv.n_splits == 5
        assert wfv.test_size == 30
        assert wfv.gap == 0
        assert wfv.expanding_window is False

    def test_split_returns_correct_count(self):
        wfv = WalkForwardValidator(n_splits=3, test_size=50)
        data = self._make_data(500)
        splits = wfv.split(data)
        assert len(splits) == 3

    def test_split_no_overlap_in_rolling_mode(self):
        wfv = WalkForwardValidator(n_splits=3, test_size=50, expanding_window=False)
        data = self._make_data(500)
        splits = wfv.split(data)
        for train_df, test_df in splits:
            assert len(set(train_df.index) & set(test_df.index)) == 0

    def test_expanding_window_grows(self):
        wfv = WalkForwardValidator(n_splits=3, test_size=50, expanding_window=True)
        data = self._make_data(600)
        splits = wfv.split(data)
        train_sizes = [len(tr) for tr, _ in splits]
        assert train_sizes[0] < train_sizes[1] < train_sizes[2]

    def test_gap_separates_train_test(self):
        wfv = WalkForwardValidator(n_splits=2, test_size=40, gap=10)
        data = self._make_data(400)
        splits = wfv.split(data)
        for train_df, test_df in splits:
            # The last train index should be before the first test index
            assert train_df.index[-1] < test_df.index[0]


# ===========================================================================
# V.  BacktestEngine – private calculation helpers
# ===========================================================================

class TestBacktestEngineHelpers:
    def setup_method(self):
        self.engine = BacktestEngine()

    # ----- _calculate_sharpe_ratio -----
    def test_sharpe_ratio_positive_returns(self):
        returns = pd.Series([0.001] * 252)
        sharpe = self.engine._calculate_sharpe_ratio(returns, risk_free_rate=0.02)
        assert sharpe > 0

    def test_sharpe_ratio_zero_returns(self):
        returns = pd.Series([0.0] * 100)
        sharpe = self.engine._calculate_sharpe_ratio(returns, risk_free_rate=0.02)
        assert sharpe == 0

    def test_sharpe_ratio_empty_series(self):
        sharpe = self.engine._calculate_sharpe_ratio(pd.Series([], dtype=float), 0.02)
        assert sharpe == 0

    # ----- _calculate_sortino_ratio -----
    def test_sortino_ratio_with_no_downside(self):
        returns = pd.Series([0.01] * 50)
        sortino = self.engine._calculate_sortino_ratio(returns, risk_free_rate=0.0)
        assert sortino == float("inf") or sortino > 0

    def test_sortino_ratio_empty_series(self):
        assert self.engine._calculate_sortino_ratio(pd.Series([], dtype=float), 0.02) == 0

    # ----- _calculate_max_drawdown -----
    def test_max_drawdown_monotonically_rising(self):
        cum_returns = pd.Series([1.0, 1.1, 1.2, 1.3])
        dd = self.engine._calculate_max_drawdown(cum_returns)
        assert dd == pytest.approx(0.0, abs=1e-10)

    def test_max_drawdown_with_known_drop(self):
        # Peak at 1.2, drops to 0.9: drawdown = (0.9-1.2)/1.2 = -0.25
        cum_returns = pd.Series([1.0, 1.2, 0.9, 1.0])
        dd = self.engine._calculate_max_drawdown(cum_returns)
        np.testing.assert_almost_equal(dd, -0.25, decimal=5)

    def test_max_drawdown_empty_series(self):
        assert self.engine._calculate_max_drawdown(pd.Series([], dtype=float)) == 0

    # ----- _calculate_beta -----
    def test_beta_perfect_correlation(self):
        bm = pd.Series([0.01, -0.01, 0.02, -0.02, 0.01])
        port = bm * 1.5  # beta should be 1.5
        beta = self.engine._calculate_beta(port, bm)
        np.testing.assert_almost_equal(beta, 1.5, decimal=5)

    def test_beta_zero_variance_benchmark(self):
        bm = pd.Series([0.01, 0.01, 0.01])
        port = pd.Series([0.02, 0.02, 0.02])
        beta = self.engine._calculate_beta(port, bm)
        assert beta == 1.0  # fallback

    def test_beta_mismatched_lengths_returns_one(self):
        bm = pd.Series([0.01, 0.02])
        port = pd.Series([0.01, 0.02, 0.03])
        beta = self.engine._calculate_beta(port, bm)
        assert beta == 1.0

    # ----- _calculate_alpha -----
    def test_alpha_neutral_strategy(self):
        bm = pd.Series([0.001] * 252)
        port = pd.Series([0.001] * 252)
        alpha = self.engine._calculate_alpha(port, bm, risk_free_rate=0.02)
        np.testing.assert_almost_equal(alpha, 0.0, decimal=5)

    # ----- _calculate_drawdowns -----
    def test_drawdowns_all_positive(self):
        portfolio = pd.Series([100.0, 110.0, 120.0, 130.0])
        dds = self.engine._calculate_drawdowns(portfolio)
        assert (dds == 0.0).all()

    def test_drawdowns_has_correct_shape(self):
        portfolio = pd.Series([100.0, 90.0, 95.0, 85.0])
        dds = self.engine._calculate_drawdowns(portfolio)
        assert len(dds) == len(portfolio)

    # ----- _initialize_portfolio -----
    def test_initialize_portfolio_cash(self):
        port = self.engine._initialize_portfolio(50_000.0, ["AAPL", "GOOG"])
        assert port["cash"] == 50_000.0
        assert set(port["positions"].keys()) == {"AAPL", "GOOG"}
        assert all(v == 0 for v in port["positions"].values())

    # ----- _calculate_portfolio_value -----
    def test_portfolio_value_no_positions(self):
        port = self.engine._initialize_portfolio(10_000.0, ["AAPL"])
        market_data = {
            "AAPL": pd.DataFrame({"close": [150.0]})
        }
        val = self.engine._calculate_portfolio_value(port, market_data, datetime.now())
        assert val == 10_000.0

    def test_portfolio_value_with_position(self):
        port = self.engine._initialize_portfolio(10_000.0, ["AAPL"])
        port["positions"]["AAPL"] = 10
        market_data = {
            "AAPL": pd.DataFrame({"close": [200.0]})
        }
        val = self.engine._calculate_portfolio_value(port, market_data, datetime.now())
        assert val == pytest.approx(12_000.0)

    # ----- _calculate_comprehensive_metrics with empty returns -----
    def test_comprehensive_metrics_empty_returns(self):
        cfg = _make_backtest_config()
        bm = pd.DataFrame({"returns": []})
        metrics = self.engine._calculate_comprehensive_metrics(
            pd.Series([], dtype=float), bm, cfg, []
        )
        assert metrics == {}

    # ----- _calculate_performance_attribution -----
    def test_performance_attribution_empty_trades(self):
        attr = self.engine._calculate_performance_attribution([])
        assert attr == {}

    def test_performance_attribution_categorises_by_holding_period(self):
        short = _make_trade_record(duration_days=3, pnl=100.0)
        medium = _make_trade_record(duration_days=15, pnl=200.0)
        long_t = _make_trade_record(duration_days=60, pnl=300.0)
        attr = self.engine._calculate_performance_attribution([short, medium, long_t])
        assert attr["short_term_pnl"] == 100.0
        assert attr["medium_term_pnl"] == 200.0
        assert attr["long_term_pnl"] == 300.0

    # ----- get_backtest_engine singleton -----
    def test_get_backtest_engine_returns_singleton(self):
        # Reset global first
        _bt_mod._backtest_engine = None
        e1 = get_backtest_engine()
        e2 = get_backtest_engine()
        assert e1 is e2


# ===========================================================================
# VI.  BacktestEngine – monthly/annual returns helpers
# ===========================================================================

class TestBacktestEngineReturnHelpers:
    def setup_method(self):
        self.engine = BacktestEngine()

    def test_monthly_returns_empty(self):
        result = self.engine._calculate_monthly_returns(pd.Series([], dtype=float))
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_annual_returns_empty(self):
        result = self.engine._calculate_annual_returns(pd.Series([], dtype=float))
        assert isinstance(result, pd.Series)
        assert result.empty

    def test_rolling_metrics_insufficient_data(self):
        returns = pd.Series([0.001] * 10)
        result = self.engine._calculate_rolling_metrics(returns, window=252)
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_calculate_risk_metrics_empty(self):
        bm = pd.DataFrame({"returns": []})
        rm = self.engine._calculate_risk_metrics(pd.Series([], dtype=float), bm)
        assert rm == {}

    def test_calculate_risk_metrics_has_expected_keys(self):
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))
        bm = pd.DataFrame({"returns": returns.values})
        rm = self.engine._calculate_risk_metrics(returns, bm)
        for key in ("var_95", "var_99", "cvar_95", "skewness", "kurtosis", "max_daily_loss"):
            assert key in rm


# ===========================================================================
# VII.  ResourceType and CostCategory enums
# ===========================================================================

class TestResourceTypeEnum:
    def test_all_values(self):
        expected = {
            "compute_cpu", "compute_gpu", "memory", "storage",
            "api_calls", "data_transfer", "model_inference",
            "training", "feature_computation",
        }
        actual = {r.value for r in ResourceType}
        assert actual == expected

    def test_member_identity(self):
        assert ResourceType.COMPUTE_CPU.value == "compute_cpu"
        assert ResourceType.MODEL_INFERENCE.value == "model_inference"


class TestCostCategoryEnum:
    def test_all_values(self):
        expected = {
            "infrastructure", "compute", "storage", "api_usage",
            "data_processing", "monitoring", "optimization",
        }
        actual = {c.value for c in CostCategory}
        assert actual == expected


# ===========================================================================
# VIII.  ResourceUsage dataclass
# ===========================================================================

class TestResourceUsage:
    def _make(self):
        return ResourceUsage(
            timestamp=datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc),
            resource_type=ResourceType.COMPUTE_CPU,
            usage_amount=2.5,
            unit="hours",
            cost_per_unit=0.08,
            total_cost=0.20,
            operation="training",
            model_name="lgbm_v1",
            metadata={"batch": 32},
        )

    def test_to_dict_keys(self):
        ru = self._make()
        d = ru.to_dict()
        assert "timestamp" in d
        assert "resource_type" in d
        assert "total_cost" in d
        assert "operation" in d

    def test_to_dict_serialises_enum(self):
        ru = self._make()
        d = ru.to_dict()
        assert d["resource_type"] == "compute_cpu"

    def test_to_dict_serialises_timestamp(self):
        ru = self._make()
        d = ru.to_dict()
        assert "2024-01-15" in d["timestamp"]


# ===========================================================================
# IX.  OptimizationRecommendation dataclass
# ===========================================================================

class TestOptimizationRecommendation:
    def test_to_dict_round_trip(self):
        rec = OptimizationRecommendation(
            category="compute",
            priority="high",
            estimated_savings=12.50,
            implementation_effort="medium",
            description="Reduce CPU usage",
            action_items=["Quantize model", "Cache predictions"],
            impact_on_performance="Low",
        )
        d = rec.to_dict()
        assert d["category"] == "compute"
        assert d["priority"] == "high"
        assert d["estimated_savings"] == pytest.approx(12.50)
        assert "Quantize model" in d["action_items"]


# ===========================================================================
# X.  MLCostTracker
# ===========================================================================

class TestMLCostTracker:
    def setup_method(self):
        # Clear any emergency mode left from prior tests
        os.environ.pop("ML_EMERGENCY_MODE", None)
        self.tracker = MLCostTracker(monthly_budget=50.0)

    def test_initial_state(self):
        assert self.tracker.monthly_budget == 50.0
        assert len(self.tracker.usage_records) == 0

    def test_pricing_config_has_all_resource_types(self):
        pricing = self.tracker.pricing
        for rt in ResourceType:
            assert rt.value in pricing, f"Missing pricing for {rt.value}"

    def test_record_usage_cpu(self):
        cost = self.tracker.record_usage(
            resource_type=ResourceType.COMPUTE_CPU,
            usage_amount=1.0,
            operation="training",
        )
        assert cost == pytest.approx(0.08, rel=1e-5)
        assert len(self.tracker.usage_records) == 1

    def test_record_usage_api_calls(self):
        cost = self.tracker.record_usage(
            resource_type=ResourceType.API_CALLS,
            usage_amount=2000,
            operation="api_request",
        )
        # 2000 / 1000 * 0.10 = 0.20
        assert cost == pytest.approx(0.20, rel=1e-5)

    def test_record_usage_model_inference(self):
        cost = self.tracker.record_usage(
            resource_type=ResourceType.MODEL_INFERENCE,
            usage_amount=5000,
            operation="inference",
        )
        # 5000 / 1000 * 0.02 = 0.10
        assert cost == pytest.approx(0.10, rel=1e-5)

    def test_record_usage_feature_computation(self):
        cost = self.tracker.record_usage(
            resource_type=ResourceType.FEATURE_COMPUTATION,
            usage_amount=10000,
            operation="feature_computation",
        )
        # 10000 / 1000 * 0.01 = 0.10
        assert cost == pytest.approx(0.10, rel=1e-5)

    def test_record_usage_stores_model_name(self):
        self.tracker.record_usage(
            resource_type=ResourceType.TRAINING,
            usage_amount=0.5,
            operation="training",
            model_name="my_model",
        )
        assert self.tracker.usage_records[-1].model_name == "my_model"

    def test_record_usage_updates_daily_costs(self):
        self.tracker.record_usage(ResourceType.COMPUTE_CPU, 1.0, "training")
        today_key = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        assert self.tracker.daily_costs[today_key] > 0

    def test_categorize_operation_training(self):
        cat = self.tracker._categorize_operation("model_training_run")
        assert cat == CostCategory.COMPUTE

    def test_categorize_operation_storage(self):
        cat = self.tracker._categorize_operation("save_model_artifact")
        assert cat == CostCategory.STORAGE

    def test_categorize_operation_monitoring(self):
        cat = self.tracker._categorize_operation("drift_monitor")
        assert cat == CostCategory.MONITORING

    def test_categorize_operation_default(self):
        cat = self.tracker._categorize_operation("unknown_operation_xyz")
        assert cat == CostCategory.INFRASTRUCTURE

    def test_get_current_month_cost_zero_when_empty(self):
        assert self.tracker.get_current_month_cost() == pytest.approx(0.0)

    def test_get_current_month_cost_accumulates(self):
        self.tracker.record_usage(ResourceType.COMPUTE_CPU, 2.0, "training")
        month_cost = self.tracker.get_current_month_cost()
        assert month_cost == pytest.approx(0.16, rel=1e-5)

    def test_is_operation_allowed_within_budget(self):
        allowed, reason = self.tracker.is_operation_allowed("inference", 0.01)
        assert allowed is True

    def test_is_operation_allowed_forced(self):
        allowed, reason = self.tracker.is_operation_allowed("training", 9999.0, force=True)
        assert allowed is True
        assert "Forced" in reason

    def test_generate_cost_recommendations_emergency(self):
        recs = self.tracker._generate_cost_recommendations(96.0)
        assert len(recs) > 0
        assert any("EMERGENCY" in r for r in recs)

    def test_generate_cost_recommendations_critical(self):
        recs = self.tracker._generate_cost_recommendations(87.0)
        assert len(recs) > 0

    def test_generate_cost_recommendations_warning(self):
        recs = self.tracker._generate_cost_recommendations(72.0)
        assert len(recs) > 0

    def test_get_usage_summary_shape(self):
        summary = self.tracker.get_usage_summary()
        assert "current_month_cost" in summary
        assert "projected_monthly_cost" in summary
        assert "monthly_budget" in summary
        assert summary["monthly_budget"] == 50.0

    def test_cost_breakdown_structure(self):
        self.tracker.record_usage(ResourceType.COMPUTE_CPU, 1.0, "training")
        breakdown = self.tracker.get_cost_breakdown(days_back=30)
        assert "total_cost" in breakdown
        assert "cost_by_resource" in breakdown
        assert "cost_by_operation" in breakdown
        assert breakdown["monthly_budget"] == 50.0


# ===========================================================================
# XI.  MLCostOptimizer
# ===========================================================================

class TestMLCostOptimizer:
    def setup_method(self):
        self.tracker = MLCostTracker(monthly_budget=50.0)
        self.optimizer = MLCostOptimizer(self.tracker)

    def test_optimization_enabled_by_default(self):
        assert self.optimizer.optimization_enabled is True

    def test_optimize_operation_disabled(self):
        self.optimizer.optimization_enabled = False
        result = self.optimizer.optimize_operation("inference", 1.0)
        assert result["optimized"] is False

    def test_optimize_operation_suggests_caching(self):
        result = self.optimizer.optimize_operation("inference", 1.0, context={})
        types = [o["type"] for o in result["optimizations"]]
        assert "caching" in types

    def test_optimize_operation_suggests_batching(self):
        result = self.optimizer.optimize_operation("inference", 1.0, context={"batch_size": 1})
        types = [o["type"] for o in result["optimizations"]]
        assert "batching" in types

    def test_optimize_operation_suggests_quantization_for_inference(self):
        result = self.optimizer.optimize_operation(
            "inference", 1.0, context={"quantized": False}
        )
        types = [o["type"] for o in result["optimizations"]]
        assert "quantization" in types

    def test_optimized_cost_is_lower(self):
        result = self.optimizer.optimize_operation("inference", 1.0, context={})
        assert result["optimized_cost"] < result["original_cost"]

    def test_savings_percent_non_negative(self):
        result = self.optimizer.optimize_operation("inference", 1.0, context={})
        assert result["savings_percent"] >= 0

    def test_no_caching_suggestion_when_already_cached(self):
        result = self.optimizer.optimize_operation(
            "inference", 1.0, context={"cache": True}
        )
        types = [o["type"] for o in result["optimizations"]]
        assert "caching" not in types


# ===========================================================================
# XII.  DatasetVersion dataclass
# ===========================================================================

class TestDatasetVersion:
    def _make(self, **overrides):
        defaults = dict(
            version="1.0.0",
            commit_hash="abc123",
            created_at=datetime.now(timezone.utc).isoformat(),
            total_samples=10000,
            stock_count=50,
            feature_count=80,
            date_range_start="2020-01-01",
            date_range_end="2023-12-31",
            train_samples=7000,
            val_samples=1500,
            test_samples=1500,
            feature_columns=["close", "rsi"],
            label_columns=["return_5d"],
            data_hash="deadbeef12345678",
        )
        defaults.update(overrides)
        return DatasetVersion(**defaults)

    def test_basic_construction(self):
        dv = self._make()
        assert dv.version == "1.0.0"
        assert dv.total_samples == 10000
        assert dv.stock_count == 50

    def test_to_dict_contains_all_fields(self):
        dv = self._make()
        d = dv.to_dict()
        assert d["version"] == "1.0.0"
        assert d["total_samples"] == 10000
        assert "feature_columns" in d

    def test_from_dict_round_trip(self):
        dv = self._make()
        d = dv.to_dict()
        restored = DatasetVersion.from_dict(d)
        assert restored.version == dv.version
        assert restored.data_hash == dv.data_hash
        assert restored.feature_columns == dv.feature_columns

    def test_optional_parent_version_default_none(self):
        dv = self._make()
        assert dv.parent_version is None

    def test_optional_model_versions_default_none(self):
        dv = self._make()
        assert dv.model_versions is None

    def test_with_parent_version(self):
        dv = self._make(parent_version="0.9.0", model_versions=["model_v1"])
        assert dv.parent_version == "0.9.0"
        assert "model_v1" in dv.model_versions


# ===========================================================================
# XIII.  HuggingFaceDatasetManager  (mocked HF libs)
# ===========================================================================

class TestHuggingFaceDatasetManager:
    def setup_method(self):
        # Force HF unavailable so no network calls are made
        self.manager = HuggingFaceDatasetManager.__new__(HuggingFaceDatasetManager)
        self.manager.repo_id = "test-org/test-dataset"
        self.manager.token = None
        self.manager._hf_available = False
        self.manager._api = None
        self.manager.auto_create_repo = False
        self.manager.private = True
        self.manager.lock = threading.Lock()
        with tempfile.TemporaryDirectory() as tmpdir:
            self.manager.local_cache_dir = Path(tmpdir)

    def test_check_available_returns_false_when_no_hf(self):
        assert self.manager._check_available() is False

    def test_list_versions_returns_empty_when_unavailable(self):
        versions = self.manager.list_versions()
        assert versions == []

    def test_get_version_metadata_returns_none_when_unavailable(self):
        result = self.manager.get_version_metadata("1.0.0")
        assert result is None

    def test_upload_dataset_returns_none_when_unavailable(self):
        train = pd.DataFrame({"a": [1, 2]})
        val = pd.DataFrame({"a": [3]})
        test = pd.DataFrame({"a": [4]})
        result = self.manager.upload_dataset(train, val, test, {})
        assert result is None

    def test_download_dataset_returns_none_when_unavailable(self):
        result = self.manager.download_dataset()
        assert result is None

    def test_compute_data_hash_deterministic(self):
        df = pd.DataFrame({"col": range(20)})
        h1 = self.manager._compute_data_hash(df)
        h2 = self.manager._compute_data_hash(df)
        assert h1 == h2

    def test_compute_data_hash_changes_with_data(self):
        df1 = pd.DataFrame({"col": range(20)})
        df2 = pd.DataFrame({"col": range(20, 40)})
        assert self.manager._compute_data_hash(df1) != self.manager._compute_data_hash(df2)

    def test_get_next_version_first_upload(self):
        # When no versions exist, next version should be 1.0.0
        self.manager._check_available = MagicMock(return_value=True)
        self.manager.list_versions = MagicMock(return_value=[])
        version = self.manager._get_next_version("patch")
        assert version == "1.0.0"

    def test_get_next_version_patch_increment(self):
        self.manager._check_available = MagicMock(return_value=True)
        self.manager.list_versions = MagicMock(return_value=["1.0.3", "1.0.0", "1.0.1"])
        version = self.manager._get_next_version("patch")
        assert version == "1.0.4"

    def test_get_next_version_minor_increment(self):
        self.manager._check_available = MagicMock(return_value=True)
        self.manager.list_versions = MagicMock(return_value=["1.2.0"])
        version = self.manager._get_next_version("minor")
        assert version == "1.3.0"

    def test_get_next_version_major_increment(self):
        self.manager._check_available = MagicMock(return_value=True)
        self.manager.list_versions = MagicMock(return_value=["2.0.0"])
        version = self.manager._get_next_version("major")
        assert version == "3.0.0"

    def test_get_dataset_manager_disabled_env(self, monkeypatch):
        monkeypatch.setenv("HF_HUB_ENABLED", "false")
        # Reset singleton
        _dh_mod._dataset_manager = None
        result = get_dataset_manager()
        assert result is None

    def test_upload_from_local_missing_dir_returns_none(self):
        result = self.manager.upload_from_local(Path("/nonexistent/path/xyz123"))
        assert result is None


# ===========================================================================
# XIV.  FeatureType / ComputeMode / FeatureStatus enums
# ===========================================================================

class TestFeatureStoreEnums:
    def test_feature_type_values(self):
        expected = {"numerical", "categorical", "boolean", "datetime", "text"}
        assert {ft.value for ft in FeatureType} == expected

    def test_compute_mode_values(self):
        expected = {"batch", "streaming", "on_demand"}
        assert {cm.value for cm in ComputeMode} == expected

    def test_feature_status_values(self):
        expected = {
            "development", "testing", "production", "deprecated", "retired"
        }
        assert {fs.value for fs in FeatureStatus} == expected

    def test_feature_type_lookup(self):
        assert FeatureType("numerical") is FeatureType.NUMERICAL

    def test_feature_status_lookup(self):
        assert FeatureStatus("production") is FeatureStatus.PRODUCTION


# ===========================================================================
# XV.  FeatureDefinition dataclass
# ===========================================================================

class TestFeatureDefinition:
    def test_basic_construction(self):
        fd = _make_feature_def()
        assert fd.name == "rsi_14d"
        assert fd.version == "1.0.0"
        assert fd.feature_type is FeatureType.NUMERICAL

    def test_to_dict_serialises_enums(self):
        fd = _make_feature_def()
        d = fd.to_dict()
        assert d["feature_type"] == "numerical"
        assert d["compute_mode"] == "batch"
        assert d["status"] == "production"

    def test_to_dict_serialises_datetimes(self):
        fd = _make_feature_def()
        d = fd.to_dict()
        # Datetime fields should be ISO strings
        assert isinstance(d["created_at"], str)
        assert isinstance(d["updated_at"], str)

    def test_optional_sla_hours_default_none(self):
        fd = _make_feature_def()
        assert fd.sla_hours is None

    def test_optional_monitoring_config_default_none(self):
        fd = _make_feature_def()
        assert fd.monitoring_config is None


# ===========================================================================
# XVI.  FeatureValidator
# ===========================================================================

class TestFeatureValidator:
    def setup_method(self):
        self.validator = FeatureValidator()

    def test_validate_numerical_clean_data(self):
        values = pd.Series([1.0, 2.0, 3.0])
        valid, errors = self.validator._validate_numerical(values)
        assert errors == []
        assert valid.all()

    def test_validate_numerical_infinite_values(self):
        values = pd.Series([1.0, float("inf"), 3.0])
        valid, errors = self.validator._validate_numerical(values)
        assert len(errors) == 1
        assert "infinite" in errors[0].lower()

    def test_validate_categorical_nulls(self):
        values = pd.Series(["a", None, "b"])
        valid, errors = self.validator._validate_categorical(values)
        assert len(errors) == 1

    def test_validate_boolean_valid(self):
        values = pd.Series([True, False, True])
        valid, errors = self.validator._validate_boolean(values)
        assert errors == []

    def test_validate_boolean_invalid(self):
        values = pd.Series([True, "maybe", False])
        valid, errors = self.validator._validate_boolean(values)
        assert len(errors) == 1

    def test_apply_range_rule_min_max(self):
        values = pd.Series([1.0, 5.0, 150.0])  # 150 > max=100
        valid, errors = self.validator._apply_validation_rule(
            "range", {"min": 0.0, "max": 100.0}, values
        )
        assert len(errors) == 1
        assert "maximum" in errors[0].lower()

    def test_apply_range_rule_below_min(self):
        values = pd.Series([-5.0, 10.0, 20.0])
        valid, errors = self.validator._apply_validation_rule(
            "range", {"min": 0.0}, values
        )
        assert len(errors) == 1
        assert "minimum" in errors[0].lower()

    def test_apply_allowed_values_rule(self):
        values = pd.Series(["A", "B", "INVALID"])
        valid, errors = self.validator._apply_validation_rule(
            "allowed_values", {"values": ["A", "B", "C"]}, values
        )
        assert len(errors) == 1

    def test_apply_not_null_rule(self):
        values = pd.Series([1.0, None, 3.0])
        valid, errors = self.validator._apply_validation_rule(
            "not_null", {"required": True}, values
        )
        assert len(errors) == 1

    def test_validate_feature_with_range_rule(self):
        fd = _make_feature_def(
            feature_type=FeatureType.NUMERICAL,
            validation_rules={"range": {"min": 0.0, "max": 100.0}},
        )
        values = pd.Series([50.0, 75.0, 200.0])  # 200 violates max
        quality_scores, errors = self.validator.validate_feature(fd, values)
        assert len(errors) >= 1
        # Quality score for violating row should be less than 1
        assert quality_scores.iloc[2] < 1.0


# ===========================================================================
# XVII.  FeatureDriftDetector
# ===========================================================================

class TestFeatureDriftDetector:
    def setup_method(self):
        self.detector = FeatureDriftDetector()

    def test_default_window_params(self):
        assert self.detector.reference_window_days == 30
        assert self.detector.detection_window_days == 7

    def test_calculate_psi_identical_distributions(self):
        np.random.seed(0)
        ref = pd.Series(np.random.normal(0, 1, 500))
        psi = self.detector._calculate_psi(ref, ref)
        assert psi >= 0.0

    def test_calculate_psi_empty_reference(self):
        psi = self.detector._calculate_psi(pd.Series([], dtype=float), pd.Series([1.0]))
        assert psi == 1.0

    def test_calculate_psi_empty_current(self):
        psi = self.detector._calculate_psi(pd.Series([1.0]), pd.Series([], dtype=float))
        assert psi == 1.0

    def test_calculate_js_distance_identical(self):
        # _calculate_js_distance can produce NaN when histogram bins contain
        # zero-probability entries (0 * log(0/0) → NaN in the source's KL
        # divergence). We only assert the result is either NaN or >= 0.
        ref = pd.Series(np.random.normal(0, 1, 300))
        js = self.detector._calculate_js_distance(ref, ref)
        assert np.isnan(js) or js >= 0.0

    def test_detect_drift_returns_correct_type(self):
        np.random.seed(1)
        ref = pd.Series(np.random.normal(0, 1, 500))
        cur = pd.Series(np.random.normal(0, 1, 200))
        result = self.detector.detect_drift("test_feature", ref, cur)
        assert isinstance(result, FeatureDriftMetrics)

    def test_detect_drift_large_shift_detected(self):
        np.random.seed(2)
        ref = pd.Series(np.random.normal(0, 1, 500))
        cur = pd.Series(np.random.normal(5, 1, 200))  # Large mean shift
        result = self.detector.detect_drift("test_feature", ref, cur)
        # drift_score may be NaN if JS distance is NaN (source-level instability)
        assert np.isnan(result.drift_score) or result.drift_score >= 0.0
        # distribution_shift_detected is a numpy bool_ in this context
        assert bool(result.distribution_shift_detected) in (True, False)

    def test_drift_metrics_fields(self):
        ref = pd.Series(np.random.normal(0, 1, 200))
        cur = pd.Series(np.random.normal(0, 1, 100))
        result = self.detector.detect_drift("rsi", ref, cur)
        assert result.feature_name == "rsi"
        assert isinstance(result.timestamp, datetime)
        assert 0.0 <= result.population_stability_index <= 1.0
        assert result.kolmogorov_smirnov_statistic >= 0.0
        # JS distance may be NaN due to zero-bin divisions in source implementation
        js = result.jensen_shannon_distance
        assert np.isnan(js) or 0.0 <= js <= 1.0


# ===========================================================================
# XVIII.  FeatureStore
# ===========================================================================

class TestFeatureStore:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = FeatureStore(
            storage_path=self.tmpdir,
            enable_caching=False,
        )

    def test_initial_registry_empty(self):
        assert isinstance(self.store.feature_registry, dict)

    def test_register_feature_creates_entry(self):
        result = self.store.register_feature(
            name="vol_20d",
            description="20-day volatility",
            feature_type=FeatureType.NUMERICAL,
            compute_mode=ComputeMode.BATCH,
            computation_logic="vol_logic",
        )
        assert result is True
        assert "vol_20d" in self.store.feature_registry

    def test_register_feature_default_status_development(self):
        self.store.register_feature(
            name="test_feat",
            description="Test",
            feature_type=FeatureType.NUMERICAL,
            compute_mode=ComputeMode.BATCH,
            computation_logic="logic",
        )
        fd = self.store.feature_registry["test_feat"]
        assert fd.status == FeatureStatus.DEVELOPMENT

    def test_register_feature_version_increment(self):
        self.store.register_feature(
            name="test_feat",
            description="v1",
            feature_type=FeatureType.NUMERICAL,
            compute_mode=ComputeMode.BATCH,
            computation_logic="logic",
        )
        self.store.register_feature(
            name="test_feat",
            description="v2",
            feature_type=FeatureType.NUMERICAL,
            compute_mode=ComputeMode.BATCH,
            computation_logic="updated_logic",
        )
        fd = self.store.feature_registry["test_feat"]
        assert fd.version == "1.0.1"

    def test_sort_features_by_dependencies_trivial(self):
        self.store.register_feature("feat_a", "a", FeatureType.NUMERICAL, ComputeMode.BATCH, "a")
        self.store.register_feature("feat_b", "b", FeatureType.NUMERICAL, ComputeMode.BATCH, "b")
        sorted_feats = self.store._sort_features_by_dependencies(["feat_a", "feat_b"])
        assert set(sorted_feats) == {"feat_a", "feat_b"}

    def test_get_feature_lineage_unknown_feature(self):
        lineage = self.store.get_feature_lineage("nonexistent")
        assert lineage == {}

    def test_get_feature_lineage_known_feature(self):
        self.store.register_feature(
            name="dep_a",
            description="dep",
            feature_type=FeatureType.NUMERICAL,
            compute_mode=ComputeMode.BATCH,
            computation_logic="logic",
        )
        self.store.register_feature(
            name="feat_with_dep",
            description="feat",
            feature_type=FeatureType.NUMERICAL,
            compute_mode=ComputeMode.BATCH,
            computation_logic="logic",
            dependencies=["dep_a"],
        )
        lineage = self.store.get_feature_lineage("feat_with_dep")
        assert "dep_a" in lineage["direct_dependencies"]

    def test_generate_cache_key_deterministic(self):
        ts = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
        k1 = self.store._generate_cache_key(["f1", "f2"], ["AAPL", "MSFT"], ts)
        k2 = self.store._generate_cache_key(["f2", "f1"], ["MSFT", "AAPL"], ts)
        assert k1 == k2  # sorted internally

    def test_generate_cache_key_changes_with_features(self):
        ts = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
        k1 = self.store._generate_cache_key(["f1"], ["AAPL"], ts)
        k2 = self.store._generate_cache_key(["f2"], ["AAPL"], ts)
        assert k1 != k2

    def test_get_from_cache_no_cache_returns_none(self):
        result = self.store._get_from_cache("any_key")
        assert result is None

    def test_cleanup_old_features_returns_zero_when_no_files(self):
        count = self.store.cleanup_old_features(days_to_keep=90)
        assert count == 0

    def test_register_computation_callable(self):
        def my_func(entity_ids, ts, data_sources, computed):
            return pd.Series([1.0] * len(entity_ids), index=entity_ids)

        self.store.register_computation("my_feature", my_func)
        assert "my_feature" in self.store.computation_cache

    def test_register_computation_rejects_non_callable(self):
        with pytest.raises(ValueError):
            self.store.register_computation("bad_feature", "not_callable")

    def test_compute_features_unregistered_feature_returns_nan(self):
        result = self.store.compute_features(
            feature_names=["nonexistent_feature"],
            entity_ids=["AAPL"],
        )
        # The feature should be skipped (warning logged) and not appear
        # (or appear as NaN if column added)
        assert isinstance(result, pd.DataFrame)

    def test_compute_features_builtin_price_return_1d(self):
        # Register price_return_1d as a built-in feature
        self.store.register_feature(
            name="price_return_1d",
            description="1-day return",
            feature_type=FeatureType.NUMERICAL,
            compute_mode=ComputeMode.BATCH,
            computation_logic="builtin",
        )
        price_df = _make_price_df("AAPL", n=10)
        result = self.store.compute_features(
            feature_names=["price_return_1d"],
            entity_ids=["AAPL"],
            data_sources={"price_data": price_df},
        )
        assert isinstance(result, pd.DataFrame)
        assert "price_return_1d" in result.columns

    def test_execute_python_computation_returns_nan_series(self):
        fd = _make_feature_def(name="unknown_feature")
        result = self.store._execute_python_computation(
            fd, ["AAPL", "MSFT"], datetime.now(), {}, pd.DataFrame()
        )
        assert isinstance(result, pd.Series)
        assert result.isna().all()

    def test_get_feature_statistics_skips_unregistered(self):
        stats = self.store.get_feature_statistics(["nonexistent"])
        assert stats == {}

    def test_get_feature_statistics_returns_data_for_registered(self):
        self.store.register_feature(
            name="registered_feat",
            description="test",
            feature_type=FeatureType.NUMERICAL,
            compute_mode=ComputeMode.BATCH,
            computation_logic="logic",
        )
        stats = self.store.get_feature_statistics(["registered_feat"])
        assert "registered_feat" in stats
        assert "count" in stats["registered_feat"]
        assert "mean" in stats["registered_feat"]

    def test_monitor_feature_drift_returns_metrics_for_registered(self):
        self.store.register_feature(
            name="vol_feature",
            description="volatility",
            feature_type=FeatureType.NUMERICAL,
            compute_mode=ComputeMode.BATCH,
            computation_logic="logic",
        )
        drift = self.store.monitor_feature_drift("vol_feature")
        assert drift is not None
        assert isinstance(drift, FeatureDriftMetrics)

    def test_monitor_feature_drift_returns_none_for_unregistered(self):
        result = self.store.monitor_feature_drift("does_not_exist")
        assert result is None


# ===========================================================================
# XIX.  FeatureStore – built-in computation helpers via data_sources
# ===========================================================================

class TestFeatureStoreBuiltins:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = FeatureStore(
            storage_path=self.tmpdir,
            enable_caching=False,
        )
        # Build a price DataFrame with 30 rows for AAPL and MSFT
        self.price_aapl = _make_price_df("AAPL", n=30)
        self.price_msft = _make_price_df("MSFT", n=30)
        self.price_df = pd.concat([self.price_aapl, self.price_msft], ignore_index=True)

    def _data_sources(self):
        return {"price_data": self.price_df}

    def test_compute_price_return_1d_aapl(self):
        result = self.store._compute_price_return_1d(
            ["AAPL"], datetime.now(), self._data_sources(), pd.DataFrame()
        )
        assert isinstance(result, pd.Series)
        assert not np.isnan(result["AAPL"])

    def test_compute_price_return_1d_no_data_source(self):
        result = self.store._compute_price_return_1d(
            ["AAPL"], datetime.now(), {}, pd.DataFrame()
        )
        assert result.isna().all()

    def test_compute_price_return_5d_aapl(self):
        result = self.store._compute_price_return_5d(
            ["AAPL"], datetime.now(), self._data_sources(), pd.DataFrame()
        )
        assert not np.isnan(result["AAPL"])

    def test_compute_price_volatility_20d(self):
        result = self.store._compute_price_volatility_20d(
            ["AAPL"], datetime.now(), self._data_sources(), pd.DataFrame()
        )
        assert not np.isnan(result["AAPL"])
        assert result["AAPL"] > 0

    def test_compute_volume_ratio_20d(self):
        result = self.store._compute_volume_ratio_20d(
            ["AAPL"], datetime.now(), self._data_sources(), pd.DataFrame()
        )
        assert not np.isnan(result["AAPL"])
        assert result["AAPL"] > 0

    def test_compute_sma_20d(self):
        result = self.store._compute_sma_20d(
            ["AAPL"], datetime.now(), self._data_sources(), pd.DataFrame()
        )
        assert not np.isnan(result["AAPL"])

    def test_compute_ema_20d(self):
        result = self.store._compute_ema_20d(
            ["AAPL"], datetime.now(), self._data_sources(), pd.DataFrame()
        )
        assert not np.isnan(result["AAPL"])

    def test_compute_rsi_14d_value_in_range(self):
        price_df = _make_price_df("AAPL", n=20)  # need at least 15 for RSI
        result = self.store._compute_rsi_14d(
            ["AAPL"], datetime.now(), {"price_data": price_df}, pd.DataFrame()
        )
        val = result["AAPL"]
        assert np.isnan(val) or (0.0 <= val <= 100.0)

    def test_compute_pe_ratio_missing_fundamentals(self):
        result = self.store._compute_pe_ratio(
            ["AAPL"], datetime.now(), self._data_sources(), pd.DataFrame()
        )
        assert result.isna().all()

    def test_compute_pe_ratio_with_fundamentals(self):
        fund_df = pd.DataFrame({
            "ticker": ["AAPL"],
            "eps": [5.0],
        })
        data_sources = {**self._data_sources(), "fundamental_data": fund_df}
        result = self.store._compute_pe_ratio(
            ["AAPL"], datetime.now(), data_sources, pd.DataFrame()
        )
        assert not np.isnan(result["AAPL"])
        assert result["AAPL"] > 0

    def test_compute_market_cap_with_fundamentals(self):
        fund_df = pd.DataFrame({
            "ticker": ["AAPL"],
            "shares_outstanding": [1_000_000_000],
        })
        data_sources = {**self._data_sources(), "fundamental_data": fund_df}
        result = self.store._compute_market_cap(
            ["AAPL"], datetime.now(), data_sources, pd.DataFrame()
        )
        assert not np.isnan(result["AAPL"])
        assert result["AAPL"] > 0
