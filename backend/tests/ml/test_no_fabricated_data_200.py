"""
Regression tests for P0 finding #200 — fabricated financial data.

These tests assert that the affected production paths raise 503 (routers) or
a domain exception (ml modules) when upstream data is unavailable, rather than
silently returning synthetic / hardcoded values.

All tests are unit-level and do NOT require live providers.

Run with:
    python3 -m pytest backend/tests/ml/test_no_fabricated_data_200.py -v --noconftest
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Set minimum required env vars BEFORE any backend imports so that the
# pydantic Settings model does not raise ValidationError at import time.
# This mirrors the pattern used in backend/tests/conftest.py.
# ---------------------------------------------------------------------------
import os as _os

_os.environ.setdefault("SECRET_KEY", "test-secret-key-for-finding-200")
_os.environ.setdefault("JWT_SECRET_KEY", "test-jwt-secret-key-for-finding-200")
_os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
_os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
_os.environ.setdefault("TESTING", "True")
_os.environ.setdefault("DEBUG", "True")
# Must NOT be "production" so that security_config._require_secret uses dev defaults
_os.environ.setdefault("ENVIRONMENT", "development")
# Required by secrets_manager (fail-fast in prod, skipped in dev)
_os.environ.setdefault("MASTER_SECRET_KEY", "test-master-secret-key-finding-200")

import sys
import types
import importlib
import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Stub out heavy optional dependencies that may not be installed in CI
# (mlflow, torch, prophet, xgboost) and partially-stubbed sklearn.metrics
# which uses a MagicMock stub but whose `from ... import name` forms fail.
# We use MagicMock() for all of them so attribute access also works.
# ---------------------------------------------------------------------------

for _mod in [
    "mlflow",
    "mlflow.sklearn",
    "mlflow.pyfunc",
    "torch",
    "torch.nn",
    "torch.utils",
    "torch.utils.data",
    "prophet",
    "xgboost",
]:
    sys.modules[_mod] = MagicMock(name=_mod)

# sklearn itself is installed but model_versioning does
# `from sklearn.metrics import accuracy_score, …`; the real module is fine
# unless our earlier stub shadowed it. Ensure it is the real sklearn.
import sklearn.metrics as _real_sklearn_metrics  # noqa: E402
sys.modules["sklearn.metrics"] = _real_sklearn_metrics

# Stub the full backend.ml.pipeline chain so training_pipeline can import
for _mod in [
    "backend.ml.pipeline",
    "backend.ml.pipeline.implementations",
    "backend.ml.pipeline.registry",
    "backend.ml.pipeline.monitoring",
    "backend.ml.pipeline.deployment",
]:
    sys.modules[_mod] = MagicMock(name=_mod)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_mock_model_manager(model_available: bool = False) -> MagicMock:
    # MagicMock treats names starting with ``assert_`` as unittest assertions.
    # ModelManager.assert_real_model must be a real child mock, so use a spec.
    mm = MagicMock(
        spec=["model_metadata", "get_model", "predict", "assert_real_model"]
    )
    mm.model_metadata = {}
    mm.get_model.return_value = MagicMock() if model_available else None
    mm.predict.return_value = [100.0]
    mm.assert_real_model = MagicMock()
    return mm


# ===========================================================================
# 1. backend/api/routers/ml.py — LSTM / XGBoost without feature data
# ===========================================================================

class TestMLRouterNoFabricatedData:
    """
    _run_single_model_prediction must raise HTTP 503 when no real feature
    data is provided and BOOTSTRAP_MODELS is not set.
    """

    @pytest.mark.asyncio
    async def test_lstm_raises_503_without_feature_data(self, monkeypatch):
        monkeypatch.delenv("BOOTSTRAP_MODELS", raising=False)
        # Re-import to pick up the cleared env var
        if "backend.api.routers.ml" in sys.modules:
            del sys.modules["backend.api.routers.ml"]

        from fastapi import HTTPException
        from backend.api.routers.ml import _run_single_model_prediction, MLModelType

        mm = _make_mock_model_manager(model_available=True)

        with pytest.raises(HTTPException) as exc_info:
            await _run_single_model_prediction(
                model_manager=mm,
                model_type=MLModelType.LSTM,
                ticker="AAPL",
                horizon_days=5,
                base_price=None,
                feature_data=None,
            )

        assert exc_info.value.status_code == 503
        detail = exc_info.value.detail.lower()
        assert "synthetic" in detail or "real" in detail

    @pytest.mark.asyncio
    async def test_xgboost_raises_503_without_feature_data(self, monkeypatch):
        monkeypatch.delenv("BOOTSTRAP_MODELS", raising=False)
        if "backend.api.routers.ml" in sys.modules:
            del sys.modules["backend.api.routers.ml"]

        from fastapi import HTTPException
        from backend.api.routers.ml import _run_single_model_prediction, MLModelType

        mm = _make_mock_model_manager(model_available=True)

        with pytest.raises(HTTPException) as exc_info:
            await _run_single_model_prediction(
                model_manager=mm,
                model_type=MLModelType.XGBOOST,
                ticker="MSFT",
                horizon_days=5,
                base_price=None,
                feature_data=None,
            )

        assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    async def test_bootstrap_mode_does_not_raise(self, monkeypatch):
        """When BOOTSTRAP_MODELS=1, zero arrays are used — no 503 raised."""
        monkeypatch.setenv("BOOTSTRAP_MODELS", "1")
        if "backend.api.routers.ml" in sys.modules:
            del sys.modules["backend.api.routers.ml"]

        from backend.api.routers.ml import _run_single_model_prediction, MLModelType

        mm = _make_mock_model_manager(model_available=True)
        # Should not raise
        result = await _run_single_model_prediction(
            model_manager=mm,
            model_type=MLModelType.LSTM,
            ticker="TEST",
            horizon_days=3,
            base_price=None,
            feature_data=None,
        )
        assert result is not None

    @pytest.mark.asyncio
    async def test_lstm_with_real_feature_data_does_not_raise(self, monkeypatch):
        """Supplying real feature data bypasses the 503 guard."""
        monkeypatch.delenv("BOOTSTRAP_MODELS", raising=False)
        if "backend.api.routers.ml" in sys.modules:
            del sys.modules["backend.api.routers.ml"]
        import numpy as np
        from backend.api.routers.ml import _run_single_model_prediction, MLModelType

        mm = _make_mock_model_manager(model_available=True)
        real_features = np.zeros((1, 30, 5), dtype=np.float32)

        result = await _run_single_model_prediction(
            model_manager=mm,
            model_type=MLModelType.LSTM,
            ticker="AAPL",
            horizon_days=3,
            base_price=150.0,
            feature_data=real_features,
        )
        assert result is not None

    def test_no_random_randn_in_single_model_prediction_source(self):
        """
        Verify np.random.randn no longer appears in the LSTM/XGBoost inference
        path inside _run_single_model_prediction.
        """
        if "backend.api.routers.ml" in sys.modules:
            del sys.modules["backend.api.routers.ml"]
        from backend.api.routers.ml import _run_single_model_prediction

        source = inspect.getsource(_run_single_model_prediction)
        assert "np.random.randn" not in source, (
            "np.random.randn found in _run_single_model_prediction — "
            "synthetic feature fabrication still present (Finding #200)"
        )


# ===========================================================================
# 2. backend/ml/backtesting.py — synthetic market / benchmark data
# ===========================================================================

class TestBacktestEngineNoFabricatedData:
    """
    BacktestEngine._get_market_data / _get_benchmark_data must refuse
    fabrication when no data_provider is configured (T1.3).
    """

    def test_get_market_data_raises_without_provider(self):
        from datetime import datetime
        from backend.exceptions import ModelUnavailableError
        from backend.ml.backtesting import BacktestEngine

        engine = BacktestEngine(data_provider=None)

        with pytest.raises(ModelUnavailableError, match="no_data_provider"):
            engine._get_market_data(
                ["AAPL"],
                datetime(2023, 1, 1),
                datetime(2023, 6, 1),
            )

    def test_get_benchmark_data_raises_without_provider(self):
        from datetime import datetime
        from backend.exceptions import ModelUnavailableError
        from backend.ml.backtesting import BacktestEngine

        engine = BacktestEngine(data_provider=None)

        with pytest.raises(ModelUnavailableError, match="no_data_provider"):
            engine._get_benchmark_data(
                "SPY",
                datetime(2023, 1, 1),
                datetime(2023, 6, 1),
            )

    def test_get_market_data_raises_on_provider_failure(self):
        from datetime import datetime
        from backend.ml.backtesting import BacktestEngine

        failing_provider = MagicMock()
        failing_provider.get_bulk_historical_prices.side_effect = ConnectionError(
            "timeout"
        )
        engine = BacktestEngine(data_provider=failing_provider)

        # Fail-loud: provider errors must surface, never fall through to synthetic
        with pytest.raises(ConnectionError, match="timeout"):
            engine._get_market_data(
                ["AAPL"],
                datetime(2023, 1, 1),
                datetime(2023, 6, 1),
            )

    def test_get_benchmark_data_raises_on_provider_failure(self):
        from datetime import datetime
        from backend.ml.backtesting import BacktestEngine

        failing_provider = MagicMock()
        failing_provider.get_historical_prices.side_effect = ConnectionError("timeout")
        engine = BacktestEngine(data_provider=failing_provider)

        with pytest.raises(ConnectionError, match="timeout"):
            engine._get_benchmark_data(
                "SPY",
                datetime(2023, 1, 1),
                datetime(2023, 6, 1),
            )

    def test_no_random_seed_in_market_data_path(self):
        """
        Default path refuses fabrication; synthetic np.random is gated behind
        allow_synthetic=True (test-only opt-in).
        """
        from datetime import datetime
        from backend.exceptions import ModelUnavailableError
        from backend.ml.backtesting import BacktestEngine

        source = inspect.getsource(BacktestEngine._get_market_data)
        assert "if not self.allow_synthetic" in source
        assert "ModelUnavailableError" in source

        default = BacktestEngine(data_provider=None, allow_synthetic=False)
        with pytest.raises(ModelUnavailableError):
            default._get_market_data(
                ["AAPL"], datetime(2023, 1, 1), datetime(2023, 6, 1)
            )

        synthetic = BacktestEngine(data_provider=None, allow_synthetic=True)
        data = synthetic._get_market_data(
            ["AAPL"], datetime(2023, 1, 1), datetime(2023, 1, 10)
        )
        assert "AAPL" in data

    def test_no_random_seed_in_benchmark_data_path(self):
        from datetime import datetime
        from backend.exceptions import ModelUnavailableError
        from backend.ml.backtesting import BacktestEngine

        source = inspect.getsource(BacktestEngine._get_benchmark_data)
        assert "if not self.allow_synthetic" in source
        assert "ModelUnavailableError" in source

        default = BacktestEngine(data_provider=None, allow_synthetic=False)
        with pytest.raises(ModelUnavailableError):
            default._get_benchmark_data(
                "SPY", datetime(2023, 1, 1), datetime(2023, 6, 1)
            )

        synthetic = BacktestEngine(data_provider=None, allow_synthetic=True)
        frame = synthetic._get_benchmark_data(
            "SPY", datetime(2023, 1, 1), datetime(2023, 1, 10)
        )
        assert not frame.empty


# ===========================================================================
# 3. backend/ml/training_pipeline.py — random metric gate
# ===========================================================================

class TestTrainingPipelineNoFabricatedMetrics:
    """
    MLTrainingPipeline.train_models must raise RuntimeError (not quietly store
    np.random.uniform metrics) when the pipeline returns no real metrics.
    """

    @pytest.mark.asyncio
    async def test_train_models_raises_when_no_real_metrics(self, tmp_path):
        """
        When the orchestrator/pipeline provides no metrics, train_models must
        record the failure rather than store random values.
        """
        # Stub out all heavy imports before importing training_pipeline
        for _heavy in [
            "backend.ml.pipeline",
            "backend.ml.pipeline.implementations",
            "backend.ml.pipeline.registry",
            "backend.ml.pipeline.monitoring",
            "backend.ml.pipeline.deployment",
        ]:
            sys.modules[_heavy] = MagicMock()

        # Force re-import with stubs in place
        if "backend.ml.training_pipeline" in sys.modules:
            del sys.modules["backend.ml.training_pipeline"]

        # Patch dotenv so it does not try to read a .env file
        with patch("dotenv.load_dotenv"):
            from backend.ml.training_pipeline import MLTrainingPipeline
            import pandas as pd
            import numpy as np

        pipeline_obj = MLTrainingPipeline.__new__(MLTrainingPipeline)
        pipeline_obj.config = {
            "models_path": str(tmp_path),
            "logs_path": str(tmp_path),
            "registry_path": str(tmp_path),
            "training_data_path": str(tmp_path),
            "predictions_path": str(tmp_path),
            "enable_auto_retraining": False,
            "performance_threshold": 0.75,
            "data_drift_threshold": 0.3,
            "daily_cost_limit": 10.0,
            "database_url": None,
        }

        # Mock pipeline that returns no metrics
        mock_pipeline = MagicMock()
        mock_pipeline.get_metrics.return_value = None

        mock_orchestrator = AsyncMock()
        mock_orchestrator.submit_pipeline.return_value = "pipe-001"
        mock_orchestrator.get_pipeline_metrics.return_value = None

        pipeline_obj.orchestrator = mock_orchestrator

        dummy_data = pd.DataFrame({
            "open": np.ones(100),
            "high": np.ones(100),
            "low": np.ones(100),
            "close": np.ones(100),
            "volume": np.ones(100),
            "sma_20": np.ones(100),
            "sma_50": np.ones(100),
            "rsi_14": np.ones(100),
            "macd": np.zeros(100),
            "bollinger_upper": np.ones(100),
            "bollinger_lower": np.ones(100),
            "future_return": np.zeros(100),
        })

        with patch("backend.ml.training_pipeline.create_pipeline", return_value=mock_pipeline):
            results = await pipeline_obj.train_models(dummy_data)

        # All models must have failed (RuntimeError for missing metrics) rather
        # than succeeding with random metrics.
        for model_name, result in results.items():
            assert result["status"] == "failed", (
                f"Model '{model_name}' should have failed when no real metrics are "
                f"available, but got status='{result['status']}'"
            )
            assert "error" in result, (
                f"Model '{model_name}' failure result missing 'error' field"
            )

    def test_no_random_uniform_in_train_models_source(self):
        """Verify np.random.uniform no longer gates metric collection."""
        for _heavy in [
            "backend.ml.pipeline",
            "backend.ml.pipeline.implementations",
            "backend.ml.pipeline.registry",
            "backend.ml.pipeline.monitoring",
            "backend.ml.pipeline.deployment",
        ]:
            sys.modules[_heavy] = MagicMock()

        if "backend.ml.training_pipeline" in sys.modules:
            del sys.modules["backend.ml.training_pipeline"]

        with patch("dotenv.load_dotenv"):
            from backend.ml.training_pipeline import MLTrainingPipeline

        source = inspect.getsource(MLTrainingPipeline.train_models)
        assert "np.random.uniform" not in source, (
            "np.random.uniform found in train_models — random metrics still gate "
            "production promotion (Finding #200)"
        )


# ===========================================================================
# 4. backend/api/routers/analysis.py — hardcoded indicator / fundamental fallbacks
# ===========================================================================

class TestAnalysisRouterNoFabricatedData:
    """
    The analysis router must not return hardcoded financial values when providers
    fail; it must raise 503 or return None for optional fields.
    """

    def test_no_hardcoded_bollinger_in_indicators_source(self):
        """
        The /indicators/{symbol} endpoint must not have hardcoded 155/150/145
        Bollinger Band values wired as the default response.
        """
        from backend.api.routers import analysis as analysis_module

        source = inspect.getsource(analysis_module.get_technical_indicators)
        assert '"upper": 155.0' not in source, (
            'Hardcoded Bollinger upper=155.0 still present in get_technical_indicators'
        )
        assert '"middle": 150.0' not in source, (
            'Hardcoded Bollinger middle=150.0 still present in get_technical_indicators'
        )
        assert '"lower": 145.0' not in source, (
            'Hardcoded Bollinger lower=145.0 still present in get_technical_indicators'
        )

    def test_no_hardcoded_moving_averages_in_indicators_source(self):
        from backend.api.routers import analysis as analysis_module

        source = inspect.getsource(analysis_module.get_technical_indicators)
        assert '"sma_20": 150.5' not in source, (
            'Hardcoded sma_20=150.5 still present in get_technical_indicators'
        )
        assert '"sma_50": 148.2' not in source, (
            'Hardcoded sma_50=148.2 still present in get_technical_indicators'
        )

    def test_no_hardcoded_fundamental_fallback_in_analyze_stock_source(self):
        """
        The fundamental fallback block must not contain hardcoded pe_ratio=25.5
        or other hardcoded values.
        """
        from backend.api.routers import analysis as analysis_module

        source = inspect.getsource(analysis_module.analyze_stock)
        assert "pe_ratio=25.5" not in source, (
            "Hardcoded pe_ratio=25.5 fundamental fallback still present in analyze_stock"
        )
        assert "intrinsic_value=165.0" not in source, (
            "Hardcoded intrinsic_value=165.0 fundamental fallback still present"
        )

    def test_no_hardcoded_ml_predictions_fallback_in_analyze_stock_source(self):
        """
        The ML predictions fallback must not return hardcoded price forecasts.
        """
        from backend.api.routers import analysis as analysis_module

        source = inspect.getsource(analysis_module.analyze_stock)
        assert "price_prediction_1d=152.5" not in source, (
            "Hardcoded ML prediction price_prediction_1d=152.5 still present"
        )
        assert "price_prediction_7d=155.0" not in source, (
            "Hardcoded ML prediction price_prediction_7d=155.0 still present"
        )

    def test_no_hardcoded_sentiment_fallback_in_analyze_stock_source(self):
        """
        The sentiment error-fallback must not return fabricated positive sentiment.
        """
        from backend.api.routers import analysis as analysis_module

        source = inspect.getsource(analysis_module.analyze_stock)
        assert 'overall_sentiment=0.35' not in source, (
            "Hardcoded sentiment overall_sentiment=0.35 fallback still present"
        )
        assert '"earnings beat"' not in source, (
            "Hardcoded sentiment key_topics still present in analyze_stock"
        )

    def test_no_price_history_raises_503_not_mock_data(self):
        """
        When price_history is empty, analyze_stock must raise HTTP 503 rather
        than populating a TechnicalIndicators object with placeholder numbers.
        """
        from backend.api.routers import analysis as analysis_module

        source = inspect.getsource(analysis_module.analyze_stock)
        # The old "using mock data" log message should be gone
        assert "using mock data" not in source, (
            "Legacy 'using mock data' comment/log still present — "
            "check that hardcoded technical fallback was removed"
        )
