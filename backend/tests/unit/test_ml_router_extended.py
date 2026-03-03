"""
Unit tests for the extended ML router endpoints:
- Drift detection
- Model version management
- Backtesting
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from types import SimpleNamespace


# ============================================================================
# Drift Detection
# ============================================================================

class TestDetectDrift:
    """Tests for POST /drift/detect."""

    @pytest.mark.asyncio
    @patch("backend.ml.drift_detection.DriftDetector")
    async def test_detect_drift_no_reference(self, MockDetector):
        from backend.api.routers.ml import detect_drift

        mock_detector = MagicMock()
        mock_detector.reference_distributions = {}
        MockDetector.return_value = mock_detector

        result = await detect_drift(
            model_name="lstm_price_predictor",
            current_user=MagicMock(),
        )
        assert result.success is True
        assert result.data["status"] == "no_reference"

    @pytest.mark.asyncio
    async def test_detect_drift_with_reference(self):
        from backend.api.routers.ml import detect_drift

        mock_result = SimpleNamespace(drift_detected=True, details={"psi": 0.35})
        mock_detector = MagicMock()
        mock_detector.reference_distributions = {"my_model": {"feature_a": {}}}
        mock_detector.detect_data_drift.return_value = mock_result

        with patch(
            "backend.ml.drift_detection.DriftDetector",
            return_value=mock_detector,
        ):
            result = await detect_drift(
                model_name="my_model",
                current_user=MagicMock(),
            )
        assert result.success is True
        assert result.data["drift_detected"] is True


class TestGetDriftStatus:
    """Tests for GET /drift/status."""

    @pytest.mark.asyncio
    async def test_get_drift_status_empty(self):
        from backend.api.routers.ml import get_drift_status

        mock_detector = MagicMock()
        mock_detector.reference_distributions = {}

        with patch(
            "backend.ml.drift_detection.DriftDetector",
            return_value=mock_detector,
        ):
            result = await get_drift_status(current_user=MagicMock())
        assert result.success is True
        assert result.data["total"] == 0

    @pytest.mark.asyncio
    async def test_get_drift_status_with_models(self):
        from backend.api.routers.ml import get_drift_status

        mock_detector = MagicMock()
        mock_detector.reference_distributions = {
            "lstm": {"feature_a": {}},
            "xgboost": {"feature_b": {}},
        }

        with patch(
            "backend.ml.drift_detection.DriftDetector",
            return_value=mock_detector,
        ):
            result = await get_drift_status(current_user=MagicMock())
        assert result.data["total"] == 2
        assert "lstm" in result.data["models_with_reference"]


# ============================================================================
# Model Version Management
# ============================================================================

class TestListModelVersions:
    """Tests for GET /versions."""

    @pytest.mark.asyncio
    async def test_list_versions_empty(self):
        from backend.api.routers.ml import list_model_versions

        mock_manager = MagicMock()
        mock_manager.model_registry = {}

        with patch(
            "backend.ml.model_versioning.get_model_version_manager",
            return_value=mock_manager,
        ):
            result = await list_model_versions(current_user=MagicMock())
        assert result.success is True
        assert result.data["total_models"] == 0

    @pytest.mark.asyncio
    async def test_list_versions_with_models(self):
        from backend.api.routers.ml import list_model_versions

        mock_version = SimpleNamespace(
            stage=SimpleNamespace(value="production"),
            is_champion=True,
            created_at=None,
        )
        mock_manager = MagicMock()
        mock_manager.model_registry = {"lstm": {"v1": mock_version}}

        with patch(
            "backend.ml.model_versioning.get_model_version_manager",
            return_value=mock_manager,
        ):
            result = await list_model_versions(current_user=MagicMock())
        assert result.data["total_models"] == 1
        assert "lstm" in result.data["models"]


class TestPromoteModelVersion:
    """Tests for POST /versions/{model_name}/promote."""

    @pytest.mark.asyncio
    async def test_promote_success(self):
        from backend.api.routers.ml import promote_model_version

        mock_manager = MagicMock()
        mock_manager.promote_model.return_value = True

        with patch(
            "backend.ml.model_versioning.get_model_version_manager",
            return_value=mock_manager,
        ):
            result = await promote_model_version(
                model_name="lstm",
                version="v2",
                target_stage="production",
                current_user=MagicMock(),
            )
        assert result.success is True
        assert result.data["promoted"] is True

    @pytest.mark.asyncio
    async def test_promote_not_found(self):
        from fastapi import HTTPException
        from backend.api.routers.ml import promote_model_version

        mock_manager = MagicMock()
        mock_manager.promote_model.return_value = False

        with patch(
            "backend.ml.model_versioning.get_model_version_manager",
            return_value=mock_manager,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await promote_model_version(
                    model_name="missing",
                    version="v1",
                    target_stage="production",
                    current_user=MagicMock(),
                )
            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_promote_invalid_stage(self):
        from fastapi import HTTPException
        from backend.api.routers.ml import promote_model_version

        mock_manager = MagicMock()

        with patch(
            "backend.ml.model_versioning.get_model_version_manager",
            return_value=mock_manager,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await promote_model_version(
                    model_name="lstm",
                    version="v1",
                    target_stage="invalid_stage",
                    current_user=MagicMock(),
                )
            assert exc_info.value.status_code == 400


class TestRollbackModelVersion:
    """Tests for POST /versions/{model_name}/rollback."""

    @pytest.mark.asyncio
    async def test_rollback_success(self):
        from backend.api.routers.ml import rollback_model_version

        mock_manager = MagicMock()
        mock_manager.rollback_model.return_value = True

        with patch(
            "backend.ml.model_versioning.get_model_version_manager",
            return_value=mock_manager,
        ):
            result = await rollback_model_version(
                model_name="lstm",
                target_version="v1",
                current_user=MagicMock(),
            )
        assert result.success is True
        assert result.data["rolled_back_to"] == "v1"

    @pytest.mark.asyncio
    async def test_rollback_not_found(self):
        from fastapi import HTTPException
        from backend.api.routers.ml import rollback_model_version

        mock_manager = MagicMock()
        mock_manager.rollback_model.return_value = False

        with patch(
            "backend.ml.model_versioning.get_model_version_manager",
            return_value=mock_manager,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await rollback_model_version(
                    model_name="lstm",
                    target_version="v0",
                    current_user=MagicMock(),
                )
            assert exc_info.value.status_code == 404


# ============================================================================
# Backtesting
# ============================================================================

class TestRunBacktest:
    """Tests for POST /backtest."""

    @pytest.mark.asyncio
    async def test_backtest_success(self):
        from backend.api.routers.ml import run_backtest, BacktestRequest

        mock_result = SimpleNamespace(
            total_return=0.15,
            sharpe_ratio=1.2,
            max_drawdown=-0.08,
            total_trades=24,
        )
        mock_engine = MagicMock()
        mock_engine.backtest_strategy.return_value = mock_result

        with patch(
            "backend.ml.backtesting.get_backtest_engine",
            return_value=mock_engine,
        ):
            req = BacktestRequest(
                tickers=["AAPL", "MSFT"],
                start_date="2025-01-01",
                end_date="2025-12-31",
                initial_capital=100000,
                benchmark="SPY",
            )
            result = await run_backtest(
                request_body=req,
                current_user=MagicMock(),
            )
        assert result.success is True
        assert result.data["total_return"] == 0.15
        assert result.data["sharpe_ratio"] == 1.2

    @pytest.mark.asyncio
    async def test_backtest_failure(self):
        from fastapi import HTTPException
        from backend.api.routers.ml import run_backtest, BacktestRequest

        with patch(
            "backend.ml.backtesting.get_backtest_engine",
            side_effect=RuntimeError("Engine init failed"),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await run_backtest(
                    request_body=BacktestRequest(
                        tickers=["AAPL"],
                        start_date="2025-01-01",
                        end_date="2025-06-30",
                    ),
                    current_user=MagicMock(),
                )
            assert exc_info.value.status_code == 500

    def test_backtest_request_model(self):
        from backend.api.routers.ml import BacktestRequest

        req = BacktestRequest(
            tickers=["AAPL"],
            start_date="2025-01-01",
            end_date="2025-12-31",
        )
        assert req.initial_capital == 100000.0
        assert req.benchmark == "SPY"
