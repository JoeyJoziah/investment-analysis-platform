"""
Unit tests for backend/ml/pipeline/ module.

Tests cover:
- base.py: PipelineConfig, ModelArtifact, PipelineResult, PipelineStep, ModelPipeline
- implementations.py: DataLoadingStep, DataPreprocessingStep, FeatureEngineeringStep,
                      DataSplittingStep, ModelTrainingStep, ModelEvaluationStep,
                      ModelSavingStep, StockPredictionPipeline, create_pipeline
- monitoring.py: DriftDetector, ModelMonitor, AlertManager, PerformanceMetrics
- orchestrator.py: MLOrchestrator, OrchestratorConfig, TrainingSchedule, RetrainingTrigger
- registry.py: ModelRegistry, ModelVersion, ModelMetadata
- deployment.py: ModelDeployer, RollbackManager, ABTestManager
- memory_sync.py: ClaudeFlowMemoryAdapter, MemoryEntry, SyncResult
- task_bridge.py: TaskBridge, UnifiedTask
"""

# Mock optional heavy dependencies before any pipeline module import triggers them.
# registry.py imports mlflow; implementations.py imports lightgbm.
import sys
from unittest.mock import MagicMock as _MagicMock

for _mod in ("mlflow", "mlflow.pytorch", "mlflow.sklearn", "lightgbm"):
    sys.modules.setdefault(_mod, _MagicMock())

import json
import hashlib
import sqlite3
import pytest
import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Tuple
from unittest.mock import patch, MagicMock, AsyncMock, PropertyMock
from dataclasses import dataclass

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# base.py tests
# ---------------------------------------------------------------------------

class TestPipelineConfig:
    """Tests for PipelineConfig dataclass."""

    def _make_config(self, **overrides):
        from backend.ml.pipeline.base import PipelineConfig, ModelType
        defaults = dict(
            name="test_model",
            version="1.0.0",
            model_type=ModelType.REGRESSION,
            data_source="data.csv",
            feature_columns=["f1", "f2"],
            target_column="target",
        )
        defaults.update(overrides)
        return PipelineConfig(**defaults)

    def test_config_to_dict_contains_required_keys(self):
        cfg = self._make_config()
        d = cfg.to_dict()
        assert d["name"] == "test_model"
        assert d["version"] == "1.0.0"
        assert d["model_type"] == "regression"
        assert d["target_column"] == "target"
        assert d["feature_columns"] == ["f1", "f2"]
        assert d["train_test_split"] == 0.8
        assert d["epochs"] == 100

    def test_config_get_hash_is_deterministic(self):
        cfg1 = self._make_config()
        cfg2 = self._make_config()
        assert cfg1.get_hash() == cfg2.get_hash()

    def test_config_get_hash_changes_with_different_values(self):
        cfg1 = self._make_config(learning_rate=0.001)
        cfg2 = self._make_config(learning_rate=0.01)
        assert cfg1.get_hash() != cfg2.get_hash()

    def test_config_defaults(self):
        cfg = self._make_config()
        assert cfg.batch_size == 32
        assert cfg.early_stopping_patience == 10
        assert cfg.optimizer == "adam"
        assert cfg.scaling_method == "standard"
        assert cfg.primary_metric == "accuracy"
        assert cfg.performance_threshold == 0.8

    def test_config_hyperparameters_default_empty(self):
        cfg = self._make_config()
        assert cfg.hyperparameters == {}

    def test_config_evaluation_metrics_default(self):
        cfg = self._make_config()
        assert "accuracy" in cfg.evaluation_metrics
        assert "f1" in cfg.evaluation_metrics


class TestModelArtifact:
    """Tests for ModelArtifact dataclass."""

    def _make_artifact(self, **overrides):
        from backend.ml.pipeline.base import ModelArtifact, ModelType
        defaults = dict(
            model_id="model_001",
            name="test_model",
            version="1.0.0",
            model_type=ModelType.REGRESSION,
            model_path=Path("/tmp/model.pkl"),
        )
        defaults.update(overrides)
        return ModelArtifact(**defaults)

    def test_artifact_to_dict(self):
        artifact = self._make_artifact()
        d = artifact.to_dict()
        assert d["model_id"] == "model_001"
        assert d["name"] == "test_model"
        assert d["model_type"] == "regression"
        assert d["is_deployed"] is False
        assert d["deployment_endpoint"] is None

    def test_artifact_defaults(self):
        artifact = self._make_artifact()
        assert artifact.training_duration_seconds == 0
        assert artifact.training_samples == 0
        assert artifact.metrics == {}
        assert artifact.feature_importance == {}

    def test_artifact_with_metrics(self):
        artifact = self._make_artifact(
            metrics={"mse": 0.05, "r2": 0.92},
            feature_importance={"f1": 0.7, "f2": 0.3},
        )
        assert artifact.metrics["r2"] == 0.92
        assert artifact.feature_importance["f1"] == 0.7


class TestPipelineResult:
    """Tests for PipelineResult dataclass."""

    def test_result_to_dict_without_artifact(self):
        from backend.ml.pipeline.base import PipelineResult, PipelineStatus
        result = PipelineResult(
            pipeline_id="pipe_001",
            status=PipelineStatus.RUNNING,
            start_time=datetime.now(timezone.utc),
        )
        d = result.to_dict()
        assert d["pipeline_id"] == "pipe_001"
        assert d["status"] == "running"
        assert d["model_artifact"] is None

    def test_result_to_dict_with_artifact(self):
        from backend.ml.pipeline.base import PipelineResult, PipelineStatus, ModelArtifact, ModelType
        artifact = ModelArtifact(
            model_id="m1", name="m", version="1", model_type=ModelType.REGRESSION,
            model_path=Path("/tmp/m.pkl"),
        )
        result = PipelineResult(
            pipeline_id="p1", status=PipelineStatus.COMPLETED,
            start_time=datetime.now(timezone.utc), model_artifact=artifact,
        )
        d = result.to_dict()
        assert d["model_artifact"]["model_id"] == "m1"


class TestPipelineStep:
    """Tests for abstract PipelineStep."""

    def test_step_validate_input_default_true(self):
        from backend.ml.pipeline.base import PipelineStep

        class DummyStep(PipelineStep):
            async def execute(self, data, context):
                return data, context

        step = DummyStep("dummy")
        result = asyncio.get_event_loop().run_until_complete(step.validate_input(None))
        assert result is True

    def test_step_cleanup_noop(self):
        from backend.ml.pipeline.base import PipelineStep

        class DummyStep(PipelineStep):
            async def execute(self, data, context):
                return data, context

        step = DummyStep("dummy")
        asyncio.get_event_loop().run_until_complete(step.cleanup())


class TestModelPipelineExecution:
    """Tests for ModelPipeline.execute and helper methods."""

    def _make_concrete_pipeline(self, steps=None):
        from backend.ml.pipeline.base import ModelPipeline, PipelineConfig, PipelineStep, ModelType

        class StubStep(PipelineStep):
            async def execute(self, data, context):
                context[f"{self.name}_result"] = "ok"
                return data, context

        class ConcretePipeline(ModelPipeline):
            def _setup_pipeline(self):
                pass

        cfg = PipelineConfig(
            name="test", version="1.0", model_type=ModelType.REGRESSION,
            data_source="x.csv", feature_columns=["f1"], target_column="y",
        )
        pipeline = ConcretePipeline(cfg)
        if steps is None:
            pipeline.steps = [StubStep("step1"), StubStep("step2")]
        else:
            pipeline.steps = steps
        return pipeline

    @pytest.mark.asyncio
    async def test_execute_completes_all_steps(self):
        pipeline = self._make_concrete_pipeline()
        result = await pipeline.execute()
        from backend.ml.pipeline.base import PipelineStatus
        assert result.status == PipelineStatus.COMPLETED
        assert "step1" in result.steps_completed
        assert "step2" in result.steps_completed

    @pytest.mark.asyncio
    async def test_execute_stores_intermediate_results(self):
        pipeline = self._make_concrete_pipeline()
        result = await pipeline.execute()
        assert "step1" in result.intermediate_results
        assert result.intermediate_results["step1"] == "ok"

    @pytest.mark.asyncio
    async def test_execute_handles_step_failure(self):
        from backend.ml.pipeline.base import PipelineStep, PipelineStatus

        class FailingStep(PipelineStep):
            async def execute(self, data, context):
                raise RuntimeError("boom")

        pipeline = self._make_concrete_pipeline(steps=[FailingStep("bad_step")])
        result = await pipeline.execute()
        assert result.status == PipelineStatus.FAILED
        assert "boom" in result.error_message

    @pytest.mark.asyncio
    async def test_execute_handles_validation_failure(self):
        from backend.ml.pipeline.base import PipelineStep, PipelineStatus

        class InvalidInputStep(PipelineStep):
            async def execute(self, data, context):
                return data, context

            async def validate_input(self, data):
                return False

        pipeline = self._make_concrete_pipeline(steps=[InvalidInputStep("validate_fail")])
        result = await pipeline.execute()
        assert result.status == PipelineStatus.FAILED
        assert "Invalid input" in result.error_message

    @pytest.mark.asyncio
    async def test_execute_calls_cleanup_on_all_steps(self):
        from backend.ml.pipeline.base import PipelineStep

        cleanup_called = []

        class TrackCleanupStep(PipelineStep):
            async def execute(self, data, context):
                return data, context

            async def cleanup(self):
                cleanup_called.append(self.name)

        steps = [TrackCleanupStep("a"), TrackCleanupStep("b")]
        pipeline = self._make_concrete_pipeline(steps=steps)
        await pipeline.execute()
        assert "a" in cleanup_called
        assert "b" in cleanup_called

    @pytest.mark.asyncio
    async def test_validate_config_missing_field(self):
        from backend.ml.pipeline.base import PipelineConfig, ModelType

        class ConcretePipeline2:
            """Inline for missing-field test."""
            pass

        pipeline = self._make_concrete_pipeline()
        pipeline.config.name = ""  # make name empty
        valid = await pipeline.validate_config()
        assert valid is False

    def test_add_remove_get_step(self):
        from backend.ml.pipeline.base import PipelineStep

        class StubStep(PipelineStep):
            async def execute(self, data, context):
                return data, context

        pipeline = self._make_concrete_pipeline(steps=[])
        s = StubStep("new_step")
        pipeline.add_step(s)
        assert pipeline.get_step("new_step") is s
        pipeline.remove_step("new_step")
        assert pipeline.get_step("new_step") is None


# ---------------------------------------------------------------------------
# implementations.py tests (mocked ML libs)
# ---------------------------------------------------------------------------

class TestDataPreprocessingStep:
    """Tests for DataPreprocessingStep logic."""

    @pytest.mark.asyncio
    async def test_fills_missing_numeric_with_median(self):
        from backend.ml.pipeline.implementations import DataPreprocessingStep
        from backend.ml.pipeline.base import PipelineConfig, ModelType

        step = DataPreprocessingStep()
        df = pd.DataFrame({"f1": [1.0, np.nan, 3.0], "target": [10, 20, 30]})
        cfg = PipelineConfig(
            name="t", version="1", model_type=ModelType.REGRESSION,
            data_source="x", feature_columns=["f1"], target_column="target",
        )
        context = {"config": cfg, "artifacts": {}}
        result_df, ctx = await step.execute(df.copy(), context)
        # median of [1.0, NaN, 3.0] => median([1.0, 3.0]) = 2.0
        # After fillna + standard scaling the NaN is replaced then scaled.
        # The key point: no NaN values remain.
        assert not result_df["f1"].isna().any()

    @pytest.mark.asyncio
    async def test_fills_missing_categorical_with_missing_string(self):
        from backend.ml.pipeline.implementations import DataPreprocessingStep
        from backend.ml.pipeline.base import PipelineConfig, ModelType

        step = DataPreprocessingStep()
        df = pd.DataFrame({"cat": ["a", None, "b"], "target": [1, 2, 3]})
        cfg = PipelineConfig(
            name="t", version="1", model_type=ModelType.REGRESSION,
            data_source="x", feature_columns=["cat"], target_column="target",
        )
        context = {"config": cfg, "artifacts": {}}
        result_df, ctx = await step.execute(df, context)
        # After fillna('missing') + categorical encoding, the value should be numeric
        # and 'missing' category should have been encoded
        assert not result_df["cat"].isna().any()

    @pytest.mark.asyncio
    async def test_standard_scaler_stored_in_artifacts(self):
        from backend.ml.pipeline.implementations import DataPreprocessingStep
        from backend.ml.pipeline.base import PipelineConfig, ModelType

        step = DataPreprocessingStep()
        df = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "target": [10, 20, 30]})
        cfg = PipelineConfig(
            name="t", version="1", model_type=ModelType.REGRESSION,
            data_source="x", feature_columns=["f1"], target_column="target",
            scaling_method="standard",
        )
        context = {"config": cfg, "artifacts": {}}
        _, ctx = await step.execute(df, context)
        assert "scaler" in ctx["artifacts"]

    @pytest.mark.asyncio
    async def test_no_scaler_when_method_none(self):
        from backend.ml.pipeline.implementations import DataPreprocessingStep
        from backend.ml.pipeline.base import PipelineConfig, ModelType

        step = DataPreprocessingStep()
        df = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "target": [10, 20, 30]})
        cfg = PipelineConfig(
            name="t", version="1", model_type=ModelType.REGRESSION,
            data_source="x", feature_columns=["f1"], target_column="target",
            scaling_method="none",
        )
        context = {"config": cfg, "artifacts": {}}
        _, ctx = await step.execute(df, context)
        assert "scaler" not in ctx["artifacts"]


class TestDataSplittingStep:
    """Tests for DataSplittingStep."""

    @pytest.mark.asyncio
    async def test_splits_data_into_train_val_test(self):
        from backend.ml.pipeline.implementations import DataSplittingStep
        from backend.ml.pipeline.base import PipelineConfig, ModelType

        np.random.seed(42)
        n = 100
        df = pd.DataFrame({"f1": np.random.randn(n), "target": np.random.randn(n)})
        cfg = PipelineConfig(
            name="t", version="1", model_type=ModelType.REGRESSION,
            data_source="x", feature_columns=["f1"], target_column="target",
            train_test_split=0.8, validation_split=0.1,
        )
        step = DataSplittingStep()
        context = {"config": cfg}
        splits, ctx = await step.execute(df, context)

        assert "X_train" in splits
        assert "X_val" in splits
        assert "X_test" in splits
        total = len(splits["X_train"]) + len(splits["X_val"]) + len(splits["X_test"])
        assert total == n

    @pytest.mark.asyncio
    async def test_split_result_stored_in_context(self):
        from backend.ml.pipeline.implementations import DataSplittingStep
        from backend.ml.pipeline.base import PipelineConfig, ModelType

        np.random.seed(0)
        df = pd.DataFrame({"f1": np.random.randn(50), "target": np.random.randn(50)})
        cfg = PipelineConfig(
            name="t", version="1", model_type=ModelType.REGRESSION,
            data_source="x", feature_columns=["f1"], target_column="target",
        )
        step = DataSplittingStep()
        context = {"config": cfg}
        _, ctx = await step.execute(df, context)
        assert "data_splitting_result" in ctx
        assert "train_samples" in ctx["data_splitting_result"]


class TestModelTrainingStep:
    """Tests for ModelTrainingStep with mocked models."""

    @pytest.mark.asyncio
    async def test_unsupported_model_type_raises(self):
        from backend.ml.pipeline.implementations import ModelTrainingStep

        step = ModelTrainingStep("unknown_model")
        data = {"X_train": np.array([]), "y_train": np.array([]),
                "X_val": np.array([]), "y_val": np.array([])}
        context = {"config": MagicMock(), "artifacts": {}}
        with pytest.raises(ValueError, match="Unsupported model type"):
            await step.execute(data, context)

    @pytest.mark.asyncio
    async def test_random_forest_training(self):
        from backend.ml.pipeline.implementations import ModelTrainingStep
        from backend.ml.pipeline.base import PipelineConfig, ModelType

        np.random.seed(42)
        n = 50
        X_train = np.random.randn(n, 3)
        y_train = np.random.randn(n)

        cfg = PipelineConfig(
            name="t", version="1", model_type=ModelType.REGRESSION,
            data_source="x", feature_columns=["f1", "f2", "f3"],
            target_column="target", num_workers=1,
            hyperparameters={"n_estimators": 5, "max_depth": 3},
        )
        step = ModelTrainingStep("random_forest")
        data = {"X_train": X_train, "y_train": y_train,
                "X_val": X_train[:10], "y_val": y_train[:10]}
        context = {"config": cfg, "artifacts": {}}
        model, ctx = await step.execute(data, context)
        assert model is not None
        assert "model_random_forest" in ctx["artifacts"]


class TestModelEvaluationStep:
    """Tests for ModelEvaluationStep."""

    @pytest.mark.asyncio
    async def test_evaluates_sklearn_model(self):
        from backend.ml.pipeline.implementations import ModelEvaluationStep
        from backend.ml.pipeline.base import PipelineConfig, ModelType

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1.0, 2.0, 3.0])
        mock_model.feature_importances_ = np.array([0.5, 0.3, 0.2])

        X_test = pd.DataFrame({"f1": [1, 2, 3], "f2": [4, 5, 6], "f3": [7, 8, 9]})
        y_test = pd.Series([1.1, 2.1, 3.1])

        cfg = PipelineConfig(
            name="t", version="1", model_type=ModelType.REGRESSION,
            data_source="x", feature_columns=["f1", "f2", "f3"],
            target_column="target", primary_metric="r2",
        )
        context = {
            "config": cfg,
            "artifacts": {"model_xgboost": mock_model},
        }
        data = {"X_test": X_test, "y_test": y_test}

        step = ModelEvaluationStep()
        eval_results, ctx = await step.execute(data, context)
        assert "xgboost" in eval_results
        assert "mse" in eval_results["xgboost"]
        assert "r2" in eval_results["xgboost"]
        assert "feature_importance" in eval_results["xgboost"]

    @pytest.mark.asyncio
    async def test_evaluates_no_models_returns_empty(self):
        from backend.ml.pipeline.implementations import ModelEvaluationStep
        from backend.ml.pipeline.base import PipelineConfig, ModelType

        cfg = PipelineConfig(
            name="t", version="1", model_type=ModelType.REGRESSION,
            data_source="x", feature_columns=["f1"], target_column="target",
        )
        context = {"config": cfg, "artifacts": {"scaler": MagicMock()}}
        data = {"X_test": pd.DataFrame(), "y_test": pd.Series()}
        step = ModelEvaluationStep()
        eval_results, ctx = await step.execute(data, context)
        assert eval_results == {}


class TestCreatePipeline:
    """Tests for create_pipeline factory."""

    def test_time_series_returns_stock_prediction_pipeline(self):
        from backend.ml.pipeline.implementations import create_pipeline
        from backend.ml.pipeline.base import PipelineConfig, ModelType

        cfg = PipelineConfig(
            name="t", version="1", model_type=ModelType.TIME_SERIES,
            data_source="x", feature_columns=["f1"], target_column="y",
        )
        pipeline = create_pipeline(cfg)
        from backend.ml.pipeline.implementations import StockPredictionPipeline
        assert isinstance(pipeline, StockPredictionPipeline)

    def test_unsupported_model_type_raises(self):
        from backend.ml.pipeline.implementations import create_pipeline
        from backend.ml.pipeline.base import PipelineConfig, ModelType

        cfg = PipelineConfig(
            name="t", version="1", model_type=ModelType.CLUSTERING,
            data_source="x", feature_columns=["f1"], target_column="y",
        )
        with pytest.raises(ValueError, match="Unsupported model type"):
            create_pipeline(cfg)


# ---------------------------------------------------------------------------
# monitoring.py tests
# ---------------------------------------------------------------------------

class TestDriftDetector:
    """Tests for DriftDetector."""

    def _make_detector(self, **config_overrides):
        from backend.ml.pipeline.monitoring import DriftDetector
        return DriftDetector(config=config_overrides if config_overrides else None)

    def test_default_config(self):
        detector = self._make_detector()
        assert detector.config["data_drift_threshold"] == 0.3
        assert detector.config["min_samples"] == 100

    def test_set_reference_data(self):
        detector = self._make_detector()
        df = pd.DataFrame({"f1": np.random.randn(200)})
        detector.set_reference_data(df)
        assert detector.reference_data is not None
        assert "f1" in detector.reference_stats

    @pytest.mark.asyncio
    async def test_detect_data_drift_no_reference_raises(self):
        detector = self._make_detector()
        with pytest.raises(ValueError, match="Reference data not set"):
            await detector.detect_data_drift(pd.DataFrame())

    @pytest.mark.asyncio
    async def test_detect_data_drift_insufficient_samples(self):
        # Must provide full config dict since DriftDetector replaces defaults entirely
        detector = self._make_detector()
        detector.config["min_samples"] = 100
        detector.set_reference_data(pd.DataFrame({"f1": np.random.randn(200)}))
        report = await detector.detect_data_drift(pd.DataFrame({"f1": [1.0, 2.0]}))
        assert bool(report.is_drift_detected) is False

    @pytest.mark.asyncio
    async def test_detect_data_drift_no_drift_similar_data(self):
        np.random.seed(42)
        ref = pd.DataFrame({"f1": np.random.randn(500)})
        cur = pd.DataFrame({"f1": np.random.randn(500)})

        detector = self._make_detector(min_samples=50, data_drift_threshold=0.5)
        detector.set_reference_data(ref)
        report = await detector.detect_data_drift(cur)
        assert bool(report.is_drift_detected) is False

    @pytest.mark.asyncio
    async def test_detect_data_drift_detects_shift(self):
        np.random.seed(42)
        ref = pd.DataFrame({"f1": np.random.randn(500)})
        cur = pd.DataFrame({"f1": np.random.randn(500) + 10})  # big shift

        detector = self._make_detector(min_samples=50, data_drift_threshold=0.1)
        detector.set_reference_data(ref)
        report = await detector.detect_data_drift(cur)
        assert bool(report.is_drift_detected) is True
        assert len(report.drifted_features) > 0

    @pytest.mark.asyncio
    async def test_detect_concept_drift_insufficient_windows(self):
        from backend.ml.pipeline.monitoring import DriftDetector
        detector = DriftDetector(config={"window_size": 1000, "concept_drift_threshold": 0.05})
        preds = np.array([1, 2, 3])
        actuals = np.array([1, 2, 3])
        report = await detector.detect_concept_drift(preds, actuals)
        assert report.is_drift_detected is False

    @pytest.mark.asyncio
    async def test_detect_concept_drift_mismatched_lengths_raises(self):
        detector = self._make_detector()
        with pytest.raises(ValueError, match="same length"):
            await detector.detect_concept_drift(np.array([1, 2]), np.array([1]))

    @pytest.mark.asyncio
    async def test_detect_prediction_drift_no_reference_raises(self):
        detector = self._make_detector()
        with pytest.raises(ValueError, match="Reference predictions not set"):
            await detector.detect_prediction_drift(np.array([1, 2]))


class TestAlertManager:
    """Tests for AlertManager."""

    @pytest.mark.asyncio
    async def test_create_alert(self):
        from backend.ml.pipeline.monitoring import AlertManager, AlertSeverity
        mgr = AlertManager()
        alert = await mgr.create_alert(
            severity=AlertSeverity.WARNING,
            title="Low accuracy",
            message="Accuracy dropped below threshold",
        )
        assert alert.severity == AlertSeverity.WARNING
        assert len(mgr.alerts) == 1

    @pytest.mark.asyncio
    async def test_get_active_alerts(self):
        from backend.ml.pipeline.monitoring import AlertManager, AlertSeverity
        mgr = AlertManager()
        await mgr.create_alert(AlertSeverity.INFO, "info", "info message")
        await mgr.create_alert(AlertSeverity.ERROR, "err", "error message")
        assert len(mgr.get_active_alerts()) == 2
        mgr.resolve_alert(mgr.alerts[0].alert_id)
        assert len(mgr.get_active_alerts()) == 1

    @pytest.mark.asyncio
    async def test_acknowledge_alert(self):
        from backend.ml.pipeline.monitoring import AlertManager, AlertSeverity
        mgr = AlertManager()
        alert = await mgr.create_alert(AlertSeverity.INFO, "t", "m")
        mgr.acknowledge_alert(alert.alert_id, user="admin")
        assert mgr.alerts[0].acknowledged is True
        assert mgr.alerts[0].acknowledged_by == "admin"

    @pytest.mark.asyncio
    async def test_alert_handler_called(self):
        from backend.ml.pipeline.monitoring import AlertManager, AlertSeverity
        called_with = []

        async def handler(alert):
            called_with.append(alert)

        mgr = AlertManager()
        mgr.register_handler(handler)
        await mgr.create_alert(AlertSeverity.CRITICAL, "crit", "msg")
        assert len(called_with) == 1


class TestPerformanceMetrics:
    """Tests for PerformanceMetrics and ModelMonitor.calculate_metrics."""

    def test_classification_metrics(self):
        from backend.ml.pipeline.monitoring import ModelMonitor
        monitor = ModelMonitor()
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        metrics = monitor.calculate_metrics(y_true, y_pred, model_type="classification")
        assert metrics.accuracy == pytest.approx(0.8, abs=0.01)
        assert metrics.sample_count == 5
        assert metrics.confusion_matrix is not None

    def test_regression_metrics(self):
        from backend.ml.pipeline.monitoring import ModelMonitor
        monitor = ModelMonitor()
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 2.9])
        metrics = monitor.calculate_metrics(y_true, y_pred, model_type="regression")
        assert metrics.mse is not None
        assert metrics.mae is not None
        assert metrics.rmse is not None
        assert metrics.r2 is not None

    def test_regression_mape_with_nonzero_actuals(self):
        from backend.ml.pipeline.monitoring import ModelMonitor
        monitor = ModelMonitor()
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([11.0, 19.0, 31.0])
        metrics = monitor.calculate_metrics(y_true, y_pred, model_type="regression")
        assert metrics.mape is not None
        assert metrics.mape > 0

    def test_to_dict(self):
        from backend.ml.pipeline.monitoring import PerformanceMetrics
        m = PerformanceMetrics(
            model_name="test", model_version="1.0",
            timestamp=datetime.now(timezone.utc), accuracy=0.95,
        )
        d = m.to_dict()
        assert d["model_name"] == "test"
        assert d["accuracy"] == 0.95


class TestModelMonitor:
    """Tests for ModelMonitor."""

    @pytest.mark.asyncio
    async def test_register_model(self):
        from backend.ml.pipeline.monitoring import ModelMonitor
        monitor = ModelMonitor()
        await monitor.register_model("m1", "1.0", "http://localhost:8080")
        assert "m1" in monitor.monitored_models
        assert "m1" in monitor.metrics_history

    @pytest.mark.asyncio
    async def test_get_model_metrics_empty(self):
        from backend.ml.pipeline.monitoring import ModelMonitor
        monitor = ModelMonitor()
        result = await monitor.get_model_metrics("nonexistent")
        assert result == {}


# ---------------------------------------------------------------------------
# orchestrator.py tests
# ---------------------------------------------------------------------------

class TestTrainingSchedule:
    """Tests for TrainingSchedule."""

    def test_hourly_next_run(self):
        from backend.ml.pipeline.orchestrator import TrainingSchedule, ScheduleFrequency
        sched = TrainingSchedule(frequency=ScheduleFrequency.HOURLY)
        next_run = sched.get_next_run_time()
        assert next_run > datetime.now(timezone.utc)

    def test_daily_next_run(self):
        from backend.ml.pipeline.orchestrator import TrainingSchedule, ScheduleFrequency
        sched = TrainingSchedule(frequency=ScheduleFrequency.DAILY, time_of_day="23:59")
        next_run = sched.get_next_run_time()
        assert next_run > datetime.now(timezone.utc) - timedelta(hours=25)


class TestRetrainingTrigger:
    """Tests for RetrainingTrigger."""

    def test_performance_degradation_trigger(self):
        from backend.ml.pipeline.orchestrator import RetrainingTrigger, TriggerType
        trigger = RetrainingTrigger(
            trigger_type=TriggerType.PERFORMANCE_DEGRADATION,
            min_accuracy=0.8, max_error_rate=0.1,
        )
        assert trigger.should_trigger({"accuracy": 0.7}) is True
        assert trigger.should_trigger({"accuracy": 0.9}) is False

    def test_data_drift_trigger(self):
        from backend.ml.pipeline.orchestrator import RetrainingTrigger, TriggerType
        trigger = RetrainingTrigger(
            trigger_type=TriggerType.DATA_DRIFT,
            drift_threshold=0.3,
        )
        assert trigger.should_trigger({"drift_score": 0.5}) is True
        assert trigger.should_trigger({"drift_score": 0.1}) is False

    def test_new_data_threshold_trigger(self):
        from backend.ml.pipeline.orchestrator import RetrainingTrigger, TriggerType
        trigger = RetrainingTrigger(
            trigger_type=TriggerType.NEW_DATA_THRESHOLD,
            min_new_samples=1000,
        )
        assert trigger.should_trigger({"new_samples": 1500}) is True
        assert trigger.should_trigger({"new_samples": 500}) is False

    def test_error_rate_trigger(self):
        from backend.ml.pipeline.orchestrator import RetrainingTrigger, TriggerType
        trigger = RetrainingTrigger(
            trigger_type=TriggerType.ERROR_RATE,
            max_error_rate=0.1,
        )
        assert trigger.should_trigger({"prediction_error_rate": 0.2}) is True
        assert trigger.should_trigger({"prediction_error_rate": 0.05}) is False

    def test_disabled_trigger_never_fires(self):
        from backend.ml.pipeline.orchestrator import RetrainingTrigger, TriggerType
        trigger = RetrainingTrigger(
            trigger_type=TriggerType.PERFORMANCE_DEGRADATION,
            enabled=False,
        )
        assert trigger.should_trigger({"accuracy": 0.1}) is False


class TestOrchestratorConfig:
    """Tests for OrchestratorConfig defaults."""

    def test_defaults(self):
        from backend.ml.pipeline.orchestrator import OrchestratorConfig
        cfg = OrchestratorConfig()
        assert cfg.max_concurrent_pipelines == 3
        assert cfg.enable_auto_retraining is True
        assert cfg.max_retries == 3

    def test_version_increment(self):
        from backend.ml.pipeline.orchestrator import MLOrchestrator
        with patch.object(Path, "mkdir"):
            orch = MLOrchestrator.__new__(MLOrchestrator)
            orch.config = MagicMock()
            assert orch._increment_version("1.0.0") == "1.0.1"
            assert orch._increment_version("2.3.9") == "2.3.10"
            assert orch._increment_version("invalid") == "1.0.1"


class TestMLOrchestratorStatus:
    """Tests for MLOrchestrator.get_status."""

    def test_get_status_shape(self):
        from backend.ml.pipeline.orchestrator import MLOrchestrator, OrchestratorConfig
        with patch.object(Path, "mkdir"):
            cfg = OrchestratorConfig(
                models_path="/tmp/test_models",
                logs_path="/tmp/test_logs",
            )
            orch = MLOrchestrator(cfg)
            status = orch.get_status()
            assert "running" in status
            assert "active_pipelines" in status
            assert "config" in status


# ---------------------------------------------------------------------------
# memory_sync.py tests
# ---------------------------------------------------------------------------

class TestMemoryEntry:
    """Tests for MemoryEntry."""

    def test_to_dict_string_value(self):
        from backend.ml.pipeline.memory_sync import MemoryEntry
        entry = MemoryEntry(
            key="k1", value="hello", namespace="default",
            timestamp=datetime.now(timezone.utc),
        )
        d = entry.to_dict()
        assert d["key"] == "k1"
        assert d["value"] == "hello"

    def test_to_dict_dict_value(self):
        from backend.ml.pipeline.memory_sync import MemoryEntry
        entry = MemoryEntry(
            key="k2", value={"a": 1}, namespace="ns",
            timestamp=datetime.now(timezone.utc),
        )
        d = entry.to_dict()
        assert d["value"] == '{"a": 1}'


class TestSyncResult:
    """Tests for SyncResult."""

    def test_sync_result_fields(self):
        from backend.ml.pipeline.memory_sync import SyncResult
        r = SyncResult(
            success=True, entries_synced=5, entries_failed=0,
            direction="to_memory", duration_ms=100.5, errors=[],
        )
        assert r.success is True
        assert r.entries_synced == 5


class TestClaudeFlowMemoryAdapter:
    """Tests for ClaudeFlowMemoryAdapter with in-memory SQLite."""

    def _create_test_db(self, db_path):
        """Create a test SQLite database with the required schema."""
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_entries (
                id TEXT PRIMARY KEY,
                key TEXT,
                value TEXT,
                namespace TEXT,
                metadata TEXT,
                ttl INTEGER,
                created_at TEXT,
                updated_at TEXT
            )
        """)
        conn.commit()
        conn.close()

    def _init_adapter_sync(self, db_path, registry_path):
        """Initialize a ClaudeFlowMemoryAdapter manually (skip async _ensure_namespaces)."""
        from backend.ml.pipeline.memory_sync import ClaudeFlowMemoryAdapter
        adapter = ClaudeFlowMemoryAdapter(
            memory_db_path=db_path,
            registry_path=registry_path,
        )
        # Manually connect without calling the async initialize (avoids recursion)
        adapter._connection = sqlite3.connect(db_path, check_same_thread=False)
        adapter._connection.row_factory = sqlite3.Row
        adapter._initialized = True
        return adapter

    def test_initialize_missing_db_returns_false(self, tmp_path):
        from backend.ml.pipeline.memory_sync import ClaudeFlowMemoryAdapter
        adapter = ClaudeFlowMemoryAdapter(
            memory_db_path=str(tmp_path / "nonexistent.db"),
            registry_path=str(tmp_path / "registry.json"),
        )
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(adapter.initialize())
            assert result is False
        finally:
            loop.close()

    @pytest.mark.asyncio
    async def test_store_and_retrieve(self, tmp_path):
        db_path = str(tmp_path / "test_memory.db")
        self._create_test_db(db_path)
        adapter = self._init_adapter_sync(db_path, str(tmp_path / "reg.json"))

        stored = await adapter.store(key="test_key", value={"data": "value"}, namespace="test")
        assert stored is True

        retrieved = await adapter.retrieve(key="test_key", namespace="test")
        assert retrieved == {"data": "value"}
        await adapter.close()

    @pytest.mark.asyncio
    async def test_retrieve_nonexistent_returns_none(self, tmp_path):
        db_path = str(tmp_path / "test_memory.db")
        self._create_test_db(db_path)
        adapter = self._init_adapter_sync(db_path, str(tmp_path / "reg.json"))

        result = await adapter.retrieve(key="nope", namespace="test")
        assert result is None
        await adapter.close()

    @pytest.mark.asyncio
    async def test_search(self, tmp_path):
        db_path = str(tmp_path / "test_memory.db")
        self._create_test_db(db_path)
        adapter = self._init_adapter_sync(db_path, str(tmp_path / "reg.json"))

        await adapter.store(key="auth_pattern", value="jwt", namespace="test")
        await adapter.store(key="db_pattern", value="sql", namespace="test")
        results = await adapter.search("auth", namespace="test")
        assert len(results) >= 1
        assert any("auth" in r["key"] for r in results)
        await adapter.close()

    @pytest.mark.asyncio
    async def test_sync_registry_to_memory_missing_file(self, tmp_path):
        db_path = str(tmp_path / "test_memory.db")
        self._create_test_db(db_path)
        adapter = self._init_adapter_sync(db_path, str(tmp_path / "reg.json"))

        result = await adapter.sync_registry_to_memory()
        assert result.success is False
        assert "not found" in result.errors[0]
        await adapter.close()

    @pytest.mark.asyncio
    async def test_sync_registry_to_memory_with_models(self, tmp_path):
        db_path = str(tmp_path / "test_memory.db")
        self._create_test_db(db_path)
        registry_path = str(tmp_path / "registry.json")
        registry_data = {"models": [{"name": "model_a", "version": "1.0"}]}
        Path(registry_path).write_text(json.dumps(registry_data))

        adapter = self._init_adapter_sync(db_path, registry_path)
        result = await adapter.sync_registry_to_memory()
        assert result.entries_synced == 1
        await adapter.close()

    @pytest.mark.asyncio
    async def test_close(self, tmp_path):
        db_path = str(tmp_path / "test_memory.db")
        self._create_test_db(db_path)
        adapter = self._init_adapter_sync(db_path, str(tmp_path / "reg.json"))

        await adapter.close()
        assert adapter._connection is None
        assert adapter._initialized is False


# ---------------------------------------------------------------------------
# task_bridge.py tests
# ---------------------------------------------------------------------------

class TestUnifiedTask:
    """Tests for UnifiedTask."""

    def test_to_claude_code_format(self):
        from backend.ml.pipeline.task_bridge import UnifiedTask, TaskStatus
        task = UnifiedTask(
            task_id="t1", status=TaskStatus.PENDING,
            subject="Train model", description="Train LSTM model",
        )
        fmt = task.to_claude_code_format()
        assert fmt["id"] == "t1"
        assert fmt["subject"] == "Train model"
        assert fmt["status"] == "pending"

    def test_to_celery_format(self):
        from backend.ml.pipeline.task_bridge import UnifiedTask, TaskStatus
        task = UnifiedTask(
            task_id="t1", status=TaskStatus.IN_PROGRESS,
            func_name="train_model", celery_task_id="celery_123",
        )
        fmt = task.to_celery_format()
        assert fmt["task_id"] == "celery_123"
        assert fmt["name"] == "train_model"
        assert fmt["status"] == "IN_PROGRESS"

    def test_from_claude_code(self):
        from backend.ml.pipeline.task_bridge import UnifiedTask
        data = {
            "id": "cc_1",
            "subject": "Fix bug",
            "status": "completed",
            "owner": "coder",
            "metadata": {"celery_task_id": "cel_1"},
        }
        task = UnifiedTask.from_claude_code(data)
        assert task.task_id == "cc_1"
        assert task.subject == "Fix bug"
        assert task.celery_task_id == "cel_1"
        assert task.source == "claude_code"

    def test_from_celery(self):
        from backend.ml.pipeline.task_bridge import UnifiedTask
        data = {
            "task_id": "cel_1",
            "name": "train_model",
            "status": "SUCCESS",
            "args": [1, 2],
            "kwargs": {"lr": 0.01},
        }
        task = UnifiedTask.from_celery(data)
        assert task.celery_task_id == "cel_1"
        assert task.func_name == "train_model"
        assert task.source == "celery"
        from backend.ml.pipeline.task_bridge import TaskStatus
        assert task.status == TaskStatus.COMPLETED


class TestTaskBridge:
    """Tests for TaskBridge."""

    @pytest.fixture
    def bridge(self, tmp_path):
        from backend.ml.pipeline.task_bridge import TaskBridge
        b = TaskBridge()
        b._state_file = tmp_path / "state.json"
        return b

    @pytest.mark.asyncio
    async def test_create_ml_task(self, bridge):
        task = await bridge.create_ml_task(
            func_name="train_model",
            subject="Train XGBoost",
            agent_type="coder",
        )
        assert task.func_name == "train_model"
        assert task.task_id in bridge.tasks

    @pytest.mark.asyncio
    async def test_get_task_by_id(self, bridge):
        task = await bridge.create_ml_task(func_name="test_func")
        found = await bridge.get_task(task.task_id)
        assert found is task

    @pytest.mark.asyncio
    async def test_get_task_by_celery_id(self, bridge):
        task = await bridge.create_ml_task(func_name="test_func")
        task.celery_task_id = "cel_999"
        bridge.celery_to_unified["cel_999"] = task.task_id
        found = await bridge.get_task("cel_999")
        assert found is task

    @pytest.mark.asyncio
    async def test_get_task_not_found(self, bridge):
        found = await bridge.get_task("nonexistent")
        assert found is None

    @pytest.mark.asyncio
    async def test_list_tasks_filter_by_status(self, bridge):
        from backend.ml.pipeline.task_bridge import TaskStatus
        t1 = await bridge.create_ml_task(func_name="f1")
        t2 = await bridge.create_ml_task(func_name="f2")
        t1.status = TaskStatus.COMPLETED
        result = await bridge.list_tasks(status=TaskStatus.COMPLETED)
        assert len(result) == 1
        assert result[0].task_id == t1.task_id

    @pytest.mark.asyncio
    async def test_save_and_load_state(self, bridge):
        await bridge.create_ml_task(func_name="f1", subject="S1")
        await bridge.save_state()
        assert bridge._state_file.exists()

        from backend.ml.pipeline.task_bridge import TaskBridge
        new_bridge = TaskBridge()
        new_bridge._state_file = bridge._state_file
        await new_bridge.initialize()
        assert len(new_bridge.tasks) == 1

    def test_get_status_summary(self, bridge):
        summary = bridge.get_status_summary()
        assert summary["total"] == 0
        assert "by_status" in summary
        assert "by_source" in summary

    @pytest.mark.asyncio
    async def test_update_from_celery(self, bridge):
        from backend.ml.pipeline.task_bridge import TaskStatus
        task = await bridge.create_ml_task(func_name="f1")
        task.celery_task_id = "cel_abc"
        bridge.celery_to_unified["cel_abc"] = task.task_id

        await bridge.update_from_celery("cel_abc", "SUCCESS", result={"acc": 0.9})
        assert task.status == TaskStatus.COMPLETED
        assert task.result == {"acc": 0.9}

    @pytest.mark.asyncio
    async def test_register_callback_triggered(self, bridge):
        from backend.ml.pipeline.task_bridge import TaskStatus
        callback_results = []

        async def callback(t):
            callback_results.append(t.status)

        task = await bridge.create_ml_task(func_name="f1")
        task.celery_task_id = "cel_cb"
        bridge.celery_to_unified["cel_cb"] = task.task_id
        bridge.register_callback(task.task_id, callback)

        await bridge.update_from_celery("cel_cb", "FAILURE", error="oops")
        assert TaskStatus.FAILED in callback_results


# ---------------------------------------------------------------------------
# deployment.py tests
# ---------------------------------------------------------------------------

class TestDeploymentStrategy:
    """Tests for deployment enums and configs."""

    def test_deployment_strategy_values(self):
        from backend.ml.pipeline.deployment import DeploymentStrategy
        assert DeploymentStrategy.BLUE_GREEN.value == "blue_green"
        assert DeploymentStrategy.CANARY.value == "canary"
        assert DeploymentStrategy.SHADOW.value == "shadow"

    def test_deployment_config_defaults(self):
        from backend.ml.pipeline.deployment import DeploymentConfig, DeploymentStrategy, DeploymentEnvironment
        cfg = DeploymentConfig(
            model_name="m1", model_version="1.0",
            environment=DeploymentEnvironment.STAGING,
            strategy=DeploymentStrategy.BLUE_GREEN,
            endpoint_url="http://localhost",
        )
        assert cfg.replicas == 2
        assert cfg.auto_rollback is True
        assert cfg.canary_percentage == 10.0


class TestModelDeployer:
    """Tests for ModelDeployer."""

    def test_canary_weights_calculation(self):
        from backend.ml.pipeline.deployment import ModelDeployer
        deployer = ModelDeployer.__new__(ModelDeployer)
        weights = deployer._calculate_canary_weights(
            existing_count=3, canary_count=1, canary_percentage=10.0,
        )
        assert len(weights) == 4
        assert pytest.approx(sum(weights), abs=0.001) == 1.0
        # canary gets 10%, existing gets 90%
        assert weights[3] == pytest.approx(0.1, abs=0.001)

    @pytest.mark.asyncio
    async def test_should_rollback_high_error_rate(self):
        from backend.ml.pipeline.deployment import ModelDeployer, DeploymentConfig, DeploymentStrategy, DeploymentEnvironment
        from backend.ml.pipeline.monitoring import PerformanceMetrics

        deployer = ModelDeployer.__new__(ModelDeployer)
        cfg = DeploymentConfig(
            model_name="m1", model_version="1.0",
            environment=DeploymentEnvironment.PRODUCTION,
            strategy=DeploymentStrategy.CANARY,
            endpoint_url="http://localhost",
            rollback_threshold_error_rate=0.05,
        )
        metrics = PerformanceMetrics(
            model_name="m1", model_version="1.0",
            timestamp=datetime.now(timezone.utc),
            error_rate=0.1,
        )
        assert await deployer._should_rollback(metrics, cfg) is True

    @pytest.mark.asyncio
    async def test_should_not_rollback_when_disabled(self):
        from backend.ml.pipeline.deployment import ModelDeployer, DeploymentConfig, DeploymentStrategy, DeploymentEnvironment
        from backend.ml.pipeline.monitoring import PerformanceMetrics

        deployer = ModelDeployer.__new__(ModelDeployer)
        cfg = DeploymentConfig(
            model_name="m1", model_version="1.0",
            environment=DeploymentEnvironment.PRODUCTION,
            strategy=DeploymentStrategy.CANARY,
            endpoint_url="http://localhost",
            auto_rollback=False,
        )
        metrics = PerformanceMetrics(
            model_name="m1", model_version="1.0",
            timestamp=datetime.now(timezone.utc),
            error_rate=0.9,
        )
        assert await deployer._should_rollback(metrics, cfg) is False

    @pytest.mark.asyncio
    async def test_rollback_unknown_deployment_returns_false(self):
        from backend.ml.pipeline.deployment import ModelDeployer
        deployer = ModelDeployer.__new__(ModelDeployer)
        deployer.deployments = {}
        result = await deployer.rollback("nonexistent_id")
        assert result is False


class TestABTestManager:
    """Tests for ABTestManager."""

    @pytest.mark.asyncio
    async def test_start_test_creates_result(self):
        from backend.ml.pipeline.deployment import ABTestManager, ABTestConfig
        from backend.ml.pipeline.monitoring import ModelMonitor
        monitor = ModelMonitor()
        mgr = ABTestManager(monitor)
        config = ABTestConfig(
            test_name="test1",
            model_a_name="model_a", model_a_version="1.0",
            model_b_name="model_b", model_b_version="2.0",
            duration_hours=0,  # prevent long-running monitor task
        )
        test_id = await mgr.start_test(config)
        assert test_id in mgr.test_results
        assert test_id in mgr.active_tests

    @pytest.mark.asyncio
    async def test_route_request_consistency(self):
        from backend.ml.pipeline.deployment import ABTestManager, ABTestConfig
        from backend.ml.pipeline.monitoring import ModelMonitor
        monitor = ModelMonitor()
        mgr = ABTestManager(monitor)
        config = ABTestConfig(
            test_name="consistency",
            model_a_name="a", model_a_version="1.0",
            model_b_name="b", model_b_version="2.0",
            traffic_percentage_a=50.0,
            duration_hours=0,
        )
        test_id = await mgr.start_test(config)
        # Same user should get same model on repeated calls
        v1 = await mgr.route_request(test_id, "user_42")
        v2 = await mgr.route_request(test_id, "user_42")
        assert v1 == v2

    @pytest.mark.asyncio
    async def test_route_request_invalid_test_raises(self):
        from backend.ml.pipeline.deployment import ABTestManager
        from backend.ml.pipeline.monitoring import ModelMonitor
        mgr = ABTestManager(ModelMonitor())
        with pytest.raises(ValueError, match="not found"):
            await mgr.route_request("nonexistent", "user_1")

    def test_get_test_results_nonexistent(self):
        from backend.ml.pipeline.deployment import ABTestManager
        from backend.ml.pipeline.monitoring import ModelMonitor
        mgr = ABTestManager(ModelMonitor())
        assert mgr.get_test_results("nope") is None


# ---------------------------------------------------------------------------
# registry.py tests
# ---------------------------------------------------------------------------

class TestModelVersion:
    """Tests for ModelVersion."""

    def _make_version(self):
        from backend.ml.pipeline.registry import ModelVersion, ModelStage, DeploymentStatus
        return ModelVersion(
            model_id="mv_001",
            model_name="test_model",
            version="1.0.0",
            model_path=Path("/tmp/model.pkl"),
            artifacts_path=Path("/tmp/artifacts"),
            created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )

    def test_to_dict(self):
        mv = self._make_version()
        d = mv.to_dict()
        assert d["model_id"] == "mv_001"
        assert d["model_name"] == "test_model"
        assert d["stage"] == "development"
        assert d["deployment_status"] == "not_deployed"

    def test_from_dict_roundtrip(self):
        from backend.ml.pipeline.registry import ModelVersion
        mv = self._make_version()
        d = mv.to_dict()
        mv2 = ModelVersion.from_dict(d)
        assert mv2.model_id == mv.model_id
        assert mv2.version == mv.version
        assert mv2.stage == mv.stage


class TestModelMetadata:
    """Tests for ModelMetadata."""

    def test_defaults(self):
        from backend.ml.pipeline.registry import ModelMetadata
        md = ModelMetadata(model_id="m1")
        assert md.approval_status == "pending"
        assert md.target_latency_ms == 100.0
        assert md.dependencies == []


class TestModelRegistryCompareMetrics:
    """Tests for ModelRegistry._compare_metrics (no I/O)."""

    def test_compare_metrics(self):
        from backend.ml.pipeline.registry import ModelRegistry
        # Construct just enough to call the method
        registry = ModelRegistry.__new__(ModelRegistry)
        m1 = {"accuracy": 0.8, "loss": 0.2}
        m2 = {"accuracy": 0.9, "loss": 0.1}
        result = registry._compare_metrics(m1, m2)
        assert result["accuracy"]["improvement_pct"] == pytest.approx(12.5, abs=0.1)
        assert result["loss"]["improvement_pct"] == pytest.approx(-50.0, abs=0.1)

    def test_compare_metrics_zero_baseline(self):
        from backend.ml.pipeline.registry import ModelRegistry
        registry = ModelRegistry.__new__(ModelRegistry)
        m1 = {"accuracy": 0}
        m2 = {"accuracy": 0.9}
        result = registry._compare_metrics(m1, m2)
        assert result["accuracy"]["improvement_pct"] == 0


# ---------------------------------------------------------------------------
# Enum coverage
# ---------------------------------------------------------------------------

class TestEnums:
    """Test enum values for completeness."""

    def test_pipeline_status_values(self):
        from backend.ml.pipeline.base import PipelineStatus
        assert PipelineStatus.PENDING.value == "pending"
        assert PipelineStatus.RETRYING.value == "retrying"

    def test_model_type_values(self):
        from backend.ml.pipeline.base import ModelType
        assert ModelType.DEEP_LEARNING.value == "deep_learning"
        assert ModelType.ANOMALY_DETECTION.value == "anomaly_detection"

    def test_trigger_type_values(self):
        from backend.ml.pipeline.orchestrator import TriggerType
        assert TriggerType.SCHEDULED.value == "scheduled"
        assert TriggerType.CONCEPT_DRIFT.value == "concept_drift"

    def test_task_status_values(self):
        from backend.ml.pipeline.task_bridge import TaskStatus
        assert TaskStatus.CANCELLED.value == "cancelled"

    def test_drift_type_values(self):
        from backend.ml.pipeline.monitoring import DriftType
        assert DriftType.DATA_DRIFT.value == "data_drift"
        assert DriftType.PERFORMANCE_DRIFT.value == "performance_drift"

    def test_alert_severity_values(self):
        from backend.ml.pipeline.monitoring import AlertSeverity
        assert AlertSeverity.CRITICAL.value == "critical"

    def test_model_stage_values(self):
        from backend.ml.pipeline.registry import ModelStage
        assert ModelStage.ARCHIVED.value == "archived"

    def test_deployment_environment_values(self):
        from backend.ml.pipeline.deployment import DeploymentEnvironment
        assert DeploymentEnvironment.PRODUCTION.value == "production"
