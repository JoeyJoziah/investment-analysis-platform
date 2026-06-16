"""
Unit tests for real held-out validation metrics (#208 item 1 / Finding #200).

`MLTrainingPipeline.train_models` gates production promotion on real metrics
returned by ``pipeline.get_metrics()`` / ``orchestrator.get_pipeline_metrics()``.
These were previously unimplemented (TODO), so this verifies they now surface
the genuine held-out metrics produced by the ``model_evaluation`` step — RMSE,
MAE, R2 and directional accuracy — and return ``None`` (fail-loud) when no real
metrics exist.

Loads backend/ml/pipeline/base.py source-level (the pipeline package proper
imports mlflow, which is not available in the hermetic test env).

Run (source-level, no conftest):
    ENVIRONMENT=test ... python3 -m pytest \
        backend/tests/unit/test_pipeline_real_metrics.py --noconftest -q
"""

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

_PIPELINE_DIR = Path(__file__).resolve().parents[2] / "ml" / "pipeline"


def _load(mod_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(
        mod_name, _PIPELINE_DIR / filename
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


_base = _load("pl_base_metrics", "base.py")

ModelPipeline = _base.ModelPipeline
PipelineConfig = _base.PipelineConfig
PipelineResult = _base.PipelineResult
PipelineStatus = _base.PipelineStatus
ModelType = _base.ModelType


class _BarePipeline(ModelPipeline):
    """Minimal concrete pipeline (no steps) for testing get_metrics()."""

    def _setup_pipeline(self):
        # No steps — we inject result directly.
        pass


def _make_pipeline():
    cfg = PipelineConfig(
        name="unit_test_pipeline",
        version="1.0.0",
        model_type=ModelType.TIME_SERIES,
        data_source="dataframe",
        feature_columns=["open", "high", "low", "close", "volume"],
        target_column="future_return",
    )
    return _BarePipeline(cfg)


def _completed_result(eval_metrics):
    res = PipelineResult(
        pipeline_id="unit_test_pipeline_x",
        status=PipelineStatus.COMPLETED,
        start_time=datetime.now(timezone.utc),
        end_time=datetime.now(timezone.utc),
    )
    if eval_metrics is not None:
        res.intermediate_results["model_evaluation"] = eval_metrics
    return res


# ---------------------------------------------------------------------------
# get_metrics() surfaces REAL held-out metrics
# ---------------------------------------------------------------------------

def test_get_metrics_returns_real_held_out_metrics():
    pipe = _make_pipeline()
    pipe.result = _completed_result(
        {
            "xgboost": {"rmse": 0.12, "mae": 0.09, "r2": 0.55, "directional_accuracy": 0.61},
            "random_forest": {"rmse": 0.20, "mae": 0.15, "r2": 0.40, "directional_accuracy": 0.52},
        }
    )
    metrics = pipe.get_metrics()
    assert metrics is not None
    # Best model is the one with the lowest held-out RMSE (xgboost).
    assert metrics["best_model"] == "xgboost"
    assert metrics["rmse"] == 0.12
    assert metrics["mae"] == 0.09
    assert metrics["directional_accuracy"] == 0.61
    # Per-model breakdown is preserved.
    assert set(metrics["per_model"]) == {"xgboost", "random_forest"}


def test_get_metrics_values_are_not_constant_placeholders():
    """Metrics must reflect the injected eval output, not hardcoded constants."""
    pipe = _make_pipeline()
    pipe.result = _completed_result(
        {"m": {"rmse": 1.234, "mae": 5.678, "r2": -0.42, "directional_accuracy": 0.337}}
    )
    metrics = pipe.get_metrics()
    assert metrics["rmse"] == 1.234
    assert metrics["mae"] == 5.678
    assert metrics["r2"] == -0.42
    assert metrics["directional_accuracy"] == 0.337


# ---------------------------------------------------------------------------
# Fail-loud: no real metrics -> None
# ---------------------------------------------------------------------------

def test_get_metrics_none_before_run():
    pipe = _make_pipeline()
    assert pipe.result is None
    assert pipe.get_metrics() is None


def test_get_metrics_none_when_pipeline_failed():
    pipe = _make_pipeline()
    res = _completed_result({"m": {"rmse": 0.1, "mae": 0.1, "r2": 0.5, "directional_accuracy": 0.6}})
    res.status = PipelineStatus.FAILED
    pipe.result = res
    assert pipe.get_metrics() is None


def test_get_metrics_none_when_no_evaluation_step_output():
    pipe = _make_pipeline()
    pipe.result = _completed_result(None)  # completed but no model_evaluation
    assert pipe.get_metrics() is None
