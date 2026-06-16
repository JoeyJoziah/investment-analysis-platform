"""
Unit tests for promotion gating on real validation metrics (#208 item 1 / #200).

`MLTrainingPipeline.deploy_model` must BLOCK production promotion when the
model's real held-out validation score does not clear ``performance_threshold``,
and `evaluate_models` must score models off real metrics (directional accuracy)
rather than fabricated constants.

`backend.ml.training_pipeline` imports the heavy pipeline package (mlflow, etc.)
at module top, which is not available in the hermetic test env, so we stub those
submodules in ``sys.modules`` and load the module source-level.

Run (source-level, no conftest):
    ENVIRONMENT=test ... python3 -m pytest \
        backend/tests/unit/test_training_pipeline_promotion_gate.py --noconftest -q
"""

import asyncio
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

_TP_PATH = Path(__file__).resolve().parents[2] / "ml" / "training_pipeline.py"


def _install_stubs():
    """Stub the heavy ML pipeline submodules training_pipeline imports."""

    def _mod(name, **attrs):
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m
        return m

    _mod("dotenv", load_dotenv=lambda *a, **k: None)

    # backend.ml.pipeline package + submodules with the exact symbols imported.
    pkg = _mod(
        "backend.ml.pipeline",
        MLOrchestrator=MagicMock,
        OrchestratorConfig=MagicMock,
        TrainingSchedule=MagicMock,
        ScheduleFrequency=MagicMock,
    )
    pkg.__path__ = []  # mark as package
    _mod(
        "backend.ml.pipeline.implementations",
        create_pipeline=MagicMock(),
        PipelineConfig=MagicMock,
        ModelType=types.SimpleNamespace(
            CLASSIFICATION="c", TIME_SERIES="ts", ENSEMBLE="e"
        ),
    )
    _mod("backend.ml.pipeline.registry", ModelRegistry=MagicMock)
    _mod("backend.ml.pipeline.monitoring", ModelMonitor=MagicMock)
    _mod(
        "backend.ml.pipeline.deployment",
        ModelDeployer=MagicMock,
        DeploymentConfig=MagicMock,
        DeploymentStrategy=types.SimpleNamespace(CANARY="canary"),
        DeploymentEnvironment=types.SimpleNamespace(PRODUCTION="production"),
    )


def _load_training_pipeline():
    _install_stubs()
    spec = importlib.util.spec_from_file_location("tp_under_test", _TP_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["tp_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


_tp = _load_training_pipeline()
MLTrainingPipeline = _tp.MLTrainingPipeline


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _pipeline_with_threshold(threshold):
    pipe = MLTrainingPipeline.__new__(MLTrainingPipeline)
    pipe.config = {"performance_threshold": threshold}
    return pipe


# ---------------------------------------------------------------------------
# _promotion_score derives from REAL metrics, not constants
# ---------------------------------------------------------------------------

def test_promotion_score_uses_directional_accuracy():
    assert MLTrainingPipeline._promotion_score(
        {"directional_accuracy": 0.73}
    ) == 0.73


def test_promotion_score_zero_when_metrics_absent():
    # No real metrics -> 0.0 so the model can never clear the gate (fail-loud).
    assert MLTrainingPipeline._promotion_score({}) == 0.0
    assert MLTrainingPipeline._promotion_score(None) == 0.0
    assert MLTrainingPipeline._promotion_score({"rmse": 0.1}) == 0.0


# ---------------------------------------------------------------------------
# deploy_model BLOCKS promotion below threshold
# ---------------------------------------------------------------------------

def test_deploy_blocked_below_threshold():
    pipe = _pipeline_with_threshold(0.75)
    pipe.deployer = MagicMock()
    pipe.deployer.deploy = AsyncMock()  # must NOT be called

    result = _run(pipe.deploy_model("xgboost_classifier", {"directional_accuracy": 0.60}))

    assert result["status"] == "blocked"
    assert result["reason"] == "performance_threshold_not_met"
    assert result["validation_score"] == 0.60
    assert result["performance_threshold"] == 0.75
    pipe.deployer.deploy.assert_not_called()


def test_deploy_blocked_when_no_real_metrics():
    pipe = _pipeline_with_threshold(0.75)
    pipe.deployer = MagicMock()
    pipe.deployer.deploy = AsyncMock()

    result = _run(pipe.deploy_model("xgboost_classifier", None))

    assert result["status"] == "blocked"
    pipe.deployer.deploy.assert_not_called()


def test_deploy_proceeds_when_threshold_met():
    pipe = _pipeline_with_threshold(0.55)
    deployment = MagicMock()
    deployment.deployment_id = "dep-1"
    deployment.status = "deployed"
    deployment.endpoints = ["http://x"]
    deployment.metrics_endpoint = "http://x/metrics"
    pipe.deployer = MagicMock()
    pipe.deployer.deploy = AsyncMock(return_value=deployment)

    result = _run(pipe.deploy_model("xgboost_classifier", {"directional_accuracy": 0.62}))

    assert result["status"] != "blocked"
    assert result["deployment_id"] == "dep-1"
    pipe.deployer.deploy.assert_called_once()


# ---------------------------------------------------------------------------
# evaluate_models selects best by REAL metrics
# ---------------------------------------------------------------------------

def test_evaluate_models_picks_best_by_real_metrics():
    pipe = _pipeline_with_threshold(0.75)
    results = {
        "model_a": {"status": "completed", "metrics": {"directional_accuracy": 0.58}},
        "model_b": {"status": "completed", "metrics": {"directional_accuracy": 0.66}},
        "model_c": {"status": "failed", "error": "boom"},
    }
    best = _run(pipe.evaluate_models(results))
    assert best == "model_b"


def test_evaluate_models_none_when_no_completed_metrics():
    pipe = _pipeline_with_threshold(0.75)
    results = {"model_c": {"status": "failed", "error": "boom"}}
    assert _run(pipe.evaluate_models(results)) is None
