"""
Regression tests for F-06-008 (evaluate-on-random).

The DAG ``evaluate_models`` task previously called
``monitor.calculate_metrics(y_true=np.random.randn(100),
y_pred=np.random.randn(100), ...)`` — every "evaluation" run was
uncorrelated noise, and the resulting ``ranking_score`` was random
white noise. This means the "best model" picked for deployment was
literally random.

The fix delegates to the real ``backend.ml.training.evaluate_models.
ModelEvaluator`` which loads the persisted ``test_data.parquet``
held-out split.

Tests inspect the DAG source rather than executing the task (which
needs Airflow + Postgres + all the ML deps).
"""

from __future__ import annotations

import re
from pathlib import Path


_DAG = (
    Path(__file__).resolve().parents[3]
    / "data_pipelines"
    / "airflow"
    / "dags"
    / "ml_training_pipeline_dag.py"
)


def test_evaluate_models_does_not_use_random_placeholders() -> None:
    """F-06-008: ``np.random.randn(100)`` placeholder CALLS must be gone.

    The docstring on evaluate_models describes the legacy behavior and
    legitimately mentions the old kwargs — anchor on the call-site form
    to avoid flagging the rationale comment (per
    ``feedback_test_anchor_logic`` memory note).
    """

    text = _DAG.read_text()
    # Strip docstrings so we only inspect executable code.
    code = re.sub(r'""".*?"""', '', text, flags=re.DOTALL)
    assert "y_true=np.random.randn(100)" not in code, (
        "DAG evaluate_models still passes y_true=np.random.randn(100) "
        "in executable code"
    )
    assert "y_pred=np.random.randn(100)" not in code, (
        "DAG evaluate_models still passes y_pred=np.random.randn(100) "
        "in executable code"
    )


def test_evaluate_models_uses_real_model_evaluator() -> None:
    """F-06-008: DAG must delegate to the real ModelEvaluator."""

    text = _DAG.read_text()
    assert "from backend.ml.training.evaluate_models import ModelEvaluator" in text, (
        "DAG evaluate_models must import the real ModelEvaluator"
    )
    assert "evaluator.run_evaluation()" in text, (
        "DAG must call evaluator.run_evaluation() to compute real metrics"
    )


def test_evaluate_models_asserts_test_parquet_exists() -> None:
    """F-06-008: missing test_data.parquet must raise, not silently noop."""

    text = _DAG.read_text()
    assert re.search(
        r'test_path\s*=\s*Path\([^)]*\)\s*/\s*["\']test_data\.parquet["\']',
        text,
    ), "DAG must derive the held-out test parquet path explicitly"
    assert "raise AirflowException" in text, (
        "DAG must raise AirflowException when test_data.parquet is missing "
        "rather than fall back to random-data metrics"
    )


def test_ranking_score_uses_r2_not_random() -> None:
    """F-06-008: ranking_score must come from the real test_metrics, not noise."""

    text = _DAG.read_text()
    # ``ranking_score`` should now be derived from the model's r2 on
    # held-out data, not a randomly-generated MonitorMetrics.r2.
    assert "float(model_result.get('r2', 0.0))" in text, (
        "ranking_score must be derived from real test-set R²"
    )
