"""
Regression tests for the check_data_quality DAG task.

F-06-003 (audit 2026-04, G2a sub-theme D step 12): the task called the
non-existent method ``DataQualityChecker.check_recent_data_quality``,
raising AttributeError at runtime.
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


def test_check_data_quality_does_not_call_removed_method() -> None:
    """F-06-003: ``checker.check_recent_data_quality(...)`` call must be gone."""

    text = _DAG.read_text()
    assert not re.search(
        r"\.check_recent_data_quality\s*\(", text
    ), (
        "ml_training_pipeline_dag.py still calls the non-existent "
        "DataQualityChecker.check_recent_data_quality method"
    )


def test_check_data_quality_uses_existing_methods() -> None:
    """F-06-003: must use real public surface of DataQualityChecker."""

    text = _DAG.read_text()
    assert "validate_price_data" in text, (
        "check_data_quality must call validate_price_data on real rows"
    )
    assert "generate_quality_report" in text, (
        "check_data_quality must call generate_quality_report to aggregate"
    )
