"""
Regression tests for Airflow DAG cleanup (F-06-005, F-06-006, F-06-007, F-06-009).
"""

from __future__ import annotations

import re
from pathlib import Path


_DAGS_DIR = (
    Path(__file__).resolve().parents[3]
    / "data_pipelines"
    / "airflow"
    / "dags"
)
_DAILY = _DAGS_DIR / "daily_stock_pipeline.py"
_ML = _DAGS_DIR / "ml_training_pipeline_dag.py"


def test_no_schedule_interval_kwarg() -> None:
    """F-06-005: ``schedule_interval=`` deprecated in Airflow 2.4+."""

    for dag_path in _DAGS_DIR.glob("*.py"):
        text = dag_path.read_text()
        assert not re.search(r"\bschedule_interval\s*=", text), (
            f"{dag_path.name} still uses deprecated schedule_interval kwarg"
        )


def test_no_create_session_call_in_dag() -> None:
    """F-06-006: ``airflow.utils.db.create_session`` removed from public API."""

    text = _DAILY.read_text()
    assert not re.search(r"\bcreate_session\s*\(", text), (
        "daily_stock_pipeline.py still calls airflow.utils.db.create_session"
    )
    assert "from airflow.utils.db import create_session" not in text


def test_no_legacy_deprecated_block_in_daily_pipeline() -> None:
    """F-06-007: ``DEPRECATED``/``LEGACY`` comment block must be gone."""

    text = _DAILY.read_text()
    assert "DEPRECATED" not in text, (
        "daily_stock_pipeline.py still contains DEPRECATED label"
    )
    assert "LEGACY FUNCTIONS" not in text, (
        "daily_stock_pipeline.py still contains LEGACY FUNCTIONS block"
    )


def test_no_placeholder_alert_email() -> None:
    """F-06-009: ``ml-alerts@company.com`` placeholder must be gone."""

    text = _ML.read_text()
    assert "ml-alerts@company.com" not in text, (
        "ml_training_pipeline_dag.py still ships the placeholder alert email"
    )
    assert "ml_alert_emails" in text, (
        "ml_training_pipeline_dag.py must source alert recipients from "
        "the ``ml_alert_emails`` Airflow Variable"
    )
