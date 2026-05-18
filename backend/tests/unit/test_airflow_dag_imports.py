"""
Regression tests for Airflow 1.x → 2.x import migration.

F-06-001 (audit 2026-04, G2a sub-theme D step 10):
``data_pipelines/airflow/dags/ml_training_pipeline_dag.py`` still imported
from removed Airflow 1.x module paths:
- ``airflow.operators.python_operator``
- ``airflow.operators.bash_operator``
- ``airflow.sensors.external_task_sensor``

``airflow dags list`` surfaces ImportError for any DAG file containing
these. The fail-first tests below scan the DAG source for the legacy
paths.
"""

from __future__ import annotations

from pathlib import Path

import pytest


_DAGS_DIR = (
    Path(__file__).resolve().parents[3]
    / "data_pipelines"
    / "airflow"
    / "dags"
)

_REMOVED_IMPORTS = [
    "airflow.operators.python_operator",
    "airflow.operators.bash_operator",
    "airflow.sensors.external_task_sensor",
]


@pytest.mark.parametrize("removed", _REMOVED_IMPORTS)
def test_no_removed_airflow_1x_imports_in_dags(removed: str) -> None:
    """F-06-001: removed Airflow 1.x import paths must not appear in DAGs."""

    offenders = []
    for dag_path in _DAGS_DIR.glob("*.py"):
        text = dag_path.read_text()
        if removed in text:
            offenders.append(dag_path.name)
    assert not offenders, (
        f"DAG(s) still import the removed Airflow 1.x path {removed!r}: "
        f"{offenders}"
    )
