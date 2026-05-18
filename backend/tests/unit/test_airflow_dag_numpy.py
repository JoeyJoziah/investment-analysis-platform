"""
Regression tests for missing numpy import in ml_training_pipeline_dag.

F-06-002 (audit 2026-04, G2a sub-theme D step 11):
``evaluate_models`` referenced ``np.random.randn`` without importing
numpy, raising ``NameError`` at task execution time.
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


def test_numpy_is_imported() -> None:
    """F-06-002: numpy must be imported wherever np.* is referenced."""

    text = _DAG.read_text()
    if "np." not in text:
        return  # nothing to verify
    assert re.search(r"^import numpy(\s+as\s+np)?$", text, re.MULTILINE), (
        "ml_training_pipeline_dag.py uses np.* but does not import numpy"
    )
