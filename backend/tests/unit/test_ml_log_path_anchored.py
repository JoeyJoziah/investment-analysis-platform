"""
Regression tests for the __file__-anchored ML log path.

F-03-007 (audit 2026-04, G2a sub-theme A step 29):
``backend/ml/training_pipeline.py`` and ``simple_training_pipeline.py``
configured logging.FileHandler with the relative path
``'backend/ml_logs/...'``. Run from any cwd other than the repo root,
this created a stray ``backend/ml_logs/`` directory next to the
caller and orphaned the log files.

The fix anchors the log directory to ``Path(__file__).parent.parent.parent /
"backend" / "ml_logs"`` so the location is invariant under cwd.
"""

from __future__ import annotations

import re
from pathlib import Path


_TARGETS = [
    Path(__file__).resolve().parents[2] / "ml" / "training_pipeline.py",
    Path(__file__).resolve().parents[2] / "ml" / "simple_training_pipeline.py",
    Path(__file__).resolve().parents[2] / "ml" / "minimal_training.py",
]


def test_no_unanchored_backend_ml_logs_string() -> None:
    """F-03-007: bare ``'backend/ml_logs/...'`` literal must not appear in FileHandler."""

    for p in _TARGETS:
        text = p.read_text()
        # The legacy form was
        # ``logging.FileHandler('backend/ml_logs/training_...')``.
        assert not re.search(
            r"FileHandler\(\s*['\"]backend/ml_logs/", text
        ), (
            f"{p.name} still uses unanchored 'backend/ml_logs/' string in "
            f"FileHandler — log path is cwd-dependent"
        )


def test_file_anchored_log_dir_present() -> None:
    """F-03-007: each target derives the log dir from ``__file__``."""

    for p in _TARGETS:
        text = p.read_text()
        assert re.search(
            r"Path\(__file__\)\.resolve\(\)\.parent\.parent\.parent",
            text,
        ), (
            f"{p.name} must derive the ml_logs directory from "
            f"Path(__file__).resolve().parent.parent.parent"
        )


def test_log_dir_creation_is_idempotent() -> None:
    """F-03-007: target must mkdir(parents=True, exist_ok=True)."""

    for p in _TARGETS:
        text = p.read_text()
        # Either an explicit mkdir() on the derived path, or the target
        # is the unit test stub which doesn't need it. Both training
        # pipelines must explicitly create the directory.
        if p.name in ("training_pipeline.py", "simple_training_pipeline.py", "minimal_training.py"):
            assert "mkdir(parents=True, exist_ok=True)" in text, (
                f"{p.name} must mkdir the log dir on import"
            )
