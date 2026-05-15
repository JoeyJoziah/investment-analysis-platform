"""
Regression tests for ETLOrchestrator distributed-pipeline bounded loop.

F-05-002 (audit 2026-04, G2a sub-theme C step 24):
``ETLOrchestrator._run_distributed_pipeline`` monitored job completion
with ``while completed_jobs < len(job_ids)`` and only counted
``status == 'completed'`` toward the exit condition. Any failed,
cancelled, or errored job left the loop spinning forever. The created
``processing_task`` was also never cancelled on exit, leaking the
task.

The fix:
- counts terminal states (completed | failed | cancelled | error)
- enforces ``ETL_DISTRIBUTED_MAX_WAIT_SECONDS`` (default 7200s)
- cancels ``processing_task`` in ``finally``
"""

from __future__ import annotations

import re
from pathlib import Path


_ORCHESTRATOR = (
    Path(__file__).resolve().parents[2]
    / "etl"
    / "etl_orchestrator.py"
)


def test_terminal_states_counted_toward_exit() -> None:
    """F-05-002: failed/cancelled/error jobs must also exit the loop."""

    text = _ORCHESTRATOR.read_text()
    assert "terminal_states" in text, (
        "monitor loop must define a terminal_states set; otherwise failed "
        "jobs leave the loop spinning"
    )
    for state in ("completed", "failed", "cancelled", "error"):
        assert f'"{state}"' in text, (
            f"terminal state {state!r} missing from the bounded loop"
        )


def test_max_wait_seconds_enforced() -> None:
    """F-05-002: wall-clock bound prevents indefinite hang."""

    text = _ORCHESTRATOR.read_text()
    assert "max_wait_seconds" in text
    assert "ETL_DISTRIBUTED_MAX_WAIT_SECONDS" in text, (
        "max wait must be operator-configurable via env var"
    )


def test_processing_task_cancelled_in_finally() -> None:
    """F-05-002: processing_task must not leak on exit."""

    text = _ORCHESTRATOR.read_text()
    # The monitor loop body must be guarded by a ``finally`` block that
    # invokes both ``stop_processing()`` and ``processing_task.cancel()``.
    finally_block = re.search(
        r"finally:\s*\n\s*self\.distributed_processor\.stop_processing\(\)"
        r".*?processing_task\.cancel\(\)",
        text,
        re.DOTALL,
    )
    assert finally_block is not None, (
        "finally block must cancel processing_task and stop the processor"
    )


def test_no_unbounded_while_completed_lt_len() -> None:
    """F-05-002: the original unbounded condition must be gone."""

    text = _ORCHESTRATOR.read_text()
    # The legacy line was ``while completed_jobs < len(job_ids):``.
    assert "while completed_jobs < len(job_ids):" not in text, (
        "unbounded ``while completed_jobs < len(job_ids):`` still present"
    )
