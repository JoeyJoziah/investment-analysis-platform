"""
Regression tests for VACUUM-in-transaction bug.

F-06-004 (audit 2026-04, G2a sub-theme D step 15):
``enhanced_stock_pipeline.cleanup_and_optimize`` issued
``VACUUM ANALYZE ...`` through ``PostgresHook.run``, which wraps each
statement in a transaction — Postgres rejects this with
``ERROR: VACUUM cannot run inside a transaction block``. VACUUMs must
go through a raw connection with isolation level AUTOCOMMIT.
"""

from __future__ import annotations

import re
from pathlib import Path


_DAG = (
    Path(__file__).resolve().parents[3]
    / "data_pipelines"
    / "airflow"
    / "dags"
    / "enhanced_stock_pipeline.py"
)


def test_vacuum_not_run_through_pg_hook_run() -> None:
    """F-06-004: VACUUM must not be issued via the transactional run() path."""

    text = _DAG.read_text()
    # The VACUUM strings live in a dedicated ``vacuum_queries`` list now.
    assert "vacuum_queries" in text, (
        "VACUUM statements should be isolated into a vacuum_queries list "
        "so they bypass the transactional path"
    )
    # And the raw-connection autocommit hop must be present.
    assert "set_isolation_level(0)" in text, (
        "VACUUM block must use ``conn.set_isolation_level(0)`` "
        "(AUTOCOMMIT) to satisfy Postgres"
    )


def test_vacuum_strings_not_inside_transactional_block() -> None:
    """F-06-004: VACUUM statements must not be inside ``transactional_queries``."""

    text = _DAG.read_text()
    # Slice the file between transactional_queries = [ ... ] and verify no
    # VACUUM string appears in that slice.
    match = re.search(
        r"transactional_queries\s*=\s*\[(.*?)\]", text, re.DOTALL
    )
    assert match is not None, "transactional_queries list not found"
    assert "VACUUM" not in match.group(1).upper(), (
        "VACUUM statement leaked back into the transactional list"
    )
