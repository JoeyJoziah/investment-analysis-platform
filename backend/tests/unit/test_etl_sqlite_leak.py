"""
Regression tests for sqlite FD leak.

F-05-008 (audit 2026-04, G2a sub-theme C step 23):
``backend/etl/distributed_batch_processor.py``, ``cache_storage.py``,
and ``multi_source_extractor.py`` opened 18 raw ``sqlite3.connect(...)``
handles. The happy paths called ``conn.close()`` but any exception
between the connect and the close (e.g. SQL error mid-transaction)
leaked the file descriptor — FD count grew unbounded under load.

The fix wraps every connection in ``contextlib.closing`` so the
descriptor is released on every code path, including exceptions.
"""

from __future__ import annotations

import re
from pathlib import Path


_TARGETS = [
    Path(__file__).resolve().parents[2] / "etl" / "distributed_batch_processor.py",
    Path(__file__).resolve().parents[2] / "etl" / "cache_storage.py",
    Path(__file__).resolve().parents[2] / "etl" / "multi_source_extractor.py",
]


def test_no_unwrapped_sqlite_connect_assignment() -> None:
    """F-05-008: ``conn = sqlite3.connect(...)`` (unwrapped) must be gone."""

    offenders: list[str] = []
    for p in _TARGETS:
        text = p.read_text()
        for m in re.finditer(r"^\s*conn\s*=\s*sqlite3\.connect\(", text, re.MULTILINE):
            offenders.append(f"{p.name}:{text.count(chr(10), 0, m.start()) + 1}")
    assert not offenders, (
        f"unwrapped sqlite3.connect assignments remain — FD leak risk: {offenders}"
    )


def test_closing_helper_imported() -> None:
    """F-05-008: each target module imports ``closing`` from contextlib."""

    for p in _TARGETS:
        text = p.read_text()
        assert re.search(r"from contextlib import .*\bclosing\b", text), (
            f"{p.name} must import ``closing`` from contextlib"
        )


def test_with_closing_sqlite3_pattern_present() -> None:
    """F-05-008: each target uses ``with closing(sqlite3.connect(...))``."""

    for p in _TARGETS:
        text = p.read_text()
        count = len(re.findall(r"with\s+closing\(\s*sqlite3\.connect\(", text))
        assert count > 0, (
            f"{p.name} must use ``with closing(sqlite3.connect(...))``"
        )


def test_total_sqlite_sites_match_expected() -> None:
    """F-05-008: 18 total sqlite3.connect callsites all wrapped."""

    total_wrapped = 0
    total_calls = 0
    for p in _TARGETS:
        text = p.read_text()
        total_wrapped += len(re.findall(r"with\s+closing\(\s*sqlite3\.connect\(", text))
        total_calls += len(re.findall(r"sqlite3\.connect\(", text))
    assert total_wrapped == total_calls == 18, (
        f"expected 18/18 wrapped, got {total_wrapped}/{total_calls}"
    )
