"""
Regression tests for TradingAgents finnhub_utils.

F-04-002 (audit 2026-04, G2a sub-theme B step 2): get_data_in_range used a
bare ``open()`` and never closed the file handle. On a long-running process
this leaks one file descriptor per call. The fail-first test below uses
``mock_open`` to assert the file handle's ``__exit__`` is invoked, which
is only true once the bare ``open()`` is wrapped in a ``with`` block.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest


_FINNHUB_UTILS_PATH = (
    Path(__file__).resolve().parents[2]
    / "TradingAgents"
    / "tradingagents"
    / "dataflows"
    / "finnhub_utils.py"
)
_spec = importlib.util.spec_from_file_location(
    "tradingagents_finnhub_utils_under_test", _FINNHUB_UTILS_PATH
)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_module)
get_data_in_range = _module.get_data_in_range


def test_get_data_in_range_closes_file_handle() -> None:
    """File handle must be closed via context manager (F-04-003)."""

    payload = '{"2024-01-15": [{"headline": "x"}], "2024-02-01": []}'
    m = mock_open(read_data=payload)

    with patch.object(_module, "open", m, create=True):
        result = get_data_in_range(
            ticker="AAPL",
            start_date="2024-01-01",
            end_date="2024-01-31",
            data_type="news_data",
            data_dir="/tmp/fake",
        )

    handle = m.return_value
    handle.__exit__.assert_called()
    assert "2024-01-15" in result
    assert "2024-02-01" not in result


def test_get_data_in_range_period_branch_uses_context_manager() -> None:
    """Same closure guarantee on the ``period``-suffixed path branch."""

    payload = '{"2024-03-31": [{"metric": 1}]}'
    m = mock_open(read_data=payload)

    with patch.object(_module, "open", m, create=True):
        get_data_in_range(
            ticker="AAPL",
            start_date="2024-01-01",
            end_date="2024-12-31",
            data_type="fin_as_reported",
            data_dir="/tmp/fake",
            period="quarterly",
        )

    m.return_value.__exit__.assert_called()
