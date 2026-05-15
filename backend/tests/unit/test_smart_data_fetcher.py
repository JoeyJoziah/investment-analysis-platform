"""
Regression tests for SmartDataFetcher real-source delegation.

F-05-004 (audit 2026-04, G2a sub-theme C step 25): the previous
implementation returned hardcoded zeros / empty lists from every
``_fetch_*`` method with ``source: "mock"``. The new implementation
delegates to the real clients and returns ``source: "unavailable"``
only when every client failed or is not configured.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest


_PATH = (
    Path(__file__).resolve().parents[2]
    / "data_ingestion"
    / "smart_data_fetcher.py"
)


def _load_module(monkeypatch: pytest.MonkeyPatch):
    """Load smart_data_fetcher in isolation with logging stubbed."""

    spec = importlib.util.spec_from_file_location(
        "smart_data_fetcher_under_test", _PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_price_fetcher_delegates_to_finnhub(monkeypatch: pytest.MonkeyPatch) -> None:
    """F-05-004: finnhub get_quote result must populate the price payload."""

    mod = _load_module(monkeypatch)
    fetcher = mod.SmartDataFetcher()

    finnhub_stub = AsyncMock()
    finnhub_stub.get_quote = AsyncMock(return_value={"c": 195.5, "d": 1.5, "dp": 0.77, "v": 12345})
    monkeypatch.setattr(fetcher, "_get_client", lambda n: finnhub_stub if n == "finnhub" else None)

    result = asyncio.run(fetcher._fetch_price_data("AAPL"))
    assert result["ticker"] == "AAPL"
    assert result["price"] == 195.5
    assert result["source"] == "finnhub"


def test_price_fetcher_falls_through_to_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """F-05-004: no working client → source == 'unavailable' (not 'mock')."""

    mod = _load_module(monkeypatch)
    fetcher = mod.SmartDataFetcher()
    monkeypatch.setattr(fetcher, "_get_client", lambda n: None)

    result = asyncio.run(fetcher._fetch_price_data("AAPL"))
    assert result["source"] == "unavailable"
    assert result["price"] == 0.0  # sentinel value preserved


def test_fundamentals_uses_finnhub_metric_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    """F-05-004: finnhub.get_basic_financials shape is consumed correctly."""

    mod = _load_module(monkeypatch)
    fetcher = mod.SmartDataFetcher()

    finnhub_stub = AsyncMock()
    finnhub_stub.get_basic_financials = AsyncMock(return_value={
        "metric": {
            "peNormalizedAnnual": 28.4,
            "marketCapitalization": 3_000_000,
            "epsAnnual": 6.84,
            "dividendYieldIndicatedAnnual": 0.005,
        }
    })
    monkeypatch.setattr(fetcher, "_get_client", lambda n: finnhub_stub if n == "finnhub" else None)

    result = asyncio.run(fetcher._fetch_fundamentals("AAPL"))
    assert result["source"] == "finnhub"
    assert result["pe_ratio"] == 28.4
    assert result["market_cap"] == 3_000_000
    assert result["eps"] == 6.84


def test_no_mock_source_label_in_implementation() -> None:
    """F-05-004: ``"source": "mock"`` literal must be gone."""

    text = _PATH.read_text()
    assert '"source": "mock"' not in text, (
        "smart_data_fetcher.py must no longer ship hardcoded mock source labels"
    )


def test_client_failure_falls_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """F-05-004: a raising client must not crash the fetcher."""

    mod = _load_module(monkeypatch)
    fetcher = mod.SmartDataFetcher()

    failing = AsyncMock()
    failing.get_quote = AsyncMock(side_effect=RuntimeError("api down"))
    monkeypatch.setattr(fetcher, "_get_client", lambda n: failing if n == "finnhub" else None)

    # Should fall through to unavailable, not raise.
    result = asyncio.run(fetcher._fetch_price_data("AAPL"))
    assert result["source"] == "unavailable"
