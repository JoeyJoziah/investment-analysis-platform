"""
Unit tests for backend/ml/data_providers.py (#208 item 1).

Uses the importlib file-loading bypass (matching test_ml_extended_agent1.py)
so the data provider can be exercised source-level without pulling SQLAlchemy
or the rest of the backend package graph into the test process.

Run (source-level, no conftest):
    ENVIRONMENT=test ... python3 -m pytest \
        backend/tests/unit/test_backtest_data_provider.py --noconftest -q
"""

import importlib.util
import sys
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Load backend/ml/data_providers.py directly by file path. This avoids
# importing backend.ml (and its SQLAlchemy-heavy siblings) as a package.
# ---------------------------------------------------------------------------
_ML_DIR = Path(__file__).resolve().parents[2] / "ml"


def _load(mod_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(mod_name, _ML_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


_dp_mod = _load("data_providers_mod", "data_providers.py")

PriceHistoryDataProvider = _dp_mod.PriceHistoryDataProvider
NoMarketDataError = _dp_mod.NoMarketDataError
price_history_to_dataframe = _dp_mod.price_history_to_dataframe
OHLCV_COLUMNS = _dp_mod.OHLCV_COLUMNS


def _fake_price_row(d: date, o, h, l, c, v):
    """Mimic a PriceHistory ORM row (Decimal-ish values accepted as floats)."""
    return SimpleNamespace(date=d, open=o, high=h, low=l, close=c, volume=v)


class _FakeRepo:
    """Stand-in for PriceHistoryRepository.get_bulk_price_history."""

    def __init__(self, data):
        # data: Dict[symbol_upper, List[row]]
        self._data = data
        self.get_bulk_price_history = AsyncMock(side_effect=self._bulk)

    async def _bulk(self, symbols, start_date=None, end_date=None, limit_per_symbol=5000):
        return {s.upper(): self._data.get(s.upper(), []) for s in symbols}


# ---------------------------------------------------------------------------
# DataFrame-shape test: correct columns/index from a fake List[PriceHistory].
# ---------------------------------------------------------------------------
def test_dataframe_shape_from_price_history():
    rows = [
        _fake_price_row(date(2024, 1, 3), 11, 13, 10, 12, 1000),
        _fake_price_row(date(2024, 1, 2), 10, 12, 9, 11, 900),
        _fake_price_row(date(2024, 1, 4), 12, 14, 11, 13, 1100),
    ]
    repo = _FakeRepo({"AAPL": rows})
    provider = PriceHistoryDataProvider(repository=repo)

    frame = provider.get_historical_prices("AAPL", date(2024, 1, 1), date(2024, 1, 31))

    # Columns are exactly OHLCV, in order.
    assert list(frame.columns) == OHLCV_COLUMNS == ["open", "high", "low", "close", "volume"]
    # Index is a DatetimeIndex, sorted ascending regardless of input order.
    assert isinstance(frame.index, pd.DatetimeIndex)
    assert list(frame.index) == sorted(frame.index)
    assert len(frame) == 3
    # Values coerced to float and mapped correctly (chronological first row).
    assert frame.iloc[0]["close"] == 11.0
    assert frame.iloc[-1]["high"] == 14.0
    assert frame["volume"].dtype == float

    # repository was queried with date objects (datetime coerced to date).
    _, kwargs = repo.get_bulk_price_history.call_args
    assert kwargs["start_date"] == date(2024, 1, 1)
    assert kwargs["end_date"] == date(2024, 1, 31)


def test_datetime_bounds_are_coerced_to_date():
    repo = _FakeRepo({"MSFT": [_fake_price_row(date(2024, 2, 1), 1, 2, 0.5, 1.5, 10)]})
    provider = PriceHistoryDataProvider(repository=repo)

    provider.get_historical_prices(
        "MSFT", datetime(2024, 1, 1, 9, 30), datetime(2024, 3, 1, 16, 0)
    )

    _, kwargs = repo.get_bulk_price_history.call_args
    assert kwargs["start_date"] == date(2024, 1, 1)
    assert kwargs["end_date"] == date(2024, 3, 1)


# ---------------------------------------------------------------------------
# Empty-symbol test: no data -> fail-loud, no fabrication.
# ---------------------------------------------------------------------------
def test_empty_symbol_raises_no_market_data():
    repo = _FakeRepo({})  # no symbols have data
    provider = PriceHistoryDataProvider(repository=repo)

    with pytest.raises(NoMarketDataError) as exc:
        provider.get_historical_prices("NOPE", date(2024, 1, 1), date(2024, 1, 31))

    assert "NOPE" in str(exc.value)


def test_bulk_fails_loud_when_any_symbol_missing():
    repo = _FakeRepo({"AAPL": [_fake_price_row(date(2024, 1, 2), 10, 12, 9, 11, 900)]})
    provider = PriceHistoryDataProvider(repository=repo)

    # AAPL has data but TSLA does not -> still fail loud, no partial fabrication.
    with pytest.raises(NoMarketDataError) as exc:
        provider.get_bulk_historical_prices(["AAPL", "TSLA"], date(2024, 1, 1), date(2024, 1, 31))

    assert "TSLA" in str(exc.value)


def test_price_history_to_dataframe_rejects_empty():
    with pytest.raises(NoMarketDataError):
        price_history_to_dataframe([])


def test_no_fabrication_returns_only_real_rows():
    rows = [
        _fake_price_row(date(2024, 1, 2), 10, 12, 9, 11, 900),
        _fake_price_row(date(2024, 1, 3), 11, 13, 10, 12, 1000),
    ]
    repo = _FakeRepo({"AAPL": rows})
    provider = PriceHistoryDataProvider(repository=repo)

    frame = provider.get_historical_prices("AAPL")

    # Exactly the rows the repo returned -- nothing synthesised to fill gaps.
    assert len(frame) == 2
