"""
Unit tests for the real daily-OHLC price backfill.

Covers the new ``backend.tasks.data_tasks`` backfill path that replaced the
old fake-flat-OHLC ``store_price_data`` write:

  * candle -> PriceHistory row mapping is correct (real values, normalized date)
  * ``bulk_upsert_prices`` is called with the real candle values
  * NO rows are written when the source returns nothing (no fabrication)

Follows the same Celery-mocking strategy as ``test_celery_tasks.py`` so we test
pure function logic without Celery infrastructure.
"""
import sys
import types
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import patch, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Mock Celery before importing any task modules (mirrors test_celery_tasks.py).
# ---------------------------------------------------------------------------
_celery_mock = types.ModuleType("celery")
_celery_mock.shared_task = lambda *a, **kw: (lambda fn: fn)
_celery_mock.Celery = MagicMock
_celery_mock.Task = MagicMock
_celery_mock.group = MagicMock
_celery_mock.chain = MagicMock
_celery_mock.current_app = MagicMock

_celery_exceptions = types.ModuleType("celery.exceptions")
_celery_exceptions.SoftTimeLimitExceeded = type("SoftTimeLimitExceeded", (Exception,), {})

sys.modules.setdefault("celery", _celery_mock)
sys.modules.setdefault("celery.exceptions", _celery_exceptions)


class _FakeCeleryApp:
    """Minimal stand-in for the Celery app. .task() is a no-op decorator."""

    class Task:
        pass

    @staticmethod
    def task(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda fn: fn


class _TaskPriority:
    LOW = 0
    NORMAL = 5
    HIGH = 9
    CRITICAL = 10


_celery_app_mod = types.ModuleType("backend.tasks.celery_app")
_celery_app_mod.celery_app = _FakeCeleryApp()
_celery_app_mod.TaskPriority = _TaskPriority
sys.modules["backend.tasks.celery_app"] = _celery_app_mod

# Clear any previously-imported task modules so they re-import against the fake
# celery_app. We must also drop them as attributes on the parent package,
# because `from backend.tasks.X import Y` would otherwise find the old
# (real-celery) module via the cached attribute rather than re-importing.
_tasks_pkg = sys.modules.get("backend.tasks")
for _task_mod in list(sys.modules):
    if _task_mod.startswith("backend.tasks.") and _task_mod != "backend.tasks.celery_app":
        _attr = _task_mod.rsplit(".", 1)[-1]
        if _tasks_pkg is not None and hasattr(_tasks_pkg, _attr):
            delattr(_tasks_pkg, _attr)
        del sys.modules[_task_mod]

for mod_name in ("psutil", "redis"):
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()


# =========================================================================
# Helpers
# =========================================================================

class FakeDBContext:
    """Fake context manager returned by get_db_sync()."""

    def __init__(self, session):
        self.session = session

    def __enter__(self):
        return self.session

    def __exit__(self, *args):
        return False


def _make_stock(symbol="AAPL", stock_id=1):
    s = MagicMock()
    s.id = stock_id
    s.symbol = symbol
    return s


def _make_candle(ts, open_, high, low, close, volume):
    """Build a Finnhub-shaped candle dict."""
    return {
        "timestamp": ts,
        "date": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }


# A couple of real-looking daily candles (two distinct trading days).
_DAY1_TS = int(datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc).timestamp())
_DAY2_TS = int(datetime(2024, 1, 3, 0, 0, tzinfo=timezone.utc).timestamp())


# =========================================================================
# Candle -> row mapping
# =========================================================================

class TestCandlesToPriceRows:
    """Tests for _candles_to_price_rows mapping correctness."""

    def test_maps_real_values_and_normalizes_date(self):
        from backend.tasks.data_tasks import _candles_to_price_rows

        candles = [_make_candle(_DAY1_TS, 100.0, 105.5, 99.25, 104.75, 1_000_000)]
        rows = _candles_to_price_rows(stock_id=7, candles=candles)

        assert len(rows) == 1
        row = rows[0]
        assert row["stock_id"] == 7
        # Real OHLC preserved (as Decimal, not flat/synthetic).
        assert row["open"] == Decimal("100.0")
        assert row["high"] == Decimal("105.5")
        assert row["low"] == Decimal("99.25")
        assert row["close"] == Decimal("104.75")
        assert row["volume"] == 1_000_000
        # Not a flat candle.
        assert not (row["open"] == row["high"] == row["low"] == row["close"])
        # Date normalized to midnight (no time component), tz-naive.
        assert row["date"].hour == 0
        assert row["date"].minute == 0
        assert row["date"].tzinfo is None
        assert row["date"].date() == datetime(2024, 1, 2).date()

    def test_skips_incomplete_candles(self):
        from backend.tasks.data_tasks import _candles_to_price_rows

        candles = [
            _make_candle(_DAY1_TS, 100.0, 105.0, 99.0, 104.0, 500_000),  # valid
            {"timestamp": _DAY2_TS, "open": 10.0, "high": 11.0, "low": 9.0,
             "close": None, "volume": 100},  # missing close -> skip
        ]
        rows = _candles_to_price_rows(stock_id=1, candles=candles)
        assert len(rows) == 1
        assert rows[0]["close"] == Decimal("104.0")

    def test_skips_nonpositive_close(self):
        from backend.tasks.data_tasks import _candles_to_price_rows

        candles = [_make_candle(_DAY1_TS, 0.0, 0.0, 0.0, 0.0, 0)]
        rows = _candles_to_price_rows(stock_id=1, candles=candles)
        assert rows == []

    def test_empty_candles_yields_no_rows(self):
        from backend.tasks.data_tasks import _candles_to_price_rows

        assert _candles_to_price_rows(stock_id=1, candles=[]) == []


# =========================================================================
# backfill_symbol_prices end-to-end (mocked client + persistence)
# =========================================================================

class TestBackfillSymbolPrices:
    """Tests for backfill_symbol_prices: persistence + no fabrication.

    Daily history comes from yfinance (Finnhub's free tier blocks /stock/candle),
    so we mock ``_fetch_yfinance_candles`` as the source.
    """

    @patch("backend.tasks.data_tasks._persist_price_rows")
    @patch("backend.tasks.data_tasks._fetch_yfinance_candles")
    @patch("backend.tasks.data_tasks.get_db_sync")
    def test_persists_real_candle_values(
        self, mock_get_db, mock_fetch, mock_persist
    ):
        from backend.tasks.data_tasks import backfill_symbol_prices

        # Stock lookup returns a real stock.
        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = _make_stock(stock_id=42)
        mock_get_db.return_value = FakeDBContext(db)

        # yfinance returns two real candles.
        mock_fetch.return_value = [
            _make_candle(_DAY1_TS, 100.0, 105.0, 99.0, 104.0, 1_000_000),
            _make_candle(_DAY2_TS, 104.0, 108.0, 103.0, 107.5, 1_200_000),
        ]
        mock_persist.return_value = 2

        result = backfill_symbol_prices("AAPL", days=30)

        assert result["status"] == "success"
        assert result["rows_written"] == 2
        assert result["candles_fetched"] == 2

        mock_fetch.assert_called_once()

        # bulk_upsert path received the REAL candle values (not flat OHLC).
        mock_persist.assert_called_once()
        _, passed_rows = mock_persist.call_args[0]
        assert len(passed_rows) == 2
        assert passed_rows[0]["close"] == Decimal("104.0")
        assert passed_rows[1]["close"] == Decimal("107.5")
        assert passed_rows[0]["stock_id"] == 42
        # Confirm no flat/synthetic row was produced.
        for r in passed_rows:
            assert not (r["open"] == r["high"] == r["low"] == r["close"])

    @patch("backend.tasks.data_tasks._persist_price_rows")
    @patch("backend.tasks.data_tasks._fetch_yfinance_candles")
    @patch("backend.tasks.data_tasks.get_db_sync")
    def test_empty_source_writes_nothing(
        self, mock_get_db, mock_fetch, mock_persist
    ):
        from backend.tasks.data_tasks import backfill_symbol_prices

        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = _make_stock()
        mock_get_db.return_value = FakeDBContext(db)

        # Source returns nothing (yfinance empty or failed).
        mock_fetch.return_value = []

        result = backfill_symbol_prices("ZZZZ", days=30)

        assert result["status"] == "no_data"
        assert result["rows_written"] == 0
        # CRITICAL: never fabricate -> persistence must not be invoked.
        mock_persist.assert_not_called()

    @patch("backend.tasks.data_tasks._persist_price_rows")
    @patch("backend.tasks.data_tasks._fetch_yfinance_candles")
    @patch("backend.tasks.data_tasks.get_db_sync")
    def test_unknown_symbol_returns_not_found(
        self, mock_get_db, mock_fetch, mock_persist
    ):
        from backend.tasks.data_tasks import backfill_symbol_prices

        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = None
        mock_get_db.return_value = FakeDBContext(db)

        result = backfill_symbol_prices("NOPE", days=30)

        assert result["status"] == "not_found"
        assert result["rows_written"] == 0
        # No fetch is attempted for an unknown symbol.
        mock_fetch.assert_not_called()
        mock_persist.assert_not_called()


# =========================================================================
# store_price_data: must NOT write fake flat OHLC anymore
# =========================================================================

class TestStorePriceDataNoFabrication:
    """The repurposed store_price_data delegates to the real-candle backfill."""

    @patch("backend.tasks.data_tasks.backfill_symbol_prices")
    def test_delegates_to_real_backfill(self, mock_backfill):
        from backend.tasks.data_tasks import store_price_data

        mock_backfill.return_value = {"status": "success", "rows_written": 3}
        ok = store_price_data("AAPL", {"finnhub": {"c": 150.0, "v": 1000}})
        assert ok is True
        mock_backfill.assert_called_once()
        # It uses a small incremental window, not a full year.
        _, kwargs = mock_backfill.call_args
        assert kwargs.get("days", 0) <= 7

    @patch("backend.tasks.data_tasks.backfill_symbol_prices")
    def test_returns_false_when_no_real_candles(self, mock_backfill):
        from backend.tasks.data_tasks import store_price_data

        mock_backfill.return_value = {"status": "no_data", "rows_written": 0}
        ok = store_price_data("ZZZZ", {"finnhub": {"c": 1.0}})
        assert ok is False


# =========================================================================
# backfill_daily_prices universe orchestration
# =========================================================================

class TestBackfillDailyPricesOrchestration:
    """Tests for the universe-level backfill task summary aggregation."""

    @patch("backend.tasks.data_tasks.time.sleep", return_value=None)
    @patch("backend.tasks.data_tasks.backfill_symbol_prices")
    @patch("backend.tasks.data_tasks.get_db_sync")
    def test_explicit_symbols_aggregate_summary(
        self, mock_get_db, mock_backfill, _mock_sleep
    ):
        from backend.tasks.data_tasks import backfill_daily_prices

        # symbols path doesn't query the DB for the universe.
        mock_backfill.side_effect = [
            {"status": "success", "rows_written": 250},
            {"status": "no_data", "rows_written": 0},
            {"status": "rate_limited", "rows_written": 0},
        ]

        summary = backfill_daily_prices(symbols=["AAPL", "ZZZZ", "MSFT"])

        assert summary["total"] == 3
        assert summary["succeeded"] == 1
        assert summary["skipped_no_data"] == 1
        assert summary["rate_limited"] == 1
        assert summary["rows_written"] == 250
        assert mock_backfill.call_count == 3
