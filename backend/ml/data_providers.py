"""
Backtest data providers.

Adapts the existing :class:`PriceHistoryRepository` to the OHLCV pandas
``DataFrame`` shape that :mod:`backend.ml.backtesting` consumes.  This is a
thin adapter over the repository -- it deliberately does *not* introduce a new
``MarketDataProvider`` abstraction.  The DDD seam already exists as
``backend.domain.contracts.MarketDataContract``; this provider simply reuses
the repository to keep the backtest path concrete and fail-loud.

Design notes
------------
* The backtest engine consumes data synchronously (it iterates trading dates in
  a plain ``for`` loop).  The repository is async, so this provider fetches all
  required history up front via :meth:`PriceHistoryRepository.get_bulk_price_history`
  and converts each symbol's ``List[PriceHistory]`` into an OHLCV ``DataFrame``
  indexed by date with columns ``open/high/low/close/volume``.
* Fail-loud: if the repository returns no rows for a requested symbol the
  provider raises ``NoMarketDataError`` rather than synthesising prices.  No
  random data is ever generated here.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

# OHLCV columns the backtester expects on each per-symbol DataFrame.
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


class NoMarketDataError(RuntimeError):
    """Raised when no price history exists for a requested symbol.

    Fail-loud sentinel: the backtest data path must never fabricate prices, so
    a missing symbol is surfaced as an error rather than silently filled.
    """


def _to_naive_timestamp(value: Any) -> pd.Timestamp:
    """Coerce a date/datetime to a tz-naive pandas Timestamp for the index."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts


def price_history_to_dataframe(records: Sequence[Any]) -> pd.DataFrame:
    """Convert a list of ``PriceHistory`` rows into an OHLCV ``DataFrame``.

    Args:
        records: Sequence of ``PriceHistory`` model instances (or any object
            exposing ``date``, ``open``, ``high``, ``low``, ``close``,
            ``volume``).  Decimal/None values are coerced to float.

    Returns:
        A ``DataFrame`` indexed by date (ascending) with columns
        ``open/high/low/close/volume``.

    Raises:
        NoMarketDataError: If ``records`` is empty.  We never return a
            fabricated or empty-but-valid frame for a missing symbol.
    """
    if not records:
        raise NoMarketDataError(
            "No price history records available to build an OHLCV DataFrame"
        )

    rows = []
    index = []
    for record in records:
        index.append(_to_naive_timestamp(record.date))
        rows.append(
            {
                "open": float(record.open),
                "high": float(record.high),
                "low": float(record.low),
                "close": float(record.close),
                "volume": float(record.volume),
            }
        )

    frame = pd.DataFrame(rows, index=pd.DatetimeIndex(index), columns=OHLCV_COLUMNS)
    # Repository returns chronological order, but sort defensively so the
    # backtester's ``data[data.index <= date]`` slicing is always correct.
    frame = frame.sort_index()
    return frame


class PriceHistoryDataProvider:
    """Backtest data provider backed by :class:`PriceHistoryRepository`.

    The backtest engine calls :meth:`get_historical_prices` for each symbol in
    the universe and :meth:`get_historical_prices` again for the benchmark
    symbol.  Both go through the repository; no abstraction layer is added.

    Args:
        repository: A ``PriceHistoryRepository`` (or compatible object exposing
            ``get_bulk_price_history``).  If ``None``, the module-level
            ``price_repository`` singleton is imported lazily so that importing
            this module never pulls SQLAlchemy into light-weight test
            processes.
        limit_per_symbol: Max rows fetched per symbol (passed through to the
            repository).  Large enough to cover typical backtest windows.
    """

    def __init__(
        self,
        repository: Optional[Any] = None,
        *,
        limit_per_symbol: int = 5000,
    ) -> None:
        self._repository = repository
        self._limit_per_symbol = limit_per_symbol

    @property
    def repository(self) -> Any:
        """Return the repository, importing the default singleton lazily."""
        if self._repository is None:
            # Lazy import keeps this module import-light for hermetic tests.
            from backend.repositories.price_repository import price_repository

            self._repository = price_repository
        return self._repository

    def _normalize_dates(
        self,
        start_date: Optional[Any],
        end_date: Optional[Any],
    ) -> tuple[Optional[date], Optional[date]]:
        """Coerce datetime/Timestamp bounds to plain ``date`` for the repo."""
        def _as_date(value: Optional[Any]) -> Optional[date]:
            if value is None:
                return None
            if isinstance(value, datetime):
                return value.date()
            if isinstance(value, pd.Timestamp):
                return value.to_pydatetime().date()
            if isinstance(value, date):
                return value
            return pd.Timestamp(value).to_pydatetime().date()

        return _as_date(start_date), _as_date(end_date)

    async def _fetch_bulk(
        self,
        symbols: List[str],
        start_date: Optional[date],
        end_date: Optional[date],
    ) -> Dict[str, List[Any]]:
        return await self.repository.get_bulk_price_history(
            symbols,
            start_date=start_date,
            end_date=end_date,
            limit_per_symbol=self._limit_per_symbol,
        )

    def _run_bulk(
        self,
        symbols: List[str],
        start_date: Optional[date],
        end_date: Optional[date],
    ) -> Dict[str, List[Any]]:
        """Run the async bulk fetch from a synchronous context.

        The backtest engine is synchronous; ``asyncio.run`` bridges to the
        async repository.  If an event loop is already running (e.g. inside an
        async API handler) the caller should pre-fetch instead -- we raise a
        clear error rather than corrupting the loop.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._fetch_bulk(symbols, start_date, end_date))
        raise RuntimeError(
            "PriceHistoryDataProvider cannot fetch synchronously while an event "
            "loop is running; fetch market data before invoking the backtest."
        )

    def get_bulk_historical_prices(
        self,
        symbols: List[str],
        start_date: Optional[Any] = None,
        end_date: Optional[Any] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch OHLCV DataFrames for several symbols in one repository call.

        Raises:
            NoMarketDataError: If any requested symbol has no price history.
        """
        if not symbols:
            return {}

        start, end = self._normalize_dates(start_date, end_date)
        bulk = self._run_bulk([s.upper() for s in symbols], start, end)

        frames: Dict[str, pd.DataFrame] = {}
        missing: List[str] = []
        for symbol in symbols:
            records = bulk.get(symbol.upper()) or []
            if not records:
                missing.append(symbol)
                continue
            frames[symbol] = price_history_to_dataframe(records)

        if missing:
            raise NoMarketDataError(
                f"No price history found for symbol(s): {', '.join(sorted(missing))}"
            )
        return frames

    def get_historical_prices(
        self,
        symbol: str,
        start_date: Optional[Any] = None,
        end_date: Optional[Any] = None,
    ) -> pd.DataFrame:
        """Fetch an OHLCV DataFrame for a single symbol.

        Args:
            symbol: Stock ticker symbol.
            start_date: Inclusive start bound (date/datetime/Timestamp).
            end_date: Inclusive end bound.

        Returns:
            OHLCV ``DataFrame`` indexed by date.

        Raises:
            NoMarketDataError: If the symbol has no price history (fail-loud).
        """
        return self.get_bulk_historical_prices([symbol], start_date, end_date)[symbol]
