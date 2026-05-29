"""
Unit tests for the S&P 500 universe seed path.

Covers:
- backend/tasks/stock_universe_fetcher.fetch_sp500_universe
- backend/tasks/stock_universe_fetcher.persist_universe

The real Wikipedia fetch (pandas.read_html) is mocked to return a small fixed
list of REAL S&P 500 constituents. The DB session is a lightweight in-memory
fake so the tests stay fast and never touch a real database. Idempotency is
asserted by running persist twice and checking no duplicate Stock rows are
created on the second run.
"""

import pytest
import pandas as pd
from unittest.mock import patch

from backend.tasks.stock_universe_fetcher import (
    StockInfo,
    UniverseSourceError,
    fetch_sp500_universe,
    persist_universe,
)
from backend.models.unified_models import Stock, Sector, Exchange


# ---------------------------------------------------------------------------
# Fixtures: a small fixed list of REAL S&P 500 constituents
# ---------------------------------------------------------------------------

def _real_sp500_dataframe() -> pd.DataFrame:
    """Wikipedia-shaped table with a handful of genuine S&P 500 members.

    BRK.B is included specifically to exercise '.' -> '-' symbol normalization.
    """
    return pd.DataFrame(
        {
            "Symbol": ["AAPL", "MSFT", "JPM", "JNJ", "BRK.B"],
            "Security": [
                "Apple Inc.",
                "Microsoft Corporation",
                "JPMorgan Chase",
                "Johnson & Johnson",
                "Berkshire Hathaway",
            ],
            "GICS Sector": [
                "Information Technology",
                "Information Technology",
                "Financials",
                "Health Care",
                "Financials",
            ],
            "GICS Sub-Industry": [
                "Technology Hardware, Storage & Peripherals",
                "Systems Software",
                "Diversified Banks",
                "Pharmaceuticals",
                "Multi-Sector Holdings",
            ],
        }
    )


# ---------------------------------------------------------------------------
# In-memory fake AsyncSession for persist tests
# ---------------------------------------------------------------------------

class _FakeResult:
    def __init__(self, value):
        self._value = value

    def scalar_one_or_none(self):
        return self._value


class FakeAsyncSession:
    """Minimal stateful AsyncSession stand-in.

    Tracks added Stock/Sector/Exchange instances and resolves subsequent
    SELECT-by-unique-field queries against them, which lets us prove the
    upsert path is idempotent across repeated persist runs.
    """

    def __init__(self):
        self.stocks: dict[str, Stock] = {}
        self.sectors: dict[str, Sector] = {}
        self.exchanges: dict[str, Exchange] = {}
        self._id_counter = 0

    async def execute(self, stmt):
        # Inspect the compiled WHERE to figure out which entity is queried.
        # We rely on the column the statement filters on; the seed code only
        # ever filters Stock.symbol, Sector.name, Exchange.code.
        entity = stmt.column_descriptions[0]["entity"]
        text = str(stmt.compile(compile_kwargs={"literal_binds": True}))

        if entity is Stock:
            for symbol, obj in self.stocks.items():
                if f"'{symbol}'" in text:
                    return _FakeResult(obj)
            return _FakeResult(None)
        if entity is Sector:
            for name, obj in self.sectors.items():
                if f"'{name}'" in text:
                    return _FakeResult(obj)
            return _FakeResult(None)
        if entity is Exchange:
            for code, obj in self.exchanges.items():
                if f"'{code}'" in text:
                    return _FakeResult(obj)
            return _FakeResult(None)
        return _FakeResult(None)

    def add(self, obj):
        self._id_counter += 1
        if getattr(obj, "id", None) is None:
            obj.id = self._id_counter
        if isinstance(obj, Stock):
            self.stocks[obj.symbol] = obj
        elif isinstance(obj, Sector):
            self.sectors[obj.name] = obj
        elif isinstance(obj, Exchange):
            self.exchanges[obj.code] = obj

    async def flush(self):
        return None


# ---------------------------------------------------------------------------
# fetch_sp500_universe
# ---------------------------------------------------------------------------

class TestFetchSP500Universe:
    def test_parses_real_constituents(self):
        with patch("backend.tasks.stock_universe_fetcher.pd.read_html",
                   return_value=[_real_sp500_dataframe()]):
            stocks = fetch_sp500_universe()

        assert len(stocks) == 5
        symbols = {s.symbol for s in stocks}
        assert {"AAPL", "MSFT", "JPM", "JNJ"}.issubset(symbols)
        apple = next(s for s in stocks if s.symbol == "AAPL")
        assert apple.name == "Apple Inc."
        assert apple.sector == "Information Technology"

    def test_normalizes_share_class_symbols(self):
        """BRK.B must become BRK-B (dot -> dash) to match platform convention."""
        with patch("backend.tasks.stock_universe_fetcher.pd.read_html",
                   return_value=[_real_sp500_dataframe()]):
            stocks = fetch_sp500_universe()

        symbols = {s.symbol for s in stocks}
        assert "BRK-B" in symbols
        assert "BRK.B" not in symbols

    def test_fails_loudly_when_source_unreachable(self):
        """A network/parse error must raise UniverseSourceError, never fabricate."""
        with patch("backend.tasks.stock_universe_fetcher.pd.read_html",
                   side_effect=ValueError("no tables found")):
            with pytest.raises(UniverseSourceError):
                fetch_sp500_universe()

    def test_fails_loudly_on_empty_table(self):
        """Zero valid constituents must raise rather than seed an empty universe."""
        empty = pd.DataFrame({"Symbol": [], "Security": [],
                              "GICS Sector": [], "GICS Sub-Industry": []})
        with patch("backend.tasks.stock_universe_fetcher.pd.read_html",
                   return_value=[empty]):
            with pytest.raises(UniverseSourceError):
                fetch_sp500_universe()

    def test_fails_loudly_on_unexpected_schema(self):
        """Missing the Symbol/Security columns must raise (source shape changed)."""
        wrong = pd.DataFrame({"Ticker": ["AAPL"], "Company": ["Apple Inc."]})
        with patch("backend.tasks.stock_universe_fetcher.pd.read_html",
                   return_value=[wrong]):
            with pytest.raises(UniverseSourceError):
                fetch_sp500_universe()


# ---------------------------------------------------------------------------
# persist_universe
# ---------------------------------------------------------------------------

class TestPersistUniverse:
    def _sample(self) -> list[StockInfo]:
        return [
            StockInfo(symbol="AAPL", name="Apple Inc.", exchange="",
                      sector="Information Technology"),
            StockInfo(symbol="MSFT", name="Microsoft Corporation", exchange="",
                      sector="Information Technology"),
            StockInfo(symbol="JPM", name="JPMorgan Chase", exchange="",
                      sector="Financials"),
        ]

    @pytest.mark.asyncio
    async def test_first_run_upserts_rows(self):
        session = FakeAsyncSession()
        summary = await persist_universe(self._sample(), session=session)

        assert summary["seeded"] == 3
        assert summary["created"] == 3
        assert summary["updated"] == 0
        # Two distinct sectors -> deduped to FK rows
        assert summary["sectors"] == 2
        assert set(session.stocks.keys()) == {"AAPL", "MSFT", "JPM"}

        # Every stock linked to a sector and an exchange, flagged active/tradable.
        for stock in session.stocks.values():
            assert stock.sector_id is not None
            assert stock.exchange_id is not None
            assert stock.is_active is True
            assert stock.is_tradable is True
            # market_cap is NOT fabricated.
            assert getattr(stock, "market_cap", None) is None

    @pytest.mark.asyncio
    async def test_second_run_is_idempotent(self):
        session = FakeAsyncSession()
        stocks = self._sample()

        first = await persist_universe(stocks, session=session)
        assert first["created"] == 3

        # Re-run with the same universe: must update, not duplicate.
        second = await persist_universe(stocks, session=session)

        assert len(session.stocks) == 3  # no duplicate rows
        assert second["created"] == 0
        assert second["updated"] == 3
        assert second["seeded"] == 3
        # Sectors also reused, not re-created.
        assert len(session.sectors) == 2

    @pytest.mark.asyncio
    async def test_exchange_created_once_and_reused(self):
        session = FakeAsyncSession()
        await persist_universe(self._sample(), session=session, default_exchange="NYSE")
        assert set(session.exchanges.keys()) == {"NYSE"}
        # Second run reuses the same exchange row.
        await persist_universe(self._sample(), session=session, default_exchange="NYSE")
        assert len(session.exchanges) == 1

    @pytest.mark.asyncio
    async def test_empty_universe_refused(self):
        session = FakeAsyncSession()
        with pytest.raises(UniverseSourceError):
            await persist_universe([], session=session)
