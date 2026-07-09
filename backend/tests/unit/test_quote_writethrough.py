"""
Unit tests for the stock-quote write-through persistence and company-overview
enrichment added to backend/services/stocks_service.py.

Covers:
  (a) write-through calls bulk_upsert_prices with the REAL OHLC from a
      successful external quote (provider + repositories mocked),
  (b) company-overview fields populate when the overview is available and stay
      null when it is not,
  (c) a persistence exception does NOT break the returned quote.

No database or external services required -- all collaborators are mocked,
mirroring the style/fixtures used in test_stocks_service.py.
"""

import sys
import pytest
from datetime import date, datetime, timezone
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from backend.services.stocks_service import StocksService

# `backend.services.__init__` re-exports the singleton, shadowing the module
# under dotted patch paths -- grab the real module so patch.object works.
_stocks_mod = sys.modules["backend.services.stocks_service"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_stock(*, id=1, symbol="AAPL", name="Apple Inc."):
    """Namespace that quacks like a Stock ORM row."""
    return SimpleNamespace(id=id, symbol=symbol, name=name)


def _db_mock():
    """Async DB session mock whose write-through existence pre-check finds NO
    row for today, so persistence proceeds. (The 'already exists' skip test
    configures ``execute`` to return a row explicitly.)"""
    session = AsyncMock()
    result = MagicMock()
    result.first.return_value = None
    session.execute = AsyncMock(return_value=result)
    return session


@pytest.fixture
def service():
    return StocksService()


@pytest.fixture
def stock_repo():
    """Repository that already has the stock row (common steady-state case)."""
    repo = AsyncMock()
    repo.get_by_symbol = AsyncMock(return_value=_make_stock())
    repo.create = AsyncMock(return_value=_make_stock())
    return repo


@pytest.fixture
def price_repo():
    repo = AsyncMock()
    repo.bulk_upsert_prices = AsyncMock(return_value=1)
    return repo


def _patch_collaborators(stock_repo, price_repo, *, overview=None, overview_raises=False):
    """
    Context manager bundle that patches the module-level collaborators used by
    the write-through / enrichment paths.
    """
    overview_mock = AsyncMock()
    if overview_raises:
        overview_mock.side_effect = RuntimeError("provider down")
    else:
        overview_mock.return_value = overview

    return (
        patch.object(_stocks_mod, "stock_repository", stock_repo),
        patch.object(_stocks_mod, "price_repository", price_repo),
        patch.object(_stocks_mod, "fetch_company_overview", overview_mock),
    )


# =========================================================================
# (a) Write-through persists REAL OHLC
# =========================================================================

class TestWriteThroughPersistence:

    @pytest.mark.asyncio
    async def test_persists_real_ohlc_on_successful_quote(self, service, stock_repo, price_repo):
        """A successful external quote upserts a price row with the provider's REAL OHLCV."""
        real_time = {
            "current_price": 155.0,
            "previous_close": 150.0,
            "open": 151.0,
            "high": 157.0,
            "low": 149.0,
            "volume": 60_000_000,
            "timestamp": "2026-05-28T15:30:00+00:00",
            "source": "finnhub",
        }
        p_stock, p_price, p_overview = _patch_collaborators(stock_repo, price_repo)
        with p_stock, p_price, p_overview:
            result = await service.get_stock_quote(
                symbol="aapl", real_time_data=real_time, db=_db_mock(),
            )

        # Response is intact regardless of persistence.
        assert result["symbol"] == "AAPL"
        assert result["price"] == 155.0

        price_repo.bulk_upsert_prices.assert_awaited_once()
        rows = price_repo.bulk_upsert_prices.call_args.args[0]
        assert len(rows) == 1
        row = rows[0]

        # REAL OHLC from the provider -- NOT a flat synthetic row.
        assert float(row["open"]) == 151.0
        assert float(row["high"]) == 157.0
        assert float(row["low"]) == 149.0
        assert float(row["close"]) == 155.0  # close == current price
        assert row["volume"] == 60_000_000
        assert row["stock_id"] == 1

        # Row is NOT flat (compliance: no synthesised open=high=low=close).
        assert not (row["open"] == row["high"] == row["low"] == row["close"])

    @pytest.mark.asyncio
    async def test_persists_date_from_provider_timestamp(self, service, stock_repo, price_repo):
        """The stored date is derived from the provider timestamp, normalised to midnight."""
        real_time = {
            "current_price": 100.0, "previous_close": 99.0,
            "open": 99.5, "high": 101.0, "low": 98.0, "volume": 1234,
            "timestamp": "2026-05-28T19:45:00+00:00",
        }
        p_stock, p_price, p_overview = _patch_collaborators(stock_repo, price_repo)
        with p_stock, p_price, p_overview:
            await service.get_stock_quote(
                symbol="MSFT", real_time_data=real_time, db=_db_mock(),
            )
        row = price_repo.bulk_upsert_prices.call_args.args[0][0]
        assert row["date"] == datetime(2026, 5, 28)

    @pytest.mark.asyncio
    async def test_zero_volume_preserved_not_invented(self, service, stock_repo, price_repo):
        """Finnhub returns no intraday volume; volume is stored honestly as 0."""
        real_time = {
            "c": 200.0, "pc": 195.0, "o": 196.0, "h": 201.0, "l": 194.0,
            # no volume key at all
        }
        p_stock, p_price, p_overview = _patch_collaborators(stock_repo, price_repo)
        with p_stock, p_price, p_overview:
            await service.get_stock_quote(
                symbol="MSFT", real_time_data=real_time, db=_db_mock(),
            )
        row = price_repo.bulk_upsert_prices.call_args.args[0][0]
        assert row["volume"] == 0
        assert float(row["close"]) == 200.0

    @pytest.mark.asyncio
    async def test_missing_ohlc_falls_back_to_close_not_fabricated(self, service, stock_repo, price_repo):
        """When OHLC is genuinely absent, the close anchors the NOT NULL columns honestly."""
        real_time = {"current_price": 123.0, "previous_close": 120.0}
        p_stock, p_price, p_overview = _patch_collaborators(stock_repo, price_repo)
        with p_stock, p_price, p_overview:
            await service.get_stock_quote(
                symbol="TEST", real_time_data=real_time, db=_db_mock(),
            )
        row = price_repo.bulk_upsert_prices.call_args.args[0][0]
        # All anchored to the real close -- the honest "last known price".
        assert float(row["open"]) == 123.0
        assert float(row["close"]) == 123.0

    @pytest.mark.asyncio
    async def test_creates_minimal_stock_when_missing(self, service, price_repo):
        """When the stock row is missing, a minimal real row is created (symbol + real name)."""
        repo = AsyncMock()
        repo.get_by_symbol = AsyncMock(return_value=None)
        repo.create = AsyncMock(return_value=_make_stock(id=42, symbol="NEW", name="New Co"))

        real_time = {
            "current_price": 10.0, "previous_close": 9.5,
            "open": 9.6, "high": 10.2, "low": 9.4, "volume": 500,
        }
        p_stock, p_price, p_overview = _patch_collaborators(
            repo, price_repo, overview={"name": "New Co"},
        )
        with p_stock, p_price, p_overview:
            await service.get_stock_quote(
                symbol="NEW", real_time_data=real_time, db=_db_mock(),
            )

        repo.create.assert_awaited_once()
        created_payload = repo.create.call_args.args[0]
        assert created_payload["symbol"] == "NEW"
        assert created_payload["name"] == "New Co"
        # And the price row used the newly-created stock's id.
        row = price_repo.bulk_upsert_prices.call_args.args[0][0]
        assert row["stock_id"] == 42

    @pytest.mark.asyncio
    async def test_no_extra_provider_call_for_quote(self, service, stock_repo, price_repo):
        """Persistence reuses the supplied quote -- it never re-fetches the quote."""
        real_time = {
            "current_price": 50.0, "previous_close": 49.0,
            "open": 49.2, "high": 50.5, "low": 48.9, "volume": 100,
        }
        quote_mock = AsyncMock(return_value=real_time)
        p_stock, p_price, p_overview = _patch_collaborators(stock_repo, price_repo)
        with p_stock, p_price, p_overview, \
                patch.object(_stocks_mod, "get_real_time_quote", quote_mock):
            await service.get_stock_quote(
                symbol="AAPL", real_time_data=real_time, db=_db_mock(),
            )
        quote_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_persistence_when_today_row_exists(self, service, stock_repo, price_repo):
        """When a price row for today already exists (e.g. from the daily backfill),
        write-through skips the insert -- no duplicate, no unique-constraint error."""
        real_time = {
            "current_price": 155.0, "previous_close": 150.0,
            "open": 151.0, "high": 157.0, "low": 149.0, "volume": 1000,
        }
        # DB existence pre-check finds a row for today.
        db = AsyncMock()
        result = MagicMock()
        result.first.return_value = (1,)
        db.execute = AsyncMock(return_value=result)

        p_stock, p_price, p_overview = _patch_collaborators(stock_repo, price_repo)
        with p_stock, p_price, p_overview:
            res = await service.get_stock_quote(
                symbol="AAPL", real_time_data=real_time, db=db,
            )

        assert res["price"] == 155.0
        # Already have today's row -> no insert attempted.
        price_repo.bulk_upsert_prices.assert_not_called()


# =========================================================================
# (b) Company-overview enrichment
# =========================================================================

class TestOverviewEnrichment:

    @pytest.mark.asyncio
    async def test_overview_fields_populate_when_available(self, service, stock_repo, price_repo):
        """market_cap / pe_ratio / 52-week range are filled from the overview."""
        real_time = {"current_price": 150.0, "previous_close": 148.0}
        overview = {
            "market_cap": 2_500_000_000_000,
            "pe_ratio": 28.5,
            "52_week_high": 199.62,
            "52_week_low": 124.17,
        }
        p_stock, p_price, p_overview = _patch_collaborators(
            stock_repo, price_repo, overview=overview,
        )
        with p_stock, p_price, p_overview:
            result = await service.get_stock_quote(
                symbol="AAPL", real_time_data=real_time, db=_db_mock(),
            )
        assert result["market_cap"] == 2_500_000_000_000
        assert result["pe_ratio"] == 28.5
        assert result["fifty_two_week_high"] == 199.62
        assert result["fifty_two_week_low"] == 124.17

    @pytest.mark.asyncio
    async def test_overview_fields_null_when_unavailable(self, service, stock_repo, price_repo):
        """When no overview is returned, the fundamentals stay null (never 0/fabricated)."""
        real_time = {"current_price": 150.0, "previous_close": 148.0}
        p_stock, p_price, p_overview = _patch_collaborators(
            stock_repo, price_repo, overview=None,
        )
        with p_stock, p_price, p_overview:
            result = await service.get_stock_quote(
                symbol="AAPL", real_time_data=real_time, db=_db_mock(),
            )
        assert result.get("market_cap") is None
        assert result.get("pe_ratio") is None
        assert result.get("fifty_two_week_high") is None
        assert result.get("fifty_two_week_low") is None

    @pytest.mark.asyncio
    async def test_overview_zero_values_treated_as_unknown(self, service, stock_repo, price_repo):
        """Provider 0-defaults for missing numerics are reported as null, not as real $0."""
        real_time = {"current_price": 150.0, "previous_close": 148.0}
        overview = {"market_cap": 0, "pe_ratio": 0, "52_week_high": 0, "52_week_low": 0}
        p_stock, p_price, p_overview = _patch_collaborators(
            stock_repo, price_repo, overview=overview,
        )
        with p_stock, p_price, p_overview:
            result = await service.get_stock_quote(
                symbol="AAPL", real_time_data=real_time, db=_db_mock(),
            )
        assert result.get("market_cap") is None
        assert result.get("pe_ratio") is None
        assert result.get("fifty_two_week_high") is None
        assert result.get("fifty_two_week_low") is None

    @pytest.mark.asyncio
    async def test_overview_does_not_clobber_quote_provided_values(self, service, stock_repo, price_repo):
        """A 52-week value already present on the quote is preserved, not overwritten."""
        # The quote itself carried a real 52-week high/low (provider included it).
        real_time = {
            "current_price": 150.0, "previous_close": 148.0,
            "52_week_high": 180.0, "52_week_low": 120.0,
        }
        overview = {"52_week_high": 999.0, "52_week_low": 1.0, "market_cap": 5}
        p_stock, p_price, p_overview = _patch_collaborators(
            stock_repo, price_repo, overview=overview,
        )
        with p_stock, p_price, p_overview:
            result = await service.get_stock_quote(
                symbol="AAPL", real_time_data=real_time, db=_db_mock(),
            )
        assert result["fifty_two_week_high"] == 180.0
        assert result["fifty_two_week_low"] == 120.0

    @pytest.mark.asyncio
    async def test_overview_failure_does_not_break_quote(self, service, stock_repo, price_repo):
        """If the overview lookup raises, the quote is still returned without fundamentals."""
        real_time = {"current_price": 150.0, "previous_close": 148.0}
        p_stock, p_price, p_overview = _patch_collaborators(
            stock_repo, price_repo, overview_raises=True,
        )
        with p_stock, p_price, p_overview:
            result = await service.get_stock_quote(
                symbol="AAPL", real_time_data=real_time, db=_db_mock(),
            )
        assert result["price"] == 150.0
        assert result.get("market_cap") is None


# =========================================================================
# (c) Persistence failure must never break the response
# =========================================================================

class TestPersistenceIsBestEffort:

    @pytest.mark.asyncio
    async def test_upsert_exception_does_not_break_quote(self, service, stock_repo):
        """A bulk_upsert failure is swallowed; the quote is returned unchanged."""
        failing_price_repo = AsyncMock()
        failing_price_repo.bulk_upsert_prices = AsyncMock(
            side_effect=RuntimeError("DB write failed")
        )
        real_time = {
            "current_price": 155.0, "previous_close": 150.0,
            "open": 151.0, "high": 157.0, "low": 149.0, "volume": 60_000_000,
            "source": "finnhub",
        }
        p_stock, p_price, p_overview = _patch_collaborators(stock_repo, failing_price_repo)
        with p_stock, p_price, p_overview:
            result = await service.get_stock_quote(
                symbol="AAPL", real_time_data=real_time, db=_db_mock(),
            )
        assert result["symbol"] == "AAPL"
        assert result["price"] == 155.0
        assert result["change"] == pytest.approx(5.0)
        failing_price_repo.bulk_upsert_prices.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_by_symbol_exception_does_not_break_quote(self, service, price_repo):
        """A failure while looking up / creating the stock row is swallowed too."""
        broken_repo = AsyncMock()
        broken_repo.get_by_symbol = AsyncMock(side_effect=RuntimeError("session boom"))
        real_time = {
            "current_price": 42.0, "previous_close": 40.0,
            "open": 40.5, "high": 43.0, "low": 39.0, "volume": 10,
        }
        p_stock, p_price, p_overview = _patch_collaborators(broken_repo, price_repo)
        with p_stock, p_price, p_overview:
            result = await service.get_stock_quote(
                symbol="AAPL", real_time_data=real_time, db=_db_mock(),
            )
        assert result["price"] == 42.0
        # Upsert never reached because the lookup blew up first, but no error
        # surfaced to the caller.
        price_repo.bulk_upsert_prices.assert_not_called()

    @pytest.mark.asyncio
    async def test_minimal_stock_create_failure_skips_persistence(self, service, price_repo):
        """If creating the missing stock fails, persistence is skipped quietly."""
        repo = AsyncMock()
        repo.get_by_symbol = AsyncMock(return_value=None)
        repo.create = AsyncMock(side_effect=RuntimeError("NOT NULL exchange_id"))
        real_time = {
            "current_price": 10.0, "previous_close": 9.0,
            "open": 9.2, "high": 10.5, "low": 8.9, "volume": 5,
        }
        p_stock, p_price, p_overview = _patch_collaborators(repo, price_repo, overview=None)
        with p_stock, p_price, p_overview:
            result = await service.get_stock_quote(
                symbol="ZZZZ", real_time_data=real_time, db=_db_mock(),
            )
        assert result["price"] == 10.0
        price_repo.bulk_upsert_prices.assert_not_called()
