"""
Unit tests for backend/services/stocks_service.py

Tests all public methods of StocksService with mocked dependencies.
No database or external services required.
"""

import sys
import pytest
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from backend.services.stocks_service import StocksService, stocks_service

# The module `backend.services.__init__` re-exports the `stocks_service` singleton,
# which shadows the actual module when using dotted patch paths.  We grab the
# real module object from sys.modules so that `patch.object(MOD, ...)` works.
_stocks_mod = sys.modules["backend.services.stocks_service"]


# ---------------------------------------------------------------------------
# Helpers -- lightweight stand-ins for ORM objects
# ---------------------------------------------------------------------------

def _make_stock(*, id=1, symbol="AAPL", company_name="Apple Inc.",
                sector="Technology", market_cap=2500000000000,
                is_active=True, is_tradable=True):
    """Return a namespace that quacks like a Stock ORM object."""
    return SimpleNamespace(
        id=id,
        symbol=symbol,
        company_name=company_name,
        sector=sector,
        market_cap=market_cap,
        is_active=is_active,
        is_tradable=is_tradable,
    )


def _make_price(*, close=150.0, open=148.0, high=152.0, low=147.0,
                volume=50000000, price_date=None):
    """Return a namespace that quacks like a PriceHistory row."""
    if price_date is None:
        price_date = date.today()
    return SimpleNamespace(
        close=Decimal(str(close)),
        open=Decimal(str(open)),
        high=Decimal(str(high)),
        low=Decimal(str(low)),
        volume=volume,
        date=price_date,
    )


def _make_alert(*, alert_id="alert-001", is_active=True, is_recurring=False,
                created_at=None):
    """Return a namespace that quacks like an Alert ORM object."""
    return SimpleNamespace(
        alert_id=alert_id,
        is_active=is_active,
        is_recurring=is_recurring,
        created_at=created_at or datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# Fixture: fresh StocksService instance (no singleton state leaks)
# ---------------------------------------------------------------------------

@pytest.fixture
def service():
    return StocksService()


@pytest.fixture
def mock_stock_repo():
    """Return an AsyncMock standing in for stock_repository."""
    repo = AsyncMock()
    repo.get_multi = AsyncMock(return_value=[_make_stock()])
    repo.search_stocks = AsyncMock(return_value=[_make_stock()])
    repo.get_by_symbol = AsyncMock(return_value=_make_stock())
    repo.get_sector_summary = AsyncMock(return_value=[
        {"sector": "Technology", "count": 100, "avg_market_cap": 500e9},
        {"sector": "Healthcare", "count": 80, "avg_market_cap": 200e9},
        {"sector": None, "count": 5, "avg_market_cap": 100e6},
    ])
    repo.get_top_performers = AsyncMock(return_value=[
        {"stock": "AAPL", "start_price": 140.0, "end_price": 180.0, "performance_pct": 28.57},
        {"stock": "MSFT", "start_price": 300.0, "end_price": 370.0, "performance_pct": 23.33},
    ])
    return repo


@pytest.fixture
def mock_price_repo():
    """Return an AsyncMock standing in for price_repository."""
    repo = AsyncMock()
    repo.get_latest_price = AsyncMock(return_value=_make_price())
    repo.get_previous_price = AsyncMock(return_value=_make_price(close=145.0))
    repo.get_price_history = AsyncMock(return_value=[
        _make_price(close=150.0),
        _make_price(close=148.0),
    ])
    repo.get_price_statistics = AsyncMock(return_value={
        "avg_price": 149.0,
        "min_price": 140.0,
        "max_price": 160.0,
        "avg_volume": 45000000,
    })
    repo.get_volatility = AsyncMock(return_value=0.25)
    return repo


@pytest.fixture
def mock_alert_repo():
    """Return an AsyncMock standing in for alert_repository."""
    repo = AsyncMock()
    repo.create = AsyncMock(return_value=_make_alert())
    return repo


# =========================================================================
# get_stocks
# =========================================================================

class TestGetStocks:

    @pytest.mark.asyncio
    async def test_get_stocks_basic(self, service, mock_stock_repo):
        """Basic call with defaults should return stock list."""
        with patch.object(_stocks_mod, "stock_repository", mock_stock_repo):
            result = await service.get_stocks(
                sector=None, min_market_cap=None, max_market_cap=None,
                is_active=True, limit=50, offset=0,
                sort_by="symbol", order="asc", db=AsyncMock(),
            )
        assert len(result) == 1
        assert result[0].symbol == "AAPL"
        mock_stock_repo.get_multi.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_stocks_with_sector_filter(self, service, mock_stock_repo):
        """Sector filter should add a FilterCriteria for sector."""
        with patch.object(_stocks_mod, "stock_repository", mock_stock_repo):
            await service.get_stocks(
                sector="Technology", min_market_cap=None, max_market_cap=None,
                is_active=False, limit=20, offset=0,
                sort_by="symbol", order="asc", db=AsyncMock(),
            )
        call_kwargs = mock_stock_repo.get_multi.call_args
        filters = call_kwargs.kwargs["filters"]
        sector_filters = [f for f in filters if f.field == "sector"]
        assert len(sector_filters) == 1
        assert sector_filters[0].value == "Technology"

    @pytest.mark.asyncio
    async def test_get_stocks_with_market_cap_range(self, service, mock_stock_repo):
        """Min and max market cap should produce gte/lte filters."""
        with patch.object(_stocks_mod, "stock_repository", mock_stock_repo):
            await service.get_stocks(
                sector=None, min_market_cap=1e9, max_market_cap=1e12,
                is_active=False, limit=50, offset=0,
                sort_by="market_cap", order="desc", db=AsyncMock(),
            )
        call_kwargs = mock_stock_repo.get_multi.call_args
        filters = call_kwargs.kwargs["filters"]
        gte_filters = [f for f in filters if f.operator == "gte"]
        lte_filters = [f for f in filters if f.operator == "lte"]
        assert len(gte_filters) == 1
        assert gte_filters[0].value == int(1e9)
        assert len(lte_filters) == 1
        assert lte_filters[0].value == int(1e12)

    @pytest.mark.asyncio
    async def test_get_stocks_is_active_adds_two_filters(self, service, mock_stock_repo):
        """is_active=True should add is_active and is_tradable filters."""
        with patch.object(_stocks_mod, "stock_repository", mock_stock_repo):
            await service.get_stocks(
                sector=None, min_market_cap=None, max_market_cap=None,
                is_active=True, limit=50, offset=0,
                sort_by="symbol", order="asc", db=AsyncMock(),
            )
        call_kwargs = mock_stock_repo.get_multi.call_args
        filters = call_kwargs.kwargs["filters"]
        active_filters = [f for f in filters if f.field in ("is_active", "is_tradable")]
        assert len(active_filters) == 2

    @pytest.mark.asyncio
    async def test_get_stocks_is_active_false_no_active_filters(self, service, mock_stock_repo):
        """is_active=False should not add is_active/is_tradable filters."""
        with patch.object(_stocks_mod, "stock_repository", mock_stock_repo):
            await service.get_stocks(
                sector=None, min_market_cap=None, max_market_cap=None,
                is_active=False, limit=50, offset=0,
                sort_by="symbol", order="asc", db=AsyncMock(),
            )
        call_kwargs = mock_stock_repo.get_multi.call_args
        filters = call_kwargs.kwargs["filters"]
        active_filters = [f for f in filters if f.field in ("is_active", "is_tradable")]
        assert len(active_filters) == 0

    @pytest.mark.asyncio
    async def test_get_stocks_sort_direction_desc(self, service, mock_stock_repo):
        """order='desc' should produce SortDirection.DESC."""
        with patch.object(_stocks_mod, "stock_repository", mock_stock_repo):
            await service.get_stocks(
                sector=None, min_market_cap=None, max_market_cap=None,
                is_active=False, limit=50, offset=0,
                sort_by="market_cap", order="desc", db=AsyncMock(),
            )
        call_kwargs = mock_stock_repo.get_multi.call_args
        sort_params = call_kwargs.kwargs["sort_params"]
        from backend.repositories import SortDirection
        assert sort_params[0].direction == SortDirection.DESC

    @pytest.mark.asyncio
    async def test_get_stocks_sort_direction_asc(self, service, mock_stock_repo):
        """order='asc' should produce SortDirection.ASC."""
        with patch.object(_stocks_mod, "stock_repository", mock_stock_repo):
            await service.get_stocks(
                sector=None, min_market_cap=None, max_market_cap=None,
                is_active=False, limit=50, offset=0,
                sort_by="symbol", order="asc", db=AsyncMock(),
            )
        call_kwargs = mock_stock_repo.get_multi.call_args
        sort_params = call_kwargs.kwargs["sort_params"]
        from backend.repositories import SortDirection
        assert sort_params[0].direction == SortDirection.ASC

    @pytest.mark.asyncio
    async def test_get_stocks_pagination(self, service, mock_stock_repo):
        """Offset and limit should be passed as PaginationParams."""
        with patch.object(_stocks_mod, "stock_repository", mock_stock_repo):
            await service.get_stocks(
                sector=None, min_market_cap=None, max_market_cap=None,
                is_active=False, limit=25, offset=50,
                sort_by="symbol", order="asc", db=AsyncMock(),
            )
        call_kwargs = mock_stock_repo.get_multi.call_args
        pagination = call_kwargs.kwargs["pagination"]
        assert pagination.limit == 25
        assert pagination.offset == 50

    @pytest.mark.asyncio
    async def test_get_stocks_empty_result(self, service, mock_stock_repo):
        """Empty result from repository should return empty list."""
        mock_stock_repo.get_multi = AsyncMock(return_value=[])
        with patch.object(_stocks_mod, "stock_repository", mock_stock_repo):
            result = await service.get_stocks(
                sector="NonExistent", min_market_cap=None, max_market_cap=None,
                is_active=False, limit=50, offset=0,
                sort_by="symbol", order="asc", db=AsyncMock(),
            )
        assert result == []

    @pytest.mark.asyncio
    async def test_get_stocks_min_market_cap_only(self, service, mock_stock_repo):
        """Only min_market_cap set should add a single gte filter."""
        with patch.object(_stocks_mod, "stock_repository", mock_stock_repo):
            await service.get_stocks(
                sector=None, min_market_cap=5e9, max_market_cap=None,
                is_active=False, limit=50, offset=0,
                sort_by="symbol", order="asc", db=AsyncMock(),
            )
        call_kwargs = mock_stock_repo.get_multi.call_args
        filters = call_kwargs.kwargs["filters"]
        assert len(filters) == 1
        assert filters[0].operator == "gte"


# =========================================================================
# search_stocks
# =========================================================================

class TestSearchStocks:

    @pytest.mark.asyncio
    async def test_search_stocks_returns_results(self, service, mock_stock_repo):
        """Search should delegate to repository and return results."""
        with patch.object(_stocks_mod, "stock_repository", mock_stock_repo):
            result = await service.search_stocks(
                query="AAPL", limit=10, db=AsyncMock(),
            )
        assert len(result) == 1
        mock_stock_repo.search_stocks.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_stocks_empty_query(self, service, mock_stock_repo):
        """Empty query string should still delegate to repository."""
        mock_stock_repo.search_stocks = AsyncMock(return_value=[])
        with patch.object(_stocks_mod, "stock_repository", mock_stock_repo):
            result = await service.search_stocks(
                query="", limit=10, db=AsyncMock(),
            )
        assert result == []

    @pytest.mark.asyncio
    async def test_search_stocks_with_limit(self, service, mock_stock_repo):
        """Limit parameter should be forwarded to repository."""
        with patch.object(_stocks_mod, "stock_repository", mock_stock_repo):
            await service.search_stocks(
                query="Tech", limit=5, db=AsyncMock(),
            )
        call_kwargs = mock_stock_repo.search_stocks.call_args
        assert call_kwargs.kwargs["limit"] == 5


# =========================================================================
# get_stock_detail
# =========================================================================

class TestGetStockDetail:

    @pytest.mark.asyncio
    async def test_get_stock_detail_found(self, service, mock_stock_repo):
        """When stock exists, return the stock object."""
        with patch.object(_stocks_mod, "stock_repository", mock_stock_repo):
            result = await service.get_stock_detail(symbol="AAPL", db=AsyncMock())
        assert result.symbol == "AAPL"
        assert result.company_name == "Apple Inc."

    @pytest.mark.asyncio
    async def test_get_stock_detail_not_found(self, service, mock_stock_repo):
        """When stock does not exist, return None."""
        mock_stock_repo.get_by_symbol = AsyncMock(return_value=None)
        with patch.object(_stocks_mod, "stock_repository", mock_stock_repo):
            result = await service.get_stock_detail(symbol="ZZZZ", db=AsyncMock())
        assert result is None

    @pytest.mark.asyncio
    async def test_get_stock_detail_passes_symbol(self, service, mock_stock_repo):
        """Symbol should be forwarded to the repository."""
        with patch.object(_stocks_mod, "stock_repository", mock_stock_repo):
            await service.get_stock_detail(symbol="MSFT", db=AsyncMock())
        mock_stock_repo.get_by_symbol.assert_called_once()
        assert mock_stock_repo.get_by_symbol.call_args[0][0] == "MSFT"


# =========================================================================
# get_stock_quote -- external data path
# =========================================================================

class TestGetStockQuoteExternal:

    @pytest.mark.asyncio
    async def test_quote_with_real_time_data(self, service):
        """When real_time_data is provided, build quote from external data."""
        real_time = {
            "price": 155.0,
            "previous_close": 150.0,
            "volume": 60000000,
            "open": 151.0,
            "high": 156.0,
            "low": 149.0,
            "source": "finnhub",
        }
        result = await service.get_stock_quote(
            symbol="aapl", real_time_data=real_time, db=AsyncMock(),
        )
        assert result["symbol"] == "AAPL"
        assert result["price"] == 155.0
        assert result["change"] == pytest.approx(5.0)
        assert result["change_percent"] == pytest.approx(100 * 5.0 / 150.0)
        assert result["volume"] == 60000000
        assert result["data_source"] == "finnhub"
        assert result["is_real_time"] is True

    @pytest.mark.asyncio
    async def test_quote_with_finnhub_style_keys(self, service):
        """Finnhub uses short keys (c, pc, v, o, h, l)."""
        real_time = {
            "c": 200.0,
            "pc": 195.0,
            "v": 30000000,
            "o": 196.0,
            "h": 201.0,
            "l": 194.0,
        }
        result = await service.get_stock_quote(
            symbol="MSFT", real_time_data=real_time, db=AsyncMock(),
        )
        assert result["price"] == 200.0
        assert result["change"] == pytest.approx(5.0)
        assert result["volume"] == 30000000
        assert result["open"] == 196.0
        assert result["high"] == 201.0
        assert result["low"] == 194.0

    @pytest.mark.asyncio
    async def test_quote_external_symbol_uppercased(self, service):
        """Symbol should be uppercased in the result."""
        result = await service.get_stock_quote(
            symbol="aapl",
            real_time_data={"price": 100.0, "previous_close": 100.0},
            db=AsyncMock(),
        )
        assert result["symbol"] == "AAPL"

    @pytest.mark.asyncio
    async def test_quote_external_zero_previous_close(self, service):
        """When previous_close is 0, change and change_percent should be 0."""
        real_time = {"price": 100.0, "previous_close": 0.0, "volume": 1000}
        result = await service.get_stock_quote(
            symbol="TEST", real_time_data=real_time, db=AsyncMock(),
        )
        assert result["change"] == 0.0
        assert result["change_percent"] == 0.0

    @pytest.mark.asyncio
    async def test_quote_external_no_previous_close(self, service):
        """When previous_close is missing, it defaults to current price."""
        real_time = {"price": 100.0, "volume": 1000}
        result = await service.get_stock_quote(
            symbol="TEST", real_time_data=real_time, db=AsyncMock(),
        )
        # previous_close defaults to current_price=100, so change=0
        assert result["change"] == 0.0
        assert result["previous_close"] is None  # same as current price

    @pytest.mark.asyncio
    async def test_quote_external_optional_fields_absent(self, service):
        """Optional fields (bid, ask, pe, 52wk) should be None when absent."""
        real_time = {"price": 100.0, "previous_close": 99.0}
        result = await service.get_stock_quote(
            symbol="TEST", real_time_data=real_time, db=AsyncMock(),
        )
        assert result["bid"] is None
        assert result["ask"] is None
        assert result["pe_ratio"] is None
        assert result["fifty_two_week_high"] is None
        assert result["fifty_two_week_low"] is None

    @pytest.mark.asyncio
    async def test_quote_external_optional_fields_present(self, service):
        """Optional fields should be populated when provided."""
        real_time = {
            "price": 150.0,
            "previous_close": 148.0,
            "bid": 149.5,
            "ask": 150.5,
            "pe": 25.0,
            "52_week_high": 180.0,
            "52_week_low": 120.0,
        }
        result = await service.get_stock_quote(
            symbol="AAPL", real_time_data=real_time, db=AsyncMock(),
        )
        assert result["bid"] == 149.5
        assert result["ask"] == 150.5
        assert result["pe_ratio"] == 25.0
        assert result["fifty_two_week_high"] == 180.0
        assert result["fifty_two_week_low"] == 120.0

    @pytest.mark.asyncio
    async def test_quote_external_has_timestamp(self, service):
        """Result should contain timestamp and last_updated."""
        real_time = {"price": 100.0, "previous_close": 99.0}
        result = await service.get_stock_quote(
            symbol="TEST", real_time_data=real_time, db=AsyncMock(),
        )
        assert isinstance(result["timestamp"], datetime)
        assert isinstance(result["last_updated"], datetime)

    @pytest.mark.asyncio
    async def test_quote_external_default_source(self, service):
        """When source is not in real_time_data, default to 'external_api'."""
        real_time = {"price": 100.0}
        result = await service.get_stock_quote(
            symbol="TEST", real_time_data=real_time, db=AsyncMock(),
        )
        assert result["data_source"] == "external_api"

    @pytest.mark.asyncio
    async def test_quote_external_zero_volume_default(self, service):
        """When volume is missing, default to 0."""
        real_time = {"price": 100.0}
        result = await service.get_stock_quote(
            symbol="TEST", real_time_data=real_time, db=AsyncMock(),
        )
        assert result["volume"] == 0


# =========================================================================
# get_stock_quote -- database fallback path
# =========================================================================

class TestGetStockQuoteDatabase:

    @pytest.mark.asyncio
    async def test_quote_fallback_to_database(
        self, service, mock_stock_repo, mock_price_repo,
    ):
        """When real_time_data is None, fall back to database."""
        with patch.object(_stocks_mod, "stock_repository", mock_stock_repo), \
             patch.object(_stocks_mod, "price_repository", mock_price_repo):
            result = await service.get_stock_quote(
                symbol="AAPL", real_time_data=None, db=AsyncMock(),
            )
        assert result["symbol"] == "AAPL"
        assert result["price"] == 150.0  # latest_price.close
        assert result["data_source"] == "database"
        assert result["is_real_time"] is False

    @pytest.mark.asyncio
    async def test_quote_db_with_previous_price(
        self, service, mock_stock_repo, mock_price_repo,
    ):
        """When previous price exists, calculate change from it."""
        with patch.object(_stocks_mod, "stock_repository", mock_stock_repo), \
             patch.object(_stocks_mod, "price_repository", mock_price_repo):
            result = await service.get_stock_quote(
                symbol="AAPL", real_time_data=None, db=AsyncMock(),
            )
        # latest close=150, previous close=145
        assert result["change"] == pytest.approx(5.0)
        assert result["change_percent"] == pytest.approx(100 * 5.0 / 145.0)
        assert result["previous_close"] == 145.0

    @pytest.mark.asyncio
    async def test_quote_db_no_previous_price(
        self, service, mock_stock_repo, mock_price_repo,
    ):
        """When no previous price, previous_close falls back to current close."""
        mock_price_repo.get_previous_price = AsyncMock(return_value=None)
        with patch.object(_stocks_mod, "stock_repository", mock_stock_repo), \
             patch.object(_stocks_mod, "price_repository", mock_price_repo):
            result = await service.get_stock_quote(
                symbol="AAPL", real_time_data=None, db=AsyncMock(),
            )
        assert result["change"] == 0.0
        assert result["previous_close"] is None

    @pytest.mark.asyncio
    async def test_quote_db_stock_not_found_raises(
        self, service, mock_stock_repo, mock_price_repo,
    ):
        """When stock not found in DB, raise HTTPException 404."""
        from fastapi import HTTPException
        mock_stock_repo.get_by_symbol = AsyncMock(return_value=None)
        with patch.object(_stocks_mod, "stock_repository", mock_stock_repo), \
             patch.object(_stocks_mod, "price_repository", mock_price_repo):
            with pytest.raises(HTTPException) as exc_info:
                await service.get_stock_quote(
                    symbol="ZZZZ", real_time_data=None, db=AsyncMock(),
                )
        assert exc_info.value.status_code == 404
        assert "ZZZZ" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_quote_db_no_price_data_raises(
        self, service, mock_stock_repo, mock_price_repo,
    ):
        """When no price data exists, raise HTTPException 404."""
        from fastapi import HTTPException
        mock_price_repo.get_latest_price = AsyncMock(return_value=None)
        with patch.object(_stocks_mod, "stock_repository", mock_stock_repo), \
             patch.object(_stocks_mod, "price_repository", mock_price_repo):
            with pytest.raises(HTTPException) as exc_info:
                await service.get_stock_quote(
                    symbol="AAPL", real_time_data=None, db=AsyncMock(),
                )
        assert exc_info.value.status_code == 404
        assert "price data" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_quote_db_includes_market_cap(
        self, service, mock_stock_repo, mock_price_repo,
    ):
        """Database fallback should include market_cap from stock."""
        with patch.object(_stocks_mod, "stock_repository", mock_stock_repo), \
             patch.object(_stocks_mod, "price_repository", mock_price_repo):
            result = await service.get_stock_quote(
                symbol="AAPL", real_time_data=None, db=AsyncMock(),
            )
        assert result["market_cap"] == 2500000000000

    @pytest.mark.asyncio
    async def test_quote_db_includes_ohlv(
        self, service, mock_stock_repo, mock_price_repo,
    ):
        """Database fallback should include open, high, low, volume."""
        with patch.object(_stocks_mod, "stock_repository", mock_stock_repo), \
             patch.object(_stocks_mod, "price_repository", mock_price_repo):
            result = await service.get_stock_quote(
                symbol="AAPL", real_time_data=None, db=AsyncMock(),
            )
        assert result["open"] == 148.0
        assert result["high"] == 152.0
        assert result["low"] == 147.0
        assert result["volume"] == 50000000


# =========================================================================
# get_price_history
# =========================================================================

class TestGetPriceHistory:

    @pytest.mark.asyncio
    async def test_get_price_history_with_dates(self, service, mock_price_repo):
        """Price history with explicit dates should forward them."""
        start = date(2024, 1, 1)
        end = date(2024, 12, 31)
        with patch.object(_stocks_mod, "price_repository", mock_price_repo):
            result = await service.get_price_history(
                symbol="AAPL", start_date=start, end_date=end,
                limit=None, db=AsyncMock(),
            )
        assert len(result) == 2
        mock_price_repo.get_price_history.assert_called_once()
        call_kwargs = mock_price_repo.get_price_history.call_args.kwargs
        assert call_kwargs["start_date"] == start
        assert call_kwargs["end_date"] == end

    @pytest.mark.asyncio
    async def test_get_price_history_default_dates(self, service, mock_price_repo):
        """When no dates provided, default to 1 year ending today."""
        with patch.object(_stocks_mod, "price_repository", mock_price_repo):
            await service.get_price_history(
                symbol="AAPL", start_date=None, end_date=None,
                limit=None, db=AsyncMock(),
            )
        call_kwargs = mock_price_repo.get_price_history.call_args.kwargs
        assert call_kwargs["end_date"] == date.today()
        expected_start = date.today() - timedelta(days=365)
        assert call_kwargs["start_date"] == expected_start

    @pytest.mark.asyncio
    async def test_get_price_history_only_end_date(self, service, mock_price_repo):
        """When only end_date is missing, default end to today."""
        start = date(2024, 6, 1)
        with patch.object(_stocks_mod, "price_repository", mock_price_repo):
            await service.get_price_history(
                symbol="AAPL", start_date=start, end_date=None,
                limit=None, db=AsyncMock(),
            )
        call_kwargs = mock_price_repo.get_price_history.call_args.kwargs
        assert call_kwargs["end_date"] == date.today()
        assert call_kwargs["start_date"] == start

    @pytest.mark.asyncio
    async def test_get_price_history_with_limit(self, service, mock_price_repo):
        """Limit parameter should be forwarded to repository."""
        with patch.object(_stocks_mod, "price_repository", mock_price_repo):
            await service.get_price_history(
                symbol="AAPL", start_date=None, end_date=None,
                limit=100, db=AsyncMock(),
            )
        call_kwargs = mock_price_repo.get_price_history.call_args.kwargs
        assert call_kwargs["limit"] == 100

    @pytest.mark.asyncio
    async def test_get_price_history_empty_result(self, service, mock_price_repo):
        """No price data should return empty list."""
        mock_price_repo.get_price_history = AsyncMock(return_value=[])
        with patch.object(_stocks_mod, "price_repository", mock_price_repo):
            result = await service.get_price_history(
                symbol="ZZZZ", start_date=None, end_date=None,
                limit=None, db=AsyncMock(),
            )
        assert result == []


# =========================================================================
# get_stock_statistics
# =========================================================================

class TestGetStockStatistics:

    @pytest.mark.asyncio
    async def test_statistics_with_volatility(self, service, mock_price_repo):
        """Statistics with volatility should include volatility_annualized."""
        with patch.object(_stocks_mod, "price_repository", mock_price_repo):
            result = await service.get_stock_statistics(
                symbol="AAPL", days=90, db=AsyncMock(),
            )
        assert result is not None
        assert result["avg_price"] == 149.0
        assert result["volatility_annualized"] == 0.25

    @pytest.mark.asyncio
    async def test_statistics_no_data_returns_none(self, service, mock_price_repo):
        """When no statistics exist, return None."""
        mock_price_repo.get_price_statistics = AsyncMock(return_value=None)
        with patch.object(_stocks_mod, "price_repository", mock_price_repo):
            result = await service.get_stock_statistics(
                symbol="ZZZZ", days=30, db=AsyncMock(),
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_statistics_volatility_none_excluded(self, service, mock_price_repo):
        """When volatility is None, it should not be added to statistics."""
        mock_price_repo.get_volatility = AsyncMock(return_value=None)
        with patch.object(_stocks_mod, "price_repository", mock_price_repo):
            result = await service.get_stock_statistics(
                symbol="AAPL", days=30, db=AsyncMock(),
            )
        assert "volatility_annualized" not in result

    @pytest.mark.asyncio
    async def test_statistics_volatility_days_capped_at_30(self, service, mock_price_repo):
        """Volatility days should be min(days, 30)."""
        with patch.object(_stocks_mod, "price_repository", mock_price_repo):
            await service.get_stock_statistics(
                symbol="AAPL", days=365, db=AsyncMock(),
            )
        vol_call = mock_price_repo.get_volatility.call_args
        assert vol_call.kwargs["days"] == 30

    @pytest.mark.asyncio
    async def test_statistics_volatility_days_less_than_30(self, service, mock_price_repo):
        """When days < 30, volatility uses the actual days value."""
        with patch.object(_stocks_mod, "price_repository", mock_price_repo):
            await service.get_stock_statistics(
                symbol="AAPL", days=14, db=AsyncMock(),
            )
        vol_call = mock_price_repo.get_volatility.call_args
        assert vol_call.kwargs["days"] == 14


# =========================================================================
# get_sectors
# =========================================================================

class TestGetSectors:

    @pytest.mark.asyncio
    async def test_get_sectors_returns_names(self, service, mock_stock_repo):
        """Should return a flat list of sector name strings."""
        with patch.object(_stocks_mod, "stock_repository", mock_stock_repo):
            result = await service.get_sectors(db=AsyncMock())
        assert result == ["Technology", "Healthcare"]

    @pytest.mark.asyncio
    async def test_get_sectors_filters_none(self, service, mock_stock_repo):
        """None sectors should be excluded from the result."""
        with patch.object(_stocks_mod, "stock_repository", mock_stock_repo):
            result = await service.get_sectors(db=AsyncMock())
        assert None not in result
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_get_sectors_empty(self, service, mock_stock_repo):
        """When no sectors exist, return empty list."""
        mock_stock_repo.get_sector_summary = AsyncMock(return_value=[])
        with patch.object(_stocks_mod, "stock_repository", mock_stock_repo):
            result = await service.get_sectors(db=AsyncMock())
        assert result == []

    @pytest.mark.asyncio
    async def test_get_sectors_all_none(self, service, mock_stock_repo):
        """When all sectors are None, return empty list."""
        mock_stock_repo.get_sector_summary = AsyncMock(return_value=[
            {"sector": None, "count": 5},
        ])
        with patch.object(_stocks_mod, "stock_repository", mock_stock_repo):
            result = await service.get_sectors(db=AsyncMock())
        assert result == []


# =========================================================================
# get_sector_summary
# =========================================================================

class TestGetSectorSummary:

    @pytest.mark.asyncio
    async def test_get_sector_summary_delegates(self, service, mock_stock_repo):
        """Should delegate to repository and return raw results."""
        with patch.object(_stocks_mod, "stock_repository", mock_stock_repo):
            result = await service.get_sector_summary(db=AsyncMock())
        assert len(result) == 3
        mock_stock_repo.get_sector_summary.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_sector_summary_empty(self, service, mock_stock_repo):
        """Empty sector summary from repository should return empty list."""
        mock_stock_repo.get_sector_summary = AsyncMock(return_value=[])
        with patch.object(_stocks_mod, "stock_repository", mock_stock_repo):
            result = await service.get_sector_summary(db=AsyncMock())
        assert result == []


# =========================================================================
# get_top_performers
# =========================================================================

class TestGetTopPerformers:

    @pytest.mark.asyncio
    async def test_get_top_performers_returns_list(self, service, mock_stock_repo):
        """Should return performer dicts from repository."""
        with patch.object(_stocks_mod, "stock_repository", mock_stock_repo):
            result = await service.get_top_performers(
                timeframe="1M", limit=10, db=AsyncMock(),
            )
        assert len(result) == 2
        assert result[0]["stock"] == "AAPL"
        assert result[0]["performance_pct"] == 28.57

    @pytest.mark.asyncio
    async def test_get_top_performers_forwards_params(self, service, mock_stock_repo):
        """Timeframe and limit should be forwarded to repository."""
        with patch.object(_stocks_mod, "stock_repository", mock_stock_repo):
            await service.get_top_performers(
                timeframe="1Y", limit=5, db=AsyncMock(),
            )
        call_kwargs = mock_stock_repo.get_top_performers.call_args.kwargs
        assert call_kwargs["timeframe"] == "1Y"
        assert call_kwargs["limit"] == 5

    @pytest.mark.asyncio
    async def test_get_top_performers_empty(self, service, mock_stock_repo):
        """No performers should return empty list."""
        mock_stock_repo.get_top_performers = AsyncMock(return_value=[])
        with patch.object(_stocks_mod, "stock_repository", mock_stock_repo):
            result = await service.get_top_performers(
                timeframe="1W", limit=10, db=AsyncMock(),
            )
        assert result == []


# =========================================================================
# create_price_alert
# =========================================================================

class TestCreatePriceAlert:

    @pytest.mark.asyncio
    async def test_create_alert_success(
        self, service, mock_stock_repo, mock_alert_repo,
    ):
        """Valid alert creation returns expected dict."""
        with patch.object(_stocks_mod, "stock_repository", mock_stock_repo), \
             patch.object(_stocks_mod, "alert_repository", mock_alert_repo):
            result = await service.create_price_alert(
                user_id=1, symbol="AAPL", condition="above",
                threshold_price=200.0, is_recurring=False, db=AsyncMock(),
            )
        assert result["symbol"] == "AAPL"
        assert result["condition"] == "above"
        assert result["threshold_price"] == 200.0
        assert result["is_active"] is True
        assert result["status"] == "active"
        assert result["alert_id"] == "alert-001"

    @pytest.mark.asyncio
    async def test_create_alert_stock_not_found(
        self, service, mock_stock_repo, mock_alert_repo,
    ):
        """Alert creation for non-existent stock raises HTTPException."""
        from fastapi import HTTPException
        mock_stock_repo.get_by_symbol = AsyncMock(return_value=None)
        with patch.object(_stocks_mod, "stock_repository", mock_stock_repo), \
             patch.object(_stocks_mod, "alert_repository", mock_alert_repo):
            with pytest.raises(HTTPException) as exc_info:
                await service.create_price_alert(
                    user_id=1, symbol="ZZZZ", condition="below",
                    threshold_price=50.0, is_recurring=False, db=AsyncMock(),
                )
        assert exc_info.value.status_code == 404
        assert "ZZZZ" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_create_alert_recurring(
        self, service, mock_stock_repo, mock_alert_repo,
    ):
        """Recurring alert should pass is_recurring=True to repository."""
        recurring_alert = _make_alert(is_recurring=True)
        mock_alert_repo.create = AsyncMock(return_value=recurring_alert)
        with patch.object(_stocks_mod, "stock_repository", mock_stock_repo), \
             patch.object(_stocks_mod, "alert_repository", mock_alert_repo):
            result = await service.create_price_alert(
                user_id=1, symbol="AAPL", condition="below",
                threshold_price=120.0, is_recurring=True, db=AsyncMock(),
            )
        assert result["is_recurring"] is True

    @pytest.mark.asyncio
    async def test_create_alert_payload_structure(
        self, service, mock_stock_repo, mock_alert_repo,
    ):
        """Alert repository create should be called with correct payload."""
        with patch.object(_stocks_mod, "stock_repository", mock_stock_repo), \
             patch.object(_stocks_mod, "alert_repository", mock_alert_repo):
            await service.create_price_alert(
                user_id=42, symbol="AAPL", condition="above",
                threshold_price=175.0, is_recurring=False, db=AsyncMock(),
            )
        create_call = mock_alert_repo.create.call_args
        data = create_call.kwargs["data"]
        assert data["user_id"] == 42
        assert data["stock_id"] == 1  # from _make_stock().id
        assert data["alert_type"] == "price_threshold"
        assert data["is_active"] is True
        assert data["is_recurring"] is False
        assert data["condition"]["type"] == "price_threshold"
        assert data["condition"]["condition"] == "above"
        assert data["condition"]["threshold_price"] == 175.0

    @pytest.mark.asyncio
    async def test_create_alert_has_created_at(
        self, service, mock_stock_repo, mock_alert_repo,
    ):
        """Alert result should include created_at datetime."""
        with patch.object(_stocks_mod, "stock_repository", mock_stock_repo), \
             patch.object(_stocks_mod, "alert_repository", mock_alert_repo):
            result = await service.create_price_alert(
                user_id=1, symbol="AAPL", condition="above",
                threshold_price=200.0, is_recurring=False, db=AsyncMock(),
            )
        assert isinstance(result["created_at"], datetime)

    @pytest.mark.asyncio
    async def test_create_alert_no_created_at_on_orm_uses_fallback(
        self, service, mock_stock_repo, mock_alert_repo,
    ):
        """When ORM alert has no created_at, fallback to current time."""
        alert_no_ts = _make_alert()
        alert_no_ts.created_at = None
        mock_alert_repo.create = AsyncMock(return_value=alert_no_ts)
        with patch.object(_stocks_mod, "stock_repository", mock_stock_repo), \
             patch.object(_stocks_mod, "alert_repository", mock_alert_repo):
            result = await service.create_price_alert(
                user_id=1, symbol="AAPL", condition="above",
                threshold_price=200.0, is_recurring=False, db=AsyncMock(),
            )
        # Should have a datetime (the fallback from `or datetime.now(...)`)
        assert isinstance(result["created_at"], datetime)


# =========================================================================
# Module-level cached functions
# =========================================================================

class TestGetRealTimeQuote:

    @pytest.mark.asyncio
    async def test_quote_from_finnhub(self):
        """When finnhub_client is available, use it."""
        mock_fh = AsyncMock()
        mock_fh.get_quote = AsyncMock(return_value={"c": 155.0})
        with patch.object(_stocks_mod, "finnhub_client", mock_fh), \
             patch.object(_stocks_mod, "alpha_vantage_client", None), \
             patch.object(_stocks_mod, "polygon_client", None):
            from backend.services.stocks_service import get_real_time_quote
            result = await get_real_time_quote.__wrapped__("AAPL")
        assert result == {"c": 155.0}

    @pytest.mark.asyncio
    async def test_quote_fallback_to_alpha_vantage(self):
        """When finnhub unavailable, fall back to alpha_vantage."""
        mock_av = AsyncMock()
        mock_av.get_quote = AsyncMock(return_value={"price": 150.0})
        with patch.object(_stocks_mod, "finnhub_client", None), \
             patch.object(_stocks_mod, "alpha_vantage_client", mock_av), \
             patch.object(_stocks_mod, "polygon_client", None):
            from backend.services.stocks_service import get_real_time_quote
            result = await get_real_time_quote.__wrapped__("AAPL")
        assert result == {"price": 150.0}

    @pytest.mark.asyncio
    async def test_quote_fallback_to_polygon(self):
        """When finnhub and alpha_vantage unavailable, fall back to polygon."""
        mock_pg = AsyncMock()
        mock_pg.get_quote = AsyncMock(return_value={"price": 148.0})
        with patch.object(_stocks_mod, "finnhub_client", None), \
             patch.object(_stocks_mod, "alpha_vantage_client", None), \
             patch.object(_stocks_mod, "polygon_client", mock_pg):
            from backend.services.stocks_service import get_real_time_quote
            result = await get_real_time_quote.__wrapped__("AAPL")
        assert result == {"price": 148.0}

    @pytest.mark.asyncio
    async def test_quote_no_providers_returns_none(self):
        """When no providers available, return None."""
        with patch.object(_stocks_mod, "finnhub_client", None), \
             patch.object(_stocks_mod, "alpha_vantage_client", None), \
             patch.object(_stocks_mod, "polygon_client", None):
            from backend.services.stocks_service import get_real_time_quote
            result = await get_real_time_quote.__wrapped__("AAPL")
        assert result is None

    @pytest.mark.asyncio
    async def test_quote_provider_exception_returns_none(self):
        """When provider raises, return None."""
        mock_fh = AsyncMock()
        mock_fh.get_quote = AsyncMock(side_effect=ConnectionError("API down"))
        with patch.object(_stocks_mod, "finnhub_client", mock_fh), \
             patch.object(_stocks_mod, "alpha_vantage_client", None), \
             patch.object(_stocks_mod, "polygon_client", None):
            from backend.services.stocks_service import get_real_time_quote
            result = await get_real_time_quote.__wrapped__("AAPL")
        assert result is None


class TestFetchCompanyOverview:

    @pytest.mark.asyncio
    async def test_overview_from_alpha_vantage(self):
        """When alpha_vantage_client is available, use it."""
        mock_av = AsyncMock()
        mock_av.get_company_overview = AsyncMock(
            return_value={"sector": "Technology", "pe_ratio": 28.0}
        )
        with patch.object(_stocks_mod, "alpha_vantage_client", mock_av), \
             patch.object(_stocks_mod, "finnhub_client", None):
            from backend.services.stocks_service import fetch_company_overview
            result = await fetch_company_overview.__wrapped__("AAPL")
        assert result["sector"] == "Technology"

    @pytest.mark.asyncio
    async def test_overview_fallback_to_finnhub(self):
        """When alpha_vantage unavailable, fall back to finnhub."""
        mock_fh = AsyncMock()
        mock_fh.get_company_profile = AsyncMock(
            return_value={"name": "Apple Inc."}
        )
        with patch.object(_stocks_mod, "alpha_vantage_client", None), \
             patch.object(_stocks_mod, "finnhub_client", mock_fh):
            from backend.services.stocks_service import fetch_company_overview
            result = await fetch_company_overview.__wrapped__("AAPL")
        assert result["name"] == "Apple Inc."

    @pytest.mark.asyncio
    async def test_overview_no_providers_returns_none(self):
        """When no providers available, return None."""
        with patch.object(_stocks_mod, "alpha_vantage_client", None), \
             patch.object(_stocks_mod, "finnhub_client", None):
            from backend.services.stocks_service import fetch_company_overview
            result = await fetch_company_overview.__wrapped__("AAPL")
        assert result is None

    @pytest.mark.asyncio
    async def test_overview_exception_returns_none(self):
        """When provider raises, return None."""
        mock_av = AsyncMock()
        mock_av.get_company_overview = AsyncMock(
            side_effect=RuntimeError("rate limited")
        )
        with patch.object(_stocks_mod, "alpha_vantage_client", mock_av), \
             patch.object(_stocks_mod, "finnhub_client", None):
            from backend.services.stocks_service import fetch_company_overview
            result = await fetch_company_overview.__wrapped__("AAPL")
        assert result is None


# =========================================================================
# Singleton instance
# =========================================================================

class TestSingletonInstance:

    def test_stocks_service_is_stocks_service(self):
        """Module-level stocks_service should be a StocksService instance."""
        assert isinstance(stocks_service, StocksService)


# =========================================================================
# _build_quote_from_external edge cases
# =========================================================================

class TestBuildQuoteFromExternalEdgeCases:

    def test_negative_change(self, service):
        """Price drop should produce negative change and change_percent."""
        result = service._build_quote_from_external(
            "AAPL",
            {"price": 140.0, "previous_close": 150.0, "volume": 1000},
            "test",
        )
        assert result["change"] == pytest.approx(-10.0)
        assert result["change_percent"] < 0

    def test_large_volume(self, service):
        """Very large volume values should be handled correctly."""
        result = service._build_quote_from_external(
            "AAPL",
            {"price": 150.0, "volume": 999999999999},
            "test",
        )
        assert result["volume"] == 999999999999

    def test_missing_price_defaults_to_zero(self, service):
        """When neither 'price' nor 'c' is present, default to 0."""
        result = service._build_quote_from_external(
            "TEST", {}, "test",
        )
        assert result["price"] == 0.0
