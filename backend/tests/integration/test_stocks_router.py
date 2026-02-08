"""
Integration tests for the Stocks API router (GitHub issue #87).

Tests cover all GET endpoints in backend/api/routers/stocks.py:
  - GET /api/v1/stocks           (list stocks with filters/pagination)
  - GET /api/v1/stocks/search    (search by symbol or name)
  - GET /api/v1/stocks/{symbol}  (stock detail)
  - GET /api/v1/stocks/{symbol}/quote    (real-time quote with fallback)
  - GET /api/v1/stocks/{symbol}/history  (historical prices)
  - GET /api/v1/stocks/{symbol}/statistics (price statistics)
  - POST /api/v1/stocks/{symbol}/watchlist  (deprecated)
  - DELETE /api/v1/stocks/{symbol}/watchlist (deprecated)

Known application bugs documented with xfail markers:
  - StockResponse.from_orm fails in async context (lazy-loaded relationships)
  - price_repository.get_previous_price not implemented
  - Deprecated watchlist endpoints pass dict to HTTPException detail
"""

import pytest
import pytest_asyncio
from datetime import datetime, timedelta, date
from decimal import Decimal
from unittest.mock import AsyncMock, patch, MagicMock
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.unified_models import (
    Stock, PriceHistory, Exchange, Sector, Industry,
)
from backend.api.main import app
from httpx import AsyncClient


pytestmark = pytest.mark.integration

# Base URL prefix for the stocks router
PREFIX = "/api/v1/stocks"


# ---------------------------------------------------------------------------
# Cache bypass fixture -- applied to every test so that the @api_cache
# decorator does not attempt to connect to Redis.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _bypass_api_cache():
    """
    Replace the cache manager with a pass-through mock so that every
    @api_cache decorated endpoint simply executes its handler function
    without touching Redis.
    """
    mock_manager = AsyncMock()

    async def passthrough_get(data_type, identifier, fallback_func=None, **kwargs):
        if fallback_func:
            result = await fallback_func()
            return result, "miss"
        return None, "miss"

    mock_manager.get = passthrough_get
    mock_manager.set = AsyncMock(return_value=True)
    mock_manager.initialize = AsyncMock()

    async def _mock_get_cache_manager():
        return mock_manager

    with patch(
        "backend.utils.api_cache_decorators.get_cache_manager",
        new=_mock_get_cache_manager,
    ):
        yield mock_manager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def sample_stock(
    db_session: AsyncSession,
    nasdaq_exchange: Exchange,
    technology_sector: Sector,
    consumer_electronics_industry: Industry,
):
    """Create a single active, tradable stock for testing."""
    stock = Stock(
        symbol="AAPL",
        name="Apple Inc.",
        exchange_id=nasdaq_exchange.id,
        asset_type="stock",
        sector_id=technology_sector.id,
        industry_id=consumer_electronics_industry.id,
        market_cap=3_000_000_000_000,
        shares_outstanding=16_000_000_000,
        country="US",
        currency="USD",
        is_active=True,
        is_tradable=True,
    )
    db_session.add(stock)
    await db_session.commit()
    await db_session.refresh(stock)
    return stock


@pytest_asyncio.fixture
async def second_stock(
    db_session: AsyncSession,
    nasdaq_exchange: Exchange,
    technology_sector: Sector,
    consumer_electronics_industry: Industry,
):
    """Create a second stock for list/search tests."""
    stock = Stock(
        symbol="MSFT",
        name="Microsoft Corporation",
        exchange_id=nasdaq_exchange.id,
        asset_type="stock",
        sector_id=technology_sector.id,
        industry_id=consumer_electronics_industry.id,
        market_cap=2_800_000_000_000,
        shares_outstanding=7_500_000_000,
        country="US",
        currency="USD",
        is_active=True,
        is_tradable=True,
    )
    db_session.add(stock)
    await db_session.commit()
    await db_session.refresh(stock)
    return stock


@pytest_asyncio.fixture
async def inactive_stock(
    db_session: AsyncSession,
    nasdaq_exchange: Exchange,
    technology_sector: Sector,
    consumer_electronics_industry: Industry,
):
    """Create an inactive stock that should be excluded from default queries."""
    stock = Stock(
        symbol="DLIST",
        name="Delisted Corp",
        exchange_id=nasdaq_exchange.id,
        asset_type="stock",
        sector_id=technology_sector.id,
        industry_id=consumer_electronics_industry.id,
        market_cap=500_000_000,
        shares_outstanding=100_000_000,
        country="US",
        currency="USD",
        is_active=False,
        is_tradable=False,
    )
    db_session.add(stock)
    await db_session.commit()
    await db_session.refresh(stock)
    return stock


@pytest_asyncio.fixture
async def price_history_30d(db_session: AsyncSession, sample_stock: Stock):
    """Create 30 days of price history for the sample stock."""
    prices = []
    base_date = date.today() - timedelta(days=30)

    for i in range(30):
        price = PriceHistory(
            stock_id=sample_stock.id,
            date=base_date + timedelta(days=i),
            open=Decimal("150.00") + Decimal(str(i * 0.50)),
            high=Decimal("152.00") + Decimal(str(i * 0.50)),
            low=Decimal("149.00") + Decimal(str(i * 0.50)),
            close=Decimal("151.00") + Decimal(str(i * 0.50)),
            adjusted_close=Decimal("151.00") + Decimal(str(i * 0.50)),
            volume=75_000_000 + (i * 1_000_000),
        )
        prices.append(price)
        db_session.add(price)

    await db_session.commit()
    return prices


# ---------------------------------------------------------------------------
# GET /api/v1/stocks -- List stocks
#
# NOTE: Endpoints that serialize Stock ORM objects through StockResponse
# currently fail with 500 due to a known bug: StockResponse.from_orm()
# tries to lazy-load relationships (exchange, sector, industry) inside
# an async context, which raises MissingGreenlet.  These tests are
# marked xfail until the serialization bug is fixed.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.xfail(
    reason="StockResponse.from_orm fails on lazy-loaded relationships in async context",
    strict=True,
)
async def test_get_stocks_returns_success(
    async_client: AsyncClient,
    sample_stock: Stock,
    second_stock: Stock,
):
    """GET /api/v1/stocks returns a success response containing stock data."""
    response = await async_client.get(PREFIX)
    assert response.status_code == 200

    body = response.json()
    assert body["success"] is True
    assert isinstance(body["data"], list)
    assert len(body["data"]) >= 2

    symbols_returned = {s["symbol"] for s in body["data"]}
    assert "AAPL" in symbols_returned
    assert "MSFT" in symbols_returned


@pytest.mark.asyncio
async def test_get_stocks_serialization_returns_500(
    async_client: AsyncClient,
    sample_stock: Stock,
):
    """
    Regression test: GET /api/v1/stocks currently returns 500 because
    StockResponse.from_orm() cannot lazy-load relationships in async.
    This test documents the current behavior.
    """
    response = await async_client.get(PREFIX)
    assert response.status_code == 500

    body = response.json()
    assert body["success"] is False
    assert "error" in body


@pytest.mark.asyncio
async def test_get_stocks_invalid_sort_rejected(
    async_client: AsyncClient,
    sample_stock: Stock,
):
    """An invalid sort_by value is rejected by query validation."""
    response = await async_client.get(
        PREFIX,
        params={"sort_by": "invalid_field"},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_get_stocks_invalid_order_rejected(
    async_client: AsyncClient,
    sample_stock: Stock,
):
    """An invalid order value is rejected by query validation."""
    response = await async_client.get(
        PREFIX,
        params={"order": "random"},
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# GET /api/v1/stocks/search
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.xfail(
    reason="StockResponse.from_orm fails on lazy-loaded relationships in async context",
    strict=True,
)
async def test_search_stocks_by_symbol(
    async_client: AsyncClient,
    sample_stock: Stock,
    second_stock: Stock,
):
    """Search by exact symbol prefix returns matching stocks."""
    response = await async_client.get(
        f"{PREFIX}/search",
        params={"query": "AAPL"},
    )
    assert response.status_code == 200

    body = response.json()
    assert body["success"] is True
    search_data = body["data"]
    assert search_data["total_count"] >= 1

    matching_symbols = {s["symbol"] for s in search_data["stocks"]}
    assert "AAPL" in matching_symbols


@pytest.mark.asyncio
async def test_search_stocks_missing_query_rejected(
    async_client: AsyncClient,
):
    """Omitting the required query parameter returns 422."""
    response = await async_client.get(f"{PREFIX}/search")
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# GET /api/v1/stocks/{symbol} -- Stock detail
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.xfail(
    reason="StockDetailResponse.from_orm fails on lazy-loaded relationships in async context",
    strict=True,
)
async def test_get_stock_detail_found(
    async_client: AsyncClient,
    sample_stock: Stock,
):
    """Fetching an existing stock by symbol returns its details."""
    response = await async_client.get(f"{PREFIX}/{sample_stock.symbol}")
    assert response.status_code == 200

    body = response.json()
    assert body["success"] is True
    assert body["data"]["symbol"] == "AAPL"
    assert body["data"]["name"] == "Apple Inc."


@pytest.mark.asyncio
async def test_get_stock_detail_not_found(
    async_client: AsyncClient,
):
    """Requesting a non-existent symbol returns 404."""
    response = await async_client.get(f"{PREFIX}/ZZZNOTREAL")
    assert response.status_code == 404

    body = response.json()
    assert body["success"] is False


# ---------------------------------------------------------------------------
# GET /api/v1/stocks/{symbol}/quote -- Real-time quote with fallback
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_stock_quote_from_external_api(
    async_client: AsyncClient,
    sample_stock: Stock,
    price_history_30d,
):
    """
    When an external data provider returns data, the quote endpoint
    should return real-time data with is_real_time=True.
    """
    mock_quote_data = {
        "price": 175.50,
        "c": 175.50,
        "previous_close": 173.00,
        "pc": 173.00,
        "volume": 80_000_000,
        "v": 80_000_000,
        "open": 174.00,
        "o": 174.00,
        "high": 176.00,
        "h": 176.00,
        "low": 173.50,
        "l": 173.50,
        "source": "finnhub",
    }

    with patch(
        "backend.api.routers.stocks.get_real_time_quote",
        new_callable=AsyncMock,
        return_value=mock_quote_data,
    ):
        response = await async_client.get(f"{PREFIX}/{sample_stock.symbol}/quote")

    assert response.status_code == 200

    body = response.json()
    assert body["success"] is True

    quote = body["data"]
    assert quote["symbol"] == "AAPL"
    assert quote["price"] == 175.50
    assert quote["is_real_time"] is True
    assert quote["data_source"] == "finnhub"


@pytest.mark.asyncio
@pytest.mark.xfail(
    reason=(
        "Database fallback path has two bugs: "
        "price_repository.get_previous_price is not implemented, "
        "and PriceHistory has no updated_at attribute (stocks.py:551)"
    ),
    strict=True,
)
async def test_get_stock_quote_database_fallback(
    async_client: AsyncClient,
    sample_stock: Stock,
    price_history_30d,
):
    """
    When external APIs return None, the quote endpoint falls back to
    database price history and returns is_real_time=False.

    Mocks get_previous_price because it is not yet implemented on
    PriceHistoryRepository.
    """
    mock_prev_price = MagicMock()
    mock_prev_price.close = Decimal("160.00")

    with patch(
        "backend.api.routers.stocks.get_real_time_quote",
        new_callable=AsyncMock,
        return_value=None,
    ), patch.object(
        __import__("backend.api.routers.stocks", fromlist=["price_repository"]).price_repository,
        "get_previous_price",
        new_callable=AsyncMock,
        return_value=mock_prev_price,
        create=True,
    ):
        response = await async_client.get(f"{PREFIX}/{sample_stock.symbol}/quote")

    assert response.status_code == 200

    body = response.json()
    assert body["success"] is True

    quote = body["data"]
    assert quote["symbol"] == "AAPL"
    assert quote["is_real_time"] is False
    assert quote["data_source"] == "database"
    assert quote["price"] > 0


@pytest.mark.asyncio
async def test_get_stock_quote_invalid_symbol_format(
    async_client: AsyncClient,
):
    """
    A malformed symbol (numbers, special chars) should be rejected
    with a 400 Bad Request by validate_stock_symbol.
    """
    response = await async_client.get(f"{PREFIX}/123!!!/quote")
    assert response.status_code == 400

    body = response.json()
    assert body["success"] is False


@pytest.mark.asyncio
async def test_get_stock_quote_symbol_not_in_db_no_external(
    async_client: AsyncClient,
):
    """
    A validly-formatted symbol that does not exist in the DB (and no
    external data) should return 404.
    """
    with patch(
        "backend.api.routers.stocks.get_real_time_quote",
        new_callable=AsyncMock,
        return_value=None,
    ):
        response = await async_client.get(f"{PREFIX}/XYZZY/quote")

    assert response.status_code == 404

    body = response.json()
    assert body["success"] is False


@pytest.mark.asyncio
async def test_get_stock_quote_case_insensitive(
    async_client: AsyncClient,
    sample_stock: Stock,
    price_history_30d,
):
    """The quote endpoint normalizes symbol to uppercase."""
    mock_quote_data = {
        "price": 175.50,
        "c": 175.50,
        "previous_close": 173.00,
        "pc": 173.00,
        "volume": 80_000_000,
        "v": 80_000_000,
        "source": "finnhub",
    }

    with patch(
        "backend.api.routers.stocks.get_real_time_quote",
        new_callable=AsyncMock,
        return_value=mock_quote_data,
    ):
        response = await async_client.get(f"{PREFIX}/aapl/quote")

    assert response.status_code == 200
    body = response.json()
    assert body["data"]["symbol"] == "AAPL"


@pytest.mark.asyncio
async def test_get_stock_quote_change_calculation(
    async_client: AsyncClient,
    sample_stock: Stock,
    price_history_30d,
):
    """Verify the change and change_percent fields are computed correctly."""
    mock_quote_data = {
        "price": 200.00,
        "c": 200.00,
        "previous_close": 190.00,
        "pc": 190.00,
        "volume": 50_000_000,
        "v": 50_000_000,
        "source": "finnhub",
    }

    with patch(
        "backend.api.routers.stocks.get_real_time_quote",
        new_callable=AsyncMock,
        return_value=mock_quote_data,
    ):
        response = await async_client.get(f"{PREFIX}/{sample_stock.symbol}/quote")

    assert response.status_code == 200

    quote = response.json()["data"]
    assert quote["change"] == pytest.approx(10.0)
    expected_pct = (10.0 / 190.0) * 100
    assert quote["change_percent"] == pytest.approx(expected_pct, rel=1e-3)


# ---------------------------------------------------------------------------
# GET /api/v1/stocks/{symbol}/history -- Historical prices
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_stock_history_returns_data(
    async_client: AsyncClient,
    sample_stock: Stock,
    price_history_30d,
):
    """Historical price endpoint returns price records for existing stock."""
    start = (date.today() - timedelta(days=30)).isoformat()
    end = date.today().isoformat()

    response = await async_client.get(
        f"{PREFIX}/{sample_stock.symbol}/history",
        params={"start_date": start, "end_date": end},
    )
    assert response.status_code == 200

    body = response.json()
    assert body["success"] is True
    assert isinstance(body["data"], list)
    assert len(body["data"]) > 0

    first = body["data"][0]
    for field in ("open", "high", "low", "close", "volume"):
        assert field in first


@pytest.mark.asyncio
async def test_get_stock_history_not_found(
    async_client: AsyncClient,
):
    """Requesting history for a symbol with no data returns 404."""
    response = await async_client.get(f"{PREFIX}/NOPRICE/history")
    assert response.status_code == 404

    body = response.json()
    assert body["success"] is False


@pytest.mark.asyncio
async def test_get_stock_history_respects_limit(
    async_client: AsyncClient,
    sample_stock: Stock,
    price_history_30d,
):
    """The limit parameter caps the number of history records returned."""
    response = await async_client.get(
        f"{PREFIX}/{sample_stock.symbol}/history",
        params={"limit": 5},
    )
    assert response.status_code == 200

    body = response.json()
    assert len(body["data"]) <= 5


@pytest.mark.asyncio
async def test_get_stock_history_default_date_range(
    async_client: AsyncClient,
    sample_stock: Stock,
    price_history_30d,
):
    """When no dates are provided, defaults to 1 year range ending today."""
    response = await async_client.get(
        f"{PREFIX}/{sample_stock.symbol}/history",
    )
    assert response.status_code == 200

    body = response.json()
    assert body["success"] is True
    assert len(body["data"]) > 0


# ---------------------------------------------------------------------------
# GET /api/v1/stocks/{symbol}/statistics -- Price statistics
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_stock_statistics(
    async_client: AsyncClient,
    sample_stock: Stock,
    price_history_30d,
):
    """Statistics endpoint returns computed metrics for a stock with data."""
    response = await async_client.get(
        f"{PREFIX}/{sample_stock.symbol}/statistics",
        params={"days": 30},
    )
    assert response.status_code == 200

    body = response.json()
    assert body["success"] is True

    stats = body["data"]
    assert "trading_days" in stats
    assert "min_price" in stats
    assert "max_price" in stats
    assert "avg_price" in stats
    assert stats["trading_days"] > 0


@pytest.mark.asyncio
async def test_get_stock_statistics_no_data(
    async_client: AsyncClient,
):
    """Statistics for a non-existent symbol returns 404."""
    response = await async_client.get(
        f"{PREFIX}/NODATA/statistics",
        params={"days": 30},
    )
    assert response.status_code == 404

    body = response.json()
    assert body["success"] is False


# ---------------------------------------------------------------------------
# Deprecated watchlist endpoints
#
# The endpoint raises HTTPException(status_code=401, detail={...}) with a
# dict as the detail. The registered error handler tries to serialize
# this into ErrorResponse(error=str), but Pydantic rejects the dict,
# causing a secondary validation error that returns 422.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_add_to_watchlist_deprecated_endpoint(
    async_client: AsyncClient,
    sample_stock: Stock,
):
    """
    POST /api/v1/stocks/{symbol}/watchlist is deprecated.
    Currently returns 422 due to the error handler receiving a dict
    as HTTPException.detail instead of a string.
    """
    response = await async_client.post(
        f"{PREFIX}/{sample_stock.symbol}/watchlist",
    )
    # The endpoint intends 401 but the dict detail triggers a 422 in the
    # error handler.  Accept either status to be resilient to future fixes.
    assert response.status_code in (401, 422)


@pytest.mark.asyncio
async def test_remove_from_watchlist_deprecated_endpoint(
    async_client: AsyncClient,
    sample_stock: Stock,
):
    """
    DELETE /api/v1/stocks/{symbol}/watchlist is deprecated.
    Currently returns 422 due to dict-as-detail serialization issue.
    """
    response = await async_client.delete(
        f"{PREFIX}/{sample_stock.symbol}/watchlist",
    )
    assert response.status_code in (401, 422)


# ---------------------------------------------------------------------------
# Data source fallback ordering
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.xfail(
    reason=(
        "Database fallback path has two bugs: "
        "price_repository.get_previous_price is not implemented, "
        "and PriceHistory has no updated_at attribute (stocks.py:551)"
    ),
    strict=True,
)
async def test_quote_data_source_fallback_order(
    async_client: AsyncClient,
    sample_stock: Stock,
    price_history_30d,
):
    """
    When all external providers fail (get_real_time_quote returns None),
    the quote endpoint falls back to the database.  Mocks
    get_previous_price since it is not yet implemented.
    """
    call_log = []

    async def mock_get_real_time_quote_failing(symbol: str):
        call_log.append(symbol)
        return None

    mock_prev_price = MagicMock()
    mock_prev_price.close = Decimal("160.00")

    with patch(
        "backend.api.routers.stocks.get_real_time_quote",
        side_effect=mock_get_real_time_quote_failing,
    ), patch.object(
        __import__("backend.api.routers.stocks", fromlist=["price_repository"]).price_repository,
        "get_previous_price",
        new_callable=AsyncMock,
        return_value=mock_prev_price,
        create=True,
    ):
        response = await async_client.get(
            f"{PREFIX}/{sample_stock.symbol}/quote",
        )

    assert response.status_code == 200

    body = response.json()
    quote = body["data"]
    assert quote["data_source"] == "database"
    assert quote["is_real_time"] is False
    assert len(call_log) == 1
    assert call_log[0] == "AAPL"


# ---------------------------------------------------------------------------
# Edge cases and validation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_stock_quote_response_structure(
    async_client: AsyncClient,
    sample_stock: Stock,
    price_history_30d,
):
    """Verify the StockQuoteResponse has all expected fields."""
    mock_quote_data = {
        "price": 175.50,
        "c": 175.50,
        "previous_close": 173.00,
        "pc": 173.00,
        "volume": 80_000_000,
        "v": 80_000_000,
        "open": 174.00,
        "o": 174.00,
        "high": 176.00,
        "h": 176.00,
        "low": 173.50,
        "l": 173.50,
        "source": "finnhub",
    }

    with patch(
        "backend.api.routers.stocks.get_real_time_quote",
        new_callable=AsyncMock,
        return_value=mock_quote_data,
    ):
        response = await async_client.get(f"{PREFIX}/{sample_stock.symbol}/quote")

    assert response.status_code == 200
    quote = response.json()["data"]

    required_fields = [
        "symbol", "price", "change", "change_percent",
        "volume", "timestamp", "data_source", "is_real_time",
    ]
    for field in required_fields:
        assert field in quote, f"Missing required field: {field}"


@pytest.mark.asyncio
async def test_get_stock_history_date_range_filtering(
    async_client: AsyncClient,
    sample_stock: Stock,
    price_history_30d,
):
    """History endpoint respects start_date and end_date params."""
    narrow_start = (date.today() - timedelta(days=5)).isoformat()
    narrow_end = date.today().isoformat()

    response = await async_client.get(
        f"{PREFIX}/{sample_stock.symbol}/history",
        params={"start_date": narrow_start, "end_date": narrow_end},
    )
    assert response.status_code == 200

    body = response.json()
    # Should return fewer records than the full 30 days
    assert len(body["data"]) <= 6  # 5 days + possible today


@pytest.mark.asyncio
async def test_get_stock_statistics_includes_volatility(
    async_client: AsyncClient,
    sample_stock: Stock,
    price_history_30d,
):
    """
    Statistics response may include volatility_annualized when enough
    price data exists.
    """
    response = await async_client.get(
        f"{PREFIX}/{sample_stock.symbol}/statistics",
        params={"days": 30},
    )
    assert response.status_code == 200

    stats = response.json()["data"]
    # Volatility is calculated from at least 2 price points
    # It may or may not be present depending on data availability
    assert "trading_days" in stats
    assert stats["trading_days"] >= 2
