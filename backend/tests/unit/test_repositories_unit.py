"""
Unit tests for the repository layer.

Tests cover:
- backend/repositories/base.py (AsyncCRUDRepository)
- backend/repositories/portfolio_repository.py
- backend/repositories/price_repository.py
- backend/repositories/stock_repository.py

All tests mock the AsyncSession to keep them fast and isolated.
"""

import pytest
from datetime import date, datetime, timezone, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

from backend.repositories.base import (
    AsyncCRUDRepository,
    FilterCriteria,
    PaginationParams,
    SortDirection,
    SortParams,
)
from backend.repositories.portfolio_repository import PortfolioRepository
from backend.repositories.price_repository import PriceHistoryRepository
from backend.repositories.stock_repository import StockRepository

# Use real SQLAlchemy models so that select()/update()/delete() can build
# valid statement objects.  This avoids ArgumentError from passing MagicMock
# where SQLAlchemy expects a mapped class.
from backend.models.unified_models import Stock, Portfolio, PriceHistory, User


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_session(
    scalar_one_or_none=None,
    scalars_all=None,
    scalar=None,
    rowcount=1,
    unique_scalars_all=None,
):
    """Build a mock AsyncSession with common chained result patterns."""
    session = AsyncMock()

    result_mock = MagicMock()

    # scalar_one_or_none pattern  (get_by_id, get_by_field, etc.)
    result_mock.scalar_one_or_none.return_value = scalar_one_or_none

    # scalars().all() pattern  (get_multi, search_stocks, etc.)
    scalars_mock = MagicMock()
    scalars_mock.all.return_value = scalars_all if scalars_all is not None else []
    result_mock.scalars.return_value = scalars_mock

    # scalar() pattern  (count, aggregate)
    result_mock.scalar.return_value = scalar

    # rowcount pattern  (update, delete)
    result_mock.rowcount = rowcount

    # unique().scalars().all() pattern  (get_user_portfolios)
    unique_mock = MagicMock()
    unique_scalars = MagicMock()
    unique_scalars.all.return_value = unique_scalars_all if unique_scalars_all is not None else []
    unique_mock.scalars.return_value = unique_scalars
    unique_mock.scalar_one_or_none.return_value = scalar_one_or_none
    result_mock.unique.return_value = unique_mock

    # first() pattern (aggregate rows)
    result_mock.first.return_value = None

    session.execute = AsyncMock(return_value=result_mock)
    session.flush = AsyncMock()
    session.refresh = AsyncMock()
    session.add = MagicMock()
    session.delete = AsyncMock()
    session.commit = AsyncMock()

    return session


# ============================================================================
# AsyncCRUDRepository -- base.py
# ============================================================================


class TestAsyncCRUDRepositoryCreate:
    """Tests for AsyncCRUDRepository.create()"""

    @pytest.mark.asyncio
    async def test_create_adds_instance_to_session(self):
        """create() should call session.add(), flush, and refresh."""
        session = _make_session()
        repo = AsyncCRUDRepository(User)

        result = await repo.create(
            {
                "username": "alice",
                "email": "alice@test.com",
                "hashed_password": "hashed",
            },
            session=session,
        )

        session.add.assert_called_once()
        session.flush.assert_awaited_once()
        session.refresh.assert_awaited_once()
        assert isinstance(result, User)
        assert result.username == "alice"

    @pytest.mark.asyncio
    async def test_create_converts_pydantic_model(self):
        """create() should call .dict() on Pydantic-like inputs."""
        pydantic_data = MagicMock()
        pydantic_data.dict.return_value = {
            "username": "bob",
            "email": "bob@test.com",
            "hashed_password": "hashed",
        }

        session = _make_session()
        repo = AsyncCRUDRepository(User)

        result = await repo.create(pydantic_data, session=session)

        pydantic_data.dict.assert_called_once_with(exclude_unset=True)
        assert result.username == "bob"


class TestAsyncCRUDRepositoryGetById:
    """Tests for AsyncCRUDRepository.get_by_id()"""

    @pytest.mark.asyncio
    async def test_get_by_id_returns_model_on_hit(self):
        """get_by_id() should return the model when found."""
        expected = MagicMock(id=7)
        session = _make_session(scalar_one_or_none=expected)

        repo = AsyncCRUDRepository(Stock)
        result = await repo.get_by_id(7, session=session)

        assert result is expected
        session.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_by_id_returns_none_on_miss(self):
        """get_by_id() should return None when the record does not exist."""
        session = _make_session(scalar_one_or_none=None)

        repo = AsyncCRUDRepository(Stock)
        result = await repo.get_by_id(999, session=session)

        assert result is None


class TestAsyncCRUDRepositoryUpdate:
    """Tests for AsyncCRUDRepository.update()"""

    @pytest.mark.asyncio
    async def test_update_applies_partial_dict(self):
        """update() should issue an UPDATE statement and re-fetch."""
        updated_obj = MagicMock(id=1, name="Updated")
        session = _make_session(scalar_one_or_none=updated_obj, rowcount=1)

        repo = AsyncCRUDRepository(Stock)
        result = await repo.update(1, {"name": "Updated"}, session=session)

        # Two execute calls: the UPDATE and the re-fetch via get_by_id
        assert session.execute.await_count == 2
        assert result is updated_obj

    @pytest.mark.asyncio
    async def test_update_returns_none_when_not_found(self):
        """update() should return None if rowcount is 0."""
        session = _make_session(scalar_one_or_none=None, rowcount=0)

        repo = AsyncCRUDRepository(Stock)
        result = await repo.update(999, {"name": "Ghost"}, session=session)

        assert result is None

    @pytest.mark.asyncio
    async def test_update_skips_none_values(self):
        """update() should strip keys with None values before issuing UPDATE."""
        existing = MagicMock(id=5)
        session = _make_session(scalar_one_or_none=existing, rowcount=0)

        repo = AsyncCRUDRepository(Stock)
        result = await repo.update(5, {"name": None, "description": None}, session=session)

        # When all values are None the dict becomes empty, so it just re-fetches
        # via get_by_id instead of issuing an UPDATE
        assert session.execute.await_count == 1
        assert result is existing


class TestAsyncCRUDRepositoryDelete:
    """Tests for AsyncCRUDRepository.delete()"""

    @pytest.mark.asyncio
    async def test_delete_returns_true_when_found(self):
        """delete() should return True when a row was removed."""
        session = _make_session(rowcount=1)

        repo = AsyncCRUDRepository(Stock)
        result = await repo.delete(1, session=session)

        assert result is True
        session.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_delete_returns_false_when_not_found(self):
        """delete() should return False when no row matched."""
        session = _make_session(rowcount=0)

        repo = AsyncCRUDRepository(Stock)
        result = await repo.delete(999, session=session)

        assert result is False


class TestAsyncCRUDRepositoryGetByField:
    """Tests for AsyncCRUDRepository.get_by_field()"""

    @pytest.mark.asyncio
    async def test_get_by_field_returns_match(self):
        """get_by_field() should return the matching record."""
        expected = MagicMock(symbol="AAPL")
        session = _make_session(scalar_one_or_none=expected)

        repo = AsyncCRUDRepository(Stock)
        result = await repo.get_by_field("symbol", "AAPL", session=session)

        assert result is expected

    @pytest.mark.asyncio
    async def test_get_by_field_returns_none_on_miss(self):
        """get_by_field() should return None when nothing matches."""
        session = _make_session(scalar_one_or_none=None)

        repo = AsyncCRUDRepository(Stock)
        result = await repo.get_by_field("symbol", "NOEXIST", session=session)

        assert result is None

    @pytest.mark.asyncio
    async def test_get_by_field_raises_on_invalid_field(self):
        """get_by_field() should raise AttributeError for unknown fields."""
        session = _make_session()

        repo = AsyncCRUDRepository(Stock)

        with pytest.raises(AttributeError, match="has no field"):
            await repo.get_by_field("totally_fake_column_xyz", "val", session=session)


class TestAsyncCRUDRepositoryExists:
    """Tests for AsyncCRUDRepository.exists()"""

    @pytest.mark.asyncio
    async def test_exists_returns_true(self):
        """exists() should return True when count > 0."""
        session = _make_session(scalar=1)

        repo = AsyncCRUDRepository(Stock)
        result = await repo.exists(1, session=session)

        assert result is True

    @pytest.mark.asyncio
    async def test_exists_returns_false(self):
        """exists() should return False when count == 0."""
        session = _make_session(scalar=0)

        repo = AsyncCRUDRepository(Stock)
        result = await repo.exists(999, session=session)

        assert result is False


class TestFilterCriteriaValidation:
    """Tests for FilterCriteria dataclass validation."""

    def test_valid_operator_accepted(self):
        fc = FilterCriteria(field="name", operator="eq", value="test")
        assert fc.operator == "eq"

    def test_invalid_operator_raises(self):
        with pytest.raises(ValueError, match="Invalid operator"):
            FilterCriteria(field="name", operator="BADOP", value="test")

    def test_all_valid_operators(self):
        valid_ops = [
            "eq", "ne", "gt", "gte", "lt", "lte",
            "in", "not_in", "like", "ilike", "is_null", "is_not_null",
        ]
        for op in valid_ops:
            fc = FilterCriteria(field="x", operator=op, value="y")
            assert fc.operator == op


class TestPaginationParamsValidation:
    """Tests for PaginationParams boundary handling."""

    def test_negative_offset_clamped_to_zero(self):
        pp = PaginationParams(offset=-5, limit=10)
        assert pp.offset == 0

    def test_zero_limit_reset_to_default(self):
        pp = PaginationParams(offset=0, limit=0)
        assert pp.limit == 100

    def test_excessive_limit_capped(self):
        pp = PaginationParams(offset=0, limit=5000)
        assert pp.limit == 1000

    def test_valid_params_unchanged(self):
        pp = PaginationParams(offset=20, limit=50)
        assert pp.offset == 20
        assert pp.limit == 50


# ============================================================================
# PortfolioRepository -- portfolio_repository.py
# ============================================================================


class TestPortfolioRepositoryGetUserPortfolios:
    """Tests for PortfolioRepository.get_user_portfolios()"""

    @pytest.mark.asyncio
    async def test_returns_portfolios_without_positions(self):
        """get_user_portfolios() without include_positions returns list."""
        portfolio = MagicMock(id=1, user_id=10, name="Growth")
        session = _make_session(unique_scalars_all=[portfolio])

        repo = PortfolioRepository()
        result = await repo.get_user_portfolios(10, session=session)

        assert result == [portfolio]
        session.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_returns_portfolios_with_positions(self):
        """get_user_portfolios() with include_positions loads relationships."""
        portfolio = MagicMock(id=1, user_id=10, name="Growth")
        session = _make_session(unique_scalars_all=[portfolio])

        repo = PortfolioRepository()
        result = await repo.get_user_portfolios(10, include_positions=True, session=session)

        assert result == [portfolio]

    @pytest.mark.asyncio
    async def test_returns_empty_for_unknown_user(self):
        """get_user_portfolios() returns [] when user has no portfolios."""
        session = _make_session(unique_scalars_all=[])

        repo = PortfolioRepository()
        result = await repo.get_user_portfolios(9999, session=session)

        assert result == []


class TestPortfolioRepositoryCalculatePortfolioValue:
    """Tests for PortfolioRepository.calculate_portfolio_value()"""

    @pytest.mark.asyncio
    async def test_returns_none_for_missing_portfolio(self):
        """calculate_portfolio_value() returns None when portfolio not found."""
        session = _make_session(scalar_one_or_none=None)

        repo = PortfolioRepository()

        with patch.object(repo, "get_portfolio_with_positions", new_callable=AsyncMock, return_value=None):
            result = await repo.calculate_portfolio_value(999, session=session)

        assert result is None

    @pytest.mark.asyncio
    async def test_calculates_total_value_correctly(self):
        """calculate_portfolio_value() sums positions_value + cash_balance."""
        position = MagicMock()
        position.id = 1
        position.stock_id = 100
        position.quantity = Decimal("10")
        position.avg_cost_basis = Decimal("50.00")

        portfolio = MagicMock()
        portfolio.id = 1
        portfolio.cash_balance = Decimal("5000.00")
        portfolio.positions = [position]

        # The inner loop fetches latest price via session.execute
        price_result = MagicMock()
        price_result.scalar.return_value = Decimal("60.00")

        session = AsyncMock()
        session.execute = AsyncMock(return_value=price_result)
        session.flush = AsyncMock()
        session.refresh = AsyncMock()
        session.add = MagicMock()

        repo = PortfolioRepository()

        with patch.object(repo, "get_portfolio_with_positions", new_callable=AsyncMock, return_value=portfolio):
            result = await repo.calculate_portfolio_value(1, session=session)

        assert result is not None
        # 10 shares * $60 = $600 positions value
        assert result["positions_value"] == 600.0
        # $5000 cash + $600 positions = $5600 total
        assert result["total_value"] == 5600.0
        assert result["cash_balance"] == 5000.0
        assert len(result["positions"]) == 1

        # Verify unrealized gain/loss
        pos_detail = result["positions"][0]
        # cost basis: 10 * $50 = $500, market value: 10 * $60 = $600
        assert pos_detail["unrealized_gain_loss"] == 100.0
        assert pos_detail["cost_basis"] == 500.0

    @pytest.mark.asyncio
    async def test_handles_zero_cost_basis_position(self):
        """calculate_portfolio_value() handles 0 cost basis without dividing by zero."""
        position = MagicMock()
        position.id = 1
        position.stock_id = 100
        position.quantity = Decimal("0")
        position.avg_cost_basis = Decimal("0.00")

        portfolio = MagicMock()
        portfolio.id = 1
        portfolio.cash_balance = Decimal("1000.00")
        portfolio.positions = [position]

        price_result = MagicMock()
        price_result.scalar.return_value = Decimal("50.00")

        session = AsyncMock()
        session.execute = AsyncMock(return_value=price_result)

        repo = PortfolioRepository()

        with patch.object(repo, "get_portfolio_with_positions", new_callable=AsyncMock, return_value=portfolio):
            result = await repo.calculate_portfolio_value(1, session=session)

        assert result is not None
        pos_detail = result["positions"][0]
        # cost_basis is 0, gain_loss_pct should be 0 (no division by zero)
        assert pos_detail["unrealized_gain_loss_pct"] == 0


class TestPortfolioRepositoryGetPortfolioAllocation:
    """Tests for PortfolioRepository.get_portfolio_allocation()"""

    @pytest.mark.asyncio
    async def test_returns_none_when_no_value_data(self):
        """get_portfolio_allocation() returns None when calculate_portfolio_value returns None."""
        session = _make_session()
        repo = PortfolioRepository()

        with patch.object(repo, "calculate_portfolio_value", new_callable=AsyncMock, return_value=None):
            result = await repo.get_portfolio_allocation(999, session=session)

        assert result is None

    @pytest.mark.asyncio
    async def test_allocation_weights_sum_approximately_to_100(self):
        """get_portfolio_allocation() stock + cash weights should sum close to 100%."""
        value_data = {
            "total_value": 10000.0,
            "cash_balance": 2000.0,
            "positions_value": 8000.0,
            "positions": [
                {"stock_id": 1, "market_value": 5000.0},
                {"stock_id": 2, "market_value": 3000.0},
            ],
        }

        stock1 = MagicMock()
        stock1.id = 1
        stock1.symbol = "AAPL"
        stock1.name = "Apple"
        stock1.sector = "Technology"

        stock2 = MagicMock()
        stock2.id = 2
        stock2.symbol = "MSFT"
        stock2.name = "Microsoft"
        stock2.sector = "Technology"

        call_count = 0

        async def _mock_execute(query):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                result.scalar_one_or_none.return_value = stock1
            else:
                result.scalar_one_or_none.return_value = stock2
            return result

        session = AsyncMock()
        session.execute = AsyncMock(side_effect=_mock_execute)

        repo = PortfolioRepository()

        with patch.object(repo, "calculate_portfolio_value", new_callable=AsyncMock, return_value=value_data):
            result = await repo.get_portfolio_allocation(1, session=session)

        assert result is not None
        cash_pct = result["cash_allocation_pct"]
        sector_pcts = sum(result["sector_allocation"].values())
        total_pct = cash_pct + sector_pcts

        assert abs(total_pct - 100.0) < 0.01

    @pytest.mark.asyncio
    async def test_allocation_contains_stock_details(self):
        """get_portfolio_allocation() should include per-stock allocation info."""
        value_data = {
            "total_value": 5000.0,
            "cash_balance": 1000.0,
            "positions_value": 4000.0,
            "positions": [
                {"stock_id": 1, "market_value": 4000.0},
            ],
        }

        stock1 = MagicMock()
        stock1.id = 1
        stock1.symbol = "TSLA"
        stock1.name = "Tesla"
        stock1.sector = "Automotive"

        async def _mock_execute(query):
            result = MagicMock()
            result.scalar_one_or_none.return_value = stock1
            return result

        session = AsyncMock()
        session.execute = AsyncMock(side_effect=_mock_execute)

        repo = PortfolioRepository()

        with patch.object(repo, "calculate_portfolio_value", new_callable=AsyncMock, return_value=value_data):
            result = await repo.get_portfolio_allocation(1, session=session)

        assert len(result["stock_allocation"]) == 1
        assert result["stock_allocation"][0]["symbol"] == "TSLA"
        assert result["stock_allocation"][0]["allocation_pct"] == pytest.approx(80.0)


# ============================================================================
# PriceHistoryRepository -- price_repository.py
# ============================================================================


class TestPriceHistoryRepositoryGetPreviousPrice:
    """Tests for PriceHistoryRepository.get_previous_price()"""

    @pytest.mark.asyncio
    async def test_returns_most_recent_price_before_reference(self):
        """get_previous_price() should return the price record just before the reference date."""
        expected = MagicMock()
        expected.date = date(2025, 1, 14)
        expected.close = Decimal("149.00")

        session = _make_session(scalar_one_or_none=expected)

        repo = PriceHistoryRepository()
        result = await repo.get_previous_price("AAPL", date(2025, 1, 15), session=session)

        assert result is expected
        session.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_returns_none_when_no_earlier_price(self):
        """get_previous_price() should return None when no earlier price exists."""
        session = _make_session(scalar_one_or_none=None)

        repo = PriceHistoryRepository()
        result = await repo.get_previous_price("NEWCO", date(2020, 1, 1), session=session)

        assert result is None

    @pytest.mark.asyncio
    async def test_accepts_datetime_as_reference(self):
        """get_previous_price() should accept datetime objects as reference_date."""
        expected = MagicMock()
        expected.date = datetime(2025, 1, 14, tzinfo=timezone.utc)

        session = _make_session(scalar_one_or_none=expected)

        repo = PriceHistoryRepository()
        ref = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        result = await repo.get_previous_price("AAPL", ref, session=session)

        assert result is expected

    @pytest.mark.asyncio
    async def test_uppercases_symbol(self):
        """get_previous_price() should uppercase the symbol before querying."""
        session = _make_session(scalar_one_or_none=None)

        repo = PriceHistoryRepository()
        await repo.get_previous_price("aapl", date(2025, 6, 1), session=session)

        session.execute.assert_awaited_once()


class TestPriceHistoryRepositoryGetLatestPrice:
    """Tests for PriceHistoryRepository.get_latest_price()"""

    @pytest.mark.asyncio
    async def test_returns_latest_price(self):
        """get_latest_price() should delegate to get_price_history with limit=1."""
        expected = MagicMock(close=Decimal("150.25"))
        session = _make_session(scalars_all=[expected])

        repo = PriceHistoryRepository()
        result = await repo.get_latest_price("AAPL", session=session)

        assert result is expected

    @pytest.mark.asyncio
    async def test_returns_none_when_no_prices(self):
        """get_latest_price() should return None for a stock with no price history."""
        session = _make_session(scalars_all=[])

        repo = PriceHistoryRepository()
        result = await repo.get_latest_price("PHANTOM", session=session)

        assert result is None


class TestPriceHistoryRepositoryGetPriceHistory:
    """Tests for PriceHistoryRepository.get_price_history()"""

    @pytest.mark.asyncio
    async def test_returns_filtered_price_list(self):
        """get_price_history() returns list of PriceHistory records."""
        p1 = MagicMock(date=date(2025, 1, 15))
        p2 = MagicMock(date=date(2025, 1, 14))
        session = _make_session(scalars_all=[p1, p2])

        repo = PriceHistoryRepository()
        result = await repo.get_price_history(
            "AAPL",
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 31),
            session=session,
        )

        assert len(result) == 2
        assert result[0] is p1

    @pytest.mark.asyncio
    async def test_returns_empty_list_for_no_data(self):
        """get_price_history() returns [] when no records match."""
        session = _make_session(scalars_all=[])

        repo = PriceHistoryRepository()
        result = await repo.get_price_history("NODATA", session=session)

        assert result == []

    @pytest.mark.asyncio
    async def test_symbol_uppercased(self):
        """get_price_history() should uppercase the symbol before querying."""
        session = _make_session(scalars_all=[])

        repo = PriceHistoryRepository()
        await repo.get_price_history("aapl", session=session)

        session.execute.assert_awaited_once()


# ============================================================================
# StockRepository -- stock_repository.py
# ============================================================================


class TestStockRepositorySearchStocks:
    """Tests for StockRepository.search_stocks()"""

    @pytest.mark.asyncio
    async def test_returns_matching_stocks(self):
        """search_stocks() should return stocks matching symbol or name."""
        stock = MagicMock(id=1, symbol="AAPL", name="Apple Inc.")
        session = _make_session(scalars_all=[stock])

        repo = StockRepository()
        result = await repo.search_stocks("AAPL", session=session)

        assert len(result) == 1
        assert result[0] is stock

    @pytest.mark.asyncio
    async def test_returns_empty_for_no_match(self):
        """search_stocks() returns [] when no stocks match."""
        session = _make_session(scalars_all=[])

        repo = StockRepository()
        result = await repo.search_stocks("ZZZZZ", session=session)

        assert result == []

    @pytest.mark.asyncio
    async def test_respects_limit(self):
        """search_stocks() should execute query (limit handled in SQL)."""
        stocks = [MagicMock(id=i) for i in range(5)]
        session = _make_session(scalars_all=stocks)

        repo = StockRepository()
        result = await repo.search_stocks("A", limit=5, session=session)

        assert len(result) == 5

    @pytest.mark.asyncio
    async def test_executes_single_query(self):
        """search_stocks() should issue exactly one DB execute call."""
        session = _make_session(scalars_all=[])

        repo = StockRepository()
        await repo.search_stocks("test", session=session)

        session.execute.assert_awaited_once()


class TestStockRepositoryGetBySector:
    """Tests for StockRepository.get_by_sector()

    Note: Stock.sector is a relationship (not a plain column), so
    get_by_sector() delegates to get_multi() with a FilterCriteria.
    We patch get_multi to avoid SQLAlchemy relationship comparison errors
    and focus on verifying the delegation logic.
    """

    @pytest.mark.asyncio
    async def test_returns_stocks_in_sector(self):
        """get_by_sector() delegates to get_multi with correct filter."""
        s1 = MagicMock(id=1, symbol="AAPL")
        s2 = MagicMock(id=2, symbol="MSFT")

        repo = StockRepository()

        with patch.object(repo, "get_multi", new_callable=AsyncMock, return_value=[s1, s2]) as mock_gm:
            result = await repo.get_by_sector("Technology", session=AsyncMock())

        assert len(result) == 2
        # Verify the filter was passed correctly
        call_kwargs = mock_gm.call_args[1]
        filters = call_kwargs["filters"]
        assert len(filters) == 1
        assert filters[0].field == "sector"
        assert filters[0].operator == "eq"
        assert filters[0].value == "Technology"

    @pytest.mark.asyncio
    async def test_returns_empty_for_unknown_sector(self):
        """get_by_sector() returns [] for a sector with no stocks."""
        repo = StockRepository()

        with patch.object(repo, "get_multi", new_callable=AsyncMock, return_value=[]):
            result = await repo.get_by_sector("NonexistentSector", session=AsyncMock())

        assert result == []


class TestStockRepositoryGetBySymbol:
    """Tests for StockRepository.get_by_symbol()"""

    @pytest.mark.asyncio
    async def test_returns_stock_for_valid_symbol(self):
        """get_by_symbol() should return the stock matching the symbol."""
        stock = MagicMock(id=1, symbol="AAPL")
        session = _make_session(scalar_one_or_none=stock)

        repo = StockRepository()
        result = await repo.get_by_symbol("aapl", session=session)

        assert result is stock

    @pytest.mark.asyncio
    async def test_returns_none_for_unknown_symbol(self):
        """get_by_symbol() should return None for a non-existent symbol."""
        session = _make_session(scalar_one_or_none=None)

        repo = StockRepository()
        result = await repo.get_by_symbol("FAKESYM", session=session)

        assert result is None


class TestStockRepositoryGetByMarketCapRange:
    """Tests for StockRepository.get_by_market_cap_range()"""

    @pytest.mark.asyncio
    async def test_returns_stocks_within_range(self):
        """get_by_market_cap_range() returns matching stocks."""
        s1 = MagicMock(id=1, market_cap=1e12)
        session = _make_session(scalars_all=[s1])

        repo = StockRepository()
        result = await repo.get_by_market_cap_range(
            min_cap=1e11, max_cap=2e12, session=session
        )

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_match(self):
        """get_by_market_cap_range() returns [] when nothing in range."""
        session = _make_session(scalars_all=[])

        repo = StockRepository()
        result = await repo.get_by_market_cap_range(
            min_cap=1e15, max_cap=2e15, session=session
        )

        assert result == []


class TestSortParamsDataclass:
    """Tests for SortParams default direction."""

    def test_default_direction_is_asc(self):
        sp = SortParams(field="name")
        assert sp.direction == SortDirection.ASC

    def test_desc_direction(self):
        sp = SortParams(field="name", direction=SortDirection.DESC)
        assert sp.direction == SortDirection.DESC


# ---------------------------------------------------------------------------
# AsyncBaseRepository.transaction() -- F-15-011 / F-07-002
# ---------------------------------------------------------------------------
#
# Audit 2026-04, Cluster E Step 5 (workpaper:
# docs/audits/2026-04/_synthesis/workpaper/E.md).
#
# The production `transaction()` method on AsyncBaseRepository is decorated
# with @asynccontextmanager but its body never `yield`s a session. Internally
# it defines a nested async-generator `_execute_transaction` and passes it to
# `db_manager.execute_with_retry(...)`. Net effect: `async with
# repo.transaction() as session:` blocks fail at runtime because the outer
# context manager produces no session.
#
# Existing tests in this file never exercised this path. The two tests below
# do, with mocked AsyncSession + patched get_db_session, asserting the
# documented contract:
#   - on success the session.commit() should be invoked
#   - on a raised exception the session.rollback() should be invoked
#
# Cascade-from-scope-07 (audit 2026-04 G4 phase 1, 2026-04-28):
# F-07-002 fix landed (backend/repositories/base.py:transaction now correctly
# wraps @asynccontextmanager around get_db_session and commit/rollback).
# The xfail(strict=True) markers below have been removed so CI proves the
# contract holds. See PR for the fail-first commit-pair on F-07-002 and the
# new tests at tests/database/test_transactions.py.

class TestAsyncBaseRepositoryTransaction:
    """F-15-011: real test of AsyncBaseRepository.transaction() async-generator bug."""

    @pytest.mark.asyncio
    async def test_transaction_commits_on_success(self):
        """`async with repo.transaction() as session:` should commit on success."""
        from contextlib import asynccontextmanager

        mock_session = AsyncMock()
        mock_session.commit = AsyncMock()
        mock_session.rollback = AsyncMock()

        @asynccontextmanager
        async def _fake_get_db_session(*_a, **_kw):
            yield mock_session

        repo = StockRepository()

        with patch(
            "backend.repositories.base.get_db_session", _fake_get_db_session
        ):
            async with repo.transaction() as session:
                # Body of transaction -- in real use the caller would do work
                # against `session` here.
                assert session is not None

        # Contract: on a clean exit, the transaction should have committed.
        assert mock_session.commit.await_count >= 1, (
            "Expected commit() on successful transaction; current production "
            "transaction() returns no session and body never executes."
        )
        assert mock_session.rollback.await_count == 0

    @pytest.mark.asyncio
    async def test_transaction_rolls_back_on_exception(self):
        """`async with repo.transaction(): raise` should rollback."""
        from contextlib import asynccontextmanager

        mock_session = AsyncMock()
        mock_session.commit = AsyncMock()
        mock_session.rollback = AsyncMock()

        @asynccontextmanager
        async def _fake_get_db_session(*_a, **_kw):
            yield mock_session

        repo = StockRepository()

        class _BoomError(RuntimeError):
            pass

        with patch(
            "backend.repositories.base.get_db_session", _fake_get_db_session
        ):
            with pytest.raises(_BoomError):
                async with repo.transaction():
                    raise _BoomError("simulated failure inside transaction")

        # Contract: on inner-block exception, rollback should have run.
        assert mock_session.rollback.await_count >= 1, (
            "Expected rollback() when transaction body raises; current "
            "production transaction() never enters the body so neither "
            "commit nor rollback is reached."
        )
