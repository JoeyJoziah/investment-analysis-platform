"""
Row Locking Security Tests
Tests optimistic and pessimistic locking to prevent concurrent update issues.
"""

import pytest
import pytest_asyncio
import asyncio
from decimal import Decimal
from datetime import datetime
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.unified_models import Portfolio, Position, Stock, User
from backend.models.thesis import InvestmentThesis
from backend.repositories.portfolio_repository import PortfolioRepository
from backend.repositories.thesis_repository import InvestmentThesisRepository
from backend.exceptions import StaleDataError, InsufficientBalanceError, InvalidPositionError
from backend.config.database import get_db_session


@pytest_asyncio.fixture
async def test_user(db_session: AsyncSession) -> User:
    """Create a test user"""
    user = User(
        email="test@example.com",
        username="testuser",
        hashed_password="hashed_password",
        full_name="Test User"
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


@pytest_asyncio.fixture
async def test_stock(db_session: AsyncSession) -> Stock:
    """Create a test stock"""
    from backend.models.unified_models import Exchange, Sector

    # Create exchange
    exchange = Exchange(code="NYSE", name="New York Stock Exchange", timezone="America/New_York")
    db_session.add(exchange)
    await db_session.flush()

    # Create sector
    sector = Sector(name="Technology", description="Tech sector")
    db_session.add(sector)
    await db_session.flush()

    # Create stock
    stock = Stock(
        symbol="TEST",
        name="Test Stock",
        exchange_id=exchange.id,
        sector_id=sector.id
    )
    db_session.add(stock)
    await db_session.commit()
    await db_session.refresh(stock)
    return stock


@pytest_asyncio.fixture
async def test_portfolio(db_session: AsyncSession, test_user: User) -> Portfolio:
    """Create a test portfolio"""
    portfolio = Portfolio(
        user_id=test_user.id,
        name="Test Portfolio",
        cash_balance=Decimal("10000.00"),
        version=1
    )
    db_session.add(portfolio)
    await db_session.commit()
    await db_session.refresh(portfolio)
    return portfolio


@pytest_asyncio.fixture
async def test_thesis(db_session: AsyncSession, test_user: User, test_stock: Stock) -> InvestmentThesis:
    """Create a test investment thesis"""
    thesis = InvestmentThesis(
        user_id=test_user.id,
        stock_id=test_stock.id,
        investment_objective="Growth",
        time_horizon="long-term",
        target_price=Decimal("150.00"),
        version=1
    )
    db_session.add(thesis)
    await db_session.commit()
    await db_session.refresh(thesis)
    return thesis


class TestPortfolioRowLocking:
    """Test row locking for Portfolio operations"""

    @pytest.mark.asyncio
    async def test_optimistic_lock_prevents_lost_update(
        self,
        db_session: AsyncSession,
        test_portfolio: Portfolio
    ):
        """Test that optimistic locking prevents lost updates"""
        repo = PortfolioRepository()

        # Read portfolio version
        portfolio1 = await repo.get_by_id(test_portfolio.id)
        portfolio2 = await repo.get_by_id(test_portfolio.id)

        assert portfolio1.version == portfolio2.version == 1

        # Update portfolio1 - should succeed
        updated1 = await repo.update_with_lock(
            id=portfolio1.id,
            data={"name": "Updated Portfolio 1"},
            expected_version=1
        )
        assert updated1.version == 2
        assert updated1.name == "Updated Portfolio 1"

        # Try to update portfolio2 with stale version - should fail
        with pytest.raises(StaleDataError) as exc_info:
            await repo.update_with_lock(
                id=portfolio2.id,
                data={"name": "Updated Portfolio 2"},
                expected_version=1
            )

        assert exc_info.value.entity_type == "Portfolio"
        assert exc_info.value.expected_version == 1
        assert exc_info.value.current_version == 2

    @pytest.mark.asyncio
    async def test_concurrent_balance_updates_with_lock(
        self,
        test_portfolio: Portfolio,
        test_stock: Stock
    ):
        """Test concurrent balance updates use row locking"""
        repo = PortfolioRepository()

        async def buy_stock(quantity: int):
            """Simulate buying stock"""
            try:
                await repo.add_position(
                    portfolio_id=test_portfolio.id,
                    stock_id=test_stock.id,
                    quantity=Decimal(quantity),
                    price=Decimal("100.00"),
                    transaction_type='buy'
                )
                return True
            except InsufficientBalanceError:
                return False

        # Start with $10,000, try to buy 50 + 50 = 100 shares ($10,000 total)
        # Both should succeed as they're serialized by the lock
        results = await asyncio.gather(
            buy_stock(50),
            buy_stock(50),
            return_exceptions=True
        )

        # At least one should succeed
        success_count = sum(1 for r in results if r is True)
        assert success_count >= 1

        # Check final balance is correct
        final_portfolio = await repo.get_by_id(test_portfolio.id)
        assert final_portfolio.cash_balance >= 0  # Should never be negative

    @pytest.mark.asyncio
    async def test_insufficient_balance_error(
        self,
        test_portfolio: Portfolio,
        test_stock: Stock
    ):
        """Test that insufficient balance is properly detected"""
        repo = PortfolioRepository()

        # Try to buy more than we have cash for
        with pytest.raises(InsufficientBalanceError) as exc_info:
            await repo.add_position(
                portfolio_id=test_portfolio.id,
                stock_id=test_stock.id,
                quantity=Decimal("200"),
                price=Decimal("100.00"),
                transaction_type='buy'
            )

        assert "$10000" in str(exc_info.value)
        assert "$20000" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_invalid_position_sell_error(
        self,
        test_portfolio: Portfolio,
        test_stock: Stock
    ):
        """Test that selling more than owned is prevented"""
        repo = PortfolioRepository()

        # Try to sell without owning any
        with pytest.raises(InvalidPositionError) as exc_info:
            await repo.add_position(
                portfolio_id=test_portfolio.id,
                stock_id=test_stock.id,
                quantity=Decimal("10"),
                price=Decimal("100.00"),
                transaction_type='sell'
            )

        assert "only 0 owned" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_position_version_increments(
        self,
        test_portfolio: Portfolio,
        test_stock: Stock
    ):
        """Test that position version increments on each update"""
        repo = PortfolioRepository()

        # Buy initial position
        await repo.add_position(
            portfolio_id=test_portfolio.id,
            stock_id=test_stock.id,
            quantity=Decimal("10"),
            price=Decimal("100.00"),
            transaction_type='buy'
        )

        async with get_db_session() as session:
            position_query = select(Position).where(
                Position.portfolio_id == test_portfolio.id,
                Position.stock_id == test_stock.id
            )
            result = await session.execute(position_query)
            position = result.scalar_one()

            assert position.version == 1

        # Buy more - version should increment
        await repo.add_position(
            portfolio_id=test_portfolio.id,
            stock_id=test_stock.id,
            quantity=Decimal("5"),
            price=Decimal("100.00"),
            transaction_type='buy'
        )

        async with get_db_session() as session:
            position_query = select(Position).where(
                Position.portfolio_id == test_portfolio.id,
                Position.stock_id == test_stock.id
            )
            result = await session.execute(position_query)
            position = result.scalar_one()

            assert position.version == 2

    @pytest.mark.asyncio
    async def test_portfolio_version_increments_on_trade(
        self,
        test_portfolio: Portfolio,
        test_stock: Stock
    ):
        """Test that portfolio version increments on each trade"""
        repo = PortfolioRepository()

        initial_version = test_portfolio.version

        # Execute trade
        await repo.add_position(
            portfolio_id=test_portfolio.id,
            stock_id=test_stock.id,
            quantity=Decimal("10"),
            price=Decimal("100.00"),
            transaction_type='buy'
        )

        updated_portfolio = await repo.get_by_id(test_portfolio.id)
        assert updated_portfolio.version == initial_version + 1

    @pytest.mark.asyncio
    async def test_get_with_lock_nowait(self, test_portfolio: Portfolio):
        """Test get_with_lock with nowait option"""
        repo = PortfolioRepository()

        async with get_db_session() as session:
            # Lock the portfolio
            locked = await repo.get_with_lock(
                test_portfolio.id,
                for_update=True,
                session=session
            )
            assert locked.id == test_portfolio.id

            # Try to lock again with nowait - should raise exception
            from sqlalchemy.exc import OperationalError
            with pytest.raises(OperationalError):
                await repo.get_with_lock(
                    test_portfolio.id,
                    for_update=True,
                    nowait=True
                )

    @pytest.mark.asyncio
    async def test_get_with_lock_skip_locked(self, test_portfolio: Portfolio):
        """Test get_with_lock with skip_locked option"""
        repo = PortfolioRepository()

        async with get_db_session() as session:
            # Lock the portfolio
            locked = await repo.get_with_lock(
                test_portfolio.id,
                for_update=True,
                session=session
            )
            assert locked.id == test_portfolio.id

            # Try to lock again with skip_locked - should return None
            result = await repo.get_with_lock(
                test_portfolio.id,
                for_update=True,
                skip_locked=True
            )
            assert result is None


class TestThesisRowLocking:
    """Test row locking for InvestmentThesis operations"""

    @pytest.mark.asyncio
    async def test_thesis_optimistic_lock(
        self,
        test_thesis: InvestmentThesis,
        test_user: User
    ):
        """Test optimistic locking for thesis updates"""
        repo = InvestmentThesisRepository()

        # Read thesis twice
        thesis1 = await repo.get_by_id(test_thesis.id)
        thesis2 = await repo.get_by_id(test_thesis.id)

        assert thesis1.version == thesis2.version == 1

        # Update thesis1 - should succeed
        updated1 = await repo.update_thesis(
            thesis_id=thesis1.id,
            user_id=test_user.id,
            data={"investment_objective": "Value"},
            expected_version=1
        )
        assert updated1.version == 2
        assert updated1.investment_objective == "Value"

        # Try to update thesis2 with stale version - should fail
        with pytest.raises(StaleDataError) as exc_info:
            await repo.update_thesis(
                thesis_id=thesis2.id,
                user_id=test_user.id,
                data={"investment_objective": "Momentum"},
                expected_version=1
            )

        assert exc_info.value.entity_type == "InvestmentThesis"
        assert exc_info.value.expected_version == 1
        assert exc_info.value.current_version == 2

    @pytest.mark.asyncio
    async def test_thesis_version_increments(
        self,
        test_thesis: InvestmentThesis,
        test_user: User
    ):
        """Test that thesis version increments on each update"""
        repo = InvestmentThesisRepository()

        initial_version = test_thesis.version

        # Update 1
        updated = await repo.update_thesis(
            thesis_id=test_thesis.id,
            user_id=test_user.id,
            data={"investment_objective": "Value"}
        )
        assert updated.version == initial_version + 1

        # Update 2
        updated = await repo.update_thesis(
            thesis_id=test_thesis.id,
            user_id=test_user.id,
            data={"investment_objective": "Growth"}
        )
        assert updated.version == initial_version + 2

    @pytest.mark.asyncio
    async def test_concurrent_thesis_updates(
        self,
        test_thesis: InvestmentThesis,
        test_user: User
    ):
        """Test concurrent thesis updates are serialized"""
        repo = InvestmentThesisRepository()

        async def update_thesis(objective: str):
            """Update thesis objective"""
            try:
                # Get current version
                thesis = await repo.get_by_id(test_thesis.id)
                await asyncio.sleep(0.01)  # Small delay to encourage conflicts

                updated = await repo.update_thesis(
                    thesis_id=test_thesis.id,
                    user_id=test_user.id,
                    data={"investment_objective": objective},
                    expected_version=thesis.version
                )
                return updated.investment_objective
            except StaleDataError:
                return None

        # Try concurrent updates
        results = await asyncio.gather(
            update_thesis("Value"),
            update_thesis("Growth"),
            update_thesis("Momentum"),
            return_exceptions=True
        )

        # At least one should succeed
        successful_updates = [r for r in results if r is not None]
        assert len(successful_updates) >= 1

        # Final version should reflect number of successful updates
        final_thesis = await repo.get_by_id(test_thesis.id)
        expected_version = 1 + len(successful_updates)
        assert final_thesis.version == expected_version


class TestPositionRowLocking:
    """Test row locking for Position operations"""

    @pytest.mark.asyncio
    async def test_position_concurrent_buy_sell(
        self,
        test_portfolio: Portfolio,
        test_stock: Stock
    ):
        """Test concurrent buy and sell operations"""
        repo = PortfolioRepository()

        # Buy initial position
        await repo.add_position(
            portfolio_id=test_portfolio.id,
            stock_id=test_stock.id,
            quantity=Decimal("20"),
            price=Decimal("100.00"),
            transaction_type='buy'
        )

        async def sell_shares(quantity: int):
            """Sell shares"""
            try:
                await repo.add_position(
                    portfolio_id=test_portfolio.id,
                    stock_id=test_stock.id,
                    quantity=Decimal(quantity),
                    price=Decimal("100.00"),
                    transaction_type='sell'
                )
                return True
            except InvalidPositionError:
                return False

        # Try to sell 10 + 10 = 20 shares concurrently
        results = await asyncio.gather(
            sell_shares(10),
            sell_shares(10),
            return_exceptions=True
        )

        # Both should succeed as we own 20 shares
        success_count = sum(1 for r in results if r is True)
        assert success_count == 2

        # Position should be deleted (quantity = 0)
        async with get_db_session() as session:
            position_query = select(Position).where(
                Position.portfolio_id == test_portfolio.id,
                Position.stock_id == test_stock.id
            )
            result = await session.execute(position_query)
            position = result.scalar_one_or_none()
            assert position is None

    @pytest.mark.asyncio
    async def test_position_oversell_prevention(
        self,
        test_portfolio: Portfolio,
        test_stock: Stock
    ):
        """Test that overselling is prevented with row locking"""
        repo = PortfolioRepository()

        # Buy initial position
        await repo.add_position(
            portfolio_id=test_portfolio.id,
            stock_id=test_stock.id,
            quantity=Decimal("10"),
            price=Decimal("100.00"),
            transaction_type='buy'
        )

        async def sell_shares(quantity: int):
            """Sell shares"""
            try:
                await repo.add_position(
                    portfolio_id=test_portfolio.id,
                    stock_id=test_stock.id,
                    quantity=Decimal(quantity),
                    price=Decimal("100.00"),
                    transaction_type='sell'
                )
                return "success"
            except InvalidPositionError:
                return "invalid_position"

        # Try to sell 10 + 10 = 20 shares (but we only have 10)
        results = await asyncio.gather(
            sell_shares(10),
            sell_shares(10),
            return_exceptions=True
        )

        # One should succeed, one should fail
        success_count = sum(1 for r in results if r == "success")
        failure_count = sum(1 for r in results if r == "invalid_position")

        assert success_count == 1
        assert failure_count == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
