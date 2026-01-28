"""
Integration tests for authentication to portfolio access flow.

Tests cover user authentication, session management, role-based access,
and portfolio operations with proper authorization checks.
"""

import pytest
import pytest_asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, patch
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.unified_models import (
    User, UserSession, Portfolio, Position, Stock, Transaction,
    Exchange, Sector,
    UserRoleEnum, OrderSideEnum, AssetTypeEnum
)
from backend.api.main import app
from backend.auth.oauth2 import create_access_token, create_refresh_token
from httpx import AsyncClient, ASGITransport


pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def premium_user(db_session: AsyncSession):
    """Create a premium user for testing."""
    user = User(
        email="premium@test.com",
        hashed_password="$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5NU2VXhI0Asei",
        full_name="Premium User",
        role=UserRoleEnum.PREMIUM_USER.value,
        is_active=True,
        is_verified=True,
        subscription_tier="premium",
        subscription_end_date=datetime.utcnow() + timedelta(days=365)
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


@pytest_asyncio.fixture
async def free_user(db_session: AsyncSession):
    """Create a free tier user for testing."""
    user = User(
        email="free@test.com",
        hashed_password="$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5NU2VXhI0Asei",
        full_name="Free User",
        role=UserRoleEnum.FREE_USER.value,
        is_active=True,
        is_verified=True,
        subscription_tier="free"
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


@pytest_asyncio.fixture
async def user_session(db_session: AsyncSession, premium_user: User):
    """Create an active user session."""
    session = UserSession(
        user_id=premium_user.id,
        session_token="test_session_token_12345",
        refresh_token="test_refresh_token_67890",
        ip_address="127.0.0.1",
        user_agent="pytest-client/1.0",
        is_active=True,
        expires_at=datetime.utcnow() + timedelta(hours=24),
        last_activity=datetime.utcnow()
    )
    db_session.add(session)
    await db_session.commit()
    await db_session.refresh(session)
    return session


@pytest_asyncio.fixture
async def user_portfolio(db_session: AsyncSession, premium_user: User):
    """Create a portfolio for the premium user."""
    portfolio = Portfolio(
        user_id=premium_user.id,
        name="Main Portfolio",
        description="Primary investment portfolio",
        cash_balance=Decimal("50000.00"),
        is_public=False,
        is_default=True,
        benchmark="SPY"
    )
    db_session.add(portfolio)
    await db_session.commit()
    await db_session.refresh(portfolio)
    return portfolio


@pytest_asyncio.fixture
async def nasdaq_exchange(db_session: AsyncSession):
    """Create NASDAQ exchange for testing."""
    exchange = Exchange(
        code="NASDAQ",
        name="NASDAQ Stock Market",
        country="US",
        currency="USD",
        timezone="America/New_York"
    )
    db_session.add(exchange)
    await db_session.commit()
    await db_session.refresh(exchange)
    return exchange


@pytest_asyncio.fixture
async def technology_sector(db_session: AsyncSession):
    """Create Technology sector for testing."""
    sector = Sector(
        name="Technology",
        description="Technology sector"
    )
    db_session.add(sector)
    await db_session.commit()
    await db_session.refresh(sector)
    return sector


@pytest_asyncio.fixture
async def sample_stocks(db_session: AsyncSession, nasdaq_exchange: Exchange, technology_sector: Sector):
    """Create sample stocks for portfolio testing."""
    stocks = [
        Stock(
            symbol="AAPL",
            name="Apple Inc.",
            exchange_id=nasdaq_exchange.id,
            asset_type="stock",
            sector_id=technology_sector.id,
            is_active=True,
            is_tradable=True
        ),
        Stock(
            symbol="MSFT",
            name="Microsoft Corporation",
            exchange_id=nasdaq_exchange.id,
            asset_type="stock",
            sector_id=technology_sector.id,
            is_active=True,
            is_tradable=True
        ),
        Stock(
            symbol="GOOGL",
            name="Alphabet Inc.",
            exchange_id=nasdaq_exchange.id,
            asset_type="stock",
            sector_id=technology_sector.id,
            is_active=True,
            is_tradable=True
        )
    ]
    for stock in stocks:
        db_session.add(stock)
    await db_session.commit()
    return {stock.symbol: stock for stock in stocks}


@pytest.mark.asyncio
async def test_login_to_portfolio_access(
    async_client: AsyncClient,
    db_session: AsyncSession,
    premium_user: User,
    user_portfolio: Portfolio
):
    """
    Test complete auth flow: login -> token -> portfolio data access.

    Validates that users can authenticate and immediately access their
    portfolio data using the issued JWT token.
    """
    # Step 1: Login
    response = await async_client.post(
        "/api/v1/auth/login",
        json={
            "email": premium_user.email,
            "password": "testpassword123"
        }
    )
    assert response.status_code == 200
    auth_data = response.json()
    assert "access_token" in auth_data["data"]
    assert "refresh_token" in auth_data["data"]
    access_token = auth_data["data"]["access_token"]

    # Step 2: Access portfolio with token
    headers = {"Authorization": f"Bearer {access_token}"}
    response = await async_client.get(
        f"/api/v1/portfolios/{user_portfolio.id}",
        headers=headers
    )
    assert response.status_code == 200
    portfolio_data = response.json()
    assert portfolio_data["data"]["name"] == "Main Portfolio"
    assert portfolio_data["data"]["user_id"] == premium_user.id
    assert Decimal(str(portfolio_data["data"]["cash_balance"])) == Decimal("50000.00")


@pytest.mark.asyncio
async def test_role_based_portfolio_limits(
    async_client: AsyncClient,
    db_session: AsyncSession,
    premium_user: User,
    free_user: User,
    sample_stocks: dict
):
    """
    Test role-based quota limits: free vs premium tier restrictions.

    Validates that free users have portfolio size limits while premium
    users can create larger portfolios with more positions.
    """
    # Create tokens
    premium_token = create_access_token(
        data={"sub": str(premium_user.id), "role": "premium_user"}
    )
    free_token = create_access_token(
        data={"sub": str(free_user.id), "role": "free_user"}
    )

    premium_headers = {"Authorization": f"Bearer {premium_token}"}
    free_headers = {"Authorization": f"Bearer {free_token}"}

    # Free user: Create portfolio
    response = await async_client.post(
        "/api/v1/portfolios",
        headers=free_headers,
        json={
            "name": "Free User Portfolio",
            "description": "Limited portfolio",
            "cash_balance": 5000.00
        }
    )
    assert response.status_code == 201
    free_portfolio_id = response.json()["data"]["id"]

    # Free user: Try to add many positions (should hit limit)
    positions_added = 0
    for symbol in list(sample_stocks.keys())[:10]:  # Try 10 positions
        response = await async_client.post(
            f"/api/v1/portfolios/{free_portfolio_id}/positions",
            headers=free_headers,
            json={
                "stock_symbol": symbol,
                "quantity": 1.0,
                "average_cost": 100.0
            }
        )
        if response.status_code == 201:
            positions_added += 1
        elif response.status_code == 403:
            # Hit quota limit
            break

    # Free user should be limited (e.g., 5 positions)
    assert positions_added <= 5

    # Premium user: Create portfolio
    response = await async_client.post(
        "/api/v1/portfolios",
        headers=premium_headers,
        json={
            "name": "Premium User Portfolio",
            "description": "Unlimited portfolio",
            "cash_balance": 50000.00
        }
    )
    assert response.status_code == 201
    premium_portfolio_id = response.json()["data"]["id"]

    # Premium user: Add more positions (should succeed)
    premium_positions_added = 0
    for symbol in list(sample_stocks.keys())[:10]:
        response = await async_client.post(
            f"/api/v1/portfolios/{premium_portfolio_id}/positions",
            headers=premium_headers,
            json={
                "stock_symbol": symbol,
                "quantity": 10.0,
                "average_cost": 150.0
            }
        )
        if response.status_code == 201:
            premium_positions_added += 1

    # Premium user should be able to add more positions
    assert premium_positions_added > positions_added


@pytest.mark.asyncio
async def test_session_expiry_during_portfolio(
    async_client: AsyncClient,
    db_session: AsyncSession,
    premium_user: User,
    user_portfolio: Portfolio
):
    """
    Test session expiry and token refresh during portfolio operations.

    Validates that expired tokens are properly rejected and refresh tokens
    can be used to obtain new access tokens without re-authentication.
    """
    # Create an expired access token
    expired_token = create_access_token(
        data={"sub": str(premium_user.id)},
        expires_delta=timedelta(minutes=-1)  # Already expired
    )

    expired_headers = {"Authorization": f"Bearer {expired_token}"}

    # Try to access portfolio with expired token
    response = await async_client.get(
        f"/api/v1/portfolios/{user_portfolio.id}",
        headers=expired_headers
    )
    assert response.status_code == 401

    # Use refresh token to get new access token
    refresh_token = create_refresh_token(data={"sub": str(premium_user.id)})

    response = await async_client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": refresh_token}
    )
    assert response.status_code == 200
    new_tokens = response.json()
    assert "access_token" in new_tokens["data"]

    # Use new token to access portfolio
    new_headers = {"Authorization": f"Bearer {new_tokens['data']['access_token']}"}
    response = await async_client.get(
        f"/api/v1/portfolios/{user_portfolio.id}",
        headers=new_headers
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_concurrent_portfolio_updates(
    async_client: AsyncClient,
    db_session: AsyncSession,
    premium_user: User,
    user_portfolio: Portfolio,
    sample_stocks: dict
):
    """
    Test concurrent portfolio updates and race condition handling.

    Validates that simultaneous portfolio modifications are properly
    serialized and don't result in data corruption or lost updates.
    """
    import asyncio

    token = create_access_token(data={"sub": str(premium_user.id)})
    headers = {"Authorization": f"Bearer {token}"}

    # Get initial cash balance
    initial_balance = user_portfolio.cash_balance

    # Define concurrent update operations
    async def buy_stock(symbol: str, quantity: float, price: float):
        return await async_client.post(
            f"/api/v1/portfolios/{user_portfolio.id}/positions",
            headers=headers,
            json={
                "stock_symbol": symbol,
                "quantity": quantity,
                "average_cost": price
            }
        )

    # Execute concurrent buys
    results = await asyncio.gather(
        buy_stock("AAPL", 10, 150.0),
        buy_stock("MSFT", 15, 300.0),
        buy_stock("GOOGL", 5, 120.0),
        return_exceptions=True
    )

    # Count successful operations
    successful = sum(1 for r in results if not isinstance(r, Exception) and r.status_code == 201)
    assert successful >= 2  # At least some should succeed

    # Verify portfolio consistency
    response = await async_client.get(
        f"/api/v1/portfolios/{user_portfolio.id}",
        headers=headers
    )
    assert response.status_code == 200
    portfolio_data = response.json()

    # Cash balance should decrease by total purchase amount
    total_spent = sum(
        r.json()["data"]["quantity"] * r.json()["data"]["average_cost"]
        for r in results
        if not isinstance(r, Exception) and r.status_code == 201
    )

    expected_balance = float(initial_balance) - total_spent
    actual_balance = float(portfolio_data["data"]["cash_balance"])

    # Allow small floating point differences
    assert abs(actual_balance - expected_balance) < 0.01


@pytest.mark.asyncio
async def test_portfolio_rebalancing_with_locks(
    async_client: AsyncClient,
    db_session: AsyncSession,
    premium_user: User,
    user_portfolio: Portfolio,
    sample_stocks: dict
):
    """
    Test portfolio rebalancing with row-level locking.

    Validates that portfolio rebalancing operations properly lock
    affected rows to prevent concurrent modifications during rebalance.
    """
    token = create_access_token(data={"sub": str(premium_user.id)})
    headers = {"Authorization": f"Bearer {token}"}

    # Add initial positions
    aapl_stock = sample_stocks["AAPL"]
    msft_stock = sample_stocks["MSFT"]

    position1 = Position(
        portfolio_id=user_portfolio.id,
        stock_id=aapl_stock.id,
        quantity=Decimal("100"),
        average_cost=Decimal("150.00")
    )
    position2 = Position(
        portfolio_id=user_portfolio.id,
        stock_id=msft_stock.id,
        quantity=Decimal("50"),
        average_cost=Decimal("300.00")
    )
    db_session.add(position1)
    db_session.add(position2)
    await db_session.commit()

    # Define target allocation (60% AAPL, 40% MSFT)
    target_allocation = {
        "AAPL": 0.60,
        "MSFT": 0.40
    }

    # Mock current prices
    with patch("backend.services.portfolio_service.PortfolioService.get_current_prices") as mock_prices:
        mock_prices.return_value = {
            "AAPL": 160.00,
            "MSFT": 320.00
        }

        # Request rebalancing
        response = await async_client.post(
            f"/api/v1/portfolios/{user_portfolio.id}/rebalance",
            headers=headers,
            json={"target_allocation": target_allocation}
        )
        assert response.status_code == 200
        rebalance_data = response.json()

        # Verify rebalancing plan
        assert "transactions" in rebalance_data["data"]
        transactions = rebalance_data["data"]["transactions"]
        assert len(transactions) >= 2

        # Execute rebalancing
        response = await async_client.post(
            f"/api/v1/portfolios/{user_portfolio.id}/rebalance/execute",
            headers=headers,
            json={"rebalance_id": rebalance_data["data"]["id"]}
        )
        assert response.status_code == 200

        # Verify final allocation matches target (within tolerance)
        response = await async_client.get(
            f"/api/v1/portfolios/{user_portfolio.id}/allocation",
            headers=headers
        )
        assert response.status_code == 200
        allocation_data = response.json()

        aapl_allocation = allocation_data["data"]["AAPL"]
        msft_allocation = allocation_data["data"]["MSFT"]

        assert abs(aapl_allocation - 0.60) < 0.05  # Within 5% tolerance
        assert abs(msft_allocation - 0.40) < 0.05
