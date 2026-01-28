"""
Integration tests for stock lookup to analysis recommendation flow.

Tests cover the complete pipeline from stock data retrieval through to
final investment recommendations, including caching, real-time data, and thesis generation.
"""

import pytest
import pytest_asyncio
from datetime import datetime, timedelta, date
from decimal import Decimal
from unittest.mock import AsyncMock, patch, MagicMock
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.tables import (
    Stock, PriceHistory, Recommendation, Fundamental,
    Alert, Portfolio, Position, RecommendationTypeEnum, AssetTypeEnum
)
from backend.api.main import app
from httpx import AsyncClient, ASGITransport


pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def sample_stock(db_session: AsyncSession):
    """Create a sample stock for testing."""
    stock = Stock(
        symbol="AAPL",
        name="Apple Inc.",
        exchange="NASDAQ",
        asset_type=AssetTypeEnum.STOCK,
        sector="Technology",
        industry="Consumer Electronics",
        market_cap=3000000000000,
        shares_outstanding=16000000000,
        country="US",
        currency="USD",
        is_active=True,
        is_tradable=True
    )
    db_session.add(stock)
    await db_session.commit()
    await db_session.refresh(stock)
    return stock


@pytest_asyncio.fixture
async def sample_price_history(db_session: AsyncSession, sample_stock: Stock):
    """Create price history for the sample stock."""
    prices = []
    base_date = date.today() - timedelta(days=30)

    for i in range(30):
        price = PriceHistory(
            stock_id=sample_stock.id,
            date=base_date + timedelta(days=i),
            open=Decimal("150.00") + Decimal(str(i * 0.5)),
            high=Decimal("152.00") + Decimal(str(i * 0.5)),
            low=Decimal("149.00") + Decimal(str(i * 0.5)),
            close=Decimal("151.00") + Decimal(str(i * 0.5)),
            adjusted_close=Decimal("151.00") + Decimal(str(i * 0.5)),
            volume=75000000 + (i * 1000000)
        )
        prices.append(price)
        db_session.add(price)

    await db_session.commit()
    return prices


@pytest_asyncio.fixture
async def sample_fundamentals(db_session: AsyncSession, sample_stock: Stock):
    """Create fundamental data for the sample stock."""
    fundamental = Fundamental(
        stock_id=sample_stock.id,
        report_date=date.today() - timedelta(days=90),
        period="Q4",
        revenue=90000000000,
        gross_profit=40000000000,
        operating_income=25000000000,
        net_income=22000000000,
        eps=Decimal("5.50"),
        eps_diluted=Decimal("5.45"),
        total_assets=350000000000,
        total_liabilities=280000000000,
        total_equity=70000000000,
        cash=50000000000,
        debt=120000000000,
        free_cash_flow=28000000000,
        pe_ratio=28.5,
        peg_ratio=1.8,
        ps_ratio=7.2,
        pb_ratio=40.0,
        dividend_yield=0.5,
        roe=0.35,
        roa=0.22,
        roic=0.28,
        gross_margin=0.44,
        operating_margin=0.28,
        net_margin=0.24
    )
    db_session.add(fundamental)
    await db_session.commit()
    await db_session.refresh(fundamental)
    return fundamental


@pytest.mark.asyncio
async def test_stock_lookup_to_recommendation(
    async_client: AsyncClient,
    db_session: AsyncSession,
    sample_stock: Stock,
    sample_price_history,
    sample_fundamentals,
    auth_headers
):
    """
    Test complete pipeline: stock lookup -> data retrieval -> analysis -> recommendation.

    Validates that stock data flows correctly through the system to generate
    investment recommendations based on technical and fundamental analysis.
    """
    # Step 1: Lookup stock by symbol
    response = await async_client.get(
        f"/api/v1/stocks/{sample_stock.symbol}",
        headers=auth_headers
    )
    assert response.status_code == 200
    stock_data = response.json()
    assert stock_data["data"]["symbol"] == "AAPL"
    assert stock_data["data"]["name"] == "Apple Inc."

    # Step 2: Fetch price history
    response = await async_client.get(
        f"/api/v1/stocks/{sample_stock.symbol}/prices",
        headers=auth_headers,
        params={"days": 30}
    )
    assert response.status_code == 200
    price_data = response.json()
    assert len(price_data["data"]) == 30
    assert price_data["data"][-1]["close"] > price_data["data"][0]["close"]  # Upward trend

    # Step 3: Fetch fundamentals
    response = await async_client.get(
        f"/api/v1/stocks/{sample_stock.symbol}/fundamentals",
        headers=auth_headers
    )
    assert response.status_code == 200
    fundamental_data = response.json()
    assert fundamental_data["data"]["pe_ratio"] == 28.5
    assert fundamental_data["data"]["roe"] == 0.35

    # Step 4: Request recommendation generation
    with patch("backend.services.recommendation_service.RecommendationService.generate_recommendation") as mock_rec:
        mock_rec.return_value = Recommendation(
            id=1,
            stock_id=sample_stock.id,
            recommendation_type=RecommendationTypeEnum.BUY,
            confidence_score=0.82,
            current_price=Decimal("165.00"),
            target_price=Decimal("185.00"),
            stop_loss=Decimal("155.00"),
            time_horizon_days=90,
            reasoning="Strong fundamentals with positive price momentum. High ROE and growing revenue.",
            key_factors=["strong_fundamentals", "upward_trend", "high_roe"],
            risk_level="medium",
            technical_score=0.78,
            fundamental_score=0.86,
            sentiment_score=0.75,
            is_active=True,
            valid_until=datetime.utcnow() + timedelta(days=30)
        )

        response = await async_client.post(
            f"/api/v1/recommendations/generate/{sample_stock.symbol}",
            headers=auth_headers
        )
        assert response.status_code == 200
        recommendation_data = response.json()
        assert recommendation_data["data"]["recommendation_type"] == "buy"
        assert recommendation_data["data"]["confidence_score"] >= 0.80
        assert "reasoning" in recommendation_data["data"]


@pytest.mark.asyncio
async def test_stock_data_caching(
    async_client: AsyncClient,
    db_session: AsyncSession,
    sample_stock: Stock,
    sample_price_history,
    auth_headers,
    mock_cache
):
    """
    Test cache hit/miss scenarios for stock data retrieval.

    Validates that frequently accessed stock data is properly cached
    and subsequent requests hit the cache for improved performance.
    """
    # First request - cache miss
    mock_cache.get.return_value = None

    response = await async_client.get(
        f"/api/v1/stocks/{sample_stock.symbol}",
        headers=auth_headers
    )
    assert response.status_code == 200

    # Verify cache was set
    assert mock_cache.set.called
    cache_key = f"stock:{sample_stock.symbol}"

    # Second request - cache hit
    cached_data = {
        "id": sample_stock.id,
        "symbol": sample_stock.symbol,
        "name": sample_stock.name,
        "exchange": sample_stock.exchange
    }
    mock_cache.get.return_value = cached_data

    response = await async_client.get(
        f"/api/v1/stocks/{sample_stock.symbol}",
        headers=auth_headers
    )
    assert response.status_code == 200

    # Verify cache was checked
    assert mock_cache.get.call_count >= 2


@pytest.mark.asyncio
async def test_stock_to_portfolio_addition(
    async_client: AsyncClient,
    db_session: AsyncSession,
    sample_stock: Stock,
    sample_price_history,
    auth_headers,
    test_user
):
    """
    Test adding analyzed stock to user portfolio.

    Validates the workflow of analyzing a stock and adding it to a portfolio,
    including position creation and portfolio value calculation.
    """
    # Create a portfolio for the test user
    portfolio = Portfolio(
        user_id=test_user.id,
        name="Test Portfolio",
        description="Integration test portfolio",
        cash_balance=Decimal("10000.00"),
        is_public=False,
        is_default=True,
        benchmark="SPY"
    )
    db_session.add(portfolio)
    await db_session.commit()
    await db_session.refresh(portfolio)

    # Step 1: Get stock analysis
    response = await async_client.get(
        f"/api/v1/stocks/{sample_stock.symbol}/analysis",
        headers=auth_headers
    )
    assert response.status_code == 200

    # Step 2: Add stock to portfolio
    current_price = Decimal("165.00")
    quantity = Decimal("10")

    response = await async_client.post(
        f"/api/v1/portfolios/{portfolio.id}/positions",
        headers=auth_headers,
        json={
            "stock_symbol": sample_stock.symbol,
            "quantity": float(quantity),
            "average_cost": float(current_price)
        }
    )
    assert response.status_code == 201
    position_data = response.json()
    assert position_data["data"]["stock_symbol"] == "AAPL"
    assert Decimal(str(position_data["data"]["quantity"])) == quantity

    # Step 3: Verify portfolio updated
    stmt = select(Position).where(Position.portfolio_id == portfolio.id)
    result = await db_session.execute(stmt)
    position = result.scalar_one()

    assert position.stock_id == sample_stock.id
    assert position.quantity == quantity
    assert position.average_cost == current_price


@pytest.mark.asyncio
async def test_real_time_quote_to_alert(
    async_client: AsyncClient,
    db_session: AsyncSession,
    sample_stock: Stock,
    auth_headers,
    test_user
):
    """
    Test real-time price change triggering alert.

    Validates that price alerts are correctly evaluated and triggered
    when real-time stock prices cross specified thresholds.
    """
    # Create a price alert
    alert = Alert(
        user_id=test_user.id,
        stock_id=sample_stock.id,
        alert_type="price_threshold",
        condition={
            "type": "above",
            "threshold": 170.00
        },
        is_active=True,
        notification_methods=["email", "push"]
    )
    db_session.add(alert)
    await db_session.commit()
    await db_session.refresh(alert)

    # Mock real-time price update (price crosses threshold)
    with patch("backend.services.realtime_price_service.RealtimePriceService.get_quote") as mock_quote:
        mock_quote.return_value = {
            "symbol": "AAPL",
            "price": 171.50,
            "change": 6.50,
            "change_percent": 3.94,
            "timestamp": datetime.utcnow()
        }

        # Trigger alert check
        response = await async_client.post(
            "/api/v1/alerts/check",
            headers=auth_headers,
            json={"stock_symbol": "AAPL"}
        )
        assert response.status_code == 200

        # Verify alert was triggered
        await db_session.refresh(alert)
        assert alert.triggered_count == 1
        assert alert.last_triggered is not None


@pytest.mark.asyncio
async def test_stock_fundamentals_to_thesis(
    async_client: AsyncClient,
    db_session: AsyncSession,
    sample_stock: Stock,
    sample_fundamentals,
    auth_headers
):
    """
    Test generating investment thesis from fundamental analysis.

    Validates that fundamental data is properly analyzed and used to
    generate a comprehensive investment thesis with bull/bear cases.
    """
    # Request thesis generation
    with patch("backend.services.thesis_service.ThesisService.generate_thesis") as mock_thesis:
        mock_thesis.return_value = {
            "symbol": "AAPL",
            "title": "Apple Inc. - Strong Fundamentals with Premium Valuation",
            "summary": "Apple demonstrates exceptional profitability metrics with ROE of 35% and strong cash generation.",
            "bull_case": [
                "Industry-leading margins with 44% gross margin",
                "Strong balance sheet with $50B cash",
                "Consistent revenue growth and innovation"
            ],
            "bear_case": [
                "High P/E ratio of 28.5 suggests premium valuation",
                "Significant debt of $120B",
                "Market saturation in core products"
            ],
            "key_metrics": {
                "roe": 0.35,
                "pe_ratio": 28.5,
                "free_cash_flow": 28000000000
            },
            "valuation_assessment": "fairly_valued",
            "confidence": 0.78
        }

        response = await async_client.post(
            f"/api/v1/thesis/generate",
            headers=auth_headers,
            json={"symbol": sample_stock.symbol}
        )
        assert response.status_code == 200
        thesis_data = response.json()

        assert thesis_data["data"]["symbol"] == "AAPL"
        assert "bull_case" in thesis_data["data"]
        assert "bear_case" in thesis_data["data"]
        assert len(thesis_data["data"]["bull_case"]) >= 3
        assert len(thesis_data["data"]["bear_case"]) >= 3
        assert thesis_data["data"]["confidence"] > 0.7
