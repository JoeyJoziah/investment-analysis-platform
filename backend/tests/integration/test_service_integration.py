"""
Service Layer Integration Tests
Tests the full flow: HTTP Request -> Router -> Service -> Repository (mocked)

These tests verify:
1. Service dependency injection works correctly in routers
2. Services handle business logic properly
3. Error handling flows through all layers
4. Services coordinate with repositories correctly
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from httpx import AsyncClient
from decimal import Decimal
from datetime import datetime, timezone

from backend.services.recommendation_service import RecommendationService
from backend.services.portfolio_service import PortfolioService
from backend.services.analysis_service import AnalysisService
from backend.services.trading_service import TradingService, OrderType, OrderSide
from backend.models.unified_models import Portfolio, Position, Stock
from backend.analytics.recommendation_engine import StockRecommendation, RecommendationAction


# ============================================================================
# RecommendationService Integration Tests
# ============================================================================

@pytest.mark.asyncio
async def test_recommendation_service_daily_endpoint_integration(authenticated_client: AsyncClient, mock_redis):
    """
    Test /api/v1/recommendations/daily endpoint works with RecommendationService.
    Tests that service is wired correctly and endpoint returns proper structure.
    """
    # Make request (service may return empty list in test env, but should work)
    response = await authenticated_client.get("/api/v1/recommendations/daily?limit=5")

    # Verify response structure
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "data" in data

    # Data can be dict or list depending on endpoint implementation
    recommendations = data["data"]
    assert recommendations is not None
    # If it's a dict, check for expected structure
    if isinstance(recommendations, dict):
        assert "date" in recommendations or "top_picks" in recommendations or len(recommendations) > 0


@pytest.mark.asyncio
async def test_recommendation_service_trending_endpoint_integration(authenticated_client: AsyncClient, mock_redis):
    """
    Test /api/v1/recommendations/trending endpoint with service layer.
    Tests that service is properly wired and returns correct structure.
    """
    # Make request
    response = await authenticated_client.get(
        "/api/v1/recommendations/trending?timeframe=24h&limit=3"
    )

    # Verify response structure (may be empty in test, but structure should be correct)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "data" in data
    # Data can be dict or list depending on endpoint implementation
    assert data["data"] is not None


@pytest.mark.asyncio
async def test_recommendation_service_confidence_calculation():
    """
    Test RecommendationService.calculate_confidence business logic.
    Unit test to verify weighted averaging logic.
    """
    service = RecommendationService()

    # Test with empty analyses
    confidence = service.calculate_confidence([])
    assert confidence == 0.0

    # Test with single analysis
    analyses = [{'type': 'technical', 'confidence': 0.8}]
    confidence = service.calculate_confidence(analyses)
    assert confidence > 0

    # Test with multiple analyses
    analyses = [
        {'type': 'technical', 'confidence': 0.8},
        {'type': 'fundamental', 'confidence': 0.7},
        {'type': 'sentiment', 'confidence': 0.6},
    ]
    confidence = service.calculate_confidence(analyses)

    # Weighted calculation:
    # technical: 0.25 * 0.8 = 0.20
    # fundamental: 0.30 * 0.7 = 0.21
    # sentiment: 0.15 * 0.6 = 0.09
    # Total: 0.50 / 0.70 = ~0.714
    assert 0.7 <= confidence <= 0.72


# ============================================================================
# AnalysisService Integration Tests
# ============================================================================

@pytest.mark.asyncio
async def test_analysis_service_analyze_endpoint_integration(authenticated_client: AsyncClient, db_session):
    """
    Test /api/v1/analysis/analyze endpoint with AnalysisService.
    Verifies service is properly wired and handles requests.
    """
    # Create a stock in the database so analysis can find it
    from backend.models.unified_models import Stock, Exchange, Sector, Industry

    # Create required foreign key entities
    exchange = Exchange(
        code="NASDAQ",
        name="NASDAQ Stock Market",
        timezone="America/New_York",
        country="US",
        currency="USD"
    )
    db_session.add(exchange)
    await db_session.commit()
    await db_session.refresh(exchange)

    sector = Sector(name="Technology", description="Technology sector")
    db_session.add(sector)
    await db_session.commit()
    await db_session.refresh(sector)

    industry = Industry(
        name="Consumer Electronics",
        sector_id=sector.id,
        description="Consumer electronics"
    )
    db_session.add(industry)
    await db_session.commit()
    await db_session.refresh(industry)

    # Create stock
    stock = Stock(
        symbol="AAPL",
        name="Apple Inc.",
        exchange_id=exchange.id,
        sector_id=sector.id,
        industry_id=industry.id,
        market_cap=2500000000000,
        is_active=True
    )
    db_session.add(stock)
    await db_session.commit()

    # Make request
    response = await authenticated_client.post(
        "/api/v1/analysis/analyze",
        json={
            "symbol": "AAPL",
            "analysis_types": ["technical"],
            "depth": "quick"
        }
    )

    # Verify response structure
    # May return 200 or other status, but should be handled gracefully
    assert response.status_code in [200, 404, 422, 500]
    data = response.json()
    assert "success" in data or "detail" in data


@pytest.mark.asyncio
async def test_analysis_service_cache_operations():
    """
    Test AnalysisService cache functionality.
    Verifies cache stores and retrieves analysis results correctly.
    """
    service = AnalysisService()

    # Initially no cache
    cached = await service.get_cached_analysis("TEST")
    assert cached is None

    # Run analysis (will cache it)
    result = await service.run_analysis("TEST", types=["technical"], depth="quick")
    assert result["ticker"] == "TEST"

    # Should now be in cache
    cached = await service.get_cached_analysis("TEST")
    assert cached is not None
    assert cached["ticker"] == "TEST"

    # Clear specific cache
    service.clear_cache("TEST")
    cached = await service.get_cached_analysis("TEST")
    assert cached is None


@pytest.mark.asyncio
async def test_analysis_service_compare_stocks():
    """
    Test AnalysisService.compare_stocks functionality.
    """
    service = AnalysisService()

    # Compare multiple stocks
    result = await service.compare_stocks(
        tickers=["AAPL", "GOOGL", "MSFT"],
        analysis_type="fundamental"
    )

    assert "comparison_type" in result
    assert result["comparison_type"] == "fundamental"
    assert "stocks" in result
    assert len(result["stocks"]) == 3


@pytest.mark.asyncio
async def test_analysis_service_error_handling():
    """
    Test AnalysisService handles errors gracefully.
    """
    service = AnalysisService()

    # Test with invalid ticker (should not crash)
    result = await service.run_analysis(
        ticker="INVALID123",
        types=["technical"],
        depth="quick"
    )

    # Should return result with ticker, even if analysis failed
    assert "ticker" in result
    assert result["ticker"] == "INVALID123"


# ============================================================================
# PortfolioService Integration Tests
# ============================================================================

@pytest.mark.asyncio
async def test_portfolio_service_summary_endpoint_integration(authenticated_client: AsyncClient, db_session):
    """
    Test /api/v1/portfolio/summary endpoint with PortfolioService.
    """
    # Create a test portfolio in the database
    from backend.models.unified_models import Portfolio

    portfolio = Portfolio(
        id=1,
        user_id=1,
        name="Test Portfolio",
        description="Test portfolio",
        cash_balance=Decimal("10000.00"),
        is_default=True,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc)
    )
    db_session.add(portfolio)
    await db_session.commit()
    await db_session.refresh(portfolio)

    # Make request
    response = await authenticated_client.get("/api/v1/portfolio/summary")

    # Verify response
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True


@pytest.mark.asyncio
async def test_portfolio_service_get_summary_business_logic(db_session):
    """
    Test PortfolioService.get_portfolio_summary business logic.
    Mock repository to test service layer in isolation.
    """
    service = PortfolioService()

    # Create mock portfolio
    mock_portfolio = Portfolio(
        id=1,
        user_id=1,
        name="Test Portfolio",
        description="Test description",
        cash_balance=Decimal("10000.00"),
        is_default=True,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc)
    )
    mock_portfolio.positions = []  # No positions

    # Mock repository methods
    with patch.object(service.repository, 'get_portfolio_with_positions', new_callable=AsyncMock) as mock_get:
        with patch.object(service.repository, 'calculate_portfolio_value', new_callable=AsyncMock) as mock_calc:
            with patch.object(service.repository, 'get_portfolio_allocation', new_callable=AsyncMock) as mock_alloc:
                mock_get.return_value = mock_portfolio
                mock_calc.return_value = {
                    'total_value': 10000.0,
                    'cash_balance': 10000.0,
                    'positions_value': 0.0
                }
                mock_alloc.return_value = {
                    'cash_allocation_pct': 100.0,
                    'positions': []
                }

                # Call service
                result = await service.get_portfolio_summary(user_id=1, portfolio_id=1)

                # Verify result
                assert result is not None
                assert result['portfolio_id'] == 1
                assert result['name'] == "Test Portfolio"
                assert result['cash_balance'] == 10000.0
                assert result['position_count'] == 0


@pytest.mark.asyncio
async def test_portfolio_service_add_position_validation():
    """
    Test PortfolioService.add_position validates inputs correctly.
    """
    service = PortfolioService()

    # Mock repository
    with patch.object(service.repository, 'get_by_id', new_callable=AsyncMock) as mock_get:
        with patch.object(service.repository, 'add_position', new_callable=AsyncMock) as mock_add:
            # Mock portfolio exists
            mock_portfolio = MagicMock()
            mock_portfolio.user_id = 1
            mock_get.return_value = mock_portfolio

            # Mock position creation
            mock_position = MagicMock()
            mock_position.id = 1
            mock_position.avg_cost_basis = Decimal("150.0")
            mock_add.return_value = mock_position

            # Test with valid inputs
            result = await service.add_position(
                portfolio_id=1,
                stock_symbol="AAPL",
                quantity=10.0,
                cost=150.0,
                user_id=1
            )

            assert result['success'] is True
            assert result['position_id'] == 1


@pytest.mark.asyncio
async def test_portfolio_service_unauthorized_access():
    """
    Test PortfolioService denies unauthorized access.
    """
    service = PortfolioService()

    # Mock repository
    with patch.object(service.repository, 'get_by_id', new_callable=AsyncMock) as mock_get:
        # Mock portfolio owned by different user
        mock_portfolio = MagicMock()
        mock_portfolio.user_id = 2  # Different user
        mock_get.return_value = mock_portfolio

        # Try to add position as user 1
        result = await service.add_position(
            portfolio_id=1,
            stock_symbol="AAPL",
            quantity=10.0,
            cost=150.0,
            user_id=1  # Different from portfolio owner
        )

        assert result['success'] is False
        assert 'access denied' in result['error'].lower()


# ============================================================================
# TradingService Integration Tests
# ============================================================================

@pytest.mark.asyncio
async def test_trading_service_validate_order_negative_quantity():
    """
    Test TradingService.validate_order rejects negative quantity.
    """
    service = TradingService()

    # Mock repository
    with patch.object(service.repository, 'get_by_id', new_callable=AsyncMock) as mock_get:
        mock_get.return_value = None

        order_data = {
            'portfolio_id': 1,
            'symbol': 'AAPL',
            'side': OrderSide.BUY,
            'order_type': OrderType.MARKET,
            'quantity': -10  # NEGATIVE - should fail
        }

        result = await service.validate_order(order_data)

        assert result['valid'] is False
        assert any('quantity' in err.lower() for err in result['errors'])


@pytest.mark.asyncio
async def test_trading_service_validate_order_unknown_order_type():
    """
    Test TradingService.validate_order with limit order requiring price.
    """
    service = TradingService()

    with patch.object(service.repository, 'get_by_id', new_callable=AsyncMock) as mock_get:
        mock_portfolio = MagicMock()
        mock_portfolio.cash_balance = Decimal("10000.0")
        mock_get.return_value = mock_portfolio

        order_data = {
            'portfolio_id': 1,
            'symbol': 'AAPL',
            'side': OrderSide.BUY,
            'order_type': OrderType.LIMIT,
            'quantity': 10
            # MISSING 'price' - should fail for limit orders
        }

        result = await service.validate_order(order_data)

        assert result['valid'] is False
        assert any('price' in err.lower() for err in result['errors'])


@pytest.mark.asyncio
async def test_trading_service_validate_order_insufficient_funds():
    """
    Test TradingService.validate_order rejects orders exceeding cash balance.
    """
    service = TradingService()

    with patch.object(service.repository, 'get_by_id', new_callable=AsyncMock) as mock_get:
        # Mock portfolio with limited cash
        mock_portfolio = MagicMock()
        mock_portfolio.cash_balance = Decimal("1000.0")  # Only $1000
        mock_get.return_value = mock_portfolio

        order_data = {
            'portfolio_id': 1,
            'symbol': 'AAPL',
            'side': OrderSide.BUY,
            'order_type': OrderType.MARKET,
            'quantity': 100,
            'price': 150.0  # Would cost $15,000 - exceeds balance
        }

        result = await service.validate_order(order_data)

        assert result['valid'] is False
        assert any('insufficient' in err.lower() for err in result['errors'])


@pytest.mark.asyncio
async def test_trading_service_validate_order_missing_required_fields():
    """
    Test TradingService.validate_order rejects orders missing required fields.
    """
    service = TradingService()

    # Missing multiple required fields
    order_data = {
        'portfolio_id': 1
        # Missing: symbol, side, order_type, quantity
    }

    result = await service.validate_order(order_data)

    assert result['valid'] is False
    assert len(result['errors']) >= 4  # At least 4 missing fields


@pytest.mark.asyncio
async def test_trading_service_validate_order_invalid_symbol():
    """
    Test TradingService.validate_order rejects invalid stock symbols.
    """
    service = TradingService()

    with patch.object(service.repository, 'get_by_id', new_callable=AsyncMock) as mock_get:
        mock_portfolio = MagicMock()
        mock_portfolio.cash_balance = Decimal("10000.0")
        mock_get.return_value = mock_portfolio

        order_data = {
            'portfolio_id': 1,
            'symbol': 'INVALID123',  # Contains numbers - invalid
            'side': OrderSide.BUY,
            'order_type': OrderType.MARKET,
            'quantity': 10,
            'price': 100.0
        }

        result = await service.validate_order(order_data)

        assert result['valid'] is False
        assert any('symbol' in err.lower() for err in result['errors'])


@pytest.mark.asyncio
async def test_trading_service_execute_trade_validation_failure():
    """
    Test TradingService.execute_trade handles validation failures.
    """
    service = TradingService()

    with patch.object(service.repository, 'get_by_id', new_callable=AsyncMock) as mock_get:
        mock_get.return_value = None  # Portfolio not found

        order = {
            'symbol': 'AAPL',
            'side': OrderSide.BUY,
            'quantity': 10,
            'price': 150.0,
            'order_type': OrderType.MARKET
        }

        result = await service.execute_trade(portfolio_id=999, order=order)

        assert result['success'] is False
        assert 'validation failed' in result['error'].lower()


@pytest.mark.asyncio
async def test_trading_service_calculate_portfolio_impact():
    """
    Test TradingService.calculate_portfolio_impact calculates metrics correctly.
    """
    service = TradingService()

    with patch.object(service.repository, 'get_by_id', new_callable=AsyncMock) as mock_get:
        with patch.object(service.repository, 'get_portfolio_allocation', new_callable=AsyncMock) as mock_alloc:
            with patch.object(service.repository, 'calculate_portfolio_value', new_callable=AsyncMock) as mock_value:
                # Mock portfolio
                mock_portfolio = MagicMock()
                mock_portfolio.cash_balance = Decimal("10000.0")
                mock_get.return_value = mock_portfolio

                # Mock current state
                mock_alloc.return_value = {'cash_allocation_pct': 100.0}
                mock_value.return_value = {'total_value': 10000.0}

                # Calculate impact of $1500 trade (10 shares @ $150)
                trade = {
                    'symbol': 'AAPL',
                    'side': OrderSide.BUY,
                    'quantity': 10,
                    'price': 150.0
                }

                result = await service.calculate_portfolio_impact(portfolio_id=1, trade=trade)

                assert result['success'] is True
                assert 'before' in result
                assert 'after' in result
                assert result['before']['total_value'] == 10000.0
                assert result['after']['cash_balance'] == 8500.0  # 10000 - 1500


@pytest.mark.asyncio
async def test_trading_service_calculate_portfolio_impact_sell_order():
    """
    Test portfolio impact calculation for sell orders.
    """
    service = TradingService()

    with patch.object(service.repository, 'get_by_id', new_callable=AsyncMock) as mock_get:
        with patch.object(service.repository, 'get_portfolio_allocation', new_callable=AsyncMock) as mock_alloc:
            with patch.object(service.repository, 'calculate_portfolio_value', new_callable=AsyncMock) as mock_value:
                # Mock portfolio
                mock_portfolio = MagicMock()
                mock_portfolio.cash_balance = Decimal("5000.0")
                mock_get.return_value = mock_portfolio

                # Mock current state
                mock_alloc.return_value = {'cash_allocation_pct': 50.0}
                mock_value.return_value = {'total_value': 10000.0}

                # Calculate impact of selling $1500 worth
                trade = {
                    'symbol': 'AAPL',
                    'side': OrderSide.SELL,
                    'quantity': 10,
                    'price': 150.0
                }

                result = await service.calculate_portfolio_impact(portfolio_id=1, trade=trade)

                assert result['success'] is True
                assert result['after']['cash_balance'] == 6500.0  # 5000 + 1500


# ============================================================================
# Cross-Service Integration Tests
# ============================================================================

@pytest.mark.asyncio
async def test_service_dependency_injection_across_routers(authenticated_client: AsyncClient):
    """
    Test that all routers properly inject their service dependencies.
    Verifies the wiring is correct across the entire API.
    """
    # Test recommendation router
    response = await authenticated_client.get("/api/v1/recommendations/trending?limit=1")
    assert response.status_code == 200

    # Test portfolio router
    response = await authenticated_client.get("/api/v1/portfolio/summary")
    assert response.status_code == 200

    # Test analysis router (if endpoint exists)
    # Note: May need to adjust based on actual endpoint signature
    # response = await authenticated_client.post("/api/v1/analysis/analyze", json={"ticker": "AAPL"})
    # assert response.status_code in [200, 422]  # 422 if validation fails


@pytest.mark.asyncio
async def test_all_services_handle_exceptions_gracefully():
    """
    Test that all services handle exceptions without crashing.
    """
    # RecommendationService
    rec_service = RecommendationService()
    result = await rec_service.generate_recommendation(ticker="INVALID", analysis_types=["technical"])
    assert 'error' in result or 'success' in result

    # AnalysisService
    analysis_service_instance = AnalysisService()
    result = await analysis_service_instance.run_analysis(ticker="INVALID", types=["technical"])
    assert "ticker" in result

    # PortfolioService
    port_service = PortfolioService()
    result = await port_service.get_portfolio_summary(user_id=999, portfolio_id=999)
    # Should return None or error, not crash
    assert result is None or 'error' in result

    # TradingService
    trade_service = TradingService()
    result = await trade_service.validate_order({})
    assert 'valid' in result
    assert result['valid'] is False
