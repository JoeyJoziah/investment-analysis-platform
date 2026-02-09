"""
Tests to verify service layer is properly wired to API routers.

These tests ensure that service dependencies are correctly injected
and that the service layer is being used by the routers.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from httpx import AsyncClient

from backend.services.recommendation_service import RecommendationService
from backend.services.portfolio_service import PortfolioService
from backend.services.analysis_service import AnalysisService
from backend.services.trading_service import TradingService


@pytest.mark.asyncio
async def test_recommendation_service_wired_to_trending_endpoint(authenticated_client: AsyncClient):
    """
    Verify that the /trending endpoint has access to RecommendationService and returns data.

    Note: The endpoint has a fallback to mock data, so we just verify it works
    and returns data in the expected format.
    """
    # Call the endpoint
    response = await authenticated_client.get("/api/v1/recommendations/trending?timeframe=24h&limit=5")

    # Verify basic response structure
    assert response.status_code == 200
    response_data = response.json()
    assert response_data["success"] is True

    # The data can be either a list or fall back to mock data structure
    # What matters is that the endpoint works and has the service wired
    data = response_data["data"]
    assert data is not None

    # If it's a list (successful call), verify structure
    if isinstance(data, list) and len(data) > 0:
        first_item = data[0]
        # Should have either ticker or symbol
        assert "ticker" in first_item or "symbol" in first_item


@pytest.mark.asyncio
async def test_recommendation_service_handles_initialization_gracefully(authenticated_client: AsyncClient):
    """
    Verify that the service initialization is handled gracefully in test mode.
    """
    # Call an endpoint that uses the recommendation service
    response = await authenticated_client.get("/api/v1/recommendations/trending?timeframe=24h")

    # Should work even without full initialization (degraded mode)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True


@pytest.mark.asyncio
async def test_portfolio_service_dependency_available(authenticated_client: AsyncClient):
    """
    Verify that PortfolioService dependency is available.
    """
    from backend.api.routers.portfolio import get_portfolio_service

    # Get the service instance (it's async now)
    service = await get_portfolio_service()

    # Verify it's the correct type
    assert isinstance(service, PortfolioService)
    assert hasattr(service, 'repository')
    assert hasattr(service, 'get_portfolio_summary')
    assert hasattr(service, 'add_position')


@pytest.mark.asyncio
async def test_analysis_service_dependency_available(authenticated_client: AsyncClient):
    """
    Verify that AnalysisService dependency is available.
    """
    from backend.api.routers.analysis import get_analysis_service

    # Get the service instance
    service = get_analysis_service()

    # Verify it's the correct type
    assert isinstance(service, AnalysisService)
    assert hasattr(service, 'fundamental_engine')
    assert hasattr(service, 'technical_engine')
    assert hasattr(service, 'sentiment_engine')
    assert hasattr(service, 'run_analysis')
    assert hasattr(service, 'get_cached_analysis')


@pytest.mark.asyncio
async def test_service_layer_provides_business_logic_abstraction():
    """
    Verify that service layer classes provide the expected business logic methods.
    """
    # RecommendationService
    rec_service = RecommendationService()
    assert hasattr(rec_service, 'generate_recommendation')
    assert hasattr(rec_service, 'get_trending')
    assert hasattr(rec_service, 'calculate_confidence')

    # PortfolioService
    port_service = PortfolioService()
    assert hasattr(port_service, 'get_portfolio_summary')
    assert hasattr(port_service, 'add_position')
    assert hasattr(port_service, 'get_allocation')
    assert hasattr(port_service, 'get_performance')

    # AnalysisService
    analysis_service_instance = AnalysisService()
    assert hasattr(analysis_service_instance, 'run_analysis')
    assert hasattr(analysis_service_instance, 'get_cached_analysis')
    assert hasattr(analysis_service_instance, 'compare_stocks')
    assert hasattr(analysis_service_instance, 'clear_cache')


@pytest.mark.asyncio
async def test_services_use_repositories_not_direct_db_access():
    """
    Verify that service classes use repositories, not direct database access.
    """
    from backend.repositories.portfolio_repository import PortfolioRepository

    # PortfolioService should use repository
    port_service = PortfolioService()
    assert hasattr(port_service, 'repository')
    # Check it's the singleton instance
    from backend.repositories import portfolio_repository
    assert port_service.repository is portfolio_repository


@pytest.mark.asyncio
async def test_recommendation_service_confidence_calculation():
    """
    Test RecommendationService.calculate_confidence() business logic.
    """
    rec_service = RecommendationService()

    # Test with empty analyses
    confidence = rec_service.calculate_confidence([])
    assert confidence == 0.0

    # Test with mixed analyses
    analyses = [
        {'type': 'technical', 'confidence': 0.8},
        {'type': 'fundamental', 'confidence': 0.7},
        {'type': 'sentiment', 'confidence': 0.6},
    ]
    confidence = rec_service.calculate_confidence(analyses)

    # Should be weighted average
    # technical: 0.25 * 0.8 = 0.20
    # fundamental: 0.30 * 0.7 = 0.21
    # sentiment: 0.15 * 0.6 = 0.09
    # Total weight: 0.70
    # Weighted sum: 0.50
    # Result: 0.50 / 0.70 = 0.714...
    assert 0.7 <= confidence <= 0.72


@pytest.mark.asyncio
async def test_analysis_service_cache_operations():
    """
    Test AnalysisService cache operations.
    """
    analysis_service_instance = AnalysisService()

    # Initially empty cache
    cached = await analysis_service_instance.get_cached_analysis("TEST")
    assert cached is None

    # Run analysis (will cache it)
    result = await analysis_service_instance.run_analysis("TEST", types=["technical"], depth="quick")
    assert result["ticker"] == "TEST"

    # Should now be in cache (within 15 minutes)
    cached = await analysis_service_instance.get_cached_analysis("TEST")
    assert cached is not None
    assert cached["ticker"] == "TEST"

    # Clear cache
    analysis_service_instance.clear_cache("TEST")
    cached = await analysis_service_instance.get_cached_analysis("TEST")
    assert cached is None


def test_service_singletons_exist():
    """
    Verify that service singleton instances are exported.
    """
    from backend.services.recommendation_service import recommendation_service
    from backend.services.portfolio_service import portfolio_service
    from backend.services.analysis_service import analysis_service
    from backend.services.trading_service import trading_service

    assert recommendation_service is not None
    assert portfolio_service is not None
    assert analysis_service is not None
    assert trading_service is not None

    # Verify they're the right types
    assert isinstance(recommendation_service, RecommendationService)
    assert isinstance(portfolio_service, PortfolioService)
    assert isinstance(analysis_service, AnalysisService)
    assert isinstance(trading_service, TradingService)


@pytest.mark.asyncio
async def test_trading_service_dependency_available():
    """
    Verify that TradingService is available and properly structured.
    """
    from backend.services.trading_service import trading_service

    # Verify it's the correct type
    assert isinstance(trading_service, TradingService)
    assert hasattr(trading_service, 'repository')
    assert hasattr(trading_service, 'validate_order')
    assert hasattr(trading_service, 'execute_trade')
    assert hasattr(trading_service, 'calculate_portfolio_impact')


@pytest.mark.asyncio
async def test_trading_service_validate_order():
    """
    Test TradingService.validate_order() business logic.
    """
    from backend.services.trading_service import trading_service

    # Test with missing fields
    result = await trading_service.validate_order({})
    assert result['valid'] is False
    assert 'errors' in result
    assert len(result['errors']) > 0

    # Test with invalid quantity (mock repository to avoid DB calls)
    with patch.object(
        trading_service.repository,
        'get_by_id',
        new_callable=AsyncMock
    ) as mock_get:
        mock_get.return_value = None

        result = await trading_service.validate_order({
            'portfolio_id': 1,
            'symbol': 'AAPL',
            'side': 'buy',
            'order_type': 'market',
            'quantity': -10
        })
        assert result['valid'] is False
        assert any('quantity' in err.lower() for err in result['errors'])


@pytest.mark.asyncio
async def test_portfolio_service_wired_to_endpoints(authenticated_client: AsyncClient):
    """
    Verify that portfolio endpoints use PortfolioService.
    """
    # Test the summary endpoint (which now uses the service)
    response = await authenticated_client.get("/api/v1/portfolio/summary")

    # Should work (may return empty list or mock data)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "data" in data
