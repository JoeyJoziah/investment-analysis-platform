"""
Comprehensive API Integration Tests for Investment Analysis Platform
Tests all critical API endpoints with real-world scenarios and error conditions.
"""

import pytest
import asyncio
import json
from datetime import datetime, date, timedelta
from typing import Dict, Any, List
from unittest.mock import AsyncMock, patch, MagicMock
from httpx import AsyncClient
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.main import app
from backend.config.database import get_async_db_session
from backend.auth.oauth2 import get_current_user
from backend.models.unified_models import User, Portfolio, Stock
from backend.utils.cache import get_cache_manager


class TestAPIEndpointsIntegration:
    """Test API endpoints with real database and cache integration."""

    @pytest.fixture
    async def async_client(self):
        """Create async HTTP client for testing."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            yield client

    @pytest.fixture
    def mock_user(self):
        """Create mock authenticated user."""
        return User(
            id=1,
            username="testuser",
            email="test@example.com",
            is_active=True,
            created_at=datetime.utcnow()
        )

    @pytest.fixture
    async def mock_db_session(self):
        """Create mock database session."""
        session = AsyncMock(spec=AsyncSession)
        session.commit = AsyncMock()
        session.rollback = AsyncMock()
        session.close = AsyncMock()
        return session

    @pytest.fixture
    def auth_headers(self):
        """Create authentication headers."""
        return {"Authorization": "Bearer test_token"}

    def override_dependencies(self, mock_user, mock_db_session):
        """Override app dependencies for testing."""
        app.dependency_overrides[get_current_user] = lambda: mock_user
        app.dependency_overrides[get_async_db_session] = lambda: mock_db_session

    def cleanup_dependencies(self):
        """Clean up dependency overrides."""
        app.dependency_overrides.clear()

    @pytest.mark.asyncio
    @pytest.mark.api
    async def test_health_endpoint_integration(self, async_client):
        """Test health endpoint with all components."""
        with patch('backend.config.database.get_async_db_session') as mock_db:
            mock_session = AsyncMock()
            mock_db.return_value = mock_session
            
            response = await async_client.get("/api/health/status")
            
            assert response.status_code == 200
            data = response.json()
            assert "status" in data
            assert "timestamp" in data
            assert "components" in data
            
            # Verify all critical components are checked
            components = data["components"]
            assert "database" in components
            assert "cache" in components
            assert "external_apis" in components

    @pytest.mark.asyncio
    @pytest.mark.api
    async def test_recommendations_endpoint_integration(self, async_client, mock_user, mock_db_session):
        """Test recommendations endpoint with ML integration."""
        self.override_dependencies(mock_user, mock_db_session)
        
        try:
            # Mock portfolio repository responses
            mock_portfolios = [
                MagicMock(
                    id=1,
                    name="Test Portfolio",
                    cash_balance=10000,
                    strategy="balanced",
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
            ]
            
            # Mock stock repository responses
            mock_stocks = [
                MagicMock(
                    symbol="AAPL",
                    name="Apple Inc.",
                    sector="Technology", 
                    market_cap=3000000000000
                ),
                MagicMock(
                    symbol="GOOGL",
                    name="Alphabet Inc.",
                    sector="Technology",
                    market_cap=2000000000000
                )
            ]
            
            with patch('backend.repositories.portfolio_repository.get_user_portfolios', return_value=mock_portfolios):
                with patch('backend.repositories.stock_repository.get_top_stocks', return_value=mock_stocks):
                    with patch('backend.repositories.price_repository.get_price_history') as mock_price_history:
                        # Mock price data
                        mock_price_data = [
                            MagicMock(
                                open=150.0, high=155.0, low=149.0, close=154.0,
                                volume=1000000, date=date.today() - timedelta(days=i)
                            )
                            for i in range(30)
                        ]
                        mock_price_history.return_value = mock_price_data
                        
                        response = await async_client.get(
                            "/api/recommendations/daily",
                            headers={"Authorization": "Bearer test_token"}
                        )
                        
                        assert response.status_code == 200
                        data = response.json()
                        
                        # Verify response structure
                        assert "date" in data
                        assert "market_outlook" in data
                        assert "top_picks" in data
                        assert "watchlist" in data
                        assert "sector_focus" in data
                        assert "market_sentiment" in data
                        assert "risk_assessment" in data
                        
                        # Verify recommendations quality
                        top_picks = data["top_picks"]
                        assert isinstance(top_picks, list)
                        
                        if top_picks:
                            rec = top_picks[0]
                            assert "symbol" in rec
                            assert "confidence_score" in rec
                            assert "target_price" in rec
                            assert "current_price" in rec
                            assert "reasoning" in rec
                            assert 0 <= rec["confidence_score"] <= 1
                            
        finally:
            self.cleanup_dependencies()

    @pytest.mark.asyncio
    @pytest.mark.api
    async def test_portfolio_endpoint_integration(self, async_client, mock_user, mock_db_session):
        """Test portfolio endpoints with database integration."""
        self.override_dependencies(mock_user, mock_db_session)
        
        try:
            # Mock portfolio data
            mock_portfolio = MagicMock(
                id=1,
                portfolio_id="portfolio-1",
                name="Test Portfolio",
                cash_balance=50000.0,
                strategy="balanced",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            mock_positions = [
                MagicMock(
                    id=1,
                    symbol="AAPL",
                    quantity=100,
                    average_cost=150.0,
                    realized_gain=1000.0
                ),
                MagicMock(
                    id=2,
                    symbol="GOOGL",
                    quantity=50,
                    average_cost=2800.0,
                    realized_gain=2000.0
                )
            ]
            
            mock_transactions = [
                MagicMock(
                    id=1,
                    symbol="AAPL",
                    transaction_type="buy",
                    quantity=100,
                    price=150.0,
                    total_amount=15000.0,
                    fees=10.0,
                    notes="Initial purchase",
                    created_at=datetime.utcnow()
                )
            ]
            
            with patch('backend.repositories.portfolio_repository.get_user_portfolios', return_value=[mock_portfolio]):
                with patch('backend.repositories.portfolio_repository.get_user_portfolio', return_value=mock_portfolio):
                    with patch('backend.repositories.portfolio_repository.get_portfolio_positions', return_value=mock_positions):
                        with patch('backend.repositories.portfolio_repository.get_recent_transactions', return_value=mock_transactions):
                            with patch('backend.repositories.stock_repository.get_by_symbol') as mock_stock:
                                mock_stock.return_value = MagicMock(
                                    name="Apple Inc.",
                                    sector="Technology"
                                )
                                
                                # Test portfolio summary
                                response = await async_client.get(
                                    "/api/portfolio/summary",
                                    headers={"Authorization": "Bearer test_token"}
                                )
                                
                                assert response.status_code == 200
                                data = response.json()
                                assert isinstance(data, list)
                                
                                if data:
                                    portfolio = data[0]
                                    assert "id" in portfolio
                                    assert "name" in portfolio
                                    assert "total_value" in portfolio
                                    assert "total_gain" in portfolio
                                    assert "positions_count" in portfolio
                                    assert "risk_score" in portfolio
                                
                                # Test portfolio detail
                                response = await async_client.get(
                                    "/api/portfolio/portfolio-1",
                                    headers={"Authorization": "Bearer test_token"}
                                )
                                
                                assert response.status_code == 200
                                data = response.json()
                                
                                # Verify detailed response structure
                                assert "positions" in data
                                assert "asset_allocation" in data
                                assert "sector_allocation" in data
                                assert "performance_metrics" in data
                                assert "recent_transactions" in data
                                
                                positions = data["positions"]
                                assert isinstance(positions, list)
                                
                                if positions:
                                    position = positions[0]
                                    assert "symbol" in position
                                    assert "quantity" in position
                                    assert "market_value" in position
                                    assert "unrealized_gain" in position
                                    
        finally:
            self.cleanup_dependencies()

    @pytest.mark.asyncio
    @pytest.mark.api
    async def test_stocks_endpoint_integration(self, async_client, mock_user, mock_db_session):
        """Test stocks endpoints with data retrieval."""
        self.override_dependencies(mock_user, mock_db_session)
        
        try:
            # Mock stock data
            mock_stocks = [
                MagicMock(
                    symbol="AAPL",
                    name="Apple Inc.",
                    sector="Technology",
                    industry="Consumer Electronics",
                    market_cap=3000000000000,
                    price=154.25,
                    change=2.15,
                    change_percent=1.41
                ),
                MagicMock(
                    symbol="GOOGL", 
                    name="Alphabet Inc.",
                    sector="Technology",
                    industry="Internet",
                    market_cap=2000000000000,
                    price=2850.50,
                    change=-15.25,
                    change_percent=-0.53
                )
            ]
            
            with patch('backend.repositories.stock_repository.search_stocks', return_value=mock_stocks):
                with patch('backend.repositories.stock_repository.get_by_symbol', return_value=mock_stocks[0]):
                    # Test stock search
                    response = await async_client.get(
                        "/api/stocks/search?query=apple&limit=10",
                        headers={"Authorization": "Bearer test_token"}
                    )
                    
                    assert response.status_code == 200
                    data = response.json()
                    assert isinstance(data, list)
                    
                    if data:
                        stock = data[0]
                        assert "symbol" in stock
                        assert "name" in stock
                        assert "sector" in stock
                        assert "market_cap" in stock
                    
                    # Test individual stock data
                    response = await async_client.get(
                        "/api/stocks/AAPL",
                        headers={"Authorization": "Bearer test_token"}
                    )
                    
                    assert response.status_code == 200
                    data = response.json()
                    assert data["symbol"] == "AAPL"
                    assert "name" in data
                    assert "current_price" in data
                    assert "market_cap" in data
                    
        finally:
            self.cleanup_dependencies()

    @pytest.mark.asyncio
    @pytest.mark.api
    async def test_analysis_endpoint_integration(self, async_client, mock_user, mock_db_session):
        """Test analysis endpoints with ML model integration."""
        self.override_dependencies(mock_user, mock_db_session)
        
        try:
            # Mock price history for technical analysis
            mock_price_data = [
                MagicMock(
                    date=date.today() - timedelta(days=i),
                    open=150.0 + i * 0.5,
                    high=155.0 + i * 0.5,
                    low=149.0 + i * 0.5,
                    close=154.0 + i * 0.5,
                    volume=1000000 + i * 10000
                )
                for i in range(100)
            ]
            
            mock_stock = MagicMock(
                symbol="AAPL",
                name="Apple Inc.",
                sector="Technology",
                market_cap=3000000000000
            )
            
            with patch('backend.repositories.price_repository.get_price_history', return_value=mock_price_data):
                with patch('backend.repositories.stock_repository.get_by_symbol', return_value=mock_stock):
                    with patch('backend.analytics.technical_analysis.TechnicalAnalysis') as mock_tech:
                        # Mock technical analysis results
                        mock_tech_instance = AsyncMock()
                        mock_tech_instance.analyze_stock.return_value = {
                            'indicators': {
                                'rsi': 65.5,
                                'macd': {'macd': 2.5, 'signal': 2.0, 'histogram': 0.5},
                                'bollinger_bands': {'upper': 160, 'middle': 154, 'lower': 148},
                                'moving_averages': {'sma_20': 152.5, 'sma_50': 150.0, 'ema_20': 153.0}
                            },
                            'signals': {
                                'trend': 'bullish',
                                'momentum': 'positive',
                                'support_levels': [148, 145, 142],
                                'resistance_levels': [158, 162, 165]
                            },
                            'score': 7.5,
                            'recommendation': 'buy'
                        }
                        mock_tech.return_value = mock_tech_instance
                        
                        response = await async_client.get(
                            "/api/analysis/technical/AAPL",
                            headers={"Authorization": "Bearer test_token"}
                        )
                        
                        assert response.status_code == 200
                        data = response.json()
                        
                        # Verify analysis structure
                        assert "symbol" in data
                        assert "indicators" in data
                        assert "signals" in data
                        assert "score" in data
                        assert "recommendation" in data
                        assert data["symbol"] == "AAPL"
                        
                        # Verify technical indicators
                        indicators = data["indicators"]
                        assert "rsi" in indicators
                        assert "macd" in indicators
                        assert "bollinger_bands" in indicators
                        assert "moving_averages" in indicators
                        
        finally:
            self.cleanup_dependencies()

    @pytest.mark.asyncio
    @pytest.mark.api
    async def test_api_error_handling(self, async_client, mock_user, mock_db_session):
        """Test API error handling and resilience."""
        self.override_dependencies(mock_user, mock_db_session)
        
        try:
            # Test 404 error
            response = await async_client.get(
                "/api/stocks/NONEXISTENT",
                headers={"Authorization": "Bearer test_token"}
            )
            assert response.status_code == 404
            
            # Test validation error
            response = await async_client.post(
                "/api/portfolio/test-portfolio/positions",
                headers={"Authorization": "Bearer test_token"},
                json={"symbol": "", "quantity": -1}  # Invalid data
            )
            assert response.status_code == 422
            
            # Test database error simulation
            with patch('backend.repositories.stock_repository.get_by_symbol', side_effect=Exception("DB Error")):
                response = await async_client.get(
                    "/api/stocks/AAPL",
                    headers={"Authorization": "Bearer test_token"}
                )
                assert response.status_code == 500
                
                error_data = response.json()
                assert "error" in error_data
                assert "timestamp" in error_data
                
        finally:
            self.cleanup_dependencies()

    @pytest.mark.asyncio
    @pytest.mark.api
    async def test_api_pagination_and_filtering(self, async_client, mock_user, mock_db_session):
        """Test API pagination and filtering functionality."""
        self.override_dependencies(mock_user, mock_db_session)
        
        try:
            # Mock large dataset
            mock_recommendations = [
                {
                    "id": f"rec-{i}",
                    "symbol": f"STOCK{i}",
                    "company_name": f"Company {i}",
                    "recommendation_type": "buy" if i % 2 == 0 else "hold",
                    "category": "growth" if i % 3 == 0 else "value",
                    "confidence_score": 0.8 - (i * 0.01),
                    "target_price": 100 + i,
                    "current_price": 90 + i,
                    "expected_return": 0.1 + (i * 0.001),
                    "time_horizon": "medium_term",
                    "risk_level": "moderate",
                    "created_at": datetime.utcnow(),
                    "valid_until": datetime.utcnow() + timedelta(days=30),
                    "reasoning": f"Analysis for stock {i}",
                    "key_factors": ["Factor 1", "Factor 2"],
                    "technical_signals": {"rsi": 60 + i},
                    "fundamental_metrics": {"pe_ratio": 15 + i},
                    "risk_factors": ["Risk 1"],
                    "entry_points": [90 + i, 85 + i],
                    "exit_points": [100 + i, 105 + i],
                    "stop_loss": 80 + i,
                    "sector": "Technology",
                    "market_cap": 1000000000 + i * 100000000,
                    "volume": 1000000 + i * 10000,
                    "analyst_consensus": "Buy",
                    "similar_stocks": []
                }
                for i in range(50)
            ]
            
            with patch('backend.api.routers.recommendations.generate_recommendation') as mock_gen:
                mock_gen.side_effect = lambda: mock_recommendations.pop(0) if mock_recommendations else None
                
                # Test pagination
                response = await async_client.get(
                    "/api/recommendations/list?limit=10&offset=0",
                    headers={"Authorization": "Bearer test_token"}
                )
                
                assert response.status_code == 200
                data = response.json()
                assert isinstance(data, list)
                assert len(data) <= 10
                
                # Test filtering
                response = await async_client.get(
                    "/api/recommendations/list?recommendation_type=buy&min_confidence=0.7",
                    headers={"Authorization": "Bearer test_token"}
                )
                
                assert response.status_code == 200
                data = response.json()
                
                # Verify filtering works
                for item in data:
                    assert item["recommendation_type"] == "buy"
                    assert item["confidence_score"] >= 0.7
                    
        finally:
            self.cleanup_dependencies()

    @pytest.mark.asyncio
    @pytest.mark.api
    async def test_api_performance_under_load(self, async_client, mock_user, mock_db_session):
        """Test API performance under concurrent load."""
        self.override_dependencies(mock_user, mock_db_session)
        
        try:
            # Mock fast responses
            with patch('backend.repositories.stock_repository.search_stocks') as mock_search:
                mock_search.return_value = [
                    MagicMock(symbol="AAPL", name="Apple Inc.", sector="Technology")
                ]
                
                # Simulate concurrent requests
                tasks = []
                for i in range(20):  # 20 concurrent requests
                    task = async_client.get(
                        f"/api/stocks/search?query=apple&page={i}",
                        headers={"Authorization": "Bearer test_token"}
                    )
                    tasks.append(task)
                
                # Execute all requests concurrently
                start_time = datetime.utcnow()
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                end_time = datetime.utcnow()
                
                # Verify performance
                duration = (end_time - start_time).total_seconds()
                assert duration < 10.0, f"Concurrent requests took {duration}s, should be under 10s"
                
                # Verify all requests succeeded
                successful_responses = [r for r in responses if hasattr(r, 'status_code') and r.status_code == 200]
                assert len(successful_responses) >= 18, f"Only {len(successful_responses)}/20 requests succeeded"
                
        finally:
            self.cleanup_dependencies()

    @pytest.mark.asyncio
    @pytest.mark.api
    async def test_api_caching_integration(self, async_client, mock_user, mock_db_session):
        """Test API response caching integration."""
        self.override_dependencies(mock_user, mock_db_session)
        
        try:
            with patch('backend.utils.cache.cache_with_ttl') as mock_cache:
                # Mock cache decorator behavior
                call_count = {"count": 0}
                
                def cache_side_effect(ttl):
                    def decorator(func):
                        async def wrapper(*args, **kwargs):
                            call_count["count"] += 1
                            return await func(*args, **kwargs)
                        return wrapper
                    return decorator
                
                mock_cache.side_effect = cache_side_effect
                
                with patch('backend.repositories.portfolio_repository.get_user_portfolios') as mock_portfolios:
                    mock_portfolios.return_value = [
                        MagicMock(
                            id=1,
                            name="Test Portfolio",
                            cash_balance=10000,
                            strategy="balanced",
                            created_at=datetime.utcnow(),
                            updated_at=datetime.utcnow()
                        )
                    ]
                    
                    # First request - should hit the function
                    response1 = await async_client.get(
                        "/api/portfolio/summary",
                        headers={"Authorization": "Bearer test_token"}
                    )
                    
                    # Second request - should use cache (in real scenario)
                    response2 = await async_client.get(
                        "/api/portfolio/summary",
                        headers={"Authorization": "Bearer test_token"}
                    )
                    
                    assert response1.status_code == 200
                    assert response2.status_code == 200
                    
                    # Verify both requests return same data
                    assert response1.json() == response2.json()
                    
        finally:
            self.cleanup_dependencies()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])