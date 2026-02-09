"""
Tests for Service Layer
Verify that service layer classes are properly structured and importable.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.services import (
    RecommendationService,
    recommendation_service,
    PortfolioService,
    portfolio_service,
    AnalysisService,
    analysis_service,
    TradingService,
    trading_service,
)


class TestRecommendationService:
    """Test RecommendationService"""

    def test_service_instance(self):
        """Test that service instance is properly created"""
        assert isinstance(recommendation_service, RecommendationService)
        assert recommendation_service.recommendation_engine is not None
        assert recommendation_service.fundamental_engine is not None

    def test_calculate_confidence_empty(self):
        """Test confidence calculation with empty input"""
        service = RecommendationService()
        confidence = service.calculate_confidence([])
        assert confidence == 0.0

    def test_calculate_confidence_weighted(self):
        """Test confidence calculation with weighted analyses"""
        service = RecommendationService()
        analyses = [
            {'type': 'technical', 'confidence': 0.8},
            {'type': 'fundamental', 'confidence': 0.9},
            {'type': 'sentiment', 'confidence': 0.7},
        ]
        confidence = service.calculate_confidence(analyses)
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.7  # Should be weighted average around 0.8

    @pytest.mark.asyncio
    async def test_get_trending_returns_list(self):
        """Test that get_trending returns a list"""
        service = RecommendationService()

        # Mock the recommendation engine
        with patch.object(
            service.recommendation_engine,
            'generate_daily_recommendations',
            new_callable=AsyncMock
        ) as mock_generate:
            mock_generate.return_value = []

            result = await service.get_trending(limit=5)
            assert isinstance(result, list)


class TestPortfolioService:
    """Test PortfolioService"""

    def test_service_instance(self):
        """Test that service instance is properly created"""
        assert isinstance(portfolio_service, PortfolioService)
        assert portfolio_service.repository is not None

    @pytest.mark.asyncio
    async def test_get_allocation_empty_portfolio(self):
        """Test allocation for non-existent portfolio"""
        service = PortfolioService()

        # Mock repository to return None
        with patch.object(
            service.repository,
            'get_portfolio_allocation',
            new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = None

            result = await service.get_allocation(portfolio_id=999)
            assert isinstance(result, dict)
            assert result['portfolio_id'] == 999
            assert result['cash_allocation_pct'] == 100

    @pytest.mark.asyncio
    async def test_get_transactions_returns_list(self):
        """Test that get_transactions returns a list"""
        service = PortfolioService()

        # Mock repository
        with patch.object(
            service.repository,
            'get_portfolio_transactions',
            new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = []

            result = await service.get_transactions(portfolio_id=1)
            assert isinstance(result, list)


class TestAnalysisService:
    """Test AnalysisService"""

    def test_service_instance(self):
        """Test that service instance is properly created"""
        assert isinstance(analysis_service, AnalysisService)
        assert analysis_service.fundamental_engine is not None
        assert analysis_service.technical_engine is not None
        assert analysis_service.sentiment_engine is not None

    def test_cache_key_generation(self):
        """Test cache key generation"""
        service = AnalysisService()
        key = service._get_cache_key('AAPL', ['technical', 'fundamental'], 'standard')
        assert 'AAPL' in key
        assert 'technical' in key or 'fundamental' in key
        assert 'standard' in key

    def test_composite_score_empty(self):
        """Test composite score calculation with no analyses"""
        service = AnalysisService()
        score = service._calculate_composite_score({})
        assert score == 0.0

    def test_composite_score_with_analyses(self):
        """Test composite score calculation"""
        service = AnalysisService()
        analyses = {
            'technical': {'composite_score': 80},
            'fundamental': {'composite_score': 70},
        }
        score = service._calculate_composite_score(analyses)
        assert 0.0 <= score <= 100.0

    @pytest.mark.asyncio
    async def test_get_cached_analysis_empty(self):
        """Test getting cached analysis when cache is empty"""
        service = AnalysisService()
        result = await service.get_cached_analysis('AAPL')
        assert result is None

    def test_clear_cache(self):
        """Test cache clearing"""
        service = AnalysisService()

        # Add something to cache
        service._cache['TEST:technical:standard'] = {
            'data': {},
            'cached_at': MagicMock()
        }

        # Clear all
        service.clear_cache()
        assert len(service._cache) == 0

    def test_clear_cache_specific_ticker(self):
        """Test clearing cache for specific ticker"""
        service = AnalysisService()

        # Add multiple entries
        service._cache['AAPL:technical:standard'] = {'data': {}, 'cached_at': MagicMock()}
        service._cache['MSFT:technical:standard'] = {'data': {}, 'cached_at': MagicMock()}

        # Clear only AAPL
        service.clear_cache('AAPL')
        assert len(service._cache) == 1
        assert 'MSFT:technical:standard' in service._cache


class TestServiceIntegration:
    """Integration tests for service layer"""

    def test_all_services_importable(self):
        """Test that all services can be imported"""
        from backend.services import (
            RecommendationService,
            PortfolioService,
            AnalysisService,
        )

        assert RecommendationService is not None
        assert PortfolioService is not None
        assert AnalysisService is not None

    def test_singleton_instances_exist(self):
        """Test that singleton instances are created"""
        from backend.services import (
            recommendation_service,
            portfolio_service,
            analysis_service,
        )

        assert recommendation_service is not None
        assert portfolio_service is not None
        assert analysis_service is not None

    def test_services_have_required_methods(self):
        """Test that services have required public methods"""
        # RecommendationService
        assert hasattr(recommendation_service, 'generate_recommendation')
        assert hasattr(recommendation_service, 'get_trending')
        assert hasattr(recommendation_service, 'calculate_confidence')

        # PortfolioService
        assert hasattr(portfolio_service, 'get_portfolio_summary')
        assert hasattr(portfolio_service, 'add_position')
        assert hasattr(portfolio_service, 'get_allocation')

        # AnalysisService
        assert hasattr(analysis_service, 'run_analysis')
        assert hasattr(analysis_service, 'get_cached_analysis')
        assert hasattr(analysis_service, 'compare_stocks')


class TestTradingService:
    """Test TradingService"""

    def test_service_instance(self):
        """Test that service instance is properly created"""
        assert isinstance(trading_service, TradingService)
        assert trading_service.repository is not None

    @pytest.mark.asyncio
    async def test_validate_order_missing_fields(self):
        """Test order validation with missing required fields"""
        service = TradingService()

        result = await service.validate_order({})
        assert result['valid'] is False
        assert 'errors' in result
        assert len(result['errors']) > 0

    @pytest.mark.asyncio
    async def test_validate_order_invalid_quantity(self):
        """Test order validation with invalid quantity"""
        service = TradingService()

        # Mock repository to avoid database calls
        with patch.object(
            service.repository,
            'get_by_id',
            new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = None

            order_data = {
                'portfolio_id': 1,
                'symbol': 'AAPL',
                'side': 'buy',
                'order_type': 'market',
                'quantity': -10
            }

            result = await service.validate_order(order_data)
            assert result['valid'] is False
            assert any('quantity' in err.lower() for err in result['errors'])

    @pytest.mark.asyncio
    async def test_validate_order_limit_without_price(self):
        """Test limit order validation without price"""
        service = TradingService()

        # Mock repository to avoid database calls
        with patch.object(
            service.repository,
            'get_by_id',
            new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = None

            order_data = {
                'portfolio_id': 1,
                'symbol': 'AAPL',
                'side': 'buy',
                'order_type': 'limit',
                'quantity': 10
            }

            result = await service.validate_order(order_data)
            assert result['valid'] is False
            assert any('price' in err.lower() for err in result['errors'])

    @pytest.mark.asyncio
    async def test_validate_order_valid_market_order(self):
        """Test valid market order validation"""
        service = TradingService()

        # Mock repository to return a portfolio
        with patch.object(
            service.repository,
            'get_by_id',
            new_callable=AsyncMock
        ) as mock_get:
            from decimal import Decimal
            mock_portfolio = MagicMock()
            mock_portfolio.cash_balance = Decimal('10000.00')
            mock_get.return_value = mock_portfolio

            order_data = {
                'portfolio_id': 1,
                'symbol': 'AAPL',
                'side': 'buy',
                'order_type': 'market',
                'quantity': 10,
                'price': 150.00
            }

            result = await service.validate_order(order_data)
            assert result['valid'] is True

    @pytest.mark.asyncio
    async def test_execute_trade_returns_result(self):
        """Test that execute_trade returns a result dictionary"""
        service = TradingService()

        # Mock repository methods
        with patch.object(
            service.repository,
            'get_by_id',
            new_callable=AsyncMock
        ) as mock_get, \
        patch.object(
            service.repository,
            'add_position',
            new_callable=AsyncMock
        ) as mock_add:
            from decimal import Decimal

            mock_portfolio = MagicMock()
            mock_portfolio.cash_balance = Decimal('10000.00')
            mock_get.return_value = mock_portfolio

            mock_position = MagicMock()
            mock_position.id = 1
            mock_add.return_value = mock_position

            order = {
                'symbol': 'AAPL',
                'side': 'buy',
                'order_type': 'market',
                'quantity': 10,
                'price': 150.00
            }

            result = await service.execute_trade(portfolio_id=1, order=order)
            assert isinstance(result, dict)
            assert 'success' in result

    @pytest.mark.asyncio
    async def test_calculate_portfolio_impact_returns_dict(self):
        """Test that calculate_portfolio_impact returns impact analysis"""
        service = TradingService()

        # Mock repository methods
        with patch.object(
            service.repository,
            'get_by_id',
            new_callable=AsyncMock
        ) as mock_get, \
        patch.object(
            service.repository,
            'get_portfolio_allocation',
            new_callable=AsyncMock
        ) as mock_alloc, \
        patch.object(
            service.repository,
            'calculate_portfolio_value',
            new_callable=AsyncMock
        ) as mock_value:
            from decimal import Decimal

            mock_portfolio = MagicMock()
            mock_portfolio.cash_balance = Decimal('10000.00')
            mock_get.return_value = mock_portfolio

            mock_alloc.return_value = {}
            mock_value.return_value = {'total_value': 50000.0}

            trade = {
                'symbol': 'AAPL',
                'side': 'buy',
                'quantity': 10,
                'price': 150.00
            }

            result = await service.calculate_portfolio_impact(portfolio_id=1, trade=trade)
            assert isinstance(result, dict)
            assert 'success' in result
            if result['success']:
                assert 'trade_impact' in result
                assert 'before' in result
                assert 'after' in result
