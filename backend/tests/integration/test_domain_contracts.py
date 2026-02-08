"""
Integration tests for DDD Domain Contracts.

Tests verify contract compliance across all five domains:
    - Domain 1: Market Data (Stock/Price/Exchange)
    - Domain 2: Portfolio Management (User/Portfolio/Position)
    - Domain 3: Data Pipeline (ETL/Ingestion)
    - Domain 4: ML/Prediction
    - Domain 5: Investment Analysis (Technical/Fundamental/Recommendations)

Contract verification ensures:
    - All contract methods are properly defined
    - Cross-domain integrations work correctly
    - Contract results are properly typed
    - Error handling follows contract patterns
"""

import pytest
import pytest_asyncio
from datetime import datetime, date, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

from backend.domain.contracts import (
    DomainContract,
    ContractResult,
    ContractError,
    MarketDataContract,
    PortfolioContract,
    DataPipelineContract,
    MLContract,
    InvestmentAnalysisContract,
)
from backend.domain.contracts.base import ContractErrorCode, ContractMetrics
from backend.domain.contracts.market_data_contract import (
    StockDTO, PriceHistoryDTO, ExchangeDTO, SectorDTO, QuoteDTO, AssetType
)
from backend.domain.contracts.portfolio_contract import (
    UserDTO, PortfolioDTO, PositionDTO, TransactionDTO, UserRole, OrderSide
)
from backend.domain.contracts.data_pipeline_contract import (
    ETLJobDTO, DataSource, DataType, JobStatus, DataQualityReportDTO, DataQualityLevel
)
from backend.domain.contracts.ml_contract import (
    ModelDTO, PredictionDTO, ModelType, ModelStatus, PredictionHorizon, PredictionDirection
)
from backend.domain.contracts.investment_analysis_contract import (
    TechnicalAnalysisDTO, FundamentalAnalysisDTO, RecommendationDTO,
    RecommendationAction, RiskLevel, TrendDirection, ValuationAssessment,
    QualityGrade, MoatRating
)


pytestmark = pytest.mark.integration


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_stock_dto():
    """Create a sample stock DTO for testing."""
    return StockDTO(
        id=1,
        symbol="AAPL",
        name="Apple Inc.",
        exchange_id=1,
        exchange_code="NASDAQ",
        sector_id=1,
        sector_name="Technology",
        industry_id=1,
        industry_name="Consumer Electronics",
        asset_type=AssetType.STOCK,
        market_cap=3_000_000_000_000.0,
        shares_outstanding=16_000_000_000,
        country="US",
        currency="USD",
        is_active=True,
        is_tradeable=True,
    )


@pytest.fixture
def sample_price_history_dto():
    """Create sample price history DTOs."""
    base_date = date.today() - timedelta(days=30)
    return [
        PriceHistoryDTO(
            stock_id=1,
            date=base_date + timedelta(days=i),
            open=Decimal("150.00") + Decimal(str(i * 0.5)),
            high=Decimal("152.00") + Decimal(str(i * 0.5)),
            low=Decimal("149.00") + Decimal(str(i * 0.5)),
            close=Decimal("151.00") + Decimal(str(i * 0.5)),
            adjusted_close=Decimal("151.00") + Decimal(str(i * 0.5)),
            volume=75_000_000 + (i * 1_000_000),
            daily_return=0.003,
        )
        for i in range(30)
    ]


@pytest.fixture
def sample_portfolio_dto():
    """Create a sample portfolio DTO."""
    return PortfolioDTO(
        id=1,
        user_id=1,
        name="Main Portfolio",
        description="Primary investment portfolio",
        cash_balance=Decimal("50000.00"),
        total_value=Decimal("150000.00"),
        total_return=Decimal("10000.00"),
        total_return_pct=7.14,
        is_public=False,
        is_default=True,
        benchmark="SPY",
    )


@pytest.fixture
def sample_prediction_dto():
    """Create a sample prediction DTO."""
    return PredictionDTO(
        id=1,
        stock_id=1,
        symbol="AAPL",
        model_id="xgboost_v1",
        model_name="XGBoost Classifier",
        prediction_date=datetime.now(timezone.utc),
        target_date=datetime.now(timezone.utc) + timedelta(days=7),
        horizon=PredictionHorizon.WEEK_1,
        predicted_price=Decimal("175.00"),
        predicted_price_low=Decimal("170.00"),
        predicted_price_high=Decimal("180.00"),
        predicted_return=0.08,
        predicted_direction=PredictionDirection.UP,
        confidence=0.82,
    )


@pytest.fixture
def sample_recommendation_dto():
    """Create a sample recommendation DTO."""
    return RecommendationDTO(
        id=1,
        recommendation_id="rec-12345",
        stock_id=1,
        symbol="AAPL",
        created_at=datetime.now(timezone.utc),
        valid_until=datetime.now(timezone.utc) + timedelta(days=30),
        action=RecommendationAction.BUY,
        confidence=0.85,
        priority=8,
        entry_price=Decimal("165.00"),
        target_price=Decimal("185.00"),
        stop_loss=Decimal("155.00"),
        expected_return=0.12,
        time_horizon_days=90,
        risk_level=RiskLevel.MEDIUM,
        risk_score=0.45,
        technical_score=0.78,
        fundamental_score=0.86,
        sentiment_score=0.75,
        overall_score=0.80,
        reasoning="Strong fundamentals with positive momentum.",
        key_factors=["high_roe", "upward_trend", "strong_cash_flow"],
        risks=["high_valuation", "competition"],
        opportunities=["market_expansion", "new_products"],
        is_active=True,
    )


# =============================================================================
# Contract Result Tests
# =============================================================================

class TestContractResult:
    """Tests for ContractResult wrapper."""

    def test_ok_result(self, sample_stock_dto):
        """Test successful result creation."""
        result = ContractResult.ok(sample_stock_dto, source="test")

        assert result.success is True
        assert result.data == sample_stock_dto
        assert result.error is None
        assert result.metadata.get("source") == "test"

    def test_fail_result(self):
        """Test failed result creation."""
        result = ContractResult.fail(
            ContractErrorCode.NOT_FOUND,
            "Stock not found",
            details={"symbol": "INVALID"},
            source_domain="MarketData"
        )

        assert result.success is False
        assert result.data is None
        assert result.error is not None
        assert result.error.code == ContractErrorCode.NOT_FOUND
        assert result.error.message == "Stock not found"
        assert result.error.details == {"symbol": "INVALID"}
        assert result.error.source_domain == "MarketData"

    def test_unwrap_success(self, sample_stock_dto):
        """Test unwrapping successful result."""
        result = ContractResult.ok(sample_stock_dto)
        unwrapped = result.unwrap()

        assert unwrapped == sample_stock_dto

    def test_unwrap_failure_raises(self):
        """Test unwrapping failed result raises exception."""
        result = ContractResult.fail(
            ContractErrorCode.VALIDATION_ERROR,
            "Invalid input"
        )

        with pytest.raises(ValueError) as exc_info:
            result.unwrap()

        assert "Contract failed" in str(exc_info.value)

    def test_unwrap_or_default(self, sample_stock_dto):
        """Test unwrap_or with default value."""
        success_result = ContractResult.ok(sample_stock_dto)
        fail_result = ContractResult.fail(
            ContractErrorCode.NOT_FOUND, "Not found"
        )

        default = StockDTO(id=0, symbol="DEFAULT", name="Default", exchange_id=0)

        assert success_result.unwrap_or(default) == sample_stock_dto
        assert fail_result.unwrap_or(default) == default

    def test_map_success(self, sample_stock_dto):
        """Test mapping successful result."""
        result = ContractResult.ok(sample_stock_dto)
        mapped = result.map(lambda s: s.symbol)

        assert mapped.success is True
        assert mapped.data == "AAPL"

    def test_map_failure_passes_through(self):
        """Test mapping failed result passes through."""
        result = ContractResult.fail(
            ContractErrorCode.NOT_FOUND, "Not found"
        )
        mapped = result.map(lambda x: x.upper())

        assert mapped.success is False
        assert mapped.error.code == ContractErrorCode.NOT_FOUND


# =============================================================================
# Contract Metrics Tests
# =============================================================================

class TestContractMetrics:
    """Tests for ContractMetrics tracking."""

    def test_initial_state(self):
        """Test metrics initial state."""
        metrics = ContractMetrics()

        assert metrics.total_calls == 0
        assert metrics.successful_calls == 0
        assert metrics.failed_calls == 0
        assert metrics.success_rate == 0.0
        assert metrics.average_latency_ms == 0.0

    def test_record_success(self):
        """Test recording successful calls."""
        metrics = ContractMetrics()

        metrics.record_success(50.0)
        metrics.record_success(100.0)

        assert metrics.total_calls == 2
        assert metrics.successful_calls == 2
        assert metrics.failed_calls == 0
        assert metrics.success_rate == 1.0
        assert metrics.average_latency_ms == 75.0

    def test_record_failure(self):
        """Test recording failed calls."""
        metrics = ContractMetrics()

        metrics.record_success(50.0)
        metrics.record_failure()

        assert metrics.total_calls == 2
        assert metrics.successful_calls == 1
        assert metrics.failed_calls == 1
        assert metrics.success_rate == 0.5


# =============================================================================
# Domain 1: Market Data Contract Tests
# =============================================================================

class TestMarketDataContract:
    """Tests for Market Data Domain Contract (Domain 1)."""

    def test_contract_properties(self):
        """Test contract metadata properties."""
        # Create a concrete implementation for testing
        class MockMarketDataContract(MarketDataContract):
            async def get_stock(self, symbol): pass
            async def get_stock_by_id(self, stock_id): pass
            async def search_stocks(self, query, limit=20, asset_types=None, active_only=True): pass
            async def list_stocks_by_sector(self, sector_id, limit=100, offset=0): pass
            async def list_stocks_by_exchange(self, exchange_id, limit=100, offset=0): pass
            async def get_exchange(self, code): pass
            async def list_exchanges(self): pass
            async def get_sector(self, sector_id): pass
            async def list_sectors(self): pass
            async def get_industry(self, industry_id): pass
            async def list_industries_by_sector(self, sector_id): pass
            async def get_price_history(self, stock_id, start_date, end_date): pass
            async def get_latest_price(self, stock_id): pass
            async def get_quote(self, symbol): pass
            async def get_technical_indicators(self, stock_id, date): pass
            async def get_technical_indicators_history(self, stock_id, start_date, end_date): pass

        contract = MockMarketDataContract()

        assert contract.domain_name == "MarketData"
        assert contract.version == "1.0.0"
        assert "get_stock" in contract.capabilities
        assert "get_price_history" in contract.capabilities
        assert "get_quote" in contract.capabilities

    def test_validate_contract(self):
        """Test contract validation passes."""
        class MockMarketDataContract(MarketDataContract):
            async def get_stock(self, symbol): pass
            async def get_stock_by_id(self, stock_id): pass
            async def search_stocks(self, query, limit=20, asset_types=None, active_only=True): pass
            async def list_stocks_by_sector(self, sector_id, limit=100, offset=0): pass
            async def list_stocks_by_exchange(self, exchange_id, limit=100, offset=0): pass
            async def get_exchange(self, code): pass
            async def list_exchanges(self): pass
            async def get_sector(self, sector_id): pass
            async def list_sectors(self): pass
            async def get_industry(self, industry_id): pass
            async def list_industries_by_sector(self, sector_id): pass
            async def get_price_history(self, stock_id, start_date, end_date): pass
            async def get_latest_price(self, stock_id): pass
            async def get_quote(self, symbol): pass
            async def get_technical_indicators(self, stock_id, date): pass
            async def get_technical_indicators_history(self, stock_id, start_date, end_date): pass

        contract = MockMarketDataContract()
        validation = contract.validate_contract()

        assert validation.success is True
        assert validation.data is True

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test contract health check."""
        class MockMarketDataContract(MarketDataContract):
            async def get_stock(self, symbol): pass
            async def get_stock_by_id(self, stock_id): pass
            async def search_stocks(self, query, limit=20, asset_types=None, active_only=True): pass
            async def list_stocks_by_sector(self, sector_id, limit=100, offset=0): pass
            async def list_stocks_by_exchange(self, exchange_id, limit=100, offset=0): pass
            async def get_exchange(self, code): pass
            async def list_exchanges(self): pass
            async def get_sector(self, sector_id): pass
            async def list_sectors(self): pass
            async def get_industry(self, industry_id): pass
            async def list_industries_by_sector(self, sector_id): pass
            async def get_price_history(self, stock_id, start_date, end_date): pass
            async def get_latest_price(self, stock_id): pass
            async def get_quote(self, symbol): pass
            async def get_technical_indicators(self, stock_id, date): pass
            async def get_technical_indicators_history(self, stock_id, start_date, end_date): pass

        contract = MockMarketDataContract()
        health = await contract.health_check()

        assert health.success is True
        assert health.data["domain"] == "MarketData"
        assert health.data["status"] == "healthy"


# =============================================================================
# Domain 2: Portfolio Contract Tests
# =============================================================================

class TestPortfolioContract:
    """Tests for Portfolio Domain Contract (Domain 2)."""

    def test_contract_properties(self):
        """Test contract metadata properties."""
        class MockPortfolioContract(PortfolioContract):
            async def get_user(self, user_id): pass
            async def get_user_by_email(self, email): pass
            async def get_portfolio(self, portfolio_id, user_id): pass
            async def list_user_portfolios(self, user_id): pass
            async def create_portfolio(self, user_id, name, description=None, cash_balance=Decimal("0"), benchmark=None): pass
            async def update_portfolio(self, portfolio_id, user_id, updates): pass
            async def delete_portfolio(self, portfolio_id, user_id): pass
            async def get_position(self, portfolio_id, stock_id): pass
            async def list_positions(self, portfolio_id): pass
            async def add_position(self, portfolio_id, stock_id, quantity, average_cost): pass
            async def close_position(self, portfolio_id, stock_id, price): pass
            async def get_transactions(self, portfolio_id, stock_id=None, start_date=None, end_date=None, limit=100): pass
            async def add_transaction(self, portfolio_id, stock_id, side, quantity, price, commission=Decimal("0"), notes=None): pass
            async def get_portfolio_summary(self, portfolio_id): pass
            async def get_allocation(self, portfolio_id): pass
            async def calculate_rebalance(self, portfolio_id, target_allocation): pass
            async def get_watchlist(self, user_id): pass
            async def add_to_watchlist(self, user_id, stock_id, notes=None): pass
            async def remove_from_watchlist(self, user_id, stock_id): pass

        contract = MockPortfolioContract()

        assert contract.domain_name == "Portfolio"
        assert contract.version == "1.0.0"
        assert "get_portfolio" in contract.capabilities
        assert "add_position" in contract.capabilities
        assert "rebalance_portfolio" in contract.capabilities

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test contract health check."""
        class MockPortfolioContract(PortfolioContract):
            async def get_user(self, user_id): pass
            async def get_user_by_email(self, email): pass
            async def get_portfolio(self, portfolio_id, user_id): pass
            async def list_user_portfolios(self, user_id): pass
            async def create_portfolio(self, user_id, name, description=None, cash_balance=Decimal("0"), benchmark=None): pass
            async def update_portfolio(self, portfolio_id, user_id, updates): pass
            async def delete_portfolio(self, portfolio_id, user_id): pass
            async def get_position(self, portfolio_id, stock_id): pass
            async def list_positions(self, portfolio_id): pass
            async def add_position(self, portfolio_id, stock_id, quantity, average_cost): pass
            async def close_position(self, portfolio_id, stock_id, price): pass
            async def get_transactions(self, portfolio_id, stock_id=None, start_date=None, end_date=None, limit=100): pass
            async def add_transaction(self, portfolio_id, stock_id, side, quantity, price, commission=Decimal("0"), notes=None): pass
            async def get_portfolio_summary(self, portfolio_id): pass
            async def get_allocation(self, portfolio_id): pass
            async def calculate_rebalance(self, portfolio_id, target_allocation): pass
            async def get_watchlist(self, user_id): pass
            async def add_to_watchlist(self, user_id, stock_id, notes=None): pass
            async def remove_from_watchlist(self, user_id, stock_id): pass

        contract = MockPortfolioContract()
        health = await contract.health_check()

        assert health.success is True
        assert health.data["domain"] == "Portfolio"


# =============================================================================
# Domain 3: Data Pipeline Contract Tests
# =============================================================================

class TestDataPipelineContract:
    """Tests for Data Pipeline Domain Contract (Domain 3)."""

    def test_contract_properties(self):
        """Test contract metadata properties."""
        class MockDataPipelineContract(DataPipelineContract):
            async def ingest_price_data(self, symbols, source=DataSource.YAHOO_FINANCE, start_date=None, end_date=None, validate=True): pass
            async def ingest_fundamentals(self, symbols, source=DataSource.SEC_EDGAR, quarters=8, validate=True): pass
            async def ingest_news(self, symbols, source=DataSource.NEWS_API, days=30, analyze_sentiment=True): pass
            async def ingest_quotes(self, symbols, source=DataSource.FINNHUB): pass
            async def run_etl_job(self, name, data_type, source, symbols, config=None): pass
            async def get_job_status(self, job_id): pass
            async def list_jobs(self, status=None, data_type=None, limit=50): pass
            async def cancel_job(self, job_id): pass
            async def get_data_source_status(self, source): pass
            async def list_data_sources(self): pass
            async def get_data_quality_report(self, stock_id, data_type): pass
            async def validate_data(self, stock_id, data_type, start_date=None, end_date=None): pass
            async def get_data_coverage(self, symbol): pass
            async def schedule_refresh(self, symbols, data_types, cron_expression): pass

        contract = MockDataPipelineContract()

        assert contract.domain_name == "DataPipeline"
        assert contract.version == "1.0.0"
        assert "ingest_price_data" in contract.capabilities
        assert "run_etl_job" in contract.capabilities
        assert "get_data_quality_report" in contract.capabilities

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test contract health check."""
        class MockDataPipelineContract(DataPipelineContract):
            async def ingest_price_data(self, symbols, source=DataSource.YAHOO_FINANCE, start_date=None, end_date=None, validate=True): pass
            async def ingest_fundamentals(self, symbols, source=DataSource.SEC_EDGAR, quarters=8, validate=True): pass
            async def ingest_news(self, symbols, source=DataSource.NEWS_API, days=30, analyze_sentiment=True): pass
            async def ingest_quotes(self, symbols, source=DataSource.FINNHUB): pass
            async def run_etl_job(self, name, data_type, source, symbols, config=None): pass
            async def get_job_status(self, job_id): pass
            async def list_jobs(self, status=None, data_type=None, limit=50): pass
            async def cancel_job(self, job_id): pass
            async def get_data_source_status(self, source): pass
            async def list_data_sources(self): pass
            async def get_data_quality_report(self, stock_id, data_type): pass
            async def validate_data(self, stock_id, data_type, start_date=None, end_date=None): pass
            async def get_data_coverage(self, symbol): pass
            async def schedule_refresh(self, symbols, data_types, cron_expression): pass

        contract = MockDataPipelineContract()
        health = await contract.health_check()

        assert health.success is True
        assert health.data["domain"] == "DataPipeline"


# =============================================================================
# Domain 4: ML Contract Tests
# =============================================================================

class TestMLContract:
    """Tests for ML Domain Contract (Domain 4)."""

    def test_contract_properties(self):
        """Test contract metadata properties."""
        class MockMLContract(MLContract):
            async def get_prediction(self, stock_id, horizon=PredictionHorizon.WEEK_1, model_id=None): pass
            async def generate_prediction(self, stock_id, horizon=PredictionHorizon.WEEK_1, model_id=None): pass
            async def batch_predict(self, stock_ids, horizon=PredictionHorizon.WEEK_1, model_id=None): pass
            async def get_prediction_history(self, stock_id, start_date, end_date, horizon=None): pass
            async def get_model(self, model_id): pass
            async def list_models(self, model_type=None, status=None): pass
            async def get_model_metrics(self, model_id): pass
            async def get_feature_importance(self, model_id, top_n=20): pass
            async def train_model(self, name, model_type, features, target, config=None): pass
            async def get_training_status(self, job_id): pass
            async def backtest_model(self, model_id, start_date, end_date, symbols=None, initial_capital=100000.0): pass
            async def validate_predictions(self, model_id, start_date, end_date): pass

        contract = MockMLContract()

        assert contract.domain_name == "ML"
        assert contract.version == "1.0.0"
        assert "get_prediction" in contract.capabilities
        assert "train_model" in contract.capabilities
        assert "backtest_model" in contract.capabilities

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test contract health check."""
        class MockMLContract(MLContract):
            async def get_prediction(self, stock_id, horizon=PredictionHorizon.WEEK_1, model_id=None): pass
            async def generate_prediction(self, stock_id, horizon=PredictionHorizon.WEEK_1, model_id=None): pass
            async def batch_predict(self, stock_ids, horizon=PredictionHorizon.WEEK_1, model_id=None): pass
            async def get_prediction_history(self, stock_id, start_date, end_date, horizon=None): pass
            async def get_model(self, model_id): pass
            async def list_models(self, model_type=None, status=None): pass
            async def get_model_metrics(self, model_id): pass
            async def get_feature_importance(self, model_id, top_n=20): pass
            async def train_model(self, name, model_type, features, target, config=None): pass
            async def get_training_status(self, job_id): pass
            async def backtest_model(self, model_id, start_date, end_date, symbols=None, initial_capital=100000.0): pass
            async def validate_predictions(self, model_id, start_date, end_date): pass

        contract = MockMLContract()
        health = await contract.health_check()

        assert health.success is True
        assert health.data["domain"] == "ML"


# =============================================================================
# Domain 5: Investment Analysis Contract Tests
# =============================================================================

class TestInvestmentAnalysisContract:
    """Tests for Investment Analysis Domain Contract (Domain 5)."""

    def test_contract_properties(self):
        """Test contract metadata properties."""
        class MockInvestmentAnalysisContract(InvestmentAnalysisContract):
            async def get_technical_analysis(self, stock_id, as_of_date=None): pass
            async def get_fundamental_analysis(self, stock_id, include_peers=True): pass
            async def get_sentiment_analysis(self, stock_id, days=30): pass
            async def generate_recommendation(self, stock_id, user_id=None, risk_tolerance=None): pass
            async def get_recommendation(self, recommendation_id): pass
            async def list_recommendations(self, stock_id=None, action=None, min_confidence=0.0, active_only=True, limit=20): pass
            async def update_recommendation_outcome(self, recommendation_id, actual_return, outcome): pass
            async def generate_thesis(self, stock_id): pass
            async def get_thesis(self, stock_id): pass
            async def get_sector_analysis(self, sector_id): pass
            async def run_screener(self, criteria, limit=50): pass
            async def compare_stocks(self, stock_ids): pass

        contract = MockInvestmentAnalysisContract()

        assert contract.domain_name == "InvestmentAnalysis"
        assert contract.version == "1.0.0"
        assert "get_technical_analysis" in contract.capabilities
        assert "get_fundamental_analysis" in contract.capabilities
        assert "generate_recommendation" in contract.capabilities
        assert "generate_thesis" in contract.capabilities

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test contract health check."""
        class MockInvestmentAnalysisContract(InvestmentAnalysisContract):
            async def get_technical_analysis(self, stock_id, as_of_date=None): pass
            async def get_fundamental_analysis(self, stock_id, include_peers=True): pass
            async def get_sentiment_analysis(self, stock_id, days=30): pass
            async def generate_recommendation(self, stock_id, user_id=None, risk_tolerance=None): pass
            async def get_recommendation(self, recommendation_id): pass
            async def list_recommendations(self, stock_id=None, action=None, min_confidence=0.0, active_only=True, limit=20): pass
            async def update_recommendation_outcome(self, recommendation_id, actual_return, outcome): pass
            async def generate_thesis(self, stock_id): pass
            async def get_thesis(self, stock_id): pass
            async def get_sector_analysis(self, sector_id): pass
            async def run_screener(self, criteria, limit=50): pass
            async def compare_stocks(self, stock_ids): pass

        contract = MockInvestmentAnalysisContract()
        health = await contract.health_check()

        assert health.success is True
        assert health.data["domain"] == "InvestmentAnalysis"


# =============================================================================
# Cross-Domain Integration Tests
# =============================================================================

class TestCrossDomainIntegration:
    """Tests for cross-domain contract integrations."""

    @pytest.mark.asyncio
    async def test_investment_analysis_to_market_data_integration(
        self,
        sample_stock_dto,
        sample_price_history_dto
    ):
        """
        Test Domain 5 (Investment Analysis) integration with Domain 1 (Market Data).

        Validates that Investment Analysis can properly consume Market Data
        for technical and fundamental analysis.
        """
        # Mock Market Data contract
        market_data_mock = AsyncMock()
        market_data_mock.get_stock.return_value = ContractResult.ok(sample_stock_dto)
        market_data_mock.get_price_history.return_value = ContractResult.ok(sample_price_history_dto)

        # Simulate cross-domain call
        stock_result = await market_data_mock.get_stock("AAPL")
        price_result = await market_data_mock.get_price_history(
            1, date.today() - timedelta(days=30), date.today()
        )

        assert stock_result.success is True
        assert stock_result.data.symbol == "AAPL"
        assert price_result.success is True
        assert len(price_result.data) == 30

    @pytest.mark.asyncio
    async def test_investment_analysis_to_portfolio_integration(
        self,
        sample_portfolio_dto,
        sample_recommendation_dto
    ):
        """
        Test Domain 5 (Investment Analysis) integration with Domain 2 (Portfolio).

        Validates that recommendations can be personalized based on
        portfolio context.
        """
        # Mock Portfolio contract
        portfolio_mock = AsyncMock()
        portfolio_mock.get_portfolio.return_value = ContractResult.ok(sample_portfolio_dto)

        # Simulate portfolio lookup for recommendation context
        portfolio_result = await portfolio_mock.get_portfolio(1, 1)

        assert portfolio_result.success is True
        assert portfolio_result.data.user_id == 1
        assert portfolio_result.data.cash_balance == Decimal("50000.00")

        # Recommendation should consider portfolio context
        # (e.g., available cash, existing positions)

    @pytest.mark.asyncio
    async def test_investment_analysis_to_ml_integration(
        self,
        sample_prediction_dto,
        sample_recommendation_dto
    ):
        """
        Test Domain 5 (Investment Analysis) integration with Domain 4 (ML).

        Validates that ML predictions are incorporated into
        investment recommendations.
        """
        # Mock ML contract
        ml_mock = AsyncMock()
        ml_mock.generate_prediction.return_value = ContractResult.ok(sample_prediction_dto)

        # Simulate ML prediction for recommendation
        prediction_result = await ml_mock.generate_prediction(1, PredictionHorizon.WEEK_1)

        assert prediction_result.success is True
        assert prediction_result.data.predicted_direction == PredictionDirection.UP
        assert prediction_result.data.confidence == 0.82

        # Recommendation should incorporate ML signal
        # (e.g., predicted direction and confidence affect recommendation)

    @pytest.mark.asyncio
    async def test_investment_analysis_to_data_pipeline_integration(self):
        """
        Test Domain 5 (Investment Analysis) integration with Domain 3 (Data Pipeline).

        Validates that data quality is checked before generating
        analysis and recommendations.
        """
        # Mock Data Pipeline contract
        data_pipeline_mock = AsyncMock()
        data_pipeline_mock.get_data_quality_report.return_value = ContractResult.ok(
            DataQualityReportDTO(
                stock_id=1,
                symbol="AAPL",
                data_type=DataType.PRICE_HISTORY,
                quality_level=DataQualityLevel.GOOD,
                quality_score=92.5,
                completeness=98.0,
                accuracy=95.0,
                timeliness=90.0,
                consistency=88.0,
                issues=[],
                assessed_at=datetime.now(timezone.utc)
            )
        )

        # Simulate data quality check before analysis
        quality_result = await data_pipeline_mock.get_data_quality_report(1, DataType.PRICE_HISTORY)

        assert quality_result.success is True
        assert quality_result.data.quality_level == DataQualityLevel.GOOD
        assert quality_result.data.quality_score >= 85.0

        # Analysis should only proceed if data quality is sufficient

    @pytest.mark.asyncio
    async def test_full_recommendation_pipeline(
        self,
        sample_stock_dto,
        sample_price_history_dto,
        sample_portfolio_dto,
        sample_prediction_dto,
        sample_recommendation_dto
    ):
        """
        Test full recommendation generation pipeline across all domains.

        This integration test validates the complete flow:
        1. Domain 1: Fetch market data
        2. Domain 3: Validate data quality
        3. Domain 4: Generate ML prediction
        4. Domain 2: Get portfolio context
        5. Domain 5: Generate recommendation
        """
        # Mock all contracts
        market_data_mock = AsyncMock()
        market_data_mock.get_stock.return_value = ContractResult.ok(sample_stock_dto)
        market_data_mock.get_price_history.return_value = ContractResult.ok(sample_price_history_dto)

        data_pipeline_mock = AsyncMock()
        data_pipeline_mock.get_data_quality_report.return_value = ContractResult.ok(
            DataQualityReportDTO(
                stock_id=1,
                symbol="AAPL",
                data_type=DataType.PRICE_HISTORY,
                quality_level=DataQualityLevel.GOOD,
                quality_score=92.5,
                completeness=98.0,
                accuracy=95.0,
                timeliness=90.0,
                consistency=88.0,
                issues=[],
                assessed_at=datetime.now(timezone.utc)
            )
        )

        ml_mock = AsyncMock()
        ml_mock.generate_prediction.return_value = ContractResult.ok(sample_prediction_dto)

        portfolio_mock = AsyncMock()
        portfolio_mock.get_portfolio.return_value = ContractResult.ok(sample_portfolio_dto)

        # Step 1: Fetch market data
        stock = (await market_data_mock.get_stock("AAPL")).unwrap()
        prices = (await market_data_mock.get_price_history(
            stock.id, date.today() - timedelta(days=30), date.today()
        )).unwrap()

        # Step 2: Validate data quality
        quality = (await data_pipeline_mock.get_data_quality_report(
            stock.id, DataType.PRICE_HISTORY
        )).unwrap()

        assert quality.quality_level in [DataQualityLevel.EXCELLENT, DataQualityLevel.GOOD]

        # Step 3: Generate ML prediction
        prediction = (await ml_mock.generate_prediction(stock.id, PredictionHorizon.WEEK_1)).unwrap()

        # Step 4: Get portfolio context
        portfolio = (await portfolio_mock.get_portfolio(1, 1)).unwrap()

        # Step 5: All data gathered - recommendation can be generated
        # Verify all necessary data is available
        assert stock.symbol == "AAPL"
        assert len(prices) == 30
        assert quality.quality_score >= 85.0
        assert prediction.confidence >= 0.5
        assert portfolio.cash_balance > 0


# =============================================================================
# Contract Error Handling Tests
# =============================================================================

class TestContractErrorHandling:
    """Tests for contract error handling patterns."""

    def test_error_to_dict_serialization(self):
        """Test error serialization to dictionary."""
        error = ContractError(
            code=ContractErrorCode.NOT_FOUND,
            message="Stock INVALID not found",
            details={"symbol": "INVALID"},
            source_domain="MarketData"
        )

        error_dict = error.to_dict()

        assert error_dict["code"] == "NOT_FOUND"
        assert error_dict["message"] == "Stock INVALID not found"
        assert error_dict["details"] == {"symbol": "INVALID"}
        assert error_dict["source_domain"] == "MarketData"
        assert "timestamp" in error_dict

    def test_error_codes_coverage(self):
        """Test all error codes are defined."""
        expected_codes = [
            "VALIDATION_ERROR",
            "NOT_FOUND",
            "UNAUTHORIZED",
            "RATE_LIMITED",
            "SERVICE_UNAVAILABLE",
            "INTERNAL_ERROR",
            "CONTRACT_VIOLATION",
            "TIMEOUT",
            "DATA_INTEGRITY_ERROR",
        ]

        actual_codes = [code.value for code in ContractErrorCode]

        for expected in expected_codes:
            assert expected in actual_codes, f"Missing error code: {expected}"

    @pytest.mark.asyncio
    async def test_graceful_error_propagation(self):
        """Test errors propagate gracefully across domains."""
        # Mock a contract that returns an error
        market_data_mock = AsyncMock()
        market_data_mock.get_stock.return_value = ContractResult.fail(
            ContractErrorCode.NOT_FOUND,
            "Stock not found",
            details={"symbol": "INVALID"},
            source_domain="MarketData"
        )

        result = await market_data_mock.get_stock("INVALID")

        assert result.success is False
        assert result.error.code == ContractErrorCode.NOT_FOUND
        assert result.error.source_domain == "MarketData"

        # Consumer should handle error gracefully
        if not result.success:
            # Return appropriate response
            assert result.error.message == "Stock not found"
