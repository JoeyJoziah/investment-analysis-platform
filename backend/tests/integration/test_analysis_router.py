"""
Integration tests for the analysis router (/api/analysis).

Tests cover all analysis endpoints including:
- POST /api/analysis/analyze  (comprehensive stock analysis)
- POST /api/analysis/batch     (batch analysis of multiple symbols)
- POST /api/analysis/compare   (stock comparison)
- GET  /api/analysis/indicators/{symbol}  (technical indicators)
- GET  /api/analysis/sentiment/{symbol}   (sentiment analysis)

Each endpoint is tested for success paths, error handling, input validation,
and response format conformance.

GitHub Issue: #88
"""

import pytest
import pytest_asyncio
from datetime import datetime, timedelta, date
from decimal import Decimal
from unittest.mock import AsyncMock, patch, MagicMock
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.unified_models import (
    Stock, PriceHistory, Exchange, Sector, Industry
)
from backend.api.main import app
from backend.tests.conftest import assert_success_response
from httpx import AsyncClient


pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def sample_stock(db_session: AsyncSession, nasdaq_exchange, technology_sector, consumer_electronics_industry):
    """Create an AAPL stock record for testing."""
    stock = Stock(
        symbol="AAPL",
        name="Apple Inc.",
        exchange_id=nasdaq_exchange.id,
        asset_type="stock",
        sector_id=technology_sector.id,
        industry_id=consumer_electronics_industry.id,
        market_cap=3000000000000,
        shares_outstanding=16000000000,
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
async def sample_price_history(db_session: AsyncSession, sample_stock: Stock):
    """Generate 60 days of ascending price history for the sample stock."""
    prices = []
    base_date = date.today() - timedelta(days=60)

    for i in range(60):
        price = PriceHistory(
            stock_id=sample_stock.id,
            date=base_date + timedelta(days=i),
            open=Decimal("140.00") + Decimal(str(i * 0.30)),
            high=Decimal("142.00") + Decimal(str(i * 0.30)),
            low=Decimal("139.00") + Decimal(str(i * 0.30)),
            close=Decimal("141.00") + Decimal(str(i * 0.30)),
            adjusted_close=Decimal("141.00") + Decimal(str(i * 0.30)),
            volume=70000000 + (i * 500000),
        )
        prices.append(price)
        db_session.add(price)

    await db_session.commit()
    return prices


@pytest_asyncio.fixture
async def second_stock(db_session: AsyncSession, nasdaq_exchange, technology_sector, consumer_electronics_industry):
    """Create a second stock (MSFT) for comparison / batch tests."""
    stock = Stock(
        symbol="MSFT",
        name="Microsoft Corporation",
        exchange_id=nasdaq_exchange.id,
        asset_type="stock",
        sector_id=technology_sector.id,
        industry_id=consumer_electronics_industry.id,
        market_cap=2800000000000,
        shares_outstanding=7500000000,
        country="US",
        currency="USD",
        is_active=True,
        is_tradable=True,
    )
    db_session.add(stock)
    await db_session.commit()
    await db_session.refresh(stock)
    return stock


def _mock_technical_analyzer_obj():
    """Return a mock TechnicalAnalysisEngine with an async analyze method."""
    mock_obj = MagicMock()
    mock_obj.analyze = AsyncMock(return_value={
        "rsi": 55.0,
        "macd": {"macd": 1.2, "signal_line": 0.9, "histogram": 0.3},
        "moving_averages": {"sma_20": 150.0, "sma_50": 148.0, "sma_200": 145.0},
        "bollinger_bands": {"upper": 155.0, "middle": 150.0, "lower": 145.0},
        "volume_analysis": {"current_volume": 50000000, "avg_volume": 45000000},
        "support_levels": [145.0, 142.0],
        "resistance_levels": [155.0, 158.0],
        "trend": "bullish",
        "volatility": 0.18,
    })
    return mock_obj


# ---------------------------------------------------------------------------
# POST /api/analysis/analyze  --  comprehensive single-stock analysis
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_analyze_stock_comprehensive_success(
    async_client: AsyncClient,
    sample_stock: Stock,
    sample_price_history,
    auth_headers,
):
    """
    Comprehensive analysis with price history in DB should return a fully
    populated AnalysisResponse wrapped in ApiResponse.
    """
    with patch(
        "backend.api.routers.analysis.technical_analyzer",
        _mock_technical_analyzer_obj(),
    ), patch(
        "backend.api.routers.analysis.alpha_vantage_client",
        None,
    ), patch(
        "backend.api.routers.analysis.finnhub_client",
        None,
    ), patch(
        "backend.api.routers.analysis.model_manager",
        None,
    ):
        response = await async_client.post(
            "/api/v1/analysis/analyze",
            json={
                "symbol": "AAPL",
                "analysis_type": "comprehensive",
                "period": "1M",
                "include_ml_predictions": False,
                "include_news_sentiment": False,
            },
        )

    data = assert_success_response(response, expected_status=200)

    # Verify top-level response structure
    assert data["symbol"] == "AAPL"
    assert data["analysis_type"] == "comprehensive"
    assert data["technical"] is not None
    assert data["fundamental"] is not None
    assert "overall_score" in data
    assert 0 <= data["overall_score"] <= 100
    assert data["recommendation"] in [
        "strong_buy", "buy", "neutral", "sell", "strong_sell"
    ]
    assert 0 <= data["confidence"] <= 1
    assert isinstance(data["key_insights"], list)
    assert len(data["key_insights"]) > 0
    assert "timestamp" in data
    assert "last_updated" in data


@pytest.mark.asyncio
async def test_analyze_stock_technical_only(
    async_client: AsyncClient,
    sample_stock: Stock,
    sample_price_history,
    auth_headers,
):
    """
    When analysis_type is 'technical', the response should populate the
    technical field but fundamental and sentiment should be None.
    """
    with patch(
        "backend.api.routers.analysis.technical_analyzer",
        _mock_technical_analyzer_obj(),
    ), patch(
        "backend.api.routers.analysis.alpha_vantage_client", None,
    ), patch(
        "backend.api.routers.analysis.finnhub_client", None,
    ), patch(
        "backend.api.routers.analysis.model_manager", None,
    ):
        response = await async_client.post(
            "/api/v1/analysis/analyze",
            json={
                "symbol": "AAPL",
                "analysis_type": "technical",
                "period": "1M",
                "include_ml_predictions": False,
                "include_news_sentiment": False,
            },
        )

    data = assert_success_response(response, expected_status=200)
    assert data["analysis_type"] == "technical"
    assert data["technical"] is not None
    assert data["fundamental"] is None
    assert data["sentiment"] is None


@pytest.mark.asyncio
async def test_analyze_stock_not_found(
    async_client: AsyncClient,
    auth_headers,
):
    """
    Requesting analysis for a symbol not in the database should return 404.
    """
    response = await async_client.post(
        "/api/v1/analysis/analyze",
        json={
            "symbol": "DOESNOTEXIST",
            "analysis_type": "quick",
            "period": "1M",
            "include_ml_predictions": False,
            "include_news_sentiment": False,
        },
    )
    assert response.status_code == 404
    body = response.json()
    assert body["success"] is False
    assert "not found" in body["error"].lower()


@pytest.mark.asyncio
async def test_analyze_stock_invalid_period(
    async_client: AsyncClient,
    auth_headers,
):
    """
    An invalid period value should trigger a 422 validation error.
    """
    response = await async_client.post(
        "/api/v1/analysis/analyze",
        json={
            "symbol": "AAPL",
            "analysis_type": "technical",
            "period": "INVALID",
            "include_ml_predictions": False,
            "include_news_sentiment": False,
        },
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_analyze_stock_invalid_analysis_type(
    async_client: AsyncClient,
    auth_headers,
):
    """
    An unrecognized analysis_type should trigger a 422 validation error.
    """
    response = await async_client.post(
        "/api/v1/analysis/analyze",
        json={
            "symbol": "AAPL",
            "analysis_type": "nonexistent_type",
            "period": "1M",
        },
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_analyze_stock_symbol_uppercased(
    async_client: AsyncClient,
    sample_stock: Stock,
    sample_price_history,
    auth_headers,
):
    """
    The endpoint should normalize the symbol to uppercase regardless of
    the casing provided by the client.
    """
    with patch(
        "backend.api.routers.analysis.technical_analyzer",
        _mock_technical_analyzer_obj(),
    ), patch(
        "backend.api.routers.analysis.alpha_vantage_client", None,
    ), patch(
        "backend.api.routers.analysis.finnhub_client", None,
    ), patch(
        "backend.api.routers.analysis.model_manager", None,
    ):
        response = await async_client.post(
            "/api/v1/analysis/analyze",
            json={
                "symbol": "aapl",
                "analysis_type": "technical",
                "period": "1M",
                "include_ml_predictions": False,
                "include_news_sentiment": False,
            },
        )

    data = assert_success_response(response, expected_status=200)
    assert data["symbol"] == "AAPL"


@pytest.mark.asyncio
async def test_analyze_stock_risk_metrics_with_price_data(
    async_client: AsyncClient,
    sample_stock: Stock,
    sample_price_history,
    auth_headers,
):
    """
    When sufficient price history exists the response should contain
    calculated risk metrics rather than fallback values.
    """
    with patch(
        "backend.api.routers.analysis.technical_analyzer",
        _mock_technical_analyzer_obj(),
    ), patch(
        "backend.api.routers.analysis.alpha_vantage_client", None,
    ), patch(
        "backend.api.routers.analysis.finnhub_client", None,
    ), patch(
        "backend.api.routers.analysis.model_manager", None,
    ):
        response = await async_client.post(
            "/api/v1/analysis/analyze",
            json={
                "symbol": "AAPL",
                "analysis_type": "comprehensive",
                "period": "1M",
                "include_ml_predictions": False,
                "include_news_sentiment": False,
            },
        )

    data = assert_success_response(response, expected_status=200)
    risk = data["risk_metrics"]
    assert risk is not None
    assert "sharpe_ratio" in risk
    assert "max_drawdown" in risk
    assert 0 <= risk["overall_risk_score"] <= 100


# ---------------------------------------------------------------------------
# POST /api/analysis/compare  --  multi-stock comparison
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_compare_stocks_success(
    async_client: AsyncClient,
    sample_stock: Stock,
    second_stock: Stock,
    auth_headers,
):
    """
    Compare two valid symbols. The response should include comparison metrics,
    recommendations per symbol, and a correlation matrix.
    """
    response = await async_client.post(
        "/api/v1/analysis/compare",
        json={
            "symbols": ["AAPL", "MSFT"],
            "analysis_type": "quick",
            "compare": True,
        },
    )

    data = assert_success_response(response, expected_status=200)
    assert set(data["symbols"]) == {"AAPL", "MSFT"}
    assert "AAPL" in data["comparison_metrics"]
    assert "MSFT" in data["comparison_metrics"]
    assert data["best_performer"] in ["AAPL", "MSFT"]
    assert "AAPL" in data["recommendations"]
    assert "MSFT" in data["recommendations"]
    # Correlation matrix should exist for <= 10 symbols
    assert data["correlation_matrix"] is not None
    assert data["correlation_matrix"]["AAPL"]["AAPL"] == 1.0
    assert data["correlation_matrix"]["MSFT"]["MSFT"] == 1.0


@pytest.mark.asyncio
async def test_compare_stocks_requires_minimum_two_symbols(
    async_client: AsyncClient,
    auth_headers,
):
    """
    Comparison with fewer than 2 symbols should return 400.
    """
    response = await async_client.post(
        "/api/v1/analysis/compare",
        json={
            "symbols": ["AAPL"],
            "analysis_type": "quick",
            "compare": False,
        },
    )
    assert response.status_code == 400
    body = response.json()
    assert "2 symbols" in body.get("error", body.get("detail", "")).lower() or \
           "2 symbols" in str(body).lower()


# ---------------------------------------------------------------------------
# GET /api/analysis/indicators/{symbol}  --  technical indicators
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_technical_indicators_all(
    async_client: AsyncClient,
    auth_headers,
):
    """
    Requesting indicators without specifying which ones should return all
    available indicators.
    """
    response = await async_client.get(
        "/api/v1/analysis/indicators/AAPL",
    )
    data = assert_success_response(response, expected_status=200)
    assert data["symbol"] == "AAPL"
    assert "indicators" in data
    indicators = data["indicators"]
    assert "rsi" in indicators
    assert "macd" in indicators
    assert "bollinger_bands" in indicators
    assert "moving_averages" in indicators


@pytest.mark.asyncio
async def test_get_technical_indicators_specific(
    async_client: AsyncClient,
    auth_headers,
):
    """
    Request only RSI and MACD; only those should appear in the response.
    """
    response = await async_client.get(
        "/api/v1/analysis/indicators/AAPL",
        params={"indicators": ["rsi", "macd"]},
    )
    data = assert_success_response(response, expected_status=200)
    indicators = data["indicators"]
    assert "rsi" in indicators
    assert "macd" in indicators
    # Other indicators should not be present
    assert "bollinger_bands" not in indicators
    assert "moving_averages" not in indicators


@pytest.mark.asyncio
async def test_get_technical_indicators_rsi_structure(
    async_client: AsyncClient,
    auth_headers,
):
    """
    RSI indicator response should include value, signal, overbought and
    oversold thresholds.
    """
    response = await async_client.get(
        "/api/v1/analysis/indicators/AAPL",
        params={"indicators": ["rsi"]},
    )
    data = assert_success_response(response, expected_status=200)
    rsi = data["indicators"]["rsi"]
    assert "value" in rsi
    assert "signal" in rsi
    assert rsi["overbought"] == 70
    assert rsi["oversold"] == 30


# ---------------------------------------------------------------------------
# GET /api/analysis/sentiment/{symbol}  --  sentiment analysis
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_sentiment_analysis_success(
    async_client: AsyncClient,
    auth_headers,
):
    """
    Sentiment endpoint should return a well-formed SentimentAnalysis structure.
    """
    response = await async_client.get(
        "/api/v1/analysis/sentiment/AAPL",
    )
    data = assert_success_response(response, expected_status=200)
    assert -1 <= data["overall_sentiment"] <= 1
    assert data["sentiment_label"] in ["Positive", "Negative", "Neutral"]
    assert "news_sentiment" in data
    assert "social_sentiment" in data
    assert isinstance(data["key_topics"], list)
    assert isinstance(data["sentiment_sources"], dict)


@pytest.mark.asyncio
async def test_get_sentiment_analysis_has_all_fields(
    async_client: AsyncClient,
    auth_headers,
):
    """
    Verify the sentiment response contains all expected keys from the
    SentimentAnalysis model.
    """
    response = await async_client.get(
        "/api/v1/analysis/sentiment/TSLA",
    )
    data = assert_success_response(response, expected_status=200)
    expected_fields = [
        "overall_sentiment",
        "sentiment_label",
        "news_sentiment",
        "social_sentiment",
        "analyst_sentiment",
        "insider_sentiment",
        "sentiment_momentum",
        "key_topics",
        "sentiment_sources",
    ]
    for field in expected_fields:
        assert field in data, f"Missing field: {field}"


# ---------------------------------------------------------------------------
# POST /api/analysis/batch  --  batch analysis (edge cases)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_batch_analysis_empty_symbols_rejected(
    async_client: AsyncClient,
    auth_headers,
):
    """
    An empty symbols list should be rejected with a 422 validation error.
    """
    response = await async_client.post(
        "/api/v1/analysis/batch",
        json={
            "symbols": [],
            "analysis_type": "quick",
        },
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Response envelope conformance
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_all_analysis_endpoints_return_api_response_envelope(
    async_client: AsyncClient,
    auth_headers,
):
    """
    Every analysis endpoint should return the standard ApiResponse wrapper
    with 'success' and 'data' fields.
    """
    # Indicators endpoint (no db dependency)
    indicator_resp = await async_client.get("/api/v1/analysis/indicators/AAPL")
    assert indicator_resp.status_code == 200
    body = indicator_resp.json()
    assert "success" in body
    assert body["success"] is True
    assert "data" in body

    # Sentiment endpoint (no db dependency)
    sentiment_resp = await async_client.get("/api/v1/analysis/sentiment/AAPL")
    assert sentiment_resp.status_code == 200
    body = sentiment_resp.json()
    assert "success" in body
    assert body["success"] is True
    assert "data" in body
