"""
Integration tests for AI agents to investment recommendations flow.

Tests cover LLM agent analysis, ML model predictions, multi-agent consensus,
and automated recommendation generation with confidence scoring.
"""

import pytest
import pytest_asyncio
from datetime import datetime, timedelta, date
from decimal import Decimal
from unittest.mock import AsyncMock, patch, MagicMock
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.tables import (
    Stock, Recommendation, RecommendationTypeEnum,
    Position, Portfolio, AssetTypeEnum
)
from backend.api.main import app
from httpx import AsyncClient, ASGITransport


pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def ml_model_mock():
    """Mock ML model predictions."""
    with patch("backend.ml.model_manager.ModelManager") as mock_manager:
        mock_instance = AsyncMock()
        mock_instance.predict_price.return_value = {
            "predicted_price": 175.50,
            "confidence": 0.82,
            "direction": "bullish",
            "volatility": 0.18,
            "support_levels": [165.0, 160.0, 155.0],
            "resistance_levels": [180.0, 185.0, 190.0]
        }
        mock_instance.predict_trend.return_value = {
            "trend": "upward",
            "strength": 0.75,
            "duration_days": 30,
            "reversal_probability": 0.15
        }
        mock_instance.calculate_risk_score.return_value = {
            "overall_risk": 0.35,
            "volatility_risk": 0.30,
            "liquidity_risk": 0.15,
            "market_risk": 0.40
        }
        mock_manager.return_value = mock_instance
        yield mock_instance


@pytest_asyncio.fixture
async def llm_agent_mock():
    """Mock LLM agent for fundamental analysis."""
    with patch("backend.agents.analysis_agent.AnalysisAgent") as mock_agent:
        mock_instance = AsyncMock()
        mock_instance.analyze_fundamentals.return_value = {
            "summary": "Strong company with solid fundamentals and growth potential",
            "strengths": [
                "High profit margins",
                "Strong balance sheet",
                "Growing market share",
                "Innovation pipeline"
            ],
            "weaknesses": [
                "Premium valuation",
                "Market concentration risk",
                "Regulatory concerns"
            ],
            "opportunities": [
                "Expanding into new markets",
                "Product diversification",
                "Strategic partnerships"
            ],
            "threats": [
                "Increasing competition",
                "Economic headwinds",
                "Supply chain disruptions"
            ],
            "overall_score": 0.78
        }
        mock_instance.generate_thesis.return_value = {
            "investment_case": "Buy recommendation based on strong fundamentals",
            "key_points": [
                "Consistent revenue growth",
                "Market leadership position",
                "Strong cash generation"
            ],
            "confidence": 0.82
        }
        mock_agent.return_value = mock_instance
        yield mock_instance


@pytest_asyncio.fixture
async def sample_stock_with_data(db_session: AsyncSession):
    """Create stock with complete data for agent analysis."""
    stock = Stock(
        symbol="NVDA",
        name="NVIDIA Corporation",
        exchange="NASDAQ",
        asset_type=AssetTypeEnum.STOCK,
        sector="Technology",
        industry="Semiconductors",
        market_cap=1500000000000,
        is_active=True,
        is_tradable=True
    )
    db_session.add(stock)
    await db_session.commit()
    await db_session.refresh(stock)
    return stock


@pytest.mark.asyncio
async def test_agent_analysis_to_recommendation(
    async_client: AsyncClient,
    db_session: AsyncSession,
    sample_stock_with_data: Stock,
    ml_model_mock,
    llm_agent_mock,
    auth_headers
):
    """
    Test LLM agent analysis -> ML prediction -> final recommendation.

    Validates that agent analysis is combined with ML predictions to
    generate comprehensive investment recommendations with reasoning.
    """
    # Step 1: Trigger agent analysis
    response = await async_client.post(
        f"/api/v1/agents/analyze/{sample_stock_with_data.symbol}",
        headers=auth_headers,
        json={
            "analysis_type": "comprehensive",
            "include_ml_predictions": True
        }
    )
    assert response.status_code == 200
    analysis_data = response.json()

    # Verify agent analysis structure
    assert "fundamental_analysis" in analysis_data["data"]
    assert "ml_predictions" in analysis_data["data"]
    assert analysis_data["data"]["fundamental_analysis"]["overall_score"] >= 0.7

    # Step 2: Generate recommendation from analysis
    response = await async_client.post(
        f"/api/v1/recommendations/from-analysis",
        headers=auth_headers,
        json={
            "stock_symbol": sample_stock_with_data.symbol,
            "analysis_id": analysis_data["data"]["id"]
        }
    )
    assert response.status_code == 201
    recommendation_data = response.json()

    # Verify recommendation combines both sources
    assert recommendation_data["data"]["stock_symbol"] == "NVDA"
    assert recommendation_data["data"]["confidence_score"] >= 0.75
    assert "reasoning" in recommendation_data["data"]
    assert "fundamental_score" in recommendation_data["data"]
    assert "technical_score" in recommendation_data["data"]

    # Reasoning should include both fundamental and technical insights
    reasoning = recommendation_data["data"]["reasoning"]
    assert any(word in reasoning.lower() for word in ["fundamental", "earnings", "growth"])
    assert any(word in reasoning.lower() for word in ["technical", "trend", "price"])


@pytest.mark.asyncio
async def test_ml_prediction_to_agent_analysis(
    async_client: AsyncClient,
    db_session: AsyncSession,
    sample_stock_with_data: Stock,
    ml_model_mock,
    llm_agent_mock,
    auth_headers
):
    """
    Test ML model predictions feeding into agent interpretation.

    Validates that ML predictions are correctly interpreted by LLM agents
    to provide context-aware investment insights.
    """
    # Step 1: Get ML predictions
    response = await async_client.get(
        f"/api/v1/ml/predict/{sample_stock_with_data.symbol}",
        headers=auth_headers,
        params={"horizon_days": 30}
    )
    assert response.status_code == 200
    ml_data = response.json()

    assert ml_data["data"]["predicted_price"] > 0
    assert ml_data["data"]["confidence"] > 0.7
    assert ml_data["data"]["direction"] in ["bullish", "bearish", "neutral"]

    # Step 2: Pass ML predictions to agent for interpretation
    response = await async_client.post(
        f"/api/v1/agents/interpret-predictions",
        headers=auth_headers,
        json={
            "stock_symbol": sample_stock_with_data.symbol,
            "ml_predictions": ml_data["data"]
        }
    )
    assert response.status_code == 200
    interpretation_data = response.json()

    # Verify agent provides context
    assert "interpretation" in interpretation_data["data"]
    assert "action_recommendation" in interpretation_data["data"]
    assert "risk_assessment" in interpretation_data["data"]

    # Agent should identify key factors
    interpretation = interpretation_data["data"]["interpretation"]
    assert len(interpretation) > 100  # Detailed interpretation
    assert interpretation_data["data"]["action_recommendation"] in [
        "strong_buy", "buy", "hold", "sell", "strong_sell"
    ]


@pytest.mark.asyncio
async def test_recommendation_confidence_scoring(
    async_client: AsyncClient,
    db_session: AsyncSession,
    sample_stock_with_data: Stock,
    ml_model_mock,
    llm_agent_mock,
    auth_headers
):
    """
    Test multi-agent consensus for confidence scoring.

    Validates that recommendations from multiple agents/models are
    aggregated to produce a weighted confidence score.
    """
    # Configure multiple agent responses with varying confidence
    agent_responses = [
        {"agent": "fundamental_agent", "recommendation": "buy", "confidence": 0.85},
        {"agent": "technical_agent", "recommendation": "buy", "confidence": 0.78},
        {"agent": "sentiment_agent", "recommendation": "hold", "confidence": 0.65},
        {"agent": "risk_agent", "recommendation": "buy", "confidence": 0.72}
    ]

    # Mock multi-agent analysis
    with patch("backend.services.recommendation_service.RecommendationService.multi_agent_consensus") as mock_consensus:
        mock_consensus.return_value = {
            "consensus_recommendation": "buy",
            "overall_confidence": 0.75,  # Weighted average
            "agent_responses": agent_responses,
            "agreement_level": 0.75,  # 3 out of 4 agree on buy
            "dissenting_opinions": [
                {
                    "agent": "sentiment_agent",
                    "recommendation": "hold",
                    "reasoning": "Mixed market sentiment"
                }
            ]
        }

        # Request multi-agent recommendation
        response = await async_client.post(
            f"/api/v1/recommendations/multi-agent/{sample_stock_with_data.symbol}",
            headers=auth_headers,
            json={"use_all_agents": True}
        )
        assert response.status_code == 200
        consensus_data = response.json()

        # Verify consensus structure
        assert consensus_data["data"]["consensus_recommendation"] == "buy"
        assert consensus_data["data"]["overall_confidence"] >= 0.70
        assert len(consensus_data["data"]["agent_responses"]) == 4
        assert consensus_data["data"]["agreement_level"] >= 0.50

        # High agreement should increase final confidence
        if consensus_data["data"]["agreement_level"] > 0.80:
            assert consensus_data["data"]["overall_confidence"] >= 0.75


@pytest.mark.asyncio
async def test_recommendation_to_portfolio_action(
    async_client: AsyncClient,
    db_session: AsyncSession,
    sample_stock_with_data: Stock,
    auth_headers,
    test_user
):
    """
    Test auto-executing trades based on high-confidence recommendations.

    Validates that recommendations with high confidence can trigger
    automated portfolio actions when user has enabled auto-trading.
    """
    # Create user portfolio with auto-trade settings
    portfolio = Portfolio(
        user_id=test_user.id,
        name="Auto-Trade Portfolio",
        description="Automated trading enabled",
        cash_balance=Decimal("100000.00"),
        is_public=False,
        is_default=True,
        target_allocation={"auto_trade": True, "confidence_threshold": 0.85}
    )
    db_session.add(portfolio)
    await db_session.commit()
    await db_session.refresh(portfolio)

    # Create high-confidence recommendation
    recommendation = Recommendation(
        stock_id=sample_stock_with_data.id,
        recommendation_type=RecommendationTypeEnum.STRONG_BUY,
        confidence_score=0.88,
        current_price=Decimal("450.00"),
        target_price=Decimal("525.00"),
        stop_loss=Decimal("420.00"),
        time_horizon_days=60,
        reasoning="Multi-agent consensus with 88% confidence for strong buy",
        key_factors=["strong_fundamentals", "positive_ml_prediction", "high_consensus"],
        risk_level="medium",
        technical_score=0.85,
        fundamental_score=0.90,
        sentiment_score=0.82,
        is_active=True,
        valid_until=datetime.utcnow() + timedelta(days=30)
    )
    db_session.add(recommendation)
    await db_session.commit()
    await db_session.refresh(recommendation)

    # Trigger auto-execution check
    with patch("backend.services.trading_service.TradingService.execute_order") as mock_execute:
        mock_execute.return_value = {
            "order_id": "AUTO_ORDER_12345",
            "status": "submitted",
            "quantity": 10,
            "estimated_cost": 4500.00
        }

        response = await async_client.post(
            f"/api/v1/portfolios/{portfolio.id}/auto-trade/check",
            headers=auth_headers
        )
        assert response.status_code == 200
        auto_trade_data = response.json()

        # Verify trade was executed
        assert "executed_orders" in auto_trade_data["data"]
        executed = auto_trade_data["data"]["executed_orders"]

        if recommendation.confidence_score >= 0.85:
            assert len(executed) > 0
            assert executed[0]["stock_symbol"] == "NVDA"
            assert executed[0]["action"] == "buy"
            assert executed[0]["reason"] == "high_confidence_recommendation"


@pytest.mark.asyncio
async def test_agent_error_handling_cascade(
    async_client: AsyncClient,
    db_session: AsyncSession,
    sample_stock_with_data: Stock,
    auth_headers
):
    """
    Test graceful degradation when agents/models fail.

    Validates that recommendation system continues to operate with
    reduced functionality when individual agents or models fail.
    """
    # Mock various agent failures
    with patch("backend.agents.analysis_agent.AnalysisAgent.analyze_fundamentals") as mock_fundamental:
        # Fundamental agent fails
        mock_fundamental.side_effect = Exception("LLM API timeout")

        # ML model succeeds
        with patch("backend.ml.model_manager.ModelManager.predict_price") as mock_ml:
            mock_ml.return_value = {
                "predicted_price": 475.00,
                "confidence": 0.75,
                "direction": "bullish"
            }

            # Request recommendation (should succeed with degraded data)
            response = await async_client.post(
                f"/api/v1/recommendations/generate/{sample_stock_with_data.symbol}",
                headers=auth_headers,
                json={"fallback_on_errors": True}
            )
            assert response.status_code == 200
            recommendation_data = response.json()

            # Recommendation generated but with warnings
            assert "warnings" in recommendation_data["data"]
            assert any("fundamental" in w.lower() for w in recommendation_data["data"]["warnings"])

            # Should still have ML-based recommendation
            assert recommendation_data["data"]["technical_score"] is not None
            assert recommendation_data["data"]["confidence_score"] > 0

            # But confidence should be lower due to missing fundamental data
            assert recommendation_data["data"]["confidence_score"] < 0.70

    # Test complete failure scenario
    with patch("backend.agents.analysis_agent.AnalysisAgent.analyze_fundamentals") as mock_fundamental:
        with patch("backend.ml.model_manager.ModelManager.predict_price") as mock_ml:
            # Both fail
            mock_fundamental.side_effect = Exception("LLM API timeout")
            mock_ml.side_effect = Exception("Model loading failed")

            # Request should fail gracefully
            response = await async_client.post(
                f"/api/v1/recommendations/generate/{sample_stock_with_data.symbol}",
                headers=auth_headers,
                json={"fallback_on_errors": True}
            )

            # Should return error but with helpful message
            assert response.status_code == 503  # Service unavailable
            error_data = response.json()
            assert "error" in error_data
            assert "temporarily unavailable" in error_data["error"].lower()
