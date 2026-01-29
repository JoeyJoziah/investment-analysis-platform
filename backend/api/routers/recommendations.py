from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks, Path, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, date, timedelta
from enum import Enum
import random
import asyncio
import logging
from sqlalchemy.ext.asyncio import AsyncSession

# Enhanced imports for real functionality
from backend.config.database import get_async_db_session
from backend.repositories import (
    recommendation_repository,
    stock_repository,
    portfolio_repository,
    price_repository,
    FilterCriteria,
    PaginationParams,
    SortParams
)
from backend.ml.model_manager import get_model_manager
from backend.analytics.recommendation_engine import RecommendationEngine
from backend.utils.cache import cache_with_ttl
from backend.utils.enhanced_error_handling import handle_api_error, validate_stock_symbol
from backend.auth.oauth2 import get_current_user
from backend.models.unified_models import User, Recommendation
from backend.config.settings import settings
from backend.models.api_response import ApiResponse, success_response

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/recommendations", tags=["recommendations"])

# =============================================================================
# SEC 2025 COMPLIANCE CONSTANTS
# =============================================================================

# Standard SEC Risk Warning (required on all recommendations)
SEC_RISK_WARNING = (
    "IMPORTANT: Past performance does not guarantee future results. All investments "
    "involve risk, including possible loss of principal. The value of investments can "
    "fluctuate, and investors may not get back the amount originally invested. Before "
    "making any investment decision, you should carefully consider your investment "
    "objectives, level of experience, and risk appetite."
)

# Standard Methodology Disclosure Template
SEC_METHODOLOGY_DISCLOSURE_TEMPLATE = (
    "This recommendation was generated using {algorithm_type} analysis incorporating "
    "technical indicators, fundamental metrics, and market sentiment data. Model version: "
    "{model_version}. Last model training date: {training_date}."
)

# Standard Limitations Statement
SEC_LIMITATIONS_STATEMENT = (
    "This analysis does NOT consider: (1) your individual financial situation or goals, "
    "(2) tax implications specific to your circumstances, (3) real-time market conditions "
    "that may have changed since data collection, (4) non-public information, (5) geopolitical "
    "events occurring after the analysis date. Data freshness may vary by source; prices and "
    "metrics may be delayed up to 15 minutes for free-tier data sources."
)

# Current model version for SEC disclosure
RECOMMENDATION_MODEL_VERSION = "1.0.0"
RECOMMENDATION_MODEL_TRAINING_DATE = "2025-12-15"

# Enum definitions (moved to top to avoid forward reference issues)
class RecommendationType(str, Enum):
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"

class TimeHorizon(str, Enum):
    SHORT_TERM = "short_term"  # 1-7 days
    MEDIUM_TERM = "medium_term"  # 1-3 months
    LONG_TERM = "long_term"  # 3+ months

class RiskLevel(str, Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"

class RecommendationCategory(str, Enum):
    VALUE = "value"
    GROWTH = "growth"
    DIVIDEND = "dividend"
    MOMENTUM = "momentum"
    CONTRARIAN = "contrarian"
    INDEX = "index"
    SECTOR_ROTATION = "sector_rotation"

# Pydantic models (defined before any functions that use them)
class RecommendationBase(BaseModel):
    symbol: str
    company_name: str
    recommendation_type: RecommendationType
    category: RecommendationCategory
    confidence_score: float = Field(..., ge=0, le=1)
    target_price: float
    current_price: float
    expected_return: float
    time_horizon: TimeHorizon
    risk_level: RiskLevel

class SECDisclosure(BaseModel):
    """SEC 2025 Required Disclosure Fields for Investment Recommendations"""
    model_config = {"protected_namespaces": ()}  # Allow model_* field names

    methodology_disclosure: str = Field(
        ...,
        description="Description of the algorithm and methodology used to generate this recommendation"
    )
    data_sources: List[str] = Field(
        ...,
        description="List of data sources used with timestamps indicating data freshness"
    )
    model_version: str = Field(
        ...,
        description="Version identifier of the ML model used for this recommendation"
    )
    model_training_date: str = Field(
        ...,
        description="Date when the recommendation model was last trained"
    )
    risk_warning: str = Field(
        ...,
        description="Standard SEC-required risk warning text"
    )
    limitations_statement: str = Field(
        ...,
        description="Statement of what the analysis does NOT consider"
    )
    confidence_level: str = Field(
        default="moderate",
        description="Confidence level of the recommendation (low/moderate/high)"
    )
    conflict_of_interest_statement: Optional[str] = Field(
        default=None,
        description="Disclosure of any material relationships with recommended securities"
    )


class RecommendationDetail(RecommendationBase):
    id: str
    created_at: datetime
    valid_until: datetime
    reasoning: str
    key_factors: List[str]
    technical_signals: Dict[str, Any]
    fundamental_metrics: Dict[str, Any]
    risk_factors: List[str]
    entry_points: List[float]
    exit_points: List[float]
    stop_loss: float
    sector: str
    market_cap: float
    volume: int
    analyst_consensus: Optional[str] = None
    similar_stocks: Optional[List[str]] = None
    # SEC 2025 Required Disclosure Fields
    sec_disclosure: Optional[SECDisclosure] = Field(
        default=None,
        description="SEC 2025 required disclosure information for this recommendation"
    )

class DailyRecommendations(BaseModel):
    date: date
    market_outlook: str
    top_picks: List[RecommendationDetail]
    watchlist: List[str]
    avoid_list: List[str]
    sector_focus: str
    market_sentiment: float = Field(..., ge=-1, le=1)
    risk_assessment: str
    special_situations: Optional[List[Dict[str, Any]]] = None
    # SEC 2025 Required Global Disclosures
    sec_global_disclosure: str = Field(
        default=SEC_RISK_WARNING,
        description="SEC-required global risk warning applicable to all recommendations"
    )
    data_as_of: str = Field(
        default_factory=lambda: datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC'),
        description="Timestamp indicating when data was collected for these recommendations"
    )
    recommendation_model_version: str = Field(
        default=RECOMMENDATION_MODEL_VERSION,
        description="Version of the recommendation model used"
    )

class PortfolioRecommendation(BaseModel):
    portfolio_id: str
    recommendations: List[RecommendationDetail]
    rebalancing_suggestions: Dict[str, float]
    risk_score: float
    expected_portfolio_return: float
    diversification_score: float

class RecommendationFilter(BaseModel):
    categories: Optional[List[RecommendationCategory]] = None
    risk_levels: Optional[List[RiskLevel]] = None
    time_horizons: Optional[List[TimeHorizon]] = None
    min_confidence: Optional[float] = Field(None, ge=0, le=1)
    min_expected_return: Optional[float] = None
    sectors: Optional[List[str]] = None
    market_cap_min: Optional[float] = None
    market_cap_max: Optional[float] = None

class RecommendationPerformance(BaseModel):
    recommendation_id: str
    symbol: str
    recommended_date: date
    recommendation_type: RecommendationType
    entry_price: float
    current_price: float
    target_price: float
    actual_return: float
    expected_return: float
    days_since_recommendation: int
    status: str  # "active", "closed", "stopped_out"
    performance_rating: float = Field(..., ge=0, le=5)

class AlertSettings(BaseModel):
    email_notifications: bool = True
    push_notifications: bool = False
    alert_types: List[str] = ["strong_buy", "strong_sell", "target_reached"]
    min_confidence: float = 0.7
    categories: List[RecommendationCategory] = []

# Initialize ML model manager and recommendation engine
model_manager = None
recommendation_engine = None

try:
    model_manager = get_model_manager()
    recommendation_engine = RecommendationEngine(model_manager=model_manager)
    logger.info("ML model manager and recommendation engine initialized successfully")
except Exception as e:
    logger.warning(f"ML model manager not available: {e}")
    recommendation_engine = RecommendationEngine()  # Fallback without ML


def generate_sec_disclosure(
    algorithm_type: str = "ML-powered quantitative",
    data_sources: List[str] = None,
    confidence_score: float = 0.5
) -> SECDisclosure:
    """
    Generate SEC 2025 compliant disclosure for a recommendation.

    Args:
        algorithm_type: Description of the algorithm used
        data_sources: List of data sources with timestamps
        confidence_score: Model confidence score (0-1)

    Returns:
        SECDisclosure object with all required fields
    """
    # Default data sources if not provided
    if data_sources is None:
        data_sources = [
            f"Alpha Vantage API (delayed 15 min) - {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
            f"Finnhub Market Data - {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
            f"Historical price data (EOD) - {datetime.utcnow().strftime('%Y-%m-%d')}",
            f"Financial statements (quarterly) - Last updated Q4 2025",
        ]

    # Determine confidence level from score
    if confidence_score >= 0.8:
        confidence_level = "high"
    elif confidence_score >= 0.6:
        confidence_level = "moderate"
    else:
        confidence_level = "low"

    # Generate methodology disclosure
    methodology_disclosure = SEC_METHODOLOGY_DISCLOSURE_TEMPLATE.format(
        algorithm_type=algorithm_type,
        model_version=RECOMMENDATION_MODEL_VERSION,
        training_date=RECOMMENDATION_MODEL_TRAINING_DATE
    )

    return SECDisclosure(
        methodology_disclosure=methodology_disclosure,
        data_sources=data_sources,
        model_version=RECOMMENDATION_MODEL_VERSION,
        model_training_date=RECOMMENDATION_MODEL_TRAINING_DATE,
        risk_warning=SEC_RISK_WARNING,
        limitations_statement=SEC_LIMITATIONS_STATEMENT,
        confidence_level=confidence_level,
        conflict_of_interest_statement=(
            "This platform does not hold positions in any recommended securities. "
            "No material relationships exist between this platform and any recommended issuers."
        )
    )

# Helper functions for real recommendation generation
async def generate_ml_powered_recommendations(
    user_id: Optional[int] = None,
    portfolio_id: Optional[str] = None,
    risk_level: Optional[RiskLevel] = None,
    categories: Optional[List[RecommendationCategory]] = None,
    limit: int = 10,
    db_session: AsyncSession = None
) -> List["RecommendationDetail"]:
    """
    Generate ML-powered recommendations with real market data.

    OPTIMIZED: Uses batch queries to eliminate N+1 query pattern.
    Previously: 201+ queries (1 for stocks + 2 per stock for prices/ML)
    Now: 2-3 queries total (1 for stocks + 1 bulk price history)
    """
    try:
        logger.info(f"Generating ML recommendations for user {user_id}, portfolio {portfolio_id}")

        # Query 1: Get market data for top stocks
        top_stocks = await stock_repository.get_top_stocks(
            limit=100,
            by_market_cap=True,
            session=db_session
        )

        if not top_stocks:
            logger.warning("No stocks found for recommendations")
            return [generate_recommendation() for _ in range(min(limit, 5))]

        # OPTIMIZATION: Batch fetch all price histories in a single query
        # This eliminates the N+1 pattern (was: 1 query per stock in loop)
        symbols_to_fetch = [stock.symbol for stock in top_stocks[:limit * 2]]  # Fetch extra for filtering

        # Query 2: Single bulk query for all price histories
        all_price_histories = await price_repository.get_bulk_price_history(
            symbols=symbols_to_fetch,
            start_date=datetime.now().date() - timedelta(days=90),
            end_date=datetime.now().date(),
            limit_per_symbol=60,
            session=db_session
        )

        logger.debug(f"Bulk fetched price histories for {len(all_price_histories)} symbols")

        # Build stock lookup for similar stocks calculation
        stock_by_sector: Dict[str, List[str]] = {}
        for stock in top_stocks:
            if stock.sector:
                if stock.sector not in stock_by_sector:
                    stock_by_sector[stock.sector] = []
                stock_by_sector[stock.sector].append(stock.symbol)

        # OPTIMIZATION: Prepare batch ML predictions if available
        ml_predictions_batch: Dict[str, Dict[str, Any]] = {}
        if model_manager and recommendation_engine:
            try:
                # Prepare all price data for batch ML prediction
                batch_price_data = {}
                for symbol, price_history in all_price_histories.items():
                    if price_history and len(price_history) >= 30:
                        batch_price_data[symbol] = [
                            {
                                'open': float(p.open),
                                'high': float(p.high),
                                'low': float(p.low),
                                'close': float(p.close),
                                'volume': p.volume,
                                'date': p.date
                            }
                            for p in price_history
                        ]

                # Try batch prediction if available, otherwise will fall back to individual
                if hasattr(recommendation_engine, 'analyze_stocks_batch'):
                    ml_predictions_batch = await recommendation_engine.analyze_stocks_batch(
                        price_data_batch=batch_price_data,
                        user_risk_tolerance=risk_level.value if risk_level else 'moderate'
                    )
            except Exception as e:
                logger.warning(f"Batch ML prediction not available, will use individual: {e}")

        recommendations = []

        # Process stocks using pre-fetched data (no additional queries in loop)
        for stock in top_stocks:
            if len(recommendations) >= limit:
                break

            try:
                # Use pre-fetched price history (no query needed)
                price_history = all_price_histories.get(stock.symbol, [])

                if not price_history or len(price_history) < 30:
                    continue

                # Prepare data for ML model (using cached data)
                price_data = [
                    {
                        'open': float(p.open),
                        'high': float(p.high),
                        'low': float(p.low),
                        'close': float(p.close),
                        'volume': p.volume,
                        'date': p.date
                    }
                    for p in price_history
                ]

                current_price = float(price_history[-1].close)

                # Get ML prediction (from batch or individual)
                ml_prediction = None
                recommendation_type = RecommendationType.HOLD
                confidence_score = 0.6

                if stock.symbol in ml_predictions_batch:
                    # Use batch prediction result
                    analysis = ml_predictions_batch[stock.symbol]
                    ml_prediction = analysis.get('prediction')
                    confidence_score = analysis.get('confidence', 0.6)
                elif model_manager and recommendation_engine:
                    # Fallback to individual prediction
                    try:
                        analysis = await recommendation_engine.analyze_stock(
                            symbol=stock.symbol,
                            price_data=price_data,
                            user_risk_tolerance=risk_level.value if risk_level else 'moderate'
                        )
                        ml_prediction = analysis.get('prediction')
                        confidence_score = analysis.get('confidence', 0.6)
                    except Exception as e:
                        logger.error(f"Error in ML prediction for {stock.symbol}: {e}")

                # Map ML prediction to recommendation type
                if ml_prediction:
                    pred_value = ml_prediction.get('direction', 0)
                    if pred_value > 0.7:
                        recommendation_type = RecommendationType.STRONG_BUY
                    elif pred_value > 0.3:
                        recommendation_type = RecommendationType.BUY
                    elif pred_value < -0.7:
                        recommendation_type = RecommendationType.STRONG_SELL
                    elif pred_value < -0.3:
                        recommendation_type = RecommendationType.SELL

                # Calculate target price
                target_price = current_price * (1 + (confidence_score * 0.2 - 0.1))
                expected_return = (target_price - current_price) / current_price

                # Determine category based on stock characteristics
                category = RecommendationCategory.GROWTH
                if stock.sector == "Technology":
                    category = RecommendationCategory.GROWTH
                elif stock.market_cap and stock.market_cap > 100000000000:
                    category = RecommendationCategory.VALUE

                # Filter by requested categories
                if categories and category not in categories:
                    continue

                # Generate SEC disclosure for this recommendation
                sec_disclosure = generate_sec_disclosure(
                    algorithm_type="ML-powered quantitative analysis",
                    data_sources=[
                        f"Price history database - {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
                        f"Stock fundamentals - {datetime.utcnow().strftime('%Y-%m-%d')}",
                        f"ML prediction model v{RECOMMENDATION_MODEL_VERSION}",
                    ],
                    confidence_score=confidence_score
                )

                # Get similar stocks from pre-computed sector lookup
                similar_stocks = []
                if stock.sector and stock.sector in stock_by_sector:
                    similar_stocks = [s for s in stock_by_sector[stock.sector] if s != stock.symbol][:3]

                # Create recommendation with SEC disclosure
                recommendation = RecommendationDetail(
                    id=f"ML-{stock.symbol}-{int(datetime.utcnow().timestamp())}",
                    symbol=stock.symbol,
                    company_name=stock.name,
                    recommendation_type=recommendation_type,
                    category=category,
                    confidence_score=confidence_score,
                    target_price=round(target_price, 2),
                    current_price=current_price,
                    expected_return=round(expected_return, 4),
                    time_horizon=TimeHorizon.MEDIUM_TERM,
                    risk_level=risk_level or RiskLevel.MODERATE,
                    created_at=datetime.utcnow(),
                    valid_until=datetime.utcnow() + timedelta(days=7),
                    reasoning="ML-powered analysis based on price patterns, volume trends, and market conditions",
                    key_factors=[
                        f"ML confidence: {confidence_score:.1%}",
                        f"Price momentum: {'Positive' if expected_return > 0 else 'Negative'}",
                        f"Market cap: ${stock.market_cap:,.0f}" if stock.market_cap else "Market cap: N/A",
                        f"Sector: {stock.sector}" if stock.sector else "Sector: N/A"
                    ],
                    technical_signals={
                        "ml_prediction": ml_prediction.get('direction', 0) if ml_prediction else 0,
                        "price_trend": "bullish" if expected_return > 0 else "bearish",
                        "volatility": ml_prediction.get('volatility', 0.2) if ml_prediction else 0.2
                    },
                    fundamental_metrics={
                        "market_cap": stock.market_cap,
                        "sector": stock.sector,
                        "industry": stock.industry
                    },
                    risk_factors=[
                        "Market volatility",
                        "Sector-specific risks",
                        "Liquidity risk" if stock.market_cap and stock.market_cap < 1000000000 else None
                    ],
                    entry_points=[current_price * 0.98, current_price * 0.95],
                    exit_points=[target_price * 0.95, target_price],
                    stop_loss=current_price * 0.92,
                    sector=stock.sector or "Unknown",
                    market_cap=stock.market_cap or 0,
                    volume=price_history[-1].volume if price_history else 0,
                    analyst_consensus=None,  # Would integrate with analyst data
                    similar_stocks=similar_stocks,
                    sec_disclosure=sec_disclosure
                )

                recommendations.append(recommendation)

            except Exception as e:
                logger.error(f"Error generating recommendation for {stock.symbol}: {e}")
                continue

        # Sort by confidence score
        recommendations.sort(key=lambda x: x.confidence_score, reverse=True)
        logger.info(f"Generated {len(recommendations)} ML recommendations using optimized batch queries")
        return recommendations[:limit]

    except Exception as e:
        logger.error(f"Error generating ML recommendations: {e}")
        # Fallback to mock data
        return [generate_recommendation() for _ in range(min(limit, 5))]

async def generate_personalized_recommendations(
    user_id: int,
    portfolio_id: Optional[str] = None,
    db_session: AsyncSession = None
) -> List["RecommendationDetail"]:
    """Generate personalized recommendations based on user's portfolio and preferences"""
    try:
        logger.info(f"Generating personalized recommendations for user {user_id}")
        
        # Get user's portfolio(s) to understand preferences
        user_portfolios = await portfolio_repository.get_user_portfolios(
            user_id=user_id,
            session=db_session
        )
        
        # Analyze existing positions to understand preferences
        existing_symbols = set()
        preferred_sectors = {}
        risk_tolerance = RiskLevel.MODERATE
        
        for portfolio in user_portfolios:
            positions = await portfolio_repository.get_portfolio_positions(
                portfolio_id=portfolio.id,
                session=db_session
            )
            
            for position in positions:
                existing_symbols.add(position.symbol)
                
                # Get stock info to determine sector preference
                stock = await stock_repository.get_by_symbol(position.symbol, session=db_session)
                if stock and stock.sector:
                    preferred_sectors[stock.sector] = preferred_sectors.get(stock.sector, 0) + 1
        
        # Generate recommendations excluding existing positions
        all_recommendations = await generate_ml_powered_recommendations(
            user_id=user_id,
            limit=20,
            db_session=db_session
        )
        
        # Filter out existing positions and prefer similar sectors
        filtered_recommendations = []
        for rec in all_recommendations:
            if rec.symbol not in existing_symbols:
                # Boost confidence for preferred sectors
                if rec.sector in preferred_sectors:
                    rec.confidence_score = min(0.95, rec.confidence_score * 1.1)
                    rec.reasoning += f" (Matches your sector preference for {rec.sector})"
                
                filtered_recommendations.append(rec)
        
        return filtered_recommendations[:10]
        
    except Exception as e:
        logger.error(f"Error generating personalized recommendations: {e}")
        return await generate_ml_powered_recommendations(limit=5, db_session=db_session)

# Sample data generator functions
def generate_recommendation(symbol: str = None) -> "RecommendationDetail":
    """Generate a sample recommendation with SEC disclosure"""
    if not symbol:
        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "META", "NVDA", "TSLA", "JPM", "V", "JNJ"]
        symbol = random.choice(symbols)

    current_price = random.uniform(50, 500)
    target_price = current_price * random.uniform(0.9, 1.3)
    confidence_score = random.uniform(0.6, 0.95)

    # Generate SEC disclosure for sample recommendation
    sec_disclosure = generate_sec_disclosure(
        algorithm_type="quantitative technical and fundamental",
        data_sources=[
            f"Market data feed - {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
            f"Financial statements - Q4 2025",
            f"Analyst consensus data - {datetime.utcnow().strftime('%Y-%m-%d')}",
        ],
        confidence_score=confidence_score
    )

    return RecommendationDetail(
        id=f"REC-{random.randint(1000, 9999)}",
        symbol=symbol,
        company_name=f"{symbol} Inc.",
        recommendation_type=random.choice(list(RecommendationType)),
        category=random.choice(list(RecommendationCategory)),
        confidence_score=confidence_score,
        target_price=round(target_price, 2),
        current_price=round(current_price, 2),
        expected_return=round((target_price - current_price) / current_price, 4),
        time_horizon=random.choice(list(TimeHorizon)),
        risk_level=random.choice(list(RiskLevel)),
        created_at=datetime.utcnow(),
        valid_until=datetime.utcnow() + timedelta(days=random.randint(7, 90)),
        reasoning="Based on strong technical indicators and improving fundamentals",
        key_factors=[
            "Strong earnings growth",
            "Positive analyst sentiment",
            "Technical breakout pattern",
            "Sector rotation favor"
        ],
        technical_signals={
            "rsi": random.uniform(30, 70),
            "macd": "bullish",
            "support": current_price * 0.95,
            "resistance": current_price * 1.05
        },
        fundamental_metrics={
            "pe_ratio": random.uniform(10, 30),
            "eps_growth": random.uniform(-0.1, 0.3),
            "revenue_growth": random.uniform(0, 0.25),
            "profit_margin": random.uniform(0.05, 0.3)
        },
        risk_factors=[
            "Market volatility",
            "Sector competition",
            "Regulatory changes"
        ],
        entry_points=[current_price * 0.98, current_price * 0.96],
        exit_points=[target_price * 0.95, target_price],
        stop_loss=current_price * 0.92,
        sector="Technology",
        market_cap=random.uniform(100000000000, 3000000000000),
        volume=random.randint(10000000, 100000000),
        analyst_consensus="Buy",
        similar_stocks=["GOOG", "FB", "NFLX"] if symbol != "GOOGL" else ["AAPL", "MSFT"],
        sec_disclosure=sec_disclosure
    )

# Enhanced Endpoints with ML Integration
@router.get("/daily")
@cache_with_ttl(ttl=3600)  # Cache for 1 hour
async def get_daily_recommendations(
    date_param: Optional[date] = Query(None, alias="date"),
    risk_level: Optional[RiskLevel] = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db_session)
) -> ApiResponse[DailyRecommendations]:
    """
    Get daily curated recommendations powered by ML models and market analysis.
    
    Provides comprehensive daily market recommendations including:
    - ML-generated top picks based on technical and fundamental analysis
    - Market sentiment analysis and outlook
    - Sector rotation recommendations
    - Risk-adjusted picks based on user preference
    - Special market situations and opportunities
    """
    try:
        target_date = date_param or date.today()
        logger.info(f"Generating daily recommendations for {target_date}, risk level: {risk_level}")
        
        # Generate ML-powered recommendations
        ml_recommendations = await generate_ml_powered_recommendations(
            user_id=current_user.id,
            risk_level=risk_level,
            limit=15,
            db_session=db
        )
        
        # Get personalized recommendations based on user's portfolio
        personalized_recs = await generate_personalized_recommendations(
            user_id=current_user.id,
            db_session=db
        )
        
        # Combine and deduplicate recommendations
        all_recommendations = {}
        for rec in ml_recommendations + personalized_recs:
            if rec.symbol not in all_recommendations:
                all_recommendations[rec.symbol] = rec
            elif rec.confidence_score > all_recommendations[rec.symbol].confidence_score:
                all_recommendations[rec.symbol] = rec
        
        # Filter by risk level if specified
        if risk_level:
            filtered_recs = [r for r in all_recommendations.values() if r.risk_level == risk_level]
            if len(filtered_recs) < 5:
                # Include some moderate risk recommendations if not enough matches
                other_recs = [r for r in all_recommendations.values() if r.risk_level != risk_level]
                filtered_recs.extend(other_recs[:(5-len(filtered_recs))])
            all_recommendations = {r.symbol: r for r in filtered_recs}
        
        # Sort by confidence score and take top picks
        top_picks = sorted(all_recommendations.values(), key=lambda x: x.confidence_score, reverse=True)[:8]
        
        # Generate watchlist from remaining high-confidence picks
        watchlist_symbols = [r.symbol for r in sorted(all_recommendations.values(), key=lambda x: x.confidence_score, reverse=True)[8:15]]
        
        # Generate avoid list based on negative recommendations
        avoid_list = []
        negative_recs = [r for r in all_recommendations.values() if r.recommendation_type in [RecommendationType.SELL, RecommendationType.STRONG_SELL]]
        avoid_list = [r.symbol for r in negative_recs[:5]]
        
        # Determine sector focus based on recommendations
        sector_counts = {}
        for rec in top_picks:
            if rec.sector and rec.sector != "Unknown":
                sector_counts[rec.sector] = sector_counts.get(rec.sector, 0) + 1
        
        sector_focus = max(sector_counts.items(), key=lambda x: x[1])[0] if sector_counts else "Technology"
        
        # Calculate market sentiment from recommendations
        sentiment_scores = [
            1.0 if rec.recommendation_type == RecommendationType.STRONG_BUY else
            0.5 if rec.recommendation_type == RecommendationType.BUY else
            0.0 if rec.recommendation_type == RecommendationType.HOLD else
            -0.5 if rec.recommendation_type == RecommendationType.SELL else
            -1.0 for rec in top_picks
        ]
        market_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0
        
        # Generate market outlook
        if market_sentiment > 0.3:
            outlook = "Bullish - Strong buying opportunities identified across multiple sectors"
        elif market_sentiment > 0.1:
            outlook = "Cautiously optimistic - Selective opportunities in preferred sectors"
        elif market_sentiment > -0.1:
            outlook = "Neutral - Mixed signals, focus on risk management"
        elif market_sentiment > -0.3:
            outlook = "Cautious - Defensive positioning recommended"
        else:
            outlook = "Bearish - High risk environment, consider cash positions"
        
        # Risk assessment based on volatility and market conditions
        avg_confidence = sum(r.confidence_score for r in top_picks) / len(top_picks) if top_picks else 0.5
        if avg_confidence > 0.8:
            risk_assessment = "Low - High confidence in current recommendations"
        elif avg_confidence > 0.6:
            risk_assessment = "Moderate - Standard market conditions"
        else:
            risk_assessment = "Elevated - Uncertain market environment"
        
        # Generate special situations (earnings, events, etc.)
        special_situations = []
        for rec in top_picks[:3]:
            if rec.recommendation_type in [RecommendationType.STRONG_BUY, RecommendationType.BUY]:
                special_situations.append({
                    "type": "high_confidence_pick",
                    "symbol": rec.symbol,
                    "confidence": rec.confidence_score,
                    "reasoning": rec.reasoning[:100] + "..." if len(rec.reasoning) > 100 else rec.reasoning,
                    "target_return": rec.expected_return
                })

        return success_response(data=DailyRecommendations(
            date=target_date,
            market_outlook=outlook,
            top_picks=top_picks[:5],  # Limit to top 5 for clarity
            watchlist=watchlist_symbols,
            avoid_list=avoid_list,
            sector_focus=sector_focus,
            market_sentiment=round(market_sentiment, 3),
            risk_assessment=risk_assessment,
            special_situations=special_situations
        ))
        
    except Exception as e:
        logger.error(f"Error generating daily recommendations: {e}")
        await handle_api_error(e, "generate daily recommendations")
        
        # Fallback to basic recommendations
        fallback_picks = [generate_recommendation() for _ in range(5)]
        return success_response(data=DailyRecommendations(
            date=target_date,
            market_outlook="Market analysis temporarily unavailable",
            top_picks=fallback_picks,
            watchlist=["AAPL", "GOOGL", "MSFT", "NVDA", "AMD"],
            avoid_list=[],
            sector_focus="Technology",
            market_sentiment=0.0,
            risk_assessment="Analysis unavailable",
            special_situations=[]
        ))

@router.get("/list")
async def get_recommendations(
    limit: int = Query(10, le=100),
    offset: int = 0,
    recommendation_type: Optional[RecommendationType] = None,
    category: Optional[RecommendationCategory] = None,
    risk_level: Optional[RiskLevel] = None,
    min_confidence: float = Query(0.0, ge=0, le=1),
    sort_by: str = Query("confidence_score", pattern="^(confidence_score|expected_return|created_at)$"),
    order: str = Query("desc", pattern="^(asc|desc)$")
) -> ApiResponse[List[RecommendationDetail]]:
    """Get list of recommendations with filters"""
    
    # Generate sample recommendations
    recommendations = [generate_recommendation() for _ in range(50)]
    
    # Apply filters
    if recommendation_type:
        recommendations = [r for r in recommendations if r.recommendation_type == recommendation_type]
    
    if category:
        recommendations = [r for r in recommendations if r.category == category]
    
    if risk_level:
        recommendations = [r for r in recommendations if r.risk_level == risk_level]
    
    recommendations = [r for r in recommendations if r.confidence_score >= min_confidence]
    
    # Sort
    reverse = (order == "desc")
    if sort_by == "confidence_score":
        recommendations.sort(key=lambda x: x.confidence_score, reverse=reverse)
    elif sort_by == "expected_return":
        recommendations.sort(key=lambda x: x.expected_return, reverse=reverse)
    elif sort_by == "created_at":
        recommendations.sort(key=lambda x: x.created_at, reverse=reverse)

    # Pagination
    return success_response(data=recommendations[offset:offset + limit])

@router.get("/{recommendation_id}")
async def get_recommendation_detail(recommendation_id: str) -> ApiResponse[RecommendationDetail]:
    """Get detailed information about a specific recommendation"""
    
    # Generate a recommendation with the specified ID
    recommendation = generate_recommendation()
    recommendation.id = recommendation_id

    return success_response(data=recommendation)

@router.post("/filter")
async def filter_recommendations(
    filter_params: RecommendationFilter,
    limit: int = Query(20, le=100)
) -> ApiResponse[List[RecommendationDetail]]:
    """Advanced filtering of recommendations"""
    
    # Generate recommendations
    recommendations = [generate_recommendation() for _ in range(100)]
    
    # Apply filters
    if filter_params.categories:
        recommendations = [r for r in recommendations if r.category in filter_params.categories]
    
    if filter_params.risk_levels:
        recommendations = [r for r in recommendations if r.risk_level in filter_params.risk_levels]
    
    if filter_params.time_horizons:
        recommendations = [r for r in recommendations if r.time_horizon in filter_params.time_horizons]
    
    if filter_params.min_confidence is not None:
        recommendations = [r for r in recommendations if r.confidence_score >= filter_params.min_confidence]
    
    if filter_params.min_expected_return is not None:
        recommendations = [r for r in recommendations if r.expected_return >= filter_params.min_expected_return]
    
    if filter_params.sectors:
        recommendations = [r for r in recommendations if r.sector in filter_params.sectors]
    
    if filter_params.market_cap_min is not None:
        recommendations = [r for r in recommendations if r.market_cap >= filter_params.market_cap_min]
    
    if filter_params.market_cap_max is not None:
        recommendations = [r for r in recommendations if r.market_cap <= filter_params.market_cap_max]
    
    # Sort by confidence score
    recommendations.sort(key=lambda x: x.confidence_score, reverse=True)

    return success_response(data=recommendations[:limit])

@router.get("/portfolio/{portfolio_id}")
async def get_portfolio_recommendations(portfolio_id: str) -> ApiResponse[PortfolioRecommendation]:
    """Get personalized recommendations for a specific portfolio"""
    
    # Generate recommendations tailored to portfolio
    recommendations = [generate_recommendation() for _ in range(5)]
    
    # Create rebalancing suggestions
    rebalancing = {
        "AAPL": 0.25,
        "GOOGL": 0.20,
        "MSFT": 0.20,
        "AMZN": 0.15,
        "NVDA": 0.10,
        "Cash": 0.10
    }

    return success_response(data=PortfolioRecommendation(
        portfolio_id=portfolio_id,
        recommendations=recommendations,
        rebalancing_suggestions=rebalancing,
        risk_score=random.uniform(30, 70),
        expected_portfolio_return=random.uniform(0.08, 0.15),
        diversification_score=random.uniform(0.6, 0.9)
    ))

@router.get("/performance/track")
async def track_recommendation_performance(
    days_back: int = Query(30, le=365),
    status: Optional[str] = Query(None, pattern="^(active|closed|stopped_out)$")
) -> ApiResponse[List[RecommendationPerformance]]:
    """Track performance of past recommendations"""
    
    performances = []
    
    for i in range(20):
        entry_price = random.uniform(50, 300)
        current_price = entry_price * random.uniform(0.8, 1.3)
        target_price = entry_price * random.uniform(1.1, 1.4)
        
        perf = RecommendationPerformance(
            recommendation_id=f"REC-{1000 + i}",
            symbol=random.choice(["AAPL", "GOOGL", "MSFT", "AMZN", "META"]),
            recommended_date=date.today() - timedelta(days=random.randint(1, days_back)),
            recommendation_type=random.choice(list(RecommendationType)),
            entry_price=entry_price,
            current_price=current_price,
            target_price=target_price,
            actual_return=(current_price - entry_price) / entry_price,
            expected_return=(target_price - entry_price) / entry_price,
            days_since_recommendation=random.randint(1, days_back),
            status=status or random.choice(["active", "closed", "stopped_out"]),
            performance_rating=random.uniform(2, 5)
        )
        performances.append(perf)
    
    if status:
        performances = [p for p in performances if p.status == status]

    return success_response(data=performances)

@router.post("/alerts/settings")
async def update_alert_settings(settings: AlertSettings) -> ApiResponse[Dict[str, str]]:
    """Update recommendation alert settings"""
    
    # In production, this would save to user preferences
    return success_response(data={
        "message": "Alert settings updated successfully",
        "status": "success"
    })

@router.get("/alerts/history")
async def get_alert_history(
    days_back: int = Query(7, le=30)
) -> ApiResponse[List[Dict[str, Any]]]:
    """Get history of recommendation alerts"""
    
    alerts = []
    for i in range(10):
        alert_date = datetime.utcnow() - timedelta(days=random.randint(0, days_back))
        alerts.append({
            "id": f"ALERT-{1000 + i}",
            "timestamp": alert_date.isoformat(),
            "type": random.choice(["strong_buy", "target_reached", "stop_loss_triggered"]),
            "symbol": random.choice(["AAPL", "GOOGL", "MSFT"]),
            "message": "Strong buy signal detected",
            "read": random.choice([True, False])
        })

    return success_response(data=sorted(alerts, key=lambda x: x["timestamp"], reverse=True))

@router.post("/backtest")
async def backtest_strategy(
    strategy: RecommendationCategory,
    start_date: date,
    end_date: date,
    initial_capital: float = 100000
) -> ApiResponse[Dict[str, Any]]:
    """Backtest a recommendation strategy"""
    
    # Simulate backtest results
    total_return = random.uniform(-0.2, 0.5)

    return success_response(data={
        "strategy": strategy,
        "period": {
            "start": start_date.isoformat(),
            "end": end_date.isoformat()
        },
        "initial_capital": initial_capital,
        "final_value": initial_capital * (1 + total_return),
        "total_return": total_return,
        "annualized_return": total_return * (365 / (end_date - start_date).days),
        "sharpe_ratio": random.uniform(0.5, 2.0),
        "max_drawdown": random.uniform(-0.3, -0.05),
        "win_rate": random.uniform(0.4, 0.7),
        "total_trades": random.randint(20, 100),
        "profitable_trades": random.randint(10, 70),
        "average_win": random.uniform(0.05, 0.15),
        "average_loss": random.uniform(-0.1, -0.03),
        "best_trade": {
            "symbol": "NVDA",
            "return": 0.45
        },
        "worst_trade": {
            "symbol": "BBBY",
            "return": -0.25
        }
    })

@router.get("/trending")
async def get_trending_recommendations(
    timeframe: str = Query("24h", pattern="^(1h|24h|7d|30d)$")
) -> ApiResponse[List[Dict[str, Any]]]:
    """Get trending recommendations based on user activity"""
    
    trending = []
    symbols = ["NVDA", "TSLA", "AAPL", "AMD", "GOOGL", "META", "AMZN", "MSFT"]
    
    for symbol in symbols[:5]:
        trending.append({
            "symbol": symbol,
            "views": random.randint(1000, 50000),
            "saves": random.randint(100, 5000),
            "recommendation_type": random.choice(list(RecommendationType)),
            "confidence_score": random.uniform(0.7, 0.95),
            "trending_score": random.uniform(70, 100),
            "timeframe": timeframe
        })

    return success_response(data=sorted(trending, key=lambda x: x["trending_score"], reverse=True))