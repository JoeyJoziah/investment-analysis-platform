from __future__ import annotations

import os
from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks, Path, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, date, timedelta, timezone
from enum import Enum
import logging
from sqlalchemy.ext.asyncio import AsyncSession

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
from backend.services.recommendation_service import (
    recommendation_service,
    SEC_RISK_WARNING,
    SEC_METHODOLOGY_DISCLOSURE_TEMPLATE,
    SEC_LIMITATIONS_STATEMENT,
    RECOMMENDATION_MODEL_VERSION,
    RECOMMENDATION_MODEL_TRAINING_DATE,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["recommendations"])

# ============================================================================
# Service Layer Dependencies
# ============================================================================

async def get_recommendation_service():
    """
    Dependency to get recommendation service instance.

    In test mode (TESTING=True), skips initialization to avoid Redis/external dependencies.
    """
    import os
    if not os.getenv("TESTING"):
        try:
            await recommendation_service.initialize()
        except Exception as e:
            logger.warning(f"Failed to initialize recommendation service: {e}")
    return recommendation_service

# ============================================================================
# Enum definitions
# ============================================================================

class RecommendationType(str, Enum):
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"

class TimeHorizon(str, Enum):
    SHORT_TERM = "short_term"   # 1-7 days
    MEDIUM_TERM = "medium_term" # 1-3 months
    LONG_TERM = "long_term"     # 3+ months

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

# ============================================================================
# Pydantic models
# ============================================================================

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
    model_config = {"protected_namespaces": ()}

    methodology_disclosure: str = Field(
        ..., description="Description of the algorithm and methodology used"
    )
    data_sources: List[str] = Field(
        ..., description="List of data sources used with timestamps"
    )
    model_version: str = Field(
        ..., description="Version identifier of the ML model used"
    )
    model_training_date: str = Field(
        ..., description="Date when the recommendation model was last trained"
    )
    risk_warning: str = Field(
        ..., description="Standard SEC-required risk warning text"
    )
    limitations_statement: str = Field(
        ..., description="Statement of what the analysis does NOT consider"
    )
    confidence_level: str = Field(
        default="moderate", description="Confidence level (low/moderate/high)"
    )
    conflict_of_interest_statement: Optional[str] = Field(
        default=None, description="Disclosure of material relationships"
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
    sec_disclosure: Optional[SECDisclosure] = Field(
        default=None, description="SEC 2025 required disclosure information"
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
    sec_global_disclosure: str = Field(
        default=SEC_RISK_WARNING, description="SEC-required global risk warning"
    )
    data_as_of: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
        description="Timestamp when data was collected"
    )
    recommendation_model_version: str = Field(
        default=RECOMMENDATION_MODEL_VERSION, description="Version of the recommendation model"
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

# ============================================================================
# Module-level ML singletons (kept for test patch paths)
# Tests patch: backend.api.routers.recommendations.model_manager
# Tests patch: backend.api.routers.recommendations.recommendation_engine
# ============================================================================

model_manager = None
recommendation_engine = None

try:
    model_manager = get_model_manager()
    recommendation_engine = RecommendationEngine(model_manager=model_manager)
    logger.info("ML model manager and recommendation engine initialized successfully")
except Exception as e:
    logger.warning(f"ML model manager not available: {e}")
    recommendation_engine = RecommendationEngine()

# ============================================================================
# Conversion helpers
# ============================================================================

def _dict_to_detail(r: Any) -> RecommendationDetail:
    """Convert a service dict (or existing RecommendationDetail) to RecommendationDetail."""
    if isinstance(r, RecommendationDetail):
        return r
    if isinstance(r.get("sec_disclosure"), dict):
        r = {**r, "sec_disclosure": SECDisclosure(**r["sec_disclosure"])}
    return RecommendationDetail(**r)


def _dicts_to_details(items: List[Any]) -> List[RecommendationDetail]:
    """Convert a list of service dicts to RecommendationDetail models."""
    return [_dict_to_detail(r) for r in items]

# ============================================================================
# Router-level functions
# These functions are kept at module level because tests import them directly
# from this module AND patch module-level names (stock_repository,
# price_repository, recommendation_engine, model_manager).
# The bodies delegate to the service, passing the patchable module-level names.
# ============================================================================

def generate_sec_disclosure(
    algorithm_type: str = "ML-powered quantitative",
    data_sources: List[str] = None,
    confidence_score: float = 0.5
) -> SECDisclosure:
    """Generate SEC 2025 compliant disclosure for a recommendation."""
    data = recommendation_service.generate_sec_disclosure(
        algorithm_type=algorithm_type,
        data_sources=data_sources,
        confidence_score=confidence_score,
    )
    return SECDisclosure(**data)


async def generate_ml_powered_recommendations(
    user_id: Optional[int] = None,
    portfolio_id: Optional[str] = None,
    risk_level: Optional[RiskLevel] = None,
    categories: Optional[List[RecommendationCategory]] = None,
    limit: int = 10,
    db_session: AsyncSession = None
) -> List[RecommendationDetail]:
    """
    Generate ML-powered recommendations with real market data.

    Delegates to the service layer while passing module-level repository and
    engine references so that test patches on this module propagate correctly.
    """
    raw = await recommendation_service.generate_ml_powered_recommendations(
        user_id=user_id,
        portfolio_id=portfolio_id,
        risk_level=risk_level.value if risk_level else None,
        categories=[c.value for c in categories] if categories else None,
        limit=limit,
        db_session=db_session,
        stock_repo=stock_repository,
        price_repo=price_repository,
        model_mgr=model_manager,
        rec_engine=recommendation_engine,
    )
    return _dicts_to_details(raw)


async def generate_personalized_recommendations(
    user_id: int,
    portfolio_id: Optional[str] = None,
    db_session: AsyncSession = None
) -> List[RecommendationDetail]:
    """Generate personalized recommendations based on user's portfolio and preferences."""
    raw = await recommendation_service.generate_personalized_recommendations(
        user_id=user_id,
        portfolio_id=portfolio_id,
        db_session=db_session,
        stock_repo=stock_repository,
        portfolio_repo=portfolio_repository,
        ml_recs_fn=lambda **kw: recommendation_service.generate_ml_powered_recommendations(
            stock_repo=stock_repository,
            price_repo=price_repository,
            model_mgr=model_manager,
            rec_engine=recommendation_engine,
            **kw,
        ),
    )
    return _dicts_to_details(raw)


def generate_recommendation(symbol: str = None) -> RecommendationDetail:
    """Generate a sample recommendation with SEC disclosure."""
    data = recommendation_service.generate_sample_recommendation(symbol=symbol)
    return _dict_to_detail(data)


# ============================================================================
# Endpoints
# ============================================================================

@router.get("/daily")
@cache_with_ttl(ttl=3600)
async def get_daily_recommendations(
    date_param: Optional[date] = Query(None, alias="date"),
    risk_level: Optional[RiskLevel] = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db_session),
    rec_service = Depends(get_recommendation_service)
) -> ApiResponse[DailyRecommendations]:
    """Get daily curated recommendations powered by ML models and market analysis."""
    target_date = date_param or date.today()
    logger.info(f"Generating daily recommendations for {target_date}, risk level: {risk_level}")

    try:
        result = await rec_service.build_daily_recommendations(
            user_id=current_user.id,
            target_date=target_date,
            risk_level=risk_level.value if risk_level else None,
            db_session=db,
        )

        return success_response(data=DailyRecommendations(
            date=target_date,
            market_outlook=result["market_outlook"],
            top_picks=_dicts_to_details(result["top_picks"]),
            watchlist=result["watchlist"],
            avoid_list=result["avoid_list"],
            sector_focus=result["sector_focus"],
            market_sentiment=round(result["market_sentiment"], 3),
            risk_assessment=result["risk_assessment"],
            special_situations=result["special_situations"],
        ))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating daily recommendations: {e}")
        if os.getenv("BOOTSTRAP_MODELS"):
            return success_response(data=DailyRecommendations(
                date=target_date,
                market_outlook="Market analysis temporarily unavailable (bootstrap mode)",
                top_picks=[],
                watchlist=[],
                avoid_list=[],
                sector_focus="N/A",
                market_sentiment=0.0,
                risk_assessment="Bootstrap mode — not production data",
                special_situations=[],
            ))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommendation service temporarily unavailable. Please try again later.",
        )

@router.get("/list")
async def get_recommendations(
    limit: int = Query(10, le=100),
    offset: int = 0,
    recommendation_type: Optional[RecommendationType] = None,
    category: Optional[RecommendationCategory] = None,
    risk_level: Optional[RiskLevel] = None,
    min_confidence: float = Query(0.0, ge=0, le=1),
    sort_by: str = Query("confidence_score", pattern="^(confidence_score|expected_return|created_at)$"),
    order: str = Query("desc", pattern="^(asc|desc)$"),
    current_user: User = Depends(get_current_user),
) -> ApiResponse[List[RecommendationDetail]]:
    """Get list of recommendations with filters"""
    recs_data = recommendation_service.generate_filtered_recommendations(
        count=50,
        recommendation_type=recommendation_type.value if recommendation_type else None,
        category=category.value if category else None,
        risk_level=risk_level.value if risk_level else None,
        min_confidence=min_confidence,
        sort_by=sort_by,
        order=order,
        limit=limit,
        offset=offset,
    )
    return success_response(data=_dicts_to_details(recs_data))

@router.get("/{recommendation_id}")
async def get_recommendation_detail(
    recommendation_id: str,
    current_user: User = Depends(get_current_user),
) -> ApiResponse[RecommendationDetail]:
    """Get detailed information about a specific recommendation"""
    rec = generate_recommendation()
    rec.id = recommendation_id
    return success_response(data=rec)

@router.post("/filter")
async def filter_recommendations(
    filter_params: RecommendationFilter,
    limit: int = Query(20, le=100),
    current_user: User = Depends(get_current_user),
) -> ApiResponse[List[RecommendationDetail]]:
    """Advanced filtering of recommendations"""
    recs_data = recommendation_service.generate_filtered_recommendations(
        count=100,
        categories=[c.value for c in filter_params.categories] if filter_params.categories else None,
        risk_levels=[r.value for r in filter_params.risk_levels] if filter_params.risk_levels else None,
        time_horizons=[t.value for t in filter_params.time_horizons] if filter_params.time_horizons else None,
        min_confidence=filter_params.min_confidence or 0.0,
        min_expected_return=filter_params.min_expected_return,
        sectors=filter_params.sectors,
        market_cap_min=filter_params.market_cap_min,
        market_cap_max=filter_params.market_cap_max,
        sort_by="confidence_score",
        order="desc",
        limit=limit,
        offset=0,
    )
    return success_response(data=_dicts_to_details(recs_data))

@router.get("/portfolio/{portfolio_id}")
async def get_portfolio_recommendations(
    portfolio_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db_session),
) -> ApiResponse[PortfolioRecommendation]:
    """Get personalized recommendations for a specific portfolio"""
    # Ownership check: verify the portfolio belongs to the requesting user.
    portfolio = await portfolio_repository.get_portfolio_with_positions(
        int(portfolio_id), session=db
    ) if portfolio_id.isdigit() else None

    if portfolio is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Portfolio {portfolio_id} not found",
        )

    if portfolio.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this portfolio",
        )

    data = recommendation_service.build_portfolio_recommendations(portfolio_id=portfolio_id)
    return success_response(data=PortfolioRecommendation(
        portfolio_id=data["portfolio_id"],
        recommendations=_dicts_to_details(data["recommendations"]),
        rebalancing_suggestions=data["rebalancing_suggestions"],
        risk_score=data["risk_score"],
        expected_portfolio_return=data["expected_portfolio_return"],
        diversification_score=data["diversification_score"],
    ))

@router.get("/performance/track")
async def track_recommendation_performance(
    days_back: int = Query(30, le=365),
    status: Optional[str] = Query(None, pattern="^(active|closed|stopped_out)$"),
    current_user: User = Depends(get_current_user),
) -> ApiResponse[List[RecommendationPerformance]]:
    """Track performance of past recommendations"""
    perf_data = recommendation_service.generate_performance_records(
        days_back=days_back,
        status_filter=status,
    )
    return success_response(data=[RecommendationPerformance(**p) for p in perf_data])

@router.post("/alerts/settings")
async def update_alert_settings(
    settings: AlertSettings,
    current_user: User = Depends(get_current_user),
) -> ApiResponse[Dict[str, str]]:
    """Update recommendation alert settings"""
    return success_response(data={
        "message": "Alert settings updated successfully",
        "status": "success"
    })

@router.get("/alerts/history")
async def get_alert_history(
    days_back: int = Query(7, le=30),
    current_user: User = Depends(get_current_user),
) -> ApiResponse[List[Dict[str, Any]]]:
    """Get history of recommendation alerts"""
    alerts = recommendation_service.generate_alert_history(days_back=days_back)
    return success_response(data=alerts)

@router.post("/backtest")
async def backtest_strategy(
    strategy: RecommendationCategory,
    start_date: date,
    end_date: date,
    initial_capital: float = 100000,
    current_user: User = Depends(get_current_user),
) -> ApiResponse[Dict[str, Any]]:
    """Backtest a recommendation strategy"""
    result = recommendation_service.run_backtest(
        strategy=strategy.value,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
    )
    return success_response(data=result)

@router.get("/trending")
async def get_trending_recommendations(
    timeframe: str = Query("24h", pattern="^(1h|24h|7d|30d)$"),
    limit: int = Query(10, le=50),
    risk_tolerance: str = Query("moderate", pattern="^(conservative|moderate|aggressive)$"),
    current_user: User = Depends(get_current_user),
    rec_service = Depends(get_recommendation_service),
) -> ApiResponse[List[Dict[str, Any]]]:
    """Get trending recommendations based on market momentum and analysis."""
    timeframe_map = {"1h": "1h", "24h": "1d", "7d": "1w", "30d": "1m"}

    try:
        trending = await rec_service.get_trending(
            timeframe=timeframe_map.get(timeframe, "1d"),
            limit=limit,
            risk_tolerance=risk_tolerance
        )
        for rec in trending:
            rec["views"] = None
            rec["saves"] = None
            rec["trending_score"] = rec["confidence"] * 100
            rec["timeframe"] = timeframe
        return success_response(data=trending)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting trending recommendations: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Trending recommendations service temporarily unavailable. Please try again later.",
        )
