from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, date, timedelta
from enum import Enum
import random
import asyncio

router = APIRouter(prefix="/recommendations", tags=["recommendations"])

# Enums
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

# Pydantic models
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

# Sample data generator functions
def generate_recommendation(symbol: str = None) -> RecommendationDetail:
    """Generate a sample recommendation"""
    if not symbol:
        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "META", "NVDA", "TSLA", "JPM", "V", "JNJ"]
        symbol = random.choice(symbols)
    
    current_price = random.uniform(50, 500)
    target_price = current_price * random.uniform(0.9, 1.3)
    
    return RecommendationDetail(
        id=f"REC-{random.randint(1000, 9999)}",
        symbol=symbol,
        company_name=f"{symbol} Inc.",
        recommendation_type=random.choice(list(RecommendationType)),
        category=random.choice(list(RecommendationCategory)),
        confidence_score=random.uniform(0.6, 0.95),
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
        similar_stocks=["GOOG", "FB", "NFLX"] if symbol != "GOOGL" else ["AAPL", "MSFT"]
    )

# Endpoints
@router.get("/daily", response_model=DailyRecommendations)
async def get_daily_recommendations(
    date_param: Optional[date] = Query(None, alias="date"),
    risk_level: Optional[RiskLevel] = None
) -> DailyRecommendations:
    """Get daily curated recommendations"""
    
    target_date = date_param or date.today()
    
    # Generate top picks based on risk level
    num_picks = 5 if risk_level == RiskLevel.CONSERVATIVE else 8 if risk_level == RiskLevel.MODERATE else 10
    top_picks = [generate_recommendation() for _ in range(num_picks)]
    
    # Filter by risk level if specified
    if risk_level:
        top_picks = [r for r in top_picks if r.risk_level == risk_level or random.random() > 0.3]
    
    return DailyRecommendations(
        date=target_date,
        market_outlook="Bullish with caution - Fed meeting ahead",
        top_picks=top_picks[:5],  # Limit to top 5
        watchlist=["AAPL", "GOOGL", "MSFT", "NVDA", "AMD"],
        avoid_list=["BBBY", "AMC", "GME"],
        sector_focus="Technology",
        market_sentiment=0.35,
        risk_assessment="Moderate - Elevated VIX levels",
        special_situations=[
            {
                "type": "earnings_play",
                "symbol": "AAPL",
                "event_date": target_date + timedelta(days=3),
                "expected_move": 0.05
            }
        ]
    )

@router.get("/list", response_model=List[RecommendationDetail])
async def get_recommendations(
    limit: int = Query(10, le=100),
    offset: int = 0,
    recommendation_type: Optional[RecommendationType] = None,
    category: Optional[RecommendationCategory] = None,
    risk_level: Optional[RiskLevel] = None,
    min_confidence: float = Query(0.0, ge=0, le=1),
    sort_by: str = Query("confidence_score", regex="^(confidence_score|expected_return|created_at)$"),
    order: str = Query("desc", regex="^(asc|desc)$")
) -> List[RecommendationDetail]:
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
    return recommendations[offset:offset + limit]

@router.get("/{recommendation_id}", response_model=RecommendationDetail)
async def get_recommendation_detail(recommendation_id: str) -> RecommendationDetail:
    """Get detailed information about a specific recommendation"""
    
    # Generate a recommendation with the specified ID
    recommendation = generate_recommendation()
    recommendation.id = recommendation_id
    
    return recommendation

@router.post("/filter", response_model=List[RecommendationDetail])
async def filter_recommendations(
    filter_params: RecommendationFilter,
    limit: int = Query(20, le=100)
) -> List[RecommendationDetail]:
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
    
    return recommendations[:limit]

@router.get("/portfolio/{portfolio_id}", response_model=PortfolioRecommendation)
async def get_portfolio_recommendations(portfolio_id: str) -> PortfolioRecommendation:
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
    
    return PortfolioRecommendation(
        portfolio_id=portfolio_id,
        recommendations=recommendations,
        rebalancing_suggestions=rebalancing,
        risk_score=random.uniform(30, 70),
        expected_portfolio_return=random.uniform(0.08, 0.15),
        diversification_score=random.uniform(0.6, 0.9)
    )

@router.get("/performance/track", response_model=List[RecommendationPerformance])
async def track_recommendation_performance(
    days_back: int = Query(30, le=365),
    status: Optional[str] = Query(None, regex="^(active|closed|stopped_out)$")
) -> List[RecommendationPerformance]:
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
    
    return performances

@router.post("/alerts/settings")
async def update_alert_settings(settings: AlertSettings) -> Dict[str, str]:
    """Update recommendation alert settings"""
    
    # In production, this would save to user preferences
    return {
        "message": "Alert settings updated successfully",
        "status": "success"
    }

@router.get("/alerts/history")
async def get_alert_history(
    days_back: int = Query(7, le=30)
) -> List[Dict[str, Any]]:
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
    
    return sorted(alerts, key=lambda x: x["timestamp"], reverse=True)

@router.post("/backtest")
async def backtest_strategy(
    strategy: RecommendationCategory,
    start_date: date,
    end_date: date,
    initial_capital: float = 100000
) -> Dict[str, Any]:
    """Backtest a recommendation strategy"""
    
    # Simulate backtest results
    total_return = random.uniform(-0.2, 0.5)
    
    return {
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
    }

@router.get("/trending")
async def get_trending_recommendations(
    timeframe: str = Query("24h", regex="^(1h|24h|7d|30d)$")
) -> List[Dict[str, Any]]:
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
    
    return sorted(trending, key=lambda x: x["trending_score"], reverse=True)