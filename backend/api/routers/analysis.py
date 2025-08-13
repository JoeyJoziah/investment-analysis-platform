from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta, date
from enum import Enum
import asyncio
import random

router = APIRouter(prefix="/analysis", tags=["analysis"])

# Enums
class AnalysisType(str, Enum):
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    SENTIMENT = "sentiment"
    COMPREHENSIVE = "comprehensive"
    QUICK = "quick"

class Indicator(str, Enum):
    RSI = "rsi"
    MACD = "macd"
    BOLLINGER_BANDS = "bollinger_bands"
    MOVING_AVERAGE = "moving_average"
    VOLUME = "volume"
    STOCHASTIC = "stochastic"
    ADX = "adx"
    ATR = "atr"

class SignalStrength(str, Enum):
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    NEUTRAL = "neutral"
    SELL = "sell"
    STRONG_SELL = "strong_sell"

# Pydantic models
class AnalysisRequest(BaseModel):
    symbol: str
    analysis_type: AnalysisType = AnalysisType.COMPREHENSIVE
    period: str = Field(default="1M", pattern="^(1D|1W|1M|3M|6M|1Y|5Y)$")
    indicators: Optional[List[Indicator]] = None
    include_ml_predictions: bool = True
    include_news_sentiment: bool = True

class TechnicalIndicators(BaseModel):
    rsi: Optional[float] = Field(None, ge=0, le=100)
    rsi_signal: Optional[SignalStrength] = None
    macd: Optional[Dict[str, float]] = None
    macd_signal: Optional[SignalStrength] = None
    moving_averages: Optional[Dict[str, float]] = None
    bollinger_bands: Optional[Dict[str, float]] = None
    volume_analysis: Optional[Dict[str, Any]] = None
    support_levels: Optional[List[float]] = None
    resistance_levels: Optional[List[float]] = None
    trend: Optional[str] = None
    volatility: Optional[float] = None

class FundamentalMetrics(BaseModel):
    pe_ratio: Optional[float] = None
    peg_ratio: Optional[float] = None
    eps: Optional[float] = None
    revenue_growth: Optional[float] = None
    profit_margin: Optional[float] = None
    debt_to_equity: Optional[float] = None
    roe: Optional[float] = None
    current_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    market_cap: Optional[float] = None
    enterprise_value: Optional[float] = None
    book_value: Optional[float] = None
    intrinsic_value: Optional[float] = None
    valuation_score: Optional[float] = Field(None, ge=0, le=100)

class SentimentAnalysis(BaseModel):
    overall_sentiment: float = Field(..., ge=-1, le=1)
    sentiment_label: str
    news_sentiment: Optional[float] = Field(None, ge=-1, le=1)
    social_sentiment: Optional[float] = Field(None, ge=-1, le=1)
    analyst_sentiment: Optional[float] = Field(None, ge=-1, le=1)
    insider_sentiment: Optional[float] = Field(None, ge=-1, le=1)
    sentiment_momentum: Optional[str] = None
    key_topics: Optional[List[str]] = None
    sentiment_sources: Optional[Dict[str, int]] = None

class MLPredictions(BaseModel):
    price_prediction_1d: Optional[float] = None
    price_prediction_7d: Optional[float] = None
    price_prediction_30d: Optional[float] = None
    confidence_score: float = Field(..., ge=0, le=1)
    predicted_volatility: Optional[float] = None
    risk_score: float = Field(..., ge=0, le=100)
    pattern_recognition: Optional[List[str]] = None
    anomaly_detection: Optional[bool] = None
    trend_prediction: Optional[str] = None

class RiskMetrics(BaseModel):
    beta: Optional[float] = None
    alpha: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    var_95: Optional[float] = None
    cvar_95: Optional[float] = None
    correlation_with_market: Optional[float] = None
    risk_adjusted_return: Optional[float] = None
    overall_risk_score: float = Field(..., ge=0, le=100)

class AnalysisResponse(BaseModel):
    symbol: str
    timestamp: datetime
    analysis_type: AnalysisType
    technical: Optional[TechnicalIndicators] = None
    fundamental: Optional[FundamentalMetrics] = None
    sentiment: Optional[SentimentAnalysis] = None
    ml_predictions: Optional[MLPredictions] = None
    risk_metrics: Optional[RiskMetrics] = None
    overall_score: float = Field(..., ge=0, le=100)
    recommendation: SignalStrength
    confidence: float = Field(..., ge=0, le=1)
    key_insights: List[str]
    warnings: Optional[List[str]] = None
    next_earnings_date: Optional[date] = None
    last_updated: datetime

class BatchAnalysisRequest(BaseModel):
    symbols: List[str] = Field(..., min_items=1, max_items=50)
    analysis_type: AnalysisType = AnalysisType.QUICK
    compare: bool = False

class ComparisonResult(BaseModel):
    symbols: List[str]
    comparison_metrics: Dict[str, Dict[str, Any]]
    best_performer: str
    recommendations: Dict[str, SignalStrength]
    correlation_matrix: Optional[Dict[str, Dict[str, float]]] = None

# Utility functions
def calculate_rsi(prices: List[float], period: int = 14) -> float:
    """Calculate Relative Strength Index"""
    # Simplified RSI calculation for demonstration
    return random.uniform(30, 70)

def calculate_macd(prices: List[float]) -> Dict[str, float]:
    """Calculate MACD indicator"""
    return {
        "macd": random.uniform(-2, 2),
        "signal": random.uniform(-2, 2),
        "histogram": random.uniform(-1, 1)
    }

def analyze_sentiment(text_data: List[str]) -> float:
    """Analyze sentiment from text data"""
    # Placeholder for sentiment analysis
    return random.uniform(-0.5, 0.5)

def generate_insights(analysis: Dict) -> List[str]:
    """Generate key insights from analysis"""
    insights = []
    
    if analysis.get("technical", {}).get("rsi", 0) > 70:
        insights.append("RSI indicates overbought conditions - potential pullback ahead")
    elif analysis.get("technical", {}).get("rsi", 0) < 30:
        insights.append("RSI indicates oversold conditions - potential bounce opportunity")
    
    if analysis.get("fundamental", {}).get("pe_ratio", 0) < 15:
        insights.append("Stock appears undervalued based on P/E ratio")
    
    if analysis.get("sentiment", {}).get("overall_sentiment", 0) > 0.5:
        insights.append("Strong positive sentiment detected in recent news and social media")
    
    if not insights:
        insights.append("Stock showing neutral signals across indicators")
    
    return insights

# Endpoints
@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_stock(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """Perform comprehensive analysis on a single stock"""
    
    # Initialize response components
    technical = None
    fundamental = None
    sentiment = None
    ml_predictions = None
    risk_metrics = None
    
    # Technical Analysis
    if request.analysis_type in [AnalysisType.TECHNICAL, AnalysisType.COMPREHENSIVE]:
        technical = TechnicalIndicators(
            rsi=calculate_rsi([]),
            rsi_signal=SignalStrength.NEUTRAL,
            macd=calculate_macd([]),
            macd_signal=SignalStrength.BUY,
            moving_averages={
                "sma_20": 150.5,
                "sma_50": 148.2,
                "sma_200": 145.0,
                "ema_12": 151.0,
                "ema_26": 149.5
            },
            bollinger_bands={
                "upper": 155.0,
                "middle": 150.0,
                "lower": 145.0
            },
            volume_analysis={
                "current_volume": 50000000,
                "avg_volume": 45000000,
                "volume_trend": "increasing"
            },
            support_levels=[145.0, 142.0, 140.0],
            resistance_levels=[155.0, 158.0, 160.0],
            trend="bullish",
            volatility=0.25
        )
    
    # Fundamental Analysis
    if request.analysis_type in [AnalysisType.FUNDAMENTAL, AnalysisType.COMPREHENSIVE]:
        fundamental = FundamentalMetrics(
            pe_ratio=25.5,
            peg_ratio=1.8,
            eps=6.5,
            revenue_growth=0.15,
            profit_margin=0.22,
            debt_to_equity=0.45,
            roe=0.28,
            current_ratio=1.8,
            dividend_yield=0.015,
            market_cap=2500000000000,
            enterprise_value=2600000000000,
            book_value=45.0,
            intrinsic_value=165.0,
            valuation_score=72.5
        )
    
    # Sentiment Analysis
    if request.analysis_type in [AnalysisType.SENTIMENT, AnalysisType.COMPREHENSIVE]:
        sentiment = SentimentAnalysis(
            overall_sentiment=0.35,
            sentiment_label="Positive",
            news_sentiment=0.45,
            social_sentiment=0.25,
            analyst_sentiment=0.40,
            insider_sentiment=0.30,
            sentiment_momentum="improving",
            key_topics=["earnings beat", "product launch", "market expansion"],
            sentiment_sources={"news": 150, "social": 5000, "analysts": 25}
        )
    
    # ML Predictions
    if request.include_ml_predictions:
        ml_predictions = MLPredictions(
            price_prediction_1d=152.5,
            price_prediction_7d=155.0,
            price_prediction_30d=160.0,
            confidence_score=0.75,
            predicted_volatility=0.22,
            risk_score=35.0,
            pattern_recognition=["ascending triangle", "bullish flag"],
            anomaly_detection=False,
            trend_prediction="upward"
        )
    
    # Risk Metrics
    risk_metrics = RiskMetrics(
        beta=1.15,
        alpha=0.02,
        sharpe_ratio=1.85,
        sortino_ratio=2.10,
        max_drawdown=-0.15,
        var_95=-0.025,
        cvar_95=-0.035,
        correlation_with_market=0.75,
        risk_adjusted_return=0.18,
        overall_risk_score=42.0
    )
    
    # Calculate overall score and recommendation
    overall_score = random.uniform(60, 85)
    
    if overall_score >= 80:
        recommendation = SignalStrength.STRONG_BUY
    elif overall_score >= 65:
        recommendation = SignalStrength.BUY
    elif overall_score >= 35:
        recommendation = SignalStrength.NEUTRAL
    elif overall_score >= 20:
        recommendation = SignalStrength.SELL
    else:
        recommendation = SignalStrength.STRONG_SELL
    
    # Generate insights
    analysis_data = {
        "technical": technical.dict() if technical else {},
        "fundamental": fundamental.dict() if fundamental else {},
        "sentiment": sentiment.dict() if sentiment else {}
    }
    insights = generate_insights(analysis_data)
    
    # Background task to cache results
    background_tasks.add_task(cache_analysis_results, request.symbol, overall_score)
    
    return AnalysisResponse(
        symbol=request.symbol.upper(),
        timestamp=datetime.utcnow(),
        analysis_type=request.analysis_type,
        technical=technical,
        fundamental=fundamental,
        sentiment=sentiment,
        ml_predictions=ml_predictions,
        risk_metrics=risk_metrics,
        overall_score=overall_score,
        recommendation=recommendation,
        confidence=0.75,
        key_insights=insights,
        warnings=["High volatility expected around earnings"] if random.random() > 0.7 else None,
        next_earnings_date=datetime.utcnow().date() + timedelta(days=random.randint(10, 90)),
        last_updated=datetime.utcnow()
    )

@router.post("/batch", response_model=List[AnalysisResponse])
async def batch_analysis(request: BatchAnalysisRequest):
    """Analyze multiple stocks at once"""
    
    results = []
    for symbol in request.symbols:
        # Create individual analysis request
        analysis_req = AnalysisRequest(
            symbol=symbol,
            analysis_type=request.analysis_type,
            include_ml_predictions=False,
            include_news_sentiment=False
        )
        
        # Perform quick analysis for each symbol
        # In production, this would be parallelized
        result = await analyze_stock(analysis_req, BackgroundTasks())
        results.append(result)
    
    return results

@router.post("/compare", response_model=ComparisonResult)
async def compare_stocks(request: BatchAnalysisRequest):
    """Compare multiple stocks side by side"""
    
    if len(request.symbols) < 2:
        raise HTTPException(status_code=400, detail="At least 2 symbols required for comparison")
    
    comparison_metrics = {}
    recommendations = {}
    
    for symbol in request.symbols:
        comparison_metrics[symbol] = {
            "score": random.uniform(50, 90),
            "pe_ratio": random.uniform(10, 40),
            "rsi": random.uniform(30, 70),
            "sentiment": random.uniform(-0.5, 0.5),
            "risk_score": random.uniform(20, 80),
            "expected_return": random.uniform(-0.1, 0.3)
        }
        
        score = comparison_metrics[symbol]["score"]
        if score >= 75:
            recommendations[symbol] = SignalStrength.BUY
        elif score >= 50:
            recommendations[symbol] = SignalStrength.NEUTRAL
        else:
            recommendations[symbol] = SignalStrength.SELL
    
    # Determine best performer
    best_performer = max(comparison_metrics.keys(), 
                        key=lambda x: comparison_metrics[x]["score"])
    
    # Generate correlation matrix if requested
    correlation_matrix = None
    if len(request.symbols) <= 10:  # Limit correlation matrix for performance
        correlation_matrix = {}
        for symbol1 in request.symbols:
            correlation_matrix[symbol1] = {}
            for symbol2 in request.symbols:
                if symbol1 == symbol2:
                    correlation_matrix[symbol1][symbol2] = 1.0
                else:
                    correlation_matrix[symbol1][symbol2] = random.uniform(-0.5, 0.9)
    
    return ComparisonResult(
        symbols=request.symbols,
        comparison_metrics=comparison_metrics,
        best_performer=best_performer,
        recommendations=recommendations,
        correlation_matrix=correlation_matrix
    )

@router.get("/indicators/{symbol}")
async def get_technical_indicators(
    symbol: str,
    indicators: List[Indicator] = Query(None)
) -> Dict[str, Any]:
    """Get specific technical indicators for a stock"""
    
    result = {}
    
    if not indicators:
        indicators = list(Indicator)
    
    for indicator in indicators:
        if indicator == Indicator.RSI:
            result["rsi"] = {
                "value": random.uniform(30, 70),
                "signal": "neutral",
                "overbought": 70,
                "oversold": 30
            }
        elif indicator == Indicator.MACD:
            result["macd"] = {
                "macd": random.uniform(-2, 2),
                "signal": random.uniform(-2, 2),
                "histogram": random.uniform(-1, 1),
                "trend": "bullish" if random.random() > 0.5 else "bearish"
            }
        elif indicator == Indicator.BOLLINGER_BANDS:
            result["bollinger_bands"] = {
                "upper": 155.0,
                "middle": 150.0,
                "lower": 145.0,
                "bandwidth": 10.0,
                "percent_b": 0.6
            }
        elif indicator == Indicator.MOVING_AVERAGE:
            result["moving_averages"] = {
                "sma_20": 150.5,
                "sma_50": 148.2,
                "sma_200": 145.0,
                "ema_12": 151.0,
                "ema_26": 149.5,
                "golden_cross": False,
                "death_cross": False
            }
    
    return {
        "symbol": symbol.upper(),
        "timestamp": datetime.utcnow().isoformat(),
        "indicators": result
    }

@router.get("/sentiment/{symbol}")
async def get_sentiment_analysis(symbol: str) -> SentimentAnalysis:
    """Get detailed sentiment analysis for a stock"""
    
    return SentimentAnalysis(
        overall_sentiment=random.uniform(-0.5, 0.5),
        sentiment_label="Positive" if random.random() > 0.5 else "Neutral",
        news_sentiment=random.uniform(-0.5, 0.5),
        social_sentiment=random.uniform(-0.5, 0.5),
        analyst_sentiment=random.uniform(-0.5, 0.5),
        insider_sentiment=random.uniform(-0.5, 0.5),
        sentiment_momentum="improving" if random.random() > 0.5 else "stable",
        key_topics=["earnings", "product launch", "market conditions"],
        sentiment_sources={"news": 100, "social": 5000, "analysts": 20}
    )

# Background task function
async def cache_analysis_results(symbol: str, score: float):
    """Cache analysis results for quick retrieval"""
    # In production, this would save to Redis or database
    await asyncio.sleep(0.1)  # Simulate async operation
    print(f"Cached analysis for {symbol} with score {score}")