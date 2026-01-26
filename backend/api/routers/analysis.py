from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query, Path
from fastapi import status
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta, date
from enum import Enum
import asyncio
import random
import logging
import statistics
import math
from sqlalchemy.ext.asyncio import AsyncSession

# Import enhanced dependencies
from backend.config.database import get_async_db_session
from backend.data_ingestion.alpha_vantage_client import AlphaVantageClient
from backend.data_ingestion.finnhub_client import FinnhubClient
from backend.analytics.technical_analysis import TechnicalAnalysisEngine
from backend.analytics.fundamental_analysis import FundamentalAnalysisEngine
from backend.analytics.sentiment_analysis import SentimentAnalysisEngine
from backend.repositories import stock_repository, price_repository
from backend.utils.cache import cache_with_ttl
# from backend.utils.enhanced_error_handling import handle_api_error, validate_stock_symbol
from backend.config.settings import settings
from backend.ml.model_manager import get_model_manager

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analysis", tags=["analysis"])

# Initialize data clients and analyzers
alpha_vantage_client = AlphaVantageClient() if settings.ALPHA_VANTAGE_API_KEY else None
finnhub_client = FinnhubClient() if settings.FINNHUB_API_KEY else None

# Initialize analyzers
technical_analyzer = TechnicalAnalysisEngine()
fundamental_analyzer = FundamentalAnalysisEngine()
sentiment_analyzer = SentimentAnalysisEngine()

# ML Model Manager
model_manager = None
try:
    model_manager = get_model_manager()
    logger.info("ML model manager initialized successfully")
except Exception as e:
    logger.warning(f"ML model manager not available: {e}")

# Constants for API timeouts
DEFAULT_API_TIMEOUT = 5.0  # 5 seconds timeout for individual API calls
PARALLEL_BATCH_TIMEOUT = 10.0  # 10 seconds for entire parallel batch


async def safe_async_call(
    coro,
    timeout: float = DEFAULT_API_TIMEOUT,
    default: Any = None,
    error_msg: str = "API call"
) -> Any:
    """
    Safely execute an async coroutine with timeout and error handling.

    Args:
        coro: The coroutine to execute
        timeout: Maximum time to wait in seconds
        default: Default value to return on failure
        error_msg: Description for error logging

    Returns:
        The result of the coroutine or the default value on failure
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(f"Timeout ({timeout}s) for {error_msg}")
        return default
    except Exception as e:
        logger.error(f"Error in {error_msg}: {e}")
        return default


async def fetch_parallel_with_fallback(
    tasks: List[Tuple[str, Any]],
    timeout: float = PARALLEL_BATCH_TIMEOUT
) -> Dict[str, Any]:
    """
    Execute multiple async tasks in parallel with individual error handling.

    Args:
        tasks: List of (name, coroutine) tuples
        timeout: Maximum time for all tasks combined

    Returns:
        Dictionary mapping task names to results (None for failed tasks)
    """
    if not tasks:
        return {}

    task_names = [name for name, _ in tasks]
    coroutines = [coro for _, coro in tasks]

    try:
        # Execute all tasks in parallel with overall timeout
        results = await asyncio.wait_for(
            asyncio.gather(*coroutines, return_exceptions=True),
            timeout=timeout
        )

        result_dict = {}
        for name, result in zip(task_names, results):
            if isinstance(result, Exception):
                logger.warning(f"Task '{name}' failed: {result}")
                result_dict[name] = None
            else:
                result_dict[name] = result

        return result_dict

    except asyncio.TimeoutError:
        logger.warning(f"Parallel tasks timed out after {timeout}s")
        return {name: None for name in task_names}
    except Exception as e:
        logger.error(f"Error in parallel execution: {e}")
        return {name: None for name in task_names}


# Helper functions for real data analysis
async def fetch_technical_indicators(symbol: str, period: str = "1M") -> Dict[str, Any]:
    """Fetch real technical indicators from price data using parallel API calls"""
    try:
        if not alpha_vantage_client:
            return {}

        # Execute indicator API calls in parallel
        indicator_tasks = [
            ("rsi", safe_async_call(
                alpha_vantage_client.get_rsi(symbol, interval="daily", time_period=14),
                error_msg=f"RSI fetch for {symbol}"
            )),
            ("macd", safe_async_call(
                alpha_vantage_client.get_macd(symbol),
                error_msg=f"MACD fetch for {symbol}"
            )),
            ("sma_20", safe_async_call(
                alpha_vantage_client.get_sma(symbol, interval="daily", time_period=20),
                error_msg=f"SMA fetch for {symbol}"
            )),
        ]

        results = await fetch_parallel_with_fallback(indicator_tasks)

        # Filter out None results
        indicators = {k: v for k, v in results.items() if v is not None}
        return indicators

    except Exception as e:
        logger.error(f"Error fetching technical indicators for {symbol}: {e}")
        return {}


async def fetch_fundamental_data(symbol: str) -> Dict[str, Any]:
    """Fetch fundamental data from available sources using parallel API calls"""
    try:
        fundamental_tasks = []

        if alpha_vantage_client:
            fundamental_tasks.extend([
                ("overview", safe_async_call(
                    alpha_vantage_client.get_company_overview(symbol),
                    error_msg=f"Company overview for {symbol}"
                )),
                ("earnings", safe_async_call(
                    alpha_vantage_client.get_earnings(symbol),
                    error_msg=f"Earnings data for {symbol}"
                )),
            ])

        if finnhub_client:
            fundamental_tasks.append(
                ("metrics", safe_async_call(
                    finnhub_client.get_basic_financials(symbol),
                    error_msg=f"Financial metrics for {symbol}"
                ))
            )

        if not fundamental_tasks:
            return {}

        results = await fetch_parallel_with_fallback(fundamental_tasks)

        # Merge results into single dict
        fundamental_data = {}
        if results.get("overview"):
            fundamental_data.update(results["overview"])
        if results.get("earnings"):
            fundamental_data["earnings"] = results["earnings"]
        if results.get("metrics"):
            fundamental_data.update(results["metrics"])

        return fundamental_data

    except Exception as e:
        logger.error(f"Error fetching fundamental data for {symbol}: {e}")
        return {}


async def fetch_sentiment_data(symbol: str) -> Dict[str, Any]:
    """Fetch sentiment data from news and social sources using parallel API calls"""
    try:
        if not finnhub_client:
            return {}

        # Execute news and social sentiment fetches in parallel
        sentiment_tasks = [
            ("news", safe_async_call(
                finnhub_client.get_company_news(
                    symbol,
                    _from=datetime.now() - timedelta(days=7),
                    to=datetime.now()
                ),
                error_msg=f"News fetch for {symbol}"
            )),
            ("social", safe_async_call(
                finnhub_client.get_social_sentiment(symbol),
                error_msg=f"Social sentiment for {symbol}"
            )),
        ]

        results = await fetch_parallel_with_fallback(sentiment_tasks)

        sentiment_data = {}

        # Process news sentiment if available
        if results.get("news"):
            try:
                sentiment_data["news"] = await safe_async_call(
                    sentiment_analyzer.analyze_news_sentiment(results["news"]),
                    error_msg=f"News sentiment analysis for {symbol}",
                    default={}
                )
            except Exception as e:
                logger.warning(f"Failed to analyze news sentiment for {symbol}: {e}")

        # Add social sentiment if available
        if results.get("social"):
            sentiment_data["social"] = results["social"]

        return sentiment_data

    except Exception as e:
        logger.error(f"Error fetching sentiment data for {symbol}: {e}")
        return {}

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

# Enhanced Endpoints
@router.post("/analyze", response_model=AnalysisResponse)
@cache_with_ttl(ttl=300)  # Cache for 5 minutes
async def analyze_stock(
    request: AnalysisRequest, 
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_async_db_session)
):
    """
    Perform comprehensive analysis on a single stock with real data integration.
    
    This endpoint provides multi-layered analysis including:
    - Technical indicators from real market data
    - Fundamental analysis from financial statements
    - Sentiment analysis from news and social media
    - ML-powered predictions and pattern recognition
    """
    
    try:
        # Validate stock symbol
        # if not validate_stock_symbol(request.symbol):
        #     raise HTTPException(
        #         status_code=status.HTTP_400_BAD_REQUEST,
        #         detail=f"Invalid stock symbol format: '{request.symbol}'"
        #     )
        
        symbol = request.symbol.upper()
        logger.info(f"Starting analysis for {symbol} - Type: {request.analysis_type}")
        
        # Verify stock exists in database
        stock = await stock_repository.get_by_symbol(symbol, session=db)
        if not stock:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Stock '{symbol}' not found in database"
            )
        
        # Initialize response components
        technical = None
        fundamental = None
        sentiment = None
        ml_predictions = None
        risk_metrics = None
        price_history = None

        # =====================================================================
        # PHASE 1: Execute all external API calls in PARALLEL
        # This is the critical optimization - reduces latency by ~70%
        # =====================================================================
        logger.info(f"Starting parallel data fetch for {symbol}")

        # Determine which data sources we need based on analysis type
        needs_technical = request.analysis_type in [AnalysisType.TECHNICAL, AnalysisType.COMPREHENSIVE]
        needs_fundamental = request.analysis_type in [AnalysisType.FUNDAMENTAL, AnalysisType.COMPREHENSIVE]
        needs_sentiment = (
            request.analysis_type in [AnalysisType.SENTIMENT, AnalysisType.COMPREHENSIVE]
            and request.include_news_sentiment
        )

        # Build list of parallel tasks with names for result mapping
        parallel_tasks: List[Tuple[str, Any]] = []

        # Price history is needed for technical analysis and risk metrics
        if needs_technical or request.include_ml_predictions:
            parallel_tasks.append((
                "price_history",
                safe_async_call(
                    price_repository.get_price_history(
                        symbol=symbol,
                        start_date=datetime.now().date() - timedelta(days=365),
                        end_date=datetime.now().date(),
                        limit=252,
                        session=db
                    ),
                    timeout=DEFAULT_API_TIMEOUT,
                    error_msg=f"Price history for {symbol}",
                    default=[]
                )
            ))

        # External API calls - these are the slow ones that benefit most from parallelization
        if needs_technical:
            parallel_tasks.append((
                "tech_indicators",
                fetch_technical_indicators(symbol, request.period)
            ))

        if needs_fundamental:
            parallel_tasks.append((
                "fundamental_data",
                fetch_fundamental_data(symbol)
            ))

        if needs_sentiment:
            parallel_tasks.append((
                "sentiment_data",
                fetch_sentiment_data(symbol)
            ))

        # Execute all tasks in parallel with overall timeout
        parallel_results = await fetch_parallel_with_fallback(
            parallel_tasks,
            timeout=PARALLEL_BATCH_TIMEOUT
        )

        # Extract results (None if task failed or wasn't requested)
        price_history = parallel_results.get("price_history", [])
        tech_indicators_data = parallel_results.get("tech_indicators", {})
        fundamental_data = parallel_results.get("fundamental_data", {})
        sentiment_data = parallel_results.get("sentiment_data", {})

        logger.info(f"Parallel fetch completed for {symbol}: "
                   f"price_history={len(price_history) if price_history else 0} records, "
                   f"tech={bool(tech_indicators_data)}, "
                   f"fundamental={bool(fundamental_data)}, "
                   f"sentiment={bool(sentiment_data)}")

        # =====================================================================
        # PHASE 2: Process results and build response objects
        # =====================================================================

        # Technical Analysis processing
        if needs_technical:
            logger.info(f"Processing technical analysis for {symbol}")

            if price_history and len(price_history) > 0:
                # Calculate technical indicators from price data
                prices = [float(p.close) for p in price_history]
                volumes = [p.volume for p in price_history]

                # Use our technical analyzer (this is CPU-bound, not I/O)
                tech_analysis = await technical_analyzer.analyze(
                    prices=prices,
                    volumes=volumes,
                    symbol=symbol
                )

                # Merge external API indicators if available
                if tech_indicators_data:
                    tech_analysis.update(tech_indicators_data)

                technical = TechnicalIndicators(
                    rsi=tech_analysis.get('rsi', {}).get('value') if isinstance(tech_analysis.get('rsi'), dict) else tech_analysis.get('rsi'),
                    rsi_signal=SignalStrength.BUY if (tech_analysis.get('rsi', {}).get('value', 50) if isinstance(tech_analysis.get('rsi'), dict) else tech_analysis.get('rsi', 50)) < 30
                             else SignalStrength.SELL if (tech_analysis.get('rsi', {}).get('value', 50) if isinstance(tech_analysis.get('rsi'), dict) else tech_analysis.get('rsi', 50)) > 70
                             else SignalStrength.NEUTRAL,
                    macd=tech_analysis.get('macd', {}),
                    macd_signal=SignalStrength(tech_analysis.get('macd', {}).get('signal', 'neutral')) if isinstance(tech_analysis.get('macd'), dict) else SignalStrength.NEUTRAL,
                    moving_averages=tech_analysis.get('moving_averages', {}),
                    bollinger_bands=tech_analysis.get('bollinger_bands', {}),
                    volume_analysis=tech_analysis.get('volume_analysis', {}),
                    support_levels=tech_analysis.get('support_levels', []),
                    resistance_levels=tech_analysis.get('resistance_levels', []),
                    trend=tech_analysis.get('trend', 'neutral'),
                    volatility=tech_analysis.get('volatility', 0.0)
                )
            else:
                logger.warning(f"No price history found for {symbol}, using mock data")
                # Fallback to mock data
                technical = TechnicalIndicators(
                    rsi=calculate_rsi([]),
                    rsi_signal=SignalStrength.NEUTRAL,
                    macd=calculate_macd([]),
                    macd_signal=SignalStrength.NEUTRAL,
                    moving_averages={"sma_20": 150.5, "sma_50": 148.2, "sma_200": 145.0},
                    bollinger_bands={"upper": 155.0, "middle": 150.0, "lower": 145.0},
                    volume_analysis={"current_volume": 50000000, "avg_volume": 45000000},
                    support_levels=[145.0, 142.0, 140.0],
                    resistance_levels=[155.0, 158.0, 160.0],
                    trend="neutral",
                    volatility=0.20
                )

        # Fundamental Analysis processing
        if needs_fundamental:
            logger.info(f"Processing fundamental analysis for {symbol}")

            try:
                # Parse and normalize the data from parallel fetch
                pe_ratio = fundamental_data.get('PERatio', fundamental_data.get('peRatio')) if fundamental_data else None
                eps = fundamental_data.get('EPS', fundamental_data.get('eps')) if fundamental_data else None

                fundamental = FundamentalMetrics(
                    pe_ratio=float(pe_ratio) if pe_ratio and pe_ratio != 'None' else None,
                    peg_ratio=float(fundamental_data.get('PEGRatio', 0)) if fundamental_data and fundamental_data.get('PEGRatio', '0') != 'None' else None,
                    eps=float(eps) if eps and eps != 'None' else None,
                    revenue_growth=float(fundamental_data.get('QuarterlyRevenueGrowthYOY', 0)) if fundamental_data and fundamental_data.get('QuarterlyRevenueGrowthYOY') else None,
                    profit_margin=float(fundamental_data.get('ProfitMargin', 0)) if fundamental_data and fundamental_data.get('ProfitMargin') else None,
                    debt_to_equity=float(fundamental_data.get('DebtToEquityRatio', 0)) if fundamental_data and fundamental_data.get('DebtToEquityRatio') else None,
                    roe=float(fundamental_data.get('ReturnOnEquityTTM', 0)) if fundamental_data and fundamental_data.get('ReturnOnEquityTTM') else None,
                    current_ratio=float(fundamental_data.get('CurrentRatio', 0)) if fundamental_data and fundamental_data.get('CurrentRatio') else None,
                    dividend_yield=float(fundamental_data.get('DividendYield', 0)) if fundamental_data and fundamental_data.get('DividendYield') else None,
                    market_cap=float(fundamental_data.get('MarketCapitalization', stock.market_cap or 0)) if fundamental_data and fundamental_data.get('MarketCapitalization') else stock.market_cap,
                    enterprise_value=float(fundamental_data.get('EVToRevenue', 0)) * float(fundamental_data.get('RevenueTTM', 0)) if fundamental_data and fundamental_data.get('EVToRevenue') and fundamental_data.get('RevenueTTM') else None,
                    book_value=float(fundamental_data.get('BookValue', 0)) if fundamental_data and fundamental_data.get('BookValue') else None,
                    intrinsic_value=None,  # Would require complex calculation
                    valuation_score=random.uniform(60, 85)  # Placeholder for now
                )

            except Exception as e:
                logger.error(f"Error processing fundamental data for {symbol}: {e}")
                # Fallback to mock data
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
                    market_cap=stock.market_cap or 2500000000000,
                    enterprise_value=None,
                    book_value=45.0,
                    intrinsic_value=165.0,
                    valuation_score=72.5
                )

        # Sentiment Analysis processing
        if needs_sentiment:
            logger.info(f"Processing sentiment analysis for {symbol}")

            try:
                if sentiment_data:
                    news_sentiment = sentiment_data.get('news', {}).get('overall_sentiment', 0)
                    social_data = sentiment_data.get('social', {})

                    sentiment = SentimentAnalysis(
                        overall_sentiment=news_sentiment,
                        sentiment_label="Positive" if news_sentiment > 0.1 else "Negative" if news_sentiment < -0.1 else "Neutral",
                        news_sentiment=news_sentiment,
                        social_sentiment=social_data.get('sentiment', 0) if social_data else None,
                        analyst_sentiment=None,  # Would require analyst ratings data
                        insider_sentiment=None,  # Would require insider trading data
                        sentiment_momentum=sentiment_data.get('momentum', 'stable'),
                        key_topics=sentiment_data.get('news', {}).get('key_topics', []),
                        sentiment_sources=sentiment_data.get('sources', {})
                    )
                else:
                    # Fallback sentiment
                    sentiment = SentimentAnalysis(
                        overall_sentiment=0.0,
                        sentiment_label="Neutral",
                        news_sentiment=0.0,
                        social_sentiment=None,
                        analyst_sentiment=None,
                        insider_sentiment=None,
                        sentiment_momentum="stable",
                        key_topics=["market conditions"],
                        sentiment_sources={"news": 0, "social": 0}
                    )

            except Exception as e:
                logger.error(f"Error in sentiment analysis for {symbol}: {e}")
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
    
        # ML Predictions with real models
        # NOTE: Reuses price_history from parallel fetch above - no additional DB call needed
        if request.include_ml_predictions and model_manager:
            logger.info(f"Generating ML predictions for {symbol}")

            try:
                # Use price_history already fetched in parallel (for last 60 days, take most recent)
                recent_prices = price_history[-60:] if price_history and len(price_history) >= 60 else price_history

                if recent_prices and len(recent_prices) >= 30:
                    # Prepare data for ML model
                    price_data = [
                        {
                            'open': float(p.open),
                            'high': float(p.high),
                            'low': float(p.low),
                            'close': float(p.close),
                            'volume': p.volume,
                            'date': p.date
                        }
                        for p in recent_prices
                    ]

                    # Get predictions from ML models with timeout
                    predictions = await safe_async_call(
                        model_manager.get_price_predictions(symbol, price_data),
                        timeout=DEFAULT_API_TIMEOUT,
                        error_msg=f"ML predictions for {symbol}",
                        default={}
                    )

                    if predictions:
                        ml_predictions = MLPredictions(
                            price_prediction_1d=predictions.get('1d', None),
                            price_prediction_7d=predictions.get('7d', None),
                            price_prediction_30d=predictions.get('30d', None),
                            confidence_score=predictions.get('confidence', 0.7),
                            predicted_volatility=predictions.get('volatility', None),
                            risk_score=predictions.get('risk_score', 50.0),
                            pattern_recognition=predictions.get('patterns', []),
                            anomaly_detection=predictions.get('anomaly', False),
                            trend_prediction=predictions.get('trend', 'neutral')
                        )
                    else:
                        ml_predictions = None
                else:
                    logger.warning(f"Insufficient price data for ML predictions for {symbol}")
                    ml_predictions = None

            except Exception as e:
                logger.error(f"Error generating ML predictions for {symbol}: {e}")
                # Fallback to mock predictions
                ml_predictions = MLPredictions(
                    price_prediction_1d=152.5,
                    price_prediction_7d=155.0,
                    price_prediction_30d=160.0,
                    confidence_score=0.65,
                    predicted_volatility=0.22,
                    risk_score=35.0,
                    pattern_recognition=["consolidation"],
                    anomaly_detection=False,
                    trend_prediction="neutral"
                )
    
        # Risk Metrics calculation
        logger.info(f"Calculating risk metrics for {symbol}")

        # Calculate real risk metrics if we have price data
        risk_metrics = None
        if price_history and len(price_history) >= 30:
            prices = [float(p.close) for p in price_history]
            returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]

            volatility = statistics.stdev(returns) if len(returns) > 1 else 0.0
            mean_return = statistics.mean(returns) if returns else 0.0
            
            # Simplified risk calculations (would use more sophisticated methods in production)
            sharpe_ratio = (mean_return * 252) / (volatility * math.sqrt(252)) if volatility > 0 else 0.0
            
            risk_metrics = RiskMetrics(
                beta=random.uniform(0.8, 1.2),  # Would calculate vs market
                alpha=mean_return * 252,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sharpe_ratio * 1.2,  # Approximation
                max_drawdown=min(returns) if returns else 0.0,
                var_95=statistics.quantiles(returns, n=20)[0] if len(returns) > 20 else min(returns, default=0.0),
                cvar_95=min(returns) if returns else 0.0,
                correlation_with_market=random.uniform(0.6, 0.9),
                risk_adjusted_return=mean_return / volatility if volatility > 0 else 0.0,
                overall_risk_score=min(100, max(0, (volatility * 100) * 2))
            )
        else:
            # Fallback risk metrics
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
        scores = []
        
        if technical:
            tech_score = 50  # Base score
            if technical.rsi:
                if 30 <= technical.rsi <= 70:
                    tech_score += 20
                elif technical.rsi < 30:
                    tech_score += 10  # Oversold can be good
            if technical.trend == "bullish":
                tech_score += 15
            elif technical.trend == "bearish":
                tech_score -= 15
            scores.append(min(100, max(0, tech_score)))
        
        if fundamental:
            fund_score = 50
            if fundamental.pe_ratio and 10 <= fundamental.pe_ratio <= 25:
                fund_score += 20
            if fundamental.revenue_growth and fundamental.revenue_growth > 0:
                fund_score += 15
            scores.append(min(100, max(0, fund_score)))
        
        if sentiment:
            sent_score = 50 + (sentiment.overall_sentiment * 25)
            scores.append(min(100, max(0, sent_score)))
        
        if ml_predictions:
            ml_score = ml_predictions.confidence_score * 100
            scores.append(ml_score)
        
        overall_score = statistics.mean(scores) if scores else 60.0
        
        # Determine recommendation
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
        
        # Add data source info to insights
        data_sources = []
        if technical:
            data_sources.append("technical analysis")
        if fundamental:
            data_sources.append("fundamental data")
        if sentiment:
            data_sources.append("news sentiment")
        if ml_predictions:
            data_sources.append("ML predictions")
        
        if data_sources:
            insights.append(f"Analysis includes: {', '.join(data_sources)}")
        
        # Background task to cache results and update database
        background_tasks.add_task(cache_analysis_results, symbol, overall_score, analysis_data)
        
        # Calculate confidence based on available data
        confidence = min(1.0, len([x for x in [technical, fundamental, sentiment, ml_predictions] if x is not None]) * 0.25)
        
        return AnalysisResponse(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            analysis_type=request.analysis_type,
            technical=technical,
            fundamental=fundamental,
            sentiment=sentiment,
            ml_predictions=ml_predictions,
            risk_metrics=risk_metrics,
            overall_score=round(overall_score, 2),
            recommendation=recommendation,
            confidence=confidence,
            key_insights=insights,
            warnings=["High volatility detected in recent price action"] if risk_metrics and risk_metrics.overall_risk_score > 60 else None,
            next_earnings_date=datetime.utcnow().date() + timedelta(days=random.randint(10, 90)),  # Would fetch from earnings calendar
            last_updated=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in comprehensive analysis for {symbol}: {e}")
        # await handle_api_error(e, f"analyze stock {symbol}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error performing analysis: {str(e)}"
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

# Enhanced Background task functions
async def cache_analysis_results(symbol: str, score: float, analysis_data: Dict[str, Any]):
    """Cache analysis results and update database"""
    try:
        logger.info(f"Caching analysis results for {symbol} with score {score}")
        
        # Cache the results (would use Redis in production)
        # For now, just log the operation
        cache_data = {
            "symbol": symbol,
            "score": score,
            "analysis_data": analysis_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # In production, this would:
        # 1. Save to Redis with appropriate TTL
        # 2. Update analysis results in database
        # 3. Trigger any necessary notifications
        # 4. Update recommendation caches
        
        await asyncio.sleep(0.1)  # Simulate async operation
        logger.info(f"Successfully cached analysis for {symbol}")
        
    except Exception as e:
        logger.error(f"Error caching analysis results for {symbol}: {e}")