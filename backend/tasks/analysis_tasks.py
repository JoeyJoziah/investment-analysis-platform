"""
Celery tasks for analysis and ML predictions
"""
from celery import shared_task, group
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, date
import numpy as np
import pandas as pd
import logging
import json
from decimal import Decimal

from backend.tasks.celery_app import celery_app
from backend.analytics.technical_analysis import TechnicalAnalyzer
from backend.analytics.fundamental_analysis import FundamentalAnalyzer
from backend.analytics.sentiment_analysis import SentimentAnalyzer
from backend.analytics.recommendation_engine import RecommendationEngine
from backend.utils.database import get_db_sync
from backend.utils.cache import get_redis_client, cache_key
from backend.models.tables import (
    Stock, PriceHistory, Recommendation, RecommendationPerformance,
    Fundamental, News
)
from sqlalchemy import select, and_, func
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

@celery_app.task(bind=True, max_retries=3)
def analyze_stock(self, symbol: str, analysis_types: List[str] = None) -> Dict[str, Any]:
    """
    Perform comprehensive analysis on a stock
    
    Args:
        symbol: Stock symbol
        analysis_types: List of analysis types to perform
    """
    try:
        if not analysis_types:
            analysis_types = ['technical', 'fundamental', 'sentiment']
        
        result = {
            'symbol': symbol,
            'timestamp': datetime.utcnow().isoformat(),
            'analysis': {}
        }
        
        with get_db_sync() as db:
            # Get stock data
            stock = db.query(Stock).filter(Stock.symbol == symbol).first()
            if not stock:
                return {'error': f'Stock {symbol} not found'}
            
            # Get price history for analysis
            price_history = db.query(PriceHistory).filter(
                PriceHistory.stock_id == stock.id
            ).order_by(PriceHistory.date.desc()).limit(252).all()  # 1 year of data
            
            if not price_history:
                return {'error': f'No price history for {symbol}'}
            
            # Convert to DataFrame for analysis
            df = pd.DataFrame([{
                'date': ph.date,
                'open': float(ph.open),
                'high': float(ph.high),
                'low': float(ph.low),
                'close': float(ph.close),
                'volume': ph.volume
            } for ph in reversed(price_history)])
            
            # Technical Analysis
            if 'technical' in analysis_types:
                try:
                    tech_analyzer = TechnicalAnalyzer()
                    technical_result = tech_analyzer.analyze(df)
                    result['analysis']['technical'] = technical_result
                except Exception as e:
                    logger.error(f"Technical analysis error for {symbol}: {e}")
                    result['analysis']['technical'] = {'error': str(e)}
            
            # Fundamental Analysis
            if 'fundamental' in analysis_types:
                try:
                    fundamentals = db.query(Fundamental).filter(
                        Fundamental.stock_id == stock.id
                    ).order_by(Fundamental.report_date.desc()).first()
                    
                    if fundamentals:
                        fund_analyzer = FundamentalAnalyzer()
                        fundamental_result = fund_analyzer.analyze(fundamentals)
                        result['analysis']['fundamental'] = fundamental_result
                    else:
                        result['analysis']['fundamental'] = {'status': 'no_data'}
                except Exception as e:
                    logger.error(f"Fundamental analysis error for {symbol}: {e}")
                    result['analysis']['fundamental'] = {'error': str(e)}
            
            # Sentiment Analysis
            if 'sentiment' in analysis_types:
                try:
                    # Get recent news
                    recent_news = db.query(News).filter(
                        News.stock_id == stock.id,
                        News.published_at >= datetime.utcnow() - timedelta(days=7)
                    ).all()
                    
                    if recent_news:
                        sent_analyzer = SentimentAnalyzer()
                        news_texts = [f"{n.headline} {n.summary or ''}" for n in recent_news]
                        sentiment_result = sent_analyzer.analyze_batch(news_texts)
                        result['analysis']['sentiment'] = sentiment_result
                    else:
                        result['analysis']['sentiment'] = {'status': 'no_recent_news'}
                except Exception as e:
                    logger.error(f"Sentiment analysis error for {symbol}: {e}")
                    result['analysis']['sentiment'] = {'error': str(e)}
        
        # Generate overall score and recommendation
        overall_score = calculate_overall_score(result['analysis'])
        result['overall_score'] = overall_score
        result['recommendation'] = generate_recommendation(overall_score)
        
        # Cache results
        redis_client = get_redis_client()
        cache_key = f"analysis:{symbol}"
        redis_client.setex(cache_key, 3600, json.dumps(result))  # Cache for 1 hour
        
        # Store recommendation if score is significant
        if overall_score > 70 or overall_score < 30:
            create_recommendation.delay(symbol, result)
        
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {e}")
        raise self.retry(exc=e, countdown=60)

@celery_app.task
def run_daily_analysis() -> Dict[str, Any]:
    """Run comprehensive daily analysis for all active stocks"""
    try:
        with get_db_sync() as db:
            # Get top stocks by market cap
            top_stocks = db.query(Stock).filter(
                Stock.is_active == True,
                Stock.is_tradable == True,
                Stock.market_cap > 1000000000  # $1B+ market cap
            ).order_by(Stock.market_cap.desc()).limit(100).all()
            
            # Create analysis tasks
            analysis_group = group(
                analyze_stock.si(stock.symbol, ['technical', 'fundamental', 'sentiment'])
                for stock in top_stocks
            )
            
            # Execute analysis
            results = analysis_group.apply_async()
            
            # Wait for completion (with timeout)
            analysis_results = results.get(timeout=1800)  # 30 minutes timeout
            
            # Process results
            strong_buys = []
            strong_sells = []
            
            for result in analysis_results:
                if result.get('overall_score', 0) > 80:
                    strong_buys.append(result['symbol'])
                elif result.get('overall_score', 0) < 20:
                    strong_sells.append(result['symbol'])
            
            summary = {
                'date': date.today().isoformat(),
                'stocks_analyzed': len(analysis_results),
                'strong_buys': strong_buys,
                'strong_sells': strong_sells,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Store daily summary
            redis_client = get_redis_client()
            redis_client.setex('daily_analysis_summary', 86400, json.dumps(summary))
            
            logger.info(f"Daily analysis completed: {len(analysis_results)} stocks analyzed")
            return summary
            
    except Exception as e:
        logger.error(f"Error in daily analysis: {e}")
        return {'error': str(e)}

@celery_app.task
def calculate_all_indicators() -> Dict[str, Any]:
    """Calculate technical indicators for all stocks"""
    try:
        with get_db_sync() as db:
            stocks = db.query(Stock).filter(
                Stock.is_active == True
            ).all()
            
            results = {
                'processed': 0,
                'errors': []
            }
            
            for stock in stocks:
                try:
                    # Get price history
                    price_history = db.query(PriceHistory).filter(
                        PriceHistory.stock_id == stock.id
                    ).order_by(PriceHistory.date.desc()).limit(100).all()
                    
                    if len(price_history) < 20:
                        continue
                    
                    # Calculate indicators
                    df = pd.DataFrame([{
                        'close': float(ph.close),
                        'high': float(ph.high),
                        'low': float(ph.low),
                        'volume': ph.volume
                    } for ph in reversed(price_history)])
                    
                    indicators = {
                        'rsi': calculate_rsi(df['close']),
                        'macd': calculate_macd(df['close']),
                        'bollinger_bands': calculate_bollinger_bands(df['close']),
                        'moving_averages': {
                            'sma_20': df['close'].rolling(20).mean().iloc[-1],
                            'sma_50': df['close'].rolling(50).mean().iloc[-1] if len(df) >= 50 else None,
                            'ema_12': df['close'].ewm(span=12).mean().iloc[-1],
                            'ema_26': df['close'].ewm(span=26).mean().iloc[-1]
                        }
                    }
                    
                    # Cache indicators
                    redis_client = get_redis_client()
                    cache_key = f"indicators:{stock.symbol}"
                    redis_client.setex(cache_key, 3600, json.dumps(indicators))
                    
                    results['processed'] += 1
                    
                except Exception as e:
                    results['errors'].append({
                        'symbol': stock.symbol,
                        'error': str(e)
                    })
            
            logger.info(f"Calculated indicators for {results['processed']} stocks")
            return results
            
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        return {'error': str(e)}

@celery_app.task
def update_all_recommendations() -> Dict[str, Any]:
    """Update all active recommendations"""
    try:
        engine = RecommendationEngine()
        results = engine.update_all_recommendations()
        
        logger.info(f"Updated {results['updated']} recommendations")
        return results
        
    except Exception as e:
        logger.error(f"Error updating recommendations: {e}")
        return {'error': str(e)}

@celery_app.task
def create_recommendation(symbol: str, analysis_result: Dict[str, Any]) -> bool:
    """Create a new recommendation based on analysis"""
    try:
        with get_db_sync() as db:
            stock = db.query(Stock).filter(Stock.symbol == symbol).first()
            if not stock:
                return False
            
            # Determine recommendation type
            score = analysis_result.get('overall_score', 50)
            if score >= 80:
                rec_type = 'strong_buy'
            elif score >= 65:
                rec_type = 'buy'
            elif score >= 35:
                rec_type = 'hold'
            elif score >= 20:
                rec_type = 'sell'
            else:
                rec_type = 'strong_sell'
            
            # Get current price
            latest_price = db.query(PriceHistory).filter(
                PriceHistory.stock_id == stock.id
            ).order_by(PriceHistory.date.desc()).first()
            
            if not latest_price:
                return False
            
            current_price = float(latest_price.close)
            
            # Calculate target price (simplified)
            if rec_type in ['strong_buy', 'buy']:
                target_price = current_price * 1.15  # 15% upside
                stop_loss = current_price * 0.95  # 5% stop loss
            else:
                target_price = current_price * 0.85  # 15% downside
                stop_loss = current_price * 1.05  # 5% stop loss
            
            # Create recommendation
            recommendation = Recommendation(
                stock_id=stock.id,
                recommendation_type=rec_type,
                confidence_score=abs(score - 50) / 50,  # Convert to 0-1 scale
                current_price=current_price,
                target_price=target_price,
                stop_loss=stop_loss,
                time_horizon_days=30,
                reasoning=generate_reasoning(analysis_result),
                key_factors=extract_key_factors(analysis_result),
                risk_level=calculate_risk_level(analysis_result),
                technical_score=analysis_result.get('analysis', {}).get('technical', {}).get('score'),
                fundamental_score=analysis_result.get('analysis', {}).get('fundamental', {}).get('score'),
                sentiment_score=analysis_result.get('analysis', {}).get('sentiment', {}).get('score'),
                is_active=True,
                valid_until=datetime.utcnow() + timedelta(days=30)
            )
            
            db.add(recommendation)
            db.commit()
            
            # Create performance tracking record
            performance = RecommendationPerformance(
                recommendation_id=recommendation.id,
                entry_price=current_price,
                current_price=current_price,
                highest_price=current_price,
                lowest_price=current_price,
                actual_return=0,
                max_return=0,
                max_drawdown=0,
                days_active=0,
                target_hit=False,
                stop_loss_hit=False
            )
            
            db.add(performance)
            db.commit()
            
            logger.info(f"Created {rec_type} recommendation for {symbol}")
            return True
            
    except Exception as e:
        logger.error(f"Error creating recommendation for {symbol}: {e}")
        return False

@celery_app.task
def track_recommendation_performance() -> Dict[str, Any]:
    """Track and update performance of all active recommendations"""
    try:
        with get_db_sync() as db:
            # Get active recommendations
            active_recs = db.query(Recommendation).filter(
                Recommendation.is_active == True
            ).all()
            
            updated = 0
            closed = 0
            
            for rec in active_recs:
                try:
                    # Get current price
                    latest_price = db.query(PriceHistory).filter(
                        PriceHistory.stock_id == rec.stock_id
                    ).order_by(PriceHistory.date.desc()).first()
                    
                    if not latest_price:
                        continue
                    
                    current_price = float(latest_price.close)
                    
                    # Update performance
                    perf = rec.performance
                    if perf:
                        perf.current_price = current_price
                        perf.highest_price = max(perf.highest_price, current_price)
                        perf.lowest_price = min(perf.lowest_price, current_price)
                        perf.actual_return = (current_price - perf.entry_price) / perf.entry_price
                        perf.max_return = (perf.highest_price - perf.entry_price) / perf.entry_price
                        perf.max_drawdown = (perf.lowest_price - perf.entry_price) / perf.entry_price
                        perf.days_active = (datetime.utcnow() - rec.created_at).days
                        
                        # Check if targets hit
                        if rec.recommendation_type in ['buy', 'strong_buy']:
                            if current_price >= rec.target_price:
                                perf.target_hit = True
                                rec.is_active = False
                                closed += 1
                            elif current_price <= rec.stop_loss:
                                perf.stop_loss_hit = True
                                rec.is_active = False
                                closed += 1
                        else:  # sell recommendations
                            if current_price <= rec.target_price:
                                perf.target_hit = True
                                rec.is_active = False
                                closed += 1
                            elif current_price >= rec.stop_loss:
                                perf.stop_loss_hit = True
                                rec.is_active = False
                                closed += 1
                        
                        # Check if expired
                        if datetime.utcnow() > rec.valid_until:
                            rec.is_active = False
                            closed += 1
                        
                        perf.last_updated = datetime.utcnow()
                        updated += 1
                    
                except Exception as e:
                    logger.error(f"Error updating recommendation {rec.id}: {e}")
            
            db.commit()
            
            return {
                'updated': updated,
                'closed': closed,
                'active': len(active_recs) - closed
            }
            
    except Exception as e:
        logger.error(f"Error tracking recommendation performance: {e}")
        return {'error': str(e)}

# Helper functions
def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return float(rsi.iloc[-1]) if not rsi.empty else 50

def calculate_macd(prices: pd.Series) -> Dict[str, float]:
    """Calculate MACD indicator"""
    ema12 = prices.ewm(span=12, adjust=False).mean()
    ema26 = prices.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return {
        'macd': float(macd_line.iloc[-1]),
        'signal': float(signal_line.iloc[-1]),
        'histogram': float(histogram.iloc[-1])
    }

def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: int = 2) -> Dict[str, float]:
    """Calculate Bollinger Bands"""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    
    return {
        'upper': float(upper_band.iloc[-1]),
        'middle': float(sma.iloc[-1]),
        'lower': float(lower_band.iloc[-1])
    }

def calculate_overall_score(analysis: Dict[str, Any]) -> float:
    """Calculate overall score from multiple analysis types"""
    scores = []
    weights = {
        'technical': 0.4,
        'fundamental': 0.3,
        'sentiment': 0.3
    }
    
    for analysis_type, weight in weights.items():
        if analysis_type in analysis and 'score' in analysis[analysis_type]:
            scores.append(analysis[analysis_type]['score'] * weight)
    
    return sum(scores) / sum(weights.values()) if scores else 50

def generate_recommendation(score: float) -> str:
    """Generate recommendation based on score"""
    if score >= 80:
        return 'strong_buy'
    elif score >= 65:
        return 'buy'
    elif score >= 35:
        return 'hold'
    elif score >= 20:
        return 'sell'
    else:
        return 'strong_sell'

def generate_reasoning(analysis_result: Dict[str, Any]) -> str:
    """Generate reasoning text from analysis"""
    reasoning_parts = []
    
    if 'technical' in analysis_result.get('analysis', {}):
        tech = analysis_result['analysis']['technical']
        if 'rsi' in tech:
            reasoning_parts.append(f"RSI at {tech['rsi']:.1f}")
    
    if 'fundamental' in analysis_result.get('analysis', {}):
        fund = analysis_result['analysis']['fundamental']
        if 'pe_ratio' in fund:
            reasoning_parts.append(f"P/E ratio of {fund['pe_ratio']:.1f}")
    
    if 'sentiment' in analysis_result.get('analysis', {}):
        sent = analysis_result['analysis']['sentiment']
        if 'overall' in sent:
            reasoning_parts.append(f"Sentiment score {sent['overall']:.2f}")
    
    return ". ".join(reasoning_parts) if reasoning_parts else "Based on comprehensive analysis"

def extract_key_factors(analysis_result: Dict[str, Any]) -> List[str]:
    """Extract key factors from analysis"""
    factors = []
    
    if analysis_result.get('overall_score', 0) > 70:
        factors.append("Strong technical indicators")
    
    if 'fundamental' in analysis_result.get('analysis', {}):
        factors.append("Solid fundamentals")
    
    if 'sentiment' in analysis_result.get('analysis', {}):
        sent_score = analysis_result['analysis']['sentiment'].get('overall', 0)
        if sent_score > 0.5:
            factors.append("Positive market sentiment")
        elif sent_score < -0.5:
            factors.append("Negative market sentiment")
    
    return factors if factors else ["Market conditions"]

def calculate_risk_level(analysis_result: Dict[str, Any]) -> str:
    """Calculate risk level from analysis"""
    # Simplified risk calculation
    volatility = analysis_result.get('analysis', {}).get('technical', {}).get('volatility', 0.2)
    
    if volatility < 0.15:
        return 'low'
    elif volatility < 0.25:
        return 'medium'
    else:
        return 'high'