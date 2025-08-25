"""
Simplified Data Pipeline using Celery
Replaces complex Airflow DAGs with lightweight Celery tasks
"""

from celery import Celery, chain, group
from celery.schedules import crontab
from datetime import datetime, timedelta
import logging
import os
from typing import List, Dict, Any
import yfinance as yf
import requests
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Celery configuration
app = Celery('data_pipeline')
app.config_from_object('backend.config.celery_config')

# Database connection
DATABASE_URL = os.getenv('DATABASE_URL', 
    f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@"
    f"{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
)
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)

# API Configuration
FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY')
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
NEWS_API_KEY = os.getenv('NEWS_API_KEY')


@app.task(bind=True, max_retries=3)
def fetch_stock_prices(self, ticker: str) -> Dict:
    """Fetch latest stock prices using yfinance (no rate limits)"""
    try:
        logger.info(f"Fetching prices for {ticker}")
        stock = yf.Ticker(ticker)
        
        # Get historical data (last 30 days)
        hist = stock.history(period="1mo")
        
        if hist.empty:
            raise ValueError(f"No data available for {ticker}")
        
        # Get latest price info
        latest = hist.iloc[-1]
        
        return {
            'ticker': ticker,
            'date': hist.index[-1].strftime('%Y-%m-%d'),
            'open': float(latest['Open']),
            'high': float(latest['High']),
            'low': float(latest['Low']),
            'close': float(latest['Close']),
            'volume': int(latest['Volume']),
            'success': True
        }
    except Exception as e:
        logger.error(f"Error fetching {ticker}: {e}")
        raise self.retry(exc=e, countdown=60)


@app.task(bind=True, max_retries=3)
def calculate_technical_indicators(self, price_data: Dict) -> Dict:
    """Calculate technical indicators from price data"""
    try:
        ticker = price_data['ticker']
        logger.info(f"Calculating indicators for {ticker}")
        
        # Fetch more historical data for calculations
        stock = yf.Ticker(ticker)
        hist = stock.history(period="3mo")
        
        if len(hist) < 20:
            raise ValueError(f"Insufficient data for {ticker}")
        
        # Calculate indicators
        close_prices = hist['Close'].values
        
        # Simple Moving Averages
        sma_20 = np.mean(close_prices[-20:]) if len(close_prices) >= 20 else close_prices[-1]
        sma_50 = np.mean(close_prices[-50:]) if len(close_prices) >= 50 else close_prices[-1]
        
        # RSI (14-day)
        def calculate_rsi(prices, period=14):
            deltas = np.diff(prices)
            seed = deltas[:period+1]
            up = seed[seed >= 0].sum() / period
            down = -seed[seed < 0].sum() / period
            rs = up / down if down != 0 else 100
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        rsi = calculate_rsi(close_prices) if len(close_prices) > 14 else 50
        
        # MACD
        if len(close_prices) >= 26:
            exp1 = pd.Series(close_prices).ewm(span=12, adjust=False).mean()
            exp2 = pd.Series(close_prices).ewm(span=26, adjust=False).mean()
            macd = exp1.iloc[-1] - exp2.iloc[-1]
            signal = exp1.ewm(span=9, adjust=False).mean().iloc[-1]
        else:
            macd = 0
            signal = 0
        
        # Bollinger Bands
        std_20 = np.std(close_prices[-20:]) if len(close_prices) >= 20 else 0
        bollinger_upper = sma_20 + (2 * std_20)
        bollinger_lower = sma_20 - (2 * std_20)
        
        return {
            'ticker': ticker,
            'date': datetime.now(),
            'sma_20': round(sma_20, 2),
            'sma_50': round(sma_50, 2),
            'rsi_14': round(rsi, 2),
            'macd': round(macd, 4),
            'macd_signal': round(signal, 4),
            'bollinger_upper': round(bollinger_upper, 2),
            'bollinger_middle': round(sma_20, 2),
            'bollinger_lower': round(bollinger_lower, 2),
            'success': True
        }
    except Exception as e:
        logger.error(f"Error calculating indicators for {ticker}: {e}")
        return {'ticker': ticker, 'success': False, 'error': str(e)}


@app.task(bind=True, max_retries=3)
def fetch_news_sentiment(self, ticker: str) -> Dict:
    """Fetch news and calculate sentiment (using free tier APIs carefully)"""
    try:
        logger.info(f"Fetching news for {ticker}")
        
        # Use Finnhub for news (60 calls/minute free tier)
        if FINNHUB_API_KEY:
            url = f"https://finnhub.io/api/v1/company-news"
            params = {
                'symbol': ticker,
                'from': (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
                'to': datetime.now().strftime('%Y-%m-%d'),
                'token': FINNHUB_API_KEY
            }
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                news = response.json()[:5]  # Limit to 5 articles
                
                # Simple sentiment calculation (replace with FinBERT later)
                sentiments = []
                for article in news:
                    # Basic keyword-based sentiment
                    positive_words = ['gain', 'rise', 'up', 'high', 'buy', 'growth', 'profit']
                    negative_words = ['loss', 'fall', 'down', 'low', 'sell', 'decline', 'loss']
                    
                    headline = article.get('headline', '').lower()
                    summary = article.get('summary', '').lower()
                    text = headline + ' ' + summary
                    
                    pos_count = sum(1 for word in positive_words if word in text)
                    neg_count = sum(1 for word in negative_words if word in text)
                    
                    if pos_count > neg_count:
                        sentiment = 0.5 + (0.5 * min(pos_count / 10, 0.5))
                    elif neg_count > pos_count:
                        sentiment = -0.5 - (0.5 * min(neg_count / 10, 0.5))
                    else:
                        sentiment = 0
                    
                    sentiments.append(sentiment)
                
                avg_sentiment = np.mean(sentiments) if sentiments else 0
                
                return {
                    'ticker': ticker,
                    'sentiment_score': round(avg_sentiment, 3),
                    'news_count': len(news),
                    'success': True
                }
        
        # Fallback: return neutral sentiment
        return {
            'ticker': ticker,
            'sentiment_score': 0,
            'news_count': 0,
            'success': True
        }
        
    except Exception as e:
        logger.error(f"Error fetching news for {ticker}: {e}")
        return {'ticker': ticker, 'success': False, 'error': str(e)}


@app.task(bind=True)
def save_to_database(self, data: Dict, data_type: str) -> bool:
    """Save data to database"""
    try:
        session = Session()
        
        if data_type == 'price':
            # Save price data
            query = """
                INSERT INTO price_history (stock_id, date, open, high, low, close, volume)
                SELECT id, %s, %s, %s, %s, %s, %s
                FROM stocks WHERE ticker = %s
                ON CONFLICT (stock_id, date) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume
            """
            session.execute(query, (
                data['date'], data['open'], data['high'], 
                data['low'], data['close'], data['volume'], 
                data['ticker']
            ))
            
        elif data_type == 'indicators':
            # Save technical indicators
            query = """
                INSERT INTO technical_indicators 
                (stock_id, date, sma_20, sma_50, rsi_14, macd, macd_signal,
                 bollinger_upper, bollinger_middle, bollinger_lower)
                SELECT id, %s, %s, %s, %s, %s, %s, %s, %s, %s
                FROM stocks WHERE ticker = %s
                ON CONFLICT (stock_id, date) DO UPDATE SET
                    sma_20 = EXCLUDED.sma_20,
                    sma_50 = EXCLUDED.sma_50,
                    rsi_14 = EXCLUDED.rsi_14
            """
            session.execute(query, (
                data['date'], data['sma_20'], data['sma_50'],
                data['rsi_14'], data['macd'], data['macd_signal'],
                data['bollinger_upper'], data['bollinger_middle'],
                data['bollinger_lower'], data['ticker']
            ))
            
        elif data_type == 'sentiment':
            # Update sentiment in stocks table or separate sentiment table
            query = """
                UPDATE stocks 
                SET last_updated = CURRENT_TIMESTAMP
                WHERE ticker = %s
            """
            session.execute(query, (data['ticker'],))
        
        session.commit()
        logger.info(f"Saved {data_type} data for {data.get('ticker')}")
        return True
        
    except Exception as e:
        logger.error(f"Database save error: {e}")
        session.rollback()
        return False
    finally:
        session.close()


@app.task
def generate_recommendations(tickers: List[str]) -> List[Dict]:
    """Generate simple rule-based recommendations"""
    try:
        session = Session()
        recommendations = []
        
        for ticker in tickers[:10]:  # Limit to top 10
            # Fetch latest indicators
            query = """
                SELECT ti.rsi_14, ti.sma_20, ti.sma_50, ph.close
                FROM technical_indicators ti
                JOIN stocks s ON ti.stock_id = s.id
                JOIN price_history ph ON ph.stock_id = s.id
                WHERE s.ticker = %s
                ORDER BY ti.date DESC, ph.date DESC
                LIMIT 1
            """
            result = session.execute(query, (ticker,)).fetchone()
            
            if result:
                rsi, sma_20, sma_50, close = result
                
                # Simple rule-based recommendation
                score = 0
                reasons = []
                
                # RSI signals
                if rsi and rsi < 30:
                    score += 2
                    reasons.append("Oversold (RSI < 30)")
                elif rsi and rsi > 70:
                    score -= 2
                    reasons.append("Overbought (RSI > 70)")
                
                # Moving average signals
                if sma_20 and sma_50:
                    if sma_20 > sma_50:
                        score += 1
                        reasons.append("Bullish MA crossover")
                    else:
                        score -= 1
                        reasons.append("Bearish MA pattern")
                
                # Price vs MA
                if close and sma_20:
                    if close > sma_20:
                        score += 1
                        reasons.append("Price above MA20")
                    else:
                        score -= 1
                        reasons.append("Price below MA20")
                
                # Generate recommendation
                if score >= 2:
                    action = 'buy'
                    confidence = min(0.5 + (score * 0.1), 0.9)
                elif score <= -2:
                    action = 'sell'
                    confidence = min(0.5 + (abs(score) * 0.1), 0.9)
                else:
                    action = 'hold'
                    confidence = 0.5
                
                recommendations.append({
                    'ticker': ticker,
                    'action': action,
                    'confidence': round(confidence, 2),
                    'score': score,
                    'reasons': reasons,
                    'generated_at': datetime.now()
                })
        
        # Save top recommendations to database
        for rec in recommendations[:5]:
            if rec['action'] != 'sell':  # Only save buy/hold recommendations
                query = """
                    INSERT INTO recommendations
                    (stock_id, action, confidence, reasoning, 
                     technical_score, is_active, created_at, priority)
                    SELECT id, %s, %s, %s, %s, true, %s, %s
                    FROM stocks WHERE ticker = %s
                """
                session.execute(query, (
                    rec['action'], rec['confidence'],
                    '; '.join(rec['reasons']),
                    rec['score'] / 10.0,  # Normalize score
                    rec['generated_at'],
                    int(rec['confidence'] * 10),
                    rec['ticker']
                ))
        
        session.commit()
        logger.info(f"Generated {len(recommendations)} recommendations")
        return recommendations
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        return []
    finally:
        session.close()


@app.task
def run_daily_pipeline():
    """Main pipeline orchestrator - runs all tasks in sequence"""
    try:
        logger.info("Starting daily data pipeline")
        
        # Get active stock tickers
        session = Session()
        tickers = session.execute(
            "SELECT ticker FROM stocks WHERE is_active = true LIMIT 100"
        ).fetchall()
        tickers = [t[0] for t in tickers]
        session.close()
        
        if not tickers:
            logger.warning("No active stocks found")
            return
        
        logger.info(f"Processing {len(tickers)} stocks")
        
        # Create task chains for each ticker
        for ticker in tickers:
            # Chain tasks: fetch prices -> calculate indicators -> fetch sentiment -> save
            chain(
                fetch_stock_prices.s(ticker),
                calculate_technical_indicators.s(),
                save_to_database.s('indicators')
            ).apply_async()
            
            # Separate chain for sentiment (to avoid rate limits)
            if tickers.index(ticker) < 20:  # Limit sentiment to first 20 stocks
                fetch_news_sentiment.apply_async(args=[ticker], countdown=tickers.index(ticker) * 2)
        
        # Generate recommendations after a delay
        generate_recommendations.apply_async(args=[tickers[:20]], countdown=300)
        
        logger.info("Daily pipeline scheduled successfully")
        return {'status': 'success', 'stocks_processed': len(tickers)}
        
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        return {'status': 'error', 'error': str(e)}


# Celery Beat Schedule
app.conf.beat_schedule = {
    'daily-data-pipeline': {
        'task': 'backend.tasks.data_pipeline.run_daily_pipeline',
        'schedule': crontab(hour=6, minute=0),  # Run at 6 AM daily
    },
    'hourly-price-update': {
        'task': 'backend.tasks.data_pipeline.fetch_stock_prices',
        'schedule': crontab(minute=0),  # Run every hour
        'args': ['AAPL']  # Monitor Apple as a canary
    },
}


if __name__ == "__main__":
    # Test the pipeline
    result = run_daily_pipeline()
    print(f"Pipeline result: {result}")