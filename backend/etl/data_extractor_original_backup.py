"""
Enhanced Multi-Source Data Extractor for 6000+ Stocks
Implements robust extraction with intelligent rate limiting, caching, and fallbacks
"""

import asyncio
import aiohttp
import aiofiles
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Any, Optional, Union
import yfinance as yf
import pandas as pd
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential, wait_fixed
import os
from dotenv import load_dotenv
import json
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import sqlite3
import hashlib
from urllib.parse import urlencode
import requests
from bs4 import BeautifulSoup
import pickle
from dataclasses import dataclass
import re

load_dotenv()

logger = logging.getLogger(__name__)

@dataclass
class RateLimitConfig:
    calls_per_minute: int
    calls_per_hour: int
    calls_per_day: int
    delay_between_calls: float
    backoff_multiplier: float = 2.0
    max_backoff: float = 300.0

@dataclass
class DataSourceConfig:
    name: str
    priority: int  # Lower is higher priority
    rate_limit: RateLimitConfig
    enabled: bool = True
    requires_api_key: bool = False
    api_key_env: str = None
    base_delay: float = 1.0


class MultiSourceDataExtractor:
    """Enhanced multi-source data extractor for large-scale stock data collection"""
    
    def __init__(self, cache_dir: str = "/tmp/stock_cache"):
        # Initialize cache directory
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize progress tracking database
        self.progress_db_path = os.path.join(cache_dir, "progress.db")
        self._init_progress_db()
        
        # Configure data sources with realistic rate limits
        self.data_sources = {
            'yahoo_scraper': DataSourceConfig(
                name='yahoo_scraper',
                priority=1,
                rate_limit=RateLimitConfig(
                    calls_per_minute=10,
                    calls_per_hour=300,
                    calls_per_day=5000,
                    delay_between_calls=6.0
                ),
                base_delay=2.0
            ),
            'yfinance': DataSourceConfig(
                name='yfinance',
                priority=2,
                rate_limit=RateLimitConfig(
                    calls_per_minute=5,
                    calls_per_hour=100,
                    calls_per_day=1000,
                    delay_between_calls=12.0
                ),
                base_delay=3.0
            ),
            'alpha_vantage': DataSourceConfig(
                name='alpha_vantage',
                priority=3,
                rate_limit=RateLimitConfig(
                    calls_per_minute=1,
                    calls_per_hour=5,
                    calls_per_day=25,
                    delay_between_calls=60.0
                ),
                requires_api_key=True,
                api_key_env='ALPHA_VANTAGE_API_KEY',
                base_delay=5.0
            ),
            'finnhub': DataSourceConfig(
                name='finnhub',
                priority=4,
                rate_limit=RateLimitConfig(
                    calls_per_minute=20,
                    calls_per_hour=60,
                    calls_per_day=1000,
                    delay_between_calls=3.0
                ),
                requires_api_key=True,
                api_key_env='FINNHUB_API_KEY',
                base_delay=2.0
            ),
            'polygon': DataSourceConfig(
                name='polygon',
                priority=5,
                rate_limit=RateLimitConfig(
                    calls_per_minute=5,
                    calls_per_hour=60,
                    calls_per_day=500,
                    delay_between_calls=12.0
                ),
                requires_api_key=True,
                api_key_env='POLYGON_API_KEY',
                base_delay=5.0
            ),
            'marketwatch_scraper': DataSourceConfig(
                name='marketwatch_scraper',
                priority=6,
                rate_limit=RateLimitConfig(
                    calls_per_minute=8,
                    calls_per_hour=200,
                    calls_per_day=3000,
                    delay_between_calls=7.5
                ),
                base_delay=3.0
            )
        }
        
        # Initialize API keys
        self.api_keys = {}
        for source_name, config in self.data_sources.items():
            if config.requires_api_key and config.api_key_env:
                key = os.getenv(config.api_key_env)
                if key:
                    self.api_keys[source_name] = key
                    logger.info(f"Loaded API key for {source_name}")
                else:
                    config.enabled = False
                    logger.warning(f"No API key found for {source_name}, disabling")
        
        # Rate limiting trackers
        self.call_history = {source: [] for source in self.data_sources.keys()}
        self.backoff_delays = {source: config.base_delay for source, config in self.data_sources.items()}
        
        # Session for web scraping
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Cache settings
        self.cache_expiry_hours = 6  # Cache data for 6 hours
        
        logger.info(f"Initialized MultiSourceDataExtractor with {len([s for s in self.data_sources.values() if s.enabled])} enabled sources")
    
    def _init_progress_db(self):
        """Initialize SQLite database for progress tracking"""
        conn = sqlite3.connect(self.progress_db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS extraction_progress (
                ticker TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                last_attempt TIMESTAMP,
                last_success TIMESTAMP,
                attempts INTEGER DEFAULT 0,
                data_sources TEXT,
                error_message TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cache_index (
                cache_key TEXT PRIMARY KEY,
                ticker TEXT,
                data_type TEXT,
                source TEXT,
                created_at TIMESTAMP,
                expires_at TIMESTAMP,
                file_path TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _get_cache_key(self, ticker: str, data_type: str, source: str, params: Dict = None) -> str:
        """Generate cache key for data"""
        key_data = f"{ticker}_{data_type}_{source}"
        if params:
            key_data += "_" + hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]
        return key_data
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        conn = sqlite3.connect(self.progress_db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT expires_at FROM cache_index WHERE cache_key = ?",
            (cache_key,)
        )
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return False
        
        expires_at = datetime.fromisoformat(result[0])
        return datetime.now() < expires_at
    
    def _save_to_cache(self, cache_key: str, data: Any, ticker: str, data_type: str, source: str):
        """Save data to cache"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            
            expires_at = datetime.now() + timedelta(hours=self.cache_expiry_hours)
            
            conn = sqlite3.connect(self.progress_db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO cache_index 
                (cache_key, ticker, data_type, source, created_at, expires_at, file_path)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (cache_key, ticker, data_type, source, datetime.now().isoformat(),
                  expires_at.isoformat(), cache_file))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving cache for {cache_key}: {e}")
    
    def _load_from_cache(self, cache_key: str) -> Optional[Any]:
        """Load data from cache"""
        if not self._is_cache_valid(cache_key):
            return None
        
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        try:
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading cache for {cache_key}: {e}")
        
        return None
    
    def _can_make_request(self, source_name: str) -> bool:
        """Check if we can make a request to this source based on rate limits"""
        if source_name not in self.data_sources:
            return False
        
        config = self.data_sources[source_name]
        if not config.enabled:
            return False
        
        current_time = time.time()
        history = self.call_history[source_name]
        
        # Clean old history
        minute_ago = current_time - 60
        hour_ago = current_time - 3600
        day_ago = current_time - 86400
        
        recent_calls = [t for t in history if t > minute_ago]
        hourly_calls = [t for t in history if t > hour_ago]
        daily_calls = [t for t in history if t > day_ago]
        
        # Check limits
        if (len(recent_calls) >= config.rate_limit.calls_per_minute or
            len(hourly_calls) >= config.rate_limit.calls_per_hour or
            len(daily_calls) >= config.rate_limit.calls_per_day):
            return False
        
        return True
    
    def _record_request(self, source_name: str):
        """Record a successful request"""
        current_time = time.time()
        self.call_history[source_name].append(current_time)
        
        # Keep only recent history
        day_ago = current_time - 86400
        self.call_history[source_name] = [
            t for t in self.call_history[source_name] if t > day_ago
        ]
    
    def _get_delay_for_source(self, source_name: str) -> float:
        """Get current delay for source (with backoff)"""
        return self.backoff_delays[source_name]
    
    def _increase_backoff(self, source_name: str):
        """Increase backoff delay for source"""
        config = self.data_sources[source_name]
        current_delay = self.backoff_delays[source_name]
        new_delay = min(current_delay * config.rate_limit.backoff_multiplier, 
                       config.rate_limit.max_backoff)
        self.backoff_delays[source_name] = new_delay
        logger.warning(f"Increased backoff for {source_name} to {new_delay:.1f}s")
    
    def _reset_backoff(self, source_name: str):
        """Reset backoff delay for source"""
        config = self.data_sources[source_name]
        self.backoff_delays[source_name] = config.base_delay
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def fetch_yfinance_data(self, ticker: str, period: str = "1mo") -> Dict:
        """Fetch stock data from Yahoo Finance"""
        try:
            if not self.check_rate_limit('yfinance'):
                logger.warning(f"Rate limit reached for yfinance")
                return None
                
            logger.info(f"Fetching yfinance data for {ticker}")
            stock = yf.Ticker(ticker)
            
            # Get historical data
            hist = stock.history(period=period)
            
            if hist.empty:
                logger.warning(f"No data available for {ticker}")
                return None
            
            # Get additional info
            info = stock.info
            
            # Get latest data
            latest = hist.iloc[-1]
            
            return {
                'ticker': ticker,
                'source': 'yfinance',
                'timestamp': datetime.now(),
                'price_data': {
                    'date': hist.index[-1].strftime('%Y-%m-%d'),
                    'open': float(latest['Open']),
                    'high': float(latest['High']),
                    'low': float(latest['Low']),
                    'close': float(latest['Close']),
                    'volume': int(latest['Volume']),
                    'history': hist.to_dict('records')
                },
                'company_info': {
                    'market_cap': info.get('marketCap'),
                    'pe_ratio': info.get('trailingPE'),
                    'dividend_yield': info.get('dividendYield'),
                    'beta': info.get('beta'),
                    'sector': info.get('sector'),
                    'industry': info.get('industry')
                }
            }
        except Exception as e:
            logger.error(f"Error fetching yfinance data for {ticker}: {e}")
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def fetch_finnhub_data(self, ticker: str) -> Dict:
        """Fetch data from Finnhub API"""
        if not self.finnhub_key or not self.check_rate_limit('finnhub'):
            return None
            
        try:
            async with aiohttp.ClientSession() as session:
                # Get quote data
                quote_url = f"https://finnhub.io/api/v1/quote"
                params = {'symbol': ticker, 'token': self.finnhub_key}
                
                async with session.get(quote_url, params=params) as response:
                    if response.status == 200:
                        quote_data = await response.json()
                    else:
                        logger.error(f"Finnhub quote API error: {response.status}")
                        return None
                
                # Get company profile
                profile_url = f"https://finnhub.io/api/v1/stock/profile2"
                async with session.get(profile_url, params=params) as response:
                    if response.status == 200:
                        profile_data = await response.json()
                    else:
                        profile_data = {}
                
                # Get recommendations
                rec_url = f"https://finnhub.io/api/v1/stock/recommendation"
                async with session.get(rec_url, params=params) as response:
                    if response.status == 200:
                        rec_data = await response.json()
                    else:
                        rec_data = []
                
                return {
                    'ticker': ticker,
                    'source': 'finnhub',
                    'timestamp': datetime.now(),
                    'quote': quote_data,
                    'profile': profile_data,
                    'recommendations': rec_data[:5] if rec_data else []
                }
                
        except Exception as e:
            logger.error(f"Error fetching Finnhub data for {ticker}: {e}")
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def fetch_polygon_data(self, ticker: str) -> Dict:
        """Fetch data from Polygon.io API"""
        if not self.polygon_key or not self.check_rate_limit('polygon'):
            return None
            
        try:
            async with aiohttp.ClientSession() as session:
                # Get aggregates (bars)
                date_from = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
                date_to = datetime.now().strftime('%Y-%m-%d')
                
                agg_url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{date_from}/{date_to}"
                params = {'apiKey': self.polygon_key}
                
                async with session.get(agg_url, params=params) as response:
                    if response.status == 200:
                        agg_data = await response.json()
                    else:
                        logger.error(f"Polygon API error: {response.status}")
                        return None
                
                # Get ticker details
                details_url = f"https://api.polygon.io/v3/reference/tickers/{ticker}"
                async with session.get(details_url, params=params) as response:
                    if response.status == 200:
                        details_data = await response.json()
                    else:
                        details_data = {}
                
                return {
                    'ticker': ticker,
                    'source': 'polygon',
                    'timestamp': datetime.now(),
                    'aggregates': agg_data.get('results', []),
                    'details': details_data.get('results', {})
                }
                
        except Exception as e:
            logger.error(f"Error fetching Polygon data for {ticker}: {e}")
            raise
    
    async def fetch_news_sentiment(self, ticker: str) -> Dict:
        """Fetch news and sentiment data"""
        if not self.news_api_key or not self.check_rate_limit('news_api'):
            # Fallback to Finnhub news
            return await self.fetch_finnhub_news(ticker)
            
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://newsapi.org/v2/everything"
                params = {
                    'q': ticker,
                    'apiKey': self.news_api_key,
                    'language': 'en',
                    'sortBy': 'relevancy',
                    'pageSize': 10,
                    'from': (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
                }
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        articles = data.get('articles', [])
                        
                        # Simple sentiment analysis
                        sentiments = []
                        for article in articles:
                            sentiment = self.analyze_sentiment(
                                article.get('title', '') + ' ' + article.get('description', '')
                            )
                            sentiments.append(sentiment)
                        
                        return {
                            'ticker': ticker,
                            'source': 'newsapi',
                            'timestamp': datetime.now(),
                            'articles': articles[:5],
                            'sentiment_score': np.mean(sentiments) if sentiments else 0,
                            'article_count': len(articles)
                        }
                    else:
                        logger.error(f"NewsAPI error: {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"Error fetching news for {ticker}: {e}")
            return None
    
    async def fetch_finnhub_news(self, ticker: str) -> Dict:
        """Fetch news from Finnhub as fallback"""
        if not self.finnhub_key or not self.check_rate_limit('finnhub'):
            return None
            
        try:
            async with aiohttp.ClientSession() as session:
                url = f"https://finnhub.io/api/v1/company-news"
                params = {
                    'symbol': ticker,
                    'from': (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
                    'to': datetime.now().strftime('%Y-%m-%d'),
                    'token': self.finnhub_key
                }
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        news = await response.json()
                        
                        # Analyze sentiment
                        sentiments = []
                        for article in news[:10]:
                            sentiment = self.analyze_sentiment(
                                article.get('headline', '') + ' ' + article.get('summary', '')
                            )
                            sentiments.append(sentiment)
                        
                        return {
                            'ticker': ticker,
                            'source': 'finnhub',
                            'timestamp': datetime.now(),
                            'articles': news[:5],
                            'sentiment_score': np.mean(sentiments) if sentiments else 0,
                            'article_count': len(news)
                        }
                    else:
                        return None
                        
        except Exception as e:
            logger.error(f"Error fetching Finnhub news for {ticker}: {e}")
            return None
    
    def analyze_sentiment(self, text: str) -> float:
        """Simple rule-based sentiment analysis (to be replaced with FinBERT)"""
        positive_words = [
            'gain', 'rise', 'up', 'high', 'buy', 'growth', 'profit',
            'positive', 'strong', 'outperform', 'upgrade', 'beat'
        ]
        negative_words = [
            'loss', 'fall', 'down', 'low', 'sell', 'decline', 'loss',
            'negative', 'weak', 'underperform', 'downgrade', 'miss'
        ]
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            return min(0.5 + (0.1 * (pos_count - neg_count)), 1.0)
        elif neg_count > pos_count:
            return max(-0.5 - (0.1 * (neg_count - pos_count)), -1.0)
        else:
            return 0
    
    async def extract_all_data(self, ticker: str) -> Dict:
        """Extract data from all available sources"""
        tasks = [
            self.fetch_yfinance_data(ticker),
            self.fetch_finnhub_data(ticker),
            self.fetch_polygon_data(ticker),
            self.fetch_news_sentiment(ticker)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        combined_data = {
            'ticker': ticker,
            'extraction_time': datetime.now(),
            'sources': {}
        }
        
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Task failed: {result}")
                continue
            if result and isinstance(result, dict):
                source = result.get('source')
                if source:
                    combined_data['sources'][source] = result
        
        return combined_data
    
    async def batch_extract(self, tickers: List[str], batch_size: int = 10) -> List[Dict]:
        """Extract data for multiple tickers in batches"""
        all_results = []
        
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}: {batch}")
            
            tasks = [self.extract_all_data(ticker) for ticker in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Batch extraction failed: {result}")
                else:
                    all_results.append(result)
            
            # Delay between batches to respect rate limits
            await asyncio.sleep(2)
        
        return all_results


class DataValidator:
    """Validate extracted data for quality and completeness"""
    
    @staticmethod
    def validate_price_data(data: Dict) -> bool:
        """Validate price data structure and values"""
        required_fields = ['open', 'high', 'low', 'close', 'volume']
        
        if 'price_data' not in data:
            return False
            
        price_data = data['price_data']
        
        # Check required fields
        for field in required_fields:
            if field not in price_data:
                return False
            
            # Validate numeric values
            if field != 'volume':
                value = price_data[field]
                if not isinstance(value, (int, float)) or value <= 0:
                    return False
        
        # Validate OHLC relationships
        if not (price_data['low'] <= price_data['high']):
            return False
        if not (price_data['low'] <= price_data['open'] <= price_data['high']):
            return False
        if not (price_data['low'] <= price_data['close'] <= price_data['high']):
            return False
        
        return True
    
    @staticmethod
    def validate_company_info(data: Dict) -> bool:
        """Validate company information"""
        if 'company_info' not in data:
            return True  # Company info is optional
            
        info = data['company_info']
        
        # Validate market cap if present
        if 'market_cap' in info and info['market_cap']:
            if not isinstance(info['market_cap'], (int, float)) or info['market_cap'] < 0:
                return False
        
        # Validate PE ratio if present
        if 'pe_ratio' in info and info['pe_ratio']:
            if not isinstance(info['pe_ratio'], (int, float)):
                return False
        
        return True
    
    @staticmethod
    def clean_data(data: Dict) -> Dict:
        """Clean and standardize extracted data"""
        cleaned = data.copy()
        
        # Standardize None values
        for source, source_data in cleaned.get('sources', {}).items():
            if isinstance(source_data, dict):
                for key, value in source_data.items():
                    if value == 'N/A' or value == '':
                        source_data[key] = None
        
        return cleaned


if __name__ == "__main__":
    # Test the extractor
    async def test():
        extractor = DataExtractor()
        
        # Test single ticker
        result = await extractor.extract_all_data('AAPL')
        print(json.dumps(result, indent=2, default=str))
        
        # Test batch extraction
        tickers = ['AAPL', 'GOOGL', 'MSFT']
        batch_results = await extractor.batch_extract(tickers)
        print(f"Extracted data for {len(batch_results)} tickers")
    
    asyncio.run(test())