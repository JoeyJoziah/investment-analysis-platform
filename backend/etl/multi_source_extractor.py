"""
Multi-Source Data Extractor with Intelligence Routing and Fallback
Handles 6000+ stocks with optimal rate limiting and data source prioritization
"""

import asyncio
import aiohttp
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Any, Optional, Set
import yfinance as yf
import pandas as pd
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential
import os
from dotenv import load_dotenv
import json
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import sqlite3
import hashlib
from dataclasses import dataclass
# SECURITY: Removed pickle - using JSON for safer serialization

# Import our web scrapers
from .web_scrapers import get_scraper, WebScraperBase

load_dotenv()
logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    ticker: str
    success: bool
    data: Optional[Dict] = None
    source: Optional[str] = None
    error: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class SourcePriority:
    name: str
    priority: int
    success_rate: float
    last_success: Optional[datetime] = None
    consecutive_failures: int = 0
    enabled: bool = True


class IntelligentSourceRouter:
    """Routes requests to optimal data sources based on success rates and availability"""
    
    def __init__(self):
        self.source_stats = {}
        self.source_priorities = {
            'yahoo_scraper': SourcePriority('yahoo_scraper', 1, 0.9),
            'yfinance': SourcePriority('yfinance', 2, 0.8),
            'alpha_vantage': SourcePriority('alpha_vantage', 3, 0.95),
            'finnhub': SourcePriority('finnhub', 4, 0.85),
            'polygon': SourcePriority('polygon', 5, 0.75),
            'marketwatch_scraper': SourcePriority('marketwatch_scraper', 6, 0.7),
            'google_finance_scraper': SourcePriority('google_finance_scraper', 7, 0.6)
        }
    
    def get_optimal_sources(self, ticker: str, max_sources: int = 3) -> List[str]:
        """Get optimal data sources for a ticker based on current conditions"""
        available_sources = []
        
        for name, priority in self.source_priorities.items():
            if not priority.enabled:
                continue
            
            # Calculate dynamic score based on success rate and recent failures
            base_score = priority.success_rate
            failure_penalty = min(priority.consecutive_failures * 0.1, 0.5)
            final_score = base_score - failure_penalty
            
            # Boost score if source had recent success
            if priority.last_success:
                hours_since_success = (datetime.now() - priority.last_success).total_seconds() / 3600
                if hours_since_success < 1:
                    final_score += 0.1
            
            available_sources.append((name, final_score, priority.priority))
        
        # Sort by score (descending) then by priority (ascending)
        available_sources.sort(key=lambda x: (-x[1], x[2]))
        
        return [source[0] for source in available_sources[:max_sources]]
    
    def record_success(self, source: str):
        """Record successful extraction from source"""
        if source in self.source_priorities:
            priority = self.source_priorities[source]
            priority.last_success = datetime.now()
            priority.consecutive_failures = 0
            # Gradually improve success rate
            priority.success_rate = min(priority.success_rate + 0.01, 1.0)
    
    def record_failure(self, source: str):
        """Record failed extraction from source"""
        if source in self.source_priorities:
            priority = self.source_priorities[source]
            priority.consecutive_failures += 1
            # Degrade success rate
            priority.success_rate = max(priority.success_rate - 0.02, 0.1)
            
            # Temporarily disable source if too many consecutive failures
            if priority.consecutive_failures >= 5:
                priority.enabled = False
                logger.warning(f"Temporarily disabled {source} due to consecutive failures")
    
    def reset_source(self, source: str):
        """Reset source stats (e.g., after rate limit period)"""
        if source in self.source_priorities:
            priority = self.source_priorities[source]
            priority.enabled = True
            priority.consecutive_failures = 0


class MultiSourceStockExtractor:
    """Main extractor class that orchestrates multiple data sources"""
    
    def __init__(self, cache_dir: str = "/tmp/stock_cache", max_concurrent: int = 10):
        self.cache_dir = cache_dir
        self.max_concurrent = max_concurrent
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize components
        self.router = IntelligentSourceRouter()
        self.progress_db_path = os.path.join(cache_dir, "extraction_progress.db")
        self._init_progress_db()
        
        # Load API keys
        self.api_keys = {
            'alpha_vantage': os.getenv('ALPHA_VANTAGE_API_KEY'),
            'finnhub': os.getenv('FINNHUB_API_KEY'),
            'polygon': os.getenv('POLYGON_API_KEY'),
            'news_api': os.getenv('NEWS_API_KEY')
        }
        
        # Rate limiting trackers
        self.rate_limits = {
            'yahoo_scraper': {'calls': [], 'max_per_hour': 300, 'delay': 2},
            'yfinance': {'calls': [], 'max_per_hour': 100, 'delay': 3},
            'alpha_vantage': {'calls': [], 'max_per_day': 25, 'delay': 60},
            'finnhub': {'calls': [], 'max_per_hour': 60, 'delay': 3},
            'polygon': {'calls': [], 'max_per_hour': 60, 'delay': 5},
            'marketwatch_scraper': {'calls': [], 'max_per_hour': 200, 'delay': 4},
            'google_finance_scraper': {'calls': [], 'max_per_hour': 150, 'delay': 5}
        }
        
        # Cache settings
        self.cache_expiry_hours = 4
        
        logger.info(f"Initialized MultiSourceStockExtractor with {len(self.api_keys)} API keys")
    
    def _init_progress_db(self):
        """Initialize progress tracking database"""
        conn = sqlite3.connect(self.progress_db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS extraction_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT,
                source TEXT,
                status TEXT,
                timestamp TIMESTAMP,
                error_message TEXT,
                data_quality_score REAL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ticker_progress (
                ticker TEXT PRIMARY KEY,
                last_successful_extraction TIMESTAMP,
                successful_sources TEXT,
                failed_sources TEXT,
                total_attempts INTEGER DEFAULT 0,
                data_completeness_score REAL DEFAULT 0.0
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _can_make_request(self, source: str) -> bool:
        """Check if we can make a request to the source"""
        if source not in self.rate_limits:
            return True
        
        limit_config = self.rate_limits[source]
        current_time = time.time()
        
        # Clean old calls
        if 'max_per_hour' in limit_config:
            hour_ago = current_time - 3600
            limit_config['calls'] = [t for t in limit_config['calls'] if t > hour_ago]
            return len(limit_config['calls']) < limit_config['max_per_hour']
        elif 'max_per_day' in limit_config:
            day_ago = current_time - 86400
            limit_config['calls'] = [t for t in limit_config['calls'] if t > day_ago]
            return len(limit_config['calls']) < limit_config['max_per_day']
        
        return True
    
    def _record_request(self, source: str):
        """Record a request to track rate limits"""
        if source in self.rate_limits:
            self.rate_limits[source]['calls'].append(time.time())
    
    async def _get_cached_data(self, ticker: str, source: str) -> Optional[Dict]:
        """Check for cached data"""
        cache_key = f"{ticker}_{source}_{datetime.now().strftime('%Y%m%d')}"
        # SECURITY: Use JSON files instead of pickle
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")

        if os.path.exists(cache_file):
            try:
                # Check if cache is still valid
                file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
                if (datetime.now() - file_time).total_seconds() < self.cache_expiry_hours * 3600:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        return json.load(f)
            except Exception as e:
                logger.debug(f"Error loading cache for {cache_key}: {e}")

        return None

    async def _save_to_cache(self, ticker: str, source: str, data: Dict):
        """Save data to cache"""
        cache_key = f"{ticker}_{source}_{datetime.now().strftime('%Y%m%d')}"
        # SECURITY: Use JSON files instead of pickle
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")

        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f)
        except Exception as e:
            logger.debug(f"Error saving cache for {cache_key}: {e}")
    
    async def extract_from_source(self, ticker: str, source: str) -> ExtractionResult:
        """Extract data from a specific source"""
        # Check cache first
        cached_data = await self._get_cached_data(ticker, source)
        if cached_data:
            logger.debug(f"Using cached data for {ticker} from {source}")
            return ExtractionResult(
                ticker=ticker,
                success=True,
                data=cached_data,
                source=source
            )
        
        # Check rate limits
        if not self._can_make_request(source):
            return ExtractionResult(
                ticker=ticker,
                success=False,
                error=f"Rate limit exceeded for {source}",
                source=source
            )
        
        try:
            data = None
            
            if source == 'yahoo_scraper':
                scraper = get_scraper('yahoo_scraper')
                data = await scraper.scrape_stock_data(ticker)
            
            elif source == 'yfinance':
                data = await self._fetch_yfinance_data(ticker)
            
            elif source == 'alpha_vantage' and self.api_keys.get('alpha_vantage'):
                data = await self._fetch_alpha_vantage_data(ticker)
            
            elif source == 'finnhub' and self.api_keys.get('finnhub'):
                data = await self._fetch_finnhub_data(ticker)
            
            elif source == 'polygon' and self.api_keys.get('polygon'):
                data = await self._fetch_polygon_data(ticker)
            
            elif source == 'marketwatch_scraper':
                scraper = get_scraper('marketwatch_scraper')
                data = await scraper.scrape_stock_data(ticker)
            
            elif source == 'google_finance_scraper':
                scraper = get_scraper('google_finance_scraper')
                data = await scraper.scrape_stock_data(ticker)
            
            if data:
                self._record_request(source)
                await self._save_to_cache(ticker, source, data)
                self.router.record_success(source)
                
                return ExtractionResult(
                    ticker=ticker,
                    success=True,
                    data=data,
                    source=source
                )
            else:
                self.router.record_failure(source)
                return ExtractionResult(
                    ticker=ticker,
                    success=False,
                    error=f"No data returned from {source}",
                    source=source
                )
        
        except Exception as e:
            self.router.record_failure(source)
            logger.error(f"Error extracting {ticker} from {source}: {e}")
            return ExtractionResult(
                ticker=ticker,
                success=False,
                error=str(e),
                source=source
            )
    
    async def _fetch_yfinance_data(self, ticker: str) -> Optional[Dict]:
        """Fetch data using yfinance library"""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1mo")
            info = stock.info
            
            if hist.empty:
                return None
            
            latest = hist.iloc[-1]
            
            return {
                'ticker': ticker,
                'source': 'yfinance',
                'timestamp': datetime.now(),
                'current_price': float(latest['Close']),
                'volume': int(latest['Volume']),
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'company_name': info.get('shortName', ticker),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'historical_data': hist.tail(30).to_dict('records')
            }
        except Exception as e:
            logger.debug(f"yfinance error for {ticker}: {e}")
            return None
    
    async def _fetch_alpha_vantage_data(self, ticker: str) -> Optional[Dict]:
        """Fetch data from Alpha Vantage API"""
        api_key = self.api_keys.get('alpha_vantage')
        if not api_key:
            return None
        
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://www.alphavantage.co/query"
                params = {
                    'function': 'GLOBAL_QUOTE',
                    'symbol': ticker,
                    'apikey': api_key
                }
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if 'Global Quote' in data:
                            quote = data['Global Quote']
                            return {
                                'ticker': ticker,
                                'source': 'alpha_vantage',
                                'timestamp': datetime.now(),
                                'current_price': float(quote.get('05. price', 0)),
                                'price_change': float(quote.get('09. change', 0)),
                                'price_change_pct': float(quote.get('10. change percent', '0').replace('%', '')),
                                'volume': int(quote.get('06. volume', 0)),
                                'previous_close': float(quote.get('08. previous close', 0))
                            }
            
            await asyncio.sleep(1)  # Alpha Vantage rate limit
            return None
            
        except Exception as e:
            logger.debug(f"Alpha Vantage error for {ticker}: {e}")
            return None
    
    async def _fetch_finnhub_data(self, ticker: str) -> Optional[Dict]:
        """Fetch data from Finnhub API"""
        api_key = self.api_keys.get('finnhub')
        if not api_key:
            return None
        
        try:
            async with aiohttp.ClientSession() as session:
                # Get quote
                quote_url = "https://finnhub.io/api/v1/quote"
                params = {'symbol': ticker, 'token': api_key}
                
                async with session.get(quote_url, params=params) as response:
                    if response.status == 200:
                        quote_data = await response.json()
                        
                        if quote_data.get('c'):  # Current price exists
                            return {
                                'ticker': ticker,
                                'source': 'finnhub',
                                'timestamp': datetime.now(),
                                'current_price': float(quote_data.get('c', 0)),
                                'high': float(quote_data.get('h', 0)),
                                'low': float(quote_data.get('l', 0)),
                                'open': float(quote_data.get('o', 0)),
                                'previous_close': float(quote_data.get('pc', 0))
                            }
            
            return None
            
        except Exception as e:
            logger.debug(f"Finnhub error for {ticker}: {e}")
            return None
    
    async def _fetch_polygon_data(self, ticker: str) -> Optional[Dict]:
        """Fetch data from Polygon API"""
        api_key = self.api_keys.get('polygon')
        if not api_key:
            return None
        
        try:
            async with aiohttp.ClientSession() as session:
                # Get previous close
                url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/prev"
                params = {'apikey': api_key}
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if 'results' in data and data['results']:
                            result = data['results'][0]
                            return {
                                'ticker': ticker,
                                'source': 'polygon',
                                'timestamp': datetime.now(),
                                'current_price': float(result.get('c', 0)),
                                'high': float(result.get('h', 0)),
                                'low': float(result.get('l', 0)),
                                'open': float(result.get('o', 0)),
                                'volume': int(result.get('v', 0))
                            }
            
            return None
            
        except Exception as e:
            logger.debug(f"Polygon error for {ticker}: {e}")
            return None
    
    async def extract_stock_data(self, ticker: str, max_attempts: int = 3) -> ExtractionResult:
        """Extract data for a single stock using multiple sources with fallback"""
        optimal_sources = self.router.get_optimal_sources(ticker, max_attempts)
        
        for source in optimal_sources:
            result = await self.extract_from_source(ticker, source)
            
            if result.success and result.data:
                # Log successful extraction
                self._log_extraction(ticker, source, 'success', None)
                return result
            else:
                # Log failed extraction
                self._log_extraction(ticker, source, 'failed', result.error)
                
                # Add delay before trying next source
                delay = self.rate_limits.get(source, {}).get('delay', 1)
                await asyncio.sleep(delay + random.uniform(0, 2))
        
        # All sources failed
        return ExtractionResult(
            ticker=ticker,
            success=False,
            error=f"All sources failed for {ticker}: {optimal_sources}"
        )
    
    def _log_extraction(self, ticker: str, source: str, status: str, error: str):
        """Log extraction attempt to database"""
        try:
            conn = sqlite3.connect(self.progress_db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO extraction_log (ticker, source, status, timestamp, error_message)
                VALUES (?, ?, ?, ?, ?)
            """, (ticker, source, status, datetime.now().isoformat(), error))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"Error logging extraction: {e}")
    
    async def batch_extract(self, tickers: List[str], 
                          batch_size: int = 50, 
                          max_concurrent: int = 10) -> List[ExtractionResult]:
        """Extract data for multiple tickers in batches with concurrency control"""
        all_results = []
        
        # Process in batches to manage memory and rate limits
        for i in range(0, len(tickers), batch_size):
            batch_tickers = tickers[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}: {len(batch_tickers)} tickers")
            
            # Control concurrency within each batch
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def extract_with_semaphore(ticker):
                async with semaphore:
                    return await self.extract_stock_data(ticker)
            
            # Execute batch with controlled concurrency
            tasks = [extract_with_semaphore(ticker) for ticker in batch_tickers]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Batch extraction error: {result}")
                else:
                    all_results.append(result)
            
            # Progress report
            successful = len([r for r in all_results if r.success])
            logger.info(f"Batch {i//batch_size + 1} complete: {successful}/{len(all_results)} successful")
            
            # Inter-batch delay to be respectful to sources
            if i + batch_size < len(tickers):
                await asyncio.sleep(random.uniform(5, 10))
        
        return all_results
    
    def get_extraction_stats(self) -> Dict:
        """Get statistics about extraction performance"""
        try:
            conn = sqlite3.connect(self.progress_db_path)
            cursor = conn.cursor()
            
            # Get overall stats
            cursor.execute("""
                SELECT source, status, COUNT(*) as count
                FROM extraction_log
                WHERE timestamp > datetime('now', '-24 hours')
                GROUP BY source, status
            """)
            
            stats = {'sources': {}}
            for row in cursor.fetchall():
                source, status, count = row
                if source not in stats['sources']:
                    stats['sources'][source] = {}
                stats['sources'][source][status] = count
            
            conn.close()
            return stats
            
        except Exception as e:
            logger.error(f"Error getting extraction stats: {e}")
            return {}
    
    def reset_failed_sources(self):
        """Reset sources that were temporarily disabled"""
        for source in self.router.source_priorities:
            self.router.reset_source(source)
        logger.info("Reset all temporarily disabled sources")


# Convenience function for easy usage
async def extract_stocks_data(tickers: List[str], 
                            cache_dir: str = "/tmp/stock_cache",
                            batch_size: int = 50,
                            max_concurrent: int = 10) -> List[ExtractionResult]:
    """
    Main function to extract data for multiple stocks
    
    Args:
        tickers: List of ticker symbols
        cache_dir: Directory for caching data
        batch_size: Number of tickers to process in each batch
        max_concurrent: Maximum concurrent requests
    
    Returns:
        List of ExtractionResult objects
    """
    extractor = MultiSourceStockExtractor(cache_dir=cache_dir, max_concurrent=max_concurrent)
    
    try:
        results = await extractor.batch_extract(
            tickers=tickers,
            batch_size=batch_size,
            max_concurrent=max_concurrent
        )
        
        # Print summary
        successful = len([r for r in results if r.success])
        logger.info(f"Extraction complete: {successful}/{len(results)} successful")
        
        # Print source statistics
        stats = extractor.get_extraction_stats()
        if stats.get('sources'):
            logger.info(f"Source performance: {stats['sources']}")
        
        return results
        
    except Exception as e:
        logger.error(f"Batch extraction failed: {e}")
        return []


if __name__ == "__main__":
    # Test the extractor
    async def test():
        test_tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
        results = await extract_stocks_data(test_tickers, batch_size=5, max_concurrent=3)
        
        for result in results:
            if result.success:
                print(f"✓ {result.ticker}: {result.source} - ${result.data.get('current_price', 'N/A')}")
            else:
                print(f"✗ {result.ticker}: {result.error}")
    
    asyncio.run(test())