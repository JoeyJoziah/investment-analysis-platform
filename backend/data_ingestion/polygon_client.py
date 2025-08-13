"""
Polygon.io API client for market data ingestion
Free tier: 5 API calls per minute
"""
import aiohttp
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta, date
import logging
import os
import json

from backend.utils.rate_limiter import RateLimiter
from backend.utils.circuit_breaker import CircuitBreaker
from backend.utils.cache import get_redis_client

logger = logging.getLogger(__name__)

class PolygonClient:
    """Client for Polygon.io API with rate limiting and circuit breaker"""
    
    BASE_URL = "https://api.polygon.io"
    
    def __init__(self):
        self.api_key = os.getenv('POLYGON_API_KEY')
        if not self.api_key:
            raise ValueError("POLYGON_API_KEY not set in environment")
        
        # Simple rate limiting tracking
        self.last_call_time = 0
        self.calls_per_minute = 5
        self.min_interval = 60 / self.calls_per_minute  # 12 seconds between calls
        
        # Circuit breaker to handle API failures
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=300,  # 5 minutes
            expected_exception=aiohttp.ClientError
        )
        
        self.session = None
        self.redis_client = get_redis_client()
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make API request with rate limiting and circuit breaker"""
        
        # Simple rate limiting
        import time
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_time
        if time_since_last_call < self.min_interval:
            await asyncio.sleep(self.min_interval - time_since_last_call)
        self.last_call_time = time.time()
        
        # Add API key to params
        if params is None:
            params = {}
        params['apiKey'] = self.api_key
        
        # Check cache first
        cache_key = f"polygon:{endpoint}:{json.dumps(sorted(params.items()))}"
        cached = self.redis_client.get(cache_key)
        if cached:
            logger.info(f"Cache hit for {cache_key}")
            return json.loads(cached)
        
        # Make request with circuit breaker
        async def make_call():
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            url = f"{self.BASE_URL}{endpoint}"
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Check for API errors
                    if data.get('status') == 'ERROR':
                        raise ValueError(f"API Error: {data.get('error', 'Unknown error')}")
                    
                    # Cache successful response
                    cache_ttl = 300  # 5 minutes for real-time data
                    self.redis_client.setex(cache_key, cache_ttl, json.dumps(data))
                    
                    return data
                elif response.status == 429:
                    raise Exception("Rate limit exceeded")
                else:
                    raise aiohttp.ClientError(f"HTTP {response.status}: {await response.text()}")
        
        return await self.circuit_breaker.call(make_call)
    
    async def get_ticker_details(self, symbol: str) -> Dict[str, Any]:
        """Get detailed information about a ticker"""
        try:
            endpoint = f"/v3/reference/tickers/{symbol}"
            data = await self._make_request(endpoint)
            
            if data.get('status') == 'OK' and data.get('results'):
                result = data['results']
                return {
                    'symbol': result.get('ticker'),
                    'name': result.get('name'),
                    'market': result.get('market'),
                    'locale': result.get('locale'),
                    'primary_exchange': result.get('primary_exchange'),
                    'type': result.get('type'),
                    'active': result.get('active'),
                    'currency': result.get('currency_name'),
                    'cik': result.get('cik'),
                    'composite_figi': result.get('composite_figi'),
                    'share_class_figi': result.get('share_class_figi'),
                    'market_cap': result.get('market_cap'),
                    'phone_number': result.get('phone_number'),
                    'address': result.get('address'),
                    'description': result.get('description'),
                    'sic_code': result.get('sic_code'),
                    'sic_description': result.get('sic_description'),
                    'ticker_root': result.get('ticker_root'),
                    'homepage_url': result.get('homepage_url'),
                    'total_employees': result.get('total_employees'),
                    'list_date': result.get('list_date'),
                    'branding': result.get('branding'),
                    'share_class_shares_outstanding': result.get('share_class_shares_outstanding'),
                    'weighted_shares_outstanding': result.get('weighted_shares_outstanding')
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting ticker details for {symbol}: {e}")
            raise
    
    async def get_last_quote(self, symbol: str) -> Dict[str, Any]:
        """Get the most recent NBBO (quote) for a ticker"""
        try:
            endpoint = f"/v2/last/nbbo/{symbol}"
            data = await self._make_request(endpoint)
            
            if data.get('status') == 'OK' and data.get('results'):
                result = data['results']
                return {
                    'symbol': result.get('T'),
                    'bid_price': result.get('p'),
                    'bid_size': result.get('s'),
                    'bid_exchange': result.get('x'),
                    'ask_price': result.get('P'),
                    'ask_size': result.get('S'),
                    'ask_exchange': result.get('X'),
                    'timestamp': result.get('t'),
                    'conditions': result.get('c'),
                    'indicators': result.get('i')
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting last quote for {symbol}: {e}")
            raise
    
    async def get_last_trade(self, symbol: str) -> Dict[str, Any]:
        """Get the most recent trade for a ticker"""
        try:
            endpoint = f"/v2/last/trade/{symbol}"
            data = await self._make_request(endpoint)
            
            if data.get('status') == 'OK' and data.get('results'):
                result = data['results']
                return {
                    'symbol': result.get('T'),
                    'price': result.get('p'),
                    'size': result.get('s'),
                    'exchange': result.get('x'),
                    'conditions': result.get('c'),
                    'timestamp': result.get('t'),
                    'participant_timestamp': result.get('y')
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting last trade for {symbol}: {e}")
            raise
    
    async def get_daily_open_close(self, symbol: str, date_str: str) -> Dict[str, Any]:
        """Get the open, high, low, close prices for a stock on a specific date"""
        try:
            endpoint = f"/v1/open-close/{symbol}/{date_str}"
            data = await self._make_request(endpoint)
            
            if data.get('status') == 'OK':
                return {
                    'symbol': data.get('symbol'),
                    'date': data.get('from'),
                    'open': data.get('open'),
                    'high': data.get('high'),
                    'low': data.get('low'),
                    'close': data.get('close'),
                    'volume': data.get('volume'),
                    'after_hours': data.get('afterHours'),
                    'pre_market': data.get('preMarket')
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting daily OHLC for {symbol} on {date_str}: {e}")
            raise
    
    async def get_aggregates(self, 
                            symbol: str,
                            multiplier: int = 1,
                            timespan: str = 'day',
                            from_date: str = None,
                            to_date: str = None,
                            limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get aggregate bars for a ticker over a given date range
        
        Args:
            symbol: Ticker symbol
            multiplier: Size of the timespan multiplier
            timespan: minute, hour, day, week, month, quarter, year
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            limit: Number of results
        """
        try:
            if not from_date:
                from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            if not to_date:
                to_date = datetime.now().strftime('%Y-%m-%d')
            
            endpoint = f"/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
            params = {
                'adjusted': 'true',
                'sort': 'desc',
                'limit': str(limit)
            }
            
            data = await self._make_request(endpoint, params)
            
            if data.get('status') == 'OK' and data.get('results'):
                results = []
                for bar in data['results']:
                    results.append({
                        'timestamp': bar.get('t'),
                        'open': bar.get('o'),
                        'high': bar.get('h'),
                        'low': bar.get('l'),
                        'close': bar.get('c'),
                        'volume': bar.get('v'),
                        'vwap': bar.get('vw'),
                        'transactions': bar.get('n')
                    })
                return results
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting aggregates for {symbol}: {e}")
            raise
    
    async def get_snapshot(self, symbol: str) -> Dict[str, Any]:
        """Get snapshot of a ticker"""
        try:
            endpoint = f"/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}"
            data = await self._make_request(endpoint)
            
            if data.get('status') == 'OK' and data.get('ticker'):
                ticker_data = data['ticker']
                return {
                    'symbol': ticker_data.get('ticker'),
                    'day': ticker_data.get('day', {}),
                    'last_quote': ticker_data.get('lastQuote', {}),
                    'last_trade': ticker_data.get('lastTrade', {}),
                    'min': ticker_data.get('min', {}),
                    'prev_day': ticker_data.get('prevDay', {}),
                    'updated': ticker_data.get('updated')
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting snapshot for {symbol}: {e}")
            raise
    
    async def get_market_status(self) -> Dict[str, Any]:
        """Get current market status"""
        try:
            endpoint = "/v1/marketstatus/now"
            data = await self._make_request(endpoint)
            
            return {
                'market': data.get('market'),
                'server_time': data.get('serverTime'),
                'exchanges': data.get('exchanges', {}),
                'currencies': data.get('currencies', {})
            }
            
        except Exception as e:
            logger.error(f"Error getting market status: {e}")
            raise
    
    async def get_tickers(self, 
                         market: str = 'stocks',
                         exchange: str = None,
                         type: str = None,
                         active: bool = True,
                         limit: int = 100) -> List[Dict[str, Any]]:
        """Get a list of tickers"""
        try:
            endpoint = "/v3/reference/tickers"
            params = {
                'market': market,
                'active': str(active).lower(),
                'limit': str(limit),
                'sort': 'ticker',
                'order': 'asc'
            }
            
            if exchange:
                params['exchange'] = exchange
            if type:
                params['type'] = type
            
            data = await self._make_request(endpoint, params)
            
            if data.get('status') == 'OK' and data.get('results'):
                results = []
                for ticker in data['results']:
                    results.append({
                        'symbol': ticker.get('ticker'),
                        'name': ticker.get('name'),
                        'market': ticker.get('market'),
                        'locale': ticker.get('locale'),
                        'primary_exchange': ticker.get('primary_exchange'),
                        'type': ticker.get('type'),
                        'active': ticker.get('active'),
                        'currency': ticker.get('currency_name'),
                        'cik': ticker.get('cik'),
                        'last_updated': ticker.get('last_updated_utc')
                    })
                return results
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting tickers: {e}")
            raise
    
    async def get_dividends(self, symbol: str) -> List[Dict[str, Any]]:
        """Get dividend history for a ticker"""
        try:
            endpoint = f"/v3/reference/dividends"
            params = {
                'ticker': symbol,
                'limit': '100',
                'sort': 'ex_dividend_date',
                'order': 'desc'
            }
            
            data = await self._make_request(endpoint, params)
            
            if data.get('status') == 'OK' and data.get('results'):
                results = []
                for div in data['results']:
                    results.append({
                        'symbol': div.get('ticker'),
                        'ex_dividend_date': div.get('ex_dividend_date'),
                        'payment_date': div.get('payment_date'),
                        'record_date': div.get('record_date'),
                        'amount': div.get('cash_amount'),
                        'currency': div.get('currency'),
                        'declaration_date': div.get('declaration_date'),
                        'frequency': div.get('frequency'),
                        'type': div.get('dividend_type')
                    })
                return results
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting dividends for {symbol}: {e}")
            raise
    
    async def get_splits(self, symbol: str) -> List[Dict[str, Any]]:
        """Get stock split history for a ticker"""
        try:
            endpoint = f"/v3/reference/splits"
            params = {
                'ticker': symbol,
                'limit': '100',
                'sort': 'execution_date',
                'order': 'desc'
            }
            
            data = await self._make_request(endpoint, params)
            
            if data.get('status') == 'OK' and data.get('results'):
                results = []
                for split in data['results']:
                    results.append({
                        'symbol': split.get('ticker'),
                        'execution_date': split.get('execution_date'),
                        'split_from': split.get('split_from'),
                        'split_to': split.get('split_to')
                    })
                return results
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting splits for {symbol}: {e}")
            raise
    
    async def get_financials(self, symbol: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get financial statements for a ticker"""
        try:
            endpoint = "/v2/reference/financials/{symbol}"
            params = {
                'limit': str(limit),
                'sort': 'report_period',
                'order': 'desc'
            }
            
            data = await self._make_request(endpoint, params)
            
            if data.get('status') == 'OK' and data.get('results'):
                return data['results']
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting financials for {symbol}: {e}")
            raise
    
    async def batch_snapshot(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get snapshots for multiple symbols (be careful with rate limits)
        
        Args:
            symbols: List of stock symbols (max 5 due to rate limit)
        
        Returns:
            Dictionary mapping symbols to snapshot data
        """
        results = {}
        
        # Limit to 5 symbols to respect rate limit
        for symbol in symbols[:5]:
            try:
                # Add delay to respect rate limits (5 per minute = 12 seconds between calls)
                await asyncio.sleep(12)
                
                snapshot = await self.get_snapshot(symbol)
                if snapshot:
                    results[symbol] = snapshot
                
            except Exception as e:
                logger.error(f"Error getting snapshot for {symbol}: {e}")
                results[symbol] = {'error': str(e)}
        
        return results
    
    async def get_news(self, symbol: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get news articles for a ticker or market-wide"""
        try:
            endpoint = "/v2/reference/news"
            params = {
                'limit': str(limit),
                'sort': 'published_utc',
                'order': 'desc'
            }
            
            if symbol:
                params['ticker'] = symbol
            
            data = await self._make_request(endpoint, params)
            
            if data.get('status') == 'OK' and data.get('results'):
                results = []
                for article in data['results']:
                    results.append({
                        'title': article.get('title'),
                        'author': article.get('author'),
                        'published': article.get('published_utc'),
                        'article_url': article.get('article_url'),
                        'tickers': article.get('tickers', []),
                        'amp_url': article.get('amp_url'),
                        'id': article.get('id'),
                        'publisher': article.get('publisher', {}).get('name'),
                        'keywords': article.get('keywords', []),
                        'description': article.get('description')
                    })
                return results
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting news: {e}")
            raise
    
    def close(self):
        """Close the client session"""
        if self.session:
            asyncio.create_task(self.session.close())

# Example usage
async def main():
    """Example usage of PolygonClient"""
    async with PolygonClient() as client:
        # Get ticker details
        details = await client.get_ticker_details('AAPL')
        print(f"AAPL Details: {details}")
        
        # Get last trade
        trade = await client.get_last_trade('AAPL')
        print(f"AAPL Last Trade: {trade}")
        
        # Get market status
        status = await client.get_market_status()
        print(f"Market Status: {status}")

if __name__ == "__main__":
    asyncio.run(main())