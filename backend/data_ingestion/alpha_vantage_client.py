"""
Alpha Vantage API Client - Free tier: 25 API requests per day
"""

import asyncio
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import logging
import json

from backend.data_ingestion.base_client import BaseAPIClient
from backend.utils.cache import get_redis

logger = logging.getLogger(__name__)


class AlphaVantageClient(BaseAPIClient):
    """
    Alpha Vantage API client optimized for free tier usage
    """
    
    def __init__(self):
        super().__init__("alpha_vantage")
        self.functions = {
            'quote': 'GLOBAL_QUOTE',
            'daily': 'TIME_SERIES_DAILY_ADJUSTED',
            'intraday': 'TIME_SERIES_INTRADAY',
            'weekly': 'TIME_SERIES_WEEKLY_ADJUSTED',
            'monthly': 'TIME_SERIES_MONTHLY_ADJUSTED',
            'earnings': 'EARNINGS',
            'overview': 'OVERVIEW',
            'income': 'INCOME_STATEMENT',
            'balance': 'BALANCE_SHEET',
            'cash_flow': 'CASH_FLOW',
            'sma': 'SMA',
            'ema': 'EMA',
            'macd': 'MACD',
            'rsi': 'RSI',
            'bbands': 'BBANDS',
            'adx': 'ADX',
            'cci': 'CCI',
            'aroon': 'AROON',
            'stoch': 'STOCH'
        }
    
    def _get_base_url(self) -> str:
        return "https://www.alphavantage.co/query"
    
    def _add_auth_params(self, params: Dict) -> Dict:
        """Add API key to parameters"""
        params['apikey'] = self.api_key
        return params
    
    async def get_quote(self, symbol: str) -> Optional[Dict]:
        """
        Get real-time quote (counts as 1 API call)
        """
        cache_key = f"av:quote:{symbol}"
        
        async def fetch():
            params = {
                'function': self.functions['quote'],
                'symbol': symbol
            }
            
            response = await self._make_request("", params)
            if response and 'Global Quote' in response:
                quote_data = response['Global Quote']
                return {
                    'symbol': symbol,
                    'price': float(quote_data.get('05. price', 0)),
                    'change': float(quote_data.get('09. change', 0)),
                    'change_percent': quote_data.get('10. change percent', '0%'),
                    'volume': int(quote_data.get('06. volume', 0)),
                    'latest_trading_day': quote_data.get('07. latest trading day'),
                    'previous_close': float(quote_data.get('08. previous close', 0)),
                    'open': float(quote_data.get('02. open', 0)),
                    'high': float(quote_data.get('03. high', 0)),
                    'low': float(quote_data.get('04. low', 0)),
                    'timestamp': datetime.utcnow().isoformat()
                }
            return None
        
        # Cache for 5 minutes for real-time quotes
        return await self.get_cached_or_fetch(cache_key, fetch, ttl=300)
    
    async def get_daily_prices(
        self,
        symbol: str,
        outputsize: str = 'compact'
    ) -> Optional[Dict]:
        """
        Get daily adjusted prices (counts as 1 API call)
        outputsize: 'compact' (100 days) or 'full' (20+ years)
        """
        cache_key = f"av:daily:{symbol}:{outputsize}"
        
        async def fetch():
            params = {
                'function': self.functions['daily'],
                'symbol': symbol,
                'outputsize': outputsize
            }
            
            response = await self._make_request("", params)
            if response and 'Time Series (Daily)' in response:
                time_series = response['Time Series (Daily)']
                
                # Convert to standardized format
                prices = []
                for date_str, data in time_series.items():
                    prices.append({
                        'date': date_str,
                        'open': float(data['1. open']),
                        'high': float(data['2. high']),
                        'low': float(data['3. low']),
                        'close': float(data['4. close']),
                        'adjusted_close': float(data['5. adjusted close']),
                        'volume': int(data['6. volume']),
                        'dividend': float(data.get('7. dividend amount', 0)),
                        'split_coefficient': float(data.get('8. split coefficient', 1))
                    })
                
                return {
                    'symbol': symbol,
                    'prices': sorted(prices, key=lambda x: x['date'], reverse=True),
                    'timestamp': datetime.utcnow().isoformat()
                }
            return None
        
        # Cache for 1 hour for daily data
        return await self.get_cached_or_fetch(cache_key, fetch, ttl=3600)
    
    async def get_company_overview(self, symbol: str) -> Optional[Dict]:
        """
        Get company fundamentals overview (counts as 1 API call)
        """
        cache_key = f"av:overview:{symbol}"
        
        async def fetch():
            params = {
                'function': self.functions['overview'],
                'symbol': symbol
            }
            
            response = await self._make_request("", params)
            if response and 'Symbol' in response:
                return {
                    'symbol': response['Symbol'],
                    'name': response.get('Name'),
                    'description': response.get('Description'),
                    'exchange': response.get('Exchange'),
                    'currency': response.get('Currency'),
                    'country': response.get('Country'),
                    'sector': response.get('Sector'),
                    'industry': response.get('Industry'),
                    'market_cap': int(response.get('MarketCapitalization', 0)),
                    'pe_ratio': float(response.get('PERatio', 0) or 0),
                    'peg_ratio': float(response.get('PEGRatio', 0) or 0),
                    'book_value': float(response.get('BookValue', 0) or 0),
                    'dividend_yield': float(response.get('DividendYield', 0) or 0),
                    'eps': float(response.get('EPS', 0) or 0),
                    'revenue_ttm': int(response.get('RevenueTTM', 0) or 0),
                    'profit_margin': float(response.get('ProfitMargin', 0) or 0),
                    'operating_margin': float(response.get('OperatingMarginTTM', 0) or 0),
                    'return_on_assets': float(response.get('ReturnOnAssetsTTM', 0) or 0),
                    'return_on_equity': float(response.get('ReturnOnEquityTTM', 0) or 0),
                    'revenue_per_share': float(response.get('RevenuePerShareTTM', 0) or 0),
                    'quarterly_earnings_growth': float(response.get('QuarterlyEarningsGrowthYOY', 0) or 0),
                    'quarterly_revenue_growth': float(response.get('QuarterlyRevenueGrowthYOY', 0) or 0),
                    'analyst_target_price': float(response.get('AnalystTargetPrice', 0) or 0),
                    'trailing_pe': float(response.get('TrailingPE', 0) or 0),
                    'forward_pe': float(response.get('ForwardPE', 0) or 0),
                    'price_to_sales': float(response.get('PriceToSalesRatioTTM', 0) or 0),
                    'price_to_book': float(response.get('PriceToBookRatio', 0) or 0),
                    'ev_to_revenue': float(response.get('EVToRevenue', 0) or 0),
                    'ev_to_ebitda': float(response.get('EVToEBITDA', 0) or 0),
                    'beta': float(response.get('Beta', 0) or 0),
                    '52_week_high': float(response.get('52WeekHigh', 0) or 0),
                    '52_week_low': float(response.get('52WeekLow', 0) or 0),
                    '50_day_ma': float(response.get('50DayMovingAverage', 0) or 0),
                    '200_day_ma': float(response.get('200DayMovingAverage', 0) or 0),
                    'shares_outstanding': int(response.get('SharesOutstanding', 0) or 0),
                    'dividend_date': response.get('DividendDate'),
                    'ex_dividend_date': response.get('ExDividendDate'),
                    'timestamp': datetime.utcnow().isoformat()
                }
            return None
        
        # Cache for 24 hours for fundamental data
        return await self.get_cached_or_fetch(cache_key, fetch, ttl=86400)
    
    async def get_technical_indicator(
        self,
        symbol: str,
        indicator: str,
        interval: str = 'daily',
        time_period: int = None,
        series_type: str = 'close'
    ) -> Optional[Dict]:
        """
        Get technical indicator data (counts as 1 API call per indicator)
        """
        cache_key = f"av:technical:{symbol}:{indicator}:{interval}:{time_period}"
        
        async def fetch():
            params = {
                'function': self.functions.get(indicator.lower()),
                'symbol': symbol,
                'interval': interval
            }
            
            # Add indicator-specific parameters
            if indicator.lower() in ['sma', 'ema', 'rsi', 'adx', 'cci']:
                params['time_period'] = time_period or 14
                params['series_type'] = series_type
            elif indicator.lower() == 'macd':
                params['series_type'] = series_type
            elif indicator.lower() == 'bbands':
                params['time_period'] = time_period or 20
                params['series_type'] = series_type
                params['nbdevup'] = 2
                params['nbdevdn'] = 2
            elif indicator.lower() == 'stoch':
                params['fastkperiod'] = 5
                params['slowkperiod'] = 3
                params['slowdperiod'] = 3
            
            response = await self._make_request("", params)
            if response:
                # Extract the technical analysis data
                ta_key = None
                for key in response.keys():
                    if 'Technical Analysis' in key:
                        ta_key = key
                        break
                
                if ta_key:
                    ta_data = response[ta_key]
                    
                    # Convert to standardized format
                    values = []
                    for date_str, data in ta_data.items():
                        value_dict = {'date': date_str}
                        value_dict.update({k: float(v) for k, v in data.items()})
                        values.append(value_dict)
                    
                    return {
                        'symbol': symbol,
                        'indicator': indicator,
                        'interval': interval,
                        'values': sorted(values, key=lambda x: x['date'], reverse=True),
                        'timestamp': datetime.utcnow().isoformat()
                    }
            return None
        
        # Cache for 1 hour for technical indicators
        return await self.get_cached_or_fetch(cache_key, fetch, ttl=3600)
    
    async def get_batch_quotes(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Get quotes for multiple symbols efficiently
        Note: Each symbol counts as 1 API call
        """
        # Alpha Vantage doesn't support true batch requests
        # We need to be very careful with the 25 daily limit
        
        results = {}
        redis = await get_redis()
        
        # Check how many API calls we've made today
        today = datetime.utcnow().strftime('%Y%m%d')
        daily_key = f"api_usage:alpha_vantage:daily:{today}"
        daily_count = int(await redis.get(daily_key) or 0)
        
        remaining_calls = max(0, 25 - daily_count)
        
        if remaining_calls == 0:
            logger.warning("Alpha Vantage daily limit reached, using cached data only")
            # Return cached data for all symbols
            for symbol in symbols:
                cache_key = f"av:quote:{symbol}"
                cached = await redis.get(cache_key)
                if cached:
                    results[symbol] = json.loads(cached)
            return results
        
        # Prioritize symbols that don't have recent cache
        symbols_to_fetch = []
        for symbol in symbols[:remaining_calls]:  # Only fetch what we can
            cache_key = f"av:quote:{symbol}"
            if not await redis.exists(cache_key):
                symbols_to_fetch.append(symbol)
        
        # Fetch with rate limiting (5 per minute = 12 seconds between requests)
        for i, symbol in enumerate(symbols_to_fetch):
            if i > 0:
                # Wait 12 seconds between each request for smooth rate limiting
                logger.debug(f"Alpha Vantage rate limit pause (12s between requests)")
                await asyncio.sleep(12)
            
            try:
                quote = await self.get_quote(symbol)
                if quote:
                    results[symbol] = quote
            except Exception as e:
                logger.error(f"Error fetching quote for {symbol}: {e}")
        
        # Fill in remaining with cached data
        for symbol in symbols:
            if symbol not in results:
                cache_key = f"av:quote:{symbol}"
                cached = await redis.get(cache_key)
                if cached:
                    results[symbol] = json.loads(cached)
        
        return results
    
    async def optimize_daily_calls(self, tickers: List[str]) -> Dict[str, str]:
        """
        Optimize the 25 daily API calls across different data types
        Returns a plan of which API calls to make
        """
        redis = await get_redis()
        today = datetime.utcnow().strftime('%Y%m%d')
        
        # Priority order for data freshness needs
        priorities = [
            ('quote', 300),  # 5 minutes
            ('overview', 86400),  # 24 hours
            ('daily', 3600),  # 1 hour
            ('technical', 3600)  # 1 hour
        ]
        
        call_plan = {}
        allocated_calls = 0
        
        for data_type, cache_ttl in priorities:
            if allocated_calls >= 25:
                break
                
            for ticker in tickers:
                if allocated_calls >= 25:
                    break
                    
                cache_key = f"av:{data_type}:{ticker}"
                
                # Check if data is stale
                ttl = await redis.ttl(cache_key)
                if ttl < 0 or ttl < cache_ttl * 0.1:  # Refresh if less than 10% TTL remaining
                    call_plan[f"{ticker}:{data_type}"] = {
                        'ticker': ticker,
                        'data_type': data_type,
                        'priority': priorities.index((data_type, cache_ttl))
                    }
                    allocated_calls += 1
        
        logger.info(f"Alpha Vantage daily call plan: {len(call_plan)} calls allocated")
        return call_plan