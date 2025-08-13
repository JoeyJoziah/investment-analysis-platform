"""
Finnhub API Client - Free tier: 60 API calls per minute
Best free tier for real-time data
"""

import asyncio
from typing import Dict, Optional, List, Any
from datetime import datetime, timedelta
import logging
import json

from backend.data_ingestion.base_client import BaseAPIClient
from backend.utils.cache import get_redis

logger = logging.getLogger(__name__)


class FinnhubClient(BaseAPIClient):
    """
    Finnhub API client - primary data source due to generous free tier
    """
    
    def __init__(self):
        super().__init__("finnhub")
        self.websocket_url = "wss://ws.finnhub.io"
    
    def _get_base_url(self) -> str:
        return "https://finnhub.io/api/v1"
    
    def _add_auth_params(self, params: Dict) -> Dict:
        """Add API key to parameters"""
        params['token'] = self.api_key
        return params
    
    async def get_quote(self, symbol: str) -> Optional[Dict]:
        """
        Get real-time quote data
        """
        cache_key = f"finnhub:quote:{symbol}"
        
        async def fetch():
            params = {'symbol': symbol}
            response = await self._make_request("quote", params)
            
            if response:
                return {
                    'symbol': symbol,
                    'current_price': response.get('c', 0),
                    'change': response.get('d', 0),
                    'percent_change': response.get('dp', 0),
                    'high': response.get('h', 0),
                    'low': response.get('l', 0),
                    'open': response.get('o', 0),
                    'previous_close': response.get('pc', 0),
                    'timestamp': datetime.fromtimestamp(response.get('t', 0)).isoformat() if response.get('t') else datetime.utcnow().isoformat()
                }
            return None
        
        # Cache for 1 minute for real-time quotes
        return await self.get_cached_or_fetch(cache_key, fetch, ttl=60)
    
    async def get_company_profile(self, symbol: str) -> Optional[Dict]:
        """
        Get company profile information
        """
        cache_key = f"finnhub:profile:{symbol}"
        
        async def fetch():
            params = {'symbol': symbol}
            response = await self._make_request("stock/profile2", params)
            
            if response:
                return {
                    'symbol': symbol,
                    'name': response.get('name'),
                    'country': response.get('country'),
                    'currency': response.get('currency'),
                    'exchange': response.get('exchange'),
                    'industry': response.get('finnhubIndustry'),
                    'ipo_date': response.get('ipo'),
                    'market_cap': response.get('marketCapitalization', 0) * 1000000,  # Convert to actual value
                    'shares_outstanding': response.get('shareOutstanding', 0) * 1000000,
                    'logo': response.get('logo'),
                    'phone': response.get('phone'),
                    'weburl': response.get('weburl'),
                    'ticker': response.get('ticker'),
                    'timestamp': datetime.utcnow().isoformat()
                }
            return None
        
        # Cache for 24 hours
        return await self.get_cached_or_fetch(cache_key, fetch, ttl=86400)
    
    async def get_candles(
        self,
        symbol: str,
        resolution: str = 'D',
        from_timestamp: int = None,
        to_timestamp: int = None
    ) -> Optional[Dict]:
        """
        Get candlestick data
        resolution: 1, 5, 15, 30, 60, D, W, M
        """
        if not from_timestamp:
            from_timestamp = int((datetime.utcnow() - timedelta(days=365)).timestamp())
        if not to_timestamp:
            to_timestamp = int(datetime.utcnow().timestamp())
        
        cache_key = f"finnhub:candles:{symbol}:{resolution}:{from_timestamp}:{to_timestamp}"
        
        async def fetch():
            params = {
                'symbol': symbol,
                'resolution': resolution,
                'from': from_timestamp,
                'to': to_timestamp
            }
            response = await self._make_request("stock/candle", params)
            
            if response and response.get('s') == 'ok':
                candles = []
                for i in range(len(response['t'])):
                    candles.append({
                        'timestamp': response['t'][i],
                        'date': datetime.fromtimestamp(response['t'][i]).isoformat(),
                        'open': response['o'][i],
                        'high': response['h'][i],
                        'low': response['l'][i],
                        'close': response['c'][i],
                        'volume': response['v'][i]
                    })
                
                return {
                    'symbol': symbol,
                    'resolution': resolution,
                    'candles': candles,
                    'timestamp': datetime.utcnow().isoformat()
                }
            return None
        
        # Cache based on resolution
        ttl = 300 if resolution in ['1', '5', '15', '30', '60'] else 3600
        return await self.get_cached_or_fetch(cache_key, fetch, ttl=ttl)
    
    async def get_basic_financials(self, symbol: str, metric: str = 'all') -> Optional[Dict]:
        """
        Get basic financial metrics
        """
        cache_key = f"finnhub:financials:{symbol}"
        
        async def fetch():
            params = {
                'symbol': symbol,
                'metric': metric
            }
            response = await self._make_request("stock/metric", params)
            
            if response and 'metric' in response:
                metrics = response['metric']
                return {
                    'symbol': symbol,
                    'pe_ratio': metrics.get('peBasicExclExtraTTM'),
                    'pe_annual': metrics.get('peExclExtraTTM'),
                    'pe_ttm': metrics.get('peTTM'),
                    'eps_ttm': metrics.get('epsExclExtraItemsTTM'),
                    'eps_growth_3y': metrics.get('epsGrowth3Y'),
                    'eps_growth_5y': metrics.get('epsGrowth5Y'),
                    'eps_growth_ttm': metrics.get('epsGrowthTTMYoy'),
                    'revenue_per_share_ttm': metrics.get('revenuePerShareTTM'),
                    'revenue_growth_3y': metrics.get('revenueGrowth3Y'),
                    'revenue_growth_5y': metrics.get('revenueGrowth5Y'),
                    'revenue_growth_ttm': metrics.get('revenueGrowthTTMYoy'),
                    'roe': metrics.get('roeTTM'),
                    'roa': metrics.get('roaTTM'),
                    'roi': metrics.get('roiTTM'),
                    'gross_margin': metrics.get('grossMarginTTM'),
                    'operating_margin': metrics.get('operatingMarginTTM'),
                    'net_margin': metrics.get('netProfitMarginTTM'),
                    'debt_to_equity': metrics.get('totalDebtToEquityQuarterly'),
                    'current_ratio': metrics.get('currentRatioQuarterly'),
                    'quick_ratio': metrics.get('quickRatioQuarterly'),
                    'cash_ratio': metrics.get('cashRatioQuarterly'),
                    'book_value_per_share': metrics.get('bookValuePerShareQuarterly'),
                    'dividend_yield': metrics.get('dividendYieldIndicatedAnnual'),
                    'dividend_rate': metrics.get('dividendsPerShareTTM'),
                    'payout_ratio': metrics.get('payoutRatioTTM'),
                    'beta': metrics.get('beta'),
                    '52_week_high': metrics.get('52WeekHigh'),
                    '52_week_low': metrics.get('52WeekLow'),
                    '52_week_high_date': metrics.get('52WeekHighDate'),
                    '52_week_low_date': metrics.get('52WeekLowDate'),
                    'market_cap': metrics.get('marketCapitalization'),
                    'timestamp': datetime.utcnow().isoformat()
                }
            return None
        
        # Cache for 6 hours
        return await self.get_cached_or_fetch(cache_key, fetch, ttl=21600)
    
    async def get_news(
        self,
        symbol: str = None,
        category: str = 'general',
        min_id: int = 0
    ) -> Optional[List[Dict]]:
        """
        Get company news or general market news
        """
        cache_key = f"finnhub:news:{symbol or category}:{min_id}"
        
        async def fetch():
            if symbol:
                # Company-specific news
                params = {
                    'symbol': symbol,
                    'from': (datetime.utcnow() - timedelta(days=7)).strftime('%Y-%m-%d'),
                    'to': datetime.utcnow().strftime('%Y-%m-%d')
                }
                endpoint = "company-news"
            else:
                # General market news
                params = {
                    'category': category,
                    'minId': min_id
                }
                endpoint = "news"
            
            response = await self._make_request(endpoint, params)
            
            if response:
                news_items = []
                for item in response[:50]:  # Limit to 50 most recent
                    news_items.append({
                        'id': item.get('id'),
                        'headline': item.get('headline'),
                        'summary': item.get('summary'),
                        'source': item.get('source'),
                        'url': item.get('url'),
                        'datetime': datetime.fromtimestamp(item.get('datetime', 0)).isoformat(),
                        'category': item.get('category'),
                        'related': item.get('related', '').split(',') if item.get('related') else [],
                        'image': item.get('image')
                    })
                
                return news_items
            return None
        
        # Cache for 15 minutes
        return await self.get_cached_or_fetch(cache_key, fetch, ttl=900)
    
    async def get_sentiment(self, symbol: str) -> Optional[Dict]:
        """
        Get news sentiment for a symbol
        """
        cache_key = f"finnhub:sentiment:{symbol}"
        
        async def fetch():
            params = {'symbol': symbol}
            response = await self._make_request("news-sentiment", params)
            
            if response:
                return {
                    'symbol': symbol,
                    'articles_in_last_week': response.get('buzz', {}).get('articlesInLastWeek', 0),
                    'buzz_score': response.get('buzz', {}).get('buzz', 0),
                    'weekly_average': response.get('buzz', {}).get('weeklyAverage', 0),
                    'sentiment_score': response.get('sentiment', {}).get('bearishPercent', 0),
                    'bullish_percent': response.get('sentiment', {}).get('bullishPercent', 0),
                    'bearish_percent': response.get('sentiment', {}).get('bearishPercent', 0),
                    'sector_average_bullish': response.get('sectorAverageBullishPercent', 0),
                    'sector_average_news_score': response.get('sectorAverageNewsScore', 0),
                    'company_news_score': response.get('companyNewsScore', 0),
                    'timestamp': datetime.utcnow().isoformat()
                }
            return None
        
        # Cache for 1 hour
        return await self.get_cached_or_fetch(cache_key, fetch, ttl=3600)
    
    async def get_recommendations(self, symbol: str) -> Optional[List[Dict]]:
        """
        Get analyst recommendations
        """
        cache_key = f"finnhub:recommendations:{symbol}"
        
        async def fetch():
            params = {'symbol': symbol}
            response = await self._make_request("stock/recommendation", params)
            
            if response:
                recommendations = []
                for rec in response[:12]:  # Last 12 months
                    recommendations.append({
                        'period': rec.get('period'),
                        'strong_buy': rec.get('strongBuy', 0),
                        'buy': rec.get('buy', 0),
                        'hold': rec.get('hold', 0),
                        'sell': rec.get('sell', 0),
                        'strong_sell': rec.get('strongSell', 0),
                        'total': sum([
                            rec.get('strongBuy', 0),
                            rec.get('buy', 0),
                            rec.get('hold', 0),
                            rec.get('sell', 0),
                            rec.get('strongSell', 0)
                        ])
                    })
                
                return recommendations
            return None
        
        # Cache for 24 hours
        return await self.get_cached_or_fetch(cache_key, fetch, ttl=86400)
    
    async def get_price_target(self, symbol: str) -> Optional[Dict]:
        """
        Get analyst price targets
        """
        cache_key = f"finnhub:price_target:{symbol}"
        
        async def fetch():
            params = {'symbol': symbol}
            response = await self._make_request("stock/price-target", params)
            
            if response:
                return {
                    'symbol': symbol,
                    'target_high': response.get('targetHigh', 0),
                    'target_low': response.get('targetLow', 0),
                    'target_mean': response.get('targetMean', 0),
                    'target_median': response.get('targetMedian', 0),
                    'last_updated': response.get('lastUpdated'),
                    'timestamp': datetime.utcnow().isoformat()
                }
            return None
        
        # Cache for 24 hours
        return await self.get_cached_or_fetch(cache_key, fetch, ttl=86400)
    
    async def get_technical_indicators(
        self,
        symbol: str,
        resolution: str = 'D',
        from_timestamp: int = None,
        to_timestamp: int = None,
        indicator: str = 'sma',
        indicator_fields: Dict = None
    ) -> Optional[Dict]:
        """
        Get technical indicator data
        """
        if not from_timestamp:
            from_timestamp = int((datetime.utcnow() - timedelta(days=365)).timestamp())
        if not to_timestamp:
            to_timestamp = int(datetime.utcnow().timestamp())
        
        cache_key = f"finnhub:indicator:{symbol}:{indicator}:{resolution}"
        
        async def fetch():
            params = {
                'symbol': symbol,
                'resolution': resolution,
                'from': from_timestamp,
                'to': to_timestamp,
                'indicator': indicator
            }
            
            if indicator_fields:
                params['indicatorFields'] = indicator_fields
            
            response = await self._make_request("indicator", params)
            
            if response:
                return {
                    'symbol': symbol,
                    'indicator': indicator,
                    'resolution': resolution,
                    'values': response,
                    'timestamp': datetime.utcnow().isoformat()
                }
            return None
        
        # Cache for 1 hour
        return await self.get_cached_or_fetch(cache_key, fetch, ttl=3600)
    
    async def get_batch_quotes(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Get quotes for multiple symbols efficiently
        Finnhub allows 60 calls/minute, so we can fetch many symbols
        """
        results = {}
        
        # Process in batches of 50 to stay well under rate limit
        batch_size = 50
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            
            # Fetch quotes concurrently
            tasks = [self.get_quote(symbol) for symbol in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Store results
            for symbol, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Error fetching quote for {symbol}: {result}")
                else:
                    results[symbol] = result
            
            # Small delay between batches
            if i + batch_size < len(symbols):
                await asyncio.sleep(1)
        
        return results
    
    async def get_market_status(self) -> Optional[Dict]:
        """
        Get current market status
        """
        cache_key = "finnhub:market_status"
        
        async def fetch():
            response = await self._make_request("stock/market-status", {})
            
            if response:
                return {
                    'exchange': response.get('exchange'),
                    'timezone': response.get('timezone'),
                    'is_open': response.get('isOpen'),
                    'session': response.get('session'),
                    'holiday': response.get('holiday'),
                    'timestamp': datetime.utcnow().isoformat()
                }
            return None
        
        # Cache for 5 minutes
        return await self.get_cached_or_fetch(cache_key, fetch, ttl=300)
    
    async def get_economic_calendar(self) -> Optional[List[Dict]]:
        """
        Get economic calendar events
        """
        cache_key = "finnhub:economic_calendar"
        
        async def fetch():
            response = await self._make_request("calendar/economic", {})
            
            if response and 'economicCalendar' in response:
                events = []
                for event in response['economicCalendar']:
                    events.append({
                        'country': event.get('country'),
                        'event': event.get('event'),
                        'impact': event.get('impact'),
                        'actual': event.get('actual'),
                        'estimate': event.get('estimate'),
                        'previous': event.get('prev'),
                        'time': event.get('time'),
                        'unit': event.get('unit')
                    })
                
                return events
            return None
        
        # Cache for 1 hour
        return await self.get_cached_or_fetch(cache_key, fetch, ttl=3600)