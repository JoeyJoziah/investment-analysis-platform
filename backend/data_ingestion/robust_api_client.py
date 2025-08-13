"""
Robust API Client with both async and sync fallback mechanisms
Supports aiohttp (preferred) and requests (fallback) with proper error handling
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, List, Union
from datetime import datetime, timedelta
import backoff
import json

# Try to import aiohttp (preferred)
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    aiohttp = None

# Fallback to requests
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    requests = None

from backend.utils.cost_monitor import cost_monitor
from backend.utils.cache import get_redis
from backend.config.settings import settings
from backend.utils.circuit_breaker import CircuitBreaker, CircuitBreakerError

logger = logging.getLogger(__name__)


class RobustAPIClient(ABC):
    """
    Robust API client that works with both aiohttp and requests
    Automatically falls back to sync mode if async dependencies unavailable
    """
    
    def __init__(self, provider_name: str, api_key: Optional[str] = None):
        self.provider_name = provider_name
        self.api_key = api_key or settings.get_api_key(provider_name)
        self.base_url = self._get_base_url()
        
        # Determine available HTTP client
        self.async_available = AIOHTTP_AVAILABLE
        self.sync_available = REQUESTS_AVAILABLE
        
        if not self.async_available and not self.sync_available:
            raise ImportError("Neither aiohttp nor requests is available. Install one of them.")
        
        # Async session for aiohttp
        self.session: Optional[Union['aiohttp.ClientSession', None]] = None
        
        # Timeouts
        if AIOHTTP_AVAILABLE:
            self.timeout = aiohttp.ClientTimeout(total=30)
        else:
            self.timeout = 30  # requests timeout in seconds
        
        # Initialize circuit breaker for this API client
        expected_exceptions = [Exception]
        if AIOHTTP_AVAILABLE:
            expected_exceptions.extend([aiohttp.ClientError, asyncio.TimeoutError])
        if REQUESTS_AVAILABLE:
            expected_exceptions.extend([requests.RequestException, requests.Timeout])
        
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60,
            expected_exception=tuple(expected_exceptions),
            success_threshold=2,
            name=f"{provider_name}_circuit_breaker",
            on_open=self._on_circuit_open,
            on_close=self._on_circuit_close
        )
        
        logger.info(f"RobustAPIClient initialized for {provider_name}:")
        logger.info(f"  - Async (aiohttp): {'✅' if self.async_available else '❌'}")
        logger.info(f"  - Sync (requests): {'✅' if self.sync_available else '❌'}")
    
    @abstractmethod
    def _get_base_url(self) -> str:
        """Get the base URL for the API"""
        pass
    
    async def __aenter__(self):
        """Async context manager entry"""
        if self.async_available:
            self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def _on_circuit_open(self):
        """Callback when circuit breaker opens"""
        logger.warning(f"Circuit breaker opened for {self.provider_name}")
    
    def _on_circuit_close(self):
        """Callback when circuit breaker closes"""
        logger.info(f"Circuit breaker closed for {self.provider_name}, service recovered")
    
    @backoff.on_exception(
        backoff.expo,
        (Exception,),
        max_tries=3,
        max_time=60
    )
    async def make_request_async(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        method: str = "GET"
    ) -> Optional[Dict]:
        """
        Make async HTTP request using aiohttp
        """
        if not self.async_available:
            raise RuntimeError("Async requests not available, aiohttp not installed")
        
        # Wrap the actual request in circuit breaker
        async def _do_request():
            return await self._make_request_async_internal(endpoint, params, method)
        
        try:
            return await self.circuit_breaker.call(_do_request)
        except CircuitBreakerError:
            logger.warning(f"Circuit breaker is open for {self.provider_name}, using fallback")
            # Try to get stale data from cache as fallback
            redis = await get_redis()
            stale_key = f"{self.provider_name}:{endpoint}:stale"
            cached = await redis.get(stale_key)
            if cached:
                logger.info(f"Using stale cache for {endpoint} due to circuit breaker")
                return json.loads(cached)
            return None
    
    async def _make_request_async_internal(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        method: str = "GET"
    ) -> Optional[Dict]:
        """
        Internal async method to make HTTP request with rate limiting and error handling
        """
        # Check rate limits
        if not await cost_monitor.check_api_limit(self.provider_name, endpoint):
            logger.warning(f"Rate limit reached for {self.provider_name}")
            return None
        
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        # Add API key to params if needed
        if self.api_key and params is None:
            params = {}
        if self.api_key:
            params = self._add_auth_params(params)
        
        start_time = datetime.utcnow()
        
        try:
            async with self.session.request(method, url, params=params) as response:
                response_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
                
                # Record API usage
                await cost_monitor.record_api_call(
                    provider=self.provider_name,
                    endpoint=endpoint,
                    success=response.status == 200,
                    response_time_ms=response_time
                )
                
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"API error {response.status} from {self.provider_name}: {await response.text()}")
                    return None
                    
        except Exception as e:
            logger.error(f"Async request failed for {self.provider_name}/{endpoint}: {e}")
            await cost_monitor.record_api_call(
                provider=self.provider_name,
                endpoint=endpoint,
                success=False
            )
            raise
    
    def make_request_sync(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        method: str = "GET"
    ) -> Optional[Dict]:
        """
        Make sync HTTP request using requests as fallback
        """
        if not self.sync_available:
            raise RuntimeError("Sync requests not available, requests library not installed")
        
        # Check rate limits (sync version)
        # Note: This is a simplified sync version - full implementation would need sync cost monitor
        logger.debug(f"Making sync request to {self.provider_name}/{endpoint}")
        
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        # Add API key to params if needed
        if self.api_key and params is None:
            params = {}
        if self.api_key:
            params = self._add_auth_params(params)
        
        start_time = datetime.utcnow()
        
        try:
            response = requests.request(method, url, params=params, timeout=self.timeout)
            response_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            if response.status_code == 200:
                logger.debug(f"Sync request successful: {self.provider_name}/{endpoint} ({response_time}ms)")
                return response.json()
            else:
                logger.error(f"API error {response.status_code} from {self.provider_name}: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Sync request failed for {self.provider_name}/{endpoint}: {e}")
            raise
    
    def _add_auth_params(self, params: Dict) -> Dict:
        """Add authentication parameters - override in subclasses"""
        return params
    
    async def get_cached_or_fetch(
        self,
        cache_key: str,
        fetch_func,
        ttl: int = 3600,
        force_refresh: bool = False,
        use_sync: bool = False
    ) -> Optional[Any]:
        """
        Get data from cache or fetch if not available
        Supports both async and sync fetch functions
        """
        redis = await get_redis()
        
        # Check cache first
        if not force_refresh:
            cached = await redis.get(cache_key)
            if cached:
                logger.debug(f"Cache hit for {cache_key}")
                return json.loads(cached)
        
        # Fetch fresh data
        if use_sync or not self.async_available:
            # Sync fetch
            data = fetch_func()
        else:
            # Async fetch
            data = await fetch_func()
        
        # Cache the result
        if data:
            await redis.set(cache_key, json.dumps(data), ex=ttl)
            
            # Also set a longer stale cache
            stale_key = f"{cache_key}:stale"
            await redis.set(stale_key, json.dumps(data), ex=ttl * 10)
        
        return data
    
    async def batch_request(
        self,
        items: List[str],
        fetch_func,
        batch_size: int = 10,
        delay: float = 0.1,
        use_sync: bool = False
    ) -> Dict[str, Any]:
        """
        Batch requests to avoid rate limits
        Supports both async and sync operations
        """
        results = {}
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            
            if use_sync or not self.async_available:
                # Sync processing
                for item in batch:
                    try:
                        result = fetch_func(item)
                        results[item] = result
                    except Exception as e:
                        logger.error(f"Error fetching {item}: {e}")
                        results[item] = None
            else:
                # Async processing
                tasks = [fetch_func(item) for item in batch]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Store results
                for item, result in zip(batch, batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"Error fetching {item}: {result}")
                        results[item] = None
                    else:
                        results[item] = result
            
            # Delay between batches to respect rate limits
            if i + batch_size < len(items):
                if use_sync or not self.async_available:
                    import time
                    time.sleep(delay)
                else:
                    await asyncio.sleep(delay)
        
        return results
    
    def get_client_info(self) -> Dict[str, Any]:
        """Get information about available HTTP clients"""
        return {
            'provider': self.provider_name,
            'async_available': self.async_available,
            'sync_available': self.sync_available,
            'preferred_mode': 'async' if self.async_available else 'sync',
            'aiohttp_version': getattr(aiohttp, '__version__', None) if AIOHTTP_AVAILABLE else None,
            'requests_version': getattr(requests, '__version__', None) if REQUESTS_AVAILABLE else None,
            'has_api_key': bool(self.api_key)
        }


class RobustFinnhubClient(RobustAPIClient):
    """Robust Finnhub client with fallback mechanisms"""
    
    def _get_base_url(self) -> str:
        return "https://finnhub.io/api/v1"
    
    def _add_auth_params(self, params: Dict) -> Dict:
        """Add API key to parameters"""
        params['token'] = self.api_key
        return params
    
    async def get_quote_async(self, symbol: str) -> Optional[Dict]:
        """Get quote using async method"""
        params = {'symbol': symbol}
        response = await self.make_request_async("quote", params)
        
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
    
    def get_quote_sync(self, symbol: str) -> Optional[Dict]:
        """Get quote using sync method as fallback"""
        params = {'symbol': symbol}
        response = self.make_request_sync("quote", params)
        
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
    
    async def get_quote(self, symbol: str) -> Optional[Dict]:
        """Get quote with automatic fallback"""
        cache_key = f"finnhub:quote:{symbol}"
        
        if self.async_available:
            fetch_func = lambda: self.get_quote_async(symbol)
            return await self.get_cached_or_fetch(cache_key, fetch_func, ttl=60)
        else:
            fetch_func = lambda: self.get_quote_sync(symbol)
            return await self.get_cached_or_fetch(cache_key, fetch_func, ttl=60, use_sync=True)


class RobustAlphaVantageClient(RobustAPIClient):
    """Robust Alpha Vantage client with fallback mechanisms"""
    
    def _get_base_url(self) -> str:
        return "https://www.alphavantage.co/query"
    
    def _add_auth_params(self, params: Dict) -> Dict:
        """Add API key to parameters"""
        params['apikey'] = self.api_key
        return params
    
    async def get_quote_async(self, symbol: str) -> Optional[Dict]:
        """Get quote using async method"""
        params = {
            'function': 'GLOBAL_QUOTE',
            'symbol': symbol
        }
        
        response = await self.make_request_async("", params)
        if response and 'Global Quote' in response:
            quote_data = response['Global Quote']
            return {
                'symbol': symbol,
                'price': float(quote_data.get('05. price', 0)),
                'change': float(quote_data.get('09. change', 0)),
                'change_percent': quote_data.get('10. change percent', '0%'),
                'timestamp': datetime.utcnow().isoformat()
            }
        return None
    
    def get_quote_sync(self, symbol: str) -> Optional[Dict]:
        """Get quote using sync method as fallback"""
        params = {
            'function': 'GLOBAL_QUOTE',
            'symbol': symbol
        }
        
        response = self.make_request_sync("", params)
        if response and 'Global Quote' in response:
            quote_data = response['Global Quote']
            return {
                'symbol': symbol,
                'price': float(quote_data.get('05. price', 0)),
                'change': float(quote_data.get('09. change', 0)),
                'change_percent': quote_data.get('10. change percent', '0%'),
                'timestamp': datetime.utcnow().isoformat()
            }
        return None
    
    async def get_quote(self, symbol: str) -> Optional[Dict]:
        """Get quote with automatic fallback"""
        cache_key = f"av:quote:{symbol}"
        
        if self.async_available:
            fetch_func = lambda: self.get_quote_async(symbol)
            return await self.get_cached_or_fetch(cache_key, fetch_func, ttl=300)
        else:
            fetch_func = lambda: self.get_quote_sync(symbol)
            return await self.get_cached_or_fetch(cache_key, fetch_func, ttl=300, use_sync=True)