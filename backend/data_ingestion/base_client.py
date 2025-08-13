"""
Base API Client with rate limiting and cost monitoring
"""

import asyncio
import aiohttp
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, List
import logging
from datetime import datetime, timedelta
import backoff
from aiohttp import ClientTimeout
import json

from backend.utils.cost_monitor import cost_monitor
from backend.utils.cache import get_redis
from backend.config.settings import settings
from backend.utils.circuit_breaker import CircuitBreaker, CircuitBreakerError

logger = logging.getLogger(__name__)


class BaseAPIClient(ABC):
    """
    Base class for all API clients with built-in rate limiting and caching
    """
    
    def __init__(self, provider_name: str, api_key: Optional[str] = None):
        self.provider_name = provider_name
        self.api_key = api_key or settings.get_api_key(provider_name)
        self.base_url = self._get_base_url()
        self.session: Optional[aiohttp.ClientSession] = None
        self.timeout = ClientTimeout(total=30)
        
        # Initialize circuit breaker for this API client
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60,
            expected_exception=(aiohttp.ClientError, asyncio.TimeoutError, Exception),
            success_threshold=2,
            name=f"{provider_name}_circuit_breaker",
            on_open=self._on_circuit_open,
            on_close=self._on_circuit_close
        )
        
    @abstractmethod
    def _get_base_url(self) -> str:
        """Get the base URL for the API"""
        pass
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def _on_circuit_open(self):
        """Callback when circuit breaker opens"""
        logger.warning(f"Circuit breaker opened for {self.provider_name}")
        # Could send alerts or switch to fallback provider here
    
    def _on_circuit_close(self):
        """Callback when circuit breaker closes"""
        logger.info(f"Circuit breaker closed for {self.provider_name}, service recovered")
    
    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError),
        max_tries=3,
        max_time=60
    )
    async def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        method: str = "GET"
    ) -> Optional[Dict]:
        """
        Make HTTP request with circuit breaker, rate limiting and error handling
        """
        # Wrap the actual request in circuit breaker
        async def _do_request():
            return await self._make_request_internal(endpoint, params, method)
        
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
    
    async def _make_request_internal(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        method: str = "GET"
    ) -> Optional[Dict]:
        """
        Internal method to make HTTP request with rate limiting and error handling
        """
        # Check rate limits
        if not await cost_monitor.check_api_limit(self.provider_name, endpoint):
            logger.warning(f"Rate limit reached for {self.provider_name}")
            return None
        
        url = f"{self.base_url}/{endpoint}"
        
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
            logger.error(f"Request failed for {self.provider_name}/{endpoint}: {e}")
            await cost_monitor.record_api_call(
                provider=self.provider_name,
                endpoint=endpoint,
                success=False
            )
            raise
    
    def _add_auth_params(self, params: Dict) -> Dict:
        """Add authentication parameters - override in subclasses"""
        return params
    
    async def get_cached_or_fetch(
        self,
        cache_key: str,
        fetch_func,
        ttl: int = 3600,
        force_refresh: bool = False
    ) -> Optional[Any]:
        """
        Get data from cache or fetch if not available
        """
        redis = await get_redis()
        
        # Check cache first
        if not force_refresh:
            cached = await redis.get(cache_key)
            if cached:
                logger.debug(f"Cache hit for {cache_key}")
                return json.loads(cached)
        
        # Fetch fresh data
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
        delay: float = 0.1
    ) -> Dict[str, Any]:
        """
        Batch requests to avoid rate limits
        """
        results = {}
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            
            # Process batch concurrently
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
                await asyncio.sleep(delay)
        
        return results