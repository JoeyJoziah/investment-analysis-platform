"""
API Response Caching Decorators and Middleware

This module provides decorators and middleware for caching API responses,
with intelligent cache invalidation and cost optimization features.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Union
import hashlib
import inspect

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from backend.utils.comprehensive_cache import get_cache_manager
from backend.utils.intelligent_cache_policies import get_policy_manager
from backend.config.settings import settings

logger = logging.getLogger(__name__)


def generate_cache_key(
    func_name: str, 
    args: tuple = (), 
    kwargs: dict = None, 
    request: Optional[Request] = None,
    include_user: bool = False
) -> str:
    """
    Generate a consistent cache key for API responses
    """
    key_parts = [func_name]
    
    # Add function arguments
    if args:
        key_parts.extend([str(arg) for arg in args])
    
    if kwargs:
        # Sort kwargs for consistent key generation
        sorted_kwargs = sorted(kwargs.items())
        key_parts.extend([f"{k}={v}" for k, v in sorted_kwargs])
    
    # Add request parameters if provided
    if request:
        # Add query parameters
        if request.query_params:
            sorted_params = sorted(request.query_params.items())
            key_parts.extend([f"q_{k}={v}" for k, v in sorted_params])
        
        # Add user ID if required
        if include_user and hasattr(request, 'state') and hasattr(request.state, 'user_id'):
            key_parts.append(f"user={request.state.user_id}")
    
    # Create hash for very long keys
    key_str = ":".join(key_parts)
    if len(key_str) > 200:
        key_hash = hashlib.md5(key_str.encode()).hexdigest()
        return f"{func_name}:hash_{key_hash}"
    
    return key_str


def api_cache(
    data_type: str,
    ttl_override: Optional[Dict[str, int]] = None,
    include_user_context: bool = False,
    cache_key_func: Optional[Callable] = None,
    invalidation_tags: Optional[List[str]] = None,
    cost_tracking: bool = True
):
    """
    Decorator for caching API endpoint responses with intelligent policies
    
    Args:
        data_type: Type of data being cached (maps to cache policies)
        ttl_override: Custom TTL values for L1/L2/L3 cache layers
        include_user_context: Whether to include user ID in cache key
        cache_key_func: Custom function to generate cache key
        invalidation_tags: Tags for selective cache invalidation
        cost_tracking: Whether to track API costs for this endpoint
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache_manager = await get_cache_manager()
            policy_manager = get_policy_manager()
            
            # Extract request object if available
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            
            # Generate cache key
            if cache_key_func:
                cache_key = cache_key_func(*args, **kwargs)
            else:
                cache_key = generate_cache_key(
                    func.__name__, 
                    args, 
                    kwargs, 
                    request, 
                    include_user_context
                )
            
            # Get cache policy and adjust TTL for current conditions
            policy = policy_manager.get_policy(data_type)
            ttl_config = ttl_override or policy_manager.adjust_ttl_for_market_hours(policy)
            
            # Track access pattern
            if request:
                identifier = cache_key
                if include_user_context and hasattr(request, 'state'):
                    identifier = f"{cache_key}:user_{getattr(request.state, 'user_id', 'anonymous')}"
                
                policy_manager.track_access(data_type, identifier)
            
            # Try to get from cache with fallback
            async def compute_result():
                if cost_tracking:
                    # Track API usage for cost optimization
                    api_provider = _detect_api_provider(func)
                    if api_provider:
                        policy_manager.track_api_usage(api_provider, data_type)
                
                return await func(*args, **kwargs)
            
            result, source = await cache_manager.get(
                data_type=data_type,
                identifier=cache_key,
                fallback_func=compute_result
            )
            
            # Add cache headers to response if it's a FastAPI response
            if isinstance(result, (JSONResponse, Response)):
                result.headers["X-Cache-Status"] = source
                result.headers["X-Cache-Key"] = cache_key[:100]  # Truncate for header size
                
                if source != 'miss':
                    # Add cache hit timestamp
                    result.headers["X-Cache-Time"] = datetime.utcnow().isoformat()
                    
                    # Add TTL information
                    if source == 'l1':
                        result.headers["X-Cache-TTL"] = str(ttl_config['l1'])
                    elif source == 'l2':
                        result.headers["X-Cache-TTL"] = str(ttl_config['l2'])
                    elif source == 'l3':
                        result.headers["X-Cache-TTL"] = str(ttl_config['l3'])
            
            return result
        
        # Add metadata for monitoring
        wrapper._cache_enabled = True
        wrapper._data_type = data_type
        wrapper._invalidation_tags = invalidation_tags or []
        
        return wrapper
    
    return decorator


def _detect_api_provider(func: Callable) -> Optional[str]:
    """Detect which API provider is being used based on function context"""
    func_module = getattr(func, '__module__', '')
    func_name = getattr(func, '__name__', '')
    
    if 'alpha_vantage' in func_module.lower() or 'alpha_vantage' in func_name.lower():
        return 'alpha_vantage'
    elif 'finnhub' in func_module.lower() or 'finnhub' in func_name.lower():
        return 'finnhub'
    elif 'polygon' in func_module.lower() or 'polygon' in func_name.lower():
        return 'polygon'
    elif 'news' in func_module.lower() or 'news' in func_name.lower():
        return 'newsapi'
    
    return None


class CacheControlMiddleware(BaseHTTPMiddleware):
    """
    Middleware for adding cache control headers and handling cache-related logic
    """
    
    def __init__(
        self,
        app: ASGIApp,
        default_cache_control: str = "public, max-age=300",
        cache_excluded_paths: Optional[List[str]] = None
    ):
        super().__init__(app)
        self.default_cache_control = default_cache_control
        self.cache_excluded_paths = cache_excluded_paths or [
            "/api/auth/",
            "/api/admin/",
            "/api/ws/"
        ]
    
    async def dispatch(self, request: Request, call_next):
        # Check if path should be excluded from caching
        path = request.url.path
        excluded = any(excluded_path in path for excluded_path in self.cache_excluded_paths)
        
        # Process request
        response = await call_next(request)
        
        if not excluded and response.status_code == 200:
            # Add cache control headers
            if "Cache-Control" not in response.headers:
                response.headers["Cache-Control"] = self.default_cache_control
            
            # Add cache-friendly headers
            response.headers["Vary"] = "Accept-Encoding, Authorization"
            
            # Add ETag if not present (for conditional requests)
            if "ETag" not in response.headers and hasattr(response, 'body'):
                try:
                    body_hash = hashlib.md5(response.body).hexdigest()[:16]
                    response.headers["ETag"] = f'"{body_hash}"'
                except:
                    pass
        
        return response


class CacheInvalidationManager:
    """
    Manager for intelligent cache invalidation based on data updates
    """
    
    def __init__(self):
        self.invalidation_rules: Dict[str, List[str]] = {}
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default invalidation rules"""
        
        # When stock price updates, invalidate related cache entries
        self.invalidation_rules["stock_price_update"] = [
            "real_time_quote:*",
            "technical_indicators:*",
            "analysis_result:*"
        ]
        
        # When company fundamentals update
        self.invalidation_rules["company_fundamentals_update"] = [
            "company_overview:*",
            "analysis_result:*"
        ]
        
        # When market hours change
        self.invalidation_rules["market_status_change"] = [
            "real_time_quote:*"
        ]
        
        # When news/sentiment updates
        self.invalidation_rules["news_update"] = [
            "news_sentiment:*",
            "analysis_result:*"
        ]
    
    async def invalidate_by_event(self, event_type: str, context: Dict[str, Any] = None):
        """
        Invalidate cache entries based on an event
        
        Args:
            event_type: Type of event that occurred
            context: Additional context (e.g., {'symbol': 'AAPL'})
        """
        if event_type not in self.invalidation_rules:
            logger.debug(f"No invalidation rules for event: {event_type}")
            return
        
        cache_manager = await get_cache_manager()
        patterns = self.invalidation_rules[event_type]
        
        invalidated_count = 0
        for pattern in patterns:
            # Replace context variables in pattern
            if context:
                for key, value in context.items():
                    pattern = pattern.replace(f"{{{key}}}", str(value))
            
            try:
                await cache_manager.invalidate_pattern(pattern)
                invalidated_count += 1
            except Exception as e:
                logger.error(f"Failed to invalidate pattern {pattern}: {e}")
        
        logger.info(f"Invalidated {invalidated_count} cache patterns for event: {event_type}")
    
    async def invalidate_by_symbol(self, symbol: str):
        """Invalidate all cache entries for a specific stock symbol"""
        patterns = [
            f"*:{symbol}:*",
            f"*:{symbol}",
            f"api:resp:*{symbol}*"
        ]
        
        cache_manager = await get_cache_manager()
        for pattern in patterns:
            try:
                await cache_manager.invalidate_pattern(pattern)
            except Exception as e:
                logger.error(f"Failed to invalidate symbol cache {pattern}: {e}")
        
        logger.info(f"Invalidated cache entries for symbol: {symbol}")
    
    async def schedule_periodic_invalidation(self):
        """Schedule periodic cache invalidation for stale data"""
        while True:
            try:
                # Every 4 hours, clean up potentially stale analysis results
                cache_manager = await get_cache_manager()
                await cache_manager.invalidate_pattern("analysis_result:*")
                logger.info("Performed periodic cache invalidation")
                
                # Sleep for 4 hours
                await asyncio.sleep(14400)
                
            except Exception as e:
                logger.error(f"Error in periodic cache invalidation: {e}")
                await asyncio.sleep(14400)


# Global cache invalidation manager
_invalidation_manager: Optional[CacheInvalidationManager] = None


def get_invalidation_manager() -> CacheInvalidationManager:
    """Get global cache invalidation manager"""
    global _invalidation_manager
    
    if _invalidation_manager is None:
        _invalidation_manager = CacheInvalidationManager()
    
    return _invalidation_manager


# Specialized decorators for common use cases

def cache_stock_data(
    ttl_hours: int = 1,
    include_user: bool = False
):
    """Decorator specifically for stock data endpoints"""
    return api_cache(
        data_type="real_time_quote",
        ttl_override={
            'l1': ttl_hours * 300,   # 5 min * hours
            'l2': ttl_hours * 1800,  # 30 min * hours  
            'l3': ttl_hours * 3600   # 1 hour * hours
        },
        include_user_context=include_user,
        invalidation_tags=["stock_data"],
        cost_tracking=True
    )


def cache_analysis_result(
    ttl_hours: int = 4,
    include_user: bool = True
):
    """Decorator specifically for analysis result endpoints"""
    return api_cache(
        data_type="analysis_result",
        ttl_override={
            'l1': ttl_hours * 450,   # 7.5 min * hours
            'l2': ttl_hours * 1800,  # 30 min * hours
            'l3': ttl_hours * 3600   # 1 hour * hours  
        },
        include_user_context=include_user,
        invalidation_tags=["analysis"],
        cost_tracking=False  # Analysis results don't consume external APIs
    )


def cache_portfolio_data(
    ttl_minutes: int = 15,
    include_user: bool = True
):
    """Decorator specifically for portfolio-related endpoints"""
    return api_cache(
        data_type="user_portfolio",
        ttl_override={
            'l1': ttl_minutes * 60,    # Convert to seconds
            'l2': ttl_minutes * 120,   # 2x in L2
            'l3': ttl_minutes * 240    # 4x in L3
        },
        include_user_context=include_user,
        invalidation_tags=["portfolio", "user_data"],
        cost_tracking=False
    )