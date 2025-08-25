"""
Database Query Caching Layer

This module provides intelligent caching for database queries to reduce
database load and improve response times.
"""

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import sqlalchemy
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import Select
from sqlalchemy.engine import Result

from backend.utils.comprehensive_cache import get_cache_manager
from backend.utils.intelligent_cache_policies import get_policy_manager
from backend.config.database import get_async_db_session

logger = logging.getLogger(__name__)


class QueryCacheManager:
    """
    Manager for database query caching with intelligent invalidation
    """
    
    def __init__(self):
        self.query_stats: Dict[str, Dict] = {}
        self.table_dependencies: Dict[str, List[str]] = {}
        self._setup_table_dependencies()
    
    def _setup_table_dependencies(self):
        """Setup table dependency mapping for cache invalidation"""
        self.table_dependencies = {
            # Stock-related tables
            'stocks': ['stock_prices', 'technical_indicators', 'recommendations', 'analysis_results'],
            'stock_prices': ['technical_indicators', 'analysis_results'],
            'technical_indicators': ['analysis_results'],
            
            # User-related tables
            'users': ['portfolios', 'user_preferences', 'alerts'],
            'portfolios': ['portfolio_stocks', 'portfolio_history'],
            
            # Market data tables
            'market_data': ['analysis_results', 'recommendations'],
            'news': ['sentiment_analysis', 'analysis_results'],
            
            # Cache table (self-maintaining)
            'cache_storage': []
        }
    
    def generate_query_key(
        self, 
        query: Union[str, Select], 
        params: Optional[Dict] = None,
        table_hint: Optional[str] = None
    ) -> str:
        """
        Generate a consistent cache key for database queries
        """
        # Convert query to string if it's a SQLAlchemy object
        if hasattr(query, 'compile'):
            query_str = str(query.compile(compile_kwargs={"literal_binds": True}))
        else:
            query_str = str(query)
        
        # Normalize query string (remove extra whitespace, case-insensitive)
        normalized_query = ' '.join(query_str.split()).lower()
        
        # Add parameters to the key
        key_components = [normalized_query]
        if params:
            sorted_params = sorted(params.items())
            key_components.append(json.dumps(sorted_params, sort_keys=True))
        
        # Add table hint for better cache organization
        if table_hint:
            key_components.append(f"table:{table_hint}")
        
        # Create hash for long keys
        full_key = "|".join(key_components)
        if len(full_key) > 200:
            key_hash = hashlib.sha256(full_key.encode()).hexdigest()[:16]
            table_prefix = table_hint or "query"
            return f"db:{table_prefix}:hash_{key_hash}"
        
        # Return readable key for shorter queries
        table_prefix = table_hint or "query"
        safe_key = full_key.replace(" ", "_").replace("|", "_")[:100]
        return f"db:{table_prefix}:{safe_key}"
    
    async def get_cached_query(
        self,
        query: Union[str, Select],
        params: Optional[Dict] = None,
        table_hint: Optional[str] = None,
        ttl_override: Optional[Dict[str, int]] = None
    ) -> Tuple[Optional[List[Dict]], bool]:
        """
        Get query results from cache
        
        Returns:
            (results, cache_hit) tuple
        """
        cache_key = self.generate_query_key(query, params, table_hint)
        cache_manager = await get_cache_manager()
        
        result, source = await cache_manager.get(
            data_type="db_query",
            identifier=cache_key
        )
        
        # Update query statistics
        self._update_query_stats(cache_key, source != 'miss')
        
        if source != 'miss':
            logger.debug(f"Database query cache hit ({source}): {cache_key[:50]}...")
            return result, True
        
        return None, False
    
    async def cache_query_result(
        self,
        query: Union[str, Select],
        result: List[Dict],
        params: Optional[Dict] = None,
        table_hint: Optional[str] = None,
        ttl_override: Optional[Dict[str, int]] = None
    ):
        """Cache query results"""
        cache_key = self.generate_query_key(query, params, table_hint)
        cache_manager = await get_cache_manager()
        
        # Use intelligent TTL based on table volatility
        policy_manager = get_policy_manager()
        data_type = f"db_query_{table_hint}" if table_hint else "db_query"
        
        # Get or create policy for this query type
        if data_type not in policy_manager.policies:
            # Create dynamic policy based on table characteristics
            ttl_config = self._get_table_ttl_config(table_hint)
        else:
            policy = policy_manager.get_policy(data_type)
            ttl_config = policy_manager.adjust_ttl_for_market_hours(policy)
        
        if ttl_override:
            ttl_config.update(ttl_override)
        
        await cache_manager.set(
            data_type="db_query",
            identifier=cache_key,
            data=result,
            custom_ttl=ttl_config
        )
        
        logger.debug(f"Cached database query result: {cache_key[:50]}...")
    
    async def invalidate_table_cache(self, table_name: str):
        """Invalidate all cached queries related to a table"""
        cache_manager = await get_cache_manager()
        
        # Invalidate direct table queries
        await cache_manager.invalidate_pattern(f"db:{table_name}:*")
        
        # Invalidate dependent table queries
        dependent_tables = self.table_dependencies.get(table_name, [])
        for dep_table in dependent_tables:
            await cache_manager.invalidate_pattern(f"db:{dep_table}:*")
        
        logger.info(f"Invalidated cache for table: {table_name} and {len(dependent_tables)} dependent tables")
    
    def _get_table_ttl_config(self, table_hint: Optional[str]) -> Dict[str, int]:
        """Get TTL configuration based on table characteristics"""
        if not table_hint:
            return {'l1': 300, 'l2': 1800, 'l3': 7200}  # Default 5min/30min/2hr
        
        # High-frequency tables (shorter TTL)
        if table_hint in ['stock_prices', 'real_time_quotes', 'market_data']:
            return {'l1': 60, 'l2': 300, 'l3': 1800}  # 1min/5min/30min
        
        # Medium-frequency tables
        elif table_hint in ['technical_indicators', 'news', 'sentiment_analysis']:
            return {'l1': 300, 'l2': 1800, 'l3': 7200}  # 5min/30min/2hr
        
        # Low-frequency tables (longer TTL)
        elif table_hint in ['stocks', 'companies', 'sectors', 'industries']:
            return {'l1': 1800, 'l2': 7200, 'l3': 86400}  # 30min/2hr/24hr
        
        # User-specific tables (medium TTL, but user-scoped)
        elif table_hint in ['portfolios', 'user_preferences', 'alerts']:
            return {'l1': 300, 'l2': 1800, 'l3': 7200}  # 5min/30min/2hr
        
        # Analysis results (medium-long TTL)
        elif table_hint in ['analysis_results', 'recommendations']:
            return {'l1': 900, 'l2': 3600, 'l3': 14400}  # 15min/1hr/4hr
        
        # Default for unknown tables
        return {'l1': 600, 'l2': 3600, 'l3': 14400}  # 10min/1hr/4hr
    
    def _update_query_stats(self, cache_key: str, cache_hit: bool):
        """Update query statistics for monitoring"""
        if cache_key not in self.query_stats:
            self.query_stats[cache_key] = {
                'total_requests': 0,
                'cache_hits': 0,
                'last_accessed': None,
                'created_at': datetime.utcnow()
            }
        
        stats = self.query_stats[cache_key]
        stats['total_requests'] += 1
        stats['last_accessed'] = datetime.utcnow()
        
        if cache_hit:
            stats['cache_hits'] += 1
    
    def get_query_statistics(self) -> Dict[str, Any]:
        """Get query cache statistics"""
        total_requests = sum(stats['total_requests'] for stats in self.query_stats.values())
        total_hits = sum(stats['cache_hits'] for stats in self.query_stats.values())
        
        hit_rate = (total_hits / total_requests) if total_requests > 0 else 0
        
        # Find most accessed queries
        top_queries = sorted(
            self.query_stats.items(),
            key=lambda x: x[1]['total_requests'],
            reverse=True
        )[:10]
        
        return {
            'total_queries': len(self.query_stats),
            'total_requests': total_requests,
            'total_hits': total_hits,
            'hit_rate': hit_rate,
            'top_queries': [
                {
                    'query_key': key[:50] + '...' if len(key) > 50 else key,
                    'requests': stats['total_requests'],
                    'hits': stats['cache_hits'],
                    'hit_rate': stats['cache_hits'] / stats['total_requests'] if stats['total_requests'] > 0 else 0
                }
                for key, stats in top_queries
            ]
        }


# Global query cache manager
_query_cache_manager: Optional[QueryCacheManager] = None


def get_query_cache_manager() -> QueryCacheManager:
    """Get global query cache manager"""
    global _query_cache_manager
    
    if _query_cache_manager is None:
        _query_cache_manager = QueryCacheManager()
    
    return _query_cache_manager


def cached_query(
    table_hint: Optional[str] = None,
    ttl_override: Optional[Dict[str, int]] = None,
    bypass_cache: bool = False
):
    """
    Decorator for caching database query results
    
    Args:
        table_hint: Primary table being queried (for better cache organization)
        ttl_override: Custom TTL values for cache layers
        bypass_cache: Skip caching (useful for development/debugging)
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if bypass_cache:
                return await func(*args, **kwargs)
            
            query_cache = get_query_cache_manager()
            
            # Try to extract query and parameters from function arguments
            query = None
            params = None
            
            # Look for query in arguments
            for arg in args:
                if isinstance(arg, (str, Select)) or hasattr(arg, 'compile'):
                    query = arg
                    break
            
            # Look for params in kwargs
            if 'params' in kwargs:
                params = kwargs['params']
            
            if not query:
                # If no query found, just execute the function
                logger.debug(f"No query found in {func.__name__}, skipping cache")
                return await func(*args, **kwargs)
            
            # Try to get from cache
            cached_result, cache_hit = await query_cache.get_cached_query(
                query, params, table_hint, ttl_override
            )
            
            if cache_hit:
                return cached_result
            
            # Execute query and cache result
            result = await func(*args, **kwargs)
            
            # Convert result to cacheable format
            if hasattr(result, 'fetchall'):
                # SQLAlchemy Result object
                rows = result.fetchall()
                cacheable_result = [dict(row._mapping) for row in rows]
            elif isinstance(result, list):
                # Already a list of dicts
                cacheable_result = result
            else:
                # Try to convert to list of dicts
                try:
                    cacheable_result = [dict(item) for item in result] if result else []
                except:
                    # If conversion fails, don't cache
                    logger.debug(f"Cannot convert result to cacheable format in {func.__name__}")
                    return result
            
            # Cache the result
            await query_cache.cache_query_result(
                query, cacheable_result, params, table_hint, ttl_override
            )
            
            return result
        
        return wrapper
    return decorator


class CachedDatabase:
    """
    Database wrapper with built-in query caching
    """
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.query_cache = get_query_cache_manager()
    
    async def execute_cached(
        self,
        query: Union[str, Select],
        params: Optional[Dict] = None,
        table_hint: Optional[str] = None,
        ttl_override: Optional[Dict[str, int]] = None
    ) -> List[Dict]:
        """Execute query with caching"""
        
        # Try cache first
        cached_result, cache_hit = await self.query_cache.get_cached_query(
            query, params, table_hint, ttl_override
        )
        
        if cache_hit:
            return cached_result
        
        # Execute query
        if isinstance(query, str):
            result = await self.session.execute(text(query), params or {})
        else:
            result = await self.session.execute(query, params or {})
        
        # Convert to list of dicts
        rows = result.fetchall()
        result_list = [dict(row._mapping) for row in rows]
        
        # Cache result
        await self.query_cache.cache_query_result(
            query, result_list, params, table_hint, ttl_override
        )
        
        return result_list
    
    async def invalidate_cache(self, table_name: str):
        """Invalidate cache for a specific table"""
        await self.query_cache.invalidate_table_cache(table_name)


# Common cached query functions for frequent operations

@cached_query(table_hint="stocks")
async def get_stock_by_symbol(symbol: str) -> Optional[Dict]:
    """Get stock information by symbol with caching"""
    async with get_async_db_session() as db:
        result = await db.execute(
            text("SELECT * FROM stocks WHERE symbol = :symbol LIMIT 1"),
            {"symbol": symbol}
        )
        row = result.fetchone()
        return dict(row._mapping) if row else None


@cached_query(table_hint="stock_prices", ttl_override={'l1': 300, 'l2': 1800, 'l3': 7200})
async def get_latest_stock_price(symbol: str) -> Optional[Dict]:
    """Get latest stock price with aggressive caching"""
    async with get_async_db_session() as db:
        result = await db.execute(
            text("""
                SELECT * FROM stock_prices 
                WHERE symbol = :symbol 
                ORDER BY date DESC 
                LIMIT 1
            """),
            {"symbol": symbol}
        )
        row = result.fetchone()
        return dict(row._mapping) if row else None


@cached_query(table_hint="technical_indicators")
async def get_technical_indicators(symbol: str, indicator_type: str, limit: int = 50) -> List[Dict]:
    """Get technical indicators with caching"""
    async with get_async_db_session() as db:
        result = await db.execute(
            text("""
                SELECT * FROM technical_indicators 
                WHERE symbol = :symbol AND indicator_type = :indicator_type
                ORDER BY date DESC 
                LIMIT :limit
            """),
            {"symbol": symbol, "indicator_type": indicator_type, "limit": limit}
        )
        rows = result.fetchall()
        return [dict(row._mapping) for row in rows]


@cached_query(table_hint="recommendations")
async def get_active_recommendations(limit: int = 100) -> List[Dict]:
    """Get active recommendations with caching"""
    async with get_async_db_session() as db:
        result = await db.execute(
            text("""
                SELECT r.*, s.company_name, s.sector
                FROM recommendations r
                JOIN stocks s ON r.symbol = s.symbol
                WHERE r.is_active = true
                ORDER BY r.confidence_score DESC, r.created_at DESC
                LIMIT :limit
            """),
            {"limit": limit}
        )
        rows = result.fetchall()
        return [dict(row._mapping) for row in rows]


# Database event listeners for cache invalidation

async def setup_cache_invalidation_triggers():
    """
    Setup database triggers or event listeners for automatic cache invalidation
    This would be called during application startup
    """
    try:
        query_cache = get_query_cache_manager()
        
        # For now, we'll set up a periodic cleanup
        # In production, you might want to use database triggers or change data capture
        
        async def periodic_invalidation():
            while True:
                try:
                    # Invalidate volatile data caches every 5 minutes
                    await query_cache.invalidate_table_cache('stock_prices')
                    await query_cache.invalidate_table_cache('real_time_quotes')
                    
                    # Sleep for 5 minutes
                    await asyncio.sleep(300)
                    
                except Exception as e:
                    logger.error(f"Error in periodic cache invalidation: {e}")
                    await asyncio.sleep(300)
        
        # Start the periodic invalidation task
        asyncio.create_task(periodic_invalidation())
        
        logger.info("Database cache invalidation system initialized")
        
    except Exception as e:
        logger.error(f"Failed to setup cache invalidation triggers: {e}")
        raise