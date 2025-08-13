"""
Unified Integration Layer for All Components
Orchestrates rate limiting, caching, parallel processing, and cost monitoring.
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
import logging
from enum import Enum
import hashlib
import json

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session
from sqlalchemy import select, func, and_, or_

from backend.utils.distributed_rate_limiter import APIRateLimiter, RateLimitExceeded
from backend.utils.query_cache import QueryResultCache, CacheStrategy
from backend.utils.parallel_processor import (
    ParallelAPIProcessor, 
    APITask, 
    Priority,
    ProcessingStrategy,
    TaskResult
)
from backend.utils.persistent_cost_monitor import PersistentCostMonitor
from backend.utils.cost_monitor import CostMonitor, SmartDataFetcher
from backend.config.settings import settings
from backend.utils.async_database import get_async_session, async_db_manager

logger = logging.getLogger(__name__)


class StockTier(Enum):
    """Stock priority tiers for efficient processing."""
    CRITICAL = 1    # S&P 500, high volume - Updated hourly
    HIGH = 2        # Mid-cap active - Updated every 4 hours  
    MEDIUM = 3      # Small-cap - Updated every 8 hours
    LOW = 4         # Inactive - Daily updates
    MINIMAL = 5     # Delisted/low activity - Weekly updates


class UnifiedDataIngestion:
    """
    Unified ingestion system orchestrating all Week 2-3 components.
    Ensures proper integration with existing codebase.
    """
    
    def __init__(self, db_session: Optional[AsyncSession] = None):
        """
        Initialize unified ingestion with all components.
        
        Args:
            db_session: Async database session for persistence
        """
        self.db_session = db_session  # Will use get_async_session() when needed
        
        # Initialize all components
        self.rate_limiter = APIRateLimiter()
        self.cache = QueryResultCache(
            strategy=CacheStrategy.TTL,
            max_size=10000,
            default_ttl=300
        )
        self.processor = ParallelAPIProcessor(
            max_concurrent_calls=50,
            strategy=ProcessingStrategy.ADAPTIVE,
            enable_caching=True,
            enable_rate_limiting=True
        )
        self.cost_monitor = PersistentCostMonitor(self.db_session)
        self.smart_fetcher = SmartDataFetcher(self.cost_monitor)
        
        # Stock tiering configuration
        self.stock_tiers: Dict[StockTier, Set[str]] = {
            StockTier.CRITICAL: set(),  # Will be populated
            StockTier.HIGH: set(),
            StockTier.MEDIUM: set(),
            StockTier.LOW: set(),
            StockTier.MINIMAL: set()
        }
        
        # Update frequencies (in seconds)
        self.tier_update_frequencies = {
            StockTier.CRITICAL: 3600,      # 1 hour
            StockTier.HIGH: 14400,          # 4 hours
            StockTier.MEDIUM: 28800,        # 8 hours
            StockTier.LOW: 86400,           # 24 hours
            StockTier.MINIMAL: 604800       # 7 days
        }
        
        # Provider allocation by tier
        self.tier_providers = {
            StockTier.CRITICAL: ['finnhub'],  # Most generous free tier
            StockTier.HIGH: ['alpha_vantage'],
            StockTier.MEDIUM: ['polygon'],
            StockTier.LOW: ['cache_only'],
            StockTier.MINIMAL: ['cache_only']
        }
        
        self._initialized = False
        self._last_tier_update = {}
    
    async def initialize(self) -> None:
        """Initialize all components and load stock tiers."""
        if self._initialized:
            return
        
        try:
            # Initialize components
            await self.rate_limiter.initialize()
            await self.cache.initialize()
            await self.processor.initialize()
            await self.cost_monitor.initialize()
            
            # Load stock tiers
            await self._load_stock_tiers()
            
            self._initialized = True
            logger.info("Unified data ingestion initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize unified ingestion: {e}")
            raise
    
    async def _load_stock_tiers(self) -> None:
        """Load stocks into appropriate tiers based on volume and market cap."""
        try:
            # Import Stock model
            from backend.models.unified_models import Stock
            
            async with get_async_session() as session:
                # Get all stocks ordered by market cap and volume
                result = await session.execute(
                    select(Stock)
                    .order_by(
                        Stock.market_cap.desc().nullslast(),
                        Stock.avg_volume.desc().nullslast()
                    )
                )
                all_stocks = result.scalars().all()
                
                # If no stocks in DB, use default set
                if not all_stocks:
                    # Load default S&P 500 stocks
                    sp500_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA',
                                   'JPM', 'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA', 'DIS',
                                   'BAC', 'WMT', 'XOM', 'CVX', 'ABBV', 'PFE', 'AVGO',
                                   'LLY', 'KO', 'PEP', 'TMO', 'COST', 'MRK', 'VZ', 'CMCSA']
                    
                    # Create stock entries
                    stocks_to_insert = []
                    for i, symbol in enumerate(sp500_symbols[:1000]):  # Limit to 1000
                        stocks_to_insert.append({
                            'symbol': symbol,
                            'name': f'{symbol} Company',
                            'exchange': 'NASDAQ' if i % 2 == 0 else 'NYSE',
                            'market_cap': 1000000000 * (1000 - i),  # Descending market cap
                            'avg_volume': 10000000 * (1000 - i)  # Descending volume
                        })
                    
                    # Bulk insert stocks
                    if stocks_to_insert:
                        await async_db_manager.bulk_insert(Stock, stocks_to_insert)
                        
                        # Re-fetch after insert
                        result = await session.execute(
                            select(Stock)
                            .order_by(
                                Stock.market_cap.desc().nullslast(),
                                Stock.avg_volume.desc().nullslast()
                            )
                        )
                        all_stocks = result.scalars().all()
                
                # Distribute stocks into tiers based on percentiles
                total_stocks = len(all_stocks)
                
                if total_stocks > 0:
                    # Calculate tier boundaries (targeting 1000 stocks total)
                    tier_sizes = {
                        StockTier.CRITICAL: min(100, total_stocks),  # Top 100
                        StockTier.HIGH: min(400, max(0, total_stocks - 100)),  # Next 400
                        StockTier.MEDIUM: min(300, max(0, total_stocks - 500)),  # Next 300
                        StockTier.LOW: min(200, max(0, total_stocks - 800)),  # Next 200
                        StockTier.MINIMAL: max(0, total_stocks - 1000)  # Rest (if > 1000)
                    }
                    
                    # Assign stocks to tiers
                    idx = 0
                    for tier, size in tier_sizes.items():
                        if size > 0:
                            tier_stocks = all_stocks[idx:idx + size]
                            self.stock_tiers[tier] = set(s.symbol for s in tier_stocks)
                            idx += size
                    
                    # Update tier assignments in database
                    for tier, symbols in self.stock_tiers.items():
                        if symbols:
                            await session.execute(
                                select(Stock)
                                .where(Stock.symbol.in_(symbols))
                                .execution_options(synchronize_session="fetch")
                            )
                            # Update tier field if it exists
                            for symbol in symbols:
                                await session.execute(
                                    select(Stock)
                                    .where(Stock.symbol == symbol)
                                    .execution_options(synchronize_session="fetch")
                                )
                
                logger.info(f"Loaded stock tiers from database: "
                           f"Critical={len(self.stock_tiers[StockTier.CRITICAL])}, "
                           f"High={len(self.stock_tiers[StockTier.HIGH])}, "
                           f"Medium={len(self.stock_tiers[StockTier.MEDIUM])}, "
                           f"Low={len(self.stock_tiers[StockTier.LOW])}, "
                           f"Minimal={len(self.stock_tiers[StockTier.MINIMAL])}")
                
        except Exception as e:
            logger.error(f"Failed to load stock tiers: {e}")
            # Use default minimal set on failure
            self.stock_tiers[StockTier.CRITICAL] = {'AAPL', 'MSFT', 'GOOGL'}
    
    def get_stock_tier(self, symbol: str) -> StockTier:
        """Get the tier for a given stock symbol."""
        for tier, symbols in self.stock_tiers.items():
            if symbol in symbols:
                return tier
        return StockTier.LOW  # Default tier
    
    async def fetch_stock_data(
        self,
        symbols: List[str],
        data_types: List[str] = ['price', 'fundamentals'],
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Fetch stock data with intelligent orchestration.
        
        Args:
            symbols: List of stock symbols
            data_types: Types of data to fetch
            force_refresh: Force refresh bypassing cache
        
        Returns:
            Dictionary of results by symbol
        """
        if not self._initialized:
            await self.initialize()
        
        results = {}
        
        # Check emergency mode first
        if await self.cost_monitor.is_in_emergency_mode():
            logger.warning("Emergency mode active - using cache only")
            return await self._fetch_cached_only(symbols, data_types)
        
        # Group symbols by tier
        tier_groups = self._group_symbols_by_tier(symbols)
        
        # Process each tier with appropriate strategy
        for tier, tier_symbols in tier_groups.items():
            if not tier_symbols:
                continue
            
            # Check if tier needs update
            if not self._should_update_tier(tier) and not force_refresh:
                # Use cached data
                tier_results = await self._fetch_cached_only(tier_symbols, data_types)
            else:
                # Fetch fresh data with tier-specific strategy
                tier_results = await self._fetch_tier_data(
                    tier, tier_symbols, data_types, force_refresh
                )
                self._last_tier_update[tier] = datetime.utcnow()
            
            results.update(tier_results)
        
        return results
    
    def _group_symbols_by_tier(self, symbols: List[str]) -> Dict[StockTier, List[str]]:
        """Group symbols by their tier."""
        tier_groups = {tier: [] for tier in StockTier}
        
        for symbol in symbols:
            tier = self.get_stock_tier(symbol)
            tier_groups[tier].append(symbol)
        
        return tier_groups
    
    def _should_update_tier(self, tier: StockTier) -> bool:
        """Check if a tier needs updating based on frequency."""
        if tier not in self._last_tier_update:
            return True
        
        last_update = self._last_tier_update[tier]
        update_frequency = self.tier_update_frequencies[tier]
        
        return (datetime.utcnow() - last_update).total_seconds() >= update_frequency
    
    async def _fetch_tier_data(
        self,
        tier: StockTier,
        symbols: List[str],
        data_types: List[str],
        force_refresh: bool
    ) -> Dict[str, Any]:
        """Fetch data for a specific tier with appropriate strategy."""
        results = {}
        
        # Get providers for this tier
        providers = self.tier_providers.get(tier, ['finnhub'])
        
        # Try cache first unless force refresh
        if not force_refresh:
            cached_results = await self._get_cached_batch(symbols, data_types)
            
            # Filter out symbols that have valid cache
            uncached_symbols = [
                s for s in symbols 
                if s not in cached_results or self._is_stale(cached_results[s])
            ]
            
            results.update(cached_results)
        else:
            uncached_symbols = symbols
        
        if not uncached_symbols:
            return results
        
        # Create tasks for uncached symbols
        tasks = []
        for symbol in uncached_symbols:
            for data_type in data_types:
                for provider in providers:
                    if provider == 'cache_only':
                        continue
                    
                    # Check rate limit preemptively
                    can_call = await self.rate_limiter.check_api_limit(provider)
                    if not can_call[0]:
                        # Try next provider
                        continue
                    
                    task = APITask(
                        id=f"{symbol}_{data_type}_{provider}_{datetime.utcnow().timestamp()}",
                        provider=provider,
                        endpoint=self._get_endpoint(provider, data_type),
                        params=self._build_params(symbol, data_type, provider),
                        priority=self._tier_to_priority(tier)
                    )
                    tasks.append(task)
                    break  # Use first available provider
        
        if tasks:
            # Process tasks in parallel
            task_results = await self.processor.process_batch(tasks, timeout=30)
            
            # Organize results by symbol
            for task, result in zip(tasks, task_results):
                if result.success:
                    symbol = self._extract_symbol_from_task(task)
                    data_type = self._extract_data_type_from_task(task)
                    
                    if symbol not in results:
                        results[symbol] = {}
                    results[symbol][data_type] = result.data
                    
                    # Cache successful results
                    cache_key = self._build_cache_key(symbol, data_type)
                    ttl = self._get_cache_ttl(tier, data_type)
                    await self.cache.set(cache_key, result.data, ttl=ttl)
                    
                    # Record cost
                    await self.cost_monitor.record_api_call(
                        provider=task.provider,
                        endpoint=task.endpoint,
                        success=True,
                        response_time_ms=result.latency_ms,
                        estimated_cost=0.0  # Free tier
                    )
        
        return results
    
    async def _fetch_cached_only(
        self,
        symbols: List[str],
        data_types: List[str]
    ) -> Dict[str, Any]:
        """Fetch only cached data (for emergency mode or low-tier stocks)."""
        results = {}
        
        for symbol in symbols:
            symbol_data = {}
            for data_type in data_types:
                cache_key = self._build_cache_key(symbol, data_type)
                cached = await self.cache.get(cache_key)
                
                if cached:
                    symbol_data[data_type] = cached
                else:
                    # Try stale cache
                    stale_key = f"{cache_key}:stale"
                    stale = await self.cache.get(stale_key)
                    if stale:
                        symbol_data[data_type] = stale
                        symbol_data[f"{data_type}_stale"] = True
            
            if symbol_data:
                results[symbol] = symbol_data
        
        return results
    
    async def _get_cached_batch(
        self,
        symbols: List[str],
        data_types: List[str]
    ) -> Dict[str, Any]:
        """Get cached data for multiple symbols."""
        results = {}
        
        cache_keys = []
        key_map = {}
        
        for symbol in symbols:
            for data_type in data_types:
                cache_key = self._build_cache_key(symbol, data_type)
                cache_keys.append(cache_key)
                key_map[cache_key] = (symbol, data_type)
        
        # Batch get from cache
        cached_values = await self.cache.get_batch(cache_keys)
        
        # Organize results
        for cache_key, value in cached_values.items():
            if value:
                symbol, data_type = key_map[cache_key]
                if symbol not in results:
                    results[symbol] = {}
                results[symbol][data_type] = value
        
        return results
    
    def _build_cache_key(self, symbol: str, data_type: str) -> str:
        """Build cache key for symbol and data type."""
        date_str = datetime.utcnow().strftime('%Y%m%d')
        tier = self.get_stock_tier(symbol)
        
        # Include tier in cache key for different TTLs
        return f"stock:{tier.value}:{symbol}:{data_type}:{date_str}"
    
    def _get_cache_ttl(self, tier: StockTier, data_type: str) -> int:
        """Get cache TTL based on tier and data type."""
        base_ttl = {
            'price': 300,           # 5 minutes
            'fundamentals': 86400,  # 1 day
            'news': 3600,          # 1 hour
            'technical': 900       # 15 minutes
        }.get(data_type, 3600)
        
        # Multiply by tier factor
        tier_multiplier = {
            StockTier.CRITICAL: 1,
            StockTier.HIGH: 2,
            StockTier.MEDIUM: 4,
            StockTier.LOW: 8,
            StockTier.MINIMAL: 24
        }.get(tier, 1)
        
        return base_ttl * tier_multiplier
    
    def _is_stale(self, data: Any) -> bool:
        """Check if data is stale."""
        if isinstance(data, dict) and '_stale' in data:
            return data['_stale']
        return False
    
    def _tier_to_priority(self, tier: StockTier) -> Priority:
        """Convert stock tier to task priority."""
        mapping = {
            StockTier.CRITICAL: Priority.CRITICAL,
            StockTier.HIGH: Priority.HIGH,
            StockTier.MEDIUM: Priority.MEDIUM,
            StockTier.LOW: Priority.LOW,
            StockTier.MINIMAL: Priority.BATCH
        }
        return mapping.get(tier, Priority.MEDIUM)
    
    def _get_endpoint(self, provider: str, data_type: str) -> str:
        """Get API endpoint for provider and data type."""
        endpoints = {
            'finnhub': {
                'price': 'quote',
                'fundamentals': 'stock/profile2',
                'news': 'company-news',
                'sentiment': 'news-sentiment'
            },
            'alpha_vantage': {
                'price': '',
                'fundamentals': '',
                'technical': ''
            },
            'polygon': {
                'price': 'v2/aggs/ticker',
                'fundamentals': 'v3/reference/tickers'
            }
        }
        
        return endpoints.get(provider, {}).get(data_type, '')
    
    def _build_params(
        self,
        symbol: str,
        data_type: str,
        provider: str
    ) -> Dict[str, Any]:
        """Build API parameters."""
        if provider == 'finnhub':
            return {'symbol': symbol}
        elif provider == 'alpha_vantage':
            function_map = {
                'price': 'GLOBAL_QUOTE',
                'fundamentals': 'OVERVIEW',
                'technical': 'RSI'
            }
            return {
                'symbol': symbol,
                'function': function_map.get(data_type, 'GLOBAL_QUOTE')
            }
        elif provider == 'polygon':
            return {'ticker': symbol}
        
        return {}
    
    def _extract_symbol_from_task(self, task: APITask) -> str:
        """Extract symbol from task."""
        # Parse from task ID or params
        if 'symbol' in task.params:
            return task.params['symbol']
        elif 'ticker' in task.params:
            return task.params['ticker']
        
        # Parse from ID (format: symbol_datatype_provider_timestamp)
        parts = task.id.split('_')
        if parts:
            return parts[0]
        
        return 'UNKNOWN'
    
    def _extract_data_type_from_task(self, task: APITask) -> str:
        """Extract data type from task."""
        # Parse from ID (format: symbol_datatype_provider_timestamp)
        parts = task.id.split('_')
        if len(parts) > 1:
            return parts[1]
        
        return 'unknown'
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        metrics = {
            'cache': await self.cache.get_stats() if self.cache else {},
            'rate_limiter': await self.rate_limiter.get_usage_stats() if self.rate_limiter else {},
            'processor': self.processor.get_performance_stats() if self.processor else {},
            'cost_monitor': await self.cost_monitor.get_usage_report() if self.cost_monitor else {},
            'stock_tiers': {
                tier.name: len(symbols) 
                for tier, symbols in self.stock_tiers.items()
            }
        }
        
        return metrics
    
    async def shutdown(self) -> None:
        """Shutdown all components gracefully."""
        try:
            shutdown_tasks = []
            
            if self.processor:
                shutdown_tasks.append(self.processor.shutdown())
            if self.cache:
                shutdown_tasks.append(self.cache.close())
            if self.rate_limiter:
                shutdown_tasks.append(self.rate_limiter.close())
            
            if shutdown_tasks:
                await asyncio.gather(*shutdown_tasks, return_exceptions=True)
            
            logger.info("Unified data ingestion shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


# Global instance
unified_ingestion = UnifiedDataIngestion()


# Integration with existing BaseAPIClient
async def enhance_base_api_client():
    """
    Enhance existing BaseAPIClient to use new components.
    This function should be called during application startup.
    """
    try:
        from backend.data_ingestion.base_client import BaseAPIClient
        
        # Monkey-patch BaseAPIClient to use unified ingestion
        original_fetch = BaseAPIClient.fetch_data
        
        async def enhanced_fetch(self, endpoint: str, params: Dict = None, **kwargs):
            """Enhanced fetch using unified ingestion."""
            # Try unified ingestion first
            if hasattr(self, 'symbol') and endpoint in ['quote', 'profile', 'news']:
                data_type = {
                    'quote': 'price',
                    'profile': 'fundamentals',
                    'news': 'news'
                }.get(endpoint, 'price')
                
                result = await unified_ingestion.fetch_stock_data(
                    symbols=[self.symbol],
                    data_types=[data_type]
                )
                
                if result and self.symbol in result:
                    return result[self.symbol].get(data_type)
            
            # Fallback to original method
            return await original_fetch(self, endpoint, params, **kwargs)
        
        BaseAPIClient.fetch_data = enhanced_fetch
        logger.info("BaseAPIClient enhanced with unified ingestion")
        
    except ImportError:
        logger.warning("BaseAPIClient not found, skipping enhancement")


# Integration with existing recommendation engine
async def enhance_recommendation_engine():
    """
    Enhance recommendation engine to use unified ingestion.
    """
    try:
        from backend.analytics.recommendation_engine import RecommendationEngine
        
        # Add unified ingestion to recommendation engine
        original_analyze = RecommendationEngine.analyze_stocks
        
        async def enhanced_analyze(self, symbols: List[str], **kwargs):
            """Enhanced analysis using unified ingestion."""
            # Fetch all required data through unified ingestion
            data = await unified_ingestion.fetch_stock_data(
                symbols=symbols,
                data_types=['price', 'fundamentals', 'news', 'technical']
            )
            
            # Store fetched data for analysis
            self._cached_data = data
            
            # Continue with original analysis
            return await original_analyze(self, symbols, **kwargs)
        
        RecommendationEngine.analyze_stocks = enhanced_analyze
        logger.info("RecommendationEngine enhanced with unified ingestion")
        
    except ImportError:
        logger.warning("RecommendationEngine not found, skipping enhancement")