"""
Cache Warming Strategy
Proactively warms caches during off-peak hours to prevent API overload.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timedelta, time
from enum import Enum
import random

from backend.utils.integration import UnifiedDataIngestion, StockTier
from backend.utils.query_cache import QueryResultCache
from backend.utils.async_database import get_async_session
from backend.utils.enhanced_cache_config import intelligent_cache_manager, StockTier as ConfigTier
from backend.config.settings import settings

logger = logging.getLogger(__name__)


class WarmingPriority(Enum):
    """Cache warming priority levels."""
    CRITICAL = 1  # Must warm before market open
    HIGH = 2      # Should warm if resources available
    MEDIUM = 3    # Warm during quiet periods
    LOW = 4       # Warm only if excess capacity


class CacheWarmingStrategy:
    """
    Implements intelligent cache warming to prevent cold start issues.
    """
    
    def __init__(
        self,
        ingestion: Optional[UnifiedDataIngestion] = None,
        cache: Optional[QueryResultCache] = None
    ):
        """
        Initialize cache warming strategy.
        
        Args:
            ingestion: Unified data ingestion instance
            cache: Query result cache instance
        """
        self.ingestion = ingestion or UnifiedDataIngestion()
        self.cache = cache or QueryResultCache()
        
        # Warming schedule (EST times)
        self.warming_windows = [
            (time(2, 0), time(4, 0)),   # 2-4 AM EST: Main warming window
            (time(13, 0), time(14, 0)),  # 1-2 PM EST: Lunch hour refresh
            (time(20, 0), time(21, 0))   # 8-9 PM EST: Evening update
        ]
        
        # Enhanced warming configuration
        self.batch_size = 20  # Increased batch size for efficiency
        self.delay_between_batches = 1.0  # Reduced delay for faster warming
        self.max_concurrent_warming = 10  # Increased concurrency
        self.predictive_warming_enabled = True
        self.market_aware_warming = True
        
        # Enhanced metrics
        self._metrics = {
            'caches_warmed': 0,
            'warming_errors': 0,
            'last_warming': None,
            'warming_duration_seconds': 0,
            'predictive_hits': 0,
            'tier_warming_stats': {tier.name: 0 for tier in StockTier},
            'market_events_processed': 0
        }
    
    def is_warming_window(self) -> bool:
        """Check if current time is within a warming window."""
        current_time = datetime.now().time()
        
        for start, end in self.warming_windows:
            if start <= current_time <= end:
                return True
        
        return False
    
    async def warm_critical_caches(self) -> Dict[str, Any]:
        """
        Warm critical caches (top-tier stocks).
        
        Returns:
            Warming results and metrics
        """
        start_time = datetime.now()
        results = {
            'warmed': [],
            'failed': [],
            'skipped': []
        }
        
        try:
            # Initialize if needed
            if not self.ingestion._initialized:
                await self.ingestion.initialize()
            
            # Get critical stocks
            critical_stocks = list(self.ingestion.stock_tiers.get(StockTier.CRITICAL, set()))
            
            if not critical_stocks:
                logger.warning("No critical stocks to warm")
                return results
            
            logger.info(f"Starting cache warming for {len(critical_stocks)} critical stocks")
            
            # Warm in batches
            for i in range(0, len(critical_stocks), self.batch_size):
                batch = critical_stocks[i:i + self.batch_size]
                
                # Check if we should continue (respect warming windows)
                if not self.is_warming_window() and i > 0:
                    logger.info("Exiting warming window, stopping cache warming")
                    results['skipped'].extend(critical_stocks[i:])
                    break
                
                # Warm batch
                batch_results = await self._warm_batch(
                    batch,
                    data_types=['price', 'fundamentals'],
                    priority=WarmingPriority.CRITICAL
                )
                
                results['warmed'].extend(batch_results['success'])
                results['failed'].extend(batch_results['failed'])
                
                # Delay between batches to avoid rate limiting
                if i + self.batch_size < len(critical_stocks):
                    await asyncio.sleep(self.delay_between_batches)
            
            # Update metrics
            self._metrics['caches_warmed'] += len(results['warmed'])
            self._metrics['warming_errors'] += len(results['failed'])
            self._metrics['last_warming'] = datetime.now()
            self._metrics['warming_duration_seconds'] = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Cache warming completed: {len(results['warmed'])} warmed, "
                       f"{len(results['failed'])} failed, {len(results['skipped'])} skipped")
            
        except Exception as e:
            logger.error(f"Critical cache warming failed: {e}")
            self._metrics['warming_errors'] += 1
        
        return results
    
    async def _warm_batch(
        self,
        symbols: List[str],
        data_types: List[str],
        priority: WarmingPriority
    ) -> Dict[str, List[str]]:
        """
        Warm cache for a batch of symbols.
        
        Args:
            symbols: Stock symbols to warm
            data_types: Types of data to cache
            priority: Warming priority
        
        Returns:
            Dict with success and failed lists
        """
        results = {'success': [], 'failed': []}
        
        # Create warming tasks
        tasks = []
        for symbol in symbols:
            task = self._warm_single_stock(symbol, data_types, priority)
            tasks.append(task)
        
        # Execute with concurrency limit
        semaphore = asyncio.Semaphore(self.max_concurrent_warming)
        
        async def limited_warm(symbol, data_types, priority):
            async with semaphore:
                return await self._warm_single_stock(symbol, data_types, priority)
        
        # Gather results
        warm_results = await asyncio.gather(
            *[limited_warm(s, data_types, priority) for s in symbols],
            return_exceptions=True
        )
        
        # Categorize results
        for symbol, result in zip(symbols, warm_results):
            if isinstance(result, Exception):
                results['failed'].append(symbol)
                logger.error(f"Failed to warm cache for {symbol}: {result}")
            elif result:
                results['success'].append(symbol)
            else:
                results['failed'].append(symbol)
        
        return results
    
    async def _warm_single_stock(
        self,
        symbol: str,
        data_types: List[str],
        priority: WarmingPriority
    ) -> bool:
        """
        Warm cache for a single stock.
        
        Args:
            symbol: Stock symbol
            data_types: Data types to cache
            priority: Warming priority
        
        Returns:
            True if successful
        """
        try:
            # Build cache keys
            for data_type in data_types:
                cache_key = self.ingestion._build_cache_key(symbol, data_type)
                
                # Check if already cached
                cached = await self.cache.get(cache_key)
                if cached and not self._is_stale_for_warming(cached):
                    logger.debug(f"Cache for {symbol}:{data_type} is fresh, skipping")
                    continue
                
                # Fetch fresh data with lower priority
                # This avoids interfering with real-time requests
                result = await self.ingestion.fetch_stock_data(
                    [symbol],
                    [data_type],
                    force_refresh=False  # Use cache if available
                )
                
                if result and symbol in result:
                    logger.debug(f"Warmed cache for {symbol}:{data_type}")
                    return True
            
            return True
            
        except Exception as e:
            logger.error(f"Error warming cache for {symbol}: {e}")
            return False
    
    def _is_stale_for_warming(self, data: Any) -> bool:
        """
        Check if cached data is stale enough to warrant warming.
        
        Args:
            data: Cached data
        
        Returns:
            True if data should be refreshed
        """
        # If data has timestamp, check age
        if isinstance(data, dict) and 'timestamp' in data:
            try:
                timestamp = datetime.fromisoformat(data['timestamp'])
                age = datetime.now() - timestamp
                
                # Consider stale if older than 4 hours for warming
                return age > timedelta(hours=4)
            except (ValueError, TypeError):
                pass
        
        # If marked as stale
        if isinstance(data, dict) and data.get('_stale'):
            return True
        
        return False
    
    async def warm_by_tier(self, tier: StockTier) -> Dict[str, Any]:
        """
        Warm caches for all stocks in a specific tier.
        
        Args:
            tier: Stock tier to warm
        
        Returns:
            Warming results
        """
        stocks = list(self.ingestion.stock_tiers.get(tier, set()))
        
        if not stocks:
            return {'warmed': [], 'failed': [], 'skipped': []}
        
        # Determine data types based on tier
        if tier == StockTier.CRITICAL:
            data_types = ['price', 'fundamentals', 'news', 'technical']
        elif tier == StockTier.HIGH:
            data_types = ['price', 'fundamentals']
        else:
            data_types = ['price']
        
        # Map tier to priority
        priority_map = {
            StockTier.CRITICAL: WarmingPriority.CRITICAL,
            StockTier.HIGH: WarmingPriority.HIGH,
            StockTier.MEDIUM: WarmingPriority.MEDIUM,
            StockTier.LOW: WarmingPriority.LOW,
            StockTier.MINIMAL: WarmingPriority.LOW
        }
        priority = priority_map.get(tier, WarmingPriority.LOW)
        
        results = {'warmed': [], 'failed': [], 'skipped': []}
        
        # Process in batches
        for i in range(0, len(stocks), self.batch_size):
            batch = stocks[i:i + self.batch_size]
            
            batch_results = await self._warm_batch(batch, data_types, priority)
            results['warmed'].extend(batch_results['success'])
            results['failed'].extend(batch_results['failed'])
            
            # Random delay to avoid patterns
            delay = self.delay_between_batches + random.uniform(0, 1)
            await asyncio.sleep(delay)
        
        return results
    
    async def adaptive_warming(self) -> None:
        """
        Adaptive cache warming based on system load and time of day.
        """
        while True:
            try:
                # Check if in warming window
                if self.is_warming_window():
                    # Check system load
                    if await self._is_system_idle():
                        # Warm critical caches first
                        await self.warm_critical_caches()
                        
                        # If still in window and idle, warm high tier
                        if self.is_warming_window() and await self._is_system_idle():
                            await self.warm_by_tier(StockTier.HIGH)
                        
                        # Continue with lower tiers if possible
                        for tier in [StockTier.MEDIUM, StockTier.LOW]:
                            if self.is_warming_window() and await self._is_system_idle():
                                await self.warm_by_tier(tier)
                            else:
                                break
                
                # Sleep until next check (5 minutes)
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Adaptive warming error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    async def _is_system_idle(self) -> bool:
        """
        Check if system is idle enough for cache warming.
        
        Returns:
            True if system is idle
        """
        try:
            # Check cost monitor emergency mode
            if await self.ingestion.cost_monitor.is_in_emergency_mode():
                return False
            
            # Check current API usage
            usage = await self.ingestion.cost_monitor.get_monthly_usage()
            if usage.get('api_calls', 0) > 8000:  # Near daily limit
                return False
            
            # Check time of day (avoid market hours 9:30 AM - 4 PM EST)
            current_hour = datetime.now().hour
            if 9 <= current_hour <= 16:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking system idle status: {e}")
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get cache warming metrics."""
        return self._metrics.copy()
    
    async def schedule_warming(self) -> None:
        """
        Schedule cache warming tasks.
        """
        logger.info("Starting cache warming scheduler")
        
        while True:
            try:
                # Calculate next warming window
                next_window = self._get_next_warming_window()
                
                if next_window:
                    wait_seconds = (next_window - datetime.now()).total_seconds()
                    
                    if wait_seconds > 0:
                        logger.info(f"Next cache warming in {wait_seconds/3600:.1f} hours")
                        await asyncio.sleep(wait_seconds)
                    
                    # Start warming
                    await self.warm_critical_caches()
                else:
                    # No window found, wait 1 hour
                    await asyncio.sleep(3600)
                    
            except Exception as e:
                logger.error(f"Cache warming scheduler error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    def _get_next_warming_window(self) -> Optional[datetime]:
        """
        Get the next warming window start time.
        
        Returns:
            Next warming window datetime or None
        """
        now = datetime.now()
        current_time = now.time()
        
        # Check today's windows
        for start, end in self.warming_windows:
            if current_time < start:
                return datetime.combine(now.date(), start)
        
        # Next day's first window
        if self.warming_windows:
            tomorrow = now + timedelta(days=1)
            return datetime.combine(tomorrow.date(), self.warming_windows[0][0])
        
        return None


# Global cache warming instance
cache_warmer = CacheWarmingStrategy()


# Utility function to start warming in background
def start_cache_warming_background():
    """Start cache warming in background task."""
    asyncio.create_task(cache_warmer.adaptive_warming())
    logger.info("Cache warming background task started")