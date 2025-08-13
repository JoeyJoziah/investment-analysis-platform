"""
Tier-Based Caching System
Advanced caching strategies aligned with stock priorities and business value.
Optimizes cache allocation, TTL, and resources based on stock importance tiers.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from datetime import datetime, timedelta
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field
from enum import Enum
import json
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from backend.utils.enhanced_cache_config import StockTier, intelligent_cache_manager
from backend.utils.cache_hit_optimization import cache_hit_optimizer
from backend.utils.monitoring import metrics

logger = logging.getLogger(__name__)


@dataclass
class TierConfiguration:
    """Configuration for a specific stock tier."""
    tier: StockTier
    priority: int  # 1=highest, 5=lowest
    cache_allocation_ratio: float  # % of total cache space
    ttl_multiplier: float  # TTL adjustment factor
    max_entries: int
    compression_level: str  # none, light, medium, heavy
    update_frequency_minutes: int
    warming_enabled: bool
    predictive_caching: bool
    batch_size: int
    prefetch_enabled: bool
    
    # Advanced settings
    memory_pressure_threshold: float = 0.85
    eviction_policy: str = "lru"  # lru, lfu, ttl-aware
    replication_factor: int = 1  # for distributed caching
    consistency_level: str = "eventual"  # strong, eventual


class TierManager:
    """
    Manages stock tier assignments and optimizations.
    """
    
    def __init__(self):
        self.tier_configs = self._initialize_tier_configs()
        self.stock_assignments: Dict[str, StockTier] = {}
        self.tier_performance: Dict[StockTier, Dict[str, float]] = defaultdict(dict)
        self.assignment_history: Dict[str, List[Tuple[datetime, StockTier]]] = defaultdict(list)
        self._lock = threading.RLock()
        
        # Performance tracking
        self.access_counts = defaultdict(lambda: defaultdict(int))  # tier -> stock -> count
        self.response_times = defaultdict(lambda: defaultdict(list))  # tier -> stock -> times
        self.cache_hit_rates = defaultdict(float)  # tier -> hit_rate
        
        # Dynamic tier adjustment
        self.auto_adjustment_enabled = True
        self.adjustment_threshold = 0.1  # Minimum change to trigger reassignment
        self.evaluation_window = timedelta(hours=24)
        
    def _initialize_tier_configs(self) -> Dict[StockTier, TierConfiguration]:
        """Initialize tier configurations optimized for financial data."""
        return {
            StockTier.CRITICAL: TierConfiguration(
                tier=StockTier.CRITICAL,
                priority=1,
                cache_allocation_ratio=0.40,  # 40% of cache for critical stocks
                ttl_multiplier=1.0,
                max_entries=2000,
                compression_level="light",
                update_frequency_minutes=1,
                warming_enabled=True,
                predictive_caching=True,
                batch_size=50,
                prefetch_enabled=True,
                eviction_policy="lfu",  # Least Frequently Used for critical
                replication_factor=2,
                consistency_level="strong"
            ),
            StockTier.HIGH: TierConfiguration(
                tier=StockTier.HIGH,
                priority=2,
                cache_allocation_ratio=0.30,  # 30% of cache
                ttl_multiplier=1.2,
                max_entries=1500,
                compression_level="light",
                update_frequency_minutes=5,
                warming_enabled=True,
                predictive_caching=True,
                batch_size=30,
                prefetch_enabled=True,
                eviction_policy="lru"
            ),
            StockTier.MEDIUM: TierConfiguration(
                tier=StockTier.MEDIUM,
                priority=3,
                cache_allocation_ratio=0.20,  # 20% of cache
                ttl_multiplier=1.5,
                max_entries=1000,
                compression_level="medium",
                update_frequency_minutes=15,
                warming_enabled=False,
                predictive_caching=False,
                batch_size=20,
                prefetch_enabled=False,
                eviction_policy="lru"
            ),
            StockTier.LOW: TierConfiguration(
                tier=StockTier.LOW,
                priority=4,
                cache_allocation_ratio=0.08,  # 8% of cache
                ttl_multiplier=2.0,
                max_entries=400,
                compression_level="heavy",
                update_frequency_minutes=60,
                warming_enabled=False,
                predictive_caching=False,
                batch_size=10,
                prefetch_enabled=False,
                eviction_policy="ttl-aware"
            ),
            StockTier.MINIMAL: TierConfiguration(
                tier=StockTier.MINIMAL,
                priority=5,
                cache_allocation_ratio=0.02,  # 2% of cache
                ttl_multiplier=3.0,
                max_entries=100,
                compression_level="heavy",
                update_frequency_minutes=240,  # 4 hours
                warming_enabled=False,
                predictive_caching=False,
                batch_size=5,
                prefetch_enabled=False,
                eviction_policy="ttl-aware"
            )
        }
    
    def assign_stock_to_tier(
        self,
        symbol: str,
        tier: StockTier,
        reason: str = "manual",
        confidence: float = 1.0
    ) -> bool:
        """
        Assign stock to a specific tier with tracking.
        """
        with self._lock:
            old_tier = self.stock_assignments.get(symbol)
            
            if old_tier == tier:
                return False  # No change needed
            
            # Update assignment
            self.stock_assignments[symbol] = tier
            
            # Track assignment history
            self.assignment_history[symbol].append((datetime.now(), tier))
            
            # Limit history size
            if len(self.assignment_history[symbol]) > 50:
                self.assignment_history[symbol] = self.assignment_history[symbol][-25:]
            
            logger.info(f"Assigned {symbol} to {tier.name} tier (reason: {reason}, confidence: {confidence:.2f})")
            
            # Update intelligent cache manager
            intelligent_cache_manager.assign_stock_tier(symbol, tier)
            
            return True
    
    def get_stock_tier(self, symbol: str) -> StockTier:
        """Get current tier assignment for a stock."""
        return self.stock_assignments.get(symbol, StockTier.MEDIUM)  # Default to medium
    
    def get_tier_config(self, tier: StockTier) -> TierConfiguration:
        """Get configuration for a tier."""
        return self.tier_configs[tier]
    
    async def auto_assign_tiers(self, market_data: Dict[str, Any]) -> Dict[str, StockTier]:
        """
        Automatically assign stocks to tiers based on market data and patterns.
        """
        new_assignments = {}
        
        try:
            # Get stock metrics for tier assignment
            stock_metrics = await self._calculate_stock_metrics(market_data)
            
            # Assign based on multiple criteria
            for symbol, metrics in stock_metrics.items():
                tier = self._determine_optimal_tier(symbol, metrics)
                
                if self.assign_stock_to_tier(symbol, tier, "auto", metrics.get('confidence', 0.8)):
                    new_assignments[symbol] = tier
            
            logger.info(f"Auto-assigned {len(new_assignments)} stocks to tiers")
            
        except Exception as e:
            logger.error(f"Auto tier assignment failed: {e}")
        
        return new_assignments
    
    async def _calculate_stock_metrics(self, market_data: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Calculate metrics for tier assignment."""
        stock_metrics = {}
        
        for symbol in market_data.get('symbols', []):
            try:
                metrics = {}
                
                # Market cap tier scoring
                market_cap = market_data.get('market_caps', {}).get(symbol, 0)
                metrics['market_cap_score'] = self._score_market_cap(market_cap)
                
                # Volume tier scoring
                avg_volume = market_data.get('avg_volumes', {}).get(symbol, 0)
                metrics['volume_score'] = self._score_volume(avg_volume)
                
                # Volatility scoring
                volatility = market_data.get('volatilities', {}).get(symbol, 0)
                metrics['volatility_score'] = self._score_volatility(volatility)
                
                # Access frequency scoring
                access_freq = self.access_counts.get(self.get_stock_tier(symbol), {}).get(symbol, 0)
                metrics['access_score'] = self._score_access_frequency(access_freq)
                
                # News/interest scoring
                news_mentions = market_data.get('news_mentions', {}).get(symbol, 0)
                metrics['interest_score'] = self._score_interest(news_mentions)
                
                # Composite score
                weights = {
                    'market_cap_score': 0.25,
                    'volume_score': 0.20,
                    'volatility_score': 0.15,
                    'access_score': 0.25,
                    'interest_score': 0.15
                }
                
                composite_score = sum(
                    metrics[metric] * weights[metric]
                    for metric in weights
                )
                metrics['composite_score'] = composite_score
                metrics['confidence'] = min(1.0, composite_score / 5.0)
                
                stock_metrics[symbol] = metrics
                
            except Exception as e:
                logger.error(f"Failed to calculate metrics for {symbol}: {e}")
                continue
        
        return stock_metrics
    
    def _score_market_cap(self, market_cap: float) -> float:
        """Score stock based on market capitalization (0-5 scale)."""
        if market_cap >= 200_000_000_000:  # $200B+ (mega-cap)
            return 5.0
        elif market_cap >= 10_000_000_000:  # $10B-200B (large-cap)
            return 4.0
        elif market_cap >= 2_000_000_000:   # $2B-10B (mid-cap)
            return 3.0
        elif market_cap >= 300_000_000:     # $300M-2B (small-cap)
            return 2.0
        else:                               # <$300M (micro-cap)
            return 1.0
    
    def _score_volume(self, avg_volume: float) -> float:
        """Score stock based on average daily volume (0-5 scale)."""
        if avg_volume >= 10_000_000:    # 10M+ shares
            return 5.0
        elif avg_volume >= 1_000_000:   # 1M-10M shares
            return 4.0
        elif avg_volume >= 100_000:     # 100K-1M shares
            return 3.0
        elif avg_volume >= 10_000:      # 10K-100K shares
            return 2.0
        else:                           # <10K shares
            return 1.0
    
    def _score_volatility(self, volatility: float) -> float:
        """Score stock based on volatility (0-5 scale)."""
        # Higher volatility = more interest = higher tier
        if volatility >= 0.8:      # Very high volatility
            return 5.0
        elif volatility >= 0.4:    # High volatility
            return 4.0
        elif volatility >= 0.2:    # Medium volatility
            return 3.0
        elif volatility >= 0.1:    # Low volatility
            return 2.0
        else:                      # Very low volatility
            return 1.0
    
    def _score_access_frequency(self, access_count: int) -> float:
        """Score stock based on recent access frequency (0-5 scale)."""
        if access_count >= 100:    # Very frequently accessed
            return 5.0
        elif access_count >= 50:   # Frequently accessed
            return 4.0
        elif access_count >= 20:   # Moderately accessed
            return 3.0
        elif access_count >= 5:    # Occasionally accessed
            return 2.0
        else:                      # Rarely accessed
            return 1.0
    
    def _score_interest(self, news_mentions: int) -> float:
        """Score stock based on news mentions and interest (0-5 scale)."""
        if news_mentions >= 50:    # Very high interest
            return 5.0
        elif news_mentions >= 20:  # High interest
            return 4.0
        elif news_mentions >= 10:  # Medium interest
            return 3.0
        elif news_mentions >= 5:   # Low interest
            return 2.0
        else:                      # Minimal interest
            return 1.0
    
    def _determine_optimal_tier(self, symbol: str, metrics: Dict[str, float]) -> StockTier:
        """Determine optimal tier based on composite metrics."""
        composite_score = metrics.get('composite_score', 2.5)
        
        if composite_score >= 4.5:
            return StockTier.CRITICAL
        elif composite_score >= 3.5:
            return StockTier.HIGH
        elif composite_score >= 2.5:
            return StockTier.MEDIUM
        elif composite_score >= 1.5:
            return StockTier.LOW
        else:
            return StockTier.MINIMAL
    
    def track_access(self, symbol: str, response_time_ms: float):
        """Track access for performance monitoring and tier optimization."""
        tier = self.get_stock_tier(symbol)
        
        with self._lock:
            self.access_counts[tier][symbol] += 1
            self.response_times[tier][symbol].append(response_time_ms)
            
            # Limit response time history
            if len(self.response_times[tier][symbol]) > 100:
                self.response_times[tier][symbol] = self.response_times[tier][symbol][-50:]
    
    def update_tier_performance(self):
        """Update tier performance metrics."""
        for tier in StockTier:
            config = self.tier_configs[tier]
            
            # Calculate average response time
            tier_response_times = []
            for symbol_times in self.response_times[tier].values():
                tier_response_times.extend(symbol_times)
            
            if tier_response_times:
                avg_response_time = np.mean(tier_response_times)
                self.tier_performance[tier]['avg_response_time_ms'] = avg_response_time
                self.tier_performance[tier]['p95_response_time_ms'] = np.percentile(tier_response_times, 95)
            
            # Calculate access frequency
            total_accesses = sum(self.access_counts[tier].values())
            self.tier_performance[tier]['total_accesses'] = total_accesses
            self.tier_performance[tier]['unique_stocks'] = len(self.access_counts[tier])
            
            # Calculate efficiency score
            efficiency = self._calculate_tier_efficiency(tier)
            self.tier_performance[tier]['efficiency_score'] = efficiency
    
    def _calculate_tier_efficiency(self, tier: StockTier) -> float:
        """Calculate efficiency score for a tier (0-100)."""
        config = self.tier_configs[tier]
        
        # Factors for efficiency calculation
        factors = {}
        
        # Cache utilization
        current_entries = len(self.access_counts[tier])
        utilization = current_entries / config.max_entries if config.max_entries > 0 else 0
        factors['utilization'] = min(utilization * 100, 100)  # Cap at 100%
        
        # Response time performance
        avg_response = self.tier_performance[tier].get('avg_response_time_ms', 100)
        target_response = 50 if tier == StockTier.CRITICAL else 100  # Target response times
        response_score = max(0, 100 - ((avg_response - target_response) / target_response * 50))
        factors['response_performance'] = max(0, response_score)
        
        # Access frequency appropriateness
        total_accesses = self.tier_performance[tier].get('total_accesses', 0)
        expected_min_accesses = {
            StockTier.CRITICAL: 1000,
            StockTier.HIGH: 500,
            StockTier.MEDIUM: 200,
            StockTier.LOW: 50,
            StockTier.MINIMAL: 10
        }
        
        min_expected = expected_min_accesses[tier]
        access_score = min(100, (total_accesses / min_expected) * 100)
        factors['access_appropriateness'] = access_score
        
        # Weighted efficiency score
        weights = {
            'utilization': 0.3,
            'response_performance': 0.4,
            'access_appropriateness': 0.3
        }
        
        efficiency_score = sum(factors[factor] * weights[factor] for factor in weights)
        return min(100, efficiency_score)
    
    def get_tier_statistics(self) -> Dict[str, Any]:
        """Get comprehensive tier statistics."""
        self.update_tier_performance()
        
        stats = {}
        
        for tier in StockTier:
            config = self.tier_configs[tier]
            performance = self.tier_performance[tier]
            
            tier_stocks = [symbol for symbol, assigned_tier in self.stock_assignments.items() 
                          if assigned_tier == tier]
            
            stats[tier.name] = {
                'configuration': {
                    'priority': config.priority,
                    'cache_allocation_ratio': config.cache_allocation_ratio,
                    'max_entries': config.max_entries,
                    'ttl_multiplier': config.ttl_multiplier,
                    'compression_level': config.compression_level,
                    'update_frequency_minutes': config.update_frequency_minutes
                },
                'current_state': {
                    'assigned_stocks': len(tier_stocks),
                    'total_accesses': performance.get('total_accesses', 0),
                    'avg_response_time_ms': performance.get('avg_response_time_ms', 0),
                    'p95_response_time_ms': performance.get('p95_response_time_ms', 0),
                    'efficiency_score': performance.get('efficiency_score', 0)
                },
                'sample_stocks': tier_stocks[:10]  # First 10 stocks
            }
        
        return stats


class TierBasedCacheStrategy:
    """
    Main tier-based caching strategy implementation.
    """
    
    def __init__(self):
        self.tier_manager = TierManager()
        self.cache_pools = {}  # tier -> cache pool
        self.executor = ThreadPoolExecutor(max_workers=8)
        self._lock = threading.RLock()
        
        # Initialize cache pools for each tier
        self._initialize_cache_pools()
        
        # Background tasks
        self.optimization_task: Optional[asyncio.Task] = None
        self.monitoring_task: Optional[asyncio.Task] = None
    
    def _initialize_cache_pools(self):
        """Initialize separate cache pools for each tier."""
        for tier in StockTier:
            config = self.tier_manager.get_tier_config(tier)
            self.cache_pools[tier] = TierCachePool(config)
    
    async def get_cached_data(
        self,
        symbol: str,
        data_type: str,
        date: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Tuple[Optional[Any], bool, Dict[str, Any]]:
        """
        Get cached data using tier-based strategy.
        
        Returns:
            Tuple of (data, cache_hit, metadata)
        """
        start_time = time.time()
        
        # Get stock tier
        tier = self.tier_manager.get_stock_tier(symbol)
        config = self.tier_manager.get_tier_config(tier)
        
        # Get from appropriate cache pool
        cache_pool = self.cache_pools[tier]
        
        # Generate optimized cache key
        cache_key, _, key_metadata = await cache_hit_optimizer.get_optimized_cache_entry(
            symbol, data_type, {}, date, params
        )
        
        # Try to get from cache
        cached_data, hit = await cache_pool.get(cache_key)
        
        # Track access for optimization
        response_time_ms = (time.time() - start_time) * 1000
        self.tier_manager.track_access(symbol, response_time_ms)
        
        # Update hit rate metrics
        await cache_hit_optimizer.track_cache_access(
            cache_key, hit, response_time_ms
        )
        
        metadata = {
            'tier': tier.name,
            'cache_hit': hit,
            'response_time_ms': response_time_ms,
            'cache_key': cache_key,
            'tier_config': {
                'priority': config.priority,
                'ttl_multiplier': config.ttl_multiplier,
                'compression_level': config.compression_level
            }
        }
        
        return cached_data, hit, metadata
    
    async def set_cached_data(
        self,
        symbol: str,
        data_type: str,
        data: Any,
        date: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        custom_ttl: Optional[int] = None
    ) -> bool:
        """
        Set cached data using tier-based strategy.
        """
        try:
            # Get stock tier and configuration
            tier = self.tier_manager.get_stock_tier(symbol)
            config = self.tier_manager.get_tier_config(tier)
            
            # Get cache pool
            cache_pool = self.cache_pools[tier]
            
            # Generate optimized cache entry
            cache_key, compressed_data, entry_metadata = await cache_hit_optimizer.get_optimized_cache_entry(
                symbol, data_type, data, date, params, urgency='normal'
            )
            
            # Calculate TTL based on tier configuration
            base_ttl = intelligent_cache_manager.get_optimal_ttl(symbol, data_type)
            adjusted_ttl = int(base_ttl * config.ttl_multiplier)
            
            if custom_ttl:
                adjusted_ttl = custom_ttl
            
            # Set in cache pool
            success = await cache_pool.set(
                cache_key,
                compressed_data,
                ttl=adjusted_ttl,
                metadata=entry_metadata
            )
            
            logger.debug(f"Cached {symbol}:{data_type} in {tier.name} tier (TTL: {adjusted_ttl}s)")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to cache data for {symbol}: {e}")
            return False
    
    async def invalidate_tier_cache(
        self,
        tier: StockTier,
        pattern: Optional[str] = None
    ) -> int:
        """Invalidate cache entries for a specific tier."""
        cache_pool = self.cache_pools[tier]
        return await cache_pool.invalidate(pattern)
    
    async def warm_tier_caches(self, tier: StockTier) -> Dict[str, Any]:
        """Warm caches for a specific tier."""
        config = self.tier_manager.get_tier_config(tier)
        
        if not config.warming_enabled:
            return {'warmed': 0, 'message': f'Warming disabled for {tier.name}'}
        
        # Get stocks in this tier
        tier_stocks = [
            symbol for symbol, assigned_tier in self.tier_manager.stock_assignments.items()
            if assigned_tier == tier
        ]
        
        if not tier_stocks:
            return {'warmed': 0, 'message': f'No stocks assigned to {tier.name}'}
        
        # Warm in batches
        warmed_count = 0
        batch_size = config.batch_size
        
        for i in range(0, len(tier_stocks), batch_size):
            batch = tier_stocks[i:i + batch_size]
            
            # Warm batch concurrently
            tasks = []
            for symbol in batch:
                if config.predictive_caching:
                    # Use predictive warming
                    task = self._predictive_warm_stock(symbol, tier)
                else:
                    # Standard warming
                    task = self._warm_stock(symbol, tier)
                tasks.append(task)
            
            # Execute batch
            results = await asyncio.gather(*tasks, return_exceptions=True)
            warmed_count += sum(1 for result in results if result is True)
            
            # Delay between batches
            if i + batch_size < len(tier_stocks):
                await asyncio.sleep(1.0)
        
        logger.info(f"Warmed {warmed_count} stocks in {tier.name} tier")
        
        return {
            'warmed': warmed_count,
            'total_stocks': len(tier_stocks),
            'tier': tier.name
        }
    
    async def _warm_stock(self, symbol: str, tier: StockTier) -> bool:
        """Warm cache for a single stock."""
        config = self.tier_manager.get_tier_config(tier)
        
        # Determine data types to warm based on tier
        if tier == StockTier.CRITICAL:
            data_types = ['price', 'fundamentals', 'technical', 'news']
        elif tier == StockTier.HIGH:
            data_types = ['price', 'fundamentals', 'technical']
        else:
            data_types = ['price', 'fundamentals']
        
        # This would integrate with data fetching services
        # For now, return success
        return True
    
    async def _predictive_warm_stock(self, symbol: str, tier: StockTier) -> bool:
        """Predictively warm cache for a stock based on patterns."""
        # This would use ML predictions to determine what to cache
        return await self._warm_stock(symbol, tier)
    
    async def optimize_tier_assignments(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize tier assignments based on current performance."""
        optimization_results = {
            'reassignments': 0,
            'improvements': {},
            'performance_changes': {}
        }
        
        try:
            # Auto-assign tiers based on current market data
            new_assignments = await self.tier_manager.auto_assign_tiers(market_data)
            optimization_results['reassignments'] = len(new_assignments)
            
            # Analyze performance improvements
            for tier in StockTier:
                old_efficiency = self.tier_manager.tier_performance[tier].get('efficiency_score', 0)
                
                # Recalculate efficiency after reassignments
                self.tier_manager.update_tier_performance()
                new_efficiency = self.tier_manager.tier_performance[tier].get('efficiency_score', 0)
                
                improvement = new_efficiency - old_efficiency
                optimization_results['improvements'][tier.name] = improvement
            
            logger.info(f"Tier optimization completed: {optimization_results['reassignments']} reassignments")
            
        except Exception as e:
            logger.error(f"Tier optimization failed: {e}")
        
        return optimization_results
    
    async def start_background_tasks(self):
        """Start background optimization and monitoring tasks."""
        if not self.optimization_task:
            self.optimization_task = asyncio.create_task(self._optimization_loop())
        
        if not self.monitoring_task:
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Tier-based caching background tasks started")
    
    async def _optimization_loop(self):
        """Background optimization loop."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Get current market data (placeholder)
                market_data = await self._get_market_data()
                
                # Optimize tier assignments
                await self.optimize_tier_assignments(market_data)
                
            except Exception as e:
                logger.error(f"Optimization loop error: {e}")
                await asyncio.sleep(1800)  # Wait 30 minutes on error
    
    async def _monitoring_loop(self):
        """Background monitoring loop."""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # Update tier performance metrics
                self.tier_manager.update_tier_performance()
                
                # Log performance summary
                stats = self.tier_manager.get_tier_statistics()
                
                for tier_name, tier_stats in stats.items():
                    efficiency = tier_stats['current_state']['efficiency_score']
                    if efficiency < 70:  # Alert on low efficiency
                        logger.warning(f"{tier_name} tier efficiency is low: {efficiency:.1f}%")
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(300)
    
    async def _get_market_data(self) -> Dict[str, Any]:
        """Get current market data for optimization."""
        # Placeholder implementation
        return {
            'symbols': list(self.tier_manager.stock_assignments.keys()),
            'market_caps': {},
            'avg_volumes': {},
            'volatilities': {},
            'news_mentions': {}
        }
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics for all tiers."""
        tier_stats = self.tier_manager.get_tier_statistics()
        
        # Add cache pool stats
        for tier in StockTier:
            if tier.name in tier_stats:
                pool_stats = self.cache_pools[tier].get_stats()
                tier_stats[tier.name]['cache_pool'] = pool_stats
        
        # Add optimization metrics
        optimization_metrics = cache_hit_optimizer.get_optimization_metrics()
        
        return {
            'tier_statistics': tier_stats,
            'optimization_metrics': optimization_metrics,
            'summary': {
                'total_stocks': len(self.tier_manager.stock_assignments),
                'total_tiers': len(StockTier),
                'background_tasks_running': (
                    bool(self.optimization_task and not self.optimization_task.done()) and
                    bool(self.monitoring_task and not self.monitoring_task.done())
                )
            }
        }


class TierCachePool:
    """Cache pool for a specific tier with tier-specific optimizations."""
    
    def __init__(self, config: TierConfiguration):
        self.config = config
        self.cache: OrderedDict = OrderedDict()
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, float] = {}
        self._lock = threading.RLock()
        
        # Metrics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.total_size_bytes = 0
    
    async def get(self, key: str) -> Tuple[Optional[Any], bool]:
        """Get value from tier cache pool."""
        with self._lock:
            if key in self.cache:
                # Update access time for LRU
                self.access_times[key] = time.time()
                
                # Move to end for LRU (if using LRU policy)
                if self.config.eviction_policy == "lru":
                    self.cache.move_to_end(key)
                
                self.hits += 1
                return self.cache[key], True
            else:
                self.misses += 1
                return None, False
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Set value in tier cache pool."""
        try:
            with self._lock:
                # Check if we need to evict
                if len(self.cache) >= self.config.max_entries:
                    await self._evict()
                
                # Store value and metadata
                self.cache[key] = value
                self.metadata[key] = metadata or {}
                self.metadata[key]['ttl'] = ttl
                self.metadata[key]['created_at'] = time.time()
                self.access_times[key] = time.time()
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to set cache value: {e}")
            return False
    
    async def _evict(self) -> None:
        """Evict entries based on tier eviction policy."""
        if not self.cache:
            return
        
        if self.config.eviction_policy == "lru":
            # Remove least recently used
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            del self.metadata[oldest_key]
            del self.access_times[oldest_key]
            
        elif self.config.eviction_policy == "lfu":
            # Remove least frequently used (simplified - would need frequency tracking)
            oldest_key = min(self.access_times, key=self.access_times.get)
            del self.cache[oldest_key]
            del self.metadata[oldest_key]
            del self.access_times[oldest_key]
            
        elif self.config.eviction_policy == "ttl-aware":
            # Remove expired entries first, then oldest
            current_time = time.time()
            expired_keys = [
                key for key, meta in self.metadata.items()
                if current_time - meta.get('created_at', 0) > meta.get('ttl', 0)
            ]
            
            if expired_keys:
                key_to_remove = expired_keys[0]
            else:
                key_to_remove = next(iter(self.cache))
            
            del self.cache[key_to_remove]
            del self.metadata[key_to_remove]
            del self.access_times[key_to_remove]
        
        self.evictions += 1
    
    async def invalidate(self, pattern: Optional[str] = None) -> int:
        """Invalidate cache entries matching pattern."""
        with self._lock:
            if pattern:
                # Pattern-based invalidation
                import fnmatch
                keys_to_remove = [
                    key for key in self.cache.keys()
                    if fnmatch.fnmatch(key, pattern)
                ]
            else:
                # Invalidate all
                keys_to_remove = list(self.cache.keys())
            
            for key in keys_to_remove:
                del self.cache[key]
                if key in self.metadata:
                    del self.metadata[key]
                if key in self.access_times:
                    del self.access_times[key]
            
            return len(keys_to_remove)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache pool statistics."""
        with self._lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
            
            return {
                'entries': len(self.cache),
                'max_entries': self.config.max_entries,
                'utilization_percent': (len(self.cache) / self.config.max_entries * 100),
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate_percent': hit_rate,
                'evictions': self.evictions,
                'eviction_policy': self.config.eviction_policy
            }


# Global tier-based cache strategy
tier_based_cache = TierBasedCacheStrategy()