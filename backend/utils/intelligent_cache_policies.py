"""
Intelligent Cache Policies and Warming Strategies

This module implements smart TTL policies, cache warming strategies, and 
cost-optimization algorithms for the investment platform.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
from collections import defaultdict
import json
import numpy as np

from backend.utils.comprehensive_cache import get_cache_manager, CacheConfig
from backend.config.settings import settings

logger = logging.getLogger(__name__)


class DataVolatility(Enum):
    """Data volatility classification for TTL optimization"""
    STATIC = "static"          # Changes rarely (company info, sector data)
    SLOW = "slow"             # Changes daily (EOD prices, fundamentals)
    MEDIUM = "medium"         # Changes hourly (intraday indicators)
    FAST = "fast"             # Changes by minute (real-time quotes)
    ULTRA_FAST = "ultra_fast" # Changes by second (order book, trades)


class DataImportance(Enum):
    """Data importance for cache warming priority"""
    CRITICAL = 5      # Always cached (S&P 500 stocks, major indices)
    HIGH = 4          # High priority (Russell 1000)
    MEDIUM = 3        # Medium priority (Mid-cap stocks)
    LOW = 2           # Low priority (Small-cap stocks)
    MINIMAL = 1       # Minimal priority (Micro-cap stocks)


@dataclass
class CachePolicy:
    """Cache policy configuration for different data types"""
    data_type: str
    volatility: DataVolatility
    importance: DataImportance
    l1_ttl: int
    l2_ttl: int
    l3_ttl: int
    warming_enabled: bool = True
    max_daily_api_calls: Optional[int] = None
    cost_per_call: float = 0.0


class IntelligentCachePolicyManager:
    """
    Manages intelligent caching policies based on data characteristics,
    market conditions, and API cost optimization
    """
    
    def __init__(self):
        self.policies: Dict[str, CachePolicy] = {}
        self.access_patterns: Dict[str, List[datetime]] = defaultdict(list)
        self.api_usage_tracker: Dict[str, Dict[str, int]] = defaultdict(dict)
        self.market_hours_active = False
        self._initialize_default_policies()
    
    def _initialize_default_policies(self):
        """Initialize default caching policies for different data types"""
        
        # Real-time market data (expensive API calls)
        self.policies["real_time_quote"] = CachePolicy(
            data_type="real_time_quote",
            volatility=DataVolatility.FAST,
            importance=DataImportance.HIGH,
            l1_ttl=60,      # 1 minute in L1
            l2_ttl=300,     # 5 minutes in L2
            l3_ttl=1800,    # 30 minutes in L3
            warming_enabled=True,
            max_daily_api_calls=25,  # Alpha Vantage free limit
            cost_per_call=0.10
        )
        
        # Daily price history
        self.policies["daily_prices"] = CachePolicy(
            data_type="daily_prices", 
            volatility=DataVolatility.SLOW,
            importance=DataImportance.HIGH,
            l1_ttl=3600,    # 1 hour in L1
            l2_ttl=14400,   # 4 hours in L2
            l3_ttl=86400,   # 24 hours in L3
            warming_enabled=True,
            max_daily_api_calls=25,
            cost_per_call=0.10
        )
        
        # Company fundamentals
        self.policies["company_overview"] = CachePolicy(
            data_type="company_overview",
            volatility=DataVolatility.STATIC,
            importance=DataImportance.MEDIUM,
            l1_ttl=7200,    # 2 hours in L1
            l2_ttl=43200,   # 12 hours in L2
            l3_ttl=604800,  # 7 days in L3
            warming_enabled=True,
            max_daily_api_calls=15,
            cost_per_call=0.10
        )
        
        # Technical indicators
        self.policies["technical_indicators"] = CachePolicy(
            data_type="technical_indicators",
            volatility=DataVolatility.MEDIUM,
            importance=DataImportance.MEDIUM,
            l1_ttl=1800,    # 30 minutes in L1
            l2_ttl=7200,    # 2 hours in L2
            l3_ttl=21600,   # 6 hours in L3
            warming_enabled=True,
            max_daily_api_calls=20,
            cost_per_call=0.10
        )
        
        # News and sentiment data
        self.policies["news_sentiment"] = CachePolicy(
            data_type="news_sentiment",
            volatility=DataVolatility.MEDIUM,
            importance=DataImportance.LOW,
            l1_ttl=900,     # 15 minutes in L1
            l2_ttl=3600,    # 1 hour in L2
            l3_ttl=14400,   # 4 hours in L3
            warming_enabled=True,
            max_daily_api_calls=100,  # NewsAPI free limit
            cost_per_call=0.05
        )
        
        # Analysis results (computed data)
        self.policies["analysis_result"] = CachePolicy(
            data_type="analysis_result",
            volatility=DataVolatility.SLOW,
            importance=DataImportance.HIGH,
            l1_ttl=1800,    # 30 minutes in L1
            l2_ttl=7200,    # 2 hours in L2
            l3_ttl=28800,   # 8 hours in L3
            warming_enabled=False,  # Don't warm computed results
            cost_per_call=0.0  # No external API cost
        )
        
        # User portfolio data
        self.policies["user_portfolio"] = CachePolicy(
            data_type="user_portfolio",
            volatility=DataVolatility.MEDIUM,
            importance=DataImportance.CRITICAL,
            l1_ttl=300,     # 5 minutes in L1
            l2_ttl=1800,    # 30 minutes in L2
            l3_ttl=7200,    # 2 hours in L3
            warming_enabled=False,  # Don't warm user-specific data
            cost_per_call=0.0
        )
        
        logger.info(f"Initialized {len(self.policies)} default cache policies")
    
    def get_policy(self, data_type: str) -> CachePolicy:
        """Get cache policy for a data type"""
        policy = self.policies.get(data_type)
        if not policy:
            # Return a default policy for unknown data types
            policy = CachePolicy(
                data_type=data_type,
                volatility=DataVolatility.MEDIUM,
                importance=DataImportance.LOW,
                l1_ttl=900,
                l2_ttl=3600,
                l3_ttl=14400,
                warming_enabled=False
            )
            logger.warning(f"No policy found for {data_type}, using default")
        
        return policy
    
    def adjust_ttl_for_market_hours(self, policy: CachePolicy) -> Dict[str, int]:
        """Adjust TTL based on market hours and volatility"""
        base_ttls = {
            'l1': policy.l1_ttl,
            'l2': policy.l2_ttl,
            'l3': policy.l3_ttl
        }
        
        # During market hours, reduce TTL for volatile data
        if self.market_hours_active and policy.volatility in [DataVolatility.FAST, DataVolatility.MEDIUM]:
            multiplier = 0.5  # Reduce TTL by 50% during market hours
            return {k: max(int(v * multiplier), 60) for k, v in base_ttls.items()}
        
        # During off-hours, increase TTL for all data
        elif not self.market_hours_active:
            multiplier = 2.0  # Double TTL during off-hours
            return {k: int(v * multiplier) for k, v in base_ttls.items()}
        
        return base_ttls
    
    def should_warm_cache(self, data_type: str, identifier: str) -> bool:
        """Determine if data should be warmed in cache"""
        policy = self.get_policy(data_type)
        
        if not policy.warming_enabled:
            return False
        
        # Check importance level
        if policy.importance == DataImportance.CRITICAL:
            return True
        
        # Check access patterns
        access_history = self.access_patterns.get(f"{data_type}:{identifier}", [])
        recent_accesses = [
            access for access in access_history 
            if access > datetime.utcnow() - timedelta(hours=24)
        ]
        
        # Warm if accessed frequently (more than 3 times in last 24 hours)
        return len(recent_accesses) >= 3
    
    def track_access(self, data_type: str, identifier: str):
        """Track data access for warming decisions"""
        key = f"{data_type}:{identifier}"
        self.access_patterns[key].append(datetime.utcnow())
        
        # Keep only recent accesses (last 7 days)
        cutoff = datetime.utcnow() - timedelta(days=7)
        self.access_patterns[key] = [
            access for access in self.access_patterns[key] 
            if access > cutoff
        ]
    
    def track_api_usage(self, api_provider: str, data_type: str):
        """Track API usage for cost optimization"""
        today = datetime.utcnow().strftime('%Y%m%d')
        if today not in self.api_usage_tracker[api_provider]:
            self.api_usage_tracker[api_provider][today] = 0
        
        self.api_usage_tracker[api_provider][today] += 1
        
        # Log if approaching limits
        policy = self.get_policy(data_type)
        if policy.max_daily_api_calls:
            usage = self.api_usage_tracker[api_provider][today]
            if usage >= policy.max_daily_api_calls * 0.8:  # 80% threshold
                logger.warning(
                    f"API usage for {api_provider} at {usage}/{policy.max_daily_api_calls} "
                    f"calls (80% threshold reached)"
                )
    
    def get_remaining_api_calls(self, api_provider: str) -> int:
        """Get remaining API calls for today"""
        today = datetime.utcnow().strftime('%Y%m%d')
        used_calls = self.api_usage_tracker[api_provider].get(today, 0)
        
        # Find the most restrictive limit from all data types using this provider
        max_calls = 0
        for policy in self.policies.values():
            if policy.max_daily_api_calls and api_provider.lower() in policy.data_type.lower():
                max_calls = max(max_calls, policy.max_daily_api_calls)
        
        return max(0, max_calls - used_calls)
    
    def optimize_daily_api_allocation(self) -> Dict[str, List[Dict]]:
        """
        Optimize daily API call allocation across different data types
        Returns prioritized list of API calls to make
        """
        allocation_plan = defaultdict(list)
        
        for api_provider in ['alpha_vantage', 'finnhub', 'polygon']:
            remaining_calls = self.get_remaining_api_calls(api_provider)
            
            # Get all data types that need this API provider
            candidate_calls = []
            
            for data_type, policy in self.policies.items():
                if api_provider.lower() in data_type.lower() or api_provider in data_type:
                    # Score based on importance and staleness
                    importance_score = policy.importance.value
                    volatility_score = 5 - list(DataVolatility).index(policy.volatility)
                    
                    candidate_calls.append({
                        'data_type': data_type,
                        'policy': policy,
                        'score': importance_score + volatility_score,
                        'api_provider': api_provider
                    })
            
            # Sort by score (highest first)
            candidate_calls.sort(key=lambda x: x['score'], reverse=True)
            
            # Allocate calls based on remaining quota
            allocated = 0
            for call in candidate_calls:
                if allocated >= remaining_calls:
                    break
                
                allocation_plan[api_provider].append({
                    'data_type': call['data_type'],
                    'priority': call['score'],
                    'estimated_cost': call['policy'].cost_per_call
                })
                allocated += 1
        
        return dict(allocation_plan)
    
    def update_market_hours_status(self, is_active: bool):
        """Update market hours status for TTL adjustments"""
        if self.market_hours_active != is_active:
            self.market_hours_active = is_active
            logger.info(f"Market hours status updated: {'active' if is_active else 'closed'}")


class SmartCacheWarmer:
    """
    Intelligent cache warming based on usage patterns, market conditions,
    and cost optimization
    """
    
    def __init__(self, policy_manager: IntelligentCachePolicyManager):
        self.policy_manager = policy_manager
        self.warming_schedule: Dict[str, datetime] = {}
        self.is_warming = False
    
    async def warm_critical_data(self, stock_tiers: Dict[str, List[str]]):
        """
        Warm cache with critical data based on stock tiers
        
        Args:
            stock_tiers: Dictionary with tiers like {'sp500': [...], 'russell1000': [...]}
        """
        if self.is_warming:
            logger.info("Cache warming already in progress, skipping")
            return
        
        self.is_warming = True
        cache_manager = await get_cache_manager()
        
        try:
            warming_tasks = []
            
            # S&P 500 stocks (highest priority)
            if 'sp500' in stock_tiers:
                for symbol in stock_tiers['sp500'][:50]:  # Limit to top 50 for API constraints
                    warming_tasks.append({
                        'data_type': 'daily_prices',
                        'identifier': symbol,
                        'params': {'outputsize': 'compact'},
                        'priority': DataImportance.CRITICAL.value
                    })
                    
                    warming_tasks.append({
                        'data_type': 'company_overview', 
                        'identifier': symbol,
                        'priority': DataImportance.CRITICAL.value
                    })
            
            # Popular ETFs and indices
            popular_symbols = ['SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 'VEA', 'IEFA', 'VWO']
            for symbol in popular_symbols:
                warming_tasks.extend([
                    {
                        'data_type': 'real_time_quote',
                        'identifier': symbol,
                        'priority': DataImportance.CRITICAL.value
                    },
                    {
                        'data_type': 'technical_indicators',
                        'identifier': symbol,
                        'params': {'indicator': 'sma', 'period': 20},
                        'priority': DataImportance.HIGH.value
                    }
                ])
            
            # Sort by priority and execute in batches
            warming_tasks.sort(key=lambda x: x['priority'], reverse=True)
            
            batch_size = 10  # Process in small batches to avoid overwhelming APIs
            for i in range(0, len(warming_tasks), batch_size):
                batch = warming_tasks[i:i + batch_size]
                
                # Check API limits before warming
                allocation = self.policy_manager.optimize_daily_api_allocation()
                if not any(allocation.values()):
                    logger.warning("No API calls available for cache warming")
                    break
                
                await cache_manager.warm_cache(batch, priority=1)
                
                # Rate limiting between batches
                await asyncio.sleep(2)
            
            logger.info(f"Cache warming completed for {len(warming_tasks)} items")
            
        except Exception as e:
            logger.error(f"Error during cache warming: {e}")
        
        finally:
            self.is_warming = False
    
    async def schedule_intelligent_warming(self):
        """
        Schedule cache warming based on market hours and usage patterns
        """
        while True:
            try:
                current_hour = datetime.utcnow().hour
                
                # Pre-market warming (6 AM UTC)
                if current_hour == 6:
                    logger.info("Starting pre-market cache warming")
                    await self._warm_premarket_data()
                
                # Market open warming (2:30 PM UTC = 9:30 AM EST)
                elif current_hour == 14 and datetime.utcnow().minute >= 30:
                    logger.info("Starting market open cache warming")
                    self.policy_manager.update_market_hours_status(True)
                    await self._warm_market_open_data()
                
                # Market close cooling (9 PM UTC = 4 PM EST)
                elif current_hour == 21:
                    logger.info("Market closed, updating cache policies")
                    self.policy_manager.update_market_hours_status(False)
                    await self._cleanup_expired_cache()
                
                # Sleep for 30 minutes before next check
                await asyncio.sleep(1800)
                
            except Exception as e:
                logger.error(f"Error in intelligent warming scheduler: {e}")
                await asyncio.sleep(1800)
    
    async def _warm_premarket_data(self):
        """Warm cache with pre-market essentials"""
        cache_manager = await get_cache_manager()
        
        # Load essential market data before market opens
        premarket_tasks = [
            {
                'data_type': 'company_overview',
                'identifier': 'SPY',
                'priority': DataImportance.CRITICAL.value
            },
            {
                'data_type': 'daily_prices', 
                'identifier': 'SPY',
                'params': {'outputsize': 'compact'},
                'priority': DataImportance.CRITICAL.value
            }
        ]
        
        await cache_manager.warm_cache(premarket_tasks, priority=1)
    
    async def _warm_market_open_data(self):
        """Warm cache with market open essentials"""
        cache_manager = await get_cache_manager()
        
        # Focus on real-time data during market hours
        market_open_tasks = []
        
        high_volume_symbols = ['SPY', 'AAPL', 'MSFT', 'TSLA', 'GOOGL', 'AMZN', 'NVDA']
        for symbol in high_volume_symbols:
            market_open_tasks.append({
                'data_type': 'real_time_quote',
                'identifier': symbol,
                'priority': DataImportance.CRITICAL.value
            })
        
        await cache_manager.warm_cache(market_open_tasks, priority=1)
    
    async def _cleanup_expired_cache(self):
        """Clean up expired cache entries to free memory"""
        cache_manager = await get_cache_manager()
        
        # This will be handled by the periodic cleanup in ComprehensiveCacheManager
        # Just log the event
        logger.info("Market closed - cache cleanup will be handled by periodic tasks")


# Global instances
_policy_manager: Optional[IntelligentCachePolicyManager] = None
_cache_warmer: Optional[SmartCacheWarmer] = None


def get_policy_manager() -> IntelligentCachePolicyManager:
    """Get global policy manager instance"""
    global _policy_manager
    
    if _policy_manager is None:
        _policy_manager = IntelligentCachePolicyManager()
    
    return _policy_manager


def get_cache_warmer() -> SmartCacheWarmer:
    """Get global cache warmer instance"""
    global _cache_warmer
    
    if _cache_warmer is None:
        policy_manager = get_policy_manager()
        _cache_warmer = SmartCacheWarmer(policy_manager)
    
    return _cache_warmer


async def start_intelligent_caching():
    """Start intelligent caching services"""
    try:
        # Start cache warming scheduler
        cache_warmer = get_cache_warmer()
        asyncio.create_task(cache_warmer.schedule_intelligent_warming())
        
        # Load initial stock tiers for warming
        # This would normally come from your stock classification system
        stock_tiers = {
            'sp500': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'V'],
            'popular_etfs': ['SPY', 'QQQ', 'IWM', 'VTI', 'VOO']
        }
        
        # Initial cache warming
        await cache_warmer.warm_critical_data(stock_tiers)
        
        logger.info("Intelligent caching services started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start intelligent caching services: {e}")
        raise