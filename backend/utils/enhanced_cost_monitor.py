"""
Enhanced Cost Monitoring and API Scheduling System
Implements intelligent API scheduling to maximize data coverage within free tier limits
"""

import asyncio
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import heapq
import logging
from collections import defaultdict, deque
import json
import numpy as np
from scipy.optimize import linprog

from backend.utils.cache import get_redis
from backend.config.settings import settings

logger = logging.getLogger(__name__)


class StockPriority(Enum):
    """Stock priority levels for API scheduling"""
    CRITICAL = 1    # S&P 500, high volume, user watchlist
    HIGH = 2        # Mid-cap active stocks
    MEDIUM = 3      # Small-cap with activity
    LOW = 4         # Inactive stocks
    MINIMAL = 5     # Delisted or very low activity


class APIProvider(Enum):
    """API providers with their characteristics"""
    FINNHUB = "finnhub"
    ALPHA_VANTAGE = "alpha_vantage"
    POLYGON = "polygon"
    YAHOO = "yahoo_finance"
    FRED = "fred"
    NEWS_API = "news_api"


class EnhancedCostMonitor:
    """
    Advanced cost monitoring with intelligent API scheduling and optimization
    """
    
    def __init__(self):
        self.redis = None
        self.scheduler = APIScheduler()
        self.optimizer = CostOptimizer()
        
        # Enhanced API limits with time windows
        self.api_limits = {
            APIProvider.FINNHUB: {
                'per_minute': 60,
                'daily': float('inf'),
                'monthly': float('inf'),
                'cost_per_call': 0.0,
                'best_hours': list(range(24)),  # Available 24/7
                'data_types': ['prices', 'news', 'fundamentals'],
                'latency_ms': 100,
                'reliability': 0.99
            },
            APIProvider.ALPHA_VANTAGE: {
                'per_minute': 5,
                'daily': 25,
                'monthly': 500,
                'cost_per_call': 0.0,
                'best_hours': [2, 3, 4, 5, 14, 15, 16, 17],  # Off-peak hours
                'data_types': ['prices', 'fundamentals', 'technical'],
                'latency_ms': 200,
                'reliability': 0.95
            },
            APIProvider.POLYGON: {
                'per_minute': 5,
                'daily': float('inf'),
                'monthly': float('inf'),
                'cost_per_call': 0.0,
                'best_hours': list(range(24)),
                'data_types': ['prices', 'news', 'aggregates'],
                'latency_ms': 150,
                'reliability': 0.97
            },
            APIProvider.YAHOO: {
                'per_minute': 2000,  # Unofficial, be conservative
                'daily': float('inf'),
                'monthly': float('inf'),
                'cost_per_call': 0.0,
                'best_hours': list(range(24)),
                'data_types': ['prices', 'fundamentals'],
                'latency_ms': 300,
                'reliability': 0.90  # Unofficial API
            },
            APIProvider.NEWS_API: {
                'per_minute': 100,
                'daily': 100,
                'monthly': 3000,
                'cost_per_call': 0.0,
                'best_hours': list(range(24)),
                'data_types': ['news'],
                'latency_ms': 150,
                'reliability': 0.98
            }
        }
        
        # Budget allocation by category
        self.budget_allocation = {
            'compute': 15.0,      # $15 for servers
            'storage': 10.0,      # $10 for database/storage
            'network': 5.0,       # $5 for CDN/bandwidth
            'apis': 0.0,          # $0 - stay within free tiers
            'monitoring': 5.0,    # $5 for monitoring tools
            'buffer': 15.0        # $15 safety buffer
        }
        
        # Track API usage patterns
        self.usage_patterns = defaultdict(lambda: deque(maxlen=168))  # 7 days of hourly data
        
    async def initialize(self):
        """Initialize connections and load historical data"""
        self.redis = await get_redis()
        await self.scheduler.initialize()
        await self.optimizer.initialize()
        await self._load_usage_patterns()
    
    async def _load_usage_patterns(self):
        """Load historical usage patterns for prediction"""
        for provider in APIProvider:
            pattern_key = f"usage_pattern:{provider.value}"
            pattern_data = await self.redis.get(pattern_key)
            if pattern_data:
                self.usage_patterns[provider] = deque(
                    json.loads(pattern_data),
                    maxlen=168
                )
    
    async def get_optimal_schedule(
        self,
        stocks: List[Tuple[str, StockPriority]],
        time_window_hours: int = 24
    ) -> Dict[str, List[Dict]]:
        """
        Generate optimal API call schedule for stocks
        
        Args:
            stocks: List of (symbol, priority) tuples
            time_window_hours: Planning horizon
            
        Returns:
            Schedule organized by hour with API assignments
        """
        schedule = defaultdict(list)
        current_time = datetime.now()
        
        # Group stocks by priority
        priority_groups = defaultdict(list)
        for symbol, priority in stocks:
            priority_groups[priority].append(symbol)
        
        # Process each priority level
        for priority in sorted(priority_groups.keys(), key=lambda x: x.value):
            stocks_list = priority_groups[priority]
            
            # Determine update frequency based on priority
            update_frequency = self._get_update_frequency(priority)
            
            # Assign API providers based on availability and suitability
            for hour in range(time_window_hours):
                target_time = current_time + timedelta(hours=hour)
                hour_key = target_time.strftime('%Y-%m-%d %H:00')
                
                # Check if update needed this hour
                if hour % update_frequency == 0:
                    # Get available API capacity for this hour
                    available_apis = await self._get_available_apis(target_time)
                    
                    # Distribute stocks across available APIs
                    assignments = self._distribute_stocks(
                        stocks_list,
                        available_apis,
                        priority
                    )
                    
                    for api, assigned_stocks in assignments.items():
                        for stock in assigned_stocks:
                            schedule[hour_key].append({
                                'symbol': stock,
                                'api': api.value,
                                'priority': priority.value,
                                'data_types': self._get_required_data_types(priority)
                            })
        
        return dict(schedule)
    
    def _get_update_frequency(self, priority: StockPriority) -> int:
        """Get update frequency in hours based on priority"""
        frequencies = {
            StockPriority.CRITICAL: 1,     # Every hour
            StockPriority.HIGH: 4,          # Every 4 hours
            StockPriority.MEDIUM: 8,        # Every 8 hours
            StockPriority.LOW: 24,          # Daily
            StockPriority.MINIMAL: 168      # Weekly
        }
        return frequencies.get(priority, 24)
    
    def _get_required_data_types(self, priority: StockPriority) -> List[str]:
        """Get required data types based on priority"""
        if priority == StockPriority.CRITICAL:
            return ['prices', 'news', 'fundamentals', 'technical']
        elif priority == StockPriority.HIGH:
            return ['prices', 'fundamentals', 'technical']
        elif priority == StockPriority.MEDIUM:
            return ['prices', 'fundamentals']
        else:
            return ['prices']
    
    async def _get_available_apis(self, target_time: datetime) -> Dict[APIProvider, int]:
        """Get available API capacity for a given time"""
        available = {}
        hour = target_time.hour
        
        for provider, limits in self.api_limits.items():
            # Check if this is a good hour for the API
            if hour not in limits['best_hours']:
                continue
            
            # Calculate remaining capacity
            used_today = await self._get_usage_count(provider, 'daily', target_time)
            daily_remaining = limits['daily'] - used_today if limits['daily'] != float('inf') else 1000
            
            # Estimate capacity for this hour based on per-minute limit
            hourly_capacity = min(limits['per_minute'] * 60, daily_remaining)
            
            # Apply reliability factor
            effective_capacity = int(hourly_capacity * limits['reliability'])
            
            if effective_capacity > 0:
                available[provider] = effective_capacity
        
        return available
    
    def _distribute_stocks(
        self,
        stocks: List[str],
        available_apis: Dict[APIProvider, int],
        priority: StockPriority
    ) -> Dict[APIProvider, List[str]]:
        """Distribute stocks across available APIs optimally"""
        assignments = defaultdict(list)
        
        if not available_apis or not stocks:
            return assignments
        
        # Sort APIs by suitability for this priority
        sorted_apis = sorted(
            available_apis.items(),
            key=lambda x: (
                -x[1],  # Higher capacity first
                self.api_limits[x[0]]['latency_ms']  # Lower latency preferred
            )
        )
        
        # Round-robin assignment with capacity constraints
        stock_idx = 0
        while stock_idx < len(stocks):
            for api, capacity in sorted_apis:
                if stock_idx >= len(stocks):
                    break
                    
                if len(assignments[api]) < capacity:
                    assignments[api].append(stocks[stock_idx])
                    stock_idx += 1
            
            # Break if no more capacity
            if all(len(assignments[api]) >= capacity for api, capacity in sorted_apis):
                break
        
        return assignments
    
    async def _get_usage_count(
        self,
        provider: APIProvider,
        period: str,
        target_time: datetime
    ) -> int:
        """Get usage count for a provider in a given period"""
        if period == 'daily':
            key = f"api_usage:{provider.value}:daily:{target_time.strftime('%Y%m%d')}"
        elif period == 'minute':
            key = f"api_usage:{provider.value}:minute:{target_time.strftime('%Y%m%d%H%M')}"
        else:
            return 0
        
        count = await self.redis.get(key)
        return int(count) if count else 0
    
    async def predict_monthly_cost(self) -> Dict[str, float]:
        """Predict monthly cost based on current usage patterns"""
        predictions = {}
        current_day = datetime.now().day
        days_in_month = 30
        
        # Calculate current run rate
        for category, budget in self.budget_allocation.items():
            if category == 'apis':
                # APIs should stay at $0
                predictions[category] = 0.0
            elif category in ['compute', 'storage']:
                # Linear projection based on current usage
                current_usage = await self._get_category_usage(category)
                daily_rate = current_usage / current_day if current_day > 0 else 0
                predictions[category] = min(daily_rate * days_in_month, budget)
            else:
                # Fixed costs
                predictions[category] = budget
        
        predictions['total'] = sum(predictions.values())
        predictions['remaining'] = 50.0 - predictions['total']
        
        return predictions
    
    async def _get_category_usage(self, category: str) -> float:
        """Get current month's usage for a category"""
        key = f"cost_usage:{category}:{datetime.now().strftime('%Y%m')}"
        usage = await self.redis.get(key)
        return float(usage) if usage else 0.0
    
    async def enable_emergency_mode(self):
        """Enable emergency cost-saving mode when approaching limits"""
        logger.critical("EMERGENCY MODE: Approaching budget limit!")
        
        # Set emergency flag
        await self.redis.set("emergency_mode", "1", ex=86400)
        
        # Aggressive caching - cache everything for 24 hours
        await self.redis.set("cache_ttl_multiplier", "10", ex=86400)
        
        # Limit to critical stocks only
        await self.redis.set("stock_filter", "critical_only", ex=86400)
        
        # Disable non-essential features
        await self.redis.set("features_disabled", json.dumps([
            "sentiment_analysis",
            "alternative_data",
            "realtime_updates"
        ]), ex=86400)
        
        # Send alert
        await self._send_cost_alert("Emergency mode activated - budget limit approaching")
    
    async def _send_cost_alert(self, message: str):
        """Send cost alert to administrators"""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'level': 'critical',
            'message': message,
            'current_cost': await self._get_category_usage('total'),
            'budget': 50.0
        }
        
        # Store in Redis for dashboard
        await self.redis.lpush("cost_alerts", json.dumps(alert))
        await self.redis.ltrim("cost_alerts", 0, 99)
        
        logger.critical(f"Cost Alert: {alert}")


class APIScheduler:
    """
    Intelligent API call scheduler with rate limiting and optimization
    """
    
    def __init__(self):
        self.redis = None
        self.call_queue = []  # Priority queue for API calls
        self.rate_limiters = {}
        
    async def initialize(self):
        """Initialize scheduler"""
        self.redis = await get_redis()
        
        # Initialize rate limiters for each provider
        for provider in APIProvider:
            self.rate_limiters[provider] = RateLimiter(provider)
    
    async def schedule_call(
        self,
        provider: APIProvider,
        endpoint: str,
        params: Dict,
        priority: int = 5,
        retry_count: int = 0
    ) -> Optional[str]:
        """
        Schedule an API call with priority queuing
        
        Returns:
            Call ID for tracking
        """
        call_id = f"{provider.value}:{endpoint}:{datetime.now().timestamp()}"
        
        # Check if we can make the call immediately
        if await self.rate_limiters[provider].can_call():
            # Execute immediately
            return await self._execute_call(call_id, provider, endpoint, params)
        
        # Add to queue
        heapq.heappush(self.call_queue, (
            priority,
            datetime.now().timestamp(),
            {
                'id': call_id,
                'provider': provider,
                'endpoint': endpoint,
                'params': params,
                'retry_count': retry_count
            }
        ))
        
        # Process queue
        await self._process_queue()
        
        return call_id
    
    async def _execute_call(
        self,
        call_id: str,
        provider: APIProvider,
        endpoint: str,
        params: Dict
    ) -> str:
        """Execute an API call"""
        # Record the call
        await self.rate_limiters[provider].record_call()
        
        # Store call metadata
        await self.redis.hset(
            f"api_call:{call_id}",
            mapping={
                'provider': provider.value,
                'endpoint': endpoint,
                'params': json.dumps(params),
                'timestamp': datetime.now().isoformat(),
                'status': 'executing'
            }
        )
        
        return call_id
    
    async def _process_queue(self):
        """Process queued API calls"""
        processed = []
        
        while self.call_queue:
            # Peek at highest priority call
            priority, timestamp, call_data = self.call_queue[0]
            provider = call_data['provider']
            
            # Check if we can make this call
            if await self.rate_limiters[provider].can_call():
                # Remove from queue and execute
                heapq.heappop(self.call_queue)
                await self._execute_call(
                    call_data['id'],
                    provider,
                    call_data['endpoint'],
                    call_data['params']
                )
                processed.append(call_data['id'])
            else:
                # Can't process any more calls right now
                break
        
        return processed
    
    async def get_queue_status(self) -> Dict:
        """Get current queue status"""
        queue_by_provider = defaultdict(int)
        queue_by_priority = defaultdict(int)
        
        for priority, _, call_data in self.call_queue:
            queue_by_provider[call_data['provider'].value] += 1
            queue_by_priority[priority] += 1
        
        return {
            'total_queued': len(self.call_queue),
            'by_provider': dict(queue_by_provider),
            'by_priority': dict(queue_by_priority),
            'oldest_call_age': (
                datetime.now().timestamp() - self.call_queue[0][1]
                if self.call_queue else 0
            )
        }


class RateLimiter:
    """
    Token bucket rate limiter for API calls
    """
    
    def __init__(self, provider: APIProvider):
        self.provider = provider
        self.redis = None
        self.limits = None
        
    async def initialize(self):
        """Initialize rate limiter"""
        self.redis = await get_redis()
        # Load limits from EnhancedCostMonitor
        
    async def can_call(self) -> bool:
        """Check if a call can be made now"""
        current_time = datetime.now()
        
        # Check per-minute limit
        minute_key = f"rate:{self.provider.value}:minute:{current_time.strftime('%Y%m%d%H%M')}"
        minute_count = await self.redis.get(minute_key)
        
        if minute_count and int(minute_count) >= self.limits.get('per_minute', float('inf')):
            return False
        
        # Check daily limit
        daily_key = f"rate:{self.provider.value}:daily:{current_time.strftime('%Y%m%d')}"
        daily_count = await self.redis.get(daily_key)
        
        if daily_count and int(daily_count) >= self.limits.get('daily', float('inf')):
            return False
        
        return True
    
    async def record_call(self):
        """Record that a call was made"""
        current_time = datetime.now()
        
        # Increment per-minute counter
        minute_key = f"rate:{self.provider.value}:minute:{current_time.strftime('%Y%m%d%H%M')}"
        await self.redis.incr(minute_key)
        await self.redis.expire(minute_key, 60)
        
        # Increment daily counter
        daily_key = f"rate:{self.provider.value}:daily:{current_time.strftime('%Y%m%d')}"
        await self.redis.incr(daily_key)
        await self.redis.expire(daily_key, 86400)


class CostOptimizer:
    """
    Optimize resource allocation to minimize costs while maximizing data coverage
    """
    
    def __init__(self):
        self.redis = None
        
    async def initialize(self):
        """Initialize optimizer"""
        self.redis = await get_redis()
    
    def optimize_api_allocation(
        self,
        stocks: List[Tuple[str, StockPriority]],
        api_limits: Dict[APIProvider, Dict],
        time_window_hours: int = 24
    ) -> Dict[str, List[str]]:
        """
        Use linear programming to optimize API allocation
        
        Maximize: Data coverage (weighted by stock priority)
        Subject to: API rate limits and free tier constraints
        """
        # Define the optimization problem
        num_stocks = len(stocks)
        num_apis = len(api_limits)
        num_hours = time_window_hours
        
        # Decision variables: x[stock][api][hour] = 1 if stock uses api at hour
        # Flatten to 1D array for linprog
        num_vars = num_stocks * num_apis * num_hours
        
        # Objective: Maximize coverage (negative for minimization in linprog)
        c = []
        for stock, priority in stocks:
            weight = 6 - priority.value  # Higher priority = higher weight
            for api in api_limits:
                for hour in range(num_hours):
                    c.append(-weight)  # Negative because linprog minimizes
        
        # Constraints
        A_ub = []
        b_ub = []
        
        # API rate limit constraints
        for api_idx, (api, limits) in enumerate(api_limits.items()):
            # Per-hour constraint
            for hour in range(num_hours):
                constraint = [0] * num_vars
                for stock_idx in range(num_stocks):
                    var_idx = stock_idx * num_apis * num_hours + api_idx * num_hours + hour
                    constraint[var_idx] = 1
                
                A_ub.append(constraint)
                b_ub.append(limits['per_minute'] * 60)  # Hourly capacity
        
        # Each stock gets updated at least once per day (for critical stocks)
        A_eq = []
        b_eq = []
        
        for stock_idx, (stock, priority) in enumerate(stocks):
            if priority == StockPriority.CRITICAL:
                constraint = [0] * num_vars
                for api_idx in range(num_apis):
                    for hour in range(num_hours):
                        var_idx = stock_idx * num_apis * num_hours + api_idx * num_hours + hour
                        constraint[var_idx] = 1
                
                A_eq.append(constraint)
                b_eq.append(24)  # At least 24 updates per day for critical stocks
        
        # Solve the optimization problem
        if A_eq:  # Only if we have equality constraints
            result = linprog(
                c=np.array(c),
                A_ub=np.array(A_ub) if A_ub else None,
                b_ub=np.array(b_ub) if b_ub else None,
                A_eq=np.array(A_eq),
                b_eq=np.array(b_eq),
                bounds=(0, 1),  # Binary variables relaxed to continuous
                method='highs'
            )
        else:
            result = linprog(
                c=np.array(c),
                A_ub=np.array(A_ub) if A_ub else None,
                b_ub=np.array(b_ub) if b_ub else None,
                bounds=(0, 1),
                method='highs'
            )
        
        # Parse results
        allocation = defaultdict(list)
        
        if result.success:
            x = result.x.reshape((num_stocks, num_apis, num_hours))
            
            for stock_idx, (stock, _) in enumerate(stocks):
                for api_idx, api in enumerate(api_limits):
                    for hour in range(num_hours):
                        if x[stock_idx, api_idx, hour] > 0.5:  # Threshold for binary decision
                            hour_key = f"hour_{hour:02d}"
                            if hour_key not in allocation:
                                allocation[hour_key] = []
                            allocation[hour_key].append({
                                'stock': stock,
                                'api': api.value
                            })
        
        return dict(allocation)
    
    async def recommend_cost_reductions(self) -> List[Dict]:
        """Analyze usage and recommend cost reduction strategies"""
        recommendations = []
        
        # Analyze API usage efficiency
        api_efficiency = await self._analyze_api_efficiency()
        if api_efficiency['wasted_calls'] > 100:
            recommendations.append({
                'category': 'API Usage',
                'issue': f"{api_efficiency['wasted_calls']} redundant API calls detected",
                'recommendation': 'Increase cache TTL for low-priority stocks',
                'potential_savings': api_efficiency['wasted_calls'] * 0.001  # Estimated
            })
        
        # Check cache hit rates
        cache_stats = await self._get_cache_statistics()
        if cache_stats['hit_rate'] < 0.7:
            recommendations.append({
                'category': 'Caching',
                'issue': f"Low cache hit rate: {cache_stats['hit_rate']:.1%}",
                'recommendation': 'Optimize cache key strategy and increase TTL',
                'potential_savings': 5.0  # Estimated monthly savings
            })
        
        # Analyze compute usage
        compute_usage = await self._analyze_compute_usage()
        if compute_usage['idle_percentage'] > 30:
            recommendations.append({
                'category': 'Compute',
                'issue': f"{compute_usage['idle_percentage']:.0f}% idle time detected",
                'recommendation': 'Implement auto-scaling or use serverless functions',
                'potential_savings': compute_usage['idle_percentage'] * 0.15  # $15 * idle%
            })
        
        return recommendations
    
    async def _analyze_api_efficiency(self) -> Dict:
        """Analyze API usage efficiency"""
        # Placeholder implementation
        return {
            'total_calls': 10000,
            'wasted_calls': 150,
            'duplicate_calls': 50,
            'failed_calls': 25
        }
    
    async def _get_cache_statistics(self) -> Dict:
        """Get cache performance statistics"""
        # Placeholder implementation
        return {
            'hit_rate': 0.75,
            'miss_rate': 0.25,
            'total_requests': 50000,
            'cache_size_mb': 256
        }
    
    async def _analyze_compute_usage(self) -> Dict:
        """Analyze compute resource usage"""
        # Placeholder implementation
        return {
            'avg_cpu_usage': 45,
            'peak_cpu_usage': 85,
            'idle_percentage': 25,
            'memory_usage_gb': 8
        }


# Utility functions
async def get_cost_summary() -> Dict:
    """Get comprehensive cost summary"""
    monitor = EnhancedCostMonitor()
    await monitor.initialize()
    
    return {
        'current_month': await monitor.predict_monthly_cost(),
        'recommendations': await monitor.optimizer.recommend_cost_reductions(),
        'api_schedule': await monitor.scheduler.get_queue_status(),
        'emergency_mode': await monitor.redis.get("emergency_mode") == "1"
    }


async def optimize_for_budget(budget: float = 50.0) -> Dict:
    """Optimize system configuration for given budget"""
    monitor = EnhancedCostMonitor()
    await monitor.initialize()
    
    if budget < 30:
        # Minimal mode
        config = {
            'mode': 'minimal',
            'stocks_limit': 500,
            'update_frequency': 'daily',
            'features_enabled': ['prices'],
            'cache_ttl_hours': 24
        }
    elif budget < 50:
        # Standard mode
        config = {
            'mode': 'standard',
            'stocks_limit': 3000,
            'update_frequency': 'mixed',
            'features_enabled': ['prices', 'fundamentals', 'technical'],
            'cache_ttl_hours': 4
        }
    else:
        # Premium mode
        config = {
            'mode': 'premium',
            'stocks_limit': 6000,
            'update_frequency': 'realtime',
            'features_enabled': ['all'],
            'cache_ttl_hours': 1
        }
    
    # Apply configuration
    await monitor.redis.set("system_config", json.dumps(config))
    
    return config