"""
Cost Monitoring System - Critical for staying under $50/month budget
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any
import logging
from collections import defaultdict
import json
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.unified_models import APIUsage
from backend.config.settings import settings
from backend.utils.cache import get_redis

logger = logging.getLogger(__name__)


class CostMonitor:
    """
    Real-time cost monitoring to prevent exceeding free tier limits
    """
    
    def __init__(self):
        self.redis = None
        self.api_limits = {
            'alpha_vantage': {
                'daily': 25,
                'per_minute': 5,
                'cost_per_call': 0.0  # Free tier
            },
            'polygon': {
                'per_minute': 5,
                'cost_per_call': 0.0  # Free tier
            },
            'finnhub': {
                'per_minute': 60,
                'cost_per_call': 0.0  # Free tier
            },
            'news_api': {
                'daily': 100,
                'cost_per_call': 0.0  # Free tier
            },
            'fred': {
                'daily': 1000,
                'cost_per_call': 0.0  # Free tier
            }
        }
        
        self.fallback_priority = [
            'finnhub',  # Most generous free tier
            'alpha_vantage',
            'polygon',
            'yahoo_finance'  # Unofficial but unlimited
        ]
    
    async def initialize(self):
        """Initialize Redis connection"""
        self.redis = await get_redis()
    
    async def check_api_limit(self, provider: str, endpoint: str = None) -> bool:
        """
        Check if API call is within limits
        Returns True if call is allowed, False if limit reached
        """
        if provider not in self.api_limits:
            return True
        
        limits = self.api_limits[provider]
        current_time = datetime.utcnow()
        
        # Check per-minute limit
        if 'per_minute' in limits:
            minute_key = f"api_usage:{provider}:minute:{current_time.strftime('%Y%m%d%H%M')}"
            minute_count = await self.redis.incr(minute_key)
            await self.redis.expire(minute_key, 60)
            
            if minute_count > limits['per_minute']:
                logger.warning(f"Per-minute limit reached for {provider}: {minute_count}/{limits['per_minute']}")
                return False
        
        # Check daily limit
        if 'daily' in limits:
            daily_key = f"api_usage:{provider}:daily:{current_time.strftime('%Y%m%d')}"
            daily_count = await self.redis.incr(daily_key)
            await self.redis.expire(daily_key, 86400)
            
            if daily_count > limits['daily']:
                logger.warning(f"Daily limit reached for {provider}: {daily_count}/{limits['daily']}")
                return False
        
        return True
    
    async def record_api_call(
        self,
        provider: str,
        endpoint: str,
        success: bool = True,
        response_time_ms: int = None,
        data_points: int = None,
        db_session: AsyncSession = None
    ):
        """Record API usage for monitoring and cost tracking"""
        
        # Record in database
        if db_session:
            usage = APIUsage(
                provider=provider,
                endpoint=endpoint,
                success=success,
                response_time_ms=response_time_ms,
                data_points=data_points,
                estimated_cost=self.api_limits.get(provider, {}).get('cost_per_call', 0.0)
            )
            db_session.add(usage)
            await db_session.commit()
        
        # Update real-time counters
        await self._update_usage_counters(provider)
        
        # Check thresholds
        await self._check_cost_thresholds()
    
    async def _update_usage_counters(self, provider: str):
        """Update real-time usage counters"""
        current_date = datetime.utcnow().strftime('%Y%m%d')
        
        # Increment provider-specific counter
        provider_key = f"usage_count:{provider}:{current_date}"
        await self.redis.incr(provider_key)
        await self.redis.expire(provider_key, 86400 * 7)  # Keep for 7 days
        
        # Update total daily API calls
        total_key = f"total_api_calls:{current_date}"
        total_calls = await self.redis.incr(total_key)
        await self.redis.expire(total_key, 86400 * 30)  # Keep for 30 days
        
        # Log every 100 calls
        if total_calls % 100 == 0:
            logger.info(f"Total API calls today: {total_calls}")
    
    async def _check_cost_thresholds(self):
        """Check if we're approaching cost thresholds"""
        current_month = datetime.utcnow().strftime('%Y%m')
        
        # Get monthly usage
        monthly_usage = await self.get_monthly_usage()
        
        # Calculate percentage of budget used
        budget_used_pct = (monthly_usage['estimated_cost'] / settings.MONTHLY_BUDGET_USD) * 100
        
        # Alert if approaching threshold
        if budget_used_pct >= settings.ALERT_THRESHOLD_PERCENT:
            alert_message = f"Cost alert: {budget_used_pct:.1f}% of monthly budget used"
            alert_details = {
                'budget': settings.MONTHLY_BUDGET_USD,
                'used': monthly_usage['estimated_cost'],
                'remaining': settings.MONTHLY_BUDGET_USD - monthly_usage['estimated_cost']
            }
            logger.warning(f"{alert_message} - Details: {alert_details}")
            
            # Store alert in Redis for dashboard display
            await self.redis.lpush(
                "cost_alerts",
                json.dumps({
                    'timestamp': datetime.utcnow().isoformat(),
                    'level': 'warning',
                    'message': alert_message,
                    'details': alert_details
                })
            )
            await self.redis.ltrim("cost_alerts", 0, 99)  # Keep last 100 alerts
            
            # Start using more aggressive caching
            await self._enable_cost_saving_mode()
    
    async def _enable_cost_saving_mode(self):
        """Enable aggressive cost-saving measures"""
        logger.warning("Enabling cost-saving mode due to budget constraints")
        
        # Increase cache TTL
        await self.redis.set("cost_saving_mode", "1", ex=86400)
        
        # Reduce API call frequency
        await self.redis.set("api_throttle_multiplier", "2", ex=86400)
    
    async def get_fallback_provider(self, original_provider: str, required_data: str) -> Optional[str]:
        """
        Get alternative data provider when primary is rate-limited
        """
        provider_capabilities = {
            'alpha_vantage': ['prices', 'fundamentals', 'technical'],
            'polygon': ['prices', 'news', 'fundamentals'],
            'finnhub': ['prices', 'news', 'fundamentals'],
            'yahoo_finance': ['prices', 'fundamentals'],
            'fred': ['macro', 'economic']
        }
        
        for fallback in self.fallback_priority:
            if fallback == original_provider:
                continue
                
            if required_data in provider_capabilities.get(fallback, []):
                if await self.check_api_limit(fallback):
                    logger.info(f"Using fallback provider {fallback} instead of {original_provider}")
                    return fallback
        
        return None
    
    async def get_usage_report(self, db_session: AsyncSession) -> Dict:
        """Generate comprehensive usage report"""
        current_time = datetime.utcnow()
        
        # Daily usage
        daily_stmt = select(
            APIUsage.provider,
            func.count(APIUsage.id).label('calls'),
            func.sum(APIUsage.estimated_cost).label('cost')
        ).where(
            APIUsage.timestamp >= current_time - timedelta(days=1)
        ).group_by(APIUsage.provider)
        
        daily_results = await db_session.execute(daily_stmt)
        daily_usage = {
            row.provider: {'calls': row.calls, 'cost': row.cost or 0}
            for row in daily_results
        }
        
        # Monthly usage
        monthly_stmt = select(
            APIUsage.provider,
            func.count(APIUsage.id).label('calls'),
            func.sum(APIUsage.estimated_cost).label('cost')
        ).where(
            APIUsage.timestamp >= current_time.replace(day=1)
        ).group_by(APIUsage.provider)
        
        monthly_results = await db_session.execute(monthly_stmt)
        monthly_usage = {
            row.provider: {'calls': row.calls, 'cost': row.cost or 0}
            for row in monthly_results
        }
        
        # Calculate totals
        total_daily_calls = sum(u['calls'] for u in daily_usage.values())
        total_daily_cost = sum(u['cost'] for u in daily_usage.values())
        total_monthly_calls = sum(u['calls'] for u in monthly_usage.values())
        total_monthly_cost = sum(u['cost'] for u in monthly_usage.values())
        
        return {
            'daily': {
                'by_provider': daily_usage,
                'total_calls': total_daily_calls,
                'total_cost': total_daily_cost
            },
            'monthly': {
                'by_provider': monthly_usage,
                'total_calls': total_monthly_calls,
                'total_cost': total_monthly_cost,
                'budget_remaining': settings.MONTHLY_BUDGET_USD - total_monthly_cost,
                'budget_used_pct': (total_monthly_cost / settings.MONTHLY_BUDGET_USD) * 100
            },
            'limits_remaining': await self._get_remaining_limits(),
            'cost_saving_mode': await self.redis.get("cost_saving_mode") == "1"
        }
    
    async def _get_remaining_limits(self) -> Dict:
        """Get remaining API calls for each provider"""
        remaining = {}
        current_time = datetime.utcnow()
        
        for provider, limits in self.api_limits.items():
            provider_remaining = {}
            
            if 'daily' in limits:
                daily_key = f"api_usage:{provider}:daily:{current_time.strftime('%Y%m%d')}"
                used = int(await self.redis.get(daily_key) or 0)
                provider_remaining['daily'] = max(0, limits['daily'] - used)
            
            if 'per_minute' in limits:
                minute_key = f"api_usage:{provider}:minute:{current_time.strftime('%Y%m%d%H%M')}"
                used = int(await self.redis.get(minute_key) or 0)
                provider_remaining['per_minute'] = max(0, limits['per_minute'] - used)
            
            remaining[provider] = provider_remaining
        
        return remaining
    
    async def get_monthly_usage(self) -> Dict:
        """Get current month's usage statistics"""
        current_month = datetime.utcnow().strftime('%Y%m')
        
        # Get from cache first
        cached = await self.redis.get(f"monthly_usage:{current_month}")
        if cached:
            return json.loads(cached)
        
        # Calculate from database
        # (Implementation would query APIUsage table)
        
        return {
            'estimated_cost': 0.0,  # Placeholder
            'api_calls': 0,
            'by_provider': {}
        }


class SmartDataFetcher:
    """
    Intelligent data fetching with caching and fallback strategies
    """
    
    def __init__(self, cost_monitor: CostMonitor):
        self.cost_monitor = cost_monitor
        self.cache_ttl = {
            'prices': 300,  # 5 minutes for real-time
            'fundamentals': 86400,  # 1 day
            'news': 3600,  # 1 hour
            'technical': 900  # 15 minutes
        }
    
    async def fetch_stock_data(
        self,
        ticker: str,
        data_type: str,
        provider: str = None,
        force_refresh: bool = False
    ) -> Optional[Dict]:
        """
        Fetch data with intelligent caching and fallback
        """
        cache_key = f"stock_data:{ticker}:{data_type}"
        
        # Check cache first (unless force refresh)
        if not force_refresh:
            cached = await self._get_cached_data(cache_key)
            if cached:
                return cached
        
        # Check if in cost-saving mode
        if await self._in_cost_saving_mode():
            # Use longer cache TTL
            cached = await self._get_cached_data(cache_key, extended_ttl=True)
            if cached:
                logger.info(f"Using extended cache for {ticker} {data_type} due to cost-saving mode")
                return cached
        
        # Determine provider
        if not provider:
            provider = self._get_preferred_provider(data_type)
        
        # Check API limits
        if not await self.cost_monitor.check_api_limit(provider):
            # Try fallback provider
            fallback = await self.cost_monitor.get_fallback_provider(provider, data_type)
            if fallback:
                provider = fallback
            else:
                # Return stale cache if available
                logger.warning(f"All providers rate-limited for {data_type}, using stale cache")
                return await self._get_cached_data(cache_key, allow_stale=True)
        
        # Fetch fresh data
        try:
            data = await self._fetch_from_provider(ticker, data_type, provider)
            
            # Cache the data
            if data:
                await self._cache_data(cache_key, data, data_type)
                
            # Record API usage
            await self.cost_monitor.record_api_call(
                provider=provider,
                endpoint=f"{data_type}/{ticker}",
                success=bool(data)
            )
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching {data_type} for {ticker} from {provider}: {e}")
            
            # Try to return cached data
            return await self._get_cached_data(cache_key, allow_stale=True)
    
    async def _get_cached_data(
        self,
        cache_key: str,
        extended_ttl: bool = False,
        allow_stale: bool = False
    ) -> Optional[Dict]:
        """Get data from cache with various strategies"""
        redis = await get_redis()
        
        # Regular cache
        cached = await redis.get(cache_key)
        if cached:
            return json.loads(cached)
        
        # Extended TTL cache (for cost-saving mode)
        if extended_ttl:
            extended_key = f"{cache_key}:extended"
            cached = await redis.get(extended_key)
            if cached:
                return json.loads(cached)
        
        # Stale cache (last resort)
        if allow_stale:
            stale_key = f"{cache_key}:stale"
            cached = await redis.get(stale_key)
            if cached:
                data = json.loads(cached)
                data['_stale'] = True
                return data
        
        return None
    
    async def _cache_data(self, cache_key: str, data: Dict, data_type: str):
        """Cache data with appropriate TTL"""
        redis = await get_redis()
        
        ttl = self.cache_ttl.get(data_type, 3600)
        
        # Regular cache
        await redis.set(cache_key, json.dumps(data), ex=ttl)
        
        # Extended cache (2x TTL)
        extended_key = f"{cache_key}:extended"
        await redis.set(extended_key, json.dumps(data), ex=ttl * 2)
        
        # Stale cache (7 days)
        stale_key = f"{cache_key}:stale"
        await redis.set(stale_key, json.dumps(data), ex=86400 * 7)
    
    async def _in_cost_saving_mode(self) -> bool:
        """Check if system is in cost-saving mode"""
        redis = await get_redis()
        return await redis.get("cost_saving_mode") == "1"
    
    def _get_preferred_provider(self, data_type: str) -> str:
        """Get preferred provider for data type"""
        preferences = {
            'prices': 'finnhub',  # Best free tier
            'fundamentals': 'alpha_vantage',
            'news': 'finnhub',
            'technical': 'alpha_vantage'
        }
        return preferences.get(data_type, 'finnhub')
    
    async def _fetch_from_provider(
        self,
        ticker: str,
        data_type: str,
        provider: str
    ) -> Optional[Dict]:
        """Fetch data from specific provider"""
        # Implementation would call specific provider APIs
        # This is a placeholder
        return {
            'ticker': ticker,
            'data_type': data_type,
            'provider': provider,
            'timestamp': datetime.utcnow().isoformat()
        }


# Global instance
cost_monitor = CostMonitor()
smart_fetcher = SmartDataFetcher(cost_monitor)