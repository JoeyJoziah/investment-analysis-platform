"""
Persistent Cost Monitor with Database Storage
Tracks API usage and costs with database persistence for accurate budget management.
"""

import logging
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum
import json
import asyncio
from collections import defaultdict

from sqlalchemy import create_engine, func, and_
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError

from backend.models.tables import CostMetrics
from backend.config.database import get_db_session
from backend.utils.cache import enhanced_cache
import redis.asyncio as aioredis

logger = logging.getLogger(__name__)


class PersistentCostMonitor:
    """
    Enhanced cost monitor with database persistence.
    Ensures accurate tracking even after system restarts.
    """
    
    def __init__(self) -> None:
        """Initialize persistent cost monitor with database backend."""
        self.cache = enhanced_cache
        self.monthly_budget_usd = 50.0
        self.emergency_threshold = 0.95  # Enter emergency mode at 95% budget
        
        # API cost configuration (per call)
        self.api_costs = {
            'alpha_vantage': 0.0,  # Free tier: 25 calls/day
            'finnhub': 0.0,        # Free tier: 60 calls/minute
            'polygon': 0.0,        # Free tier: 5 calls/minute
            'fmp': 0.0,            # Free tier: 250 calls/day
            'newsapi': 0.0,        # Free tier: 500 calls/day
            'sec': 0.0,            # Free: unlimited (rate limited)
        }
        
        # API limits
        self.api_limits = {
            'alpha_vantage': {'daily': 25, 'per_minute': 5},
            'finnhub': {'daily': float('inf'), 'per_minute': 60},
            'polygon': {'daily': float('inf'), 'per_minute': 5},
            'fmp': {'daily': 250, 'per_minute': float('inf')},
            'newsapi': {'daily': 500, 'per_minute': float('inf')},
            'sec': {'daily': float('inf'), 'per_minute': 10},
        }
        
        # Infrastructure costs (estimated monthly)
        self.infrastructure_costs = {
            'compute': 15.0,      # Kubernetes pods
            'storage': 10.0,      # PostgreSQL + Redis
            'network': 5.0,       # Bandwidth
            'monitoring': 0.0,    # Prometheus/Grafana (self-hosted)
        }
    
    async def track_api_call(
        self,
        provider: str,
        success: bool = True,
        data_points: int = 1,
        latency_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Track an API call with persistence to database.
        
        Args:
            provider: API provider name
            success: Whether the call was successful
            data_points: Number of data points fetched
            latency_ms: API response latency in milliseconds
            metadata: Additional metadata about the call
        
        Returns:
            bool: True if tracking successful, False if budget exceeded
        """
        try:
            # Check if we're in emergency mode first
            if await self.is_in_emergency_mode():
                logger.warning(f"Emergency mode active - blocking API call to {provider}")
                return False
            
            # Track in cache for real-time metrics
            cache_key = f"api_usage:{provider}:{datetime.now().date()}"
            await self.cache.increment(cache_key)
            
            # Update database
            with get_db_session() as session:
                today = datetime.now().date()
                
                # Get or create today's metrics
                metrics = session.query(CostMetrics).filter(
                    and_(
                        CostMetrics.date == today,
                        CostMetrics.provider == provider
                    )
                ).first()
                
                if not metrics:
                    metrics = CostMetrics(
                        date=today,
                        provider=provider,
                        api_calls=0,
                        successful_calls=0,
                        failed_calls=0,
                        cached_hits=0,
                        estimated_cost=0.0,
                        data_points_fetched=0,
                        metadata={}
                    )
                    session.add(metrics)
                
                # Update metrics
                metrics.api_calls += 1
                if success:
                    metrics.successful_calls += 1
                else:
                    metrics.failed_calls += 1
                
                metrics.data_points_fetched += data_points
                
                # Update latency (running average)
                if latency_ms:
                    if metrics.average_latency_ms:
                        metrics.average_latency_ms = (
                            metrics.average_latency_ms * (metrics.api_calls - 1) + latency_ms
                        ) / metrics.api_calls
                    else:
                        metrics.average_latency_ms = latency_ms
                
                # Calculate error rate
                if metrics.api_calls > 0:
                    metrics.error_rate = metrics.failed_calls / metrics.api_calls
                
                # Update cost estimate
                metrics.estimated_cost = self._calculate_provider_cost(
                    provider,
                    metrics.api_calls
                )
                
                # Store metadata if provided
                if metadata:
                    current_metadata = metrics.meta_data or {}
                    current_metadata.update(metadata)
                    metrics.meta_data = current_metadata
                
                session.commit()
                
                # Check if we're approaching limits
                await self._check_limits(provider, metrics.api_calls)
                
                return True
                
        except SQLAlchemyError as e:
            logger.error(f"Database error tracking API call: {e}")
            return False
        except Exception as e:
            logger.error(f"Error tracking API call: {e}")
            return False
    
    async def track_cache_hit(self, provider: str, data_type: Optional[str] = None) -> None:
        """Track a cache hit to show cost savings."""
        try:
            with get_db_session() as session:
                today = datetime.now().date()
                
                metrics = session.query(CostMetrics).filter(
                    and_(
                        CostMetrics.date == today,
                        CostMetrics.provider == provider
                    )
                ).first()
                
                if metrics:
                    metrics.cached_hits += 1
                    session.commit()
                
                # Also track in cache for real-time stats
                cache_key = f"cache_hits:{provider}:{today}"
                await self.cache.increment(cache_key)
                
        except Exception as e:
            logger.error(f"Error tracking cache hit: {e}")
    
    async def get_usage_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive usage metrics from database.
        
        Returns:
            Dict containing usage statistics and cost estimates
        """
        try:
            with get_db_session() as session:
                today = datetime.now().date()
                month_start = date(today.year, today.month, 1)
                
                # Get today's metrics
                today_metrics = session.query(
                    CostMetrics.provider,
                    func.sum(CostMetrics.api_calls).label('calls'),
                    func.sum(CostMetrics.successful_calls).label('successful'),
                    func.sum(CostMetrics.failed_calls).label('failed'),
                    func.sum(CostMetrics.cached_hits).label('cached'),
                    func.sum(CostMetrics.estimated_cost).label('cost'),
                    func.avg(CostMetrics.average_latency_ms).label('avg_latency')
                ).filter(
                    CostMetrics.date == today
                ).group_by(CostMetrics.provider).all()
                
                # Get monthly metrics
                monthly_metrics = session.query(
                    CostMetrics.provider,
                    func.sum(CostMetrics.api_calls).label('calls'),
                    func.sum(CostMetrics.estimated_cost).label('cost'),
                    func.sum(CostMetrics.data_points_fetched).label('data_points')
                ).filter(
                    CostMetrics.date >= month_start
                ).group_by(CostMetrics.provider).all()
                
                # Build response
                providers = {}
                
                # Process today's metrics
                for metric in today_metrics:
                    providers[metric.provider] = {
                        'calls_today': metric.calls or 0,
                        'successful_today': metric.successful or 0,
                        'failed_today': metric.failed or 0,
                        'cached_today': metric.cached or 0,
                        'cost_today': float(metric.cost or 0),
                        'avg_latency_ms': float(metric.avg_latency or 0),
                        'error_rate': (metric.failed / metric.calls * 100) if metric.calls else 0
                    }
                
                # Add monthly totals
                monthly_api_cost = 0.0
                for metric in monthly_metrics:
                    if metric.provider in providers:
                        providers[metric.provider].update({
                            'calls_month': metric.calls or 0,
                            'cost_month': float(metric.cost or 0),
                            'data_points_month': metric.data_points or 0
                        })
                    monthly_api_cost += float(metric.cost or 0)
                
                # Calculate total costs
                monthly_infrastructure = sum(self.infrastructure_costs.values())
                total_monthly_cost = monthly_api_cost + monthly_infrastructure
                
                # Check cache efficiency
                total_calls = sum(m.calls or 0 for m in today_metrics)
                total_cached = sum(m.cached or 0 for m in today_metrics)
                cache_hit_rate = (total_cached / (total_calls + total_cached) * 100) if (total_calls + total_cached) > 0 else 0
                
                return {
                    'providers': providers,
                    'summary': {
                        'total_api_calls_today': total_calls,
                        'total_cache_hits_today': total_cached,
                        'cache_hit_rate': round(cache_hit_rate, 2),
                        'estimated_api_cost_month': round(monthly_api_cost, 2),
                        'infrastructure_cost_month': round(monthly_infrastructure, 2),
                        'total_estimated_cost_month': round(total_monthly_cost, 2),
                        'budget_remaining': round(self.monthly_budget_usd - total_monthly_cost, 2),
                        'budget_usage_percent': round((total_monthly_cost / self.monthly_budget_usd) * 100, 2),
                        'in_emergency_mode': total_monthly_cost >= (self.monthly_budget_usd * self.emergency_threshold)
                    }
                }
                
        except Exception as e:
            logger.error(f"Error getting usage metrics: {e}")
            return {
                'providers': {},
                'summary': {
                    'error': str(e)
                }
            }
    
    async def is_in_emergency_mode(self) -> bool:
        """Check if we're in emergency mode due to budget constraints."""
        metrics = await self.get_usage_metrics()
        return metrics['summary'].get('in_emergency_mode', False)
    
    async def get_provider_availability(self, provider: str) -> Dict[str, Any]:
        """
        Check if a provider is available based on rate limits and budget.
        
        Args:
            provider: API provider name
        
        Returns:
            Dict with availability status and limits
        """
        try:
            if provider not in self.api_limits:
                return {'available': False, 'reason': 'Unknown provider'}
            
            # Check emergency mode
            if await self.is_in_emergency_mode():
                return {'available': False, 'reason': 'Emergency mode - budget limit reached'}
            
            limits = self.api_limits[provider]
            today = datetime.now().date()
            
            with get_db_session() as session:
                # Get today's usage
                metrics = session.query(CostMetrics).filter(
                    and_(
                        CostMetrics.date == today,
                        CostMetrics.provider == provider
                    )
                ).first()
                
                calls_today = metrics.api_calls if metrics else 0
                
                # Check daily limit
                if limits['daily'] != float('inf') and calls_today >= limits['daily']:
                    return {
                        'available': False,
                        'reason': f"Daily limit reached ({calls_today}/{limits['daily']})"
                    }
                
                # Check rate limit (last minute)
                minute_key = f"rate_limit:{provider}:{datetime.now().strftime('%Y%m%d%H%M')}"
                calls_minute = await self.cache.get(minute_key) or 0
                
                if limits['per_minute'] != float('inf') and int(calls_minute) >= limits['per_minute']:
                    return {
                        'available': False,
                        'reason': f"Rate limit reached ({calls_minute}/{limits['per_minute']} per minute)"
                    }
                
                return {
                    'available': True,
                    'calls_today': calls_today,
                    'daily_limit': limits['daily'],
                    'calls_remaining_today': limits['daily'] - calls_today if limits['daily'] != float('inf') else 'unlimited',
                    'rate_limit': limits['per_minute'],
                    'calls_this_minute': int(calls_minute)
                }
                
        except Exception as e:
            logger.error(f"Error checking provider availability: {e}")
            return {'available': False, 'reason': f"Error: {str(e)}"}
    
    async def _check_limits(self, provider: str, calls_today: int) -> None:
        """Check and alert if approaching limits."""
        if provider not in self.api_limits:
            return
        
        limits = self.api_limits[provider]
        
        # Check if approaching daily limit
        if limits['daily'] != float('inf'):
            usage_percent = (calls_today / limits['daily']) * 100
            
            if usage_percent >= 90:
                logger.warning(
                    f"API limit warning: {provider} at {usage_percent:.1f}% "
                    f"of daily limit ({calls_today}/{limits['daily']})"
                )
            elif usage_percent >= 75:
                logger.info(
                    f"API usage notice: {provider} at {usage_percent:.1f}% "
                    f"of daily limit ({calls_today}/{limits['daily']})"
                )
        
        # Track rate limit
        minute_key = f"rate_limit:{provider}:{datetime.now().strftime('%Y%m%d%H%M')}"
        await self.cache.increment(minute_key)
        await self.cache.expire(minute_key, 60)  # Expire after 1 minute
    
    def _calculate_provider_cost(self, provider: str, calls: int) -> float:
        """Calculate estimated cost for API calls."""
        base_cost = self.api_costs.get(provider, 0.0)
        
        # Most providers are free tier, but calculate overage if exceeded
        if provider == 'alpha_vantage' and calls > 25:
            # Hypothetical overage cost
            return (calls - 25) * 0.01
        elif provider == 'finnhub' and calls > 10000:  # Monthly limit estimate
            return (calls - 10000) * 0.001
        elif provider == 'polygon' and calls > 1000:  # Monthly limit estimate
            return (calls - 1000) * 0.002
        
        return base_cost * calls
    
    async def generate_cost_report(self, start_date: Optional[date] = None, end_date: Optional[date] = None) -> Dict[str, Any]:
        """
        Generate a detailed cost report for a date range.
        
        Args:
            start_date: Start date for report (default: month start)
            end_date: End date for report (default: today)
        
        Returns:
            Dict containing detailed cost breakdown
        """
        try:
            if not end_date:
                end_date = datetime.now().date()
            if not start_date:
                start_date = date(end_date.year, end_date.month, 1)
            
            with get_db_session() as session:
                # Get daily breakdown
                daily_metrics = session.query(
                    CostMetrics.date,
                    CostMetrics.provider,
                    CostMetrics.api_calls,
                    CostMetrics.cached_hits,
                    CostMetrics.estimated_cost,
                    CostMetrics.error_rate
                ).filter(
                    and_(
                        CostMetrics.date >= start_date,
                        CostMetrics.date <= end_date
                    )
                ).order_by(CostMetrics.date, CostMetrics.provider).all()
                
                # Organize by date
                report = {
                    'period': {
                        'start': start_date.isoformat(),
                        'end': end_date.isoformat()
                    },
                    'daily_breakdown': defaultdict(lambda: {
                        'providers': {},
                        'total_calls': 0,
                        'total_cached': 0,
                        'total_cost': 0.0
                    })
                }
                
                for metric in daily_metrics:
                    day_key = metric.date.isoformat()
                    report['daily_breakdown'][day_key]['providers'][metric.provider] = {
                        'calls': metric.api_calls,
                        'cached': metric.cached_hits,
                        'cost': float(metric.estimated_cost or 0),
                        'error_rate': float(metric.error_rate or 0)
                    }
                    report['daily_breakdown'][day_key]['total_calls'] += metric.api_calls
                    report['daily_breakdown'][day_key]['total_cached'] += metric.cached_hits
                    report['daily_breakdown'][day_key]['total_cost'] += float(metric.estimated_cost or 0)
                
                # Calculate totals
                total_api_calls = sum(m.api_calls for m in daily_metrics)
                total_cached = sum(m.cached_hits for m in daily_metrics)
                total_api_cost = sum(float(m.estimated_cost or 0) for m in daily_metrics)
                
                # Add infrastructure costs (prorated for period)
                days_in_period = (end_date - start_date).days + 1
                days_in_month = 30  # Approximate
                infrastructure_cost = sum(self.infrastructure_costs.values()) * (days_in_period / days_in_month)
                
                report['summary'] = {
                    'total_api_calls': total_api_calls,
                    'total_cache_hits': total_cached,
                    'cache_efficiency': round((total_cached / (total_api_calls + total_cached) * 100), 2) if (total_api_calls + total_cached) > 0 else 0,
                    'total_api_cost': round(total_api_cost, 2),
                    'infrastructure_cost': round(infrastructure_cost, 2),
                    'total_cost': round(total_api_cost + infrastructure_cost, 2),
                    'daily_average_cost': round((total_api_cost + infrastructure_cost) / days_in_period, 2),
                    'projected_monthly_cost': round((total_api_cost + infrastructure_cost) / days_in_period * 30, 2)
                }
                
                return report
                
        except Exception as e:
            logger.error(f"Error generating cost report: {e}")
            return {'error': str(e)}
    
    async def cleanup_old_metrics(self, days_to_keep: int = 90) -> None:
        """Clean up old metrics to save storage space."""
        try:
            cutoff_date = datetime.now().date() - timedelta(days=days_to_keep)
            
            with get_db_session() as session:
                deleted = session.query(CostMetrics).filter(
                    CostMetrics.date < cutoff_date
                ).delete()
                
                session.commit()
                logger.info(f"Cleaned up {deleted} old cost metric records")
                
        except Exception as e:
            logger.error(f"Error cleaning up old metrics: {e}")


# Global instance
persistent_cost_monitor = PersistentCostMonitor()