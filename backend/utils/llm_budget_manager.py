import os
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from decimal import Decimal
import redis
import json

logger = logging.getLogger(__name__)

class LLMCostTracker:
    """Track LLM usage costs and patterns"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        
    async def track_usage(
        self, 
        analysis_type: str, 
        model: str, 
        tokens_used: int, 
        cost: float,
        ticker: str = None
    ) -> None:
        """Track LLM usage with detailed metrics"""
        now = datetime.utcnow()
        date_key = now.strftime('%Y%m%d')
        month_key = now.strftime('%Y%m')
        
        usage_data = {
            'timestamp': now.isoformat(),
            'analysis_type': analysis_type,
            'model': model,
            'tokens_used': tokens_used,
            'cost': cost,
            'ticker': ticker
        }
        
        # Track daily usage
        daily_key = f"llm_usage:daily:{date_key}"
        await self.redis.lpush(daily_key, json.dumps(usage_data))
        await self.redis.expire(daily_key, 86400 * 32)  # Keep for 32 days
        
        # Track monthly totals
        monthly_key = f"llm_usage:monthly:{month_key}"
        await self.redis.hincrbyfloat(monthly_key, 'total_cost', cost)
        await self.redis.hincrbyfloat(monthly_key, 'total_tokens', tokens_used)
        await self.redis.hincrby(monthly_key, 'total_requests', 1)
        await self.redis.hincrby(monthly_key, f'requests_{analysis_type}', 1)
        await self.redis.expire(monthly_key, 86400 * 366)  # Keep for a year
        
        # Track hourly usage for burst protection
        hour_key = f"llm_usage:hourly:{now.strftime('%Y%m%d%H')}"
        await self.redis.hincrbyfloat(hour_key, 'cost', cost)
        await self.redis.expire(hour_key, 3600 * 25)  # Keep for 25 hours
        
    async def get_usage_stats(self, period: str = 'monthly') -> Dict:
        """Get usage statistics for specified period"""
        now = datetime.utcnow()
        
        if period == 'monthly':
            key = f"llm_usage:monthly:{now.strftime('%Y%m')}"
        elif period == 'daily':
            key = f"llm_usage:daily:{now.strftime('%Y%m%d')}"
        elif period == 'hourly':
            key = f"llm_usage:hourly:{now.strftime('%Y%m%d%H')}"
        else:
            raise ValueError(f"Invalid period: {period}")
            
        if period in ['monthly', 'hourly']:
            raw_stats = await self.redis.hgetall(key)
            return {k.decode(): float(v) if k.endswith(b'cost') or k.endswith(b'tokens') 
                   else int(v) for k, v in raw_stats.items()}
        else:  # daily - list of individual requests
            raw_data = await self.redis.lrange(key, 0, -1)
            return [json.loads(item.decode()) for item in raw_data]


class LLMBudgetManager:
    """Manages LLM usage budget with strict cost controls"""
    
    def __init__(
        self, 
        monthly_budget: float = 25.0,
        redis_client: Optional[redis.Redis] = None
    ):
        self.monthly_llm_budget = Decimal(str(monthly_budget))
        self.redis = redis_client or redis.Redis.from_url(
            os.getenv('REDIS_URL', 'redis://localhost:6379')
        )
        self.cost_tracker = LLMCostTracker(self.redis)
        
        # Cost estimates per analysis type (in USD)
        self.cost_estimates = {
            'single_agent': Decimal('0.15'),      # Single agent call
            'news_analysis': Decimal('0.12'),     # News analyst only
            'fundamentals_analysis': Decimal('0.18'),  # Fundamentals analyst
            'sentiment_analysis': Decimal('0.10'), # Social media analyst  
            'technical_analysis': Decimal('0.08'), # Technical analyst
            'bull_bear_debate': Decimal('1.20'),   # Bull/bear researcher debate
            'risk_assessment': Decimal('0.80'),    # Risk management team
            'full_analysis': Decimal('2.50'),     # Full multi-agent analysis
            'trader_decision': Decimal('0.25'),   # Final trader decision
        }
        
        # Model cost per 1K tokens (input/output)
        self.model_costs = {
            'gpt-4o-mini': {'input': 0.00015, 'output': 0.0006},
            'gpt-4o': {'input': 0.0025, 'output': 0.01},
            'gpt-4o-2024-08-06': {'input': 0.0025, 'output': 0.01},
            'o1-preview': {'input': 0.015, 'output': 0.06},
            'o1-mini': {'input': 0.003, 'output': 0.012},
            'claude-3-5-sonnet-20241022': {'input': 0.003, 'output': 0.015},
            'claude-3-haiku-20240307': {'input': 0.00025, 'output': 0.00125}
        }
        
        # Daily and hourly limits for burst protection
        self.daily_limit = self.monthly_llm_budget / 30  # Even distribution
        self.hourly_limit = self.daily_limit / 12  # Allow bursts during active hours
        
    async def can_afford_analysis(
        self, 
        analysis_type: str = 'single_agent',
        model: str = 'gpt-4o-mini',
        estimated_tokens: int = 2000
    ) -> Tuple[bool, str]:
        """
        Check if we can afford the requested analysis
        
        Returns:
            Tuple of (can_afford: bool, reason: str)
        """
        try:
            # Get current usage
            monthly_usage = await self._get_monthly_usage()
            daily_usage = await self._get_daily_usage()
            hourly_usage = await self._get_hourly_usage()
            
            # Calculate estimated cost
            if analysis_type in self.cost_estimates:
                estimated_cost = self.cost_estimates[analysis_type]
            else:
                # Calculate based on token usage and model
                estimated_cost = self._calculate_token_cost(model, estimated_tokens)
            
            # Check monthly budget
            if monthly_usage + estimated_cost > self.monthly_llm_budget:
                return False, f"Monthly budget exceeded ({monthly_usage + estimated_cost:.2f} > {self.monthly_llm_budget})"
            
            # Check daily limit
            if daily_usage + estimated_cost > self.daily_limit:
                return False, f"Daily limit exceeded ({daily_usage + estimated_cost:.2f} > {self.daily_limit})"
            
            # Check hourly limit
            if hourly_usage + estimated_cost > self.hourly_limit:
                return False, f"Hourly limit exceeded ({hourly_usage + estimated_cost:.2f} > {self.hourly_limit})"
            
            return True, "Budget available"
            
        except Exception as e:
            logger.error(f"Error checking budget: {e}")
            return False, f"Budget check failed: {str(e)}"
    
    async def reserve_budget(
        self, 
        analysis_type: str, 
        estimated_cost: Optional[float] = None
    ) -> str:
        """
        Reserve budget for analysis and return reservation ID
        """
        if estimated_cost is None:
            estimated_cost = float(self.cost_estimates.get(analysis_type, Decimal('0.15')))
        
        can_afford, reason = await self.can_afford_analysis(analysis_type)
        if not can_afford:
            raise BudgetExceededException(reason)
        
        # Create reservation
        reservation_id = f"llm_reservation:{datetime.utcnow().isoformat()}:{analysis_type}"
        reservation_data = {
            'analysis_type': analysis_type,
            'estimated_cost': estimated_cost,
            'timestamp': datetime.utcnow().isoformat(),
            'status': 'reserved'
        }
        
        await self.redis.setex(
            reservation_id, 
            300,  # 5 minute expiry
            json.dumps(reservation_data)
        )
        
        # Track reservation in hourly/daily counters
        now = datetime.utcnow()
        hour_key = f"llm_reservations:hourly:{now.strftime('%Y%m%d%H')}"
        daily_key = f"llm_reservations:daily:{now.strftime('%Y%m%d')}"
        
        await self.redis.hincrbyfloat(hour_key, 'reserved_cost', estimated_cost)
        await self.redis.hincrbyfloat(daily_key, 'reserved_cost', estimated_cost)
        await self.redis.expire(hour_key, 3600 * 2)
        await self.redis.expire(daily_key, 86400 * 2)
        
        return reservation_id
    
    async def confirm_usage(
        self, 
        reservation_id: str, 
        actual_cost: float,
        model: str,
        tokens_used: int,
        analysis_type: str,
        ticker: str = None
    ) -> None:
        """
        Confirm actual usage and track costs
        """
        # Track the actual usage
        await self.cost_tracker.track_usage(
            analysis_type=analysis_type,
            model=model,
            tokens_used=tokens_used,
            cost=actual_cost,
            ticker=ticker
        )
        
        # Mark reservation as completed
        reservation_data = await self.redis.get(reservation_id)
        if reservation_data:
            data = json.loads(reservation_data.decode())
            data['status'] = 'completed'
            data['actual_cost'] = actual_cost
            data['model'] = model
            data['tokens_used'] = tokens_used
            
            # Store completed reservation for audit
            completed_key = f"llm_completed:{reservation_id.split(':', 2)[-1]}"
            await self.redis.setex(completed_key, 86400 * 7, json.dumps(data))
        
        # Clean up reservation
        await self.redis.delete(reservation_id)
        
        logger.info(f"LLM usage confirmed: {analysis_type} for {ticker}, "
                   f"cost: ${actual_cost:.4f}, tokens: {tokens_used}")
    
    async def _get_monthly_usage(self) -> Decimal:
        """Get current month's LLM spending"""
        month_key = f"llm_usage:monthly:{datetime.utcnow().strftime('%Y%m')}"
        usage = await self.redis.hget(month_key, 'total_cost')
        return Decimal(str(usage.decode())) if usage else Decimal('0')
    
    async def _get_daily_usage(self) -> Decimal:
        """Get current day's LLM spending including reservations"""
        now = datetime.utcnow()
        date_key = now.strftime('%Y%m%d')
        
        # Actual usage
        daily_usage_key = f"llm_usage:daily:{date_key}"
        daily_requests = await self.redis.lrange(daily_usage_key, 0, -1)
        actual_cost = sum(
            json.loads(req.decode())['cost'] 
            for req in daily_requests
        )
        
        # Reserved usage
        reservations_key = f"llm_reservations:daily:{date_key}"
        reserved_cost = await self.redis.hget(reservations_key, 'reserved_cost')
        reserved = float(reserved_cost.decode()) if reserved_cost else 0
        
        return Decimal(str(actual_cost + reserved))
    
    async def _get_hourly_usage(self) -> Decimal:
        """Get current hour's LLM spending including reservations"""
        now = datetime.utcnow()
        hour_key_actual = f"llm_usage:hourly:{now.strftime('%Y%m%d%H')}"
        hour_key_reserved = f"llm_reservations:hourly:{now.strftime('%Y%m%d%H')}"
        
        # Actual usage
        actual_usage = await self.redis.hget(hour_key_actual, 'cost')
        actual = float(actual_usage.decode()) if actual_usage else 0
        
        # Reserved usage  
        reserved_usage = await self.redis.hget(hour_key_reserved, 'reserved_cost')
        reserved = float(reserved_usage.decode()) if reserved_usage else 0
        
        return Decimal(str(actual + reserved))
    
    def _calculate_token_cost(self, model: str, tokens: int) -> Decimal:
        """Calculate cost based on token usage and model"""
        if model not in self.model_costs:
            # Default to gpt-4o-mini if model unknown
            model = 'gpt-4o-mini'
            
        costs = self.model_costs[model]
        # Assume 70% input, 30% output split
        input_tokens = int(tokens * 0.7)
        output_tokens = int(tokens * 0.3)
        
        input_cost = (input_tokens / 1000) * costs['input']
        output_cost = (output_tokens / 1000) * costs['output']
        
        return Decimal(str(input_cost + output_cost))
    
    async def get_budget_status(self) -> Dict:
        """Get comprehensive budget status"""
        monthly_usage = await self._get_monthly_usage()
        daily_usage = await self._get_daily_usage()
        hourly_usage = await self._get_hourly_usage()
        
        stats = await self.cost_tracker.get_usage_stats('monthly')
        
        return {
            'budget': {
                'monthly_budget': float(self.monthly_llm_budget),
                'monthly_used': float(monthly_usage),
                'monthly_remaining': float(self.monthly_llm_budget - monthly_usage),
                'daily_limit': float(self.daily_limit),
                'daily_used': float(daily_usage),
                'hourly_limit': float(self.hourly_limit),
                'hourly_used': float(hourly_usage)
            },
            'usage_stats': stats,
            'cost_health': self._get_cost_health(monthly_usage),
            'recommended_actions': self._get_recommendations(monthly_usage, daily_usage)
        }
    
    def _get_cost_health(self, monthly_usage: Decimal) -> str:
        """Get budget health status"""
        usage_percentage = (monthly_usage / self.monthly_llm_budget) * 100
        
        if usage_percentage < 50:
            return "healthy"
        elif usage_percentage < 75:
            return "moderate"
        elif usage_percentage < 90:
            return "high"
        else:
            return "critical"
    
    def _get_recommendations(self, monthly_usage: Decimal, daily_usage: Decimal) -> List[str]:
        """Get budget optimization recommendations"""
        recommendations = []
        
        usage_percentage = (monthly_usage / self.monthly_llm_budget) * 100
        
        if usage_percentage > 80:
            recommendations.append("Consider using cheaper models (gpt-4o-mini instead of gpt-4o)")
            recommendations.append("Reduce analysis depth for non-critical stocks")
            recommendations.append("Increase cache TTL to reduce repeated analyses")
        
        if daily_usage > self.daily_limit * Decimal('0.8'):
            recommendations.append("Daily usage high - consider batching analyses")
            recommendations.append("Implement queue system for non-urgent analyses")
        
        if not recommendations:
            recommendations.append("Budget usage is healthy - continue current operations")
            
        return recommendations


class BudgetExceededException(Exception):
    """Raised when LLM budget would be exceeded"""
    pass


class LLMCircuitBreaker:
    """Circuit breaker to prevent LLM cost overruns"""
    
    def __init__(self, budget_manager: LLMBudgetManager):
        self.budget_manager = budget_manager
        self.failure_threshold = 5  # Number of budget failures before opening circuit
        self.recovery_timeout = 300  # 5 minutes
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        self.failure_count = 0
        self.last_failure_time = None
    
    async def call_with_circuit_breaker(self, analysis_function, *args, **kwargs):
        """Execute analysis function with circuit breaker protection"""
        if self.state == 'OPEN':
            if self._should_attempt_reset():
                self.state = 'HALF_OPEN'
            else:
                raise BudgetExceededException("Circuit breaker is OPEN - too many budget failures")
        
        try:
            # Check budget before proceeding
            can_afford, reason = await self.budget_manager.can_afford_analysis(
                kwargs.get('analysis_type', 'single_agent')
            )
            
            if not can_afford:
                await self._record_failure()
                raise BudgetExceededException(f"Budget check failed: {reason}")
            
            # Execute the analysis
            result = await analysis_function(*args, **kwargs)
            
            # Reset circuit breaker on success
            if self.state == 'HALF_OPEN':
                self.state = 'CLOSED'
                self.failure_count = 0
            
            return result
            
        except BudgetExceededException:
            await self._record_failure()
            raise
        except Exception as e:
            logger.error(f"Circuit breaker caught unexpected error: {e}")
            raise
    
    async def _record_failure(self):
        """Record a budget failure"""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit breaker"""
        if self.last_failure_time is None:
            return True
        
        time_since_failure = datetime.utcnow() - self.last_failure_time
        return time_since_failure.total_seconds() > self.recovery_timeout