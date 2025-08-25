"""
Business Metrics Tracking for Investment Analysis Platform
Tracks key business indicators and cost metrics for the $50/month budget
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json

from prometheus_client import Counter, Gauge, Histogram

# Business-specific metrics
daily_recommendations_generated = Counter(
    'daily_recommendations_generated_total',
    'Daily recommendations generated',
    ['confidence_level', 'action']
)

stocks_processed_daily = Counter(
    'stocks_processed_daily_total', 
    'Daily stocks processed',
    ['exchange', 'sector']
)

user_portfolio_performance = Histogram(
    'user_portfolio_performance_percent',
    'User portfolio performance percentage',
    ['time_horizon']
)

api_cost_tracking = Counter(
    'api_cost_tracking_dollars',
    'API costs in dollars', 
    ['provider', 'endpoint_type']
)

monthly_budget_usage = Gauge(
    'monthly_budget_usage_percent',
    'Monthly budget usage percentage'
)

ml_model_accuracy_tracking = Gauge(
    'ml_model_accuracy_current',
    'Current ML model accuracy',
    ['model_name', 'prediction_type']
)

data_pipeline_success_rate = Gauge(
    'data_pipeline_success_rate_percent', 
    'Data pipeline success rate',
    ['pipeline_name']
)

# Cost breakdown by service
service_costs = Gauge(
    'service_daily_cost_dollars',
    'Daily cost per service in dollars',
    ['service_name']
)

logger = logging.getLogger(__name__)

class BusinessMetricsTracker:
    """Tracks and reports business metrics for the investment platform"""
    
    def __init__(self):
        self.daily_budget = 50.0 / 30  # $50/month divided by 30 days
        self.current_daily_cost = 0.0
        self.api_call_costs = {
            'alpha_vantage': 0.0,  # Free tier
            'finnhub': 0.0,        # Free tier
            'polygon': 0.0,        # Free tier 
            'news_api': 0.0        # Free tier
        }
        
    async def track_recommendation_generation(
        self, 
        confidence_level: str,
        action: str,
        ticker: str
    ):
        """Track recommendation generation metrics"""
        try:
            daily_recommendations_generated.labels(
                confidence_level=confidence_level,
                action=action
            ).inc()
            
            logger.info(f"Recommendation generated: {action} for {ticker} (confidence: {confidence_level})")
            
        except Exception as e:
            logger.error(f"Error tracking recommendation metrics: {e}")
    
    async def track_stock_processing(
        self,
        ticker: str,
        exchange: str,
        sector: str,
        processing_time: float
    ):
        """Track stock processing metrics"""
        try:
            stocks_processed_daily.labels(
                exchange=exchange,
                sector=sector
            ).inc()
            
            logger.debug(f"Stock processed: {ticker} ({exchange}/{sector}) in {processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error tracking stock processing metrics: {e}")
    
    async def track_api_cost(
        self,
        provider: str,
        endpoint_type: str,
        estimated_cost: float = 0.0
    ):
        """Track API usage costs (mostly $0 for free tiers, but track usage)"""
        try:
            api_cost_tracking.labels(
                provider=provider,
                endpoint_type=endpoint_type
            ).inc(estimated_cost)
            
            # Update internal cost tracking
            if provider in self.api_call_costs:
                self.api_call_costs[provider] += estimated_cost
                
            logger.debug(f"API call tracked: {provider}/{endpoint_type} (cost: ${estimated_cost})")
            
        except Exception as e:
            logger.error(f"Error tracking API cost metrics: {e}")
    
    async def update_budget_usage(self):
        """Update monthly budget usage percentage"""
        try:
            # Calculate total daily costs
            total_daily_cost = sum(self.api_call_costs.values())
            
            # Project monthly cost
            monthly_projection = total_daily_cost * 30
            
            # Calculate budget usage percentage
            budget_usage_percent = (monthly_projection / 50.0) * 100
            
            monthly_budget_usage.set(budget_usage_percent)
            
            # Update service cost breakdown
            for service, cost in self.api_call_costs.items():
                service_costs.labels(service_name=service).set(cost)
                
            logger.info(f"Budget usage updated: {budget_usage_percent:.1f}% (${monthly_projection:.2f}/month)")
            
            # Alert if approaching budget limit
            if budget_usage_percent > 80:
                logger.warning(f"Budget usage high: {budget_usage_percent:.1f}%")
                await self._send_budget_alert(budget_usage_percent, monthly_projection)
                
        except Exception as e:
            logger.error(f"Error updating budget metrics: {e}")
    
    async def track_ml_model_accuracy(
        self,
        model_name: str,
        prediction_type: str,
        accuracy: float
    ):
        """Track ML model accuracy"""
        try:
            ml_model_accuracy_tracking.labels(
                model_name=model_name,
                prediction_type=prediction_type
            ).set(accuracy)
            
            logger.info(f"ML accuracy updated: {model_name}/{prediction_type} = {accuracy:.3f}")
            
        except Exception as e:
            logger.error(f"Error tracking ML accuracy metrics: {e}")
    
    async def track_pipeline_success_rate(
        self,
        pipeline_name: str,
        success_rate: float
    ):
        """Track data pipeline success rates"""
        try:
            data_pipeline_success_rate.labels(
                pipeline_name=pipeline_name
            ).set(success_rate)
            
            logger.info(f"Pipeline success rate: {pipeline_name} = {success_rate:.1f}%")
            
        except Exception as e:
            logger.error(f"Error tracking pipeline metrics: {e}")
    
    async def track_portfolio_performance(
        self,
        performance_percent: float,
        time_horizon: str = '1d'
    ):
        """Track user portfolio performance"""
        try:
            user_portfolio_performance.labels(
                time_horizon=time_horizon
            ).observe(performance_percent)
            
            logger.info(f"Portfolio performance tracked: {performance_percent:.2f}% ({time_horizon})")
            
        except Exception as e:
            logger.error(f"Error tracking portfolio performance: {e}")
    
    async def generate_daily_report(self) -> Dict:
        """Generate daily business metrics report"""
        try:
            report = {
                'date': datetime.utcnow().isoformat(),
                'budget_usage': {
                    'daily_cost': sum(self.api_call_costs.values()),
                    'monthly_projection': sum(self.api_call_costs.values()) * 30,
                    'budget_remaining': 50.0 - (sum(self.api_call_costs.values()) * 30),
                    'usage_percent': (sum(self.api_call_costs.values()) * 30 / 50.0) * 100
                },
                'api_usage': self.api_call_costs.copy(),
                'cost_optimization_tips': await self._get_cost_optimization_tips()
            }
            
            logger.info(f"Daily business report generated: ${report['budget_usage']['daily_cost']:.2f} spent today")
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating daily report: {e}")
            return {}
    
    async def _send_budget_alert(self, usage_percent: float, monthly_projection: float):
        """Send budget usage alert"""
        try:
            # In a real implementation, this would send notifications
            # For now, just log the alert
            alert_message = (
                f"BUDGET ALERT: Usage at {usage_percent:.1f}% "
                f"(${monthly_projection:.2f}/month projected)"
            )
            
            logger.warning(alert_message)
            
            # TODO: Integrate with AlertManager webhook
            # await self._send_webhook_alert(alert_message)
            
        except Exception as e:
            logger.error(f"Error sending budget alert: {e}")
    
    async def _get_cost_optimization_tips(self) -> List[str]:
        """Get cost optimization recommendations"""
        tips = []
        
        total_cost = sum(self.api_call_costs.values())
        monthly_projection = total_cost * 30
        
        if monthly_projection > 40:  # 80% of budget
            tips.append("Consider reducing API call frequency")
            tips.append("Implement more aggressive caching")
            tips.append("Optimize data pipeline schedules")
        
        if self.api_call_costs.get('alpha_vantage', 0) > 0.5:
            tips.append("Alpha Vantage usage high - consider batching requests")
        
        if self.api_call_costs.get('polygon', 0) > 0.3:
            tips.append("Polygon.io usage approaching free tier limits")
        
        if not tips:
            tips.append("Current usage is within optimal range")
            tips.append("Continue monitoring for any usage spikes")
        
        return tips
    
    async def reset_daily_metrics(self):
        """Reset daily tracking metrics (called daily via cron)"""
        try:
            logger.info("Resetting daily metrics...")
            
            # Archive current day's data before reset
            await self.generate_daily_report()
            
            # Reset daily counters (Prometheus counters can't be reset, but we can track resets)
            self.current_daily_cost = 0.0
            
            logger.info("Daily metrics reset completed")
            
        except Exception as e:
            logger.error(f"Error resetting daily metrics: {e}")

# Global business metrics tracker instance
business_metrics = BusinessMetricsTracker()

# Convenience functions for easy import
async def track_recommendation(confidence: str, action: str, ticker: str):
    """Convenience function to track recommendation generation"""
    await business_metrics.track_recommendation_generation(confidence, action, ticker)

async def track_stock_analysis(ticker: str, exchange: str, sector: str, duration: float):
    """Convenience function to track stock analysis"""
    await business_metrics.track_stock_processing(ticker, exchange, sector, duration)

async def track_api_usage(provider: str, endpoint: str, cost: float = 0.0):
    """Convenience function to track API usage"""
    await business_metrics.track_api_cost(provider, endpoint, cost)

async def update_model_accuracy(model: str, prediction_type: str, accuracy: float):
    """Convenience function to update ML model accuracy"""
    await business_metrics.track_ml_model_accuracy(model, prediction_type, accuracy)

async def track_portfolio_perf(performance: float, horizon: str = '1d'):
    """Convenience function to track portfolio performance"""
    await business_metrics.track_portfolio_performance(performance, horizon)
