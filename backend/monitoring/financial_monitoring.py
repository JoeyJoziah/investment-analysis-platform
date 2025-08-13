"""
Financial Metrics Monitoring
Comprehensive monitoring of financial performance, trading strategies, and market metrics.
"""

import asyncio
import logging
import time
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field
import json

from prometheus_client import (
    Counter, Histogram, Gauge, Summary, Info
)

from backend.config.monitoring_config import monitoring_config
from backend.utils.structured_logging import get_structured_logger

logger = get_structured_logger(__name__)

# Portfolio Performance Metrics
portfolio_total_value = Gauge(
    'portfolio_total_value_usd',
    'Total portfolio value in USD',
    ['portfolio_id', 'user_id']
)

portfolio_daily_return = Gauge(
    'portfolio_daily_return_percent',
    'Portfolio daily return percentage',
    ['portfolio_id', 'user_id']
)

portfolio_cumulative_return = Gauge(
    'portfolio_cumulative_return_percent',
    'Portfolio cumulative return percentage',
    ['portfolio_id', 'user_id', 'period']
)

portfolio_volatility = Gauge(
    'portfolio_volatility_annualized',
    'Annualized portfolio volatility',
    ['portfolio_id', 'user_id', 'period']
)

portfolio_sharpe_ratio = Gauge(
    'portfolio_sharpe_ratio',
    'Portfolio Sharpe ratio',
    ['portfolio_id', 'user_id', 'period']
)

portfolio_max_drawdown = Gauge(
    'portfolio_max_drawdown_percent',
    'Portfolio maximum drawdown percentage',
    ['portfolio_id', 'user_id', 'period']
)

portfolio_beta = Gauge(
    'portfolio_beta',
    'Portfolio beta vs market',
    ['portfolio_id', 'user_id', 'benchmark']
)

portfolio_alpha = Gauge(
    'portfolio_alpha_percent',
    'Portfolio alpha vs benchmark',
    ['portfolio_id', 'user_id', 'benchmark']
)

# Strategy Performance Metrics
strategy_hit_rate = Gauge(
    'strategy_hit_rate_percent',
    'Strategy success rate percentage',
    ['strategy_name', 'time_horizon', 'tier']
)

strategy_average_return = Gauge(
    'strategy_average_return_percent',
    'Strategy average return percentage',
    ['strategy_name', 'time_horizon', 'tier']
)

strategy_risk_adjusted_return = Gauge(
    'strategy_risk_adjusted_return',
    'Strategy risk-adjusted return (Sharpe)',
    ['strategy_name', 'time_horizon']
)

strategy_win_loss_ratio = Gauge(
    'strategy_win_loss_ratio',
    'Strategy win/loss ratio',
    ['strategy_name', 'tier']
)

strategy_maximum_consecutive_losses = Gauge(
    'strategy_max_consecutive_losses',
    'Strategy maximum consecutive losses',
    ['strategy_name']
)

# Recommendation Performance Metrics
recommendation_accuracy_1day = Gauge(
    'recommendation_accuracy_1day_percent',
    '1-day recommendation accuracy',
    ['model', 'recommendation_type', 'confidence_bucket']
)

recommendation_accuracy_7day = Gauge(
    'recommendation_accuracy_7day_percent',
    '7-day recommendation accuracy',
    ['model', 'recommendation_type', 'confidence_bucket']
)

recommendation_accuracy_30day = Gauge(
    'recommendation_accuracy_30day_percent',
    '30-day recommendation accuracy',
    ['model', 'recommendation_type', 'confidence_bucket']
)

recommendation_performance_vs_benchmark = Gauge(
    'recommendation_vs_benchmark_percent',
    'Recommendation performance vs benchmark',
    ['model', 'benchmark', 'time_horizon']
)

recommendation_calibration_error = Gauge(
    'recommendation_calibration_error',
    'Recommendation confidence calibration error',
    ['model', 'confidence_bucket']
)

# Market Metrics
market_regime_indicator = Gauge(
    'market_regime_indicator',
    'Market regime indicator (0=bear, 1=bull, 0.5=neutral)',
    ['market', 'timeframe']
)

market_volatility_index = Gauge(
    'market_volatility_index',
    'Market volatility index',
    ['market', 'calculation_method']
)

market_correlation_matrix = Gauge(
    'market_sector_correlation',
    'Correlation between market sectors',
    ['sector1', 'sector2', 'timeframe']
)

market_breadth_indicator = Gauge(
    'market_breadth_indicator',
    'Market breadth indicator',
    ['indicator_type', 'timeframe']
)

# Risk Metrics
portfolio_var_95 = Gauge(
    'portfolio_var_95_percent',
    'Portfolio 95% Value at Risk',
    ['portfolio_id', 'user_id', 'timeframe']
)

portfolio_expected_shortfall = Gauge(
    'portfolio_expected_shortfall_percent',
    'Portfolio Expected Shortfall (CVaR)',
    ['portfolio_id', 'user_id', 'timeframe']
)

concentration_risk = Gauge(
    'portfolio_concentration_risk',
    'Portfolio concentration risk (Herfindahl index)',
    ['portfolio_id', 'user_id']
)

sector_exposure = Gauge(
    'portfolio_sector_exposure_percent',
    'Portfolio sector exposure percentage',
    ['portfolio_id', 'user_id', 'sector']
)

# Trading Metrics
trade_execution_slippage = Histogram(
    'trade_execution_slippage_percent',
    'Trade execution slippage percentage',
    ['order_type', 'market_cap_tier', 'liquidity_bucket']
)

trade_latency = Histogram(
    'trade_latency_milliseconds',
    'Trade execution latency in milliseconds',
    ['order_type', 'venue']
)

order_fill_rate = Gauge(
    'order_fill_rate_percent',
    'Order fill rate percentage',
    ['order_type', 'market_cap_tier', 'time_period']
)

# Cost Analysis
trading_cost_total = Counter(
    'trading_cost_total_usd',
    'Total trading costs in USD',
    ['cost_type', 'venue']
)

trading_cost_per_share = Histogram(
    'trading_cost_per_share_cents',
    'Trading cost per share in cents',
    ['market_cap_tier', 'volume_bucket']
)

api_cost_per_recommendation = Gauge(
    'api_cost_per_recommendation_cents',
    'API cost per recommendation in cents',
    ['data_provider', 'recommendation_type']
)


@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics."""
    portfolio_id: str
    user_id: str
    total_value: float
    daily_return: float
    returns: List[float] = field(default_factory=list)
    positions: Dict[str, float] = field(default_factory=dict)
    benchmark_returns: List[float] = field(default_factory=list)
    risk_free_rate: float = 0.02


@dataclass
class StrategyMetrics:
    """Trading strategy performance metrics."""
    strategy_name: str
    trades: List[Dict[str, Any]] = field(default_factory=list)
    hit_rate: float = 0.0
    average_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    consecutive_losses: int = 0


@dataclass
class RecommendationTrackingRecord:
    """Individual recommendation tracking."""
    id: str
    model: str
    ticker: str
    recommendation_type: str
    confidence: float
    predicted_return: float
    timestamp: datetime
    actual_returns: Dict[str, float] = field(default_factory=dict)  # 1d, 7d, 30d
    benchmark_returns: Dict[str, float] = field(default_factory=dict)


class FinancialMonitor:
    """
    Comprehensive financial performance monitoring.
    """
    
    def __init__(self):
        self.portfolio_cache: Dict[str, PortfolioMetrics] = {}
        self.strategy_cache: Dict[str, StrategyMetrics] = {}
        self.recommendation_tracking: Dict[str, RecommendationTrackingRecord] = {}
        self.market_data_cache: Dict[str, Any] = {}
        
        # Performance calculation caches
        self.returns_cache: Dict[str, deque] = defaultdict(lambda: deque(maxlen=252))  # 1 year of daily returns
        self.benchmark_cache: Dict[str, deque] = defaultdict(lambda: deque(maxlen=252))
        
        self._monitoring_task: Optional[asyncio.Task] = None
        self._update_interval = 300  # 5 minutes
    
    async def start_monitoring(self) -> None:
        """Start financial monitoring."""
        if not self._monitoring_task:
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            logger.info("Started financial metrics monitoring")
    
    async def stop_monitoring(self) -> None:
        """Stop financial monitoring."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            logger.info("Stopped financial metrics monitoring")
    
    async def _monitoring_loop(self) -> None:
        """Background financial monitoring loop."""
        while True:
            try:
                await self._update_portfolio_metrics()
                await self._update_strategy_metrics()
                await self._update_recommendation_accuracy()
                await self._update_market_metrics()
                await self._calculate_risk_metrics()
                await asyncio.sleep(self._update_interval)
            except Exception as e:
                logger.error(f"Error in financial monitoring loop: {e}")
                await asyncio.sleep(self._update_interval)
    
    async def _update_portfolio_metrics(self) -> None:
        """Update all portfolio performance metrics."""
        try:
            from backend.repositories.portfolio_repository import PortfolioRepository
            
            portfolio_repo = PortfolioRepository()
            portfolios = await portfolio_repo.get_all_active_portfolios()
            
            for portfolio in portfolios:
                metrics = await self._calculate_portfolio_metrics(portfolio)
                await self._record_portfolio_metrics(metrics)
        
        except Exception as e:
            logger.error(f"Error updating portfolio metrics: {e}")
    
    async def _calculate_portfolio_metrics(self, portfolio) -> PortfolioMetrics:
        """Calculate comprehensive portfolio metrics."""
        try:
            from backend.repositories.portfolio_repository import PortfolioRepository
            
            portfolio_repo = PortfolioRepository()
            
            # Get current portfolio value and positions
            total_value = await portfolio_repo.get_portfolio_value(portfolio.id)
            positions = await portfolio_repo.get_portfolio_positions(portfolio.id)
            
            # Get historical returns
            returns_data = await portfolio_repo.get_portfolio_returns_history(portfolio.id, days=252)
            returns = [r['return'] for r in returns_data]
            
            # Calculate daily return
            daily_return = returns[0] if returns else 0.0
            
            metrics = PortfolioMetrics(
                portfolio_id=str(portfolio.id),
                user_id=str(portfolio.user_id),
                total_value=total_value,
                daily_return=daily_return,
                returns=returns,
                positions={p['ticker']: p['value'] for p in positions}
            )
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {e}")
            return PortfolioMetrics("", "", 0.0, 0.0)
    
    async def _record_portfolio_metrics(self, metrics: PortfolioMetrics) -> None:
        """Record portfolio metrics to Prometheus."""
        try:
            # Basic metrics
            portfolio_total_value.labels(
                portfolio_id=metrics.portfolio_id,
                user_id=metrics.user_id
            ).set(metrics.total_value)
            
            portfolio_daily_return.labels(
                portfolio_id=metrics.portfolio_id,
                user_id=metrics.user_id
            ).set(metrics.daily_return)
            
            if len(metrics.returns) >= 30:  # Need enough data for calculations
                returns_array = np.array(metrics.returns)
                
                # Cumulative returns
                cumulative_30d = (np.prod(1 + returns_array[:30] / 100) - 1) * 100
                portfolio_cumulative_return.labels(
                    portfolio_id=metrics.portfolio_id,
                    user_id=metrics.user_id,
                    period="30d"
                ).set(cumulative_30d)
                
                # Volatility (annualized)
                volatility_30d = np.std(returns_array[:30]) * np.sqrt(252)
                portfolio_volatility.labels(
                    portfolio_id=metrics.portfolio_id,
                    user_id=metrics.user_id,
                    period="30d"
                ).set(volatility_30d)
                
                # Sharpe ratio
                excess_returns = returns_array[:30] - (metrics.risk_free_rate / 252 * 100)
                if volatility_30d > 0:
                    sharpe_30d = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
                    portfolio_sharpe_ratio.labels(
                        portfolio_id=metrics.portfolio_id,
                        user_id=metrics.user_id,
                        period="30d"
                    ).set(sharpe_30d)
                
                # Maximum drawdown
                cumulative_returns = np.cumprod(1 + returns_array / 100)
                running_max = np.maximum.accumulate(cumulative_returns)
                drawdowns = (cumulative_returns - running_max) / running_max * 100
                max_dd = np.min(drawdowns[:30])
                portfolio_max_drawdown.labels(
                    portfolio_id=metrics.portfolio_id,
                    user_id=metrics.user_id,
                    period="30d"
                ).set(abs(max_dd))
                
                # Concentration risk (Herfindahl index)
                if metrics.positions:
                    total_value = sum(metrics.positions.values())
                    if total_value > 0:
                        weights = [value / total_value for value in metrics.positions.values()]
                        herfindahl = sum(w**2 for w in weights)
                        concentration_risk.labels(
                            portfolio_id=metrics.portfolio_id,
                            user_id=metrics.user_id
                        ).set(herfindahl)
        
        except Exception as e:
            logger.error(f"Error recording portfolio metrics: {e}")
    
    async def _update_strategy_metrics(self) -> None:
        """Update trading strategy performance metrics."""
        try:
            from backend.analytics.recommendation_engine import RecommendationEngine
            
            engine = RecommendationEngine()
            strategies = await engine.get_strategy_performance_data()
            
            for strategy_name, performance_data in strategies.items():
                # Hit rate
                hit_rate = performance_data.get('hit_rate', 0) * 100
                strategy_hit_rate.labels(
                    strategy_name=strategy_name,
                    time_horizon="30d",
                    tier="all"
                ).set(hit_rate)
                
                # Average return
                avg_return = performance_data.get('average_return', 0) * 100
                strategy_average_return.labels(
                    strategy_name=strategy_name,
                    time_horizon="30d",
                    tier="all"
                ).set(avg_return)
                
                # Risk-adjusted return
                risk_adj_return = performance_data.get('sharpe_ratio', 0)
                strategy_risk_adjusted_return.labels(
                    strategy_name=strategy_name,
                    time_horizon="30d"
                ).set(risk_adj_return)
                
                # Win/loss ratio
                win_loss_ratio = performance_data.get('win_loss_ratio', 0)
                strategy_win_loss_ratio.labels(
                    strategy_name=strategy_name,
                    tier="all"
                ).set(win_loss_ratio)
        
        except Exception as e:
            logger.error(f"Error updating strategy metrics: {e}")
    
    async def _update_recommendation_accuracy(self) -> None:
        """Update recommendation accuracy metrics."""
        try:
            from backend.analytics.recommendation_engine import RecommendationEngine
            
            engine = RecommendationEngine()
            accuracy_data = await engine.get_detailed_accuracy_metrics()
            
            for model, model_data in accuracy_data.items():
                for rec_type, type_data in model_data.items():
                    for confidence_bucket, bucket_data in type_data.items():
                        # 1-day accuracy
                        if '1d_accuracy' in bucket_data:
                            recommendation_accuracy_1day.labels(
                                model=model,
                                recommendation_type=rec_type,
                                confidence_bucket=confidence_bucket
                            ).set(bucket_data['1d_accuracy'] * 100)
                        
                        # 7-day accuracy
                        if '7d_accuracy' in bucket_data:
                            recommendation_accuracy_7day.labels(
                                model=model,
                                recommendation_type=rec_type,
                                confidence_bucket=confidence_bucket
                            ).set(bucket_data['7d_accuracy'] * 100)
                        
                        # 30-day accuracy
                        if '30d_accuracy' in bucket_data:
                            recommendation_accuracy_30day.labels(
                                model=model,
                                recommendation_type=rec_type,
                                confidence_bucket=confidence_bucket
                            ).set(bucket_data['30d_accuracy'] * 100)
                        
                        # Calibration error
                        if 'calibration_error' in bucket_data:
                            recommendation_calibration_error.labels(
                                model=model,
                                confidence_bucket=confidence_bucket
                            ).set(bucket_data['calibration_error'])
        
        except Exception as e:
            logger.error(f"Error updating recommendation accuracy: {e}")
    
    async def _update_market_metrics(self) -> None:
        """Update market-wide metrics."""
        try:
            from backend.analytics.market_regime.regime_detector import MarketRegimeDetector
            
            regime_detector = MarketRegimeDetector()
            
            # Market regime
            current_regime = await regime_detector.get_current_regime()
            regime_value = 1.0 if current_regime == 'bull' else 0.0 if current_regime == 'bear' else 0.5
            
            market_regime_indicator.labels(
                market="SPY",
                timeframe="daily"
            ).set(regime_value)
            
            # Market volatility
            volatility_metrics = await regime_detector.get_volatility_metrics()
            for timeframe, vol_value in volatility_metrics.items():
                market_volatility_index.labels(
                    market="SPY",
                    calculation_method="realized"
                ).set(vol_value)
        
        except Exception as e:
            logger.error(f"Error updating market metrics: {e}")
    
    async def _calculate_risk_metrics(self) -> None:
        """Calculate and update risk metrics."""
        try:
            for portfolio_id, metrics in self.portfolio_cache.items():
                if len(metrics.returns) >= 30:
                    returns_array = np.array(metrics.returns[:30]) / 100  # Convert to decimal
                    
                    # VaR 95%
                    var_95 = np.percentile(returns_array, 5) * 100  # Back to percentage
                    portfolio_var_95.labels(
                        portfolio_id=metrics.portfolio_id,
                        user_id=metrics.user_id,
                        timeframe="30d"
                    ).set(abs(var_95))
                    
                    # Expected Shortfall (CVaR)
                    var_threshold = np.percentile(returns_array, 5)
                    tail_losses = returns_array[returns_array <= var_threshold]
                    if len(tail_losses) > 0:
                        expected_shortfall = np.mean(tail_losses) * 100
                        portfolio_expected_shortfall.labels(
                            portfolio_id=metrics.portfolio_id,
                            user_id=metrics.user_id,
                            timeframe="30d"
                        ).set(abs(expected_shortfall))
        
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
    
    # Public methods for recording specific events
    
    def record_trade_execution(
        self,
        order_type: str,
        market_cap_tier: str,
        expected_price: float,
        actual_price: float,
        execution_time_ms: float,
        venue: str = "default"
    ) -> None:
        """Record trade execution metrics."""
        try:
            # Calculate slippage
            if expected_price > 0:
                slippage_pct = abs(actual_price - expected_price) / expected_price * 100
                trade_execution_slippage.labels(
                    order_type=order_type,
                    market_cap_tier=market_cap_tier,
                    liquidity_bucket=self._get_liquidity_bucket(market_cap_tier)
                ).observe(slippage_pct)
            
            # Record latency
            trade_latency.labels(
                order_type=order_type,
                venue=venue
            ).observe(execution_time_ms)
        
        except Exception as e:
            logger.error(f"Error recording trade execution metrics: {e}")
    
    def record_trading_cost(
        self,
        cost_type: str,
        amount_usd: float,
        shares: int,
        market_cap_tier: str,
        venue: str = "default"
    ) -> None:
        """Record trading cost metrics."""
        try:
            # Total cost
            trading_cost_total.labels(
                cost_type=cost_type,
                venue=venue
            ).inc(amount_usd)
            
            # Cost per share
            if shares > 0:
                cost_per_share_cents = (amount_usd / shares) * 100
                trading_cost_per_share.labels(
                    market_cap_tier=market_cap_tier,
                    volume_bucket=self._get_volume_bucket(shares)
                ).observe(cost_per_share_cents)
        
        except Exception as e:
            logger.error(f"Error recording trading cost metrics: {e}")
    
    def add_recommendation_tracking(
        self,
        model: str,
        ticker: str,
        recommendation_type: str,
        confidence: float,
        predicted_return: float
    ) -> str:
        """Add a recommendation for tracking."""
        try:
            rec_id = f"{model}_{ticker}_{int(time.time())}"
            
            record = RecommendationTrackingRecord(
                id=rec_id,
                model=model,
                ticker=ticker,
                recommendation_type=recommendation_type,
                confidence=confidence,
                predicted_return=predicted_return,
                timestamp=datetime.now()
            )
            
            self.recommendation_tracking[rec_id] = record
            return rec_id
        
        except Exception as e:
            logger.error(f"Error adding recommendation tracking: {e}")
            return ""
    
    def _get_liquidity_bucket(self, market_cap_tier: str) -> str:
        """Get liquidity bucket based on market cap tier."""
        liquidity_map = {
            "tier1": "high",
            "tier2": "medium", 
            "tier3": "low",
            "tier4": "very_low",
            "tier5": "very_low"
        }
        return liquidity_map.get(market_cap_tier, "unknown")
    
    def _get_volume_bucket(self, shares: int) -> str:
        """Get volume bucket for shares traded."""
        if shares < 100:
            return "small"
        elif shares < 1000:
            return "medium"
        elif shares < 10000:
            return "large"
        else:
            return "very_large"
    
    def get_financial_summary(self) -> Dict[str, Any]:
        """Get comprehensive financial monitoring summary."""
        try:
            return {
                "timestamp": datetime.now().isoformat(),
                "portfolios_monitored": len(self.portfolio_cache),
                "strategies_monitored": len(self.strategy_cache),
                "recommendations_tracked": len(self.recommendation_tracking),
                "market_data_points": len(self.market_data_cache),
                "last_update": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error generating financial summary: {e}")
            return {"error": str(e)}


# Global financial monitor
financial_monitor = FinancialMonitor()


# Setup function
async def setup_financial_monitoring():
    """Setup financial monitoring."""
    await financial_monitor.start_monitoring()
    logger.info("Financial metrics monitoring setup completed")


# Convenience functions
def record_trade(order_type: str, market_cap_tier: str, expected_price: float, 
                actual_price: float, execution_time_ms: float, venue: str = "default"):
    """Convenience function to record trade execution."""
    financial_monitor.record_trade_execution(
        order_type, market_cap_tier, expected_price, actual_price, execution_time_ms, venue
    )


def record_cost(cost_type: str, amount_usd: float, shares: int, 
               market_cap_tier: str, venue: str = "default"):
    """Convenience function to record trading costs."""
    financial_monitor.record_trading_cost(cost_type, amount_usd, shares, market_cap_tier, venue)


def track_recommendation(model: str, ticker: str, recommendation_type: str, 
                        confidence: float, predicted_return: float) -> str:
    """Convenience function to track recommendations."""
    return financial_monitor.add_recommendation_tracking(
        model, ticker, recommendation_type, confidence, predicted_return
    )