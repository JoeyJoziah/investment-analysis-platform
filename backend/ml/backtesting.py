"""
Comprehensive Backtesting Framework
Provides walk-forward analysis, risk-adjusted metrics, and Monte Carlo simulation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from pathlib import Path
import json

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class BacktestMetric(Enum):
    """Available backtest metrics"""
    TOTAL_RETURN = "total_return"
    ANNUALIZED_RETURN = "annualized_return"
    VOLATILITY = "volatility"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio" 
    CALMAR_RATIO = "calmar_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    DIRECTIONAL_ACCURACY = "directional_accuracy"
    INFORMATION_RATIO = "information_ratio"
    BETA = "beta"
    ALPHA = "alpha"
    TRACKING_ERROR = "tracking_error"


@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    start_date: datetime
    end_date: datetime
    initial_capital: float = 100000.0
    commission: float = 0.001  # 0.1% per trade
    slippage: float = 0.0005   # 0.05% slippage
    lookback_window: int = 252  # Trading days for rolling metrics
    rebalance_frequency: str = "daily"  # daily, weekly, monthly
    risk_free_rate: float = 0.02  # Annual risk-free rate
    benchmark_symbol: str = "SPY"
    max_position_size: float = 0.1  # 10% max position
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    transaction_costs_enabled: bool = True
    

@dataclass
class TradeRecord:
    """Individual trade record"""
    ticker: str
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    quantity: int
    side: str  # 'long' or 'short'
    pnl: float
    pnl_pct: float
    commission: float
    slippage: float
    duration_days: int
    signal_strength: float
    model_confidence: float


@dataclass
class BacktestResult:
    """Comprehensive backtest results"""
    config: BacktestConfig
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    trades: List[TradeRecord]
    daily_returns: pd.Series
    portfolio_values: pd.Series
    benchmark_returns: pd.Series
    metrics: Dict[str, float]
    monthly_returns: pd.DataFrame
    annual_returns: pd.Series
    drawdowns: pd.Series
    rolling_metrics: pd.DataFrame
    sector_performance: Dict[str, float]
    risk_metrics: Dict[str, float]
    performance_attribution: Dict[str, float]
    monte_carlo_results: Optional[Dict[str, Any]] = None


class WalkForwardValidator:
    """Walk-forward analysis for time series models"""
    
    def __init__(self, 
                 n_splits: int = 5,
                 train_size: Optional[int] = None,
                 test_size: int = 30,
                 gap: int = 0,
                 expanding_window: bool = False):
        self.n_splits = n_splits
        self.train_size = train_size
        self.test_size = test_size  
        self.gap = gap
        self.expanding_window = expanding_window
    
    def split(self, data: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Generate walk-forward splits"""
        splits = []
        n_samples = len(data)
        
        if self.train_size is None:
            # Calculate train size to have n_splits
            total_test_samples = self.n_splits * self.test_size
            available_train = n_samples - total_test_samples - (self.n_splits - 1) * self.gap
            self.train_size = available_train // self.n_splits
        
        for i in range(self.n_splits):
            if self.expanding_window:
                # Expanding window: training set grows
                train_start = 0
                train_end = self.train_size + i * (self.test_size + self.gap)
            else:
                # Rolling window: training set size stays constant
                train_start = i * (self.test_size + self.gap)
                train_end = train_start + self.train_size
            
            test_start = train_end + self.gap
            test_end = test_start + self.test_size
            
            if test_end > n_samples:
                break
            
            train_data = data.iloc[train_start:train_end]
            test_data = data.iloc[test_start:test_end]
            
            splits.append((train_data, test_data))
        
        return splits


class BacktestEngine:
    """
    Comprehensive backtesting engine with advanced analytics
    """
    
    def __init__(self, data_provider: Any = None):
        self.data_provider = data_provider
        self.results_cache = {}
        
    def backtest_strategy(self,
                         strategy_func: Callable,
                         universe: List[str],
                         config: BacktestConfig,
                         model: Any = None,
                         feature_columns: List[str] = None) -> BacktestResult:
        """
        Run comprehensive backtest of a trading strategy
        
        Args:
            strategy_func: Function that generates trading signals
            universe: List of tickers to trade
            config: Backtesting configuration
            model: ML model for predictions (optional)
            feature_columns: Feature columns for model (optional)
            
        Returns:
            BacktestResult with comprehensive metrics
        """
        logger.info(f"Starting backtest from {config.start_date} to {config.end_date}")
        
        # Get market data
        market_data = self._get_market_data(universe, config.start_date, config.end_date)
        benchmark_data = self._get_benchmark_data(config.benchmark_symbol, config.start_date, config.end_date)
        
        # Initialize portfolio
        portfolio = self._initialize_portfolio(config.initial_capital, universe)
        trades = []
        daily_values = []
        daily_returns = []
        
        # Get trading dates
        trading_dates = pd.date_range(config.start_date, config.end_date, freq='D')
        trading_dates = trading_dates[trading_dates.weekday < 5]  # Exclude weekends
        
        for date in trading_dates:
            try:
                # Get current market data up to this date
                current_data = {ticker: data[data.index <= date] 
                              for ticker, data in market_data.items() if len(data[data.index <= date]) > 0}
                
                if not current_data:
                    continue
                
                # Generate signals
                signals = strategy_func(current_data, date, model, feature_columns)
                
                # Execute trades
                day_trades = self._execute_trades(portfolio, signals, current_data, date, config)
                trades.extend(day_trades)
                
                # Update portfolio value
                portfolio_value = self._calculate_portfolio_value(portfolio, current_data, date)
                daily_values.append(portfolio_value)
                
                # Calculate daily return
                if len(daily_values) > 1:
                    daily_return = (daily_values[-1] - daily_values[-2]) / daily_values[-2]
                    daily_returns.append(daily_return)
                
            except Exception as e:
                logger.error(f"Error on date {date}: {e}")
                continue
        
        # Calculate comprehensive metrics
        portfolio_series = pd.Series(daily_values, index=trading_dates[:len(daily_values)])
        returns_series = pd.Series(daily_returns, index=trading_dates[:len(daily_returns)])
        
        metrics = self._calculate_comprehensive_metrics(
            returns_series, 
            benchmark_data, 
            config,
            trades
        )
        
        # Calculate additional analytics
        monthly_returns = self._calculate_monthly_returns(returns_series)
        annual_returns = self._calculate_annual_returns(returns_series)
        drawdowns = self._calculate_drawdowns(portfolio_series)
        rolling_metrics = self._calculate_rolling_metrics(returns_series, config.lookback_window)
        
        result = BacktestResult(
            config=config,
            start_date=config.start_date,
            end_date=config.end_date,
            initial_capital=config.initial_capital,
            final_capital=daily_values[-1] if daily_values else config.initial_capital,
            total_trades=len(trades),
            winning_trades=len([t for t in trades if t.pnl > 0]),
            losing_trades=len([t for t in trades if t.pnl < 0]),
            trades=trades,
            daily_returns=returns_series,
            portfolio_values=portfolio_series,
            benchmark_returns=self._align_benchmark_returns(benchmark_data, returns_series.index),
            metrics=metrics,
            monthly_returns=monthly_returns,
            annual_returns=annual_returns,
            drawdowns=drawdowns,
            rolling_metrics=rolling_metrics,
            sector_performance=self._calculate_sector_performance(trades, universe),
            risk_metrics=self._calculate_risk_metrics(returns_series, benchmark_data),
            performance_attribution=self._calculate_performance_attribution(trades)
        )
        
        logger.info(f"Backtest completed: {result.total_trades} trades, "
                   f"{result.metrics.get('total_return', 0):.2%} total return")
        
        return result
    
    def walk_forward_analysis(self,
                             strategy_func: Callable,
                             universe: List[str],
                             config: BacktestConfig,
                             model: Any = None,
                             n_splits: int = 5,
                             train_ratio: float = 0.7) -> Dict[str, Any]:
        """
        Perform walk-forward analysis
        """
        logger.info("Starting walk-forward analysis")
        
        # Get full dataset
        market_data = self._get_market_data(universe, config.start_date, config.end_date)
        
        # Create date splits
        all_dates = pd.date_range(config.start_date, config.end_date, freq='D')
        all_dates = all_dates[all_dates.weekday < 5]
        
        validator = WalkForwardValidator(
            n_splits=n_splits,
            test_size=len(all_dates) // (n_splits * 2),
            expanding_window=True
        )
        
        # Create splits based on dates
        date_splits = []
        split_size = len(all_dates) // n_splits
        
        for i in range(n_splits):
            if i == 0:
                train_start = 0
                train_end = int(split_size * train_ratio)
            else:
                train_start = 0  # Expanding window
                train_end = int(split_size * (i + train_ratio))
            
            test_start = train_end
            test_end = min(test_start + split_size, len(all_dates))
            
            if test_end <= test_start:
                break
                
            train_period = (all_dates[train_start], all_dates[train_end-1])
            test_period = (all_dates[test_start], all_dates[test_end-1])
            
            date_splits.append((train_period, test_period))
        
        # Run backtests for each split
        results = []
        
        for i, (train_period, test_period) in enumerate(date_splits):
            logger.info(f"Walk-forward split {i+1}/{len(date_splits)}: "
                       f"Train: {train_period[0].date()} to {train_period[1].date()}, "
                       f"Test: {test_period[0].date()} to {test_period[1].date()}")
            
            # Create config for this split
            split_config = BacktestConfig(
                start_date=test_period[0],
                end_date=test_period[1],
                initial_capital=config.initial_capital,
                commission=config.commission,
                slippage=config.slippage,
                risk_free_rate=config.risk_free_rate,
                benchmark_symbol=config.benchmark_symbol
            )
            
            try:
                # Run backtest for this period
                result = self.backtest_strategy(strategy_func, universe, split_config, model)
                results.append({
                    'split': i + 1,
                    'train_start': train_period[0],
                    'train_end': train_period[1], 
                    'test_start': test_period[0],
                    'test_end': test_period[1],
                    'result': result
                })
                
            except Exception as e:
                logger.error(f"Error in walk-forward split {i+1}: {e}")
                continue
        
        # Aggregate results
        aggregated_metrics = self._aggregate_walk_forward_results(results)
        
        return {
            'splits': results,
            'aggregated_metrics': aggregated_metrics,
            'stability_metrics': self._calculate_stability_metrics(results),
            'performance_consistency': self._calculate_performance_consistency(results)
        }
    
    def monte_carlo_simulation(self,
                              backtest_result: BacktestResult,
                              n_simulations: int = 1000,
                              confidence_levels: List[float] = [0.05, 0.95]) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation for stress testing
        """
        logger.info(f"Running Monte Carlo simulation with {n_simulations} scenarios")
        
        returns = backtest_result.daily_returns
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Generate random return scenarios
        np.random.seed(42)  # For reproducibility
        simulated_returns = np.random.normal(mean_return, std_return, 
                                           (n_simulations, len(returns)))
        
        # Calculate metrics for each simulation
        simulation_results = []
        
        for i in range(n_simulations):
            sim_returns = pd.Series(simulated_returns[i], index=returns.index)
            sim_portfolio_values = (1 + sim_returns).cumprod() * backtest_result.initial_capital
            
            sim_metrics = {
                'total_return': (sim_portfolio_values.iloc[-1] / backtest_result.initial_capital - 1),
                'volatility': sim_returns.std() * np.sqrt(252),
                'sharpe_ratio': self._calculate_sharpe_ratio(sim_returns, backtest_result.config.risk_free_rate),
                'max_drawdown': self._calculate_max_drawdown(sim_portfolio_values),
                'final_value': sim_portfolio_values.iloc[-1]
            }
            
            simulation_results.append(sim_metrics)
        
        # Calculate confidence intervals
        results_df = pd.DataFrame(simulation_results)
        confidence_intervals = {}
        
        for metric in results_df.columns:
            confidence_intervals[metric] = {
                'mean': results_df[metric].mean(),
                'std': results_df[metric].std(),
                'percentiles': {
                    f'{int(level*100)}%': results_df[metric].quantile(level) 
                    for level in confidence_levels
                }
            }
        
        # Risk metrics
        var_95 = results_df['total_return'].quantile(0.05)  # 5% VaR
        cvar_95 = results_df[results_df['total_return'] <= var_95]['total_return'].mean()  # Conditional VaR
        
        probability_of_loss = (results_df['total_return'] < 0).mean()
        probability_of_ruin = (results_df['final_value'] < backtest_result.initial_capital * 0.5).mean()
        
        return {
            'n_simulations': n_simulations,
            'confidence_intervals': confidence_intervals,
            'risk_metrics': {
                'var_95': var_95,
                'cvar_95': cvar_95,
                'probability_of_loss': probability_of_loss,
                'probability_of_ruin': probability_of_ruin
            },
            'simulation_data': results_df,
            'original_metrics': {
                'total_return': backtest_result.metrics.get('total_return', 0),
                'sharpe_ratio': backtest_result.metrics.get('sharpe_ratio', 0),
                'max_drawdown': backtest_result.metrics.get('max_drawdown', 0)
            }
        }
    
    def benchmark_comparison(self,
                           backtest_result: BacktestResult,
                           additional_benchmarks: List[str] = None) -> Dict[str, Any]:
        """
        Compare strategy performance against multiple benchmarks
        """
        benchmarks = [backtest_result.config.benchmark_symbol]
        if additional_benchmarks:
            benchmarks.extend(additional_benchmarks)
        
        comparison_results = {}
        
        for benchmark in benchmarks:
            try:
                benchmark_data = self._get_benchmark_data(
                    benchmark, 
                    backtest_result.start_date, 
                    backtest_result.end_date
                )
                
                benchmark_returns = self._align_benchmark_returns(
                    benchmark_data, 
                    backtest_result.daily_returns.index
                )
                
                # Calculate benchmark metrics
                benchmark_metrics = self._calculate_comprehensive_metrics(
                    benchmark_returns, 
                    benchmark_data, 
                    backtest_result.config,
                    []
                )
                
                # Calculate relative performance
                active_returns = backtest_result.daily_returns - benchmark_returns
                tracking_error = active_returns.std() * np.sqrt(252)
                information_ratio = active_returns.mean() / active_returns.std() * np.sqrt(252) if active_returns.std() > 0 else 0
                
                comparison_results[benchmark] = {
                    'benchmark_metrics': benchmark_metrics,
                    'relative_performance': {
                        'excess_return': backtest_result.metrics.get('annualized_return', 0) - benchmark_metrics.get('annualized_return', 0),
                        'tracking_error': tracking_error,
                        'information_ratio': information_ratio,
                        'beta': self._calculate_beta(backtest_result.daily_returns, benchmark_returns),
                        'alpha': self._calculate_alpha(backtest_result.daily_returns, benchmark_returns, backtest_result.config.risk_free_rate)
                    }
                }
                
            except Exception as e:
                logger.error(f"Error comparing to benchmark {benchmark}: {e}")
                continue
        
        return comparison_results
    
    def _get_market_data(self, universe: List[str], start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
        """Get market data for backtesting"""
        # This would integrate with the actual data provider
        # For now, return mock data structure
        market_data = {}
        
        for ticker in universe:
            # Mock data - in real implementation, fetch from data provider
            dates = pd.date_range(start_date, end_date, freq='D')
            dates = dates[dates.weekday < 5]  # Exclude weekends
            
            # Generate realistic price data
            np.random.seed(hash(ticker) % 2**32)
            returns = np.random.normal(0.0005, 0.02, len(dates))  # ~0.12% daily return, 2% volatility
            prices = 100 * (1 + returns).cumprod()
            
            market_data[ticker] = pd.DataFrame({
                'close': prices,
                'open': prices * (1 + np.random.normal(0, 0.001, len(dates))),
                'high': prices * (1 + np.abs(np.random.normal(0, 0.005, len(dates)))),
                'low': prices * (1 - np.abs(np.random.normal(0, 0.005, len(dates)))),
                'volume': np.random.lognormal(15, 0.5, len(dates)),
                'returns': returns
            }, index=dates)
        
        return market_data
    
    def _get_benchmark_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get benchmark data"""
        # Mock benchmark data
        dates = pd.date_range(start_date, end_date, freq='D')
        dates = dates[dates.weekday < 5]
        
        np.random.seed(42)  # SPY-like returns
        returns = np.random.normal(0.0004, 0.015, len(dates))  # Market-like returns
        prices = 100 * (1 + returns).cumprod()
        
        return pd.DataFrame({
            'close': prices,
            'returns': returns
        }, index=dates)
    
    def _initialize_portfolio(self, initial_capital: float, universe: List[str]) -> Dict[str, Any]:
        """Initialize portfolio state"""
        return {
            'cash': initial_capital,
            'positions': {ticker: 0 for ticker in universe},
            'last_prices': {ticker: 0 for ticker in universe}
        }
    
    def _execute_trades(self, portfolio: Dict[str, Any], signals: Dict[str, float], 
                       market_data: Dict[str, pd.DataFrame], date: datetime, 
                       config: BacktestConfig) -> List[TradeRecord]:
        """Execute trades based on signals"""
        trades = []
        
        for ticker, signal in signals.items():
            if ticker not in market_data or len(market_data[ticker]) == 0:
                continue
            
            current_price = market_data[ticker]['close'].iloc[-1]
            portfolio['last_prices'][ticker] = current_price
            
            # Calculate position size based on signal strength
            target_weight = signal * config.max_position_size
            portfolio_value = self._calculate_portfolio_value(portfolio, market_data, date)
            target_value = target_weight * portfolio_value
            
            current_position_value = portfolio['positions'][ticker] * current_price
            trade_value = target_value - current_position_value
            
            if abs(trade_value) < 100:  # Minimum trade size
                continue
            
            # Calculate shares to trade
            shares_to_trade = int(trade_value / current_price)
            if shares_to_trade == 0:
                continue
            
            # Calculate costs
            commission = abs(shares_to_trade * current_price) * config.commission if config.transaction_costs_enabled else 0
            slippage_cost = abs(shares_to_trade * current_price) * config.slippage if config.transaction_costs_enabled else 0
            
            total_cost = abs(shares_to_trade * current_price) + commission + slippage_cost
            
            # Check if we have enough cash
            if shares_to_trade > 0 and total_cost > portfolio['cash']:
                shares_to_trade = int((portfolio['cash'] - commission - slippage_cost) / current_price)
                if shares_to_trade <= 0:
                    continue
            
            # Execute trade
            if shares_to_trade != 0:
                side = 'long' if shares_to_trade > 0 else 'short'
                
                # Update portfolio
                portfolio['positions'][ticker] += shares_to_trade
                portfolio['cash'] -= shares_to_trade * current_price + commission + slippage_cost
                
                # Record trade
                trade = TradeRecord(
                    ticker=ticker,
                    entry_date=date,
                    exit_date=date,  # Will be updated when position is closed
                    entry_price=current_price,
                    exit_price=current_price,  # Will be updated
                    quantity=abs(shares_to_trade),
                    side=side,
                    pnl=0,  # Will be calculated when closed
                    pnl_pct=0,
                    commission=commission,
                    slippage=slippage_cost,
                    duration_days=0,
                    signal_strength=abs(signal),
                    model_confidence=0.8  # Default confidence
                )
                
                trades.append(trade)
        
        return trades
    
    def _calculate_portfolio_value(self, portfolio: Dict[str, Any], 
                                  market_data: Dict[str, pd.DataFrame], date: datetime) -> float:
        """Calculate current portfolio value"""
        total_value = portfolio['cash']
        
        for ticker, position in portfolio['positions'].items():
            if ticker in market_data and len(market_data[ticker]) > 0:
                current_price = market_data[ticker]['close'].iloc[-1]
                total_value += position * current_price
        
        return total_value
    
    def _calculate_comprehensive_metrics(self, returns: pd.Series, benchmark_data: pd.DataFrame, 
                                       config: BacktestConfig, trades: List[TradeRecord]) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        if len(returns) == 0:
            return {}
        
        metrics = {}
        
        # Basic return metrics
        metrics['total_return'] = (1 + returns).prod() - 1
        metrics['annualized_return'] = (1 + returns).prod() ** (252 / len(returns)) - 1
        metrics['volatility'] = returns.std() * np.sqrt(252)
        
        # Risk-adjusted metrics
        metrics['sharpe_ratio'] = self._calculate_sharpe_ratio(returns, config.risk_free_rate)
        metrics['sortino_ratio'] = self._calculate_sortino_ratio(returns, config.risk_free_rate)
        
        # Drawdown metrics
        cumulative_returns = (1 + returns).cumprod()
        metrics['max_drawdown'] = self._calculate_max_drawdown(cumulative_returns)
        metrics['calmar_ratio'] = metrics['annualized_return'] / abs(metrics['max_drawdown']) if metrics['max_drawdown'] != 0 else 0
        
        # Trading metrics
        if trades:
            winning_trades = [t for t in trades if t.pnl > 0]
            losing_trades = [t for t in trades if t.pnl < 0]
            
            metrics['win_rate'] = len(winning_trades) / len(trades)
            
            if losing_trades:
                avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
                avg_loss = np.mean([t.pnl for t in losing_trades])
                metrics['profit_factor'] = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            else:
                metrics['profit_factor'] = float('inf') if winning_trades else 0
        
        # Directional accuracy
        if len(returns) > 1:
            correct_direction = sum(1 for r in returns if (r > 0) == (returns.mean() > 0))
            metrics['directional_accuracy'] = correct_direction / len(returns)
        
        return metrics
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) == 0 or returns.std() == 0:
            return 0
        
        excess_return = returns.mean() * 252 - risk_free_rate
        return excess_return / (returns.std() * np.sqrt(252))
    
    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float) -> float:
        """Calculate Sortino ratio"""
        if len(returns) == 0:
            return 0
        
        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf')
        
        downside_std = downside_returns.std() * np.sqrt(252)
        if downside_std == 0:
            return 0
        
        return (returns.mean() * 252 - risk_free_rate) / downside_std
    
    def _calculate_max_drawdown(self, cumulative_returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        if len(cumulative_returns) == 0:
            return 0
        
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        return drawdowns.min()
    
    def _calculate_beta(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate beta vs benchmark"""
        if len(returns) != len(benchmark_returns) or benchmark_returns.var() == 0:
            return 1.0
        
        return np.cov(returns, benchmark_returns)[0, 1] / benchmark_returns.var()
    
    def _calculate_alpha(self, returns: pd.Series, benchmark_returns: pd.Series, risk_free_rate: float) -> float:
        """Calculate alpha vs benchmark"""
        beta = self._calculate_beta(returns, benchmark_returns)
        
        portfolio_return = returns.mean() * 252
        benchmark_return = benchmark_returns.mean() * 252
        
        return portfolio_return - (risk_free_rate + beta * (benchmark_return - risk_free_rate))
    
    def _calculate_drawdowns(self, portfolio_values: pd.Series) -> pd.Series:
        """Calculate rolling drawdowns"""
        rolling_max = portfolio_values.expanding().max()
        drawdowns = (portfolio_values - rolling_max) / rolling_max
        return drawdowns
    
    def _calculate_monthly_returns(self, returns: pd.Series) -> pd.DataFrame:
        """Calculate monthly returns matrix"""
        if len(returns) == 0:
            return pd.DataFrame()
        
        monthly = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        # Create matrix with years as rows and months as columns
        monthly_matrix = monthly.to_frame('returns')
        monthly_matrix['year'] = monthly_matrix.index.year
        monthly_matrix['month'] = monthly_matrix.index.month
        
        pivot_table = monthly_matrix.pivot_table(
            values='returns', 
            index='year', 
            columns='month', 
            aggfunc='first'
        )
        
        # Add month names as column headers
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        pivot_table.columns = [month_names[i-1] if i in pivot_table.columns else None 
                              for i in range(1, 13)]
        
        return pivot_table
    
    def _calculate_annual_returns(self, returns: pd.Series) -> pd.Series:
        """Calculate annual returns"""
        if len(returns) == 0:
            return pd.Series()
        
        return returns.resample('Y').apply(lambda x: (1 + x).prod() - 1)
    
    def _calculate_rolling_metrics(self, returns: pd.Series, window: int) -> pd.DataFrame:
        """Calculate rolling performance metrics"""
        if len(returns) < window:
            return pd.DataFrame()
        
        rolling_metrics = pd.DataFrame(index=returns.index)
        
        # Rolling returns
        rolling_metrics['rolling_return'] = returns.rolling(window).apply(lambda x: (1 + x).prod() - 1)
        rolling_metrics['rolling_volatility'] = returns.rolling(window).std() * np.sqrt(252)
        rolling_metrics['rolling_sharpe'] = (returns.rolling(window).mean() * 252 - 0.02) / (returns.rolling(window).std() * np.sqrt(252))
        
        # Rolling max drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.rolling(window).max()
        rolling_metrics['rolling_max_drawdown'] = ((cumulative - rolling_max) / rolling_max).rolling(window).min()
        
        return rolling_metrics
    
    def _calculate_sector_performance(self, trades: List[TradeRecord], universe: List[str]) -> Dict[str, float]:
        """Calculate performance by sector (mock implementation)"""
        # In real implementation, would map tickers to sectors
        sector_performance = {}
        
        # Mock sector mapping
        sector_map = {ticker: f"Sector_{hash(ticker) % 5}" for ticker in universe}
        
        for sector in set(sector_map.values()):
            sector_trades = [t for t in trades if sector_map.get(t.ticker) == sector]
            if sector_trades:
                sector_pnl = sum(t.pnl for t in sector_trades)
                sector_performance[sector] = sector_pnl
        
        return sector_performance
    
    def _calculate_risk_metrics(self, returns: pd.Series, benchmark_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive risk metrics"""
        risk_metrics = {}
        
        if len(returns) == 0:
            return risk_metrics
        
        # Value at Risk (VaR)
        risk_metrics['var_95'] = returns.quantile(0.05)
        risk_metrics['var_99'] = returns.quantile(0.01)
        
        # Conditional Value at Risk (CVaR)
        var_95 = risk_metrics['var_95']
        risk_metrics['cvar_95'] = returns[returns <= var_95].mean()
        
        # Skewness and Kurtosis
        risk_metrics['skewness'] = returns.skew()
        risk_metrics['kurtosis'] = returns.kurtosis()
        
        # Maximum daily loss
        risk_metrics['max_daily_loss'] = returns.min()
        
        return risk_metrics
    
    def _calculate_performance_attribution(self, trades: List[TradeRecord]) -> Dict[str, float]:
        """Calculate performance attribution"""
        if not trades:
            return {}
        
        attribution = {}
        
        # By holding period
        short_term_trades = [t for t in trades if t.duration_days <= 5]
        medium_term_trades = [t for t in trades if 5 < t.duration_days <= 30]
        long_term_trades = [t for t in trades if t.duration_days > 30]
        
        attribution['short_term_pnl'] = sum(t.pnl for t in short_term_trades)
        attribution['medium_term_pnl'] = sum(t.pnl for t in medium_term_trades)
        attribution['long_term_pnl'] = sum(t.pnl for t in long_term_trades)
        
        # By signal strength
        high_confidence_trades = [t for t in trades if t.signal_strength > 0.7]
        medium_confidence_trades = [t for t in trades if 0.3 < t.signal_strength <= 0.7]
        low_confidence_trades = [t for t in trades if t.signal_strength <= 0.3]
        
        attribution['high_confidence_pnl'] = sum(t.pnl for t in high_confidence_trades)
        attribution['medium_confidence_pnl'] = sum(t.pnl for t in medium_confidence_trades)
        attribution['low_confidence_pnl'] = sum(t.pnl for t in low_confidence_trades)
        
        return attribution
    
    def _align_benchmark_returns(self, benchmark_data: pd.DataFrame, target_index: pd.DatetimeIndex) -> pd.Series:
        """Align benchmark returns to strategy returns"""
        benchmark_returns = benchmark_data['returns']
        return benchmark_returns.reindex(target_index, method='ffill').fillna(0)
    
    def _aggregate_walk_forward_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Aggregate walk-forward results"""
        if not results:
            return {}
        
        all_metrics = {}
        for result_dict in results:
            result = result_dict['result']
            for metric, value in result.metrics.items():
                if metric not in all_metrics:
                    all_metrics[metric] = []
                all_metrics[metric].append(value)
        
        aggregated = {}
        for metric, values in all_metrics.items():
            aggregated[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }
        
        return aggregated
    
    def _calculate_stability_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """Calculate strategy stability metrics"""
        if len(results) < 2:
            return {}
        
        returns = [r['result'].metrics.get('total_return', 0) for r in results]
        sharpe_ratios = [r['result'].metrics.get('sharpe_ratio', 0) for r in results]
        
        stability = {
            'return_consistency': 1 - (np.std(returns) / np.mean(returns)) if np.mean(returns) != 0 else 0,
            'sharpe_consistency': 1 - (np.std(sharpe_ratios) / np.mean(sharpe_ratios)) if np.mean(sharpe_ratios) != 0 else 0,
            'positive_periods': sum(1 for r in returns if r > 0) / len(returns)
        }
        
        return stability
    
    def _calculate_performance_consistency(self, results: List[Dict]) -> Dict[str, Any]:
        """Calculate performance consistency metrics"""
        metrics_over_time = {}
        
        for result_dict in results:
            period = f"{result_dict['test_start'].strftime('%Y-%m')} to {result_dict['test_end'].strftime('%Y-%m')}"
            metrics_over_time[period] = result_dict['result'].metrics
        
        return metrics_over_time
    
    def save_backtest_report(self, result: BacktestResult, filepath: str):
        """Save comprehensive backtest report"""
        report = {
            'metadata': {
                'generated_at': datetime.utcnow().isoformat(),
                'backtest_period': {
                    'start_date': result.start_date.isoformat(),
                    'end_date': result.end_date.isoformat()
                }
            },
            'configuration': {
                'initial_capital': result.initial_capital,
                'commission': result.config.commission,
                'slippage': result.config.slippage,
                'benchmark': result.config.benchmark_symbol
            },
            'summary': {
                'total_return': result.metrics.get('total_return', 0),
                'annualized_return': result.metrics.get('annualized_return', 0),
                'sharpe_ratio': result.metrics.get('sharpe_ratio', 0),
                'max_drawdown': result.metrics.get('max_drawdown', 0),
                'total_trades': result.total_trades,
                'win_rate': result.metrics.get('win_rate', 0)
            },
            'detailed_metrics': result.metrics,
            'risk_metrics': result.risk_metrics,
            'trades_summary': {
                'total_trades': result.total_trades,
                'winning_trades': result.winning_trades,
                'losing_trades': result.losing_trades,
                'average_trade_pnl': np.mean([t.pnl for t in result.trades]) if result.trades else 0
            }
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Backtest report saved to {filepath}")


# Global backtest engine instance
_backtest_engine: Optional[BacktestEngine] = None

def get_backtest_engine() -> BacktestEngine:
    """Get global backtest engine instance"""
    global _backtest_engine
    if _backtest_engine is None:
        _backtest_engine = BacktestEngine()
    return _backtest_engine