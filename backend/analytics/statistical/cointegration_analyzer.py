"""
Cointegration Analysis System
Implements statistical tests for identifying cointegrated pairs and relationships
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
import warnings
from scipy import stats
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import statsmodels.api as sm

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class TestType(Enum):
    """Types of cointegration tests"""
    ENGLE_GRANGER = "engle_granger"
    JOHANSEN = "johansen"
    PHILLIPS_OULIARIS = "phillips_ouliaris"
    ADF = "augmented_dickey_fuller"


@dataclass
class CointegrationResult:
    """Result of cointegration test"""
    ticker1: str
    ticker2: str
    test_type: TestType
    is_cointegrated: bool
    confidence_level: float
    test_statistic: float
    p_value: float
    critical_values: Dict[str, float]
    hedge_ratio: float
    half_life: float  # Mean reversion half-life in days
    spread_stats: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PairTradingSignal:
    """Trading signal for cointegrated pair"""
    pair: Tuple[str, str]
    signal_type: str  # 'long_spread', 'short_spread', 'close'
    z_score: float
    spread_value: float
    entry_price_1: float
    entry_price_2: float
    hedge_ratio: float
    confidence: float
    expected_profit: float
    risk_metrics: Dict[str, float]


class CointegrationAnalyzer:
    """
    Comprehensive cointegration analysis for statistical arbitrage
    """
    
    def __init__(self):
        self.cointegrated_pairs: List[CointegrationResult] = []
        self.pair_history: Dict[Tuple[str, str], List[float]] = {}
        self.z_score_thresholds = {
            'entry': 2.0,
            'exit': 0.5,
            'stop_loss': 3.0
        }
        
    def test_cointegration(
        self,
        series1: pd.Series,
        series2: pd.Series,
        test_type: TestType = TestType.ENGLE_GRANGER,
        confidence_level: float = 0.95
    ) -> CointegrationResult:
        """
        Test for cointegration between two time series
        
        Args:
            series1: First price series
            series2: Second price series
            test_type: Type of cointegration test
            confidence_level: Confidence level for test
            
        Returns:
            Cointegration test result
        """
        # Ensure series are aligned
        series1, series2 = self._align_series(series1, series2)
        
        if test_type == TestType.ENGLE_GRANGER:
            result = self._engle_granger_test(
                series1, series2, confidence_level
            )
        elif test_type == TestType.JOHANSEN:
            result = self._johansen_test(
                series1, series2, confidence_level
            )
        elif test_type == TestType.PHILLIPS_OULIARIS:
            result = self._phillips_ouliaris_test(
                series1, series2, confidence_level
            )
        else:
            result = self._engle_granger_test(
                series1, series2, confidence_level
            )
        
        # Calculate additional metrics
        result.half_life = self._calculate_half_life(series1, series2, result.hedge_ratio)
        result.spread_stats = self._calculate_spread_statistics(
            series1, series2, result.hedge_ratio
        )
        
        # Store if cointegrated
        if result.is_cointegrated:
            self.cointegrated_pairs.append(result)
            self.pair_history[(result.ticker1, result.ticker2)] = (
                self._calculate_spread(series1, series2, result.hedge_ratio).tolist()
            )
        
        return result
    
    def find_cointegrated_pairs(
        self,
        price_data: pd.DataFrame,
        min_correlation: float = 0.5,
        max_pairs: int = 100,
        test_type: TestType = TestType.ENGLE_GRANGER
    ) -> List[CointegrationResult]:
        """
        Find all cointegrated pairs in a dataset
        
        Args:
            price_data: DataFrame with price data (columns are tickers)
            min_correlation: Minimum correlation to test for cointegration
            max_pairs: Maximum number of pairs to return
            test_type: Cointegration test type
            
        Returns:
            List of cointegrated pairs
        """
        tickers = price_data.columns.tolist()
        cointegrated_pairs = []
        
        # Calculate correlation matrix
        correlation_matrix = price_data.corr()
        
        # Test pairs with sufficient correlation
        tested_pairs = set()
        
        for i, ticker1 in enumerate(tickers):
            for j, ticker2 in enumerate(tickers):
                if i >= j:  # Skip diagonal and duplicates
                    continue
                
                pair = tuple(sorted([ticker1, ticker2]))
                if pair in tested_pairs:
                    continue
                
                # Check correlation threshold
                if abs(correlation_matrix.loc[ticker1, ticker2]) < min_correlation:
                    continue
                
                tested_pairs.add(pair)
                
                # Test for cointegration
                try:
                    result = self.test_cointegration(
                        price_data[ticker1],
                        price_data[ticker2],
                        test_type=test_type
                    )
                    
                    if result.is_cointegrated:
                        cointegrated_pairs.append(result)
                        
                        if len(cointegrated_pairs) >= max_pairs:
                            break
                
                except Exception as e:
                    logger.debug(f"Error testing {ticker1}-{ticker2}: {e}")
            
            if len(cointegrated_pairs) >= max_pairs:
                break
        
        # Sort by confidence level
        cointegrated_pairs.sort(
            key=lambda x: x.confidence_level,
            reverse=True
        )
        
        logger.info(f"Found {len(cointegrated_pairs)} cointegrated pairs")
        return cointegrated_pairs[:max_pairs]
    
    def generate_trading_signals(
        self,
        current_prices: Dict[str, float],
        lookback_prices: pd.DataFrame
    ) -> List[PairTradingSignal]:
        """
        Generate trading signals for cointegrated pairs
        
        Args:
            current_prices: Current prices for all tickers
            lookback_prices: Historical prices for spread calculation
            
        Returns:
            List of trading signals
        """
        signals = []
        
        for pair_result in self.cointegrated_pairs:
            ticker1, ticker2 = pair_result.ticker1, pair_result.ticker2
            
            # Check if we have current prices
            if ticker1 not in current_prices or ticker2 not in current_prices:
                continue
            
            # Calculate current spread
            price1 = current_prices[ticker1]
            price2 = current_prices[ticker2]
            current_spread = price1 - pair_result.hedge_ratio * price2
            
            # Calculate historical spread statistics
            if ticker1 in lookback_prices.columns and ticker2 in lookback_prices.columns:
                historical_spread = self._calculate_spread(
                    lookback_prices[ticker1],
                    lookback_prices[ticker2],
                    pair_result.hedge_ratio
                )
                
                spread_mean = historical_spread.mean()
                spread_std = historical_spread.std()
                
                # Calculate z-score
                z_score = (current_spread - spread_mean) / spread_std if spread_std > 0 else 0
                
                # Generate signal based on z-score
                signal = self._generate_signal_from_zscore(
                    z_score,
                    (ticker1, ticker2),
                    current_spread,
                    price1,
                    price2,
                    pair_result.hedge_ratio,
                    pair_result.confidence_level
                )
                
                if signal:
                    # Calculate risk metrics
                    signal.risk_metrics = self._calculate_risk_metrics(
                        historical_spread,
                        current_spread,
                        pair_result.half_life
                    )
                    
                    signals.append(signal)
        
        return signals
    
    def calculate_portfolio_cointegration(
        self,
        price_data: pd.DataFrame,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Test if a portfolio of assets is cointegrated
        
        Args:
            price_data: Price data for portfolio assets
            weights: Portfolio weights (equal weight if None)
            
        Returns:
            Portfolio cointegration analysis
        """
        n_assets = len(price_data.columns)
        
        # Use equal weights if not provided
        if weights is None:
            weights = {col: 1/n_assets for col in price_data.columns}
        
        # Johansen test for multiple time series
        try:
            result = coint_johansen(price_data, det_order=0, k_ar_diff=1)
            
            # Check if portfolio is cointegrated
            trace_stat = result.lr1  # Trace statistic
            critical_values = result.cvt  # Critical values
            
            # Test at 95% confidence level
            is_cointegrated = trace_stat[0] > critical_values[0, 1]
            
            # Extract cointegrating vectors
            cointegrating_vectors = result.evec
            
            # Calculate portfolio spread using first cointegrating vector
            if is_cointegrated:
                spread = price_data @ cointegrating_vectors[:, 0]
                
                # Test spread for stationarity
                adf_result = adfuller(spread, autolag='AIC')
                
                return {
                    'is_cointegrated': is_cointegrated,
                    'n_cointegrating_relations': sum(trace_stat > critical_values[:, 1]),
                    'trace_statistics': trace_stat.tolist(),
                    'critical_values': critical_values.tolist(),
                    'cointegrating_vectors': cointegrating_vectors.tolist(),
                    'spread_stationarity': {
                        'is_stationary': adf_result[1] < 0.05,
                        'adf_statistic': adf_result[0],
                        'p_value': adf_result[1]
                    },
                    'optimal_weights': self._calculate_optimal_weights(
                        cointegrating_vectors[:, 0]
                    )
                }
            
        except Exception as e:
            logger.error(f"Portfolio cointegration test failed: {e}")
        
        return {
            'is_cointegrated': False,
            'error': 'Test failed or insufficient data'
        }
    
    # Test implementations
    
    def _engle_granger_test(
        self,
        series1: pd.Series,
        series2: pd.Series,
        confidence_level: float
    ) -> CointegrationResult:
        """Engle-Granger two-step cointegration test"""
        
        # Step 1: Regress series1 on series2
        model = OLS(series1, sm.add_constant(series2))
        results = model.fit()
        hedge_ratio = results.params[1]
        
        # Step 2: Test residuals for stationarity
        residuals = series1 - hedge_ratio * series2
        adf_result = adfuller(residuals, autolag='AIC')
        
        # Use statsmodels coint function for comparison
        coint_result = coint(series1, series2)
        
        # Determine if cointegrated
        critical_value_index = {0.90: 0, 0.95: 1, 0.99: 2}.get(confidence_level, 1)
        is_cointegrated = coint_result[0] < coint_result[2][critical_value_index]
        
        return CointegrationResult(
            ticker1=series1.name if hasattr(series1, 'name') else 'series1',
            ticker2=series2.name if hasattr(series2, 'name') else 'series2',
            test_type=TestType.ENGLE_GRANGER,
            is_cointegrated=is_cointegrated,
            confidence_level=confidence_level if is_cointegrated else 0,
            test_statistic=coint_result[0],
            p_value=coint_result[1],
            critical_values={
                '90%': coint_result[2][0],
                '95%': coint_result[2][1],
                '99%': coint_result[2][2]
            },
            hedge_ratio=hedge_ratio,
            half_life=0,  # Calculated later
            spread_stats={}  # Calculated later
        )
    
    def _johansen_test(
        self,
        series1: pd.Series,
        series2: pd.Series,
        confidence_level: float
    ) -> CointegrationResult:
        """Johansen cointegration test"""
        
        # Combine series
        data = pd.DataFrame({
            'series1': series1,
            'series2': series2
        })
        
        # Perform Johansen test
        result = coint_johansen(data, det_order=0, k_ar_diff=1)
        
        # Check trace statistic
        confidence_index = {0.90: 0, 0.95: 1, 0.99: 2}.get(confidence_level, 1)
        trace_stat = result.lr1[0]
        critical_value = result.cvt[0, confidence_index]
        
        is_cointegrated = trace_stat > critical_value
        
        # Get hedge ratio from cointegrating vector
        if is_cointegrated:
            cointegrating_vector = result.evec[:, 0]
            hedge_ratio = -cointegrating_vector[1] / cointegrating_vector[0]
        else:
            hedge_ratio = 0
        
        return CointegrationResult(
            ticker1=series1.name if hasattr(series1, 'name') else 'series1',
            ticker2=series2.name if hasattr(series2, 'name') else 'series2',
            test_type=TestType.JOHANSEN,
            is_cointegrated=is_cointegrated,
            confidence_level=confidence_level if is_cointegrated else 0,
            test_statistic=trace_stat,
            p_value=0,  # Johansen doesn't provide p-value directly
            critical_values={
                '90%': result.cvt[0, 0],
                '95%': result.cvt[0, 1],
                '99%': result.cvt[0, 2]
            },
            hedge_ratio=hedge_ratio,
            half_life=0,
            spread_stats={}
        )
    
    def _phillips_ouliaris_test(
        self,
        series1: pd.Series,
        series2: pd.Series,
        confidence_level: float
    ) -> CointegrationResult:
        """Phillips-Ouliaris cointegration test"""
        
        # This is similar to Engle-Granger but uses different critical values
        # For simplicity, we'll use the Engle-Granger implementation
        return self._engle_granger_test(series1, series2, confidence_level)
    
    # Helper methods
    
    def _align_series(
        self,
        series1: pd.Series,
        series2: pd.Series
    ) -> Tuple[pd.Series, pd.Series]:
        """Align two series to have same index"""
        common_index = series1.index.intersection(series2.index)
        return series1[common_index], series2[common_index]
    
    def _calculate_spread(
        self,
        series1: pd.Series,
        series2: pd.Series,
        hedge_ratio: float
    ) -> pd.Series:
        """Calculate spread between two series"""
        return series1 - hedge_ratio * series2
    
    def _calculate_half_life(
        self,
        series1: pd.Series,
        series2: pd.Series,
        hedge_ratio: float
    ) -> float:
        """Calculate mean reversion half-life using OU process"""
        
        spread = self._calculate_spread(series1, series2, hedge_ratio)
        
        # Lag spread
        spread_lag = spread.shift(1)
        spread_diff = spread - spread_lag
        
        # Remove NaN
        spread_lag = spread_lag[1:]
        spread_diff = spread_diff[1:]
        
        # Regress spread diff on lagged spread
        try:
            model = OLS(spread_diff, spread_lag)
            results = model.fit()
            theta = -results.params[0]  # Mean reversion speed
            
            if theta > 0:
                half_life = np.log(2) / theta
                return min(half_life, 365)  # Cap at 1 year
        except:
            pass
        
        return 30  # Default to 30 days
    
    def _calculate_spread_statistics(
        self,
        series1: pd.Series,
        series2: pd.Series,
        hedge_ratio: float
    ) -> Dict[str, float]:
        """Calculate spread statistics"""
        
        spread = self._calculate_spread(series1, series2, hedge_ratio)
        
        return {
            'mean': float(spread.mean()),
            'std': float(spread.std()),
            'min': float(spread.min()),
            'max': float(spread.max()),
            'current': float(spread.iloc[-1]) if len(spread) > 0 else 0,
            'skewness': float(spread.skew()),
            'kurtosis': float(spread.kurtosis()),
            'sharpe_ratio': float(spread.mean() / spread.std()) if spread.std() > 0 else 0
        }
    
    def _generate_signal_from_zscore(
        self,
        z_score: float,
        pair: Tuple[str, str],
        spread_value: float,
        price1: float,
        price2: float,
        hedge_ratio: float,
        confidence: float
    ) -> Optional[PairTradingSignal]:
        """Generate trading signal based on z-score"""
        
        signal_type = None
        expected_profit = 0
        
        # Determine signal type
        if abs(z_score) > self.z_score_thresholds['stop_loss']:
            # Stop loss triggered
            signal_type = 'close'
        elif abs(z_score) < self.z_score_thresholds['exit']:
            # Close position
            signal_type = 'close'
        elif z_score > self.z_score_thresholds['entry']:
            # Spread too high, short spread (long stock2, short stock1)
            signal_type = 'short_spread'
            expected_profit = abs(z_score - self.z_score_thresholds['exit']) * 0.01
        elif z_score < -self.z_score_thresholds['entry']:
            # Spread too low, long spread (long stock1, short stock2)
            signal_type = 'long_spread'
            expected_profit = abs(z_score + self.z_score_thresholds['exit']) * 0.01
        
        if signal_type:
            return PairTradingSignal(
                pair=pair,
                signal_type=signal_type,
                z_score=z_score,
                spread_value=spread_value,
                entry_price_1=price1,
                entry_price_2=price2,
                hedge_ratio=hedge_ratio,
                confidence=confidence,
                expected_profit=expected_profit,
                risk_metrics={}
            )
        
        return None
    
    def _calculate_risk_metrics(
        self,
        historical_spread: pd.Series,
        current_spread: float,
        half_life: float
    ) -> Dict[str, float]:
        """Calculate risk metrics for pair trade"""
        
        # Historical volatility
        spread_volatility = historical_spread.std()
        
        # Value at Risk (95% confidence)
        var_95 = np.percentile(historical_spread.diff().dropna(), 5)
        
        # Maximum drawdown
        cumulative = (historical_spread - historical_spread.mean()).cumsum()
        running_max = cumulative.expanding().max()
        drawdown = cumulative - running_max
        max_drawdown = drawdown.min()
        
        # Probability of profit (assuming mean reversion)
        distance_from_mean = abs(current_spread - historical_spread.mean())
        prob_profit = 1 - np.exp(-2 * distance_from_mean / (spread_volatility * np.sqrt(half_life)))
        
        return {
            'spread_volatility': float(spread_volatility),
            'value_at_risk_95': float(var_95),
            'max_drawdown': float(max_drawdown),
            'probability_of_profit': float(prob_profit),
            'expected_holding_days': float(half_life)
        }
    
    def _calculate_optimal_weights(
        self,
        cointegrating_vector: np.ndarray
    ) -> Dict[int, float]:
        """Calculate optimal portfolio weights from cointegrating vector"""
        
        # Normalize to sum to 1
        normalized = cointegrating_vector / np.sum(np.abs(cointegrating_vector))
        
        return {i: float(w) for i, w in enumerate(normalized)}


# Statistical arbitrage strategy using cointegration
class StatisticalArbitrageStrategy:
    """
    Complete statistical arbitrage strategy using cointegration
    """
    
    def __init__(self, analyzer: CointegrationAnalyzer):
        self.analyzer = analyzer
        self.active_positions: Dict[Tuple[str, str], Dict] = {}
        self.trade_history: List[Dict] = []
        
    async def execute_strategy(
        self,
        price_data: pd.DataFrame,
        current_prices: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """
        Execute statistical arbitrage strategy
        
        Args:
            price_data: Historical price data
            current_prices: Current market prices
            
        Returns:
            List of trade recommendations
        """
        recommendations = []
        
        # Find cointegrated pairs if not already done
        if not self.analyzer.cointegrated_pairs:
            self.analyzer.find_cointegrated_pairs(price_data)
        
        # Generate signals
        signals = self.analyzer.generate_trading_signals(
            current_prices,
            price_data
        )
        
        for signal in signals:
            # Check if we have an active position
            if signal.pair in self.active_positions:
                position = self.active_positions[signal.pair]
                
                if signal.signal_type == 'close':
                    # Close position
                    recommendations.append({
                        'action': 'close',
                        'pair': signal.pair,
                        'reason': f'Z-score at {signal.z_score:.2f}',
                        'expected_pnl': self._calculate_pnl(position, current_prices)
                    })
                    
                    # Record trade
                    self.trade_history.append({
                        'pair': signal.pair,
                        'entry_time': position['entry_time'],
                        'exit_time': datetime.utcnow(),
                        'pnl': self._calculate_pnl(position, current_prices)
                    })
                    
                    # Remove from active positions
                    del self.active_positions[signal.pair]
                    
            else:
                # Check for new position
                if signal.signal_type in ['long_spread', 'short_spread']:
                    recommendations.append({
                        'action': 'open',
                        'pair': signal.pair,
                        'direction': signal.signal_type,
                        'z_score': signal.z_score,
                        'hedge_ratio': signal.hedge_ratio,
                        'confidence': signal.confidence,
                        'risk_metrics': signal.risk_metrics,
                        'position_size': self._calculate_position_size(signal)
                    })
                    
                    # Record position
                    self.active_positions[signal.pair] = {
                        'entry_time': datetime.utcnow(),
                        'entry_prices': {
                            signal.pair[0]: signal.entry_price_1,
                            signal.pair[1]: signal.entry_price_2
                        },
                        'direction': signal.signal_type,
                        'hedge_ratio': signal.hedge_ratio
                    }
        
        return recommendations
    
    def _calculate_pnl(
        self,
        position: Dict,
        current_prices: Dict[str, float]
    ) -> float:
        """Calculate P&L for a position"""
        
        ticker1, ticker2 = position['pair']
        entry_price1 = position['entry_prices'][ticker1]
        entry_price2 = position['entry_prices'][ticker2]
        current_price1 = current_prices.get(ticker1, entry_price1)
        current_price2 = current_prices.get(ticker2, entry_price2)
        
        if position['direction'] == 'long_spread':
            # Long ticker1, short ticker2
            pnl = (current_price1 - entry_price1) - position['hedge_ratio'] * (current_price2 - entry_price2)
        else:
            # Short ticker1, long ticker2
            pnl = -(current_price1 - entry_price1) + position['hedge_ratio'] * (current_price2 - entry_price2)
        
        return pnl
    
    def _calculate_position_size(self, signal: PairTradingSignal) -> float:
        """Calculate position size based on Kelly criterion"""
        
        # Simplified Kelly criterion
        win_probability = signal.risk_metrics.get('probability_of_profit', 0.5)
        win_loss_ratio = 1.5  # Assume 1.5:1 reward/risk
        
        kelly_fraction = (win_probability * win_loss_ratio - (1 - win_probability)) / win_loss_ratio
        
        # Apply safety factor
        safe_kelly = max(0, min(0.25, kelly_fraction * 0.25))
        
        return safe_kelly