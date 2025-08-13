"""
Market Regime Detection System
Identifies market conditions using statistical and ML approaches
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
import logging
from scipy import stats
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classifications"""
    BULL_QUIET = "bull_quiet"           # Low volatility uptrend
    BULL_VOLATILE = "bull_volatile"     # High volatility uptrend
    BEAR_QUIET = "bear_quiet"           # Low volatility downtrend
    BEAR_VOLATILE = "bear_volatile"     # High volatility downtrend
    SIDEWAYS_LOW_VOL = "sideways_low"   # Range-bound, low volatility
    SIDEWAYS_HIGH_VOL = "sideways_high" # Range-bound, high volatility
    CRISIS = "crisis"                    # Extreme volatility/drawdown
    RECOVERY = "recovery"                # Post-crisis recovery phase


class RegimeDetector:
    """
    Advanced market regime detection using multiple methodologies
    """
    
    def __init__(self):
        self.hmm_model = None
        self.gmm_model = None
        self.regime_history = []
        self.transition_matrix = None
        self.regime_features = {}
        
    def detect_regime(
        self,
        market_data: pd.DataFrame,
        method: str = 'ensemble'
    ) -> Dict[str, Any]:
        """
        Detect current market regime using specified method
        
        Args:
            market_data: DataFrame with OHLCV data for market index
            method: Detection method ('hmm', 'gmm', 'statistical', 'ensemble')
        
        Returns:
            Dictionary with regime classification and confidence
        """
        # Calculate regime indicators
        indicators = self._calculate_regime_indicators(market_data)
        
        if method == 'hmm':
            regime = self._detect_hmm_regime(indicators)
        elif method == 'gmm':
            regime = self._detect_gmm_regime(indicators)
        elif method == 'statistical':
            regime = self._detect_statistical_regime(indicators)
        elif method == 'ensemble':
            regime = self._detect_ensemble_regime(indicators)
        else:
            raise ValueError(f"Unknown method: {method}")
            
        # Calculate regime stability and transition probability
        stability = self._calculate_regime_stability(regime, indicators)
        transition_probs = self._calculate_transition_probabilities(regime)
        
        # Generate regime characteristics
        characteristics = self._generate_regime_characteristics(regime, indicators)
        
        return {
            'current_regime': regime['regime'],
            'confidence': regime['confidence'],
            'stability': stability,
            'transition_probabilities': transition_probs,
            'characteristics': characteristics,
            'indicators': indicators,
            'historical_context': self._get_historical_context(regime)
        }
    
    def _calculate_regime_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate key indicators for regime detection"""
        
        # Returns
        returns = data['close'].pct_change()
        log_returns = np.log(data['close'] / data['close'].shift(1))
        
        # Volatility measures
        realized_vol = returns.rolling(20).std() * np.sqrt(252)
        garch_vol = self._estimate_garch_volatility(returns)
        
        # Trend indicators
        sma_20 = data['close'].rolling(20).mean()
        sma_50 = data['close'].rolling(50).mean()
        sma_200 = data['close'].rolling(200).mean()
        
        trend_strength = (data['close'].iloc[-1] - sma_200.iloc[-1]) / sma_200.iloc[-1]
        trend_consistency = self._calculate_trend_consistency(data['close'])
        
        # Market breadth (if available)
        advance_decline = self._calculate_advance_decline_ratio(data)
        new_highs_lows = self._calculate_new_highs_lows_ratio(data)
        
        # Volatility regime
        vol_percentile = stats.percentileofscore(
            realized_vol.dropna(),
            realized_vol.iloc[-1]
        )
        
        # Drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        current_drawdown = drawdown.iloc[-1]
        max_drawdown_30d = drawdown.tail(30).min()
        
        # Correlation breakdown
        correlation_stability = self._calculate_correlation_stability(data)
        
        # Skewness and kurtosis (tail risks)
        skewness_20d = returns.tail(20).skew()
        kurtosis_20d = returns.tail(20).kurtosis()
        
        # Volume analysis
        volume_trend = self._analyze_volume_pattern(data)
        
        # Fear indicators
        put_call_ratio = self._estimate_put_call_ratio(realized_vol.iloc[-1])
        
        return {
            'returns_20d': returns.tail(20).mean() * 252,
            'returns_60d': returns.tail(60).mean() * 252,
            'volatility': realized_vol.iloc[-1],
            'volatility_percentile': vol_percentile,
            'garch_volatility': garch_vol,
            'trend_strength': trend_strength,
            'trend_consistency': trend_consistency,
            'sma_cross': 1 if sma_20.iloc[-1] > sma_50.iloc[-1] else -1,
            'golden_cross': 1 if sma_50.iloc[-1] > sma_200.iloc[-1] else -1,
            'current_drawdown': current_drawdown,
            'max_drawdown_30d': max_drawdown_30d,
            'advance_decline': advance_decline,
            'new_highs_lows': new_highs_lows,
            'correlation_stability': correlation_stability,
            'skewness': skewness_20d,
            'kurtosis': kurtosis_20d,
            'volume_trend': volume_trend,
            'put_call_ratio': put_call_ratio
        }
    
    def _detect_hmm_regime(self, indicators: Dict) -> Dict:
        """Detect regime using Hidden Markov Model"""
        try:
            from hmmlearn import hmm
            
            # Prepare features for HMM
            features = np.array([
                [indicators['returns_20d']],
                [indicators['volatility']],
                [indicators['trend_strength']]
            ]).T
            
            # Train or use existing HMM
            if self.hmm_model is None:
                self.hmm_model = hmm.GaussianHMM(
                    n_components=4,  # 4 basic regimes
                    covariance_type="full",
                    n_iter=100
                )
                # In production, train on historical data
                # self.hmm_model.fit(historical_features)
            
            # Predict current state
            # state = self.hmm_model.predict(features)[-1]
            # For now, use rule-based approximation
            state = self._rule_based_regime(indicators)
            
            return {
                'regime': state,
                'confidence': 0.75,
                'method': 'hmm'
            }
            
        except ImportError:
            logger.warning("hmmlearn not installed, using fallback")
            return self._detect_statistical_regime(indicators)
    
    def _detect_gmm_regime(self, indicators: Dict) -> Dict:
        """Detect regime using Gaussian Mixture Model"""
        
        # Prepare features
        features = np.array([
            indicators['returns_20d'],
            indicators['volatility'],
            indicators['skewness'],
            indicators['current_drawdown']
        ]).reshape(1, -1)
        
        # Initialize GMM if needed
        if self.gmm_model is None:
            self.gmm_model = GaussianMixture(
                n_components=6,  # More granular regimes
                covariance_type='full',
                random_state=42
            )
            # In production, fit on historical data
            # self.gmm_model.fit(historical_features)
        
        # For now, use statistical approach
        regime = self._statistical_regime_classification(indicators)
        
        return {
            'regime': regime,
            'confidence': 0.70,
            'method': 'gmm'
        }
    
    def _detect_statistical_regime(self, indicators: Dict) -> Dict:
        """Detect regime using statistical rules"""
        
        regime = self._statistical_regime_classification(indicators)
        confidence = self._calculate_statistical_confidence(indicators)
        
        return {
            'regime': regime,
            'confidence': confidence,
            'method': 'statistical'
        }
    
    def _detect_ensemble_regime(self, indicators: Dict) -> Dict:
        """Ensemble approach combining multiple methods"""
        
        # Get predictions from each method
        methods_results = [
            self._detect_statistical_regime(indicators),
            # self._detect_hmm_regime(indicators),
            # self._detect_gmm_regime(indicators)
        ]
        
        # For now, use the statistical method as primary
        primary_regime = methods_results[0]['regime']
        
        # Calculate ensemble confidence
        confidence = np.mean([r['confidence'] for r in methods_results])
        
        return {
            'regime': primary_regime,
            'confidence': confidence,
            'method': 'ensemble',
            'component_predictions': methods_results
        }
    
    def _statistical_regime_classification(self, indicators: Dict) -> MarketRegime:
        """Classify regime based on statistical rules"""
        
        returns = indicators['returns_20d']
        volatility = indicators['volatility']
        drawdown = indicators['current_drawdown']
        trend = indicators['trend_strength']
        
        # Crisis detection
        if drawdown < -0.15 and volatility > 0.35:
            return MarketRegime.CRISIS
        
        # Recovery detection
        if drawdown < -0.05 and returns > 0.15 and indicators['returns_60d'] < 0:
            return MarketRegime.RECOVERY
        
        # Volatility classification
        high_vol = volatility > 0.20
        
        # Trend classification
        if trend > 0.05 and returns > 0.10:
            return MarketRegime.BULL_VOLATILE if high_vol else MarketRegime.BULL_QUIET
        elif trend < -0.05 and returns < -0.10:
            return MarketRegime.BEAR_VOLATILE if high_vol else MarketRegime.BEAR_QUIET
        else:
            return MarketRegime.SIDEWAYS_HIGH_VOL if high_vol else MarketRegime.SIDEWAYS_LOW_VOL
    
    def _calculate_statistical_confidence(self, indicators: Dict) -> float:
        """Calculate confidence in statistical regime detection"""
        
        confidence_factors = []
        
        # Trend clarity
        trend_clarity = abs(indicators['trend_strength'])
        confidence_factors.append(min(trend_clarity * 2, 1.0))
        
        # Volatility stability
        vol_stability = 1.0 - abs(indicators['volatility'] - 0.15) / 0.35
        confidence_factors.append(max(0, vol_stability))
        
        # Consistency of signals
        signal_consistency = (
            (indicators['sma_cross'] * indicators['golden_cross'] > 0) * 0.3 +
            (abs(indicators['trend_consistency']) > 0.6) * 0.3 +
            (indicators['correlation_stability'] > 0.7) * 0.4
        )
        confidence_factors.append(signal_consistency)
        
        return np.mean(confidence_factors)
    
    def _calculate_regime_stability(
        self,
        current_regime: Dict,
        indicators: Dict
    ) -> float:
        """Calculate stability of current regime"""
        
        # Factors indicating regime stability
        stability_factors = []
        
        # Volatility stability
        vol_in_normal_range = 0.10 < indicators['volatility'] < 0.25
        stability_factors.append(float(vol_in_normal_range))
        
        # Trend consistency
        stability_factors.append(indicators['trend_consistency'])
        
        # Correlation stability
        stability_factors.append(indicators['correlation_stability'])
        
        # Low tail risk
        low_tail_risk = abs(indicators['skewness']) < 1 and indicators['kurtosis'] < 3
        stability_factors.append(float(low_tail_risk))
        
        return np.mean(stability_factors)
    
    def _calculate_transition_probabilities(
        self,
        current_regime: Dict
    ) -> Dict[str, float]:
        """Calculate probabilities of transitioning to other regimes"""
        
        # Simplified transition matrix (in production, learn from history)
        transitions = {
            MarketRegime.BULL_QUIET: {
                MarketRegime.BULL_QUIET: 0.7,
                MarketRegime.BULL_VOLATILE: 0.15,
                MarketRegime.SIDEWAYS_LOW_VOL: 0.10,
                MarketRegime.BEAR_QUIET: 0.05
            },
            MarketRegime.BULL_VOLATILE: {
                MarketRegime.BULL_VOLATILE: 0.5,
                MarketRegime.BULL_QUIET: 0.2,
                MarketRegime.BEAR_VOLATILE: 0.15,
                MarketRegime.CRISIS: 0.10,
                MarketRegime.SIDEWAYS_HIGH_VOL: 0.05
            },
            MarketRegime.BEAR_QUIET: {
                MarketRegime.BEAR_QUIET: 0.6,
                MarketRegime.BEAR_VOLATILE: 0.2,
                MarketRegime.SIDEWAYS_LOW_VOL: 0.15,
                MarketRegime.RECOVERY: 0.05
            },
            MarketRegime.CRISIS: {
                MarketRegime.CRISIS: 0.4,
                MarketRegime.BEAR_VOLATILE: 0.3,
                MarketRegime.RECOVERY: 0.3
            }
        }
        
        current = current_regime['regime']
        if current in transitions:
            return {k.value: v for k, v in transitions[current].items()}
        
        # Default uniform distribution
        return {regime.value: 1/8 for regime in MarketRegime}
    
    def _generate_regime_characteristics(
        self,
        regime: Dict,
        indicators: Dict
    ) -> Dict[str, Any]:
        """Generate characteristics and trading implications for regime"""
        
        regime_type = regime['regime']
        
        characteristics = {
            MarketRegime.BULL_QUIET: {
                'description': 'Steady uptrend with low volatility',
                'typical_duration_days': 180,
                'recommended_strategy': 'Buy and hold, momentum strategies',
                'risk_level': 'Low',
                'position_sizing': 'Increase to 1.2x normal',
                'sector_preference': ['Technology', 'Consumer Discretionary'],
                'hedging_needed': False
            },
            MarketRegime.BULL_VOLATILE: {
                'description': 'Uptrend with elevated volatility',
                'typical_duration_days': 90,
                'recommended_strategy': 'Selective buying, profit taking',
                'risk_level': 'Medium',
                'position_sizing': 'Normal (1.0x)',
                'sector_preference': ['Healthcare', 'Consumer Staples'],
                'hedging_needed': True
            },
            MarketRegime.BEAR_VOLATILE: {
                'description': 'Downtrend with high volatility',
                'typical_duration_days': 60,
                'recommended_strategy': 'Defensive, short opportunities',
                'risk_level': 'High',
                'position_sizing': 'Reduce to 0.6x normal',
                'sector_preference': ['Utilities', 'Consumer Staples'],
                'hedging_needed': True
            },
            MarketRegime.CRISIS: {
                'description': 'Market crisis with extreme volatility',
                'typical_duration_days': 30,
                'recommended_strategy': 'Capital preservation, quality focus',
                'risk_level': 'Extreme',
                'position_sizing': 'Reduce to 0.3x normal',
                'sector_preference': ['Gold', 'Treasury Bonds'],
                'hedging_needed': True
            }
        }
        
        base_chars = characteristics.get(
            regime_type,
            {
                'description': 'Transitional regime',
                'typical_duration_days': 45,
                'recommended_strategy': 'Neutral, wait for clarity',
                'risk_level': 'Medium',
                'position_sizing': 'Reduce to 0.8x normal',
                'sector_preference': ['Balanced'],
                'hedging_needed': False
            }
        )
        
        # Add current market metrics
        base_chars.update({
            'current_volatility': indicators['volatility'],
            'current_drawdown': indicators['current_drawdown'],
            'trend_strength': indicators['trend_strength'],
            'market_breadth': indicators['advance_decline']
        })
        
        return base_chars
    
    def _get_historical_context(self, regime: Dict) -> Dict:
        """Get historical context for current regime"""
        
        # In production, query historical regime database
        return {
            'similar_periods': [
                {'date': '2017-01-15', 'duration_days': 156, 'outcome': 'continued_rally'},
                {'date': '2019-10-01', 'duration_days': 89, 'outcome': 'volatility_spike'}
            ],
            'average_duration': 120,
            'typical_next_regime': 'sideways_low',
            'historical_return': 0.12
        }
    
    def _estimate_garch_volatility(self, returns: pd.Series) -> float:
        """Estimate GARCH volatility"""
        # Simplified - in production use arch package
        return returns.tail(20).std() * np.sqrt(252) * 1.1
    
    def _calculate_trend_consistency(self, prices: pd.Series) -> float:
        """Calculate trend consistency score"""
        sma_20 = prices.rolling(20).mean()
        above_sma = (prices > sma_20).tail(20)
        return (above_sma.sum() / 20) * 2 - 1  # Scale to [-1, 1]
    
    def _calculate_advance_decline_ratio(self, data: pd.DataFrame) -> float:
        """Calculate advance/decline ratio (simplified)"""
        # In production, use actual market breadth data
        return 0.55  # Placeholder
    
    def _calculate_new_highs_lows_ratio(self, data: pd.DataFrame) -> float:
        """Calculate new highs/lows ratio"""
        # In production, use actual market breadth data
        return 0.60  # Placeholder
    
    def _calculate_correlation_stability(self, data: pd.DataFrame) -> float:
        """Calculate correlation stability across assets"""
        # In production, calculate actual cross-asset correlations
        return 0.75  # Placeholder
    
    def _analyze_volume_pattern(self, data: pd.DataFrame) -> float:
        """Analyze volume patterns"""
        if 'volume' in data.columns:
            volume_ma = data['volume'].rolling(20).mean()
            recent_volume = data['volume'].tail(5).mean()
            return (recent_volume / volume_ma.iloc[-1]) - 1 if volume_ma.iloc[-1] > 0 else 0
        return 0.0
    
    def _estimate_put_call_ratio(self, volatility: float) -> float:
        """Estimate put/call ratio based on volatility"""
        # Simplified estimation
        return 0.7 + (volatility - 0.15) * 2
    
    def _rule_based_regime(self, indicators: Dict) -> MarketRegime:
        """Simple rule-based regime classification"""
        return self._statistical_regime_classification(indicators)


class RegimeAwareRecommendationAdjuster:
    """
    Adjusts recommendations based on detected market regime
    """
    
    def __init__(self, regime_detector: RegimeDetector):
        self.regime_detector = regime_detector
        
    def adjust_recommendations(
        self,
        recommendations: List[Dict],
        market_data: pd.DataFrame
    ) -> List[Dict]:
        """
        Adjust stock recommendations based on market regime
        """
        # Detect current regime
        regime_info = self.regime_detector.detect_regime(market_data)
        current_regime = regime_info['current_regime']
        
        adjusted_recommendations = []
        
        for rec in recommendations:
            adjusted_rec = rec.copy()
            
            # Adjust confidence based on regime
            regime_multiplier = self._get_regime_confidence_multiplier(
                current_regime,
                rec.get('sector', 'Unknown')
            )
            adjusted_rec['confidence'] *= regime_multiplier
            
            # Adjust position sizing
            position_adjustment = self._get_position_size_adjustment(
                current_regime,
                rec.get('risk_score', 0.5)
            )
            adjusted_rec['recommended_allocation'] *= position_adjustment
            
            # Add regime-specific warnings
            warnings = self._generate_regime_warnings(current_regime, rec)
            adjusted_rec['regime_warnings'] = warnings
            
            # Adjust stop-loss based on regime volatility
            if current_regime in [MarketRegime.BEAR_VOLATILE, MarketRegime.CRISIS]:
                adjusted_rec['stop_loss'] *= 0.95  # Tighter stop-loss
            
            adjusted_recommendations.append(adjusted_rec)
        
        # Re-rank based on regime suitability
        return self._rerank_by_regime_suitability(
            adjusted_recommendations,
            current_regime
        )
    
    def _get_regime_confidence_multiplier(
        self,
        regime: MarketRegime,
        sector: str
    ) -> float:
        """Get confidence multiplier based on regime and sector"""
        
        regime_sector_matrix = {
            MarketRegime.BULL_QUIET: {
                'Technology': 1.2,
                'Consumer Discretionary': 1.15,
                'Financials': 1.1,
                'default': 1.0
            },
            MarketRegime.CRISIS: {
                'Technology': 0.6,
                'Consumer Discretionary': 0.5,
                'Utilities': 1.2,
                'Consumer Staples': 1.15,
                'default': 0.7
            }
        }
        
        regime_multipliers = regime_sector_matrix.get(
            regime,
            {'default': 1.0}
        )
        
        return regime_multipliers.get(sector, regime_multipliers['default'])
    
    def _get_position_size_adjustment(
        self,
        regime: MarketRegime,
        risk_score: float
    ) -> float:
        """Adjust position size based on regime and risk"""
        
        regime_adjustments = {
            MarketRegime.BULL_QUIET: 1.2,
            MarketRegime.BULL_VOLATILE: 1.0,
            MarketRegime.SIDEWAYS_LOW_VOL: 0.9,
            MarketRegime.SIDEWAYS_HIGH_VOL: 0.8,
            MarketRegime.BEAR_QUIET: 0.7,
            MarketRegime.BEAR_VOLATILE: 0.5,
            MarketRegime.CRISIS: 0.3,
            MarketRegime.RECOVERY: 1.1
        }
        
        base_adjustment = regime_adjustments.get(regime, 1.0)
        
        # Further adjust based on individual stock risk
        if risk_score > 0.7:
            base_adjustment *= 0.8
        elif risk_score < 0.3:
            base_adjustment *= 1.1
            
        return base_adjustment
    
    def _generate_regime_warnings(
        self,
        regime: MarketRegime,
        recommendation: Dict
    ) -> List[str]:
        """Generate regime-specific warnings"""
        
        warnings = []
        
        if regime == MarketRegime.CRISIS:
            warnings.append("Market in crisis mode - consider reducing all positions")
            if recommendation.get('beta', 1) > 1.5:
                warnings.append("High-beta stock particularly vulnerable in crisis")
                
        elif regime == MarketRegime.BEAR_VOLATILE:
            warnings.append("Bear market with high volatility - use tight stops")
            
        elif regime == MarketRegime.BULL_VOLATILE:
            if recommendation.get('rsi', 50) > 70:
                warnings.append("Overbought in volatile bull market - risk of pullback")
        
        return warnings
    
    def _rerank_by_regime_suitability(
        self,
        recommendations: List[Dict],
        regime: MarketRegime
    ) -> List[Dict]:
        """Re-rank recommendations based on regime suitability"""
        
        def regime_suitability_score(rec):
            score = rec['confidence']
            
            # Boost defensive stocks in bear markets
            if regime in [MarketRegime.BEAR_VOLATILE, MarketRegime.CRISIS]:
                if rec.get('sector') in ['Utilities', 'Consumer Staples']:
                    score *= 1.3
                elif rec.get('beta', 1) < 0.8:
                    score *= 1.2
            
            # Boost growth stocks in bull markets
            elif regime in [MarketRegime.BULL_QUIET, MarketRegime.BULL_VOLATILE]:
                if rec.get('sector') in ['Technology', 'Consumer Discretionary']:
                    score *= 1.2
                elif rec.get('growth_score', 0) > 0.7:
                    score *= 1.15
            
            return score
        
        return sorted(
            recommendations,
            key=regime_suitability_score,
            reverse=True
        )