# ADR-020: Market Regime Detection Implementation

**Status**: Accepted  
**Date**: 2025-01-08  
**Deciders**: Data Science Team, Product Team

## Context

Investment strategies and stock analysis techniques that work well in bull markets may perform poorly in bear markets or high volatility periods. The Investment Analysis Application needs to adapt its recommendations and risk assessments based on current market conditions to provide more accurate and context-appropriate advice.

Traditional technical and fundamental analysis often fails to account for regime changes, leading to:
- Suboptimal recommendations during market transitions
- Inappropriate risk assessment for current conditions  
- Poor performance during market stress periods
- Lack of context for historical backtesting results

Market regimes can be characterized by different combinations of:
- Trend direction (bull/bear)
- Volatility levels (low/high)
- Market breadth and participation
- Economic cycles and external factors

## Decision

We will implement a comprehensive market regime detection system that:

### 1. Defines Market Regimes
Identify 8 distinct market regimes based on trend and volatility:
- **Bull Low Vol**: Strong uptrend, low volatility
- **Bull High Vol**: Strong uptrend, high volatility  
- **Bear Low Vol**: Strong downtrend, low volatility
- **Bear High Vol**: Strong downtrend, high volatility
- **Sideways Low Vol**: Range-bound, low volatility
- **Sideways High Vol**: Range-bound, high volatility
- **Crisis**: Extreme volatility, panic selling
- **Recovery**: Transition from crisis to normal conditions

### 2. Multi-Method Detection Approach
Implement multiple detection algorithms and combine results:
- **Hidden Markov Model (HMM)**: Probabilistic regime detection
- **Gaussian Mixture Model (GMM)**: Clustering-based approach
- **Statistical Thresholds**: Rule-based detection using key indicators
- **Ensemble Method**: Weighted combination of all approaches

### 3. Key Indicators
Monitor multiple market indicators for regime detection:
- **Trend Indicators**: Moving averages, trend strength, directional movement
- **Volatility Measures**: VIX levels, realized volatility, volatility clustering
- **Market Breadth**: Advance/decline ratios, new highs/lows, sector performance
- **Momentum**: Rate of change, momentum oscillators
- **Market Structure**: Volume patterns, bid-ask spreads, market depth

### 4. Adaptive Recommendations
Adjust recommendation models based on detected regime:
- Different factor weightings for different regimes
- Regime-specific risk adjustments
- Dynamic stop-loss and take-profit levels
- Sector rotation strategies based on regime characteristics

## Implementation Details

### Regime Detection Engine
```python
class RegimeDetector:
    def __init__(self):
        self.regimes = [
            'bull_low_vol', 'bull_high_vol',
            'bear_low_vol', 'bear_high_vol', 
            'sideways_low_vol', 'sideways_high_vol',
            'crisis', 'recovery'
        ]
        self.hmm_model = self._initialize_hmm()
        self.gmm_model = self._initialize_gmm()
        
    def detect_current_regime(self, market_data: pd.DataFrame) -> dict:
        """Detect current market regime using ensemble approach"""
        # Calculate indicators
        indicators = self._calculate_indicators(market_data)
        
        # Multiple detection methods
        hmm_regime = self._hmm_detection(indicators)
        gmm_regime = self._gmm_detection(indicators) 
        statistical_regime = self._statistical_detection(indicators)
        
        # Ensemble prediction
        regime_probs = self._ensemble_prediction([
            hmm_regime, gmm_regime, statistical_regime
        ])
        
        return {
            'current_regime': max(regime_probs, key=regime_probs.get),
            'confidence': max(regime_probs.values()),
            'regime_probabilities': regime_probs,
            'transition_probability': self._calculate_transition_prob(),
            'regime_duration': self._estimate_regime_duration()
        }
```

### Market Indicators Calculation
```python
def _calculate_indicators(self, market_data: pd.DataFrame) -> dict:
    """Calculate comprehensive market indicators"""
    indicators = {}
    
    # Trend indicators
    indicators['sma_slope'] = self._calculate_sma_slope(market_data, 20)
    indicators['trend_strength'] = self._calculate_trend_strength(market_data)
    
    # Volatility measures  
    indicators['realized_vol'] = market_data['returns'].rolling(20).std() * np.sqrt(252)
    indicators['vol_regime'] = 'high' if indicators['realized_vol'].iloc[-1] > 0.25 else 'low'
    
    # Market breadth
    indicators['advance_decline'] = self._calculate_advance_decline_ratio()
    indicators['new_highs_lows'] = self._calculate_new_highs_lows_ratio()
    
    # Momentum
    indicators['roc'] = market_data['close'].pct_change(20)
    indicators['momentum_score'] = self._calculate_momentum_score(market_data)
    
    return indicators
```

### Regime-Aware Recommendation Adjustment
```python
class RegimeAwareRecommendationAdjuster:
    def __init__(self):
        self.regime_weights = {
            'bull_low_vol': {'momentum': 0.4, 'growth': 0.4, 'technical': 0.2},
            'bull_high_vol': {'momentum': 0.3, 'growth': 0.3, 'quality': 0.4},
            'bear_low_vol': {'value': 0.4, 'quality': 0.4, 'defensive': 0.2},
            'bear_high_vol': {'defensive': 0.5, 'quality': 0.3, 'value': 0.2},
            'sideways_low_vol': {'mean_reversion': 0.5, 'pairs': 0.3, 'income': 0.2},
            'sideways_high_vol': {'volatility': 0.4, 'mean_reversion': 0.3, 'options': 0.3},
            'crisis': {'defensive': 0.6, 'cash': 0.3, 'hedge': 0.1},
            'recovery': {'contrarian': 0.4, 'momentum': 0.3, 'growth': 0.3}
        }
    
    def adjust_recommendation(self, base_rec: dict, regime: str) -> dict:
        """Adjust recommendation based on current market regime"""
        weights = self.regime_weights[regime]
        
        # Adjust confidence based on regime stability
        regime_stability = self._calculate_regime_stability(regime)
        adjusted_confidence = base_rec['confidence'] * regime_stability
        
        # Modify position sizing based on regime risk
        risk_multiplier = self._get_regime_risk_multiplier(regime)
        adjusted_position_size = base_rec['position_size'] * risk_multiplier
        
        return {
            **base_rec,
            'regime_adjusted': True,
            'market_regime': regime,
            'confidence': adjusted_confidence,
            'position_size': adjusted_position_size,
            'regime_factors': weights,
            'risk_adjustment': risk_multiplier
        }
```

## Consequences

### Positive
- **Context-Aware Analysis**: Recommendations adapt to market conditions
- **Improved Risk Management**: Better risk assessment for different market environments
- **Enhanced Backtesting**: Historical analysis can be regime-segmented for better insights  
- **Proactive Strategy**: Anticipate regime changes rather than react after they occur
- **Better Performance**: Strategy optimization for specific market conditions
- **User Education**: Help users understand market context for their investments

### Negative
- **Model Complexity**: Additional complexity in recommendation engine
- **Computational Overhead**: Real-time regime detection requires additional processing
- **Parameter Sensitivity**: Models may be sensitive to parameter choices
- **Regime Identification Lag**: May not detect regime changes immediately
- **Overfitting Risk**: Models might overfit to historical regime patterns
- **Interpretation Challenges**: Users may not understand regime-based adjustments

### Risks
- **False Regime Detection**: Incorrectly identifying regime changes
- **Model Degradation**: Regime characteristics may change over time
- **Whipsaw Conditions**: Frequent regime changes during transition periods
- **Historical Bias**: Past regime patterns may not repeat in the future
- **Computational Requirements**: May impact system performance during market hours

## Validation and Testing

### Historical Backtesting
- Test regime detection accuracy against known historical periods
- Validate regime-adjusted strategies vs. static approaches
- Analyze performance during major market transitions

### Walk-Forward Analysis
- Implement rolling regime detection with out-of-sample testing
- Monitor regime detection stability and accuracy over time
- Validate ensemble approach vs. individual methods

### Stress Testing
- Test behavior during market crash scenarios
- Validate crisis regime detection sensitivity
- Ensure system remains stable during volatile periods

## Operational Considerations

### Real-Time Updates
```python
# Daily regime update schedule
@scheduler.scheduled_job(trigger="cron", hour=16, minute=30)  # After market close
async def update_market_regime():
    """Update market regime detection daily"""
    regime_detector = RegimeDetector()
    current_regime = await regime_detector.detect_current_regime()
    
    # Store regime information
    await store_regime_data(current_regime)
    
    # Trigger recommendation recalculation if regime changed
    if regime_changed(current_regime):
        await trigger_recommendation_update()
```

### Monitoring and Alerts
- Alert on regime changes
- Monitor regime detection confidence levels
- Track recommendation performance by regime
- Alert on extended periods of regime uncertainty

### User Communication
- Display current market regime in dashboard
- Explain regime-based recommendation adjustments
- Provide historical context for regime patterns
- Educational content about different market regimes

## Integration Points

### Recommendation Engine
- Apply regime weights to factor models
- Adjust risk parameters based on regime
- Modify portfolio construction rules

### Risk Management
- Scale position sizes based on regime volatility
- Adjust stop-loss levels for regime characteristics
- Implement regime-specific hedging strategies

### Backtesting Framework
- Segment historical performance by regime
- Calculate regime-specific strategy metrics
- Enable regime-filtered backtesting

## Related ADRs
- [ADR-019: ML Model Architecture](./019-ml-model-architecture.md)
- [ADR-021: Statistical Analysis Framework](./021-statistical-analysis-framework.md)  
- [ADR-015: Error Handling Standards](./015-error-handling-standards.md)
- [ADR-013: Monitoring and Alerting](./013-monitoring-alerting.md)

## Future Enhancements
- Incorporate alternative data sources (sentiment, economic indicators)
- Implement sector-specific regime detection
- Add international market regime correlations
- Develop regime-based portfolio optimization
- Create regime transition early warning system

## Review Schedule
Market regime detection models should be reviewed quarterly to:
- Validate detection accuracy against recent market events
- Retrain models with new data
- Adjust regime definitions if market structure changes
- Update ensemble weights based on method performance