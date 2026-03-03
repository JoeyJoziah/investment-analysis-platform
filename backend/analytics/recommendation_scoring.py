"""
Recommendation Scoring - scoring helpers, price targets, confidence, and key-factor extraction.

Extracted from recommendation_engine.py. All functions here operate as pure/near-pure
helpers; they carry no engine state of their own and are used by both
RecommendationEngine and OptimizedRecommendationEngine.
"""

import numpy as np
from typing import Dict, List
from datetime import datetime, timezone

from backend.analytics.recommendation_types import RecommendationAction

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Score normalisation
# ---------------------------------------------------------------------------

def normalize_score(value: float, min_val: float, max_val: float) -> float:
    """Normalize a value to 0-1 range."""
    if max_val == min_val:
        return 0.5
    normalized = (value - min_val) / (max_val - min_val)
    return max(0.0, min(1.0, normalized))


# ---------------------------------------------------------------------------
# Action determination
# ---------------------------------------------------------------------------

def determine_action(score: float, thresholds: Dict[str, float]) -> RecommendationAction:
    """Determine recommendation action based on composite score and threshold map."""
    if score >= thresholds['strong_buy']:
        return RecommendationAction.STRONG_BUY
    elif score >= thresholds['buy']:
        return RecommendationAction.BUY
    elif score >= thresholds['hold']:
        return RecommendationAction.HOLD
    elif score >= thresholds['sell']:
        return RecommendationAction.SELL
    else:
        return RecommendationAction.STRONG_SELL


# ---------------------------------------------------------------------------
# Confidence calculation
# ---------------------------------------------------------------------------

def calculate_confidence(
    technical: Dict,
    fundamental: Dict,
    sentiment: Dict,
    ml_predictions: Dict,
    risk_metrics: Dict
) -> float:
    """Calculate overall confidence in recommendation."""
    confidence_factors = []

    # Technical confidence
    if technical:
        signal_count = len(technical.get('signals', []))
        pattern_count = len(
            technical.get('pattern_recognition', {}).get('candlestick_patterns', {})
        )
        tech_confidence = min(1.0, (signal_count + pattern_count) / 10)
        confidence_factors.append(tech_confidence)

    # Fundamental confidence
    if fundamental:
        quality_score = fundamental.get('quality_score', {}).get('overall_score', 50) / 100
        confidence_factors.append(quality_score)

    # Sentiment confidence
    if sentiment:
        sentiment_confidence = sentiment.get('overall_sentiment', {}).get('confidence', 0.5)
        confidence_factors.append(sentiment_confidence)

    # ML model confidence
    if ml_predictions:
        ml_confidences = [
            pred.model_confidence for pred in ml_predictions.values()
            if hasattr(pred, 'model_confidence')
        ]
        if ml_confidences:
            confidence_factors.append(np.mean(ml_confidences))

    # Risk adjustment
    risk_penalty = risk_metrics['risk_score'] * 0.2

    if confidence_factors:
        base_confidence = np.mean(confidence_factors)
        return max(0.1, base_confidence - risk_penalty)

    return 0.5


# ---------------------------------------------------------------------------
# Price targets
# ---------------------------------------------------------------------------

def calculate_price_targets(
    current_price: float,
    ml_predictions: Dict,
    technical_analysis: Dict,
    risk_metrics: Dict
) -> Dict[str, float]:
    """Calculate price targets and stop loss."""
    # ML targets
    ml_targets = [
        pred.predicted_price for pred in ml_predictions.values()
        if hasattr(pred, 'predicted_price')
    ]
    ml_target = np.mean(ml_targets) if ml_targets else current_price

    # Technical targets
    tech_resistance = technical_analysis.get(
        'support_resistance', {}
    ).get('primary_resistance', current_price * 1.1)
    tech_support = technical_analysis.get(
        'support_resistance', {}
    ).get('primary_support', current_price * 0.9)

    # Combined target
    target_price = 0.7 * ml_target + 0.3 * tech_resistance

    # Stop loss
    volatility_stop = current_price * (
        1 - 2 * risk_metrics['volatility'] / np.sqrt(252) * 5
    )
    support_stop = tech_support * 0.98
    stop_loss = max(volatility_stop, support_stop)

    # Expected return
    expected_return = (target_price - current_price) / current_price
    downside = abs((stop_loss - current_price) / current_price)
    risk_reward = abs(expected_return) / downside if downside else 0.0

    return {
        'target': target_price,
        'stop_loss': stop_loss,
        'expected_return': expected_return,
        'risk_reward_ratio': risk_reward,
    }


# ---------------------------------------------------------------------------
# Time horizon
# ---------------------------------------------------------------------------

def determine_time_horizon(
    action: RecommendationAction,
    technical_analysis: Dict,
    ml_predictions: Dict
) -> int:
    """Determine investment time horizon in days."""
    base_horizons = {
        RecommendationAction.STRONG_BUY: 60,
        RecommendationAction.BUY: 30,
        RecommendationAction.HOLD: 20,
        RecommendationAction.SELL: 10,
        RecommendationAction.STRONG_SELL: 5,
    }
    base_horizon = base_horizons.get(action, 20)

    if technical_analysis:
        patterns = technical_analysis.get(
            'pattern_recognition', {}
        ).get('chart_patterns', {})
        if any(p in patterns for p in ['cup_and_handle', 'ascending_triangle']):
            base_horizon = int(base_horizon * 1.5)
        elif any(p in patterns for p in ['flag', 'pennant']):
            base_horizon = int(base_horizon * 0.7)

    if ml_predictions:
        if 'horizon_60' in ml_predictions and 'horizon_5' in ml_predictions:
            long_return = ml_predictions['horizon_60'].predicted_return
            short_return = ml_predictions['horizon_5'].predicted_return
            if abs(long_return) > abs(short_return) * 2:
                base_horizon = 60

    return base_horizon


# ---------------------------------------------------------------------------
# Key factor / risk / opportunity / catalyst extraction
# ---------------------------------------------------------------------------

def extract_key_factors(
    technical: Dict,
    fundamental: Dict,
    sentiment: Dict,
    ml_predictions: Dict
) -> List[str]:
    """Extract key factors driving the recommendation."""
    factors: List[str] = []

    # Technical factors
    if technical:
        trend = technical.get('market_structure', {}).get('trend', '')
        if 'uptrend' in trend:
            factors.append("Strong technical uptrend")
        elif 'downtrend' in trend:
            factors.append("Technical downtrend warning")

        patterns = technical.get('pattern_recognition', {}).get('candlestick_patterns', {})
        if patterns:
            pattern_names = list(patterns.keys())[:2]
            factors.append(f"Technical patterns: {', '.join(pattern_names)}")

        rsi = technical.get('momentum_indicators', {}).get('rsi_14', 50)
        if rsi < 30:
            factors.append("Oversold conditions (RSI < 30)")
        elif rsi > 70:
            factors.append("Overbought conditions (RSI > 70)")

    # Fundamental factors
    if fundamental:
        valuation = fundamental.get('valuation_models', {})
        upside = valuation.get('upside_potential', 0)
        if upside > 30:
            factors.append(f"Significant undervaluation ({upside:.0f}% upside)")
        elif upside < -20:
            factors.append(f"Overvaluation concern ({abs(upside):.0f}% downside)")

        quality = fundamental.get('quality_score', {}).get('overall_score', 0)
        if quality > 80:
            factors.append("Exceptional business quality")

        growth = fundamental.get('growth_analysis', {}).get('growth_drivers', [])
        if growth:
            factors.append(f"Growth drivers: {', '.join(growth[:2])}")

    # Sentiment factors
    if sentiment:
        overall = sentiment.get('overall_sentiment', {})
        if overall.get('score', 0) > 0.5:
            factors.append("Positive market sentiment")
        elif overall.get('score', 0) < -0.5:
            factors.append("Negative sentiment warning")

        analyst = sentiment.get('source_breakdown', {}).get('analyst', {})
        if analyst.get('average_sentiment', 0) > 0.6:
            factors.append("Bullish analyst consensus")

    # ML factors
    if ml_predictions:
        strong_predictions = [
            pred for pred in ml_predictions.values()
            if hasattr(pred, 'predicted_return') and abs(pred.predicted_return) > 0.1
        ]
        if strong_predictions:
            avg_return = np.mean([p.predicted_return for p in strong_predictions])
            factors.append(f"ML models predict {avg_return * 100:.1f}% return")

    return factors[:5]


def identify_risks(
    fundamental: Dict,
    risk_metrics: Dict,
    sentiment: Dict
) -> List[str]:
    """Identify key risks."""
    risks: List[str] = []

    if fundamental:
        fund_risks = fundamental.get('risks', [])
        for risk in fund_risks[:2]:
            risks.append(risk.get('description', ''))

        health = fundamental.get('financial_health', {})
        z_score = health.get('altman_z_score', {}).get('score', 3)
        if z_score < 1.8:
            risks.append("Financial distress risk (low Altman Z-Score)")

    if risk_metrics['volatility'] > 0.4:
        risks.append(f"High volatility ({risk_metrics['volatility'] * 100:.0f}% annual)")

    if risk_metrics['beta'] > 1.5:
        risks.append(f"High market sensitivity (Beta: {risk_metrics['beta']:.1f})")

    if risk_metrics['max_drawdown'] < -0.3:
        risks.append(f"Significant drawdown risk ({risk_metrics['max_drawdown'] * 100:.0f}%)")

    if sentiment:
        anomalies = sentiment.get('anomaly_detection', {})
        if anomalies.get('anomalies_detected'):
            risks.append("Unusual sentiment patterns detected")

    return risks[:4]


def identify_opportunities(
    fundamental: Dict,
    technical: Dict,
    sentiment: Dict
) -> List[str]:
    """Identify opportunities."""
    opportunities: List[str] = []

    if fundamental:
        fund_opps = fundamental.get('opportunities', [])
        for opp in fund_opps[:2]:
            opportunities.append(opp.get('description', ''))

        moat = fundamental.get('moat_analysis', {})
        if moat.get('rating') == 'wide':
            opportunities.append("Wide economic moat provides competitive advantage")

    if technical:
        sr = technical.get('support_resistance', {})
        if sr.get('current_price') and sr.get('primary_support'):
            support_distance = (
                sr['current_price'] - sr['primary_support']
            ) / sr['current_price']
            if support_distance < 0.05:
                opportunities.append("Trading near strong support level")

        patterns = technical.get('pattern_recognition', {}).get('chart_patterns', {})
        if 'ascending_triangle' in patterns or 'cup_and_handle' in patterns:
            opportunities.append("Bullish breakout pattern forming")

    if sentiment:
        momentum = sentiment.get('temporal_analysis', {}).get('momentum', 0)
        if momentum > 0.2:
            opportunities.append("Improving sentiment momentum")

    return opportunities[:3]


def find_catalysts(
    stock_data: Dict,
    sentiment: Dict,
    fundamental: Dict
) -> List[str]:
    """Identify potential catalysts."""
    catalysts: List[str] = []

    next_earnings = stock_data.get('next_earnings_date')
    if next_earnings:
        days_to_earnings = (next_earnings - datetime.now(timezone.utc)).days
        if 0 < days_to_earnings < 30:
            catalysts.append(f"Earnings report in {days_to_earnings} days")

    if sentiment:
        keywords = sentiment.get('keyword_analysis', {}).get('top_positive', [])
        if any(k in ['merger', 'acquisition', 'partnership', 'fda', 'approval'] for k in keywords):
            catalysts.append("Potential M&A or regulatory catalyst")

    if 'product' in str(sentiment.get('keyword_analysis', {})).lower():
        catalysts.append("New product launch catalyst")

    sector = stock_data.get('sector')
    if sector and fundamental:
        growth = fundamental.get('growth_analysis', {})
        if 'market_growth' in str(growth):
            catalysts.append(f"{sector} sector rotation opportunity")

    return catalysts[:3]


# ---------------------------------------------------------------------------
# Position sizing and priority
# ---------------------------------------------------------------------------

def calculate_position_sizing(
    confidence: float,
    risk_metrics: Dict,
    action: RecommendationAction
) -> Dict[str, float]:
    """Calculate recommended position sizing via a safety-adjusted Kelly Criterion."""
    p = confidence
    q = 1 - p
    b = 2  # Assume 2:1 reward/risk ratio

    kelly_fraction = (p * b - q) / b
    safe_kelly = kelly_fraction * 0.25

    risk_adjustment = 1 - risk_metrics['risk_score'] * 0.5

    max_allocations = {
        RecommendationAction.STRONG_BUY: 0.10,
        RecommendationAction.BUY: 0.07,
        RecommendationAction.HOLD: 0.05,
        RecommendationAction.SELL: 0.0,
        RecommendationAction.STRONG_SELL: 0.0,
    }
    max_allocation = max_allocations.get(action, 0.05)

    allocation = min(safe_kelly * risk_adjustment, max_allocation)
    allocation = max(0.0, allocation)

    portfolio_size = 100_000
    max_size = allocation * portfolio_size

    return {
        'allocation': allocation,
        'max_size': max_size,
        'kelly_fraction': kelly_fraction,
        'risk_adjusted_allocation': allocation,
    }


def calculate_priority(
    score: float,
    confidence: float,
    opportunities: List[str]
) -> int:
    """Calculate recommendation priority (1-10)."""
    base_priority = int(score * 10)

    if confidence > 0.8:
        base_priority += 1

    if len(opportunities) >= 3:
        base_priority += 1

    return max(1, min(10, base_priority))
