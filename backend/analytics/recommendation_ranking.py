"""
Recommendation Ranking - filtering, ranking, portfolio optimization, and report summaries.

Extracted from recommendation_engine.py. All functions here are stateless helpers
or thin async wrappers around the portfolio optimizer; they carry no engine state.
Both RecommendationEngine and OptimizedRecommendationEngine delegate to these
functions to avoid code duplication.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

from backend.analytics.recommendation_types import RecommendationAction, StockRecommendation

if TYPE_CHECKING:
    from backend.utils.portfolio_optimizer import PortfolioOptimizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def should_recommend(
    recommendation: StockRecommendation,
    risk_tolerance: str,
) -> bool:
    """Determine if a recommendation meets inclusion criteria."""
    risk_thresholds = {
        'conservative': 0.3,
        'moderate': 0.5,
        'aggressive': 0.8,
    }
    max_risk = risk_thresholds.get(risk_tolerance, 0.5)

    if recommendation.risk_score > max_risk:
        return False

    if recommendation.confidence < 0.5:
        return False

    if recommendation.action in [RecommendationAction.SELL, RecommendationAction.STRONG_SELL]:
        return False

    if recommendation.expected_return < 0.05:
        return False

    return True


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def rank_recommendations(
    recommendations: List[StockRecommendation],
) -> List[StockRecommendation]:
    """Rank recommendations by a weighted composite score."""
    for rec in recommendations:
        return_score = min(rec.expected_return / 0.3, 1.0)
        confidence_score = rec.confidence
        risk_score = 1 - rec.risk_score
        sharpe_score = min(rec.sharpe_ratio / 2, 1.0)

        rec.ranking_score = (
            0.3 * return_score
            + 0.2 * confidence_score
            + 0.2 * risk_score
            + 0.3 * sharpe_score
        )

    ranked = sorted(recommendations, key=lambda x: x.ranking_score, reverse=True)

    for i, rec in enumerate(ranked):
        rec.priority = min(10, max(1, 10 - i // 5))

    return ranked


def rank_recommendations_optimized(
    recommendations: List[StockRecommendation],
) -> List[StockRecommendation]:
    """Memory-efficient ranking used by OptimizedRecommendationEngine."""
    for rec in recommendations:
        rec.ranking_score = (
            rec.technical_score * 0.4
            + rec.confidence * 0.3
            + (1 - rec.risk_score) * 0.3
        )

    ranked = sorted(recommendations, key=lambda x: x.ranking_score, reverse=True)

    for i, rec in enumerate(ranked):
        rec.priority = min(10, max(1, 10 - i // 5))

    return ranked


# ---------------------------------------------------------------------------
# Portfolio optimisation wrappers
# ---------------------------------------------------------------------------

async def optimize_recommendations(
    recommendations: List[StockRecommendation],
    risk_tolerance: str,
    portfolio_optimizer: "PortfolioOptimizer",
) -> List[StockRecommendation]:
    """Apply portfolio optimization to a ranked list of recommendations."""
    if len(recommendations) < 2:
        return recommendations

    expected_returns = np.array([rec.expected_return for rec in recommendations])

    n_assets = len(recommendations)
    correlations = np.eye(n_assets) * 0.5
    volatilities = np.array([rec.volatility for rec in recommendations])
    cov_matrix = np.outer(volatilities, volatilities) * correlations

    risk_params = {
        'conservative': {'max_volatility': 0.15, 'min_sharpe': 1.0},
        'moderate':     {'max_volatility': 0.25, 'min_sharpe': 0.7},
        'aggressive':   {'max_volatility': 0.40, 'min_sharpe': 0.5},
    }
    params = risk_params.get(risk_tolerance, risk_params['moderate'])

    optimal_weights = await portfolio_optimizer.optimize(
        expected_returns=expected_returns,
        cov_matrix=cov_matrix,
        constraints={
            'max_volatility': params['max_volatility'],
            'min_sharpe':     params['min_sharpe'],
            'max_position':   0.10,
            'min_position':   0.02,
        },
    )

    optimized_recs: List[StockRecommendation] = []
    for rec, weight in zip(recommendations, optimal_weights):
        if weight > 0.01:
            rec.recommended_allocation = weight
            rec.max_position_size = weight * 100_000
            optimized_recs.append(rec)

    optimized_recs.sort(key=lambda x: x.recommended_allocation, reverse=True)
    return optimized_recs


async def optimize_recommendations_streaming(
    recommendations: List[StockRecommendation],
    risk_tolerance: str,
) -> List[StockRecommendation]:
    """
    Streaming portfolio optimization to reduce memory usage.

    Used by OptimizedRecommendationEngine - no external optimizer dependency.
    """
    if len(recommendations) < 2:
        return recommendations

    risk_thresholds = {
        'conservative': 0.3,
        'moderate':     0.5,
        'aggressive':   0.8,
    }
    max_risk = risk_thresholds.get(risk_tolerance, 0.5)

    optimized: List[StockRecommendation] = []
    total_allocation = 0.0

    for rec in recommendations:
        if rec.risk_score <= max_risk and total_allocation < 0.8:
            remaining_capacity = 0.8 - total_allocation
            rec.recommended_allocation = min(rec.recommended_allocation, remaining_capacity)

            if rec.recommended_allocation > 0.01:
                optimized.append(rec)
                total_allocation += rec.recommended_allocation

    return optimized


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------

def generate_summary(recommendations: List[StockRecommendation]) -> Dict:
    """Generate summary statistics for a report."""
    if not recommendations:
        return {}

    returns = [rec.expected_return for rec in recommendations]
    risks = [rec.risk_score for rec in recommendations]

    return {
        'total_recommendations': len(recommendations),
        'average_expected_return': float(np.mean(returns)),
        'average_risk_score': float(np.mean(risks)),
        'by_action': {
            action.value: sum(1 for rec in recommendations if rec.action == action)
            for action in RecommendationAction
        },
        'top_sectors': get_top_sectors(recommendations),
        'total_allocation': sum(rec.recommended_allocation for rec in recommendations),
    }


def get_top_sectors(recommendations: List[StockRecommendation]) -> List[Dict]:
    """Get top sectors by recommendation count."""
    sector_counts: Dict[str, int] = {}

    for rec in recommendations:
        sector = "Technology"  # Placeholder - would derive from stock data
        sector_counts[sector] = sector_counts.get(sector, 0) + 1

    sorted_sectors = sorted(sector_counts.items(), key=lambda x: x[1], reverse=True)

    return [
        {'sector': sector, 'count': count}
        for sector, count in sorted_sectors[:5]
    ]
