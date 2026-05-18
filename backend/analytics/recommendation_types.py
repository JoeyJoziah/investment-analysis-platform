"""
Shared types for the recommendation engine subsystem.

Placing enums and dataclasses here breaks the circular-import chain that would
otherwise arise when scoring/ranking helpers import from recommendation_engine.py
while recommendation_engine.py also imports those helpers.

All public symbols are re-exported from recommendation_engine.py so existing
callers are completely unaffected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List

from backend.ml.runtime_models import PredictionResult


class RecommendationAction(Enum):
    """Recommendation actions."""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


@dataclass
class StockRecommendation:
    """Complete stock recommendation."""
    ticker: str
    action: RecommendationAction
    confidence: float
    priority: int  # 1-10

    # Price targets
    entry_price: float
    target_price: float
    stop_loss: float
    expected_return: float
    time_horizon_days: int

    # Risk metrics
    risk_score: float
    volatility: float
    beta: float
    sharpe_ratio: float
    max_drawdown: float

    # Analysis scores
    technical_score: float
    fundamental_score: float
    sentiment_score: float
    ml_prediction_score: float

    # Detailed analysis
    technical_analysis: Dict
    fundamental_analysis: Dict
    sentiment_analysis: Dict
    ml_predictions: Dict[str, PredictionResult]

    # Reasoning
    key_factors: List[str]
    risks: List[str]
    opportunities: List[str]
    catalysts: List[str]

    # Metadata
    generated_at: datetime
    valid_until: datetime

    # Position sizing
    recommended_allocation: float  # Percentage of portfolio
    max_position_size: float       # Dollar amount

    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            'ticker': self.ticker,
            'action': self.action.value,
            'confidence': self.confidence,
            'priority': self.priority,
            'entry_price': self.entry_price,
            'target_price': self.target_price,
            'stop_loss': self.stop_loss,
            'expected_return': self.expected_return,
            'time_horizon_days': self.time_horizon_days,
            'risk_score': self.risk_score,
            'volatility': self.volatility,
            'beta': self.beta,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'technical_score': self.technical_score,
            'fundamental_score': self.fundamental_score,
            'sentiment_score': self.sentiment_score,
            'ml_prediction_score': self.ml_prediction_score,
            'key_factors': self.key_factors,
            'risks': self.risks,
            'opportunities': self.opportunities,
            'catalysts': self.catalysts,
            'generated_at': self.generated_at.isoformat(),
            'valid_until': self.valid_until.isoformat(),
            'recommended_allocation': self.recommended_allocation,
            'max_position_size': self.max_position_size,
        }
