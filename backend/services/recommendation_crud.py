"""Recommendation SEC/model constants only.

The previous RecommendationCrudMixin (~330 lines) duplicated random.* sample
generators already gated in recommendation_service. That dead fabricator surface
is removed (Wave 2 / #242-A.1). Callers needing sample data use
RecommendationService under DEMO_MODE.

Only constants remain for import by recommendation_analysis / service modules.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

SEC_RISK_WARNING = (
    "IMPORTANT: Past performance does not guarantee future results. All investments "
    "involve risk, including possible loss of principal. The value of investments can "
    "fluctuate, and investors may not get back the amount originally invested. Before "
    "making any investment decision, you should carefully consider your investment "
    "objectives, level of experience, and risk appetite."
)

SEC_METHODOLOGY_DISCLOSURE_TEMPLATE = (
    "This recommendation was generated using {algorithm_type} analysis incorporating "
    "technical indicators, fundamental metrics, and market sentiment data. Model version: "
    "{model_version}. Last model training date: {training_date}."
)

SEC_LIMITATIONS_STATEMENT = (
    "This analysis does NOT consider: (1) your individual financial situation or goals, "
    "(2) tax implications specific to your circumstances, (3) real-time market conditions "
    "that may have changed since data collection, (4) non-public information, (5) geopolitical "
    "events occurring after the analysis date. Data freshness may vary by source; prices and "
    "metrics may be delayed up to 15 minutes for free-tier data sources."
)

RECOMMENDATION_MODEL_VERSION = "1.0.0"
RECOMMENDATION_MODEL_TRAINING_DATE = "2025-12-15"


def generate_sec_disclosure(
    algorithm_type: str = "ML-powered quantitative",
    data_sources: Optional[List[str]] = None,
    confidence_score: float = 0.5,
    *,
    model_version: str = RECOMMENDATION_MODEL_VERSION,
    training_date: str = RECOMMENDATION_MODEL_TRAINING_DATE,
) -> Dict[str, Any]:
    """Build SEC disclosure fields (no random / fabricated market data)."""
    sources = data_sources or ["Market data feed", "Financial statements"]
    return {
        "risk_warning": SEC_RISK_WARNING,
        "methodology": SEC_METHODOLOGY_DISCLOSURE_TEMPLATE.format(
            algorithm_type=algorithm_type,
            model_version=model_version,
            training_date=training_date,
        ),
        "limitations": SEC_LIMITATIONS_STATEMENT,
        "data_sources": sources,
        "confidence_score": confidence_score,
        "model_version": model_version,
        "training_date": training_date,
    }


__all__ = [
    "SEC_RISK_WARNING",
    "SEC_METHODOLOGY_DISCLOSURE_TEMPLATE",
    "SEC_LIMITATIONS_STATEMENT",
    "RECOMMENDATION_MODEL_VERSION",
    "RECOMMENDATION_MODEL_TRAINING_DATE",
    "generate_sec_disclosure",
]
