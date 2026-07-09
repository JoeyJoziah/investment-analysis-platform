"""
SEC compliance constants and disclosure helpers for recommendations (Wave 13).

Extracted from recommendation_service to keep domain policy separate and
shrink the service module footprint.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# Standard SEC Risk Warning (required on all recommendations)
SEC_RISK_WARNING = (
    "IMPORTANT: Past performance does not guarantee future results. All investments "
    "involve risk, including possible loss of principal. The value of investments can "
    "fluctuate, and investors may not get back the amount originally invested. Before "
    "making any investment decision, you should carefully consider your investment "
    "objectives, level of experience, and risk appetite."
)

# Standard Methodology Disclosure Template
SEC_METHODOLOGY_DISCLOSURE_TEMPLATE = (
    "This recommendation was generated using {algorithm_type} analysis incorporating "
    "technical indicators, fundamental metrics, and market sentiment data. Model version: "
    "{model_version}. Last model training date: {training_date}."
)

# Standard Limitations Statement
SEC_LIMITATIONS_STATEMENT = (
    "This analysis does NOT consider: (1) your individual financial situation or goals, "
    "(2) tax implications specific to your circumstances, (3) real-time market conditions "
    "that may have changed since data collection, (4) non-public information, (5) geopolitical "
    "events occurring after the analysis date. Data freshness may vary by source; prices and "
    "metrics may be delayed up to 15 minutes for free-tier data sources."
)

# Current model version for SEC disclosure
RECOMMENDATION_MODEL_VERSION = "1.0.0"
RECOMMENDATION_MODEL_TRAINING_DATE = "2025-12-15"

# Honest algorithm label for the transparent screen (must not claim ML).
RULES_BASED_ALGORITHM_TYPE = "rules-based quantitative screen"

RULES_BASED_METHODOLOGY_DISCLOSURE = (
    "This recommendation was generated using a transparent, rules-based "
    "quantitative screen over stored historical data. It does NOT use machine "
    "learning, neural networks, or predictive models. Momentum is measured as "
    "the trailing 60-trading-day price return computed from end-of-day closing "
    "prices. Valuation is measured as the cross-sectional percentile rank of the "
    "price-to-earnings (P/E) ratio across the screened universe (a lower P/E "
    "ranks more favorably), with the PEG ratio used as a tiebreaker. The "
    "composite score equally weights momentum percentile and inverse-valuation "
    "percentile when both inputs are available, and uses momentum alone "
    "otherwise. Recommendation tiers and confidence are derived deterministically "
    "from the composite rank; identical inputs always produce identical outputs."
)

MOMENTUM_WINDOW_DAYS = 60
MOMENTUM_MIN_ROWS = 30
TARGET_PRICE_MOMENTUM_CLAMP = 0.30


def build_sec_disclosure(
    algorithm_type: str = "ML-powered quantitative",
    data_sources: Optional[List[str]] = None,
    confidence_score: float = 0.5,
    methodology_disclosure: Optional[str] = None,
) -> Dict[str, Any]:
    """Build SEC 2025 compliant disclosure fields for a recommendation."""
    if data_sources is None:
        data_sources = [
            f"Alpha Vantage API (delayed 15 min) - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            f"Finnhub Market Data - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            f"Historical price data (EOD) - {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
            "Financial statements (quarterly) - Last updated Q4 2025",
        ]

    if confidence_score >= 0.8:
        confidence_level = "high"
    elif confidence_score >= 0.6:
        confidence_level = "moderate"
    else:
        confidence_level = "low"

    if methodology_disclosure is None:
        methodology_disclosure = SEC_METHODOLOGY_DISCLOSURE_TEMPLATE.format(
            algorithm_type=algorithm_type,
            model_version=RECOMMENDATION_MODEL_VERSION,
            training_date=RECOMMENDATION_MODEL_TRAINING_DATE,
        )

    return {
        "methodology_disclosure": methodology_disclosure,
        "data_sources": data_sources,
        "model_version": RECOMMENDATION_MODEL_VERSION,
        "model_training_date": RECOMMENDATION_MODEL_TRAINING_DATE,
        "risk_warning": SEC_RISK_WARNING,
        "limitations_statement": SEC_LIMITATIONS_STATEMENT,
        "confidence_level": confidence_level,
        "conflict_of_interest_statement": (
            "This platform does not hold positions in any recommended securities. "
            "No material relationships exist between this platform and any recommended issuers."
        ),
    }
