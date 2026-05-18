"""
Regression tests for risk-free rate + portfolio size config plumbing.

F-09-008 (audit 2026-04, G2a sub-theme E step 37):
backend/analytics/recommendation_engine.py:429 hardcoded
``risk_free_rate = 0.045`` inside the Sharpe-ratio calculation,
silently drifting from FundamentalAnalysisEngine.risk_free_rate.

F-09-009 (step 38):
backend/analytics/recommendation_scoring.py:398 hardcoded
``portfolio_size = 100_000`` inside ``calculate_position_sizing`` and
recommendation_ranking.py:147 multiplied ``weight * 100_000`` —
neither was configurable, so max-position-size dollars never tracked
the actual portfolio.

Source-level tests because the modules pull in a large transitive
graph (technical/fundamental/sentiment engines).
"""

from __future__ import annotations

import re
from pathlib import Path


_ENGINE = (
    Path(__file__).resolve().parents[2]
    / "analytics"
    / "recommendation_engine.py"
)
_RANKING = (
    Path(__file__).resolve().parents[2]
    / "analytics"
    / "recommendation_ranking.py"
)
_SCORING = (
    Path(__file__).resolve().parents[2]
    / "analytics"
    / "recommendation_scoring.py"
)


def test_no_hardcoded_risk_free_rate_in_engine() -> None:
    """F-09-008: bare ``risk_free_rate = 0.045`` literal must be gone."""

    text = _ENGINE.read_text()
    assert not re.search(
        r"^\s*risk_free_rate\s*=\s*0\.045\s*$", text, re.MULTILINE
    ), "recommendation_engine.py still has a hardcoded risk_free_rate constant"


def test_engine_syncs_risk_free_rate_with_fundamental_engine() -> None:
    """F-09-008: engine must pull from FundamentalAnalysisEngine."""

    text = _ENGINE.read_text()
    assert "self.risk_free_rate = self.fundamental_engine.risk_free_rate" in text, (
        "RecommendationEngine.__init__ must source risk_free_rate from "
        "self.fundamental_engine.risk_free_rate so the two stay in sync"
    )


def test_no_hardcoded_portfolio_size_constant() -> None:
    """F-09-009: bare ``portfolio_size = 100_000`` must be gone from scoring."""

    text = _SCORING.read_text()
    assert not re.search(
        r"^\s*portfolio_size\s*=\s*100_000\s*$", text, re.MULTILINE
    ), "recommendation_scoring.py still has a hardcoded portfolio_size constant"


def test_scoring_accepts_portfolio_size_param() -> None:
    """F-09-009: ``calculate_position_sizing`` must accept portfolio_size kwarg."""

    text = _SCORING.read_text()
    assert "portfolio_size: float = 100_000" in text, (
        "calculate_position_sizing must accept portfolio_size as a parameter"
    )


def test_ranking_accepts_portfolio_size_param() -> None:
    """F-09-009: ``optimize_recommendations`` must accept portfolio_size kwarg."""

    text = _RANKING.read_text()
    assert "portfolio_size: float = 100_000" in text, (
        "optimize_recommendations must accept portfolio_size as a parameter"
    )
    assert "weight * portfolio_size" in text, (
        "max_position_size assignment must use the portfolio_size parameter"
    )


def test_engine_reads_default_portfolio_size_from_env() -> None:
    """F-09-009: engine sources portfolio size from DEFAULT_PORTFOLIO_SIZE env."""

    text = _ENGINE.read_text()
    assert "DEFAULT_PORTFOLIO_SIZE" in text, (
        "engine must default portfolio_size from the DEFAULT_PORTFOLIO_SIZE env var"
    )
