"""
Regression tests for hybrid_engine._create_error_recommendation.

F-09-005 (audit 2026-04, G2a sub-theme E step 35):
The error-fallback factory passed ``recommendation="HOLD"`` and
``overall_score=0.5`` to ``EnhancedStockRecommendation(...)`` —
neither is a declared field on the dataclass or its parent. Worse,
many parent-required fields (priority, entry_price, time_horizon_days,
etc.) were missing, so the constructor TypeError'd on every error
path. The error fallback itself was broken.

This test verifies source-level shape rather than instantiating the
class, because EnhancedStockRecommendation pulls in
``backend.analytics.recommendation_engine`` which transitively imports
heavy ML deps.
"""

from __future__ import annotations

import re
from pathlib import Path


_HE = (
    Path(__file__).resolve().parents[2]
    / "analytics"
    / "agents"
    / "hybrid_engine.py"
)


def test_error_recommendation_does_not_pass_unknown_kwargs() -> None:
    """F-09-005: ``recommendation=`` and ``overall_score=`` kwargs are gone."""

    text = _HE.read_text()
    # Locate the _create_error_recommendation block.
    body = re.search(
        r"def _create_error_recommendation\(.*?\n    [^ ]",
        text,
        re.DOTALL,
    )
    assert body is not None, "could not locate _create_error_recommendation"
    block = body.group(0)
    assert "recommendation=\"HOLD\"" not in block, (
        "_create_error_recommendation must not pass ``recommendation=`` "
        "(use ``action=RecommendationAction.HOLD`` instead)"
    )
    assert "overall_score=" not in block, (
        "_create_error_recommendation must not pass ``overall_score=`` "
        "(not a declared field on StockRecommendation)"
    )


def test_error_recommendation_uses_recommendation_action_enum() -> None:
    """F-09-005: error path must use the real ``action`` field + enum."""

    text = _HE.read_text()
    assert "RecommendationAction.HOLD" in text, (
        "error path must use RecommendationAction.HOLD on the action field"
    )


def test_error_recommendation_supplies_all_parent_required_fields() -> None:
    """F-09-005: every required parent field must be passed."""

    text = _HE.read_text()
    block = re.search(
        r"return EnhancedStockRecommendation\((.*?)\n        \)",
        text,
        re.DOTALL,
    )
    assert block is not None, "error path return block not found"
    body = block.group(1)

    required = [
        "ticker", "action", "confidence", "priority",
        "entry_price", "target_price", "stop_loss", "expected_return",
        "time_horizon_days",
        "risk_score", "volatility", "beta", "sharpe_ratio", "max_drawdown",
        "technical_score", "fundamental_score", "sentiment_score",
        "ml_prediction_score",
        "technical_analysis", "fundamental_analysis", "sentiment_analysis",
        "ml_predictions",
        "key_factors", "risks", "opportunities", "catalysts",
        "generated_at", "valid_until",
        "recommended_allocation", "max_position_size",
    ]
    missing = [f for f in required if f"{f}=" not in body]
    assert not missing, (
        f"_create_error_recommendation missing parent-required kwargs: {missing}"
    )


def test_no_self_overall_score_access() -> None:
    """F-09-005: hybrid score logic must not read self.overall_score."""

    text = _HE.read_text()
    # Allow it in comments / docstrings (rationale text), forbid in code.
    code_lines = [
        line for line in text.splitlines()
        if "self.overall_score" in line and not line.lstrip().startswith("#")
    ]
    assert not code_lines, (
        f"self.overall_score is accessed in non-comment code: {code_lines}"
    )
