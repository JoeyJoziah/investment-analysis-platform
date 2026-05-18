"""
Regression tests for StockRecommendation.ranking_score field.

F-09-004 (audit 2026-04, G2a sub-theme E step 34):
The ranking layer assigned ``rec.ranking_score = X`` on every
recommendation, but ``StockRecommendation`` had no such field — the
attribute survived as an ad-hoc instance attr that disappeared from
``dataclasses.asdict(rec)`` and ``to_dict()`` output. Consumers reading
the serialized form (storage, API responses) never saw the score.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest


_PATH = (
    Path(__file__).resolve().parents[2]
    / "analytics"
    / "recommendation_types.py"
)


def _load_module(monkeypatch: pytest.MonkeyPatch):
    # Drop polluted backend.* sys.modules entries first.
    for name in list(sys.modules):
        if name == "backend" or name.startswith("backend."):
            monkeypatch.delitem(sys.modules, name, raising=False)

    name = "recommendation_types_under_test"
    spec = importlib.util.spec_from_file_location(name, _PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # @dataclass resolves forward refs by looking up cls.__module__ in
    # sys.modules; register before exec to avoid AttributeError.
    sys.modules[name] = module
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def _build_rec(mod):
    return mod.StockRecommendation(
        ticker="ACME",
        action=mod.RecommendationAction.BUY,
        confidence=0.8,
        priority=5,
        entry_price=100.0,
        target_price=120.0,
        stop_loss=90.0,
        expected_return=0.2,
        time_horizon_days=30,
        risk_score=0.3,
        volatility=0.2,
        beta=1.1,
        sharpe_ratio=1.5,
        max_drawdown=0.1,
        technical_score=0.7,
        fundamental_score=0.6,
        sentiment_score=0.5,
        ml_prediction_score=0.65,
        technical_analysis={},
        fundamental_analysis={},
        sentiment_analysis={},
        ml_predictions={},
        key_factors=[],
        risks=[],
        opportunities=[],
        catalysts=[],
        generated_at=datetime.now(timezone.utc),
        valid_until=datetime.now(timezone.utc),
        recommended_allocation=0.05,
        max_position_size=10_000.0,
    )


def test_dataclass_has_ranking_score_field(monkeypatch: pytest.MonkeyPatch) -> None:
    """F-09-004: ``ranking_score`` must be a declared field on the dataclass."""

    mod = _load_module(monkeypatch)
    fields = {f.name for f in dataclasses.fields(mod.StockRecommendation)}
    assert "ranking_score" in fields, (
        "StockRecommendation must declare ranking_score as a dataclass field"
    )


def test_default_ranking_score_is_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """F-09-004: ``ranking_score`` defaults to 0.0 when not supplied."""

    mod = _load_module(monkeypatch)
    rec = _build_rec(mod)
    assert rec.ranking_score == 0.0


def test_asdict_includes_ranking_score(monkeypatch: pytest.MonkeyPatch) -> None:
    """F-09-004: serialization paths surface ranking_score."""

    mod = _load_module(monkeypatch)
    rec = _build_rec(mod)
    rec.ranking_score = 0.87
    assert dataclasses.asdict(rec)["ranking_score"] == 0.87
    assert rec.to_dict()["ranking_score"] == 0.87
