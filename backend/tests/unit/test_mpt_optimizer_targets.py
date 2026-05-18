"""
Regression tests for MPT optimizer target enforcement.

F-09-010 (audit 2026-04, G2a sub-theme E step 40):
``PortfolioOptimizer.optimize`` ignored ``target_return`` and
``target_volatility`` parameters entirely — it always returned equal
weights regardless of what the caller requested. The efficient frontier
helper that walks ``target_return`` values produced 50 identical
results.

The fix uses ``scipy.optimize.minimize(method="SLSQP")`` to actually
solve the mean-variance problem.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


_PATH = (
    Path(__file__).resolve().parents[2]
    / "analytics"
    / "portfolio"
    / "modern_portfolio_theory.py"
)


def _load(monkeypatch: pytest.MonkeyPatch):
    for name in list(sys.modules):
        if name == "backend" or name.startswith("backend."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    spec = importlib.util.spec_from_file_location("mpt_under_test", _PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["mpt_under_test"] = module
    monkeypatch.setitem(sys.modules, "mpt_under_test", module)
    spec.loader.exec_module(module)
    return module


def _three_asset_returns():
    """Three assets with distinct mean returns so a real solver moves weights."""
    rng = np.random.default_rng(42)
    n = 252
    # Asset A: high return, high vol.
    a = rng.normal(0.0008, 0.02, n)
    # Asset B: medium.
    b = rng.normal(0.0004, 0.015, n)
    # Asset C: low return, low vol.
    c = rng.normal(0.0002, 0.008, n)
    return pd.DataFrame({"A": a, "B": b, "C": c})


def test_no_target_maximizes_sharpe(monkeypatch: pytest.MonkeyPatch) -> None:
    """F-09-010: with no target, optimizer should not produce equal weights."""

    mod = _load(monkeypatch)
    opt = mod.PortfolioOptimizer()
    result = opt.optimize(_three_asset_returns())
    weights = np.array(list(result.weights.values()))
    # Equal-weight signature would be [1/3, 1/3, 1/3] exactly.
    assert not np.allclose(weights, 1.0 / 3, atol=1e-6), (
        f"optimizer returned equal weights {weights} — the solver did not run"
    )
    assert abs(weights.sum() - 1.0) < 1e-6


def test_target_return_constraint_is_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    """F-09-010: portfolio return must meet or exceed the target."""

    mod = _load(monkeypatch)
    opt = mod.PortfolioOptimizer()
    df = _three_asset_returns()
    # Pick a target somewhere in the middle of the feasible range.
    target = df.mean().mean() * 252  # average annualized return
    result = opt.optimize(df, target_return=target)
    assert result.expected_return >= target - 1e-3, (
        f"target_return={target:.4f} not met (got {result.expected_return:.4f})"
    )


def test_target_volatility_constraint_is_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    """F-09-010: portfolio volatility must not exceed the target."""

    mod = _load(monkeypatch)
    opt = mod.PortfolioOptimizer()
    df = _three_asset_returns()
    # Pick a relatively tight vol cap that should bind.
    target_vol = 0.15
    result = opt.optimize(df, target_volatility=target_vol)
    assert result.volatility <= target_vol + 1e-3, (
        f"target_volatility={target_vol} exceeded (got {result.volatility:.4f})"
    )


def test_efficient_frontier_walks_returns(monkeypatch: pytest.MonkeyPatch) -> None:
    """F-09-010: efficient frontier must produce varying portfolios.

    ``get_efficient_frontier`` yields a list of ``(volatility, return)``
    tuples — we expect the return coordinate to vary across the 10
    points (previously every point was identical because the underlying
    optimizer ignored the target_return).
    """

    mod = _load(monkeypatch)
    opt = mod.PortfolioOptimizer()
    df = _three_asset_returns()
    frontier = opt.get_efficient_frontier(df, n_points=10)
    rets = [ret for (_vol, ret) in frontier]
    assert len(set(round(r, 4) for r in rets)) > 1, (
        f"efficient frontier collapsed to a single point: {rets}"
    )
