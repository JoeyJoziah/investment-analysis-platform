"""
Regression tests for DCF sensitivity_analysis self-mutation.

F-09-003 (audit 2026-04, G2a sub-theme E step 36):
``DCFModel.sensitivity_analysis`` set ``self.terminal_growth_rate = gr``
inside a nested loop, leaving the instance's terminal-growth-rate
attribute set to the *last* growth rate tested after the method
returned. Subsequent calls to ``calculate_intrinsic_value`` then used
the wrong terminal growth rate.

The fix threads ``terminal_growth_rate`` as a per-call parameter on
``calculate_intrinsic_value`` and removes the mutation.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


_DCF = (
    Path(__file__).resolve().parents[2]
    / "analytics"
    / "fundamental"
    / "valuation"
    / "dcf_model.py"
)


def _load_module(monkeypatch: pytest.MonkeyPatch):
    for name in list(sys.modules):
        if name == "backend" or name.startswith("backend."):
            monkeypatch.delitem(sys.modules, name, raising=False)

    name = "dcf_model_under_test"
    spec = importlib.util.spec_from_file_location(name, _DCF)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def test_terminal_growth_rate_unchanged_after_sensitivity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F-09-003: ``self.terminal_growth_rate`` must be the same value after sensitivity_analysis returns."""

    mod = _load_module(monkeypatch)
    model = mod.DCFModel(terminal_growth_rate=0.025)

    model.sensitivity_analysis(
        free_cash_flow=1_000_000,
        discount_rates=[0.08, 0.10, 0.12],
        growth_rates=[0.01, 0.02, 0.03, 0.04],
        shares_outstanding=10_000,
    )

    assert model.terminal_growth_rate == 0.025, (
        f"sensitivity_analysis mutated self.terminal_growth_rate "
        f"(now {model.terminal_growth_rate})"
    )


def test_calculate_intrinsic_value_accepts_terminal_growth_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F-09-003: calculate_intrinsic_value must accept terminal_growth_rate kwarg."""

    mod = _load_module(monkeypatch)
    model = mod.DCFModel(terminal_growth_rate=0.025)

    r1 = model.calculate_intrinsic_value(
        free_cash_flow=1_000_000,
        discount_rate=0.10,
        shares_outstanding=10_000,
        terminal_growth_rate=0.01,
    )
    r2 = model.calculate_intrinsic_value(
        free_cash_flow=1_000_000,
        discount_rate=0.10,
        shares_outstanding=10_000,
        terminal_growth_rate=0.04,
    )

    # Higher terminal growth → higher intrinsic value (everything else equal).
    assert r2.intrinsic_value > r1.intrinsic_value, (
        f"expected r2 > r1 (terminal growth 0.04 > 0.01), "
        f"got r1={r1.intrinsic_value}, r2={r2.intrinsic_value}"
    )

    # And the override must not have leaked back into the instance.
    assert model.terminal_growth_rate == 0.025
