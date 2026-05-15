"""
Regression tests for TradingAgents trader.

F-04-008 (audit 2026-04, G2a sub-theme B step 3): create_trader returned a
``functools.partial`` over a ``(state, name)`` signature. The positional
``state`` argument was vulnerable to drift if langgraph ever called the
node with a kwarg, and the name parameter leaked into the public signature.
The fail-first tests below assert that the returned callable takes a
single positional ``state`` argument (no ``name`` parameter) and that the
returned dict carries ``sender == "Trader"``.
"""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
from unittest.mock import MagicMock


_TRADER_PATH = (
    Path(__file__).resolve().parents[2]
    / "TradingAgents"
    / "tradingagents"
    / "agents"
    / "trader"
    / "trader.py"
)
_spec = importlib.util.spec_from_file_location(
    "tradingagents_trader_under_test", _TRADER_PATH
)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_module)
create_trader = _module.create_trader


def _state() -> dict:
    return {
        "company_of_interest": "ACME",
        "investment_plan": "BUY",
        "market_report": "M",
        "sentiment_report": "S",
        "news_report": "N",
        "fundamentals_report": "F",
    }


def _llm() -> MagicMock:
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="FINAL TRANSACTION PROPOSAL: **BUY**")
    return llm


def _memory() -> MagicMock:
    mem = MagicMock()
    mem.get_memories.return_value = []
    return mem


def test_create_trader_returns_single_arg_callable() -> None:
    """Closure factory: returned callable takes only ``state`` (F-04-008)."""

    node = create_trader(_llm(), _memory())
    sig = inspect.signature(node)
    params = list(sig.parameters.values())
    assert len(params) == 1, f"expected single state param, got {params!r}"
    assert params[0].name == "state"


def test_create_trader_does_not_return_functools_partial() -> None:
    """The closure-factory form must not be a ``functools.partial`` (F-04-008)."""

    import functools

    node = create_trader(_llm(), _memory())
    assert not isinstance(node, functools.partial)


def test_trader_node_sets_sender_to_trader() -> None:
    """Sender identity baked into the closure, not a runtime kwarg (F-04-008)."""

    node = create_trader(_llm(), _memory())
    result = node(_state())
    assert result["sender"] == "Trader"
