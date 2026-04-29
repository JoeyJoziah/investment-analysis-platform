"""
Regression tests for TradingAgents risk_manager.

F-04-002 (audit 2026-04, G2a): risk_manager_node assigned
``state['news_report']`` to the local ``fundamentals_report`` variable, so the
LLM prompt was built with the news content twice and never received the
fundamentals report at all. The fail-first test below proves the prompt
contains the *fundamentals* content (not duplicated news) once the bug is
fixed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

# Importing ``tradingagents.agents.managers.risk_manager`` via the normal
# package import path drags in heavy LangChain/LLM dependencies through
# ``tradingagents/agents/__init__.py``. The module under test only uses the
# ``time`` and ``json`` stdlib modules, so we load it directly from its file
# path to keep this regression test isolated and dependency-free.
_RISK_MANAGER_PATH = (
    Path(__file__).resolve().parents[2]
    / "TradingAgents"
    / "tradingagents"
    / "agents"
    / "managers"
    / "risk_manager.py"
)
_spec = importlib.util.spec_from_file_location(
    "tradingagents_risk_manager_under_test", _RISK_MANAGER_PATH
)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_module)
create_risk_manager = _module.create_risk_manager


def _build_state() -> Dict[str, Any]:
    """State fixture with deliberately distinct news/fundamentals content."""

    return {
        "company_of_interest": "ACME",
        "market_report": "MARKET",
        "news_report": "NEWS_CONTENT",
        "fundamentals_report": "FUND_CONTENT",
        "sentiment_report": "SENT",
        "investment_plan": "PLAN",
        "risk_debate_state": {
            "history": "",
            "risky_history": "",
            "safe_history": "",
            "neutral_history": "",
            "current_risky_response": "",
            "current_safe_response": "",
            "current_neutral_response": "",
            "count": 0,
        },
    }


def test_risk_manager_uses_fundamentals_report_not_news_duplicate():
    """``curr_situation`` passed to memory must include fundamentals, not 2x news.

    The ``curr_situation`` string the node builds is the semantic key handed
    to ``memory.get_memories(...)`` so the past-trade lookup is conditioned on
    the real ticker context (market + sentiment + news + fundamentals).

    Pre-fix (F-04-002): ``fundamentals_report = state['news_report']`` causes
    ``curr_situation`` to contain the news text twice and never the
    fundamentals text — so memory retrieval is conditioned on a duplicated
    news report instead of fundamentals.

    Post-fix: ``curr_situation`` contains both the news and fundamentals
    content exactly once each.
    """

    captured: Dict[str, str] = {}

    def fake_get_memories(curr_situation: str, n_matches: int = 2):  # noqa: ARG001
        captured["curr_situation"] = curr_situation
        return []

    llm = MagicMock()
    llm.invoke.return_value = SimpleNamespace(content="JUDGE_DECISION")

    memory = MagicMock()
    memory.get_memories.side_effect = fake_get_memories

    node = create_risk_manager(llm, memory)
    node(_build_state())

    curr_situation = captured["curr_situation"]

    # Sanity: market/sentiment/news included as before.
    assert "MARKET" in curr_situation
    assert "SENT" in curr_situation
    assert "NEWS_CONTENT" in curr_situation

    # Hard regression assertions for F-04-002:
    # 1. fundamentals content must appear in the memory-retrieval key.
    # 2. news content must NOT be duplicated.
    assert "FUND_CONTENT" in curr_situation, (
        "fundamentals content missing from curr_situation — F-04-002 bug present"
    )
    assert curr_situation.count("NEWS_CONTENT") == 1, (
        "news content duplicated in curr_situation — F-04-002 bug present "
        f"(count={curr_situation.count('NEWS_CONTENT')})"
    )


if __name__ == "__main__":  # pragma: no cover - manual invocation helper
    pytest.main([__file__, "-v"])
