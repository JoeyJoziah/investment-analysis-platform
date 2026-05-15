"""
Regression tests for wildcard imports in TradingAgents.

F-04-007 (audit 2026-04, G2a sub-theme B step 8): three modules used
``from tradingagents.agents import *`` which (a) defeats static analysis
(mypy/pyflakes), (b) hides the actual dependency surface, and (c) in the
case of ``agents/utils/agent_states.py`` creates a circular import path
because ``agents/__init__.py`` imports ``agent_states`` first.
"""

from __future__ import annotations

from pathlib import Path


_BASE = (
    Path(__file__).resolve().parents[2]
    / "TradingAgents"
    / "tradingagents"
)

_WILDCARD_TARGETS = [
    _BASE / "agents" / "utils" / "agent_states.py",
    _BASE / "graph" / "setup.py",
    _BASE / "graph" / "trading_graph.py",
]


def test_no_wildcard_imports_from_agents_package() -> None:
    """F-04-007: ``from tradingagents.agents import *`` must be gone."""

    offenders = []
    for p in _WILDCARD_TARGETS:
        text = p.read_text()
        if "from tradingagents.agents import *" in text:
            offenders.append(str(p))
    assert not offenders, (
        f"wildcard imports remain: {offenders}"
    )
