"""
Regression tests for OpenRouter/Ollama-specific kwargs on ChatOpenAI.

F-04-006 (audit 2026-04, G2a sub-theme B step 7): trading_graph.py
constructed ``ChatOpenAI`` with only ``model`` and ``base_url`` for all
three OpenAI-compatible providers. OpenRouter requires routing headers
(``HTTP-Referer`` / ``X-Title``) and Ollama benefits from explicit
``model_kwargs``. The fail-first tests below scan the source for the
required provider-specific construction patterns.

Source-level inspection avoids importing trading_graph.py (which pulls
in LangChain, ChromaDB, and yfinance at import time).
"""

from __future__ import annotations

from pathlib import Path


_TRADING_GRAPH_PATH = (
    Path(__file__).resolve().parents[2]
    / "TradingAgents"
    / "tradingagents"
    / "graph"
    / "trading_graph.py"
)


def test_openrouter_has_default_headers() -> None:
    """F-04-006: OpenRouter ChatOpenAI must declare default_headers."""

    text = _TRADING_GRAPH_PATH.read_text()
    assert "openrouter" in text.lower()
    assert "default_headers" in text, (
        "OpenRouter branch must pass default_headers to ChatOpenAI for routing"
    )
    assert "HTTP-Referer" in text, (
        "OpenRouter requires HTTP-Referer header for request attribution"
    )


def test_ollama_has_model_kwargs() -> None:
    """F-04-006: Ollama ChatOpenAI must declare model_kwargs."""

    text = _TRADING_GRAPH_PATH.read_text()
    assert "ollama" in text.lower()
    assert "model_kwargs" in text, (
        "Ollama branch must pass model_kwargs for ollama-specific options"
    )


def test_providers_are_branched_separately() -> None:
    """F-04-006: openai/openrouter/ollama no longer share a single branch."""

    text = _TRADING_GRAPH_PATH.read_text()
    # The pre-fix code had `elif provider == "openai" or provider == "ollama"
    # or provider == "openrouter"` collapsed into one branch. After the fix
    # each provider must have its own branch to receive provider-specific
    # kwargs.
    bad_pattern = (
        '"openai" or self.config["llm_provider"] == "ollama" or '
        'self.config["llm_provider"] == "openrouter"'
    )
    assert bad_pattern not in text, (
        "openai/ollama/openrouter must not share a single ChatOpenAI branch"
    )
