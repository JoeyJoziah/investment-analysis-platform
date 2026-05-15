"""
Regression tests for TradingAgents setup.py dependencies.

F-04-009 (audit 2026-04, G2a sub-theme B step 9): setup.py shipped only
``langchain-openai`` and pinned it at ``>=0.0.2`` even though
``trading_graph.py`` imports ``ChatAnthropic`` and ``ChatGoogleGenerativeAI``
unconditionally. A fresh ``pip install`` of the package therefore failed
at import time for any non-OpenAI provider.
"""

from __future__ import annotations

import re
from pathlib import Path


_SETUP_PATH = (
    Path(__file__).resolve().parents[2]
    / "TradingAgents"
    / "setup.py"
)


def _install_requires() -> str:
    text = _SETUP_PATH.read_text()
    match = re.search(r"install_requires\s*=\s*\[(.*?)\]", text, re.DOTALL)
    assert match is not None, "install_requires not found in setup.py"
    return match.group(1)


def test_setup_declares_langchain_anthropic() -> None:
    """F-04-009: ChatAnthropic import requires langchain-anthropic dep."""

    body = _install_requires()
    assert "langchain-anthropic" in body, (
        "trading_graph.py imports ChatAnthropic but langchain-anthropic "
        "is not declared in install_requires"
    )


def test_setup_declares_langchain_google_genai() -> None:
    """F-04-009: ChatGoogleGenerativeAI import requires langchain-google-genai dep."""

    body = _install_requires()
    assert "langchain-google-genai" in body, (
        "trading_graph.py imports ChatGoogleGenerativeAI but "
        "langchain-google-genai is not declared in install_requires"
    )


def test_langchain_openai_pinned_at_modern_minor() -> None:
    """F-04-009: bump langchain-openai pin to a version that has ChatOpenAI."""

    body = _install_requires()
    match = re.search(r'langchain-openai\s*>=\s*([0-9.]+)', body)
    assert match is not None, "langchain-openai version pin not found"
    parts = [int(p) for p in match.group(1).split(".")]
    # >=0.1.0 — pre-0.1 releases lacked stable ChatOpenAI shapes.
    assert parts >= [0, 1, 0], (
        f"langchain-openai must be pinned >=0.1.0 (got {match.group(1)})"
    )
