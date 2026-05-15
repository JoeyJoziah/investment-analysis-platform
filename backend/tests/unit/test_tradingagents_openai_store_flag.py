"""
Regression tests for the OpenAI ``store`` data-residency flag.

F-04-005 (audit 2026-04, G2a sub-theme B step 6): three callsites in
``dataflows/interface.py`` (``get_stock_news_openai``,
``get_global_news_openai``, ``get_fundamentals_openai``) hardcoded
``store=True``, which retains prompt content on OpenAI servers. This
must default to ``False`` and be driven by an explicit
``openai_store_responses`` config flag so the data-residency posture is
auditable.

These tests inspect the source file directly to avoid pulling in
LangChain/yfinance/Reddit deps that interface.py loads at import time.
"""

from __future__ import annotations

import re
from pathlib import Path


_INTERFACE_PATH = (
    Path(__file__).resolve().parents[2]
    / "TradingAgents"
    / "tradingagents"
    / "dataflows"
    / "interface.py"
)
_DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "TradingAgents"
    / "tradingagents"
    / "default_config.py"
)


def test_no_hardcoded_store_true_in_interface() -> None:
    """F-04-005: ``store=True`` literal must be gone from interface.py."""

    text = _INTERFACE_PATH.read_text()
    assert "store=True" not in text, (
        "hardcoded store=True remains in dataflows/interface.py — "
        "OpenAI prompt retention must be config-gated"
    )


def test_store_flag_threaded_from_config() -> None:
    """F-04-005: all three callsites must read ``openai_store_responses``."""

    text = _INTERFACE_PATH.read_text()
    occurrences = re.findall(r'store\s*=\s*[^,\n]*openai_store_responses', text)
    assert len(occurrences) == 3, (
        f"expected 3 config-driven ``store=...`` callsites, found {len(occurrences)}"
    )


def test_default_config_disables_store() -> None:
    """F-04-005: default posture is ``openai_store_responses=False``."""

    text = _DEFAULT_CONFIG_PATH.read_text()
    assert "openai_store_responses" in text, (
        "default_config.py must declare openai_store_responses key"
    )
    match = re.search(r'"openai_store_responses"\s*:\s*(\S+?)\s*[,}]', text)
    assert match is not None, "openai_store_responses key not parseable"
    assert match.group(1) == "False", (
        f"openai_store_responses default must be False, got {match.group(1)!r}"
    )
