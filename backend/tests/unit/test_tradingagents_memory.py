"""
Regression tests for TradingAgents memory (embeddings).

F-04-004 (audit 2026-04, G2a sub-theme B step 5): ``FinancialSituationMemory``
always constructed an OpenAI client regardless of ``llm_provider``. When
the user selected Anthropic or Google, the OpenAI client was pointed at
the wrong base URL and silently failed or sent embedding requests to the
wrong endpoint. The fail-first tests below assert that unsupported
providers raise ``NotImplementedError`` at construction time, and that
the OpenAI-compatible providers (openai, openrouter, ollama) still work.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


_MEMORY_PATH = (
    Path(__file__).resolve().parents[2]
    / "TradingAgents"
    / "tradingagents"
    / "agents"
    / "utils"
    / "memory.py"
)


def _load_memory_module(monkeypatch: pytest.MonkeyPatch):
    """Load memory.py with chromadb and openai stubbed out."""

    chromadb_stub = MagicMock()
    chromadb_stub.Client.return_value.create_collection.return_value = MagicMock()
    chromadb_config_stub = MagicMock()
    openai_stub = MagicMock()

    monkeypatch.setitem(sys.modules, "chromadb", chromadb_stub)
    monkeypatch.setitem(sys.modules, "chromadb.config", chromadb_config_stub)
    monkeypatch.setitem(sys.modules, "openai", openai_stub)

    spec = importlib.util.spec_from_file_location(
        "tradingagents_memory_under_test", _MEMORY_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_anthropic_provider_raises_not_implemented(monkeypatch: pytest.MonkeyPatch) -> None:
    """F-04-004: Anthropic provider has no OpenAI-compatible embeddings."""

    mod = _load_memory_module(monkeypatch)
    cfg = {
        "llm_provider": "anthropic",
        "backend_url": "https://api.anthropic.com",
    }
    with pytest.raises(NotImplementedError, match="anthropic"):
        mod.FinancialSituationMemory("test", cfg)


def test_google_provider_raises_not_implemented(monkeypatch: pytest.MonkeyPatch) -> None:
    """F-04-004: Google provider has no OpenAI-compatible embeddings."""

    mod = _load_memory_module(monkeypatch)
    cfg = {
        "llm_provider": "google",
        "backend_url": "https://generativelanguage.googleapis.com",
    }
    with pytest.raises(NotImplementedError, match="google"):
        mod.FinancialSituationMemory("test", cfg)


def test_openai_provider_constructs_successfully(monkeypatch: pytest.MonkeyPatch) -> None:
    """F-04-004: OpenAI provider continues to work."""

    mod = _load_memory_module(monkeypatch)
    cfg = {
        "llm_provider": "openai",
        "backend_url": "https://api.openai.com/v1",
    }
    inst = mod.FinancialSituationMemory("test", cfg)
    assert inst.embedding == "text-embedding-3-small"


def test_ollama_provider_constructs_successfully(monkeypatch: pytest.MonkeyPatch) -> None:
    """F-04-004: Ollama (OpenAI-compatible) continues to work."""

    mod = _load_memory_module(monkeypatch)
    cfg = {
        "llm_provider": "ollama",
        "backend_url": "http://localhost:11434/v1",
    }
    inst = mod.FinancialSituationMemory("test", cfg)
    assert inst.embedding == "nomic-embed-text"
