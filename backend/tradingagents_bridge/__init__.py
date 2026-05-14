"""Bridge between the canonical TradingAgents package and IAP persistence.

This module is the IAP-side glue. It:
  - Provides a single entry point that runs TradingAgents and writes the result
    into IAP's `recommendations` and (optionally) `orders` tables.
  - Adapts TradingAgents' free-text `final_trade_decision` to IAP's structured
    Recommendation schema via the LLM-based parser already in
    `tradingagents.execution.parser`.
  - Keeps the active TradingAgents codebase library-pure: no IAP imports inside
    TradingAgents itself.

Activation:
  - `TRADINGAGENTS_PATH` env var must point at the canonical workspace
    (default already wired in cache_aware_agents.py).
  - `TRADINGAGENTS_MEMORY_DIR` env var should point at IAP's shared
    `data/tradingagents_memory/` so both projects share the ChromaDB store.
  - `TRADINGAGENTS_PERSIST=1` env var enables Postgres persistence.
"""
from .persistence import (
    persist_tradingagents_decision,
    map_action_to_iap,
    PersistenceResult,
)

__all__ = [
    "persist_tradingagents_decision",
    "map_action_to_iap",
    "PersistenceResult",
]
