"""
Unified multi-provider stock data extractor (#97).

Single entrypoint for ETL extraction that consolidates Alpha Vantage,
Finnhub, Polygon, and Yahoo (and related fallbacks) behind one abstraction.

Prefer importing from this module rather than the legacy per-source modules
or unlimited_* variants.
"""

from __future__ import annotations

from backend.etl.multi_source_extractor import (
    IntelligentSourceRouter,
    MultiSourceStockExtractor,
    SourcePriority,
)

# Canonical names for consumers
UnifiedStockExtractor = MultiSourceStockExtractor
ProviderRouter = IntelligentSourceRouter

__all__ = [
    "UnifiedStockExtractor",
    "MultiSourceStockExtractor",
    "ProviderRouter",
    "IntelligentSourceRouter",
    "SourcePriority",
]
