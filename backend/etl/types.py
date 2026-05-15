"""
Shared ETL types.

F-05-005 (audit 2026-04, G2a sub-theme C): ``ExtractionResult`` was
previously defined twice — once in ``multi_source_extractor.py`` and
once in ``unlimited_data_extractor.py`` — with subtly different ``data``
field types. Consolidated here so both modules re-export the same
class and downstream isinstance checks behave consistently.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional


@dataclass
class ExtractionResult:
    """Result of a data extraction attempt."""

    ticker: str
    success: bool
    data: Any = None
    source: Optional[str] = None
    error: Optional[str] = None
    timestamp: Optional[datetime] = None

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.now()


__all__ = ["ExtractionResult"]
