"""
Regression tests for the consolidated ExtractionResult type.

F-05-005 (audit 2026-04, G2a sub-theme C step 22):
``ExtractionResult`` was defined twice — once in
``backend/etl/multi_source_extractor.py`` and once in
``backend/etl/unlimited_data_extractor.py`` — with subtly different
``data`` field types. ``isinstance()`` checks failed across the
modules. Consolidated into ``backend/etl/types.py``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


_BASE = Path(__file__).resolve().parents[2] / "etl"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_types_module_defines_extraction_result() -> None:
    """F-05-005: canonical ExtractionResult lives in backend.etl.types."""

    text = (_BASE / "types.py").read_text()
    assert "class ExtractionResult" in text


_RE_EXPORT_PATTERNS = (
    "from backend.etl.types import ExtractionResult",
    "from .types import ExtractionResult",
)


def test_multi_source_re_exports_extraction_result() -> None:
    """F-05-005: multi_source_extractor re-exports from types module.

    Accepts either the absolute (``backend.etl.types``) or relative
    (``.types``) import form — both resolve to the same module.
    """

    text = (_BASE / "multi_source_extractor.py").read_text()
    assert any(p in text for p in _RE_EXPORT_PATTERNS), (
        "multi_source_extractor.py must re-export ExtractionResult from "
        "backend.etl.types (absolute or relative form)"
    )
    occurrences = text.count("class ExtractionResult")
    assert occurrences == 0, (
        f"multi_source_extractor.py still defines its own "
        f"ExtractionResult class ({occurrences} definition(s) found)"
    )


def test_unlimited_data_extractor_re_exports_extraction_result() -> None:
    """F-05-005: unlimited_data_extractor re-exports from types module."""

    text = (_BASE / "unlimited_data_extractor.py").read_text()
    assert any(p in text for p in _RE_EXPORT_PATTERNS), (
        "unlimited_data_extractor.py must re-export ExtractionResult from "
        "backend.etl.types (absolute or relative form)"
    )
    occurrences = text.count("class ExtractionResult")
    assert occurrences == 0
