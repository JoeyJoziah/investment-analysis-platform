"""
Regression tests for the ETLOrchestrator extractor alias.

F-05-007 (audit 2026-04, G2a sub-theme C step 20):
``ETLOrchestrator.__init__`` set ``self.legacy_extractor`` but two code
paths (lines 387 and 633) referenced ``self.extractor``, raising
AttributeError on every realtime/single-ticker call.

This test inspects the source rather than instantiating the
orchestrator, because instantiation pulls in the full backend stack
(Postgres, Redis, validators).
"""

from __future__ import annotations

import re
from pathlib import Path


_ORCHESTRATOR = (
    Path(__file__).resolve().parents[2]
    / "etl"
    / "etl_orchestrator.py"
)


def test_init_sets_extractor_alias() -> None:
    """F-05-007: ``self.extractor`` must be assigned alongside legacy_extractor."""

    text = _ORCHESTRATOR.read_text()
    assert re.search(
        r"self\.extractor\s*=\s*self\.legacy_extractor", text
    ), (
        "ETLOrchestrator.__init__ must alias self.extractor to the "
        "legacy_extractor to satisfy the realtime/single-ticker paths"
    )


def test_extractor_references_have_a_defined_owner() -> None:
    """F-05-007: every ``self.extractor.<method>(`` call must be backed by an init assignment."""

    text = _ORCHESTRATOR.read_text()
    extractor_calls = re.findall(r"self\.extractor\.\w+\(", text)
    assert extractor_calls, "no self.extractor.* calls — bug spec stale?"
    assert re.search(r"self\.extractor\s*=", text), (
        f"self.extractor is referenced ({len(extractor_calls)} call sites) "
        f"but never assigned"
    )
