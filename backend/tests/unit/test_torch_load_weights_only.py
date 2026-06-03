"""
Regression tests for ``torch.load(weights_only=True)`` enforcement.

F-03-002 (audit 2026-04, G2a sub-theme A step 27):
Five ``torch.load(...)`` callsites in ``backend/ml/`` deserialized
artifacts with the default ``weights_only=False``, which is
pickle-based and equivalent to arbitrary code execution if an
attacker can write to (or substitute) a model artifact path. CVE
class: torch.load pickle RCE (cf. CVE-2025-32434).

The workpaper acceptance gate is:

    grep -rn "torch.load" backend/ml/ | grep -v weights_only

returning empty. The literal pattern collides with the explanatory
docstring/comments that document the fix, so the tighter form below
matches only the function-call syntax (``torch.load(``) — preserving
the audit intent without flagging the rationale comments. See
[[feedback_test_anchor_logic]] in user memory.
"""

from __future__ import annotations

import re
from pathlib import Path


_ML_DIR = Path(__file__).resolve().parents[2] / "ml"


def test_every_torch_load_call_passes_weights_only() -> None:
    """F-03-002: every ``torch.load(...)`` invocation must include weights_only."""

    offenders: list[str] = []
    for py in _ML_DIR.rglob("*.py"):
        text = py.read_text()
        # Match the call form ``torch.load(`` and capture up to the
        # matching close-paren on the same statement.
        for m in re.finditer(r"torch\.load\(([^)]*)\)", text):
            args = m.group(1)
            if "weights_only" not in args:
                line_no = text.count("\n", 0, m.start()) + 1
                offenders.append(f"{py.relative_to(_ML_DIR.parent)}:{line_no}")
    assert not offenders, (
        f"torch.load call(s) without weights_only kwarg — RCE risk: {offenders}"
    )


def test_weights_only_value_is_true_not_false() -> None:
    """F-03-002: ``weights_only=False`` is the unsafe default; ban it."""

    offenders: list[str] = []
    for py in _ML_DIR.rglob("*.py"):
        text = py.read_text()
        for m in re.finditer(
            r"torch\.load\([^)]*weights_only\s*=\s*False[^)]*\)", text
        ):
            line_no = text.count("\n", 0, m.start()) + 1
            offenders.append(f"{py.relative_to(_ML_DIR.parent)}:{line_no}")
    assert not offenders, (
        f"torch.load called with weights_only=False — unsafe: {offenders}"
    )


def test_expected_callsite_count_unchanged() -> None:
    """F-03-002: every torch.load callsite in backend/ml/ is guarded.

    Originally 5 callsites at audit time. After commit f5dd8ac on main
    (file-relocation refactor that introduced ``runtime_models.py``)
    the count is 7 — this PR guards the 2 new state_dict loads too.
    The meaningful contract is in
    ``test_every_torch_load_call_passes_weights_only``; this test
    just records the current expected count so unexpected new
    callsites (e.g. a future refactor) get flagged for review.
    """

    count = 0
    for py in _ML_DIR.rglob("*.py"):
        count += len(re.findall(r"torch\.load\(", py.read_text()))
    assert count == 7, (
        f"expected 7 torch.load callsites in backend/ml/ post f5dd8ac, "
        f"found {count}"
    )
