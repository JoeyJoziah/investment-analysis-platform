"""
Regression tests for the Johansen "fake test" removal.

F-09-007 (audit 2026-04, G2a sub-theme E step 39):
``CointegrationAnalyzer._johansen_test`` silently delegated to
Engle-Granger and returned its result. Callers selecting
``CointegrationMethod.JOHANSEN`` believed they were getting a
different statistical test but got identical numbers — false
confidence in a downstream stat-arb strategy.

Per workpaper §3, the cheaper-and-honest fix is to remove the
JOHANSEN enum value entirely. If real Johansen support is needed
later, implement properly via
``statsmodels.tsa.vector_ar.vecm.coint_johansen``.
"""

from __future__ import annotations

from pathlib import Path


_PATH = (
    Path(__file__).resolve().parents[2]
    / "analytics"
    / "statistical"
    / "cointegration_analyzer.py"
)


def test_johansen_enum_value_removed() -> None:
    """F-09-007: ``JOHANSEN = "johansen"`` enum member must be gone."""

    text = _PATH.read_text()
    # The actual enum-member line was ``    JOHANSEN = "johansen"``.
    # Comments documenting the removal may still mention the name —
    # match only the assignment form.
    assert "JOHANSEN = " not in text, (
        "CointegrationMethod still declares the JOHANSEN enum member"
    )


def test_johansen_method_helper_removed() -> None:
    """F-09-007: ``def _johansen_test`` must be gone."""

    text = _PATH.read_text()
    assert "def _johansen_test" not in text, (
        "_johansen_test helper still defined — it silently delegated to "
        "Engle-Granger which is exactly the false-confidence bug"
    )


def test_test_cointegration_raises_for_unknown_method() -> None:
    """F-09-007: passing a non-ENGLE_GRANGER method must raise, not fallthrough."""

    text = _PATH.read_text()
    assert "raise NotImplementedError" in text, (
        "test_cointegration must raise NotImplementedError for unsupported "
        "methods so callers cannot silently get the wrong test"
    )
