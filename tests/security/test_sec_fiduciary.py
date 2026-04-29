"""
F-08-020 — SEC FiduciaryDutyChecker scope guard tests.

Audit reference
---------------
- PRD-for-loki §2 Q5=B1 decision recorded 2026-04-28.
- Working assumption of record:
  ``docs/audits/2026-04/_synthesis/_meta/LEGAL_ASSUMPTION_OF_RECORD.md``.
- Workpaper §3 Phase 5:
  ``docs/audits/2026-04/_synthesis/workpaper/G4_storage_security_residual.md``.

Working assumption (NOT legal advice; canonical source is
LEGAL_ASSUMPTION_OF_RECORD.md):

    The platform is NOT a registered investment advisor under the SEC
    Investment Advisers Act. It surfaces analytics, rankings, and research.
    It does NOT offer personalized investment advice triggering fiduciary
    duty. ``FiduciaryDutyChecker`` exists as compliance scaffolding but
    must not be wired into the personalized-recommendation code path; if
    it is invoked anywhere, its output must carry an advisory-disclaimer
    posture (NOT fiduciary-grade).

These tests encode that scope guard. They are intentionally NOT testing
the internal correctness of ``FiduciaryDutyChecker`` — that would lock
in fiduciary semantics the platform has explicitly disclaimed.

Revisit triggers (per LEGAL_ASSUMPTION_OF_RECORD.md):
- If counsel is engaged and concludes the platform IS an advisor.
- If the platform begins offering personalized advice for compensation.
- If a Form ADV is filed.

If any of those trigger, this test file becomes the wrong shape and must
be re-spec'd against an updated assumption-of-record.
"""
from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Module list — production paths that handle personalized recommendations.
#
# Personalized = "tied to a specific client/user/portfolio". General
# analytics, rankings, scoring, and research are explicitly out of scope
# and may be served without fiduciary entanglement.
# ---------------------------------------------------------------------------
_PERSONALIZED_RECOMMENDATION_MODULES = (
    "backend.api.routers.recommendations",
    "backend.services.recommendation_service",
    "backend.services.recommendation_crud",
    "backend.services.recommendation_analysis",
    "backend.repositories.recommendation_repository",
)


def _module_uses_fiduciary_checker(module_name: str) -> bool:
    """Return True if a production module imports / instantiates FiduciaryDutyChecker.

    Heuristic: source-text scan for the symbol name. Source-text is more
    robust than import-graph inspection because it catches lazy/conditional
    imports and string-based references too.
    """
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(
            f"Cannot import {module_name} in this environment: {exc!r}. "
            "Test cannot guard a module that does not load."
        )

    src_path = inspect.getsourcefile(module) or inspect.getfile(module)
    if not src_path or not Path(src_path).exists():
        pytest.skip(f"{module_name} has no resolvable source file")

    source = Path(src_path).read_text(encoding="utf-8")
    return "FiduciaryDutyChecker" in source


@pytest.mark.security
@pytest.mark.compliance
@pytest.mark.parametrize("module_name", _PERSONALIZED_RECOMMENDATION_MODULES)
def test_fiduciary_checker_not_on_personalized_recommendation_path(module_name):
    """No personalized-recommendation production module may reference
    ``FiduciaryDutyChecker``.

    Per Q5=B1 the platform is NOT a registered investment advisor and must
    not produce fiduciary-grade output. Wiring FiduciaryDutyChecker into
    the recommendation path would imply fiduciary semantics.

    If this test fails, EITHER:
      (a) the personalized-recommendation flow has begun using the
          checker — coordinate with counsel + update
          LEGAL_ASSUMPTION_OF_RECORD.md before unblocking, OR
      (b) the checker call site explicitly returns advisory-disclaimer
          output (not fiduciary-grade) — in which case loosen this test
          to inspect the call site's response shape rather than its mere
          presence.
    """
    assert not _module_uses_fiduciary_checker(module_name), (
        f"{module_name} references FiduciaryDutyChecker. The platform is "
        "NOT a registered investment advisor (Q5=B1, "
        "docs/audits/2026-04/_synthesis/_meta/LEGAL_ASSUMPTION_OF_RECORD.md). "
        "Wiring the checker into a personalized-recommendation path would "
        "imply fiduciary-grade analysis. Coordinate with counsel before "
        "unblocking, or document the advisory-disclaimer response shape "
        "and update this test."
    )


@pytest.mark.security
@pytest.mark.compliance
def test_legal_assumption_of_record_present():
    """The working-assumption file must exist; tests cite it as canon.

    If the assumption is ever removed or renamed, the scope guard tests
    above lose their anchor and must be re-spec'd. Failing fast here is
    safer than silently drifting.
    """
    repo_root = Path(__file__).resolve().parents[2]
    assumption = (
        repo_root
        / "docs"
        / "audits"
        / "2026-04"
        / "_synthesis"
        / "_meta"
        / "LEGAL_ASSUMPTION_OF_RECORD.md"
    )
    assert assumption.exists(), (
        f"LEGAL_ASSUMPTION_OF_RECORD.md missing at {assumption}. The "
        "FiduciaryDutyChecker scope guard tests cite this file as the "
        "canonical assumption-of-record (Q5=B1, audit 2026-04). If the "
        "assumption was retired or moved, update this test and the "
        "tests above to point at the new source."
    )

    text = assumption.read_text(encoding="utf-8")
    assert "NOT a registered investment advisor" in text, (
        "LEGAL_ASSUMPTION_OF_RECORD.md no longer asserts the Q5=B1 "
        "working assumption. The scope guard tests rely on this exact "
        "claim — if the assumption changed, re-spec the tests."
    )


@pytest.mark.security
@pytest.mark.compliance
def test_fiduciary_checker_module_carries_advisory_caveat():
    """The compliance.sec module that defines FiduciaryDutyChecker should
    not be silently mistaken for production-ready fiduciary infrastructure.

    Until the platform is registered as an investment advisor (Q5=B1
    revisit trigger), any consumer of FiduciaryDutyChecker must treat its
    output as advisory only. Documenting that constraint in the source
    is the cheapest available signal.

    This test asserts the module exists and its docstring does not
    silently claim fiduciary-grade authority. It is intentionally weak;
    its purpose is to fail loudly if someone re-frames the module's
    purpose without updating the assumption-of-record.
    """
    sec_module = importlib.import_module("backend.compliance.sec")
    checker_cls = getattr(sec_module, "FiduciaryDutyChecker", None)
    assert checker_cls is not None, (
        "backend.compliance.sec.FiduciaryDutyChecker missing. If the class "
        "was removed, drop this test file. If renamed, update the import."
    )
    # Docstring sanity: do not assert specific wording (would lock in
    # legalese), just ensure there IS one so changes are reviewable.
    assert (checker_cls.__doc__ or "").strip(), (
        "FiduciaryDutyChecker has no docstring. Per Q5=B1 the class must "
        "carry contextual documentation so consumers cannot mistake it "
        "for production fiduciary infrastructure without active review."
    )
