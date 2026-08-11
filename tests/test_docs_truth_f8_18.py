"""Docs-truth regression tests for scope-18 findings (2026-08 audit).

Source-level (pure file reads, no backend imports) so they run under
``pytest --noconftest``.

F8-18-001: docs/SUPERSEDED.md must describe the real archive (out-of-repo,
125 files per the archive README), must not point readers at the emptied
in-repo directory, must not map originals that no longer exist, and must
index the 2026-06 supersession layer that STATUS.md points to.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SUPERSEDED = REPO_ROOT / "docs" / "SUPERSEDED.md"


def _text() -> str:
    return SUPERSEDED.read_text()


class TestF8_18_001_SupersededIndex:
    def test_no_stale_88_count(self):
        """The '88 prior audit reports' in-repo claim was wrong on both count
        and location (archive moved out-of-repo, 125 files)."""
        assert "88 prior audit reports" not in _text()

    def test_states_out_of_repo_location(self):
        """Must cite the out-of-repo location recorded by the archive README
        (docs/audits/2026-04/_meta/prior-reports-archive/README.md)."""
        assert "audit-archives-2026-04" in _text()

    def test_indexes_2026_06_supersession_layer(self):
        """STATUS.md sends readers here for 'the index of stale archived
        docs'; the 2026-06 archive layer must therefore be indexed."""
        assert "_superseded/2026-06" in _text()

    def test_no_mapping_rows_for_deleted_originals(self):
        """tests/TEST_*.md, E2E_*.md, QUICK_START.md, FILE_MANIFEST no longer
        exist (ls tests/*.md -> README.md only); dead rows must be dropped."""
        t = _text()
        assert "tests/TEST_*.md" not in t
        assert "QUICK_START.md, FILE_MANIFEST" not in t

    def test_no_claim_that_archive_copies_are_in_repo(self):
        """Reading-order step 1 must not instruct opening an in-repo archive
        copy path that returns only a README."""
        assert "_meta/prior-reports-archive/{name}.archived.md" not in _text()


class TestF8_18_004_005_BackendDocsSilo:
    """F8-18-004: two contradictory '100% COMPLETE' wave reports; F8-18-005:
    backend/docs/** sat outside every archive sweep. Fix = extend the T2.2
    sweep: wave artifacts move to docs/_superseded/2026-06/, the one
    reference doc stays with a version header."""

    DEPLOYMENT = REPO_ROOT / "backend" / "docs" / "deployment"
    SUPERSEDED_0606 = REPO_ROOT / "docs" / "_superseded" / "2026-06"

    def test_no_wave_artifacts_left_in_backend_docs_deployment(self):
        remaining = list(self.DEPLOYMENT.glob("*.md")) if self.DEPLOYMENT.exists() else []
        assert len(remaining) == 0, f"unswept: {[p.name for p in remaining]}"

    def test_no_100_percent_complete_claims_in_backend_docs(self):
        hits = [
            p for p in (REPO_ROOT / "backend" / "docs").rglob("*.md")
            if "100% COMPLETE" in p.read_text(errors="ignore")
        ]
        assert hits == [], f"contradictory completion claims remain: {hits}"

    def test_wave_reports_archived_with_banner(self):
        moved = sorted(self.SUPERSEDED_0606.glob("backend-deployment-*.md"))
        assert len(moved) == 7, f"expected 7 archived wave reports, got {len(moved)}"
        for p in moved:
            assert "SUPERSEDED (2026-06)" in p.read_text()[:2000], p.name

    def test_database_optimization_guide_kept_with_header(self):
        guide = REPO_ROOT / "backend" / "docs" / "DATABASE_OPTIMIZATION_GUIDE.md"
        assert guide.exists()
        head = "\n".join(guide.read_text().splitlines()[:12])
        assert "Version:" in head and "Last Updated:" in head

    def test_wave_0_14_report_swept_from_docs_root(self):
        assert not (REPO_ROOT / "docs" / "WAVE_0_14_STATUS_REPORT.cgd.md").exists()
        assert (self.SUPERSEDED_0606 / "WAVE_0_14_STATUS_REPORT.cgd.md").exists()
