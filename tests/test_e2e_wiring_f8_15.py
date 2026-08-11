"""E2E-wiring regression tests for scope-15 findings (2026-08 audit, C3).

Source-level file reads, runs under ``pytest --noconftest``.
"""

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FRONTEND = REPO_ROOT / "frontend" / "web"
CT = REPO_ROOT / ".github" / "workflows" / "comprehensive-testing.yml"


class TestF8_15_002_RealE2EInCI:
    """The CI 'End-to-End Tests' job ran an echo stub via npm run
    cypress:run under continue-on-error, so 10 Playwright specs gated
    nothing. The job must run the real suite and be able to fail."""

    def test_cypress_echo_stubs_deleted(self):
        pkg = json.loads((FRONTEND / "package.json").read_text())
        scripts = pkg.get("scripts", {})
        assert "cypress:run" not in scripts
        assert "cypress:open" not in scripts

    def test_no_cypress_references_left_in_workflow(self):
        assert "cypress" not in CT.read_text().lower()

    def _e2e_job(self):
        t = CT.read_text()
        start = t.index("  e2e-tests:")
        nxt = re.search(r"^  [a-z0-9_-]+:\s*$", t[start + 15:], re.M)
        return t[start:start + 15 + (nxt.start() if nxt else len(t))]

    def test_e2e_job_runs_playwright_without_continue_on_error(self):
        job = self._e2e_job()
        assert "playwright test" in job, "job must run the real suite"
        assert "continue-on-error" not in job, (
            "an E2E job that cannot fail gates nothing"
        )

    def test_artifacts_point_at_playwright_outputs(self):
        job = self._e2e_job()
        assert "playwright-report" in job
        assert "test-results" in job


class TestF8_15_023_ProjectMatrix:
    """PR/push runs use one browser project; the full 5-browser matrix runs
    only on the nightly schedule."""

    def test_chromium_only_outside_schedule(self):
        t = CT.read_text()
        assert "--project=chromium" in t

    def test_full_matrix_reserved_for_schedule(self):
        t = CT.read_text()
        assert re.search(r"github\.event_name\s*==\s*'schedule'", t), (
            "no schedule-gated full-matrix path"
        )
