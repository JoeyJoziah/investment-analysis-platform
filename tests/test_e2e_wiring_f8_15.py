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


class TestF8_15_012_WebServerEnv:
    """The backend webServer entry must carry an explicit env block —
    backend.config.settings hard-requires SECRET_KEY/JWT_SECRET_KEY/
    REDIS_URL at import, so without one uvicorn never binds and Playwright
    times out at webServer startup instead of failing a test."""

    def test_backend_webserver_has_env_block(self):
        t = (FRONTEND / "playwright.config.ts").read_text()
        m = re.search(r"\{[^{}]*uvicorn[^{}]*env:\s*\{(?P<env>[^}]*)\}", t, re.S)
        assert m, "backend webServer entry has no env block"
        env = m.group("env")
        for var in ("SECRET_KEY", "JWT_SECRET_KEY", "REDIS_URL", "DATABASE_URL"):
            assert var in env, f"webServer env missing {var}"


class TestF8_15_013_022_LoginHelper:
    """F8-15-013: loginAsTestUser must not silently degrade to
    'not logged in' when the form fails to render, must not accept the
    site root as a post-login URL, and must assert a real auth signal.
    F8-15-022: no hardcoded default password literal."""

    def _helpers(self):
        return (FRONTEND / "tests" / "e2e" / "helpers.ts").read_text()

    def test_no_silent_catch_fallback_on_login_form(self):
        assert ".catch(() => false)" not in self._helpers()

    def test_no_site_root_alternative_in_post_login_url(self):
        assert re.search(r"waitForURL\(/[^/]*\\\/\$", self._helpers()) is None, (
            r"the \/$ alternative matches the unauthenticated site root"
        )

    def test_asserts_post_login_auth_signal(self):
        assert "access_token" in self._helpers(), (
            "no post-login signal assertion (auth breakage would pass)"
        )

    def test_no_hardcoded_default_password(self):
        assert "PortfolioTest" not in self._helpers()
        hits = [p for p in (FRONTEND / "tests").rglob("*.ts")
                if "PortfolioTest123" in p.read_text()]
        assert hits == [], hits

    def test_missing_password_env_fails_fast(self):
        assert re.search(r"E2E_USER_PASSWORD.*(throw|Error)", self._helpers(), re.S), (
            "unset E2E_USER_PASSWORD must raise a clear error"
        )
