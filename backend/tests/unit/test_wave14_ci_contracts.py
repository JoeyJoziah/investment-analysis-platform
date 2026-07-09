"""Wave 14: CI/Docker contract locks for staging build fix."""
from __future__ import annotations

from pathlib import Path


def test_dockerfile_backend_test_stage_does_not_copy_root_tests():
    """Staging failure: root tests/ is dockerignored; only backend/tests is required."""
    df = Path("Dockerfile.backend").read_text(encoding="utf-8")
    assert "backend/tests" in df
    # Must not require COPY of repo-root tests path (excluded by .dockerignore)
    assert "COPY --chown=appuser:appuser tests /app/tests" not in df
    assert "pytest" in df and "backend/tests" in df


def test_dockerignore_scopes_root_tests_only():
    """Only /tests/ at repo root is ignored; backend/tests remains in build context."""
    text = Path(".dockerignore").read_text(encoding="utf-8")
    assert "/tests/" in text or text.count("tests/") == 1 and "/tests/" in text
    # Avoid broad pattern that drops backend/tests from context
    lines = [
        ln.strip()
        for ln in text.splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    broad = [ln for ln in lines if ln in {"tests/", "tests", "**/tests/", "**/tests"}]
    assert not broad, f"broad tests ignore still present: {broad}"
