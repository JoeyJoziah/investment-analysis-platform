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


def test_staging_sbom_steps_are_non_blocking():
    """SBOM must not fail image publish on external Syft download errors."""
    yml = Path(".github/workflows/staging-deploy.yml").read_text(encoding="utf-8")
    assert "anchore/sbom-action" in yml
    # Both SBOM steps should tolerate failure (continue-on-error)
    assert yml.count("continue-on-error: true") >= 2
    assert "Generate SBOM for backend" in yml
    assert "Generate SBOM for frontend" in yml


def test_staging_trivy_uses_single_primary_image_ref():
    """Multi-line metadata tags break Trivy image-ref parsing (Staging security-scan)."""
    yml = Path(".github/workflows/staging-deploy.yml").read_text(encoding="utf-8")
    assert "Export primary image refs" in yml
    assert "primary-images" in yml
    # Job outputs must resolve to primary single-ref step, not meta.outputs.tags
    assert "steps.primary-images.outputs.backend" in yml
    assert "steps.primary-images.outputs.frontend" in yml
    # Trivy consumes the single-ref job outputs
    assert "needs.build-images.outputs.backend-image" in yml
    assert "image-ref:" in yml


def test_staging_codeql_sarif_upload_is_v3():
    """CodeQL Action v1/v2 are deprecated; Staging must use upload-sarif@v3."""
    yml = Path(".github/workflows/staging-deploy.yml").read_text(encoding="utf-8")
    assert "github/codeql-action/upload-sarif@v3" in yml
    assert "github/codeql-action/upload-sarif@v2" not in yml
    assert "github/codeql-action/upload-sarif@v1" not in yml
    # One SARIF file per upload step (v3 does not accept multi-path lists reliably)
    assert "backend-trivy-results.sarif" in yml
    assert "frontend-trivy-results.sarif" in yml
    assert "security-events: write" in yml


def test_staging_deploy_skips_ssh_when_host_unset():
    """Without STAGING_HOST, SSH deploy must skip instead of failing appleboy missing host."""
    yml = Path(".github/workflows/staging-deploy.yml").read_text(encoding="utf-8")
    assert "Preflight staging host" in yml
    assert "configured=false" in yml
    assert "STAGING_HOST unset" in yml or "STAGING_HOST is not set" in yml
    # Smoke/rollback only when a real deploy was attempted
    assert "needs.deploy-staging.outputs.deployed == 'true'" in yml
