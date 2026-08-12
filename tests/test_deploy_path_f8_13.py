"""Deploy-path regression tests for scope-13 findings (2026-08 audit, C4).

Source-level file reads, runs under ``pytest --noconftest``. Runtime
behavior (nginx -t, docker build, compose config) is verified with the real
tools in CI/local runs; these tests pin the configuration state.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROD = REPO_ROOT / "docker-compose.production.yml"
NGINX_SSL = REPO_ROOT / "infrastructure" / "docker" / "nginx" / "nginx-ssl.conf"
HEADERS = REPO_ROOT / "infrastructure" / "docker" / "frontend" / "security-headers.conf"


class TestQ1_NginxAdoption_F8_13_001_002_003_014_017:
    """Q1: nginx-ssl.conf is authoritative. The previously mounted
    nginx.optimized.conf could not start (duplicate location /, duplicate
    /api/ via the location-bearing headers include, and user appuser which
    does not exist in nginx:alpine)."""

    def test_production_mounts_nginx_ssl_conf(self):
        t = PROD.read_text()
        assert "nginx-ssl.conf:/etc/nginx/nginx.conf" in t
        assert "nginx.optimized.conf" not in t

    def test_broken_config_retired(self):
        assert not (REPO_ROOT / "infrastructure" / "docker" / "frontend"
                    / "nginx.optimized.conf").exists()

    def test_headers_include_is_headers_only(self):
        """F8-13-002: the included snippet must not declare location blocks
        (nginx aborts on duplicate /api/ when it does)."""
        assert re.search(r"^\s*location\b", HEADERS.read_text(), re.M) is None

    def test_no_nonexistent_user_directive(self):
        """F8-13-003: nginx:alpine ships no appuser."""
        assert "user appuser" not in NGINX_SSL.read_text()

    def test_upstream_frontend_matches_container_port(self):
        """Production frontend serves on container port 80 (host 3000:80);
        an upstream of frontend:3000 would 502 every page load."""
        assert "server frontend:80;" in NGINX_SSL.read_text()

    def test_healthcheck_uses_health_endpoint(self):
        """F8-13-017: /nginx_status is allow-listed/internal; /health exists
        on both listeners and is the base-compose convention."""
        t = PROD.read_text()
        m = re.search(r"nginx:.*?healthcheck:\s*\n\s*test: \[([^\]]*)\]", t, re.S)
        assert m and "/health" in m.group(1), "nginx healthcheck must probe /health"
        assert "localhost/nginx_status" not in t

    def test_moved_deny_rules_live_in_the_server_block(self):
        """F8-13-002: the location-scoped deny rules move into the server
        that needs them, once each."""
        t = NGINX_SSL.read_text()
        assert t.count("deny all;") >= 3, "attack/dotfile/vcs denies missing"

    def test_ssl_guide_claim_is_now_true(self):
        """F8-13-014: the guide said production uses nginx-ssl.conf; the
        compose mount now makes that statement true."""
        t = PROD.read_text()
        assert "infrastructure/docker/nginx/nginx-ssl.conf" in t


class TestF8_13_005_EnvFileOptional:
    """A gitignored .env must not hard-fail every documented compose
    invocation from a clean clone."""

    def test_env_file_entries_are_optional(self):
        t = (REPO_ROOT / "docker-compose.yml").read_text()
        assert "required: false" in t
        plain = re.findall(r"env_file:\s*\n\s*- \.env\s*$", t, re.M)
        assert plain == [], "short-form env_file: - .env remains"


class TestF8_13_006_PerformanceOverlay:
    """The overlay must express limits in the same deploy.resources shape
    as the base file — mem_limit conflicts fatally with it."""

    def test_no_mem_limit_conflict_keys(self):
        t = (REPO_ROOT / "docker-compose.performance.yml").read_text()
        assert "mem_limit" not in t
        assert "deploy" in t and "resources" in t


class TestF8_13_021_ObsoleteVersionKey:
    def test_no_version_keys_in_any_compose_file(self):
        offenders = [
            p.name for p in REPO_ROOT.glob("docker-compose*.yml")
            if re.search(r"^version:", p.read_text(), re.M)
        ]
        assert offenders == [], offenders


class TestF8_13_007_DevStages:
    """The documented dev overlay requests target: development from both
    root Dockerfiles; the stages must exist."""

    def test_backend_has_development_stage(self):
        assert re.search(r"^FROM .* AS development", (REPO_ROOT / "Dockerfile.backend").read_text(), re.M)

    def test_frontend_has_development_stage(self):
        assert re.search(r"^FROM .* AS development", (REPO_ROOT / "Dockerfile.frontend").read_text(), re.M)


class TestF8_13_008_FrontendTestService:
    """frontend_test requested a nonexistent target and mounted source over
    an nginx image with no npm; unit tests run in the CI frontend job."""

    def test_frontend_test_service_removed(self):
        t = (REPO_ROOT / "docker-compose.test.yml").read_text()
        assert "frontend_test" not in t


class TestC4_DefaultTargetFootgun:
    """DISCOVERED during C4 (not in the audit): Dockerfile.backend's last
    stage is `test`, and Docker builds the LAST stage when no target is
    given — so every target-less build site (production-deploy, staging,
    ci, reusable-build, release-management, security-scan, base compose,
    deploy scripts) was building and shipping the TEST image (CMD pytest,
    dev deps). Every build of the backend Dockerfile must pin a target."""

    def _sites(self):
        import subprocess
        out = subprocess.run(
            ["git", "grep", "-n", "Dockerfile.backend", "--",
             ".github/workflows", "scripts", "docker-compose.yml",
             "docker-compose.dev.yml", "docker-compose.test.yml",
             "docker-compose.production.yml"],
            cwd=REPO_ROOT, capture_output=True, text=True,
        ).stdout.strip().split("\n")
        return [l for l in out if l]

    def test_every_backend_build_site_pins_a_target(self):
        offenders = []
        for line in self._sites():
            path, lineno, _ = line.split(":", 2)
            text = (REPO_ROOT / path).read_text().split("\n")
            ln = int(lineno)
            window = "\n".join(text[max(0, ln - 4):ln + 6])
            if ("build" in window or "docker build" in window) and "target" not in window:
                offenders.append(f"{path}:{lineno}")
        assert offenders == [], offenders


class TestF8_13_009_SingleImageLineage:
    """Two divergent production image lineages were both live (GHA built the
    root Dockerfiles; blue-green built infrastructure/.../Dockerfile.optimized
    — py3.12 vs 3.11, node 20 vs 18, checksum pinning vs none). One canonical
    lineage: the root Dockerfiles (T3.1/D4)."""

    def test_no_optimized_dockerfiles_referenced_or_present(self):
        import subprocess
        hits = subprocess.run(
            ["git", "grep", "-l", "Dockerfile.optimized", "--",
             ":!docs", ":!tests"],
            cwd=REPO_ROOT, capture_output=True, text=True,
        ).stdout.strip()
        assert hits == "", hits
        assert not list((REPO_ROOT / "infrastructure" / "docker").rglob("Dockerfile*"))

    def test_no_stray_frontend_web_dockerfile(self):
        assert not (REPO_ROOT / "frontend" / "web" / "Dockerfile").exists()

    def test_blue_green_builds_the_canonical_dockerfiles(self):
        t = (REPO_ROOT / "scripts" / "deployment" / "blue_green_deploy.sh").read_text()
        assert 'Dockerfile.backend"' in t and 'Dockerfile.frontend"' in t
        assert "--target production" in t, "frontend build must pin the production stage"

    def test_validate_structure_asserts_canonical_paths(self):
        t = (REPO_ROOT / "scripts" / "testing" / "validation" / "validate_structure.py").read_text()
        assert "infrastructure/docker/backend/Dockerfile" not in t
        assert '"Dockerfile.backend"' in t


class TestF8_13_011_DataDirOutsideRepo:
    """Default DATA_DIR must not put Postgres/Redis/monitoring storage
    inside the tracked working tree (accidental commits; git clean would
    destroy production state)."""

    def test_no_volume_device_defaults_into_the_repo(self):
        t = PROD.read_text()
        assert "${DATA_DIR:-./data}" not in t
        assert "${DATA_DIR:-/var/lib/investment-platform}" in t

    def test_env_template_documents_data_dir(self):
        t = (REPO_ROOT / ".env.production.example").read_text()
        assert "DATA_DIR" in t


class TestF8_13_018_019_StandaloneProduction:
    """Q2: production is a STANDALONE stack. The overlay usage comment was
    a lie (the two-file merge unioned source bind-mounts over the immutable
    image and collided two nginx services on 80/443)."""

    def test_no_overlay_usage_comment(self):
        t = PROD.read_text()
        assert "-f docker-compose.yml -f docker-compose.production.yml" not in t
        assert "docker compose -f docker-compose.production.yml" in t

    def test_base_compose_has_no_edge_nginx(self):
        t = (REPO_ROOT / "docker-compose.yml").read_text()
        assert "image: nginx:alpine" not in t
        assert "nginx-prometheus-exporter" not in t


class TestF8_13_024_RuntimeMounts:
    """./logs bind-mounts are created root-owned by the engine while the
    backend runs as appuser (cannot write); ./static serves nothing."""

    def test_logs_use_a_named_volume(self):
        t = PROD.read_text()
        assert "./logs:/app/logs" not in t
        assert "app_logs:/app/logs" in t

    def test_static_mount_dropped(self):
        assert "./static" not in PROD.read_text()
