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
