"""Wave 11: Sentry, backups, thesis, infra, ETL unify, workflow contracts."""
from __future__ import annotations

import inspect
import os
from pathlib import Path

os.environ.setdefault("SECRET_KEY", "test-secret-wave11")
os.environ.setdefault("JWT_SECRET_KEY", "test-jwt-wave11")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("TESTING", "True")
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("MASTER_SECRET_KEY", "test-master-wave11")


def test_sentry_backend_setup_is_optional_and_wired():
    """#102: backend Sentry bootstrap is optional and lifespan-wired."""
    from backend.monitoring import sentry_setup

    src = inspect.getsource(sentry_setup)
    assert "def init_sentry" in src
    assert "SENTRY_DSN" in src
    # Safe when unset
    assert sentry_setup.init_sentry(dsn="") is False
    assert sentry_setup.is_sentry_enabled() is False

    main_src = Path("backend/api/main.py").read_text(encoding="utf-8")
    assert "init_sentry" in main_src
    assert "sentry_setup" in main_src

    fe = Path("frontend/web/src/monitoring/sentry.ts")
    assert fe.exists()
    fe_src = fe.read_text(encoding="utf-8")
    assert "VITE_SENTRY_DSN" in fe_src
    assert "initFrontendSentry" in fe_src
    index = Path("frontend/web/src/index.tsx").read_text(encoding="utf-8")
    assert "initFrontendSentry" in index


def test_database_backup_automation_exists():
    """#38: automated DB backup task + scripts + retention env."""
    maint = Path("backend/tasks/maintenance_tasks.py").read_text(encoding="utf-8")
    assert "def backup_database" in maint
    assert "pg_dump" in maint
    assert "BACKUP_DIR" in maint

    celery = Path("backend/tasks/celery_app.py").read_text(encoding="utf-8")
    assert "backup-database" in celery
    assert "backup_database" in celery

    assert Path("scripts/backup.sh").exists()
    assert Path("scripts/restore-backup.sh").exists()
    assert Path("scripts/verify-backup.sh").exists()

    env = Path(".env.example").read_text(encoding="utf-8")
    assert "BACKUP_RETENTION_DAYS" in env
    assert "ENABLE_AUTOMATIC_BACKUPS" in env


def test_investment_thesis_templates_and_api():
    """#33: thesis documentation templates + API surface."""
    template = Path("docs/templates/investment_thesis_template.md")
    assert template.exists()
    text = template.read_text(encoding="utf-8")
    assert "Investment Thesis" in text
    assert "Executive Summary" in text

    assert Path("backend/api/routers/thesis.py").exists()
    assert Path("frontend/web/src/pages/InvestmentThesis.tsx").exists()
    public = Path("frontend/web/public/docs/templates/investment_thesis_template.md")
    assert public.exists()


def test_infrastructure_stack_docker_and_monitoring():
    """#54: Docker + Prometheus/Grafana monitoring infrastructure present."""
    assert Path("docker-compose.yml").exists()
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")
    assert "postgres" in compose.lower()
    assert "redis" in compose.lower()

    assert Path("infrastructure/monitoring/prometheus.yml").exists() or Path(
        "infrastructure/monitoring/prometheus.prod.yml"
    ).exists()
    assert Path("infrastructure/monitoring/grafana").is_dir()
    assert Path("infrastructure/docker/nginx/nginx-ssl.conf").exists()  # TLS ready (#2 related)


def test_unified_etl_extractor_consolidates_providers():
    """#97: unified extractor facade over multi-source providers."""
    from backend.etl.unified_extractor import (
        MultiSourceStockExtractor,
        UnifiedStockExtractor,
    )

    assert UnifiedStockExtractor is MultiSourceStockExtractor
    src = Path("backend/etl/multi_source_extractor.py").read_text(encoding="utf-8")
    for provider in ("Finnhub", "Polygon", "Alpha", "Yahoo"):
        assert provider.lower() in src.lower() or provider in src


def test_github_actions_typecheck_consolidated():
    """#104: redundant mypy workflow deprecated; type-check.yml is canonical."""
    type_check = Path(".github/workflows/type-check.yml")
    assert type_check.exists()
    mypy = Path(".github/workflows/mypy.yml")
    assert mypy.exists()
    mypy_text = mypy.read_text(encoding="utf-8")
    assert "DEPRECATED" in mypy_text or "deprecated" in mypy_text.lower()
    assert "type-check.yml" in mypy_text
    # Deprecated workflow should not run on every push
    assert "push:" not in mypy_text or "workflow_dispatch" in mypy_text
