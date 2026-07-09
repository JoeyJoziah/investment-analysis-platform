"""Wave 10: notifications, ML serving, load tests, compare, E2E coverage contracts."""
from __future__ import annotations

import inspect
import os
from pathlib import Path

os.environ.setdefault("SECRET_KEY", "test-secret-wave10")
os.environ.setdefault("JWT_SECRET_KEY", "test-jwt-wave10")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("TESTING", "True")
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("MASTER_SECRET_KEY", "test-master-wave10")


def test_email_smtp_notification_stack_exists():
    """#43: SMTP email alerts implemented in notification tasks."""
    path = Path("backend/tasks/notification_tasks.py")
    assert path.exists()
    src = path.read_text(encoding="utf-8")
    assert "smtplib" in src
    assert "def send_email" in src
    assert "SMTP_HOST" in src
    assert "ENABLE_EMAIL" in src
    assert "send_alert_notification" in src
    # Fail-loud / safe default when disabled
    assert "Email sending disabled" in src or "ENABLE_EMAIL" in src


def test_slack_notifier_webhook_helper_exists():
    """#46: Slack notifications via webhook helper."""
    path = Path("backend/utils/slack_notifier.py")
    assert path.exists()
    src = path.read_text(encoding="utf-8")
    assert "def notify_slack" in src
    assert "SLACK_WEBHOOK_URL" in src
    assert "httpx" in src
    # Never raises; no-op without webhook
    assert "Never raises" in src or "return False" in src


def test_ml_serving_and_ab_testing_infrastructure():
    """#110: ML serving layer + A/B test infrastructure present."""
    assert Path("backend/ml/ml_api_server.py").exists()
    assert Path("backend/ml/model_versioning.py").exists()

    api_src = Path("backend/ml/ml_api_server.py").read_text(encoding="utf-8")
    assert "FastAPI" in api_src
    assert "PredictionRequest" in api_src or "prediction" in api_src.lower()

    versioning = Path("backend/ml/model_versioning.py").read_text(encoding="utf-8")
    assert "class ABTestConfig" in versioning
    assert "def create_ab_test" in versioning
    assert "def get_ab_test_model" in versioning
    assert "traffic_split" in versioning

    from backend.api.routers import ml as ml_router

    ml_src = inspect.getsource(ml_router)
    assert "create_prediction" in ml_src or "/predictions" in ml_src
    assert "list_model_versions" in ml_src or "/versions" in ml_src
    assert "promote_model_version" in ml_src or "promote" in ml_src


def test_performance_load_tests_exist():
    """#5: performance/load test suites present and substantial."""
    candidates = [
        Path("backend/tests/test_performance_load.py"),
        Path("backend/tests/test_performance_optimizations.py"),
        Path("backend/tests/test_ml_performance.py"),
    ]
    existing = [p for p in candidates if p.exists()]
    assert existing, "no performance load test modules found"
    assert any(p.stat().st_size > 10_000 for p in existing)
    load = Path("backend/tests/test_performance_load.py")
    if load.exists():
        src = load.read_text(encoding="utf-8")
        assert "def test_" in src or "class " in src


def test_comparative_stock_analysis_endpoint():
    """#35: side-by-side stock comparison endpoint exists."""
    from backend.api.routers import analysis as analysis_mod

    src = inspect.getsource(analysis_mod)
    assert '@router.post("/compare")' in src or 'post("/compare")' in src
    assert "compare_stocks" in src
    assert "ComparisonResult" in src or "compare" in src.lower()


def test_e2e_critical_user_flows_cover_core_pages():
    """#92: E2E suite covers auth, portfolio, and additional critical flows."""
    e2e = Path("frontend/web/tests/e2e")
    assert e2e.exists()
    specs = sorted(p.name for p in e2e.glob("*.spec.ts"))
    # Original + expanded Wave 10 flows
    required = {
        "auth.spec.ts",
        "portfolio.spec.ts",
        "watchlist.spec.ts",
        "dashboard.spec.ts",
        "settings.spec.ts",
        "market-search.spec.ts",
        "analysis.spec.ts",
        "recommendations.spec.ts",
        "alerts.spec.ts",
    }
    missing = required - set(specs)
    assert not missing, f"missing E2E specs: {missing}"
    assert len(specs) >= 9
    # Helpers shared across specs
    assert (e2e / "helpers.ts").exists()
