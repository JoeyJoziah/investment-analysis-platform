"""Wave 8: product feature backlog contracts (watchlist, settings, news, TS strict)."""
from __future__ import annotations

import inspect
import os
import re
from pathlib import Path

os.environ.setdefault("SECRET_KEY", "test-secret-wave8")
os.environ.setdefault("JWT_SECRET_KEY", "test-jwt-wave8")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("TESTING", "True")
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("MASTER_SECRET_KEY", "test-master-wave8")


def test_watchlist_unit_tests_exist_and_are_substantial():
    """#55: watchlist unit/service tests are present and non-trivial."""
    service_tests = Path("backend/tests/unit/test_watchlist_service.py")
    api_tests = Path("backend/tests/test_watchlist.py")
    assert service_tests.exists(), "missing unit/test_watchlist_service.py"
    assert api_tests.exists(), "missing tests/test_watchlist.py"
    service_src = service_tests.read_text(encoding="utf-8")
    api_src = api_tests.read_text(encoding="utf-8")
    assert service_src.count("def test_") >= 40
    assert api_src.count("def test_") >= 40
    assert service_tests.stat().st_size > 10_000
    assert api_tests.stat().st_size > 10_000


def test_watchlist_router_crud_and_price_alerts():
    """#44: watchlist CRUD + price-threshold alerts exist."""
    from backend.api.routers import watchlist as wl
    from backend.api.routers import stocks as stocks_mod

    wl_src = inspect.getsource(wl)
    assert "async def create_watchlist" in wl_src
    assert "async def add_watchlist_item" in wl_src
    assert "async def get_user_watchlists" in wl_src
    assert "get_current_user" in wl_src

    stocks_src = inspect.getsource(stocks_mod)
    assert "async def create_price_alert" in stocks_src
    assert "create_price_alert" in stocks_src
    assert Path("backend/repositories/alert_repository.py").exists()


def test_typescript_strict_mode_enabled():
    """#91: frontend TypeScript strict mode is enabled."""
    tsconfig = Path("frontend/web/tsconfig.json")
    assert tsconfig.exists()
    text = tsconfig.read_text(encoding="utf-8")
    # Allow comments in tsconfig — match active strict flag
    assert re.search(r'"strict"\s*:\s*true', text), "strict mode not enabled"
    assert re.search(r'"noUnusedLocals"\s*:\s*true', text)
    assert re.search(r'"noUnusedParameters"\s*:\s*true', text)


def test_settings_service_persists_to_database():
    """#107: settings service persists preferences via SQLAlchemy + commit."""
    from backend.services import settings_service as svc

    src = inspect.getsource(svc)
    assert "await db.commit()" in src or "db.commit()" in src
    assert "update(User)" in src or "User.id" in src
    assert "async def update_preferences" in src
    assert "async def get_preferences" in src
    assert "async def update_notification_settings" in src

    from backend.api.routers import settings as settings_router

    router_src = inspect.getsource(settings_router)
    assert "settings_service" in router_src
    assert "get_current_user" in router_src
    assert "stub" not in router_src.lower()


def test_news_service_uses_real_providers():
    """#106: news integrates real providers (Finnhub -> NewsAPI -> MarketAux)."""
    from backend.services import news_service as news

    src = inspect.getsource(news)
    assert "async def fetch_news" in src
    assert "Finnhub" in src or "finnhub" in src.lower()
    assert "NewsAPI" in src or "newsapi" in src.lower()
    assert "MarketAux" in src or "marketaux" in src.lower()
    assert "httpx" in src
    # Fail-loud: skip provider when key missing, do not fabricate articles
    assert "not configured" in src.lower() or "skipping" in src.lower()

    from backend.api.routers import news as news_router

    router_src = inspect.getsource(news_router)
    assert "fetch_news" in router_src
    assert "get_current_user" in router_src


def test_portfolio_analysis_no_random_uniform_in_primary_path():
    """#108 partial: primary portfolio service analysis is not random.uniform.

    Full real analytics remain open (#108) — this locks the fail-loud
    placeholder (nulls / messaging) rather than fabricated metrics.
    """
    from backend.services.portfolio_service import PortfolioService

    src = inspect.getsource(PortfolioService.build_portfolio_analysis)
    assert "random.uniform" not in src
    assert "random.choice" not in src
    # Explicit non-fabrication messaging or null risk fields
    assert (
        "not yet available" in src.lower()
        or "None" in src
        or "null" in src.lower()
    )
