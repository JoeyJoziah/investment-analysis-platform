"""Regression tests for the 2026-08-14 audit remediations.

These stay import-light: source contracts plus isolated service/JWT units.
They do not boot the full FastAPI app.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

REPO = Path(__file__).resolve().parents[3]


def test_oauth2_decode_has_no_hs256_fallback():
    source = (REPO / "backend" / "auth" / "oauth2.py").read_text(encoding="utf-8")
    assert "HS256 fallback was removed" in source
    assert "JWT_ALGORITHM_FALLBACK" not in source.split("def decode_access_token", 1)[1]


def test_main_environment_defaults_to_settings_not_development():
    source = (REPO / "backend" / "api" / "main.py").read_text(encoding="utf-8")
    assert 'os.getenv("ENVIRONMENT", settings.ENVIRONMENT)' in source
    assert 'os.getenv("ENVIRONMENT", "development")' not in source


def test_settings_api_keys_write_is_admin_and_settings_environment():
    source = (REPO / "backend" / "api" / "routers" / "settings.py").read_text(
        encoding="utf-8"
    )
    assert "get_current_admin_user" in source
    assert "app_settings" in source
    assert 'os.getenv("ENVIRONMENT", "development")' not in source


def test_ml_promote_rollback_require_admin():
    from backend.api.routers import ml as ml_mod

    assert "get_current_admin_user" in inspect.getsource(ml_mod.promote_model_version)
    assert "get_current_admin_user" in inspect.getsource(ml_mod.rollback_model_version)


def test_analysis_analyze_and_batch_require_user():
    from backend.api.routers import analysis as analysis_mod

    assert "get_current_user" in inspect.getsource(analysis_mod.analyze_stock)
    assert "get_current_user" in inspect.getsource(analysis_mod.batch_analysis)


def test_ws_triggers_require_admin():
    from backend.api.routers import websocket as ws_mod

    assert "get_current_admin_user" in inspect.getsource(ws_mod.trigger_alert)
    assert "get_current_admin_user" in inspect.getsource(ws_mod.trigger_news_broadcast)
    assert "get_current_admin_user" in inspect.getsource(ws_mod.get_active_connections)


def test_agents_capabilities_require_user():
    from backend.api.routers import agents as agents_mod

    assert "get_current_user" in inspect.getsource(agents_mod.get_agent_capabilities)


def test_auth_refresh_consumes_refresh_token_not_access():
    from backend.api.routers import auth as auth_mod

    source = inspect.getsource(auth_mod.refresh_token)
    assert "TokenType.REFRESH" in source
    assert "get_current_user" not in source
    assert "auth_rate_limit" in source


def test_auth_logout_revokes_bearer():
    from backend.api.routers import auth as auth_mod

    source = inspect.getsource(auth_mod.logout)
    assert "revoke_token" in source


def test_repository_exposes_owned_portfolio_helpers():
    from backend.repositories.portfolio_repository import PortfolioRepository

    assert hasattr(PortfolioRepository, "get_user_portfolio")
    assert hasattr(PortfolioRepository, "get_owned_portfolio")
    assert hasattr(PortfolioRepository, "get_portfolio_positions")
    assert hasattr(PortfolioRepository, "get_recent_transactions")


def test_ml_api_server_requires_token_except_health():
    source = (REPO / "backend" / "ml" / "ml_api_server.py").read_text(encoding="utf-8")
    assert "ML_API_TOKEN" in source
    assert '"/health"' in source
    assert 'os.getenv("ML_BIND_HOST", "127.0.0.1")' in source


def test_ml_compose_binds_localhost():
    source = (REPO / "docker-compose.ml-production.yml").read_text(encoding="utf-8")
    assert "127.0.0.1:8001:8001" in source
    assert "ML_API_TOKEN" in source


@pytest.mark.asyncio
async def test_execute_trade_does_not_write_unowned_book():
    from backend.services.trading_service import TradingService

    svc = TradingService()
    svc.repository = MagicMock()
    svc.repository.get_owned_portfolio = AsyncMock(return_value=None)
    svc.repository.add_position = AsyncMock()

    result = await svc.execute_trade(
        1,
        {
            "symbol": "AAPL",
            "side": "buy",
            "order_type": "market",
            "quantity": 1,
            "price": 10,
        },
        user_id=99,
    )
    assert result.get("not_found") is True
    svc.repository.add_position.assert_not_called()


@pytest.mark.asyncio
async def test_impact_hides_unowned_portfolio():
    from backend.services.trading_service import TradingService

    svc = TradingService()
    svc.repository = MagicMock()
    svc.repository.get_owned_portfolio = AsyncMock(return_value=None)

    result = await svc.calculate_portfolio_impact(
        7,
        {"symbol": "AAPL", "side": "buy", "quantity": 1, "price": 10},
        user_id=3,
    )
    assert result.get("success") is False
    assert result.get("not_found") is True
    assert "7" not in result.get("error", "")


@pytest.mark.asyncio
async def test_validate_order_without_user_id_is_not_found():
    from backend.services.trading_service import TradingService

    svc = TradingService()
    svc.repository = MagicMock()
    svc.repository.get_by_id = AsyncMock()
    svc.repository.get_owned_portfolio = AsyncMock(return_value=None)

    result = await svc.validate_order(
        {
            "portfolio_id": 1,
            "symbol": "AAPL",
            "side": "buy",
            "order_type": "market",
            "quantity": 1,
            "price": 10,
        },
        user_id=None,
    )
    assert result.get("valid") is False
    assert result.get("not_found") is True
    svc.repository.get_by_id.assert_not_called()
