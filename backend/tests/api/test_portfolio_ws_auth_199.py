"""
Regression tests for P0 finding #199 — Broken Access Control.

Slice covered by this file
  REST: portfolio.py get_transactions / get_portfolio_performance / get_watchlist
  WS:  websocket.py /ws/market and /ws/portfolio/{portfolio_id}

Strategy
  These tests are route-signature-level: they assert that
    1. each REST endpoint declares `get_current_user` as a Depends() parameter,
    2. the WS handlers close with the correct policy-violation code (4401/4403)
       before ever calling websocket.accept() when auth/ownership fails.

  This avoids spinning up a full TestClient / real database while giving a
  clear, deterministic signal that the access-control guards are wired in.

  Environment note: module-level env setup mirrors backend/tests/conftest.py
  so this file can be loaded directly (e.g. with --noconftest) or via the
  shared conftest — both paths produce the same result.
"""

import os

# Must be set before any backend module import so settings validators pass and
# the sync SQLAlchemy engine does not try to use an asyncio-incompatible URL.
os.environ.setdefault("TESTING", "True")
os.environ.setdefault("DEBUG", "True")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("SECRET_KEY", "test-secret-key-finding-199")
os.environ.setdefault("JWT_SECRET_KEY", "test-jwt-secret-finding-199")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
# Prevent InsecureSecretError from security_config._require_secret outside prod.
os.environ.setdefault("ENVIRONMENT", "development")

import inspect
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _depends_on_get_current_user(func) -> bool:
    """
    Return True when *func* has a parameter whose default is a
    fastapi.Depends(get_current_user) sentinel.
    """
    from fastapi import params as fa_params
    from backend.auth.oauth2 import get_current_user

    sig = inspect.signature(func)
    for param in sig.parameters.values():
        default = param.default
        if (
            isinstance(default, fa_params.Depends)
            and default.dependency is get_current_user
        ):
            return True
    return False


# ---------------------------------------------------------------------------
# REST endpoint signature tests
# ---------------------------------------------------------------------------

class TestPortfolioEndpointSignatures:
    """Finding #199 REST slice: all three previously-unguarded endpoints must
    now declare get_current_user as a Depends() parameter."""

    def test_get_transactions_requires_current_user(self):
        from backend.api.routers.portfolio import get_transactions
        assert _depends_on_get_current_user(get_transactions), (
            "get_transactions must declare current_user: User = Depends(get_current_user) "
            "(finding #199 regression)"
        )

    def test_get_portfolio_performance_requires_current_user(self):
        from backend.api.routers.portfolio import get_portfolio_performance
        assert _depends_on_get_current_user(get_portfolio_performance), (
            "get_portfolio_performance must declare current_user: User = Depends(get_current_user) "
            "(finding #199 regression)"
        )

    def test_get_watchlist_requires_current_user(self):
        from backend.api.routers.portfolio import get_watchlist
        assert _depends_on_get_current_user(get_watchlist), (
            "get_watchlist must declare current_user: User = Depends(get_current_user) "
            "(finding #199 regression)"
        )


# ---------------------------------------------------------------------------
# WebSocket handler tests
# ---------------------------------------------------------------------------

class TestWebSocketAuthGuards:
    """Finding #199 WS slice: both /ws/market and /ws/portfolio/{id} must
    call _reject_unauthenticated() and close before websocket.accept()."""

    @pytest.mark.asyncio
    async def test_market_ws_closes_4401_when_no_token(self):
        """Anonymous connection to /ws/market must be rejected with code 4401."""
        from backend.api.routers.websocket import market_data_stream_endpoint

        mock_ws = MagicMock()
        mock_ws.close = AsyncMock()
        mock_ws.accept = AsyncMock()

        await market_data_stream_endpoint(websocket=mock_ws, token=None)

        mock_ws.close.assert_awaited_once_with(code=4401)
        mock_ws.accept.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_market_ws_closes_4401_when_invalid_token(self):
        """Malformed/expired token to /ws/market must be rejected with code 4401."""
        from backend.api.routers.websocket import market_data_stream_endpoint

        mock_ws = MagicMock()
        mock_ws.close = AsyncMock()
        mock_ws.accept = AsyncMock()

        with patch(
            "backend.api.routers.websocket._verify_bearer_token",
            return_value=None,
        ):
            await market_data_stream_endpoint(
                websocket=mock_ws, token="bad.token.value"
            )

        mock_ws.close.assert_awaited_once_with(code=4401)
        mock_ws.accept.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_portfolio_ws_closes_4401_when_no_token(self):
        """Anonymous connection to /ws/portfolio/{id} must be rejected with code 4401."""
        from backend.api.routers.websocket import portfolio_stream

        mock_ws = MagicMock()
        mock_ws.close = AsyncMock()
        mock_ws.accept = AsyncMock()

        await portfolio_stream(
            websocket=mock_ws, portfolio_id="port-123", token=None
        )

        mock_ws.close.assert_awaited_once_with(code=4401)
        mock_ws.accept.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_portfolio_ws_closes_4401_when_invalid_token(self):
        """Malformed/expired token to /ws/portfolio/{id} must be rejected with code 4401."""
        from backend.api.routers.websocket import portfolio_stream

        mock_ws = MagicMock()
        mock_ws.close = AsyncMock()
        mock_ws.accept = AsyncMock()

        with patch(
            "backend.api.routers.websocket._verify_bearer_token",
            return_value=None,
        ):
            await portfolio_stream(
                websocket=mock_ws, portfolio_id="port-123", token="bad.token"
            )

        mock_ws.close.assert_awaited_once_with(code=4401)
        mock_ws.accept.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_portfolio_ws_closes_4403_when_not_owner(self):
        """Valid token for a user who does not own the portfolio must yield code 4403."""
        import sys
        # Retrieve the actual portfolio_service module from sys.modules rather
        # than using the `import X as m` alias, because the services package
        # __init__ re-exports `portfolio_service` as a singleton attribute which
        # shadows the submodule when Python resolves the dotted name via getattr.
        from backend.api.routers.websocket import portfolio_stream  # ensure loaded
        import backend.services.portfolio_service  # ensure in sys.modules
        _ps_module = sys.modules["backend.services.portfolio_service"]
        _ps_singleton = _ps_module.portfolio_service

        mock_ws = MagicMock()
        mock_ws.close = AsyncMock()
        mock_ws.accept = AsyncMock()

        async def _fake_db_gen():
            yield MagicMock()

        with patch(
            "backend.api.routers.websocket._verify_bearer_token",
            return_value={"user_id": 99},
        ), patch.object(
            _ps_singleton,
            "compute_portfolio_detail",
            new=AsyncMock(return_value=None),
        ), patch(
            "backend.config.database.get_async_db_session",
            return_value=_fake_db_gen(),
        ):
            await portfolio_stream(
                websocket=mock_ws, portfolio_id="port-456", token="valid.token"
            )

        mock_ws.close.assert_awaited_once_with(code=4403)
        mock_ws.accept.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_portfolio_ws_accepts_authenticated_owner(self):
        """Valid token + confirmed ownership must result in websocket.accept() being called."""
        import sys
        from backend.api.routers.websocket import portfolio_stream  # ensure loaded
        from fastapi import WebSocketDisconnect
        import backend.services.portfolio_service  # ensure in sys.modules
        _ps_module = sys.modules["backend.services.portfolio_service"]
        _ps_singleton = _ps_module.portfolio_service

        mock_ws = MagicMock()
        mock_ws.close = AsyncMock()
        mock_ws.accept = AsyncMock()
        # Simulate disconnect so the streaming loop exits immediately.
        mock_ws.send_json = AsyncMock(side_effect=WebSocketDisconnect())

        async def _fake_db_gen():
            yield MagicMock()

        with patch(
            "backend.api.routers.websocket._verify_bearer_token",
            return_value={"user_id": 42},
        ), patch.object(
            _ps_singleton,
            "compute_portfolio_detail",
            new=AsyncMock(return_value={"id": "port-789"}),
        ), patch(
            "backend.config.database.get_async_db_session",
            return_value=_fake_db_gen(),
        ):
            await portfolio_stream(
                websocket=mock_ws, portfolio_id="port-789", token="owner.token"
            )

        mock_ws.accept.assert_awaited_once()
        mock_ws.close.assert_not_awaited()
