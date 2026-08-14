"""
Integration tests for the WebSocket router.

Tests cover:
- WebSocket connection establishment (market and portfolio endpoints)
- Subscription management via EnhancedConnectionManager
- Message broadcasting to connected clients
- Disconnect cleanup and stale connection handling
- Error handling for malformed messages and unknown types
- The legacy message handler path
"""

import asyncio
import json
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.routers.websocket import (
    EnhancedConnectionManager,
    MessageType,
    handle_client_message,
    send_alert,
    broadcast_news,
    cleanup_client_streams,
    router as ws_router,
)

# Lightweight FastAPI app containing only the websocket router.
# This avoids the heavy lifespan (database, redis, scheduler, ML models)
# that the main app requires and that cannot run in the test environment.
_ws_test_app = FastAPI()
_ws_test_app.include_router(ws_router, prefix="/api/v1/ws")


pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_websocket(client_id: str = "test-client") -> AsyncMock:
    """Create a mock WebSocket that records sent messages."""
    ws = AsyncMock()
    ws.send_text = AsyncMock()
    ws.send_json = AsyncMock()
    ws.accept = AsyncMock()
    return ws


def _make_mock_user(user_id: int = 1, username: str = "testuser", role: str = "analyst"):
    """Create a minimal mock user object compatible with EnhancedConnectionManager."""
    user = MagicMock()
    user.id = user_id
    user.username = username
    user.role = role
    return user


# ---------------------------------------------------------------------------
# EnhancedConnectionManager: connect / disconnect
# ---------------------------------------------------------------------------

class TestConnectionManager:
    """Tests for EnhancedConnectionManager connect and disconnect lifecycle."""

    @pytest.mark.asyncio
    async def test_connect_registers_client(self):
        """Connecting a client stores it in active_connections and initialises health tracking."""
        mgr = EnhancedConnectionManager()
        ws = _make_mock_websocket()
        user = _make_mock_user()

        await mgr.connect(ws, "client-1", user=user)

        assert "client-1" in mgr.active_connections
        conn = mgr.active_connections["client-1"]
        assert conn["websocket"] is ws
        assert conn["user_id"] == user.id
        assert conn["message_count"] == 0
        assert "client-1" in mgr.connection_health
        ws.accept.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_connect_anonymous_user(self):
        """Connecting without a user stores user_id as None."""
        mgr = EnhancedConnectionManager()
        ws = _make_mock_websocket()

        await mgr.connect(ws, "anon-1", user=None)

        assert mgr.active_connections["anon-1"]["user_id"] is None
        assert "anon-1" not in mgr.user_sessions

    @pytest.mark.asyncio
    async def test_disconnect_removes_all_state(self):
        """Disconnecting a client cleans up connections, sessions, subscriptions, and health."""
        mgr = EnhancedConnectionManager()
        ws = _make_mock_websocket()
        user = _make_mock_user()

        await mgr.connect(ws, "client-2", user=user)
        await mgr.subscribe("client-2", ["AAPL", "MSFT"])
        assert "client-2" in mgr.subscriptions

        await mgr.disconnect(ws, "client-2")

        assert "client-2" not in mgr.active_connections
        assert "client-2" not in mgr.user_sessions
        assert "client-2" not in mgr.subscriptions
        assert "client-2" not in mgr.connection_health

    @pytest.mark.asyncio
    async def test_disconnect_idempotent(self):
        """Disconnecting a client that is already gone does not raise."""
        mgr = EnhancedConnectionManager()
        ws = _make_mock_websocket()

        # Should not raise even though the client was never connected
        await mgr.disconnect(ws, "nonexistent-client")


# ---------------------------------------------------------------------------
# Subscription management
# ---------------------------------------------------------------------------

class TestSubscriptionManagement:
    """Tests for subscribe, unsubscribe, and get_subscriptions."""

    @pytest.mark.asyncio
    async def test_subscribe_adds_symbols(self):
        """Subscribing adds symbols to the client's subscription set."""
        mgr = EnhancedConnectionManager()
        ws = _make_mock_websocket()
        await mgr.connect(ws, "sub-client")

        result = await mgr.subscribe("sub-client", ["AAPL", "TSLA"])

        assert set(result) == {"AAPL", "TSLA"}
        assert mgr.get_subscriptions("sub-client") == {"AAPL", "TSLA"}

    @pytest.mark.asyncio
    async def test_subscribe_is_additive(self):
        """Subsequent subscribe calls add to the existing set, not replace it."""
        mgr = EnhancedConnectionManager()
        ws = _make_mock_websocket()
        await mgr.connect(ws, "sub-client-2")

        await mgr.subscribe("sub-client-2", ["AAPL"])
        await mgr.subscribe("sub-client-2", ["MSFT"])

        assert mgr.get_subscriptions("sub-client-2") == {"AAPL", "MSFT"}

    @pytest.mark.asyncio
    async def test_unsubscribe_removes_symbols(self):
        """Unsubscribing removes the specified symbols from the subscription set."""
        mgr = EnhancedConnectionManager()
        ws = _make_mock_websocket()
        await mgr.connect(ws, "unsub-client")

        await mgr.subscribe("unsub-client", ["AAPL", "MSFT", "GOOGL"])
        mgr.unsubscribe("unsub-client", ["MSFT"])

        assert mgr.get_subscriptions("unsub-client") == {"AAPL", "GOOGL"}

    @pytest.mark.asyncio
    async def test_get_subscriptions_returns_empty_for_unknown_client(self):
        """Querying subscriptions for an unknown client returns an empty set."""
        mgr = EnhancedConnectionManager()
        assert mgr.get_subscriptions("ghost-client") == set()


# ---------------------------------------------------------------------------
# Message broadcasting
# ---------------------------------------------------------------------------

class TestBroadcasting:
    """Tests for broadcast, send_personal_message, and send_to_subscribers."""

    @pytest.mark.asyncio
    async def test_broadcast_sends_to_all_clients(self):
        """broadcast sends the message to every connected client."""
        mgr = EnhancedConnectionManager()
        ws1 = _make_mock_websocket()
        ws2 = _make_mock_websocket()

        await mgr.connect(ws1, "bc-1")
        await mgr.connect(ws2, "bc-2")

        sent = await mgr.broadcast("hello everyone")

        assert sent == 2
        ws1.send_text.assert_awaited_once_with("hello everyone")
        ws2.send_text.assert_awaited_once_with("hello everyone")

    @pytest.mark.asyncio
    async def test_broadcast_excludes_specified_client(self):
        """broadcast with exclude= skips the excluded client."""
        mgr = EnhancedConnectionManager()
        ws1 = _make_mock_websocket()
        ws2 = _make_mock_websocket()

        await mgr.connect(ws1, "bc-ex-1")
        await mgr.connect(ws2, "bc-ex-2")

        sent = await mgr.broadcast("selective message", exclude="bc-ex-1")

        assert sent == 1
        ws1.send_text.assert_not_awaited()
        ws2.send_text.assert_awaited_once_with("selective message")

    @pytest.mark.asyncio
    async def test_broadcast_handles_failed_send(self):
        """broadcast removes clients that fail to receive the message."""
        mgr = EnhancedConnectionManager()
        ws_good = _make_mock_websocket()
        ws_bad = _make_mock_websocket()
        ws_bad.send_text.side_effect = Exception("Connection reset")

        await mgr.connect(ws_good, "good-client")
        await mgr.connect(ws_bad, "bad-client")

        sent = await mgr.broadcast("test")

        assert sent == 1
        # The failed client should have been cleaned up
        assert "bad-client" not in mgr.active_connections

    @pytest.mark.asyncio
    async def test_send_personal_message_to_connected_client(self):
        """send_personal_message delivers to a specific client and increments count."""
        mgr = EnhancedConnectionManager()
        ws = _make_mock_websocket()
        await mgr.connect(ws, "pm-client")

        result = await mgr.send_personal_message("private hello", "pm-client")

        assert result is True
        ws.send_text.assert_awaited_once_with("private hello")
        assert mgr.active_connections["pm-client"]["message_count"] == 1

    @pytest.mark.asyncio
    async def test_send_personal_message_to_unknown_client(self):
        """send_personal_message returns False for an unknown client."""
        mgr = EnhancedConnectionManager()

        result = await mgr.send_personal_message("hello", "missing-client")

        assert result is False

    @pytest.mark.asyncio
    async def test_send_to_subscribers(self):
        """send_to_subscribers sends only to clients subscribed to the given symbol."""
        mgr = EnhancedConnectionManager()
        ws_aapl = _make_mock_websocket()
        ws_msft = _make_mock_websocket()
        ws_both = _make_mock_websocket()

        await mgr.connect(ws_aapl, "sub-aapl")
        await mgr.connect(ws_msft, "sub-msft")
        await mgr.connect(ws_both, "sub-both")

        await mgr.subscribe("sub-aapl", ["AAPL"])
        await mgr.subscribe("sub-msft", ["MSFT"])
        await mgr.subscribe("sub-both", ["AAPL", "MSFT"])

        sent = await mgr.send_to_subscribers("AAPL", '{"price": 150}')

        assert sent == 2
        ws_aapl.send_text.assert_awaited_once()
        ws_msft.send_text.assert_not_awaited()
        ws_both.send_text.assert_awaited_once()


# ---------------------------------------------------------------------------
# Stale connection cleanup
# ---------------------------------------------------------------------------

class TestStaleConnectionCleanup:
    """Tests for cleanup_stale_connections and update_health."""

    @pytest.mark.asyncio
    async def test_cleanup_removes_stale_connections(self):
        """Connections whose last heartbeat exceeds the threshold are cleaned up."""
        mgr = EnhancedConnectionManager()
        ws = _make_mock_websocket()
        await mgr.connect(ws, "stale-client")

        # Artificially age the health timestamp
        mgr.connection_health["stale-client"] = datetime.now(timezone.utc) - timedelta(minutes=60)

        removed = await mgr.cleanup_stale_connections(max_age_minutes=30)

        assert removed == 1
        assert "stale-client" not in mgr.active_connections

    @pytest.mark.asyncio
    async def test_cleanup_keeps_healthy_connections(self):
        """Connections with a recent heartbeat are not removed."""
        mgr = EnhancedConnectionManager()
        ws = _make_mock_websocket()
        await mgr.connect(ws, "healthy-client")

        # Heartbeat is very recent (set by connect)
        removed = await mgr.cleanup_stale_connections(max_age_minutes=30)

        assert removed == 0
        assert "healthy-client" in mgr.active_connections

    @pytest.mark.asyncio
    async def test_update_health_refreshes_timestamp(self):
        """update_health should refresh the client's heartbeat timestamp."""
        mgr = EnhancedConnectionManager()
        ws = _make_mock_websocket()
        await mgr.connect(ws, "heartbeat-client")

        old_ts = mgr.connection_health["heartbeat-client"]
        # Small sleep to ensure timestamps differ
        await asyncio.sleep(0.01)
        await mgr.update_health("heartbeat-client")
        new_ts = mgr.connection_health["heartbeat-client"]

        assert new_ts >= old_ts


# ---------------------------------------------------------------------------
# Legacy message handler
# ---------------------------------------------------------------------------

class TestLegacyMessageHandler:
    """Tests for the handle_client_message function (legacy path)."""

    @pytest.mark.asyncio
    async def test_handle_subscribe_message(self):
        """A subscribe message should add symbols and acknowledge."""
        mgr = EnhancedConnectionManager()
        ws = _make_mock_websocket()
        await mgr.connect(ws, "legacy-sub")

        message = {"type": MessageType.SUBSCRIBE, "symbols": ["AAPL", "MSFT"]}

        with patch("backend.api.routers.websocket.manager", mgr), \
             patch("backend.api.routers.websocket.active_price_streams", {}):
            await handle_client_message(ws, "legacy-sub", message)

        ws.send_json.assert_awaited_once()
        sent_data = ws.send_json.call_args[0][0]
        assert sent_data["type"] == MessageType.SYSTEM
        assert "AAPL" in sent_data["symbols"]
        assert "MSFT" in sent_data["symbols"]

    @pytest.mark.asyncio
    async def test_handle_unsubscribe_message(self):
        """An unsubscribe message should remove symbols and acknowledge."""
        mgr = EnhancedConnectionManager()
        ws = _make_mock_websocket()
        await mgr.connect(ws, "legacy-unsub")
        await mgr.subscribe("legacy-unsub", ["AAPL"])

        message = {"type": MessageType.UNSUBSCRIBE, "symbols": ["AAPL"]}

        with patch("backend.api.routers.websocket.manager", mgr):
            await handle_client_message(ws, "legacy-unsub", message)

        ws.send_json.assert_awaited_once()
        sent_data = ws.send_json.call_args[0][0]
        assert sent_data["type"] == MessageType.SYSTEM
        assert "AAPL" in sent_data["symbols"]
        assert mgr.get_subscriptions("legacy-unsub") == set()

    @pytest.mark.asyncio
    async def test_handle_unknown_message_type(self):
        """An unknown message type should result in an error response."""
        mgr = EnhancedConnectionManager()
        ws = _make_mock_websocket()
        await mgr.connect(ws, "legacy-err")

        message = {"type": "totally_invalid"}

        with patch("backend.api.routers.websocket.manager", mgr):
            await handle_client_message(ws, "legacy-err", message)

        ws.send_json.assert_awaited_once()
        sent_data = ws.send_json.call_args[0][0]
        assert sent_data["type"] == MessageType.ERROR
        assert "Unknown" in sent_data["message"]

    @pytest.mark.asyncio
    async def test_handle_chat_message_broadcasts(self):
        """A chat message should be broadcast to all connected clients."""
        mgr = EnhancedConnectionManager()
        ws_sender = _make_mock_websocket()
        ws_receiver = _make_mock_websocket()

        await mgr.connect(ws_sender, "chat-sender")
        await mgr.connect(ws_receiver, "chat-receiver")

        message = {"type": MessageType.CHAT, "message": "hello world"}

        with patch("backend.api.routers.websocket.manager", mgr):
            await handle_client_message(ws_sender, "chat-sender", message)

        # broadcast sends to ALL clients (including sender in the legacy handler)
        assert ws_sender.send_text.await_count + ws_receiver.send_text.await_count >= 1


# ---------------------------------------------------------------------------
# Helper functions: send_alert and broadcast_news
# ---------------------------------------------------------------------------

class TestHelperFunctions:
    """Tests for send_alert and broadcast_news module-level helpers."""

    @pytest.mark.asyncio
    async def test_send_alert_delivers_to_client(self):
        """send_alert wraps the alert in the correct envelope and delivers it."""
        mgr = EnhancedConnectionManager()
        ws = _make_mock_websocket()
        await mgr.connect(ws, "alert-client")

        alert_data = {"alert_type": "price_threshold", "message": "AAPL above 200"}

        with patch("backend.api.routers.websocket.manager", mgr):
            await send_alert("alert-client", alert_data)

        ws.send_text.assert_awaited_once()
        payload = json.loads(ws.send_text.call_args[0][0])
        assert payload["type"] == MessageType.ALERT
        assert payload["alert"]["alert_type"] == "price_threshold"
        assert "timestamp" in payload

    @pytest.mark.asyncio
    async def test_broadcast_news_sends_to_all(self):
        """broadcast_news sends the news envelope to every connected client."""
        mgr = EnhancedConnectionManager()
        ws1 = _make_mock_websocket()
        ws2 = _make_mock_websocket()

        await mgr.connect(ws1, "news-1")
        await mgr.connect(ws2, "news-2")

        news_data = {"headline": "Market rally", "summary": "Stocks soar"}

        with patch("backend.api.routers.websocket.manager", mgr):
            await broadcast_news(news_data)

        assert ws1.send_text.await_count == 1
        assert ws2.send_text.await_count == 1
        payload = json.loads(ws1.send_text.call_args[0][0])
        assert payload["type"] == MessageType.NEWS
        assert payload["news"]["headline"] == "Market rally"


# ---------------------------------------------------------------------------
# WebSocket endpoint tests via TestClient
# ---------------------------------------------------------------------------

class TestMarketDataEndpoint:
    """Tests for the /api/ws/market endpoint via Starlette TestClient."""

    def test_market_endpoint_sends_data(self):
        """The market endpoint should accept the connection and send JSON data."""
        with TestClient(_ws_test_app) as client:
            with client.websocket_connect("/api/v1/ws/market") as ws:
                data = ws.receive_json()

                assert data["type"] == "market_overview"
                assert "timestamp" in data
                assert "indices" in data
                assert "SPY" in data["indices"]
                assert "QQQ" in data["indices"]
                assert "DIA" in data["indices"]
                assert "market_sentiment" in data
                assert "vix" in data
                assert "advance_decline" in data
                assert "volume" in data

    def test_market_endpoint_data_structure(self):
        """Each index in market data should have price, change, and change_percent."""
        with TestClient(_ws_test_app) as client:
            with client.websocket_connect("/api/v1/ws/market") as ws:
                data = ws.receive_json()

                for index_name in ("SPY", "QQQ", "DIA"):
                    index_data = data["indices"][index_name]
                    assert "price" in index_data
                    assert "change" in index_data
                    assert "change_percent" in index_data
                    assert isinstance(index_data["price"], (int, float))


class TestPortfolioStreamEndpoint:
    """Tests for the /api/ws/portfolio/{portfolio_id} endpoint."""

    def test_portfolio_endpoint_sends_updates(self):
        """The portfolio endpoint should send portfolio update data for the given ID."""
        with TestClient(_ws_test_app) as client:
            with client.websocket_connect("/api/v1/ws/portfolio/test-portfolio-123") as ws:
                data = ws.receive_json()

                assert data["type"] == "portfolio_update"
                assert data["portfolio_id"] == "test-portfolio-123"
                assert "timestamp" in data
                assert "total_value" in data
                assert "day_change" in data
                assert "day_change_percent" in data
                assert "positions" in data
                assert isinstance(data["positions"], list)
                assert len(data["positions"]) > 0

    def test_portfolio_endpoint_positions_structure(self):
        """Each position in the portfolio update should contain required fields."""
        with TestClient(_ws_test_app) as client:
            with client.websocket_connect("/api/v1/ws/portfolio/pf-001") as ws:
                data = ws.receive_json()

                for position in data["positions"]:
                    assert "symbol" in position
                    assert "current_price" in position
                    assert "change" in position
                    assert "value" in position


# ---------------------------------------------------------------------------
# REST endpoint: GET /api/ws/connections
# ---------------------------------------------------------------------------

class TestConnectionsEndpoint:
    """Tests for the GET /connections REST endpoint on the websocket router."""

    def test_get_active_connections_requires_admin(self):
        """The connections endpoint is admin-only (IAP-003)."""
        with TestClient(_ws_test_app) as client:
            response = client.get("/api/v1/ws/connections")

        assert response.status_code in (401, 403)


# ---------------------------------------------------------------------------
# Client stream cleanup
# ---------------------------------------------------------------------------

class TestClientStreamCleanup:
    """Tests for cleanup_client_streams helper."""

    @pytest.mark.asyncio
    async def test_cleanup_cancels_orphaned_price_streams(self):
        """When a client disconnects and no other client subscribes to the same symbol,
        the price stream task for that symbol should be cancelled."""
        mgr = EnhancedConnectionManager()
        ws = _make_mock_websocket()
        await mgr.connect(ws, "cleanup-client")
        await mgr.subscribe("cleanup-client", ["AAPL", "TSLA"])

        mock_task_aapl = MagicMock()
        mock_task_tsla = MagicMock()
        streams = {"AAPL": mock_task_aapl, "TSLA": mock_task_tsla}

        with patch("backend.api.routers.websocket.manager", mgr), \
             patch("backend.api.routers.websocket.active_price_streams", streams):
            await cleanup_client_streams("cleanup-client")

        mock_task_aapl.cancel.assert_called_once()
        mock_task_tsla.cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_preserves_shared_streams(self):
        """When another client still subscribes to a symbol, the stream should not
        be cancelled even when the disconnecting client leaves."""
        mgr = EnhancedConnectionManager()
        ws1 = _make_mock_websocket()
        ws2 = _make_mock_websocket()

        await mgr.connect(ws1, "leaving-client")
        await mgr.connect(ws2, "staying-client")
        await mgr.subscribe("leaving-client", ["AAPL"])
        await mgr.subscribe("staying-client", ["AAPL"])

        mock_task = MagicMock()
        streams = {"AAPL": mock_task}

        with patch("backend.api.routers.websocket.manager", mgr), \
             patch("backend.api.routers.websocket.active_price_streams", streams):
            await cleanup_client_streams("leaving-client")

        mock_task.cancel.assert_not_called()
