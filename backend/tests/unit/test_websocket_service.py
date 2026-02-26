"""
Unit tests for backend/services/websocket_service.py

Tests cover:
- MessageType: enum members and values
- EnhancedConnectionManager: init, connect, disconnect, send_personal_message,
  broadcast, subscribe, unsubscribe, get_subscriptions, send_to_subscribers,
  update_health, cleanup_stale_connections, initialize
- handle_client_message: subscribe, unsubscribe, chat, unknown type
- stream_price_updates: price generation, cancellation, error recovery
- send_heartbeat: heartbeat loop, cancellation, error handling
- cleanup_client_streams: stream cancellation when no subscribers remain
- send_alert: alert message formatting and delivery
- broadcast_news: news message formatting and broadcast
- generate_market_overview_data: structure and field presence
- generate_portfolio_update_data: structure, portfolio_id, positions
- cleanup_stale_connections_task: background loop, error recovery
"""

import asyncio
import json
from datetime import datetime, timezone, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.services.websocket_service import (
    MessageType,
    EnhancedConnectionManager,
    handle_client_message,
    stream_price_updates,
    send_heartbeat,
    cleanup_client_streams,
    send_alert,
    broadcast_news,
    generate_market_overview_data,
    generate_portfolio_update_data,
    cleanup_stale_connections_task,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_websocket():
    """Return an AsyncMock that quacks like a WebSocket."""
    ws = AsyncMock()
    ws.accept = AsyncMock()
    ws.send_text = AsyncMock()
    ws.send_json = AsyncMock()
    ws.close = AsyncMock()
    return ws


def _make_mock_user(user_id=1, username="testuser", role="user"):
    """Return a SimpleNamespace that quacks like a User ORM object."""
    return SimpleNamespace(id=user_id, username=username, role=role)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def manager():
    """Fresh EnhancedConnectionManager with no Redis."""
    return EnhancedConnectionManager()


@pytest.fixture
def mock_ws():
    """Fresh mock WebSocket."""
    return _make_mock_websocket()


@pytest.fixture
def mock_user():
    """Fresh mock user."""
    return _make_mock_user()


@pytest.fixture
async def connected_manager(manager, mock_ws, mock_user):
    """Manager with one connected client."""
    await manager.connect(mock_ws, "client-1", user=mock_user)
    return manager


# ===========================================================================
# MessageType enum
# ===========================================================================

class TestMessageType:

    def test_subscribe_value(self):
        assert MessageType.SUBSCRIBE == "subscribe"

    def test_unsubscribe_value(self):
        assert MessageType.UNSUBSCRIBE == "unsubscribe"

    def test_price_update_value(self):
        assert MessageType.PRICE_UPDATE == "price_update"

    def test_trade_executed_value(self):
        assert MessageType.TRADE_EXECUTED == "trade_executed"

    def test_alert_value(self):
        assert MessageType.ALERT == "alert"

    def test_news_value(self):
        assert MessageType.NEWS == "news"

    def test_chat_value(self):
        assert MessageType.CHAT == "chat"

    def test_system_value(self):
        assert MessageType.SYSTEM == "system"

    def test_heartbeat_value(self):
        assert MessageType.HEARTBEAT == "heartbeat"

    def test_error_value(self):
        assert MessageType.ERROR == "error"

    def test_is_string_enum(self):
        assert isinstance(MessageType.SUBSCRIBE, str)

    def test_all_members_present(self):
        expected = {
            "SUBSCRIBE", "UNSUBSCRIBE", "PRICE_UPDATE", "TRADE_EXECUTED",
            "ALERT", "NEWS", "CHAT", "SYSTEM", "HEARTBEAT", "ERROR",
        }
        assert set(MessageType.__members__.keys()) == expected


# ===========================================================================
# EnhancedConnectionManager -- __init__
# ===========================================================================

class TestConnectionManagerInit:

    def test_default_state(self, manager):
        assert manager.active_connections == {}
        assert manager.subscriptions == {}
        assert manager.user_sessions == {}
        assert manager.connection_health == {}
        assert manager.redis_client is None


# ===========================================================================
# EnhancedConnectionManager -- initialize
# ===========================================================================

class TestConnectionManagerInitialize:

    @pytest.mark.asyncio
    async def test_initialize_sets_redis_client(self, manager):
        mock_redis = AsyncMock()
        with patch(
            "backend.services.websocket_service.get_redis",
            new_callable=AsyncMock,
            return_value=mock_redis,
        ):
            await manager.initialize()

        assert manager.redis_client is mock_redis

    @pytest.mark.asyncio
    async def test_initialize_redis_failure_sets_none(self, manager):
        with patch(
            "backend.services.websocket_service.get_redis",
            new_callable=AsyncMock,
            side_effect=ConnectionError("Redis down"),
        ):
            await manager.initialize()

        assert manager.redis_client is None


# ===========================================================================
# EnhancedConnectionManager -- connect
# ===========================================================================

class TestConnectionManagerConnect:

    @pytest.mark.asyncio
    async def test_connect_accepts_websocket(self, manager, mock_ws):
        await manager.connect(mock_ws, "client-1")
        mock_ws.accept.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_stores_connection_info(self, manager, mock_ws):
        await manager.connect(mock_ws, "client-1")

        assert "client-1" in manager.active_connections
        info = manager.active_connections["client-1"]
        assert info["websocket"] is mock_ws
        assert info["message_count"] == 0
        assert info["user_id"] is None
        assert isinstance(info["connected_at"], datetime)

    @pytest.mark.asyncio
    async def test_connect_with_user_stores_session(self, manager, mock_ws, mock_user):
        await manager.connect(mock_ws, "client-1", user=mock_user)

        assert "client-1" in manager.user_sessions
        session = manager.user_sessions["client-1"]
        assert session["user_id"] == mock_user.id
        assert session["username"] == mock_user.username
        assert session["role"] == mock_user.role

    @pytest.mark.asyncio
    async def test_connect_without_user_no_session(self, manager, mock_ws):
        await manager.connect(mock_ws, "client-1", user=None)

        assert "client-1" not in manager.user_sessions

    @pytest.mark.asyncio
    async def test_connect_stores_user_id_in_connection(self, manager, mock_ws, mock_user):
        await manager.connect(mock_ws, "client-1", user=mock_user)

        assert manager.active_connections["client-1"]["user_id"] == mock_user.id

    @pytest.mark.asyncio
    async def test_connect_updates_health(self, manager, mock_ws):
        await manager.connect(mock_ws, "client-1")

        assert "client-1" in manager.connection_health
        assert isinstance(manager.connection_health["client-1"], datetime)

    @pytest.mark.asyncio
    async def test_connect_persists_to_redis(self, manager, mock_ws, mock_user):
        mock_redis = AsyncMock()
        manager.redis_client = mock_redis

        await manager.connect(mock_ws, "client-1", user=mock_user)

        mock_redis.hset.assert_called_once()
        call_args = mock_redis.hset.call_args
        assert call_args[0][0] == "websocket:connections"
        assert call_args[0][1] == "client-1"

    @pytest.mark.asyncio
    async def test_connect_redis_error_does_not_raise(self, manager, mock_ws):
        mock_redis = AsyncMock()
        mock_redis.hset.side_effect = Exception("Redis write failed")
        manager.redis_client = mock_redis

        # Should not raise
        await manager.connect(mock_ws, "client-1")

        assert "client-1" in manager.active_connections

    @pytest.mark.asyncio
    async def test_connect_websocket_accept_error_raises(self, manager):
        ws = AsyncMock()
        ws.accept.side_effect = RuntimeError("Connection refused")

        with pytest.raises(RuntimeError, match="Connection refused"):
            await manager.connect(ws, "client-1")


# ===========================================================================
# EnhancedConnectionManager -- disconnect
# ===========================================================================

class TestConnectionManagerDisconnect:

    @pytest.mark.asyncio
    async def test_disconnect_removes_connection(self, connected_manager, mock_ws):
        await connected_manager.disconnect(mock_ws, "client-1")

        assert "client-1" not in connected_manager.active_connections

    @pytest.mark.asyncio
    async def test_disconnect_removes_user_session(self, connected_manager, mock_ws):
        await connected_manager.disconnect(mock_ws, "client-1")

        assert "client-1" not in connected_manager.user_sessions

    @pytest.mark.asyncio
    async def test_disconnect_removes_subscriptions(self, connected_manager, mock_ws):
        connected_manager.subscriptions["client-1"] = {"AAPL", "GOOG"}

        await connected_manager.disconnect(mock_ws, "client-1")

        assert "client-1" not in connected_manager.subscriptions

    @pytest.mark.asyncio
    async def test_disconnect_removes_health(self, connected_manager, mock_ws):
        await connected_manager.disconnect(mock_ws, "client-1")

        assert "client-1" not in connected_manager.connection_health

    @pytest.mark.asyncio
    async def test_disconnect_cleans_redis(self, connected_manager, mock_ws):
        mock_redis = AsyncMock()
        connected_manager.redis_client = mock_redis

        await connected_manager.disconnect(mock_ws, "client-1")

        assert mock_redis.hdel.call_count == 2
        calls = [c[0] for c in mock_redis.hdel.call_args_list]
        assert ("websocket:connections", "client-1") in calls
        assert ("websocket:subscriptions", "client-1") in calls

    @pytest.mark.asyncio
    async def test_disconnect_redis_error_does_not_raise(self, connected_manager, mock_ws):
        mock_redis = AsyncMock()
        mock_redis.hdel.side_effect = Exception("Redis delete failed")
        connected_manager.redis_client = mock_redis

        # Should not raise
        await connected_manager.disconnect(mock_ws, "client-1")

    @pytest.mark.asyncio
    async def test_disconnect_unknown_client_does_not_raise(self, manager, mock_ws):
        # Should not raise
        await manager.disconnect(mock_ws, "nonexistent-client")

    @pytest.mark.asyncio
    async def test_disconnect_with_none_websocket(self, connected_manager):
        # Used internally when sending fails
        await connected_manager.disconnect(None, "client-1")

        assert "client-1" not in connected_manager.active_connections


# ===========================================================================
# EnhancedConnectionManager -- send_personal_message
# ===========================================================================

class TestSendPersonalMessage:

    @pytest.mark.asyncio
    async def test_sends_to_correct_client(self, connected_manager, mock_ws):
        result = await connected_manager.send_personal_message("hello", "client-1")

        assert result is True
        mock_ws.send_text.assert_called_once_with("hello")

    @pytest.mark.asyncio
    async def test_increments_message_count(self, connected_manager, mock_ws):
        await connected_manager.send_personal_message("msg1", "client-1")
        await connected_manager.send_personal_message("msg2", "client-1")

        assert connected_manager.active_connections["client-1"]["message_count"] == 2

    @pytest.mark.asyncio
    async def test_returns_false_for_unknown_client(self, manager):
        result = await manager.send_personal_message("hello", "ghost")

        assert result is False

    @pytest.mark.asyncio
    async def test_send_error_disconnects_client(self, connected_manager, mock_ws):
        mock_ws.send_text.side_effect = ConnectionError("broken pipe")

        result = await connected_manager.send_personal_message("msg", "client-1")

        assert result is False
        assert "client-1" not in connected_manager.active_connections


# ===========================================================================
# EnhancedConnectionManager -- broadcast
# ===========================================================================

class TestBroadcast:

    @pytest.mark.asyncio
    async def test_broadcast_sends_to_all(self, manager):
        ws1 = _make_mock_websocket()
        ws2 = _make_mock_websocket()
        await manager.connect(ws1, "c1")
        await manager.connect(ws2, "c2")

        count = await manager.broadcast("hello all")

        assert count == 2
        ws1.send_text.assert_called_once_with("hello all")
        ws2.send_text.assert_called_once_with("hello all")

    @pytest.mark.asyncio
    async def test_broadcast_excludes_client(self, manager):
        ws1 = _make_mock_websocket()
        ws2 = _make_mock_websocket()
        await manager.connect(ws1, "c1")
        await manager.connect(ws2, "c2")

        count = await manager.broadcast("hello", exclude="c1")

        assert count == 1
        ws1.send_text.assert_not_called()
        ws2.send_text.assert_called_once_with("hello")

    @pytest.mark.asyncio
    async def test_broadcast_filters_by_role(self, manager):
        ws_admin = _make_mock_websocket()
        ws_user = _make_mock_websocket()
        admin = _make_mock_user(user_id=1, username="admin", role="admin")
        user = _make_mock_user(user_id=2, username="regular", role="user")

        await manager.connect(ws_admin, "c-admin", user=admin)
        await manager.connect(ws_user, "c-user", user=user)

        count = await manager.broadcast("admin only", target_role="admin")

        assert count == 1
        ws_admin.send_text.assert_called_once_with("admin only")
        ws_user.send_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_broadcast_increments_message_count(self, connected_manager, mock_ws):
        await connected_manager.broadcast("msg")

        assert connected_manager.active_connections["client-1"]["message_count"] == 1

    @pytest.mark.asyncio
    async def test_broadcast_failed_client_disconnected(self, manager):
        ws_ok = _make_mock_websocket()
        ws_bad = _make_mock_websocket()
        ws_bad.send_text.side_effect = ConnectionError("broken")

        await manager.connect(ws_ok, "ok")
        await manager.connect(ws_bad, "bad")

        count = await manager.broadcast("msg")

        assert count == 1
        assert "bad" not in manager.active_connections
        assert "ok" in manager.active_connections

    @pytest.mark.asyncio
    async def test_broadcast_to_empty_returns_zero(self, manager):
        count = await manager.broadcast("hello")
        assert count == 0

    @pytest.mark.asyncio
    async def test_broadcast_role_filter_with_no_session_still_sends(self, manager):
        """Clients without user sessions still receive role-filtered broadcasts.

        The role filter only applies to clients that ARE in user_sessions.
        Anonymous clients (not in user_sessions) pass through the filter.
        """
        ws = _make_mock_websocket()
        await manager.connect(ws, "anon", user=None)

        count = await manager.broadcast("admins only", target_role="admin")

        assert count == 1
        ws.send_text.assert_called_once_with("admins only")


# ===========================================================================
# EnhancedConnectionManager -- subscribe
# ===========================================================================

class TestSubscribe:

    @pytest.mark.asyncio
    async def test_subscribe_adds_symbols(self, manager):
        result = await manager.subscribe("c1", ["AAPL", "GOOG"])

        assert result == ["AAPL", "GOOG"]
        assert manager.subscriptions["c1"] == {"AAPL", "GOOG"}

    @pytest.mark.asyncio
    async def test_subscribe_appends_to_existing(self, manager):
        await manager.subscribe("c1", ["AAPL"])
        await manager.subscribe("c1", ["GOOG"])

        assert manager.subscriptions["c1"] == {"AAPL", "GOOG"}

    @pytest.mark.asyncio
    async def test_subscribe_with_db_validates_symbols(self, manager):
        mock_db = AsyncMock()
        mock_stock = SimpleNamespace(id=1, symbol="AAPL")

        async def get_by_symbol_side_effect(symbol, session=None):
            if symbol == "AAPL":
                return mock_stock
            return None

        with patch(
            "backend.services.websocket_service.stock_repository"
        ) as mock_repo:
            mock_repo.get_by_symbol = AsyncMock(side_effect=get_by_symbol_side_effect)

            result = await manager.subscribe("c1", ["aapl", "INVALID"], db_session=mock_db)

        assert result == ["AAPL"]
        assert "AAPL" in manager.subscriptions["c1"]

    @pytest.mark.asyncio
    async def test_subscribe_without_db_skips_validation(self, manager):
        result = await manager.subscribe("c1", ["ANY_SYMBOL"])

        assert result == ["ANY_SYMBOL"]

    @pytest.mark.asyncio
    async def test_subscribe_persists_to_redis(self, manager):
        mock_redis = AsyncMock()
        manager.redis_client = mock_redis

        await manager.subscribe("c1", ["AAPL"])

        mock_redis.hset.assert_called_once()
        call_args = mock_redis.hset.call_args
        assert call_args[0][0] == "websocket:subscriptions"
        assert call_args[0][1] == "c1"

    @pytest.mark.asyncio
    async def test_subscribe_redis_error_does_not_raise(self, manager):
        mock_redis = AsyncMock()
        mock_redis.hset.side_effect = Exception("Redis write failed")
        manager.redis_client = mock_redis

        result = await manager.subscribe("c1", ["AAPL"])

        assert result == ["AAPL"]

    @pytest.mark.asyncio
    async def test_subscribe_empty_list(self, manager):
        result = await manager.subscribe("c1", [])

        assert result == []
        assert manager.subscriptions["c1"] == set()


# ===========================================================================
# EnhancedConnectionManager -- unsubscribe
# ===========================================================================

class TestUnsubscribe:

    def test_unsubscribe_removes_symbols(self, manager):
        manager.subscriptions["c1"] = {"AAPL", "GOOG", "MSFT"}
        manager.unsubscribe("c1", ["AAPL", "GOOG"])

        assert manager.subscriptions["c1"] == {"MSFT"}

    def test_unsubscribe_nonexistent_symbol_is_noop(self, manager):
        manager.subscriptions["c1"] = {"AAPL"}
        manager.unsubscribe("c1", ["NONEXISTENT"])

        assert manager.subscriptions["c1"] == {"AAPL"}

    def test_unsubscribe_unknown_client_is_noop(self, manager):
        # Should not raise
        manager.unsubscribe("ghost", ["AAPL"])


# ===========================================================================
# EnhancedConnectionManager -- get_subscriptions
# ===========================================================================

class TestGetSubscriptions:

    def test_returns_client_subscriptions(self, manager):
        manager.subscriptions["c1"] = {"AAPL", "GOOG"}

        result = manager.get_subscriptions("c1")

        assert result == {"AAPL", "GOOG"}

    def test_returns_empty_set_for_unknown_client(self, manager):
        result = manager.get_subscriptions("ghost")

        assert result == set()


# ===========================================================================
# EnhancedConnectionManager -- send_to_subscribers
# ===========================================================================

class TestSendToSubscribers:

    @pytest.mark.asyncio
    async def test_sends_to_subscribed_clients(self, manager):
        ws1 = _make_mock_websocket()
        ws2 = _make_mock_websocket()
        await manager.connect(ws1, "c1")
        await manager.connect(ws2, "c2")
        manager.subscriptions["c1"] = {"AAPL"}
        manager.subscriptions["c2"] = {"AAPL"}

        count = await manager.send_to_subscribers("AAPL", '{"price": 150}')

        assert count == 2
        ws1.send_text.assert_called_once_with('{"price": 150}')
        ws2.send_text.assert_called_once_with('{"price": 150}')

    @pytest.mark.asyncio
    async def test_skips_unsubscribed_clients(self, manager):
        ws1 = _make_mock_websocket()
        ws2 = _make_mock_websocket()
        await manager.connect(ws1, "c1")
        await manager.connect(ws2, "c2")
        manager.subscriptions["c1"] = {"AAPL"}
        manager.subscriptions["c2"] = {"GOOG"}

        count = await manager.send_to_subscribers("AAPL", '{"price": 150}')

        assert count == 1
        ws1.send_text.assert_called_once()
        ws2.send_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_excludes_specified_client(self, manager):
        ws1 = _make_mock_websocket()
        ws2 = _make_mock_websocket()
        await manager.connect(ws1, "c1")
        await manager.connect(ws2, "c2")
        manager.subscriptions["c1"] = {"AAPL"}
        manager.subscriptions["c2"] = {"AAPL"}

        count = await manager.send_to_subscribers(
            "AAPL", '{"price": 150}', exclude_client="c1"
        )

        assert count == 1
        ws1.send_text.assert_not_called()
        ws2.send_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_increments_message_count(self, manager):
        ws = _make_mock_websocket()
        await manager.connect(ws, "c1")
        manager.subscriptions["c1"] = {"AAPL"}

        await manager.send_to_subscribers("AAPL", "msg")

        assert manager.active_connections["c1"]["message_count"] == 1

    @pytest.mark.asyncio
    async def test_failed_send_disconnects_client(self, manager):
        ws_bad = _make_mock_websocket()
        ws_bad.send_text.side_effect = ConnectionError("broken")
        await manager.connect(ws_bad, "bad")
        manager.subscriptions["bad"] = {"AAPL"}

        count = await manager.send_to_subscribers("AAPL", "msg")

        assert count == 0
        assert "bad" not in manager.active_connections

    @pytest.mark.asyncio
    async def test_skips_subscribed_but_disconnected_client(self, manager):
        """A client in subscriptions but not in active_connections is skipped."""
        manager.subscriptions["ghost"] = {"AAPL"}

        count = await manager.send_to_subscribers("AAPL", "msg")

        assert count == 0

    @pytest.mark.asyncio
    async def test_no_subscribers_returns_zero(self, manager):
        count = await manager.send_to_subscribers("AAPL", "msg")
        assert count == 0


# ===========================================================================
# EnhancedConnectionManager -- update_health
# ===========================================================================

class TestUpdateHealth:

    @pytest.mark.asyncio
    async def test_updates_health_timestamp(self, manager):
        before = datetime.now(timezone.utc)
        await manager.update_health("c1")
        after = datetime.now(timezone.utc)

        assert "c1" in manager.connection_health
        assert before <= manager.connection_health["c1"] <= after


# ===========================================================================
# EnhancedConnectionManager -- cleanup_stale_connections
# ===========================================================================

class TestCleanupStaleConnections:

    @pytest.mark.asyncio
    async def test_removes_stale_connections(self, manager):
        ws = _make_mock_websocket()
        await manager.connect(ws, "stale")

        # Backdate the health timestamp
        manager.connection_health["stale"] = (
            datetime.now(timezone.utc) - timedelta(minutes=60)
        )

        cleaned = await manager.cleanup_stale_connections(max_age_minutes=30)

        assert cleaned == 1
        assert "stale" not in manager.active_connections

    @pytest.mark.asyncio
    async def test_keeps_fresh_connections(self, manager):
        ws = _make_mock_websocket()
        await manager.connect(ws, "fresh")

        cleaned = await manager.cleanup_stale_connections(max_age_minutes=30)

        assert cleaned == 0
        assert "fresh" in manager.active_connections

    @pytest.mark.asyncio
    async def test_mixed_stale_and_fresh(self, manager):
        ws_stale = _make_mock_websocket()
        ws_fresh = _make_mock_websocket()
        await manager.connect(ws_stale, "stale")
        await manager.connect(ws_fresh, "fresh")

        manager.connection_health["stale"] = (
            datetime.now(timezone.utc) - timedelta(minutes=60)
        )

        cleaned = await manager.cleanup_stale_connections(max_age_minutes=30)

        assert cleaned == 1
        assert "stale" not in manager.active_connections
        assert "fresh" in manager.active_connections

    @pytest.mark.asyncio
    async def test_no_connections_returns_zero(self, manager):
        cleaned = await manager.cleanup_stale_connections()
        assert cleaned == 0


# ===========================================================================
# handle_client_message -- subscribe
# ===========================================================================

class TestHandleClientMessageSubscribe:

    @pytest.mark.asyncio
    async def test_subscribe_calls_manager(self, manager, mock_ws):
        active_streams = {}
        message = {"type": "subscribe", "symbols": ["AAPL", "GOOG"]}

        with patch.object(manager, "subscribe", new_callable=AsyncMock) as mock_sub:
            # Patch stream_price_updates to prevent real task creation
            with patch(
                "backend.services.websocket_service.stream_price_updates",
                new_callable=AsyncMock,
            ):
                await handle_client_message(
                    mock_ws,
                    "c1",
                    message,
                    connection_manager=manager,
                    active_streams=active_streams,
                )

        mock_sub.assert_called_once_with("c1", ["AAPL", "GOOG"])

    @pytest.mark.asyncio
    async def test_subscribe_creates_stream_tasks(self, manager, mock_ws):
        active_streams = {}
        message = {"type": "subscribe", "symbols": ["AAPL"]}

        with patch(
            "backend.services.websocket_service.stream_price_updates",
            new_callable=AsyncMock,
        ) as mock_stream:
            # asyncio.create_task needs a coroutine, so use the mock coroutine
            await handle_client_message(
                mock_ws,
                "c1",
                message,
                connection_manager=manager,
                active_streams=active_streams,
            )

        assert "AAPL" in active_streams

    @pytest.mark.asyncio
    async def test_subscribe_sends_confirmation(self, manager, mock_ws):
        active_streams = {}
        message = {"type": "subscribe", "symbols": ["AAPL"]}

        with patch(
            "backend.services.websocket_service.stream_price_updates",
            new_callable=AsyncMock,
        ):
            await handle_client_message(
                mock_ws,
                "c1",
                message,
                connection_manager=manager,
                active_streams=active_streams,
            )

        mock_ws.send_json.assert_called_once()
        sent = mock_ws.send_json.call_args[0][0]
        assert sent["type"] == MessageType.SYSTEM
        assert "AAPL" in sent["symbols"]

    @pytest.mark.asyncio
    async def test_subscribe_does_not_duplicate_streams(self, manager, mock_ws):
        mock_task = MagicMock()
        active_streams = {"AAPL": mock_task}
        message = {"type": "subscribe", "symbols": ["AAPL"]}

        await handle_client_message(
            mock_ws,
            "c1",
            message,
            connection_manager=manager,
            active_streams=active_streams,
        )

        # Should still be the original task, not replaced
        assert active_streams["AAPL"] is mock_task


# ===========================================================================
# handle_client_message -- unsubscribe
# ===========================================================================

class TestHandleClientMessageUnsubscribe:

    @pytest.mark.asyncio
    async def test_unsubscribe_calls_manager(self, manager, mock_ws):
        active_streams = {}
        message = {"type": "unsubscribe", "symbols": ["AAPL"]}

        with patch.object(manager, "unsubscribe") as mock_unsub:
            await handle_client_message(
                mock_ws,
                "c1",
                message,
                connection_manager=manager,
                active_streams=active_streams,
            )

        mock_unsub.assert_called_once_with("c1", ["AAPL"])

    @pytest.mark.asyncio
    async def test_unsubscribe_sends_confirmation(self, manager, mock_ws):
        active_streams = {}
        message = {"type": "unsubscribe", "symbols": ["AAPL"]}

        await handle_client_message(
            mock_ws,
            "c1",
            message,
            connection_manager=manager,
            active_streams=active_streams,
        )

        mock_ws.send_json.assert_called_once()
        sent = mock_ws.send_json.call_args[0][0]
        assert sent["type"] == MessageType.SYSTEM
        assert "AAPL" in sent["symbols"]


# ===========================================================================
# handle_client_message -- chat
# ===========================================================================

class TestHandleClientMessageChat:

    @pytest.mark.asyncio
    async def test_chat_broadcasts_message(self, manager, mock_ws):
        active_streams = {}
        message = {"type": "chat", "message": "Hello everyone!"}

        with patch.object(
            manager, "broadcast", new_callable=AsyncMock
        ) as mock_broadcast:
            await handle_client_message(
                mock_ws,
                "c1",
                message,
                connection_manager=manager,
                active_streams=active_streams,
            )

        mock_broadcast.assert_called_once()
        broadcast_payload = json.loads(mock_broadcast.call_args[0][0])
        assert broadcast_payload["type"] == MessageType.CHAT
        assert broadcast_payload["from"] == "c1"
        assert broadcast_payload["message"] == "Hello everyone!"

    @pytest.mark.asyncio
    async def test_chat_with_empty_message(self, manager, mock_ws):
        active_streams = {}
        message = {"type": "chat"}

        with patch.object(
            manager, "broadcast", new_callable=AsyncMock
        ) as mock_broadcast:
            await handle_client_message(
                mock_ws,
                "c1",
                message,
                connection_manager=manager,
                active_streams=active_streams,
            )

        broadcast_payload = json.loads(mock_broadcast.call_args[0][0])
        assert broadcast_payload["message"] == ""


# ===========================================================================
# handle_client_message -- unknown type
# ===========================================================================

class TestHandleClientMessageUnknown:

    @pytest.mark.asyncio
    async def test_unknown_type_sends_error(self, manager, mock_ws):
        active_streams = {}
        message = {"type": "invalid_type"}

        await handle_client_message(
            mock_ws,
            "c1",
            message,
            connection_manager=manager,
            active_streams=active_streams,
        )

        mock_ws.send_json.assert_called_once()
        sent = mock_ws.send_json.call_args[0][0]
        assert sent["type"] == MessageType.ERROR
        assert "invalid_type" in sent["message"]

    @pytest.mark.asyncio
    async def test_none_type_sends_error(self, manager, mock_ws):
        active_streams = {}
        message = {}

        await handle_client_message(
            mock_ws,
            "c1",
            message,
            connection_manager=manager,
            active_streams=active_streams,
        )

        mock_ws.send_json.assert_called_once()
        sent = mock_ws.send_json.call_args[0][0]
        assert sent["type"] == MessageType.ERROR


# ===========================================================================
# stream_price_updates
# ===========================================================================

class TestStreamPriceUpdates:

    @pytest.mark.asyncio
    async def test_cancellation_exits_cleanly(self, manager):
        """CancelledError should break the loop without error."""
        with patch(
            "backend.services.websocket_service.asyncio.sleep",
            new_callable=AsyncMock,
            side_effect=asyncio.CancelledError,
        ):
            # Should not raise
            await stream_price_updates("AAPL", manager)

    @pytest.mark.asyncio
    async def test_sends_to_subscribers(self, manager):
        call_count = 0

        async def mock_sleep(duration):
            nonlocal call_count
            call_count += 1
            if call_count >= 1:
                raise asyncio.CancelledError

        with patch.object(
            manager, "send_to_subscribers", new_callable=AsyncMock
        ) as mock_send:
            with patch(
                "backend.services.websocket_service.asyncio.sleep",
                side_effect=mock_sleep,
            ):
                await stream_price_updates("AAPL", manager)

        mock_send.assert_called_once()
        call_args = mock_send.call_args
        assert call_args[0][0] == "AAPL"
        payload = json.loads(call_args[0][1])
        assert payload["type"] == MessageType.PRICE_UPDATE
        assert payload["symbol"] == "AAPL"
        assert "price" in payload
        assert "volume" in payload
        assert "bid" in payload
        assert "ask" in payload

    @pytest.mark.asyncio
    async def test_error_recovery_continues_loop(self, manager):
        """Non-cancellation errors should sleep 5s and continue."""
        call_count = 0

        async def mock_sleep(duration):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # This is the recovery sleep (5s) after the error
                assert duration == 5
            if call_count >= 2:
                raise asyncio.CancelledError

        with patch.object(
            manager,
            "send_to_subscribers",
            new_callable=AsyncMock,
            side_effect=[RuntimeError("oops"), AsyncMock()],
        ):
            with patch(
                "backend.services.websocket_service.asyncio.sleep",
                side_effect=mock_sleep,
            ):
                await stream_price_updates("AAPL", manager)


# ===========================================================================
# send_heartbeat
# ===========================================================================

class TestSendHeartbeat:

    @pytest.mark.asyncio
    async def test_cancellation_exits_cleanly(self, mock_ws):
        with patch(
            "backend.services.websocket_service.asyncio.sleep",
            new_callable=AsyncMock,
            side_effect=asyncio.CancelledError,
        ):
            await send_heartbeat(mock_ws, "c1")

    @pytest.mark.asyncio
    async def test_sends_heartbeat_message(self, mock_ws):
        call_count = 0

        async def mock_sleep(duration):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                raise asyncio.CancelledError

        with patch(
            "backend.services.websocket_service.asyncio.sleep",
            side_effect=mock_sleep,
        ):
            await send_heartbeat(mock_ws, "c1")

        mock_ws.send_json.assert_called_once()
        sent = mock_ws.send_json.call_args[0][0]
        assert sent["type"] == MessageType.HEARTBEAT
        assert "timestamp" in sent
        assert "server_time" in sent

    @pytest.mark.asyncio
    async def test_websocket_error_breaks_loop(self, mock_ws):
        call_count = 0

        async def mock_sleep(duration):
            nonlocal call_count
            call_count += 1

        mock_ws.send_json.side_effect = ConnectionError("broken")

        with patch(
            "backend.services.websocket_service.asyncio.sleep",
            side_effect=mock_sleep,
        ):
            await send_heartbeat(mock_ws, "c1")

        # Should have only tried once then broken
        assert mock_ws.send_json.call_count == 1


# ===========================================================================
# cleanup_client_streams
# ===========================================================================

class TestCleanupClientStreams:

    @pytest.mark.asyncio
    async def test_cancels_orphaned_streams(self, manager):
        manager.subscriptions["c1"] = {"AAPL", "GOOG"}

        mock_task_aapl = MagicMock()
        mock_task_goog = MagicMock()
        active_streams = {"AAPL": mock_task_aapl, "GOOG": mock_task_goog}

        await cleanup_client_streams(
            "c1",
            connection_manager=manager,
            active_streams=active_streams,
        )

        mock_task_aapl.cancel.assert_called_once()
        mock_task_goog.cancel.assert_called_once()
        assert "AAPL" not in active_streams
        assert "GOOG" not in active_streams

    @pytest.mark.asyncio
    async def test_keeps_streams_with_other_subscribers(self, manager):
        manager.subscriptions["c1"] = {"AAPL"}
        manager.subscriptions["c2"] = {"AAPL"}

        mock_task = MagicMock()
        active_streams = {"AAPL": mock_task}

        await cleanup_client_streams(
            "c1",
            connection_manager=manager,
            active_streams=active_streams,
        )

        mock_task.cancel.assert_not_called()
        assert "AAPL" in active_streams

    @pytest.mark.asyncio
    async def test_no_subscriptions_is_noop(self, manager):
        active_streams = {"AAPL": MagicMock()}

        await cleanup_client_streams(
            "ghost",
            connection_manager=manager,
            active_streams=active_streams,
        )

        assert "AAPL" in active_streams

    @pytest.mark.asyncio
    async def test_stream_not_in_active_streams(self, manager):
        """Subscribed to a symbol but no stream task exists -- should not raise."""
        manager.subscriptions["c1"] = {"AAPL"}
        active_streams = {}

        await cleanup_client_streams(
            "c1",
            connection_manager=manager,
            active_streams=active_streams,
        )


# ===========================================================================
# send_alert
# ===========================================================================

class TestSendAlert:

    @pytest.mark.asyncio
    async def test_sends_alert_to_client(self, manager):
        with patch.object(
            manager, "send_personal_message", new_callable=AsyncMock
        ) as mock_send:
            alert_data = {"level": "critical", "message": "Price spike!"}
            await send_alert(
                "c1", alert_data, connection_manager=manager
            )

        mock_send.assert_called_once()
        payload = json.loads(mock_send.call_args[0][0])
        assert payload["type"] == MessageType.ALERT
        assert payload["alert"]["level"] == "critical"
        assert payload["alert"]["message"] == "Price spike!"
        assert "timestamp" in payload


# ===========================================================================
# broadcast_news
# ===========================================================================

class TestBroadcastNews:

    @pytest.mark.asyncio
    async def test_broadcasts_news(self, manager):
        with patch.object(
            manager, "broadcast", new_callable=AsyncMock
        ) as mock_broadcast:
            news_data = {"headline": "Market hits all-time high"}
            await broadcast_news(news_data, connection_manager=manager)

        mock_broadcast.assert_called_once()
        payload = json.loads(mock_broadcast.call_args[0][0])
        assert payload["type"] == MessageType.NEWS
        assert payload["news"]["headline"] == "Market hits all-time high"
        assert "timestamp" in payload


# ===========================================================================
# generate_market_overview_data
# ===========================================================================

class TestGenerateMarketOverviewData:

    def test_returns_dict(self):
        result = generate_market_overview_data()
        assert isinstance(result, dict)

    def test_has_type_field(self):
        result = generate_market_overview_data()
        assert result["type"] == "market_overview"

    def test_has_timestamp(self):
        result = generate_market_overview_data()
        assert "timestamp" in result

    def test_has_all_indices(self):
        result = generate_market_overview_data()
        assert "SPY" in result["indices"]
        assert "QQQ" in result["indices"]
        assert "DIA" in result["indices"]

    def test_index_has_price_and_change(self):
        result = generate_market_overview_data()
        for symbol in ["SPY", "QQQ", "DIA"]:
            index_data = result["indices"][symbol]
            assert "price" in index_data
            assert "change" in index_data
            assert "change_percent" in index_data

    def test_has_market_sentiment(self):
        result = generate_market_overview_data()
        assert "market_sentiment" in result
        assert -1 <= result["market_sentiment"] <= 1

    def test_has_vix(self):
        result = generate_market_overview_data()
        assert "vix" in result
        assert 12 <= result["vix"] <= 30

    def test_has_advance_decline(self):
        result = generate_market_overview_data()
        ad = result["advance_decline"]
        assert "advancing" in ad
        assert "declining" in ad
        assert "unchanged" in ad

    def test_has_volume(self):
        result = generate_market_overview_data()
        vol = result["volume"]
        assert "total" in vol
        assert "up_volume" in vol
        assert "down_volume" in vol


# ===========================================================================
# generate_portfolio_update_data
# ===========================================================================

class TestGeneratePortfolioUpdateData:

    def test_returns_dict(self):
        result = generate_portfolio_update_data("port-1")
        assert isinstance(result, dict)

    def test_has_type_field(self):
        result = generate_portfolio_update_data("port-1")
        assert result["type"] == "portfolio_update"

    def test_has_correct_portfolio_id(self):
        result = generate_portfolio_update_data("my-portfolio")
        assert result["portfolio_id"] == "my-portfolio"

    def test_has_timestamp(self):
        result = generate_portfolio_update_data("port-1")
        assert "timestamp" in result

    def test_has_financial_fields(self):
        result = generate_portfolio_update_data("port-1")
        assert "total_value" in result
        assert "day_change" in result
        assert "day_change_percent" in result

    def test_has_positions(self):
        result = generate_portfolio_update_data("port-1")
        positions = result["positions"]
        assert len(positions) == 2
        symbols = {p["symbol"] for p in positions}
        assert "AAPL" in symbols
        assert "GOOGL" in symbols

    def test_position_has_fields(self):
        result = generate_portfolio_update_data("port-1")
        for pos in result["positions"]:
            assert "symbol" in pos
            assert "current_price" in pos
            assert "change" in pos
            assert "value" in pos

    def test_has_alerts_list(self):
        result = generate_portfolio_update_data("port-1")
        assert isinstance(result["alerts"], list)


# ===========================================================================
# cleanup_stale_connections_task
# ===========================================================================

class TestCleanupStaleConnectionsTask:

    @pytest.mark.asyncio
    async def test_calls_cleanup_then_sleeps(self, manager):
        call_count = 0

        async def mock_sleep(duration):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                assert duration == 300
            raise asyncio.CancelledError

        with patch.object(
            manager,
            "cleanup_stale_connections",
            new_callable=AsyncMock,
            return_value=0,
        ) as mock_cleanup:
            with patch(
                "backend.services.websocket_service.asyncio.sleep",
                side_effect=mock_sleep,
            ):
                try:
                    await cleanup_stale_connections_task(manager)
                except asyncio.CancelledError:
                    pass

        mock_cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_error_recovery_waits_60s(self, manager):
        call_count = 0
        sleep_values = []

        async def mock_sleep(duration):
            nonlocal call_count
            sleep_values.append(duration)
            call_count += 1
            if call_count >= 2:
                raise asyncio.CancelledError

        with patch.object(
            manager,
            "cleanup_stale_connections",
            new_callable=AsyncMock,
            side_effect=[RuntimeError("oops"), AsyncMock(return_value=0)],
        ):
            with patch(
                "backend.services.websocket_service.asyncio.sleep",
                side_effect=mock_sleep,
            ):
                try:
                    await cleanup_stale_connections_task(manager)
                except asyncio.CancelledError:
                    pass

        # First sleep should be 60s (error recovery), second 300s (normal)
        assert sleep_values[0] == 60
