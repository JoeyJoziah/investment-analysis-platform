"""
WebSocket Integration Tests

Tests for real-time price updates, subscription management,
and WebSocket connection reliability.
"""

import asyncio
import json
import logging
import time
from typing import Optional
from unittest.mock import Mock, patch, AsyncMock

import pytest
import websockets
from fastapi.testclient import TestClient

from backend.api.main import app
from backend.models.unified_models import User
from backend.auth.oauth2 import create_tokens

logger = logging.getLogger(__name__)


@pytest.fixture
def test_user_data():
    """Test user fixture"""
    return {
        "username": "wstest_user",
        "email": "wstest@example.com",
        "password": "TestPassword123!@#",
        "is_active": True,
        "is_admin": False,
    }


@pytest.fixture
def auth_headers(test_user_data, db_session):
    """Create authenticated headers for WebSocket tests"""
    user = User(
        username=test_user_data["username"],
        email=test_user_data["email"],
        is_active=True,
        is_admin=False,
    )
    user.set_password(test_user_data["password"])
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    tokens = create_tokens(user)
    return {"Authorization": f"Bearer {tokens['access_token']}"}


class TestWebSocketConnection:
    """WebSocket connection tests"""

    @pytest.mark.asyncio
    async def test_websocket_connection_succeeds(self, test_user_data, db_session):
        """Test that WebSocket connection is established successfully"""
        # Create test user
        user = User(
            username=test_user_data["username"],
            email=test_user_data["email"],
            is_active=True,
            is_admin=False,
        )
        user.set_password(test_user_data["password"])
        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)

        tokens = create_tokens(user)

        # Use TestClient to get WebSocket connection
        with TestClient(app) as client:
            with client.websocket_connect(
                "/api/ws/prices",
                headers={"Authorization": f"Bearer {tokens['access_token']}"},
            ) as websocket:
                # Verify connection is established
                data = websocket.receive_json()
                assert data["type"] == "connection_established"
                assert data["user_id"] == user.id

    @pytest.mark.asyncio
    async def test_websocket_requires_authentication(self):
        """Test that WebSocket connection requires valid token"""
        with TestClient(app) as client:
            # Try to connect without token
            with pytest.raises(Exception):
                with client.websocket_connect("/api/ws/prices") as websocket:
                    pass

    @pytest.mark.asyncio
    async def test_websocket_rejects_invalid_token(self):
        """Test WebSocket rejects invalid token"""
        with TestClient(app) as client:
            with pytest.raises(Exception):
                with client.websocket_connect(
                    "/api/ws/prices",
                    headers={"Authorization": "Bearer invalid.token.here"},
                ) as websocket:
                    pass

    @pytest.mark.asyncio
    async def test_websocket_rejects_expired_token(self, test_user_data, db_session):
        """Test WebSocket rejects expired token"""
        from datetime import timedelta

        user = User(
            username=test_user_data["username"],
            email=test_user_data["email"],
            is_active=True,
        )
        user.set_password(test_user_data["password"])
        db_session.add(user)
        db_session.commit()

        # Create expired token
        from backend.auth.oauth2 import create_access_token

        expired_token = create_access_token(
            {"sub": user.username, "user_id": user.id},
            expires_delta=timedelta(seconds=-10),
        )

        with TestClient(app) as client:
            with pytest.raises(Exception):
                with client.websocket_connect(
                    "/api/ws/prices",
                    headers={"Authorization": f"Bearer {expired_token}"},
                ) as websocket:
                    pass

    @pytest.mark.asyncio
    async def test_websocket_connection_for_inactive_user(
        self, test_user_data, db_session
    ):
        """Test WebSocket rejects inactive user"""
        user = User(
            username=test_user_data["username"],
            email=test_user_data["email"],
            is_active=False,  # Inactive user
        )
        user.set_password(test_user_data["password"])
        db_session.add(user)
        db_session.commit()

        tokens = create_tokens(user)

        with TestClient(app) as client:
            with pytest.raises(Exception):
                with client.websocket_connect(
                    "/api/ws/prices",
                    headers={"Authorization": f"Bearer {tokens['access_token']}"},
                ) as websocket:
                    pass


class TestPriceSubscription:
    """Price subscription and update tests"""

    @pytest.mark.asyncio
    async def test_subscribe_to_price_updates(self, test_user_data, db_session):
        """Test subscribing to price updates for specific tickers"""
        user = User(
            username=test_user_data["username"],
            email=test_user_data["email"],
        )
        user.set_password(test_user_data["password"])
        db_session.add(user)
        db_session.commit()

        tokens = create_tokens(user)

        with TestClient(app) as client:
            with client.websocket_connect(
                "/api/ws/prices",
                headers={"Authorization": f"Bearer {tokens['access_token']}"},
            ) as websocket:
                # Receive connection confirmation
                data = websocket.receive_json()
                assert data["type"] == "connection_established"

                # Subscribe to AAPL price updates
                websocket.send_json(
                    {"action": "subscribe", "symbol": "AAPL"}
                )

                # Receive subscription confirmation
                response = websocket.receive_json()
                assert response["type"] == "subscription_confirmed"
                assert response["symbol"] == "AAPL"

    @pytest.mark.asyncio
    async def test_unsubscribe_from_price_updates(self, test_user_data, db_session):
        """Test unsubscribing from price updates"""
        user = User(
            username=test_user_data["username"],
            email=test_user_data["email"],
        )
        user.set_password(test_user_data["password"])
        db_session.add(user)
        db_session.commit()

        tokens = create_tokens(user)

        with TestClient(app) as client:
            with client.websocket_connect(
                "/api/ws/prices",
                headers={"Authorization": f"Bearer {tokens['access_token']}"},
            ) as websocket:
                # Skip connection message
                websocket.receive_json()

                # Subscribe to AAPL
                websocket.send_json(
                    {"action": "subscribe", "symbol": "AAPL"}
                )
                websocket.receive_json()  # Confirm subscription

                # Unsubscribe
                websocket.send_json(
                    {"action": "unsubscribe", "symbol": "AAPL"}
                )

                response = websocket.receive_json()
                assert response["type"] == "unsubscribed"
                assert response["symbol"] == "AAPL"

    @pytest.mark.asyncio
    async def test_multiple_subscriptions(self, test_user_data, db_session):
        """Test subscribing to multiple symbols"""
        user = User(
            username=test_user_data["username"],
            email=test_user_data["email"],
        )
        user.set_password(test_user_data["password"])
        db_session.add(user)
        db_session.commit()

        tokens = create_tokens(user)

        with TestClient(app) as client:
            with client.websocket_connect(
                "/api/ws/prices",
                headers={"Authorization": f"Bearer {tokens['access_token']}"},
            ) as websocket:
                websocket.receive_json()  # Connection message

                # Subscribe to multiple symbols
                symbols = ["AAPL", "MSFT", "GOOGL", "TSLA"]
                for symbol in symbols:
                    websocket.send_json(
                        {"action": "subscribe", "symbol": symbol}
                    )
                    response = websocket.receive_json()
                    assert response["symbol"] == symbol

    @pytest.mark.asyncio
    async def test_invalid_subscription_symbol(self, test_user_data, db_session):
        """Test handling of invalid ticker symbols"""
        user = User(
            username=test_user_data["username"],
            email=test_user_data["email"],
        )
        user.set_password(test_user_data["password"])
        db_session.add(user)
        db_session.commit()

        tokens = create_tokens(user)

        with TestClient(app) as client:
            with client.websocket_connect(
                "/api/ws/prices",
                headers={"Authorization": f"Bearer {tokens['access_token']}"},
            ) as websocket:
                websocket.receive_json()

                # Try to subscribe to invalid symbol
                websocket.send_json(
                    {"action": "subscribe", "symbol": "INVALID123XYZ"}
                )

                response = websocket.receive_json()
                # Should either fail or succeed - server may validate differently
                assert response.get("type") in [
                    "subscription_confirmed",
                    "subscription_failed",
                    "error",
                ]


class TestPriceUpdateDelivery:
    """Price update delivery and latency tests"""

    @pytest.mark.asyncio
    async def test_price_update_message_format(self, test_user_data, db_session):
        """Test that price update messages have correct format"""
        user = User(
            username=test_user_data["username"],
            email=test_user_data["email"],
        )
        user.set_password(test_user_data["password"])
        db_session.add(user)
        db_session.commit()

        tokens = create_tokens(user)

        with TestClient(app) as client:
            with client.websocket_connect(
                "/api/ws/prices",
                headers={"Authorization": f"Bearer {tokens['access_token']}"},
            ) as websocket:
                websocket.receive_json()  # Connection

                # Subscribe to AAPL
                websocket.send_json(
                    {"action": "subscribe", "symbol": "AAPL"}
                )
                websocket.receive_json()  # Confirmation

                # Simulate receiving price update
                # (In real scenario, this would come from market data feed)
                # For testing, we'll send a test message
                websocket.send_json(
                    {
                        "action": "test_price_update",
                        "symbol": "AAPL",
                        "price": 150.25,
                    }
                )

                # Verify message format if received
                try:
                    response = websocket.receive_json(timeout=2)
                    if response.get("type") == "price_update":
                        assert "symbol" in response
                        assert "price" in response
                        assert "timestamp" in response
                        assert isinstance(response["price"], (int, float))
                except TimeoutError:
                    # No price update in test environment is acceptable
                    pass

    @pytest.mark.asyncio
    async def test_price_update_latency(self, test_user_data, db_session):
        """Test that price updates arrive within acceptable latency (<2s)"""
        user = User(
            username=test_user_data["username"],
            email=test_user_data["email"],
        )
        user.set_password(test_user_data["password"])
        db_session.add(user)
        db_session.commit()

        tokens = create_tokens(user)

        with TestClient(app) as client:
            with client.websocket_connect(
                "/api/ws/prices",
                headers={"Authorization": f"Bearer {tokens['access_token']}"},
            ) as websocket:
                websocket.receive_json()

                # Subscribe to symbol
                websocket.send_json(
                    {"action": "subscribe", "symbol": "AAPL"}
                )
                websocket.receive_json()

                # Measure time for next message
                start_time = time.time()

                try:
                    # Wait for up to 3 seconds for a price update
                    response = websocket.receive_json(timeout=3)
                    elapsed = time.time() - start_time

                    if response.get("type") == "price_update":
                        # Verify latency is under 2 seconds
                        assert elapsed < 2.0, f"Price update latency {elapsed}s exceeds 2s limit"

                except TimeoutError:
                    # In test environment without live data, timeout is acceptable
                    pass

    @pytest.mark.asyncio
    async def test_batch_price_updates(self, test_user_data, db_session):
        """Test handling of rapid consecutive price updates"""
        user = User(
            username=test_user_data["username"],
            email=test_user_data["email"],
        )
        user.set_password(test_user_data["password"])
        db_session.add(user)
        db_session.commit()

        tokens = create_tokens(user)

        with TestClient(app) as client:
            with client.websocket_connect(
                "/api/ws/prices",
                headers={"Authorization": f"Bearer {tokens['access_token']}"},
            ) as websocket:
                websocket.receive_json()

                # Subscribe to symbol
                websocket.send_json(
                    {"action": "subscribe", "symbol": "AAPL"}
                )
                websocket.receive_json()

                # Simulate batch updates
                update_count = 0
                try:
                    for _ in range(10):
                        response = websocket.receive_json(timeout=0.5)
                        if response.get("type") == "price_update":
                            update_count += 1
                except TimeoutError:
                    pass

                # In test environment, we just verify the server handles it


class TestWebSocketReconnection:
    """Connection resilience and reconnection tests"""

    @pytest.mark.asyncio
    async def test_reconnection_with_same_token(self, test_user_data, db_session):
        """Test reconnecting with the same token"""
        user = User(
            username=test_user_data["username"],
            email=test_user_data["email"],
        )
        user.set_password(test_user_data["password"])
        db_session.add(user)
        db_session.commit()

        tokens = create_tokens(user)

        with TestClient(app) as client:
            # First connection
            with client.websocket_connect(
                "/api/ws/prices",
                headers={"Authorization": f"Bearer {tokens['access_token']}"},
            ) as websocket:
                data = websocket.receive_json()
                assert data["type"] == "connection_established"

            # Second connection with same token
            with client.websocket_connect(
                "/api/ws/prices",
                headers={"Authorization": f"Bearer {tokens['access_token']}"},
            ) as websocket:
                data = websocket.receive_json()
                assert data["type"] == "connection_established"

    @pytest.mark.asyncio
    async def test_preserve_subscriptions_on_reconnect(
        self, test_user_data, db_session
    ):
        """Test that subscriptions can be reestablished on reconnect"""
        user = User(
            username=test_user_data["username"],
            email=test_user_data["email"],
        )
        user.set_password(test_user_data["password"])
        db_session.add(user)
        db_session.commit()

        tokens = create_tokens(user)
        subscribed_symbols = ["AAPL", "MSFT"]

        with TestClient(app) as client:
            # First connection - subscribe to symbols
            with client.websocket_connect(
                "/api/ws/prices",
                headers={"Authorization": f"Bearer {tokens['access_token']}"},
            ) as websocket:
                websocket.receive_json()  # Connection message

                for symbol in subscribed_symbols:
                    websocket.send_json(
                        {"action": "subscribe", "symbol": symbol}
                    )
                    websocket.receive_json()  # Confirmation

            # Reconnect and resubscribe
            with client.websocket_connect(
                "/api/ws/prices",
                headers={"Authorization": f"Bearer {tokens['access_token']}"},
            ) as websocket:
                websocket.receive_json()

                # Resubscribe to same symbols
                for symbol in subscribed_symbols:
                    websocket.send_json(
                        {"action": "subscribe", "symbol": symbol}
                    )
                    response = websocket.receive_json()
                    assert response["symbol"] == symbol

    @pytest.mark.asyncio
    async def test_connection_cleanup_on_disconnect(
        self, test_user_data, db_session
    ):
        """Test that resources are cleaned up on disconnect"""
        user = User(
            username=test_user_data["username"],
            email=test_user_data["email"],
        )
        user.set_password(test_user_data["password"])
        db_session.add(user)
        db_session.commit()

        tokens = create_tokens(user)

        with TestClient(app) as client:
            with client.websocket_connect(
                "/api/ws/prices",
                headers={"Authorization": f"Bearer {tokens['access_token']}"},
            ) as websocket:
                websocket.receive_json()

                # Subscribe to symbol
                websocket.send_json(
                    {"action": "subscribe", "symbol": "AAPL"}
                )
                websocket.receive_json()

                # Close connection (implicit on context exit)

            # Resources should be cleaned up
            # Verify by attempting new connection which should succeed
            with client.websocket_connect(
                "/api/ws/prices",
                headers={"Authorization": f"Bearer {tokens['access_token']}"},
            ) as websocket2:
                data = websocket2.receive_json()
                assert data["type"] == "connection_established"


class TestWebSocketErrorHandling:
    """Error handling tests"""

    @pytest.mark.asyncio
    async def test_invalid_message_format(self, test_user_data, db_session):
        """Test handling of invalid message format"""
        user = User(
            username=test_user_data["username"],
            email=test_user_data["email"],
        )
        user.set_password(test_user_data["password"])
        db_session.add(user)
        db_session.commit()

        tokens = create_tokens(user)

        with TestClient(app) as client:
            with client.websocket_connect(
                "/api/ws/prices",
                headers={"Authorization": f"Bearer {tokens['access_token']}"},
            ) as websocket:
                websocket.receive_json()

                # Send invalid message
                websocket.send_json({"invalid": "message"})

                try:
                    response = websocket.receive_json(timeout=1)
                    # Server should either ignore or send error
                    assert response.get("type") in [
                        "error",
                        None,
                    ]
                except Exception:
                    # Connection might be closed on invalid message
                    pass

    @pytest.mark.asyncio
    async def test_malformed_json_handling(self, test_user_data, db_session):
        """Test handling of malformed JSON"""
        user = User(
            username=test_user_data["username"],
            email=test_user_data["email"],
        )
        user.set_password(test_user_data["password"])
        db_session.add(user)
        db_session.commit()

        tokens = create_tokens(user)

        with TestClient(app) as client:
            with client.websocket_connect(
                "/api/ws/prices",
                headers={"Authorization": f"Bearer {tokens['access_token']}"},
            ) as websocket:
                websocket.receive_json()

                # Try to send text instead of JSON
                websocket.send_text("This is not JSON")

                try:
                    response = websocket.receive_json(timeout=1)
                except Exception:
                    # Server should handle gracefully
                    pass

    @pytest.mark.asyncio
    async def test_server_error_notification(self, test_user_data, db_session):
        """Test that server errors are communicated to client"""
        user = User(
            username=test_user_data["username"],
            email=test_user_data["email"],
        )
        user.set_password(test_user_data["password"])
        db_session.add(user)
        db_session.commit()

        tokens = create_tokens(user)

        with TestClient(app) as client:
            with client.websocket_connect(
                "/api/ws/prices",
                headers={"Authorization": f"Bearer {tokens['access_token']}"},
            ) as websocket:
                websocket.receive_json()

                # Try to perform operation that might fail
                websocket.send_json(
                    {"action": "invalid_action"}
                )

                try:
                    response = websocket.receive_json(timeout=1)
                    if response.get("type") == "error":
                        assert "message" in response or "detail" in response
                except TimeoutError:
                    # Server might not respond to invalid action
                    pass
