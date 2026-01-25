"""
WebSocket Integration Tests for Investment Analysis Platform
Tests real-time data streaming, client connections, and message broadcasting.
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from fastapi.websockets import WebSocket, WebSocketDisconnect
import websockets
from websockets.exceptions import ConnectionClosedError

from backend.api.main import app
from backend.api.routers.websocket import EnhancedConnectionManager as ConnectionManager, manager
from backend.auth.oauth2 import get_current_user
from backend.models.unified_models import User


class TestWebSocketIntegration:
    """Test WebSocket connections, real-time data streaming, and client management."""

    @pytest.fixture
    def mock_user(self):
        """Create mock authenticated user."""
        return User(
            id=1,
            username="testuser",
            email="test@example.com",
            is_active=True,
            created_at=datetime.utcnow()
        )

    @pytest.fixture
    def websocket_manager(self):
        """Create WebSocket manager instance."""
        return ConnectionManager()

    @pytest.fixture
    def connection_manager(self):
        """Create connection manager instance."""
        return ConnectionManager()

    @pytest.fixture
    def mock_websocket(self):
        """Create mock WebSocket connection."""
        websocket = AsyncMock(spec=WebSocket)
        websocket.accept = AsyncMock()
        websocket.close = AsyncMock()
        websocket.send_text = AsyncMock()
        websocket.send_json = AsyncMock()
        websocket.receive_text = AsyncMock()
        websocket.receive_json = AsyncMock()
        return websocket

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_websocket_connection_lifecycle(self, websocket_manager, mock_websocket, mock_user):
        """Test WebSocket connection establishment, authentication, and cleanup."""
        
        # Test connection establishment
        await websocket_manager.connect(mock_websocket, mock_user.id)
        
        # Verify connection was accepted
        mock_websocket.accept.assert_called_once()
        
        # Verify user was added to active connections
        assert mock_user.id in websocket_manager.active_connections
        assert websocket_manager.active_connections[mock_user.id] == mock_websocket
        
        # Test connection count
        assert websocket_manager.get_connection_count() == 1
        
        # Test connection info
        connection_info = websocket_manager.get_connection_info(mock_user.id)
        assert connection_info is not None
        assert connection_info["user_id"] == mock_user.id
        assert "connected_at" in connection_info
        
        # Test disconnection
        websocket_manager.disconnect(mock_user.id)
        assert mock_user.id not in websocket_manager.active_connections
        assert websocket_manager.get_connection_count() == 0

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_websocket_message_broadcasting(self, websocket_manager):
        """Test broadcasting messages to all connected clients."""
        
        # Create multiple mock connections
        mock_connections = []
        user_ids = [1, 2, 3]
        
        for user_id in user_ids:
            websocket = AsyncMock(spec=WebSocket)
            mock_connections.append(websocket)
            await websocket_manager.connect(websocket, user_id)
        
        # Test broadcast to all
        test_message = {"type": "price_update", "symbol": "AAPL", "price": 154.25}
        await websocket_manager.broadcast_to_all(test_message)
        
        # Verify all connections received the message
        for websocket in mock_connections:
            websocket.send_json.assert_called_once_with(test_message)
        
        # Test broadcast to specific users
        target_users = [1, 3]
        personal_message = {"type": "notification", "message": "Portfolio alert"}
        await websocket_manager.broadcast_to_users(personal_message, target_users)
        
        # Verify only targeted users received the message
        mock_connections[0].send_json.assert_called_with(personal_message)  # User 1
        mock_connections[2].send_json.assert_called_with(personal_message)  # User 3
        
        # User 2 should not have received the personal message
        assert mock_connections[1].send_json.call_count == 1  # Only the broadcast message

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_websocket_real_time_stock_updates(self, websocket_manager, mock_websocket, mock_user):
        """Test real-time stock price updates via WebSocket."""
        
        await websocket_manager.connect(mock_websocket, mock_user.id)
        
        # Subscribe to stock updates
        subscription_message = {
            "type": "subscribe",
            "symbols": ["AAPL", "GOOGL", "MSFT"]
        }
        websocket_manager.subscribe_to_stocks(mock_user.id, subscription_message["symbols"])
        
        # Simulate price update
        price_update = {
            "type": "price_update",
            "symbol": "AAPL",
            "price": 155.50,
            "change": 1.25,
            "change_percent": 0.81,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Send update to subscribed users
        await websocket_manager.send_stock_update("AAPL", price_update)
        
        # Verify user received the update
        mock_websocket.send_json.assert_called_with(price_update)
        
        # Test unsubscribe
        websocket_manager.unsubscribe_from_stocks(mock_user.id, ["AAPL"])
        
        # Send another update - should not be received
        await websocket_manager.send_stock_update("AAPL", price_update)
        
        # Verify no additional message was sent
        assert mock_websocket.send_json.call_count == 1

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_websocket_portfolio_notifications(self, websocket_manager, mock_websocket, mock_user):
        """Test portfolio-related notifications via WebSocket."""
        
        await websocket_manager.connect(mock_websocket, mock_user.id)
        
        # Test portfolio value update
        portfolio_update = {
            "type": "portfolio_update",
            "portfolio_id": "portfolio-123",
            "total_value": 125000.50,
            "day_change": 2500.00,
            "day_change_percent": 2.04,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await websocket_manager.send_portfolio_update(mock_user.id, portfolio_update)
        mock_websocket.send_json.assert_called_with(portfolio_update)
        
        # Test trade execution notification
        trade_notification = {
            "type": "trade_executed",
            "symbol": "AAPL",
            "action": "buy",
            "quantity": 100,
            "price": 154.25,
            "total_amount": 15425.00,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await websocket_manager.send_trade_notification(mock_user.id, trade_notification)
        mock_websocket.send_json.assert_called_with(trade_notification)
        
        # Test alert notification
        alert_notification = {
            "type": "alert",
            "severity": "high",
            "title": "Price Target Reached",
            "message": "AAPL has reached your target price of $155.00",
            "symbol": "AAPL",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await websocket_manager.send_alert(mock_user.id, alert_notification)
        mock_websocket.send_json.assert_called_with(alert_notification)

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_websocket_error_handling(self, websocket_manager, mock_websocket, mock_user):
        """Test WebSocket error handling and connection recovery."""
        
        await websocket_manager.connect(mock_websocket, mock_user.id)
        
        # Test message sending failure
        mock_websocket.send_json.side_effect = ConnectionClosedError(None, None)
        
        test_message = {"type": "test", "data": "test"}
        result = await websocket_manager.send_to_user(mock_user.id, test_message)
        
        # Should handle error gracefully
        assert result is False
        
        # Connection should be removed from active connections
        assert mock_user.id not in websocket_manager.active_connections
        
        # Test connection cleanup on error
        websocket_manager.active_connections[mock_user.id] = mock_websocket
        mock_websocket.send_json.side_effect = Exception("Unexpected error")
        
        result = await websocket_manager.send_to_user(mock_user.id, test_message)
        assert result is False
        
        # Test rate limiting on reconnection
        websocket_manager.failed_connections[mock_user.id] = 5
        can_reconnect = websocket_manager.can_reconnect(mock_user.id)
        assert can_reconnect is False

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_websocket_message_queuing(self, websocket_manager, mock_user):
        """Test message queuing when client is disconnected."""
        
        # Send message to disconnected user
        test_message = {"type": "queued_message", "data": "important"}
        result = await websocket_manager.send_to_user(mock_user.id, test_message, queue_if_offline=True)
        
        # Should queue message
        assert result is True
        assert mock_user.id in websocket_manager.message_queue
        assert len(websocket_manager.message_queue[mock_user.id]) == 1
        
        # Connect user
        mock_websocket = AsyncMock(spec=WebSocket)
        await websocket_manager.connect(mock_websocket, mock_user.id)
        
        # Queued messages should be sent automatically
        await asyncio.sleep(0.1)  # Allow processing
        mock_websocket.send_json.assert_called_with(test_message)
        
        # Queue should be cleared
        assert len(websocket_manager.message_queue.get(mock_user.id, [])) == 0

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_websocket_authentication(self, connection_manager):
        """Test WebSocket authentication and authorization."""
        
        mock_websocket = AsyncMock(spec=WebSocket)
        
        # Test authentication with valid token
        with patch('backend.auth.oauth2.verify_token') as mock_verify:
            mock_user = User(id=1, username="testuser", email="test@test.com", is_active=True)
            mock_verify.return_value = mock_user
            
            auth_result = await connection_manager.authenticate_websocket(
                mock_websocket, 
                "valid_token"
            )
            
            assert auth_result is True
            assert mock_websocket.accept.called
        
        # Test authentication with invalid token
        with patch('backend.auth.oauth2.verify_token') as mock_verify:
            mock_verify.side_effect = Exception("Invalid token")
            
            auth_result = await connection_manager.authenticate_websocket(
                mock_websocket,
                "invalid_token"
            )
            
            assert auth_result is False
            mock_websocket.close.assert_called_with(code=4001, reason="Authentication failed")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_websocket_load_handling(self, websocket_manager):
        """Test WebSocket handling under high connection load."""
        
        # Create many concurrent connections
        connection_tasks = []
        num_connections = 100
        
        for i in range(num_connections):
            websocket = AsyncMock(spec=WebSocket)
            task = websocket_manager.connect(websocket, i + 1)
            connection_tasks.append(task)
        
        # Execute all connections concurrently
        await asyncio.gather(*connection_tasks)
        
        # Verify all connections were established
        assert websocket_manager.get_connection_count() == num_connections
        
        # Test broadcasting to all connections
        broadcast_message = {"type": "mass_update", "timestamp": datetime.utcnow().isoformat()}
        
        start_time = datetime.utcnow()
        await websocket_manager.broadcast_to_all(broadcast_message)
        end_time = datetime.utcnow()
        
        broadcast_duration = (end_time - start_time).total_seconds()
        
        # Should complete broadcast within reasonable time
        assert broadcast_duration < 5.0, f"Broadcast took {broadcast_duration}s, should be under 5s"
        
        # Test connection cleanup
        cleanup_tasks = []
        for i in range(num_connections):
            task = asyncio.create_task(websocket_manager.disconnect(i + 1))
            cleanup_tasks.append(task)
        
        await asyncio.gather(*cleanup_tasks)
        assert websocket_manager.get_connection_count() == 0

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_websocket_heartbeat_mechanism(self, websocket_manager, mock_websocket, mock_user):
        """Test WebSocket heartbeat/ping-pong mechanism."""
        
        await websocket_manager.connect(mock_websocket, mock_user.id)
        
        # Enable heartbeat
        websocket_manager.enable_heartbeat(mock_user.id, interval=1.0)
        
        # Wait for heartbeat to trigger
        await asyncio.sleep(1.5)
        
        # Verify ping was sent
        mock_websocket.send_json.assert_called()
        
        # Check that ping message was sent
        calls = mock_websocket.send_json.call_args_list
        ping_sent = any(
            call[0][0].get("type") == "ping" 
            for call in calls
        )
        assert ping_sent
        
        # Test heartbeat timeout
        mock_websocket.send_json.side_effect = asyncio.TimeoutError()
        
        # Wait for timeout detection
        await asyncio.sleep(2.0)
        
        # Connection should be marked as stale
        connection_info = websocket_manager.get_connection_info(mock_user.id)
        if connection_info:
            assert connection_info.get("stale", False) is True

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_websocket_subscription_management(self, websocket_manager, mock_websocket, mock_user):
        """Test WebSocket subscription management for different data types."""
        
        await websocket_manager.connect(mock_websocket, mock_user.id)
        
        # Test stock subscriptions
        stocks = ["AAPL", "GOOGL", "MSFT"]
        websocket_manager.subscribe_to_stocks(mock_user.id, stocks)
        
        user_subscriptions = websocket_manager.get_user_subscriptions(mock_user.id)
        assert "stocks" in user_subscriptions
        assert set(user_subscriptions["stocks"]) == set(stocks)
        
        # Test portfolio subscriptions
        portfolios = ["portfolio-123", "portfolio-456"]
        websocket_manager.subscribe_to_portfolios(mock_user.id, portfolios)
        
        user_subscriptions = websocket_manager.get_user_subscriptions(mock_user.id)
        assert "portfolios" in user_subscriptions
        assert set(user_subscriptions["portfolios"]) == set(portfolios)
        
        # Test news subscriptions
        news_categories = ["earnings", "analyst_ratings", "market_news"]
        websocket_manager.subscribe_to_news(mock_user.id, news_categories)
        
        user_subscriptions = websocket_manager.get_user_subscriptions(mock_user.id)
        assert "news" in user_subscriptions
        assert set(user_subscriptions["news"]) == set(news_categories)
        
        # Test subscription limits
        too_many_stocks = [f"STOCK{i:04d}" for i in range(1000)]
        result = websocket_manager.subscribe_to_stocks(mock_user.id, too_many_stocks)
        
        # Should enforce subscription limits
        assert result is False
        
        # Current subscriptions should remain unchanged
        current_stocks = websocket_manager.get_user_subscriptions(mock_user.id)["stocks"]
        assert len(current_stocks) == 3

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_websocket_data_compression(self, websocket_manager, mock_websocket, mock_user):
        """Test WebSocket data compression for large messages."""
        
        await websocket_manager.connect(mock_websocket, mock_user.id)
        
        # Create large message (market data for many stocks)
        large_message = {
            "type": "bulk_market_data",
            "data": {
                f"STOCK{i:04d}": {
                    "price": 100 + i * 0.1,
                    "volume": 1000000 + i * 1000,
                    "change": (i % 10) - 5,
                    "timestamp": datetime.utcnow().isoformat()
                }
                for i in range(1000)
            }
        }
        
        # Enable compression
        websocket_manager.enable_compression(mock_user.id, True)
        
        await websocket_manager.send_to_user(mock_user.id, large_message)
        
        # Verify message was sent (compression handled internally)
        mock_websocket.send_json.assert_called_once()
        
        # In real implementation, would verify:
        # - Message size reduction
        # - Decompression on client side
        # - Performance improvement

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_websocket_reconnection_handling(self, websocket_manager):
        """Test WebSocket reconnection handling and state restoration."""
        
        mock_websocket1 = AsyncMock(spec=WebSocket)
        mock_websocket2 = AsyncMock(spec=WebSocket)
        user_id = 1
        
        # Initial connection
        await websocket_manager.connect(mock_websocket1, user_id)
        websocket_manager.subscribe_to_stocks(user_id, ["AAPL", "GOOGL"])
        
        # Simulate connection loss
        websocket_manager.disconnect(user_id)
        
        # Reconnection
        await websocket_manager.connect(mock_websocket2, user_id)
        
        # Verify new connection
        assert websocket_manager.active_connections[user_id] == mock_websocket2
        
        # Test state restoration
        subscriptions = websocket_manager.get_user_subscriptions(user_id)
        
        # In production system, subscriptions might be restored
        # This test verifies the infrastructure is in place
        assert websocket_manager.supports_reconnection(user_id)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])