"""
WebSocket Service Layer

Business logic for WebSocket connection management, subscriptions,
message handling, and real-time data streaming.

Extracted from backend/api/routers/websocket.py to keep the router thin.
"""

import asyncio
import json
import logging
import random
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from sqlalchemy.ext.asyncio import AsyncSession

from backend.repositories import stock_repository
from backend.utils.cache import get_redis

logger = logging.getLogger(__name__)


# ---- Message types ----

class MessageType(str, Enum):
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    PRICE_UPDATE = "price_update"
    TRADE_EXECUTED = "trade_executed"
    ALERT = "alert"
    NEWS = "news"
    CHAT = "chat"
    SYSTEM = "system"
    HEARTBEAT = "heartbeat"
    ERROR = "error"


# ---- Connection manager ----

# ---- Timing constants ----

HEARTBEAT_INTERVAL = 30  # seconds between heartbeat pings
PRICE_STREAM_MIN_INTERVAL = 0.5  # seconds, minimum delay between price updates
PRICE_STREAM_MAX_INTERVAL = 3  # seconds, maximum delay between price updates
STREAM_ERROR_RETRY_DELAY = 5  # seconds to wait after a price stream error
STALE_CONNECTION_CLEANUP_INTERVAL = 300  # seconds (5 min) between stale-connection sweeps
CLEANUP_ERROR_RETRY_DELAY = 60  # seconds to wait after a cleanup task error


class EnhancedConnectionManager:
    """WebSocket connection manager with error handling and persistence."""

    def __init__(self):
        self.active_connections: Dict[str, Dict[str, Any]] = {}
        self.subscriptions: Dict[str, Set[str]] = {}
        self.user_sessions: Dict[str, Dict[str, Any]] = {}
        self.connection_health: Dict[str, datetime] = {}
        self.redis_client = None

    async def initialize(self):
        """Initialize Redis connection for persistence."""
        try:
            self.redis_client = await get_redis()
            logger.info("WebSocket manager initialized with Redis persistence")
        except Exception as e:
            logger.warning(f"Redis not available for WebSocket persistence: {e}")

    async def connect(self, websocket, client_id: str, user=None):
        """Connect a WebSocket with enhanced error handling."""
        try:
            await websocket.accept()

            self.active_connections[client_id] = {
                'websocket': websocket,
                'connected_at': datetime.now(timezone.utc),
                'user_id': user.id if user else None,
                'message_count': 0
            }

            if user:
                self.user_sessions[client_id] = {
                    'user_id': user.id,
                    'username': user.username,
                    'role': user.role
                }

            self.connection_health[client_id] = datetime.now(timezone.utc)

            if self.redis_client:
                try:
                    await self.redis_client.hset(
                        "websocket:connections",
                        client_id,
                        json.dumps({
                            'connected_at': datetime.now(timezone.utc).isoformat(),
                            'user_id': user.id if user else None
                        })
                    )
                except Exception as e:
                    logger.error(f"Error persisting connection info: {e}")

            logger.info(
                f"Client {client_id} connected "
                f"(User: {user.username if user else 'Anonymous'}). "
                f"Total connections: {len(self.active_connections)}"
            )

        except Exception as e:
            logger.error(f"Error connecting client {client_id}: {e}")
            raise

    async def disconnect(self, websocket, client_id: str):
        """Disconnect a WebSocket with cleanup."""
        try:
            if client_id in self.active_connections:
                connection_info = self.active_connections[client_id]
                logger.info(
                    f"Client {client_id} disconnected after "
                    f"{connection_info.get('message_count', 0)} messages"
                )
                del self.active_connections[client_id]

            if client_id in self.user_sessions:
                del self.user_sessions[client_id]

            if client_id in self.subscriptions:
                del self.subscriptions[client_id]

            if client_id in self.connection_health:
                del self.connection_health[client_id]

            if self.redis_client:
                try:
                    await self.redis_client.hdel("websocket:connections", client_id)
                    await self.redis_client.hdel("websocket:subscriptions", client_id)
                except Exception as e:
                    logger.error(f"Error cleaning up Redis data: {e}")

            logger.info(
                f"Client {client_id} disconnected. "
                f"Total connections: {len(self.active_connections)}"
            )

        except Exception as e:
            logger.error(f"Error disconnecting client {client_id}: {e}")

    async def send_personal_message(self, message: str, client_id: str) -> bool:
        """Send message to specific client with error handling."""
        try:
            if client_id not in self.active_connections:
                logger.warning(f"Client {client_id} not found for personal message")
                return False

            websocket = self.active_connections[client_id]['websocket']
            await websocket.send_text(message)

            self.active_connections[client_id]['message_count'] += 1

            return True

        except Exception as e:
            logger.error(f"Error sending personal message to {client_id}: {e}")
            await self.disconnect(None, client_id)
            return False

    async def broadcast(
        self,
        message: str,
        exclude: Optional[str] = None,
        target_role: Optional[str] = None,
    ) -> int:
        """Broadcast message with role filtering and error handling."""
        successful_sends = 0
        failed_clients = []

        for client_id, connection_info in self.active_connections.items():
            if client_id == exclude:
                continue

            if target_role and client_id in self.user_sessions:
                user_role = self.user_sessions[client_id].get('role')
                if user_role != target_role:
                    continue

            try:
                websocket = connection_info['websocket']
                await websocket.send_text(message)
                connection_info['message_count'] += 1
                successful_sends += 1

            except Exception as e:
                logger.error(f"Error broadcasting to {client_id}: {e}")
                failed_clients.append(client_id)

        for client_id in failed_clients:
            await self.disconnect(None, client_id)

        logger.debug(
            f"Broadcast sent to {successful_sends} clients, "
            f"{len(failed_clients)} failed"
        )
        return successful_sends

    async def subscribe(
        self,
        client_id: str,
        symbols: List[str],
        db_session: Optional[AsyncSession] = None,
    ) -> List[str]:
        """Subscribe client to symbols with validation."""
        try:
            if client_id not in self.subscriptions:
                self.subscriptions[client_id] = set()

            if db_session:
                valid_symbols = []
                for symbol in symbols:
                    stock = await stock_repository.get_by_symbol(
                        symbol.upper(), session=db_session
                    )
                    if stock:
                        valid_symbols.append(symbol.upper())
                    else:
                        logger.warning(f"Invalid symbol for subscription: {symbol}")
                symbols = valid_symbols

            self.subscriptions[client_id].update(symbols)

            if self.redis_client:
                try:
                    await self.redis_client.hset(
                        "websocket:subscriptions",
                        client_id,
                        json.dumps(list(self.subscriptions[client_id]))
                    )
                except Exception as e:
                    logger.error(f"Error persisting subscriptions: {e}")

            logger.info(f"Client {client_id} subscribed to {len(symbols)} symbols")
            return symbols

        except Exception as e:
            logger.error(f"Error subscribing client {client_id}: {e}")
            return []

    def unsubscribe(self, client_id: str, symbols: List[str]):
        """Unsubscribe client from symbols."""
        if client_id in self.subscriptions:
            self.subscriptions[client_id].difference_update(symbols)
            logger.info(f"Client {client_id} unsubscribed from {len(symbols)} symbols")

    def get_subscriptions(self, client_id: str) -> Set[str]:
        """Get client's subscriptions."""
        return self.subscriptions.get(client_id, set())

    async def send_to_subscribers(
        self,
        symbol: str,
        message: str,
        exclude_client: Optional[str] = None,
    ) -> int:
        """Send message to all subscribers of a symbol."""
        sent_count = 0
        failed_clients = []

        for client_id, symbols in self.subscriptions.items():
            if (
                symbol in symbols
                and client_id != exclude_client
                and client_id in self.active_connections
            ):
                try:
                    websocket = self.active_connections[client_id]['websocket']
                    await websocket.send_text(message)
                    self.active_connections[client_id]['message_count'] += 1
                    sent_count += 1

                except Exception as e:
                    logger.error(f"Error sending to subscriber {client_id}: {e}")
                    failed_clients.append(client_id)

        for client_id in failed_clients:
            await self.disconnect(None, client_id)

        return sent_count

    async def update_health(self, client_id: str):
        """Update client health status."""
        self.connection_health[client_id] = datetime.now(timezone.utc)

    async def cleanup_stale_connections(self, max_age_minutes: int = 30) -> int:
        """Clean up stale connections."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=max_age_minutes)
        stale_clients = []

        for client_id, last_heartbeat in self.connection_health.items():
            if last_heartbeat < cutoff_time:
                stale_clients.append(client_id)

        for client_id in stale_clients:
            logger.info(f"Cleaning up stale connection: {client_id}")
            await self.disconnect(None, client_id)

        return len(stale_clients)


# ---- Helper / business-logic functions ----

async def handle_client_message(
    websocket,
    client_id: str,
    message: Dict[str, Any],
    *,
    connection_manager: EnhancedConnectionManager,
    active_streams: Dict[str, "asyncio.Task"],
):
    """Legacy message handler for backwards compatibility.

    Parameters
    ----------
    websocket:
        The WebSocket connection to respond on.
    client_id:
        Unique identifier of the sending client.
    message:
        Parsed message dict from the client.
    connection_manager:
        The ``EnhancedConnectionManager`` instance to operate on.
    active_streams:
        Dict mapping symbol -> asyncio.Task for running price streams.
    """
    msg_type = message.get("type")

    if msg_type == MessageType.SUBSCRIBE:
        symbols = message.get("symbols", [])
        await connection_manager.subscribe(client_id, symbols)

        for symbol in symbols:
            if symbol not in active_streams:
                active_streams[symbol] = asyncio.create_task(
                    stream_price_updates(symbol, connection_manager)
                )

        await websocket.send_json({
            "type": MessageType.SYSTEM,
            "message": f"Subscribed to {symbols}",
            "symbols": symbols,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

    elif msg_type == MessageType.UNSUBSCRIBE:
        symbols = message.get("symbols", [])
        connection_manager.unsubscribe(client_id, symbols)

        await websocket.send_json({
            "type": MessageType.SYSTEM,
            "message": f"Unsubscribed from {symbols}",
            "symbols": symbols,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

    elif msg_type == MessageType.CHAT:
        chat_message = {
            "type": MessageType.CHAT,
            "from": client_id,
            "message": message.get("message", ""),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        await connection_manager.broadcast(json.dumps(chat_message))

    else:
        await websocket.send_json({
            "type": MessageType.ERROR,
            "message": f"Unknown message type: {msg_type}",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })


async def stream_price_updates(
    symbol: str,
    connection_manager: EnhancedConnectionManager,
):
    """Stream real-time price updates for a symbol.

    Production (``settings.DEMO_MODE`` False, default): refuse to fabricate
    ticks — emit a single ``price_unavailable`` payload and stop (parity with
    ``socketio_service._stream_price_updates``). Demo mode keeps synthetic
    ticks tagged ``data_source: simulated``.
    """
    from backend.config.settings import settings

    if not settings.DEMO_MODE:
        unavailable = {
            "type": MessageType.ERROR,
            "error": "model_unavailable",
            "reason": "live_feed_not_configured",
            "symbol": symbol,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        await connection_manager.send_to_subscribers(
            symbol, json.dumps(unavailable)
        )
        logger.warning(
            "stream_price_updates: refusing to fabricate ticks for %s "
            "(DEMO_MODE=false)",
            symbol,
        )
        return

    while True:
        try:
            price_update = {
                "type": MessageType.PRICE_UPDATE,
                "symbol": symbol,
                "price": random.uniform(50, 500),
                "change": random.uniform(-5, 5),
                "change_percent": random.uniform(-2, 2),
                "volume": random.randint(1000000, 50000000),
                "bid": random.uniform(49, 499),
                "ask": random.uniform(51, 501),
                "bid_size": random.randint(100, 1000),
                "ask_size": random.randint(100, 1000),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data_source": "simulated",
            }

            await connection_manager.send_to_subscribers(
                symbol, json.dumps(price_update)
            )

            await asyncio.sleep(
                random.uniform(PRICE_STREAM_MIN_INTERVAL, PRICE_STREAM_MAX_INTERVAL)
            )

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error streaming price for {symbol}: {e}")
            await asyncio.sleep(STREAM_ERROR_RETRY_DELAY)


async def send_heartbeat(websocket, client_id: str):
    """Send periodic heartbeat to keep connection alive."""
    while True:
        try:
            await asyncio.sleep(HEARTBEAT_INTERVAL)

            heartbeat = {
                "type": MessageType.HEARTBEAT,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "server_time": datetime.now(timezone.utc).timestamp()
            }

            await websocket.send_json(heartbeat)

        except asyncio.CancelledError:
            break
        except Exception:
            break


async def cleanup_client_streams(
    client_id: str,
    *,
    connection_manager: EnhancedConnectionManager,
    active_streams: Dict[str, "asyncio.Task"],
):
    """Clean up resources when client disconnects.

    Cancels price-stream tasks for symbols that no longer have any subscribers.
    """
    subscriptions = connection_manager.get_subscriptions(client_id)

    for symbol in subscriptions:
        still_subscribed = False
        for other_client_id, other_subs in connection_manager.subscriptions.items():
            if other_client_id != client_id and symbol in other_subs:
                still_subscribed = True
                break

        if not still_subscribed and symbol in active_streams:
            active_streams[symbol].cancel()
            del active_streams[symbol]


async def send_alert(
    client_id: str,
    alert: Dict[str, Any],
    *,
    connection_manager: EnhancedConnectionManager,
):
    """Send alert to specific client."""
    alert_message = {
        "type": MessageType.ALERT,
        "alert": alert,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    await connection_manager.send_personal_message(
        json.dumps(alert_message), client_id
    )


async def broadcast_news(
    news: Dict[str, Any],
    *,
    connection_manager: EnhancedConnectionManager,
):
    """Broadcast news to all connected clients."""
    news_message = {
        "type": MessageType.NEWS,
        "news": news,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    await connection_manager.broadcast(json.dumps(news_message))


def generate_market_overview_data() -> Dict[str, Any]:
    """Generate a market overview data snapshot.

    Production refuses synthetic indices; demo mode returns simulated data.
    """
    from backend.config.settings import settings

    if not settings.DEMO_MODE:
        return {
            "type": "market_overview",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": "model_unavailable",
            "reason": "live_feed_not_configured",
            "indices": {},
            "data_source": "unavailable",
        }

    return {
        "type": "market_overview",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data_source": "simulated",
        "indices": {
            "SPY": {
                "price": random.uniform(400, 450),
                "change": random.uniform(-2, 2),
                "change_percent": random.uniform(-0.5, 0.5)
            },
            "QQQ": {
                "price": random.uniform(350, 400),
                "change": random.uniform(-3, 3),
                "change_percent": random.uniform(-0.75, 0.75)
            },
            "DIA": {
                "price": random.uniform(330, 370),
                "change": random.uniform(-1.5, 1.5),
                "change_percent": random.uniform(-0.4, 0.4)
            }
        },
        "market_sentiment": random.uniform(-1, 1),
        "vix": random.uniform(12, 30),
        "advance_decline": {
            "advancing": random.randint(1500, 2500),
            "declining": random.randint(500, 1500),
            "unchanged": random.randint(100, 300)
        },
        "volume": {
            "total": random.randint(5000000000, 10000000000),
            "up_volume": random.randint(2000000000, 6000000000),
            "down_volume": random.randint(1000000000, 4000000000)
        }
    }


def generate_portfolio_update_data(portfolio_id: str) -> Dict[str, Any]:
    """Generate a portfolio update data snapshot.

    Production refuses synthetic portfolio ticks; demo mode returns simulated data.
    """
    from backend.config.settings import settings

    if not settings.DEMO_MODE:
        return {
            "type": "portfolio_update",
            "portfolio_id": portfolio_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": "model_unavailable",
            "reason": "live_feed_not_configured",
            "positions": [],
            "alerts": [],
            "data_source": "unavailable",
        }

    return {
        "type": "portfolio_update",
        "portfolio_id": portfolio_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data_source": "simulated",
        "total_value": random.uniform(90000, 110000),
        "day_change": random.uniform(-2000, 2000),
        "day_change_percent": random.uniform(-2, 2),
        "positions": [
            {
                "symbol": "AAPL",
                "current_price": random.uniform(190, 200),
                "change": random.uniform(-2, 2),
                "value": random.uniform(15000, 20000)
            },
            {
                "symbol": "GOOGL",
                "current_price": random.uniform(145, 155),
                "change": random.uniform(-1.5, 1.5),
                "value": random.uniform(10000, 15000)
            }
        ],
        "alerts": []
    }


async def cleanup_stale_connections_task(
    connection_manager: EnhancedConnectionManager,
):
    """Background task to clean up stale WebSocket connections."""
    while True:
        try:
            cleaned = await connection_manager.cleanup_stale_connections()
            if cleaned > 0:
                logger.info(f"Cleaned up {cleaned} stale WebSocket connections")
            await asyncio.sleep(STALE_CONNECTION_CLEANUP_INTERVAL)
        except Exception as e:
            logger.error(f"Error in cleanup task: {e}")
            await asyncio.sleep(CLEANUP_ERROR_RETRY_DELAY)
