"""
Socket.IO Service

Provides a python-socketio AsyncServer instance that is mounted alongside the
FastAPI application.  This replaces the previous native-WebSocket implementation
with a standardised Socket.IO transport so that all clients (frontend service
and hooks) speak the same protocol.

Event model
-----------
Client -> Server events:
  connect              – Socket.IO lifecycle, handled automatically
  disconnect           – Socket.IO lifecycle, handled automatically
  subscribe_prices     – { symbols: ["AAPL", "TSLA", ...] }
  subscribe_portfolio  – { portfolio_id: "uuid" }
  subscribe_alerts     – { user_id: "uuid" }

Server -> Client events:
  price_update         – { symbol, price, change, change_percent, volume,
                           bid, ask, bid_size, ask_size, timestamp }
  portfolio_update     – { portfolio_id, total_value, day_change,
                           day_change_percent, positions, timestamp }
  alert_notification   – { alert_type, message, severity, timestamp }
  system               – { message, timestamp }
  error                – { code, message, timestamp }

Rooms
-----
- "prices:{SYMBOL}"      – clients subscribed to a given ticker
- "portfolio:{id}"       – clients subscribed to a given portfolio
- "alerts:{user_id}"     – clients subscribed to their own alerts
"""

import asyncio
import logging
import random
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import socketio

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# AsyncServer instance (module-level singleton)
# ---------------------------------------------------------------------------

sio: socketio.AsyncServer = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins="*",
    logger=False,
    engineio_logger=False,
)

# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------

# Maps socket session id -> set of rooms the client is in
_client_rooms: Dict[str, set] = {}

# Maps price-room name -> background streaming task
_price_tasks: Dict[str, asyncio.Task] = {}

# Timing constants (seconds)
_PRICE_MIN_INTERVAL = 0.5
_PRICE_MAX_INTERVAL = 3.0
_STREAM_ERROR_RETRY = 5


# ---------------------------------------------------------------------------
# Lifecycle events
# ---------------------------------------------------------------------------

@sio.event
async def connect(sid: str, environ: Dict[str, Any], auth: Optional[Dict] = None):
    """Handle a new Socket.IO connection."""
    _client_rooms[sid] = set()
    logger.info("Socket.IO client connected: %s (total: %d)", sid, len(_client_rooms))
    await sio.emit(
        "system",
        {
            "message": "Connected to Investment Analysis Platform real-time feed",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        to=sid,
    )


@sio.event
async def disconnect(sid: str):
    """Handle a Socket.IO disconnection and clean up rooms / tasks."""
    rooms = _client_rooms.pop(sid, set())
    logger.info(
        "Socket.IO client disconnected: %s (was in %d rooms)", sid, len(rooms)
    )

    for room in rooms:
        await sio.leave_room(sid, room)
        if room.startswith("prices:"):
            await _maybe_stop_price_stream(room)


# ---------------------------------------------------------------------------
# Client -> Server event handlers
# ---------------------------------------------------------------------------

@sio.event
async def subscribe_prices(sid: str, data: Dict[str, Any]):
    """
    Subscribe *sid* to one or more ticker price streams.

    Expected payload::

        { "symbols": ["AAPL", "TSLA"] }
    """
    symbols = data.get("symbols", [])
    if not isinstance(symbols, list) or not symbols:
        await sio.emit(
            "error",
            {
                "code": "INVALID_PAYLOAD",
                "message": "subscribe_prices requires a non-empty 'symbols' list",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            to=sid,
        )
        return

    subscribed = []
    for symbol in symbols:
        symbol = str(symbol).upper().strip()
        if not symbol:
            continue

        room = f"prices:{symbol}"
        await sio.enter_room(sid, room)
        _client_rooms.setdefault(sid, set()).add(room)
        subscribed.append(symbol)

        # Start background price streaming task if not already running
        if room not in _price_tasks or _price_tasks[room].done():
            _price_tasks[room] = asyncio.create_task(
                _stream_price_updates(symbol, room)
            )

    await sio.emit(
        "system",
        {
            "message": f"Subscribed to price updates for: {subscribed}",
            "symbols": subscribed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        to=sid,
    )
    logger.info("Client %s subscribed to prices: %s", sid, subscribed)


@sio.event
async def subscribe_portfolio(sid: str, data: Dict[str, Any]):
    """
    Subscribe *sid* to portfolio updates.

    Expected payload::

        { "portfolio_id": "uuid-string" }
    """
    portfolio_id = data.get("portfolio_id")
    if not portfolio_id:
        await sio.emit(
            "error",
            {
                "code": "INVALID_PAYLOAD",
                "message": "subscribe_portfolio requires a 'portfolio_id'",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            to=sid,
        )
        return

    room = f"portfolio:{portfolio_id}"
    await sio.enter_room(sid, room)
    _client_rooms.setdefault(sid, set()).add(room)

    await sio.emit(
        "system",
        {
            "message": f"Subscribed to portfolio updates for: {portfolio_id}",
            "portfolio_id": portfolio_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        to=sid,
    )
    logger.info("Client %s subscribed to portfolio: %s", sid, portfolio_id)


@sio.event
async def subscribe_alerts(sid: str, data: Dict[str, Any]):
    """
    Subscribe *sid* to alert notifications for a user.

    Expected payload::

        { "user_id": "uuid-string" }
    """
    user_id = data.get("user_id")
    if not user_id:
        await sio.emit(
            "error",
            {
                "code": "INVALID_PAYLOAD",
                "message": "subscribe_alerts requires a 'user_id'",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            to=sid,
        )
        return

    room = f"alerts:{user_id}"
    await sio.enter_room(sid, room)
    _client_rooms.setdefault(sid, set()).add(room)

    await sio.emit(
        "system",
        {
            "message": f"Subscribed to alerts for user: {user_id}",
            "user_id": user_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        to=sid,
    )
    logger.info("Client %s subscribed to alerts for user: %s", sid, user_id)


# ---------------------------------------------------------------------------
# Server -> Client broadcast helpers (callable from other modules)
# ---------------------------------------------------------------------------

async def emit_price_update(symbol: str, price_data: Dict[str, Any]) -> None:
    """Broadcast a price update to all subscribers of *symbol*."""
    await sio.emit("price_update", price_data, room=f"prices:{symbol}")


async def emit_portfolio_update(portfolio_id: str, update_data: Dict[str, Any]) -> None:
    """Broadcast a portfolio update to all subscribers of *portfolio_id*."""
    await sio.emit("portfolio_update", update_data, room=f"portfolio:{portfolio_id}")


async def emit_alert_notification(user_id: str, alert_data: Dict[str, Any]) -> None:
    """Send an alert notification to a specific user's alert room."""
    await sio.emit("alert_notification", alert_data, room=f"alerts:{user_id}")


# ---------------------------------------------------------------------------
# Background price streaming
# ---------------------------------------------------------------------------

async def _stream_price_updates(symbol: str, room: str) -> None:
    """Stream price updates to ``room`` until cancelled.

    Per PRD audit 2026-04 F-02-003 / Q4 default (recorded 2026-04-28): this
    coroutine previously fabricated price/change/volume/bid/ask values via
    ``random.uniform`` and broadcast them to subscribed clients. For an
    SEC-regulated investment platform, broadcasting fake live prices is a
    serious correctness + compliance failure.

    Production behaviour (``settings.DEMO_MODE`` False, default):
        Emit a single ``price_unavailable`` payload telling subscribers
        that the live feed is not wired up and the stream is intentionally
        stopped — avoids leaving sockets open spinning out fake ticks.

    Demo behaviour (``settings.DEMO_MODE`` True): keep the legacy synthetic
    stream, but every payload is tagged ``data_source: 'simulated'`` so
    consumers can render an explicit "demo data" badge.
    """
    from backend.config.settings import settings

    logger.info("Starting price stream for symbol: %s", symbol)

    if not settings.DEMO_MODE:
        await sio.emit(
            "price_unavailable",
            {
                "symbol": symbol,
                "error": "model_unavailable",
                "reason": "live_feed_not_configured",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            room=room,
        )
        logger.warning(
            "_stream_price_updates: refusing to fabricate ticks for %s "
            "(DEMO_MODE=false); emitting price_unavailable", symbol,
        )
        return

    try:
        while True:
            payload = {
                "symbol": symbol,
                "price": round(random.uniform(50, 500), 2),
                "change": round(random.uniform(-5, 5), 2),
                "change_percent": round(random.uniform(-2, 2), 4),
                "volume": random.randint(1_000_000, 50_000_000),
                "bid": round(random.uniform(49, 499), 2),
                "ask": round(random.uniform(51, 501), 2),
                "bid_size": random.randint(100, 1000),
                "ask_size": random.randint(100, 1000),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data_source": "simulated",  # F-02-003: explicit demo tag
            }
            await sio.emit("price_update", payload, room=room)
            await asyncio.sleep(
                random.uniform(_PRICE_MIN_INTERVAL, _PRICE_MAX_INTERVAL)
            )

    except asyncio.CancelledError:
        logger.info("Price stream cancelled for symbol: %s", symbol)
    except Exception as exc:
        logger.error("Price stream error for %s: %s", symbol, exc)
        await asyncio.sleep(_STREAM_ERROR_RETRY)


async def _maybe_stop_price_stream(room: str) -> None:
    """Cancel the price streaming task for *room* if no clients remain."""
    rooms_in_use = {r for client_rooms in _client_rooms.values() for r in client_rooms}
    if room not in rooms_in_use and room in _price_tasks:
        task = _price_tasks.pop(room)
        task.cancel()
        logger.info("Stopped price stream for room: %s (no remaining subscribers)", room)


# ---------------------------------------------------------------------------
# ASGI app factory
# ---------------------------------------------------------------------------

def create_socketio_asgi_app(fastapi_app) -> socketio.ASGIApp:
    """
    Wrap *fastapi_app* in a Socket.IO ``ASGIApp``.

    The Socket.IO server handles requests whose path starts with ``/socket.io/``
    and delegates everything else to *fastapi_app*.

    Usage in ``main.py``::

        from backend.services.socketio_service import sio, create_socketio_asgi_app
        socket_app = create_socketio_asgi_app(app)
        # Mount socket_app as the top-level ASGI callable in uvicorn.
    """
    return socketio.ASGIApp(sio, other_asgi_app=fastapi_app)
