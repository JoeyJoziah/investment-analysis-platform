"""
WebSocket Router

Thin routing layer for WebSocket endpoints.  Business logic lives in
``backend.services.websocket_service``.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from typing import Dict, Optional, Any
import asyncio
import json
import logging
from datetime import datetime, timezone
import uuid

# Security imports
from backend.security.websocket_security import (
    secure_websocket, WebSocketSecurityManager,
    WebSocketClient, WebSocketMessageType, send_error_message,
    validate_subscription_permissions
)
from backend.security.audit_logging import get_audit_logger, AuditEventType, AuditSeverity
from backend.auth.oauth2 import verify_token as _verify_bearer_token

# Service layer -- all business logic lives here
from backend.services.websocket_service import (
    EnhancedConnectionManager,
    MessageType,
    generate_market_overview_data,
    generate_portfolio_update_data,
    cleanup_stale_connections_task as _cleanup_stale_connections_task,
)
from backend.services import websocket_service as _ws_svc

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])

# ---- Module-level singletons (preserves existing patch paths) ----

manager = EnhancedConnectionManager()

active_price_streams: Dict[str, asyncio.Task] = {}
market_data_stream: Optional[asyncio.Task] = None


# ---- Startup helpers ----

async def initialize_websocket_manager():
    """Initialize the WebSocket manager."""
    await manager.initialize()


cleanup_task: Optional[asyncio.Task] = None


def start_cleanup_task():
    """Start the cleanup task -- call from FastAPI startup event."""
    global cleanup_task
    if cleanup_task is None:
        cleanup_task = asyncio.create_task(
            _cleanup_stale_connections_task(manager)
        )


# ---------------------------------------------------------------------------
# Compatibility shims -- these thin wrappers delegate to the service layer
# while binding to the module-level ``manager`` and ``active_price_streams``
# so that existing test patches on ``backend.api.routers.websocket.manager``
# and ``backend.api.routers.websocket.active_price_streams`` keep working.
# ---------------------------------------------------------------------------

async def handle_client_message(
    websocket: WebSocket,
    client_id: str,
    message: Dict[str, Any],
):
    """Legacy message handler (delegates to service)."""
    await _ws_svc.handle_client_message(
        websocket,
        client_id,
        message,
        connection_manager=manager,
        active_streams=active_price_streams,
    )


async def stream_price_updates(symbol: str):
    """Stream price updates for *symbol* (delegates to service)."""
    await _ws_svc.stream_price_updates(symbol, manager)


async def cleanup_client_streams(client_id: str):
    """Clean up resources when client disconnects (delegates to service)."""
    await _ws_svc.cleanup_client_streams(
        client_id,
        connection_manager=manager,
        active_streams=active_price_streams,
    )


async def send_alert(client_id: str, alert: Dict[str, Any]):
    """Send alert to specific client (delegates to service)."""
    await _ws_svc.send_alert(
        client_id, alert, connection_manager=manager
    )


async def broadcast_news(news: Dict[str, Any]):
    """Broadcast news to all connected clients (delegates to service)."""
    await _ws_svc.broadcast_news(news, connection_manager=manager)


# ---------------------------------------------------------------------------
# WebSocket endpoints
# ---------------------------------------------------------------------------

@router.websocket("/stream")
@secure_websocket(require_auth=False, allowed_roles=None)
async def websocket_endpoint(
    websocket: WebSocket,
    client_id: str = Query(default_factory=lambda: str(uuid.uuid4())),
    token: Optional[str] = Query(None),
    security_manager: WebSocketSecurityManager = None,
    client: WebSocketClient = None,
):
    """Main WebSocket endpoint for real-time data streaming with security."""
    audit_logger = get_audit_logger()

    try:
        welcome_message = {
            "type": WebSocketMessageType.SYSTEM.value,
            "message": "Connected to secure real-time stream",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "client_id": client_id,
            "authenticated": client.is_authenticated,
            "allowed_actions": list(client.allowed_actions),
            "server_version": "1.0.0"
        }

        await security_manager.send_secure_message(client_id, welcome_message)
        await manager.connect(websocket, client_id, client.user_session)

        try:
            while True:
                raw_data = await websocket.receive_text()

                is_valid, message, error = await security_manager.validate_message(
                    client_id, raw_data
                )

                if not is_valid:
                    await send_error_message(websocket, "VALIDATION_FAILED", error)
                    continue

                await handle_secure_client_message(
                    websocket, client_id, message, security_manager, client
                )

        except WebSocketDisconnect:
            logger.info(f"WebSocket client {client_id} disconnected normally")
        except Exception as e:
            logger.error(f"WebSocket error for client {client_id}: {e}")
            await send_error_message(websocket, "INTERNAL_ERROR", "Connection error occurred")

    finally:
        await manager.disconnect(websocket, client_id)
        await cleanup_client_streams(client_id)


def _reject_unauthenticated(token: Optional[str]) -> Optional[dict]:
    """
    Validate a bearer token passed on a WebSocket handshake.

    Returns the decoded payload dict on success, or ``None`` when the token
    is absent or invalid.  Callers must close the connection with an
    appropriate policy-violation code when ``None`` is returned.
    """
    if not token:
        return None
    return _verify_bearer_token(token)


@router.websocket("/market")
async def market_data_stream_endpoint(
    websocket: WebSocket,
    token: Optional[str] = Query(None),
):
    """Dedicated WebSocket for market-wide data streaming.

    Requires a valid bearer token supplied as the ``token`` query parameter.
    The handshake is rejected with close-code 4401 when the token is absent
    or invalid — no financial data is ever sent to unauthenticated callers.
    """
    payload = _reject_unauthenticated(token)
    if payload is None:
        await websocket.close(code=4401)
        logger.warning("Market data WS: rejected unauthenticated connection")
        return

    await websocket.accept()

    try:
        while True:
            market_data = generate_market_overview_data()
            await websocket.send_json(market_data)
            await asyncio.sleep(5)

    except WebSocketDisconnect:
        logger.info("Market data stream client disconnected")


@router.websocket("/portfolio/{portfolio_id}")
async def portfolio_stream(
    websocket: WebSocket,
    portfolio_id: str,
    token: Optional[str] = Query(None),
):
    """WebSocket for portfolio-specific updates.

    Requires a valid bearer token supplied as the ``token`` query parameter.
    The connection is rejected with close-code 4401 when no token is
    provided, and with close-code 4403 when the authenticated user does not
    own the requested portfolio.
    """
    from backend.services.portfolio_service import portfolio_service as _ps
    from backend.config.database import get_async_db_session

    payload = _reject_unauthenticated(token)
    if payload is None:
        await websocket.close(code=4401)
        logger.warning(
            "Portfolio WS %s: rejected unauthenticated connection", portfolio_id
        )
        return

    # Resolve user_id from token payload.  The canonical oauth2 token stores
    # the numeric user id under the "user_id" claim (see oauth2.py:188).
    token_user_id = payload.get("user_id")
    if token_user_id is None:
        await websocket.close(code=4401)
        logger.warning(
            "Portfolio WS %s: token missing user_id claim", portfolio_id
        )
        return

    # Verify portfolio ownership before accepting the connection.
    try:
        db_gen = get_async_db_session()
        db = await db_gen.__anext__()
        try:
            ownership = await _ps.compute_portfolio_detail(
                portfolio_id=portfolio_id,
                user_id=int(token_user_id),
                db=db,
            )
        finally:
            try:
                await db_gen.aclose()
            except Exception:
                pass
    except Exception as exc:
        logger.error(
            "Portfolio WS %s: ownership check error: %s", portfolio_id, exc
        )
        await websocket.close(code=4401)
        return

    if ownership is None:
        await websocket.close(code=4403)
        logger.warning(
            "Portfolio WS %s: access denied for user %s", portfolio_id, token_user_id
        )
        return

    await websocket.accept()

    try:
        while True:
            portfolio_update = generate_portfolio_update_data(portfolio_id)
            await websocket.send_json(portfolio_update)
            await asyncio.sleep(3)

    except WebSocketDisconnect:
        logger.info(f"Portfolio stream for {portfolio_id} disconnected")


# ---------------------------------------------------------------------------
# Secure message handler (stays in router -- tightly coupled to security)
# ---------------------------------------------------------------------------

async def handle_secure_client_message(
    websocket: WebSocket,
    client_id: str,
    message: Dict[str, Any],
    security_manager: WebSocketSecurityManager,
    client: WebSocketClient,
):
    """Handle incoming messages from clients with security validation."""
    audit_logger = get_audit_logger()
    msg_type = message.get("type")

    try:
        if msg_type == WebSocketMessageType.AUTHENTICATE.value:
            token = message.get("token")
            if token:
                user_session = await security_manager.authenticator.authenticate_connection(
                    websocket, token=token
                )
                if user_session:
                    client.user_session = user_session
                    client.is_authenticated = True
                    client.allowed_actions = security_manager._get_allowed_actions(user_session.role)

                    response = {
                        "type": WebSocketMessageType.SYSTEM.value,
                        "message": "Authentication successful",
                        "authenticated": True,
                        "user_id": user_session.user_id,
                        "role": user_session.role.value,
                        "allowed_actions": list(client.allowed_actions)
                    }
                else:
                    response = {
                        "type": WebSocketMessageType.ERROR.value,
                        "message": "Authentication failed",
                        "code": "AUTH_FAILED"
                    }
            else:
                response = {
                    "type": WebSocketMessageType.ERROR.value,
                    "message": "Token required for authentication",
                    "code": "TOKEN_REQUIRED"
                }

            await security_manager.send_secure_message(client_id, response)

        elif msg_type == WebSocketMessageType.SUBSCRIBE.value:
            symbols = message.get("symbols", [])
            allowed_symbols, denied_symbols = await validate_subscription_permissions(client, symbols)

            if denied_symbols:
                await security_manager.send_secure_message(client_id, {
                    "type": WebSocketMessageType.ERROR.value,
                    "message": f"Access denied to symbols: {denied_symbols}",
                    "code": "SUBSCRIPTION_DENIED"
                })

            if allowed_symbols:
                validated_symbols = await manager.subscribe(client_id, allowed_symbols)

                client.subscriptions.update(validated_symbols)
                client.subscription_count = len(client.subscriptions)

                for symbol in validated_symbols:
                    if symbol not in active_price_streams:
                        active_price_streams[symbol] = asyncio.create_task(
                            stream_price_updates(symbol)
                        )

                await audit_logger.log_event(
                    AuditEventType.DATA_ACCESS,
                    user_id=client.user_session.user_id if client.user_session else None,
                    ip_address=client.ip_address,
                    action="websocket_subscribe",
                    resource=",".join(validated_symbols),
                    severity=AuditSeverity.LOW,
                    details={"client_id": client_id, "symbols": validated_symbols}
                )

                await security_manager.send_secure_message(client_id, {
                    "type": WebSocketMessageType.SYSTEM.value,
                    "message": f"Subscribed to {len(validated_symbols)} symbols",
                    "symbols": validated_symbols,
                    "denied_symbols": denied_symbols
                })

        elif msg_type == WebSocketMessageType.UNSUBSCRIBE.value:
            symbols = message.get("symbols", [])
            manager.unsubscribe(client_id, symbols)

            client.subscriptions.difference_update(symbols)
            client.subscription_count = len(client.subscriptions)

            await security_manager.send_secure_message(client_id, {
                "type": WebSocketMessageType.SYSTEM.value,
                "message": f"Unsubscribed from {len(symbols)} symbols",
                "symbols": symbols
            })

        elif msg_type == WebSocketMessageType.HEARTBEAT.value:
            client.update_activity()

            await security_manager.send_secure_message(client_id, {
                "type": WebSocketMessageType.HEARTBEAT.value,
                "message": "pong",
                "server_time": datetime.now(timezone.utc).timestamp()
            })

        elif msg_type == WebSocketMessageType.CHAT.value:
            if not client.is_authenticated:
                await send_error_message(websocket, "AUTH_REQUIRED", "Authentication required for chat")
                return

            chat_content = message.get("message", "").strip()
            if not chat_content or len(chat_content) > 500:
                await send_error_message(websocket, "INVALID_MESSAGE", "Invalid chat message")
                return

            chat_message = {
                "type": WebSocketMessageType.CHAT.value,
                "from": client.user_session.username if client.user_session else client_id,
                "user_id": client.user_session.user_id if client.user_session else None,
                "message": chat_content,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            sent_count = await security_manager.broadcast_secure_message(chat_message, exclude_client=client_id)

            await audit_logger.log_event(
                AuditEventType.DATA_CREATE,
                user_id=client.user_session.user_id if client.user_session else None,
                ip_address=client.ip_address,
                action="websocket_chat",
                severity=AuditSeverity.LOW,
                details={"client_id": client_id, "recipients": sent_count, "message_length": len(chat_content)}
            )

        else:
            await send_error_message(websocket, "UNKNOWN_MESSAGE_TYPE", f"Unknown message type: {msg_type}")

    except Exception as e:
        logger.error(f"Error handling secure message from {client_id}: {e}")
        await send_error_message(websocket, "MESSAGE_HANDLING_ERROR", "Error processing message")


# ---------------------------------------------------------------------------
# REST endpoints for triggering WebSocket events
# ---------------------------------------------------------------------------

@router.post("/trigger/alert")
async def trigger_alert(client_id: str, alert_type: str, message: str):
    """Trigger an alert for a specific client."""
    alert = {
        "alert_type": alert_type,
        "message": message,
        "severity": "info"
    }

    await send_alert(client_id, alert)

    return {"status": "Alert sent", "client_id": client_id}


@router.post("/trigger/news")
async def trigger_news_broadcast(headline: str, summary: str, symbol: Optional[str] = None):
    """Broadcast news to all clients."""
    news = {
        "headline": headline,
        "summary": summary,
        "symbol": symbol,
        "source": "Internal"
    }

    await broadcast_news(news)

    return {"status": "News broadcast sent"}


@router.get("/connections")
async def get_active_connections():
    """Get information about active WebSocket connections."""
    return {
        "total_connections": len(manager.active_connections),
        "clients": list(manager.active_connections.keys()),
        "subscriptions": {
            client_id: list(symbols)
            for client_id, symbols in manager.subscriptions.items()
        },
        "active_streams": list(active_price_streams.keys())
    }
