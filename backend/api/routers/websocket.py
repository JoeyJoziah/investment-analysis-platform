from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, Depends, HTTPException, status
from typing import Dict, List, Set, Optional, Any
import json
import asyncio
import random
from datetime import datetime, timezone
from enum import Enum
import logging
from sqlalchemy.ext.asyncio import AsyncSession
import uuid

# Enhanced imports for real functionality
from backend.config.database import get_async_db_session
from backend.repositories import stock_repository, price_repository, portfolio_repository
# from backend.utils.enhanced_error_handling import handle_websocket_error
from backend.auth.oauth2 import get_current_user_from_token
from backend.models.unified_models import User
from backend.utils.cache import get_redis
from backend.config.settings import settings

# Security imports - temporarily commented out due to missing dependencies
# from backend.security.websocket_security import (
#     get_websocket_security, secure_websocket, WebSocketSecurityManager,
#     WebSocketClient, WebSocketMessageType, send_error_message,
#     validate_subscription_permissions
# )
# from backend.security.enhanced_auth import UserRole
# from backend.security.audit_logging import get_audit_logger, AuditEventType, AuditSeverity

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ws", tags=["websocket"])

# Enhanced WebSocket connection manager with error handling and persistence
class EnhancedConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, Dict[str, Any]] = {}  # client_id -> connection info
        self.subscriptions: Dict[str, Set[str]] = {}  # client_id -> set of symbols
        self.user_sessions: Dict[str, Dict[str, Any]] = {}  # client_id -> user session info
        self.connection_health: Dict[str, datetime] = {}  # client_id -> last heartbeat
        self.redis_client = None
        
    async def initialize(self):
        """Initialize Redis connection for persistence"""
        try:
            self.redis_client = await get_redis()
            logger.info("WebSocket manager initialized with Redis persistence")
        except Exception as e:
            logger.warning(f"Redis not available for WebSocket persistence: {e}")
    
    async def connect(self, websocket: WebSocket, client_id: str, user: Optional[User] = None):
        """Connect a WebSocket with enhanced error handling"""
        try:
            await websocket.accept()
            
            # Store connection info
            self.active_connections[client_id] = {
                'websocket': websocket,
                'connected_at': datetime.utcnow(),
                'user_id': user.id if user else None,
                'message_count': 0
            }
            
            # Store user session info
            if user:
                self.user_sessions[client_id] = {
                    'user_id': user.id,
                    'username': user.username,
                    'role': user.role
                }
            
            # Initialize health tracking
            self.connection_health[client_id] = datetime.utcnow()
            
            # Persist connection info to Redis if available
            if self.redis_client:
                try:
                    await self.redis_client.hset(
                        "websocket:connections",
                        client_id,
                        json.dumps({
                            'connected_at': datetime.utcnow().isoformat(),
                            'user_id': user.id if user else None
                        })
                    )
                except Exception as e:
                    logger.error(f"Error persisting connection info: {e}")
            
            logger.info(f"Client {client_id} connected (User: {user.username if user else 'Anonymous'}). Total connections: {len(self.active_connections)}")
            
        except Exception as e:
            logger.error(f"Error connecting client {client_id}: {e}")
            raise
    
    async def disconnect(self, websocket: WebSocket, client_id: str):
        """Disconnect a WebSocket with cleanup"""
        try:
            # Remove from active connections
            if client_id in self.active_connections:
                connection_info = self.active_connections[client_id]
                logger.info(f"Client {client_id} disconnected after {connection_info.get('message_count', 0)} messages")
                del self.active_connections[client_id]
            
            # Clean up user session
            if client_id in self.user_sessions:
                del self.user_sessions[client_id]
            
            # Clean up subscriptions
            if client_id in self.subscriptions:
                del self.subscriptions[client_id]
                
            # Clean up health tracking
            if client_id in self.connection_health:
                del self.connection_health[client_id]
            
            # Remove from Redis if available
            if self.redis_client:
                try:
                    await self.redis_client.hdel("websocket:connections", client_id)
                    await self.redis_client.hdel("websocket:subscriptions", client_id)
                except Exception as e:
                    logger.error(f"Error cleaning up Redis data: {e}")
            
            logger.info(f"Client {client_id} disconnected. Total connections: {len(self.active_connections)}")
            
        except Exception as e:
            logger.error(f"Error disconnecting client {client_id}: {e}")
    
    async def send_personal_message(self, message: str, client_id: str) -> bool:
        """Send message to specific client with error handling"""
        try:
            if client_id not in self.active_connections:
                logger.warning(f"Client {client_id} not found for personal message")
                return False
                
            websocket = self.active_connections[client_id]['websocket']
            await websocket.send_text(message)
            
            # Update message count
            self.active_connections[client_id]['message_count'] += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending personal message to {client_id}: {e}")
            # Remove disconnected client
            await self.disconnect(None, client_id)
            return False
    
    async def broadcast(self, message: str, exclude: Optional[str] = None, target_role: Optional[str] = None):
        """Broadcast message with role filtering and error handling"""
        successful_sends = 0
        failed_clients = []
        
        for client_id, connection_info in self.active_connections.items():
            if client_id == exclude:
                continue
                
            # Filter by role if specified
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
        
        # Clean up failed connections
        for client_id in failed_clients:
            await self.disconnect(None, client_id)
        
        logger.debug(f"Broadcast sent to {successful_sends} clients, {len(failed_clients)} failed")
        return successful_sends
    
    async def subscribe(self, client_id: str, symbols: List[str], db_session: Optional[AsyncSession] = None):
        """Subscribe client to symbols with validation"""
        try:
            if client_id not in self.subscriptions:
                self.subscriptions[client_id] = set()
            
            # Validate symbols if database session available
            if db_session:
                valid_symbols = []
                for symbol in symbols:
                    stock = await stock_repository.get_by_symbol(symbol.upper(), session=db_session)
                    if stock:
                        valid_symbols.append(symbol.upper())
                    else:
                        logger.warning(f"Invalid symbol for subscription: {symbol}")
                
                symbols = valid_symbols
            
            self.subscriptions[client_id].update(symbols)
            
            # Persist to Redis
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
        """Unsubscribe client from symbols"""
        if client_id in self.subscriptions:
            self.subscriptions[client_id].difference_update(symbols)
            logger.info(f"Client {client_id} unsubscribed from {len(symbols)} symbols")
    
    def get_subscriptions(self, client_id: str) -> Set[str]:
        """Get client's subscriptions"""
        return self.subscriptions.get(client_id, set())
    
    async def send_to_subscribers(self, symbol: str, message: str, exclude_client: Optional[str] = None) -> int:
        """Send message to all subscribers of a symbol"""
        sent_count = 0
        failed_clients = []
        
        for client_id, symbols in self.subscriptions.items():
            if symbol in symbols and client_id != exclude_client and client_id in self.active_connections:
                try:
                    websocket = self.active_connections[client_id]['websocket']
                    await websocket.send_text(message)
                    self.active_connections[client_id]['message_count'] += 1
                    sent_count += 1
                    
                except Exception as e:
                    logger.error(f"Error sending to subscriber {client_id}: {e}")
                    failed_clients.append(client_id)
        
        # Clean up failed connections
        for client_id in failed_clients:
            await self.disconnect(None, client_id)
        
        return sent_count
    
    async def update_health(self, client_id: str):
        """Update client health status"""
        self.connection_health[client_id] = datetime.utcnow()
    
    async def cleanup_stale_connections(self, max_age_minutes: int = 30):
        """Clean up stale connections"""
        cutoff_time = datetime.utcnow() - timedelta(minutes=max_age_minutes)
        stale_clients = []
        
        for client_id, last_heartbeat in self.connection_health.items():
            if last_heartbeat < cutoff_time:
                stale_clients.append(client_id)
        
        for client_id in stale_clients:
            logger.info(f"Cleaning up stale connection: {client_id}")
            await self.disconnect(None, client_id)
        
        return len(stale_clients)

# Message types
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

# Create enhanced connection manager instance
manager = EnhancedConnectionManager()

# Initialize manager on startup
async def initialize_websocket_manager():
    """Initialize the WebSocket manager"""
    await manager.initialize()

# Background task to cleanup stale connections
async def cleanup_stale_connections_task():
    """Background task to clean up stale WebSocket connections"""
    while True:
        try:
            cleaned = await manager.cleanup_stale_connections()
            if cleaned > 0:
                logger.info(f"Cleaned up {cleaned} stale WebSocket connections")
            await asyncio.sleep(300)  # Run every 5 minutes
        except Exception as e:
            logger.error(f"Error in cleanup task: {e}")
            await asyncio.sleep(60)

# Start cleanup task
cleanup_task = asyncio.create_task(cleanup_stale_connections_task())

# Data structures for real-time data
active_price_streams: Dict[str, asyncio.Task] = {}
market_data_stream: Optional[asyncio.Task] = None

# Secure WebSocket endpoints
@router.websocket("/stream")
@secure_websocket(require_auth=False, allowed_roles=None)
async def websocket_endpoint(
    websocket: WebSocket,
    client_id: str = Query(default_factory=lambda: str(uuid.uuid4())),
    token: Optional[str] = Query(None),
    security_manager: WebSocketSecurityManager = None,
    client: WebSocketClient = None
):
    """Main WebSocket endpoint for real-time data streaming with security"""
    
    audit_logger = get_audit_logger()
    
    try:
        # Send welcome message with security info
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
        
        # Connect to legacy manager for backwards compatibility
        await manager.connect(websocket, client_id, client.user_session)
        
        try:
            while True:
                # Receive message from client
                raw_data = await websocket.receive_text()
                
                # Validate message through security manager
                is_valid, message, error = await security_manager.validate_message(
                    client_id, raw_data
                )
                
                if not is_valid:
                    await send_error_message(websocket, "VALIDATION_FAILED", error)
                    continue
                
                # Handle different message types securely
                await handle_secure_client_message(
                    websocket, client_id, message, security_manager, client
                )
                
        except WebSocketDisconnect:
            logger.info(f"WebSocket client {client_id} disconnected normally")
        except Exception as e:
            logger.error(f"WebSocket error for client {client_id}: {e}")
            await send_error_message(websocket, "INTERNAL_ERROR", "Connection error occurred")
        
    finally:
        # Clean up legacy manager connection
        await manager.disconnect(websocket, client_id)
        
        # Clean up any active streams for this client
        await cleanup_client_streams(client_id)
        
        # Security manager will handle cleanup automatically

@router.websocket("/market")
async def market_data_stream_endpoint(websocket: WebSocket):
    """Dedicated WebSocket for market-wide data streaming"""
    
    await websocket.accept()
    
    try:
        while True:
            # Send market overview data
            market_data = {
                "type": "market_overview",
                "timestamp": datetime.utcnow().isoformat(),
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
            
            await websocket.send_json(market_data)
            await asyncio.sleep(5)  # Update every 5 seconds
            
    except WebSocketDisconnect:
        print("Market data stream client disconnected")

@router.websocket("/portfolio/{portfolio_id}")
async def portfolio_stream(websocket: WebSocket, portfolio_id: str):
    """WebSocket for portfolio-specific updates"""
    
    await websocket.accept()
    
    try:
        while True:
            # Send portfolio updates
            portfolio_update = {
                "type": "portfolio_update",
                "portfolio_id": portfolio_id,
                "timestamp": datetime.utcnow().isoformat(),
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
            
            await websocket.send_json(portfolio_update)
            await asyncio.sleep(3)  # Update every 3 seconds
            
    except WebSocketDisconnect:
        print(f"Portfolio stream for {portfolio_id} disconnected")

# Helper functions
async def handle_secure_client_message(
    websocket: WebSocket, 
    client_id: str, 
    message: Dict[str, Any],
    security_manager: WebSocketSecurityManager,
    client: WebSocketClient
):
    """Handle incoming messages from clients with security validation"""
    
    audit_logger = get_audit_logger()
    msg_type = message.get("type")
    
    try:
        if msg_type == WebSocketMessageType.AUTHENTICATE.value:
            # Handle authentication request
            token = message.get("token")
            if token:
                # Re-authenticate with new token
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
            # Subscribe to symbols with security validation
            symbols = message.get("symbols", [])
            
            # Validate subscription permissions
            allowed_symbols, denied_symbols = await validate_subscription_permissions(client, symbols)
            
            if denied_symbols:
                await security_manager.send_secure_message(client_id, {
                    "type": WebSocketMessageType.ERROR.value,
                    "message": f"Access denied to symbols: {denied_symbols}",
                    "code": "SUBSCRIPTION_DENIED"
                })
            
            if allowed_symbols:
                # Subscribe through legacy manager
                validated_symbols = await manager.subscribe(client_id, allowed_symbols)
                
                # Update client subscriptions
                client.subscriptions.update(validated_symbols)
                client.subscription_count = len(client.subscriptions)
                
                # Start price streams for subscribed symbols
                for symbol in validated_symbols:
                    if symbol not in active_price_streams:
                        active_price_streams[symbol] = asyncio.create_task(
                            stream_price_updates(symbol)
                        )
                
                # Log subscription
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
            # Unsubscribe from symbols
            symbols = message.get("symbols", [])
            
            # Remove from legacy manager
            manager.unsubscribe(client_id, symbols)
            
            # Update client subscriptions
            client.subscriptions.difference_update(symbols)
            client.subscription_count = len(client.subscriptions)
            
            await security_manager.send_secure_message(client_id, {
                "type": WebSocketMessageType.SYSTEM.value,
                "message": f"Unsubscribed from {len(symbols)} symbols",
                "symbols": symbols
            })
        
        elif msg_type == WebSocketMessageType.HEARTBEAT.value:
            # Handle heartbeat
            client.update_activity()
            
            await security_manager.send_secure_message(client_id, {
                "type": WebSocketMessageType.HEARTBEAT.value,
                "message": "pong",
                "server_time": datetime.now(timezone.utc).timestamp()
            })
        
        elif msg_type == WebSocketMessageType.CHAT.value:
            # Handle chat messages (only for authenticated users)
            if not client.is_authenticated:
                await send_error_message(websocket, "AUTH_REQUIRED", "Authentication required for chat")
                return
            
            chat_content = message.get("message", "").strip()
            if not chat_content or len(chat_content) > 500:
                await send_error_message(websocket, "INVALID_MESSAGE", "Invalid chat message")
                return
            
            # Broadcast chat message with user context
            chat_message = {
                "type": WebSocketMessageType.CHAT.value,
                "from": client.user_session.username if client.user_session else client_id,
                "user_id": client.user_session.user_id if client.user_session else None,
                "message": chat_content,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            sent_count = await security_manager.broadcast_secure_message(chat_message, exclude_client=client_id)
            
            # Log chat message
            await audit_logger.log_event(
                AuditEventType.DATA_CREATE,
                user_id=client.user_session.user_id if client.user_session else None,
                ip_address=client.ip_address,
                action="websocket_chat",
                severity=AuditSeverity.LOW,
                details={"client_id": client_id, "recipients": sent_count, "message_length": len(chat_content)}
            )
        
        else:
            # Unknown message type
            await send_error_message(websocket, "UNKNOWN_MESSAGE_TYPE", f"Unknown message type: {msg_type}")
    
    except Exception as e:
        logger.error(f"Error handling secure message from {client_id}: {e}")
        await send_error_message(websocket, "MESSAGE_HANDLING_ERROR", "Error processing message")


async def handle_client_message(websocket: WebSocket, client_id: str, message: Dict[str, Any]):
    """Legacy message handler for backwards compatibility"""
    
    msg_type = message.get("type")
    
    if msg_type == MessageType.SUBSCRIBE:
        # Subscribe to symbols
        symbols = message.get("symbols", [])
        await manager.subscribe(client_id, symbols)
        
        # Start price streams for subscribed symbols
        for symbol in symbols:
            if symbol not in active_price_streams:
                active_price_streams[symbol] = asyncio.create_task(
                    stream_price_updates(symbol)
                )
        
        await websocket.send_json({
            "type": MessageType.SYSTEM,
            "message": f"Subscribed to {symbols}",
            "symbols": symbols,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    elif msg_type == MessageType.UNSUBSCRIBE:
        # Unsubscribe from symbols
        symbols = message.get("symbols", [])
        manager.unsubscribe(client_id, symbols)
        
        await websocket.send_json({
            "type": MessageType.SYSTEM,
            "message": f"Unsubscribed from {symbols}",
            "symbols": symbols,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    elif msg_type == MessageType.CHAT:
        # Broadcast chat message to all connected clients
        chat_message = {
            "type": MessageType.CHAT,
            "from": client_id,
            "message": message.get("message", ""),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        await manager.broadcast(json.dumps(chat_message))
    
    else:
        # Echo back for unknown message types
        await websocket.send_json({
            "type": MessageType.ERROR,
            "message": f"Unknown message type: {msg_type}",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

async def stream_price_updates(symbol: str):
    """Stream real-time price updates for a symbol"""
    
    while True:
        try:
            # Generate random price update
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
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Send to all subscribers of this symbol
            await manager.send_to_subscribers(symbol, json.dumps(price_update))
            
            # Random delay between updates (simulate market activity)
            await asyncio.sleep(random.uniform(0.5, 3))
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"Error streaming price for {symbol}: {e}")
            await asyncio.sleep(5)

async def send_heartbeat(websocket: WebSocket, client_id: str):
    """Send periodic heartbeat to keep connection alive"""
    
    while True:
        try:
            await asyncio.sleep(30)  # Send heartbeat every 30 seconds
            
            heartbeat = {
                "type": MessageType.HEARTBEAT,
                "timestamp": datetime.utcnow().isoformat(),
                "server_time": datetime.utcnow().timestamp()
            }
            
            await websocket.send_json(heartbeat)
            
        except asyncio.CancelledError:
            break
        except Exception:
            break

async def cleanup_client_streams(client_id: str):
    """Clean up resources when client disconnects"""
    
    # Get client's subscriptions
    subscriptions = manager.get_subscriptions(client_id)
    
    # Check if any symbols are still subscribed by other clients
    for symbol in subscriptions:
        still_subscribed = False
        for other_client_id, other_subs in manager.subscriptions.items():
            if other_client_id != client_id and symbol in other_subs:
                still_subscribed = True
                break
        
        # Cancel stream if no other clients are subscribed
        if not still_subscribed and symbol in active_price_streams:
            active_price_streams[symbol].cancel()
            del active_price_streams[symbol]

async def send_alert(client_id: str, alert: Dict[str, Any]):
    """Send alert to specific client"""
    
    alert_message = {
        "type": MessageType.ALERT,
        "alert": alert,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    await manager.send_personal_message(json.dumps(alert_message), client_id)

async def broadcast_news(news: Dict[str, Any]):
    """Broadcast news to all connected clients"""
    
    news_message = {
        "type": MessageType.NEWS,
        "news": news,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    await manager.broadcast(json.dumps(news_message))

# API endpoints for triggering WebSocket events
@router.post("/trigger/alert")
async def trigger_alert(client_id: str, alert_type: str, message: str):
    """Trigger an alert for a specific client"""
    
    alert = {
        "alert_type": alert_type,
        "message": message,
        "severity": "info"
    }
    
    await send_alert(client_id, alert)
    
    return {"status": "Alert sent", "client_id": client_id}

@router.post("/trigger/news")
async def trigger_news_broadcast(headline: str, summary: str, symbol: Optional[str] = None):
    """Broadcast news to all clients"""
    
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
    """Get information about active WebSocket connections"""
    
    return {
        "total_connections": len(manager.active_connections),
        "clients": list(manager.active_connections.keys()),
        "subscriptions": {
            client_id: list(symbols) 
            for client_id, symbols in manager.subscriptions.items()
        },
        "active_streams": list(active_price_streams.keys())
    }