from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, Depends
from typing import Dict, List, Set, Optional, Any
import json
import asyncio
import random
from datetime import datetime
from enum import Enum

router = APIRouter(prefix="/ws", tags=["websocket"])

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
        self.subscriptions: Dict[str, Set[str]] = {}
        self.user_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        if client_id not in self.active_connections:
            self.active_connections[client_id] = []
        self.active_connections[client_id].append(websocket)
        self.user_connections[client_id] = websocket
        print(f"Client {client_id} connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket, client_id: str):
        if client_id in self.active_connections:
            self.active_connections[client_id].remove(websocket)
            if not self.active_connections[client_id]:
                del self.active_connections[client_id]
        
        if client_id in self.user_connections:
            del self.user_connections[client_id]
        
        if client_id in self.subscriptions:
            del self.subscriptions[client_id]
        
        print(f"Client {client_id} disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: str, client_id: str):
        if client_id in self.user_connections:
            websocket = self.user_connections[client_id]
            await websocket.send_text(message)
    
    async def broadcast(self, message: str, exclude: Optional[str] = None):
        for client_id, connections in self.active_connections.items():
            if client_id != exclude:
                for connection in connections:
                    try:
                        await connection.send_text(message)
                    except:
                        pass
    
    def subscribe(self, client_id: str, symbols: List[str]):
        if client_id not in self.subscriptions:
            self.subscriptions[client_id] = set()
        self.subscriptions[client_id].update(symbols)
    
    def unsubscribe(self, client_id: str, symbols: List[str]):
        if client_id in self.subscriptions:
            self.subscriptions[client_id].difference_update(symbols)
    
    def get_subscriptions(self, client_id: str) -> Set[str]:
        return self.subscriptions.get(client_id, set())
    
    async def send_to_subscribers(self, symbol: str, message: str):
        for client_id, symbols in self.subscriptions.items():
            if symbol in symbols and client_id in self.user_connections:
                try:
                    await self.user_connections[client_id].send_text(message)
                except:
                    pass

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

# Create connection manager instance
manager = ConnectionManager()

# Data structures for real-time data
active_price_streams: Dict[str, asyncio.Task] = {}
market_data_stream: Optional[asyncio.Task] = None

# Endpoints
@router.websocket("/stream")
async def websocket_endpoint(
    websocket: WebSocket,
    client_id: str = Query(...),
    token: Optional[str] = Query(None)
):
    """Main WebSocket endpoint for real-time data streaming"""
    
    await manager.connect(websocket, client_id)
    
    # Send welcome message
    await websocket.send_json({
        "type": MessageType.SYSTEM,
        "message": "Connected to real-time stream",
        "timestamp": datetime.utcnow().isoformat(),
        "client_id": client_id
    })
    
    # Start heartbeat
    heartbeat_task = asyncio.create_task(send_heartbeat(websocket, client_id))
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle different message types
            await handle_client_message(websocket, client_id, message)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket, client_id)
        heartbeat_task.cancel()
        
        # Clean up any active streams for this client
        await cleanup_client_streams(client_id)
        
        # Notify others that user disconnected (for chat, etc.)
        await manager.broadcast(
            json.dumps({
                "type": MessageType.SYSTEM,
                "message": f"User {client_id} disconnected",
                "timestamp": datetime.utcnow().isoformat()
            }),
            exclude=client_id
        )

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
async def handle_client_message(websocket: WebSocket, client_id: str, message: Dict[str, Any]):
    """Handle incoming messages from clients"""
    
    msg_type = message.get("type")
    
    if msg_type == MessageType.SUBSCRIBE:
        # Subscribe to symbols
        symbols = message.get("symbols", [])
        manager.subscribe(client_id, symbols)
        
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
            "timestamp": datetime.utcnow().isoformat()
        })
    
    elif msg_type == MessageType.UNSUBSCRIBE:
        # Unsubscribe from symbols
        symbols = message.get("symbols", [])
        manager.unsubscribe(client_id, symbols)
        
        await websocket.send_json({
            "type": MessageType.SYSTEM,
            "message": f"Unsubscribed from {symbols}",
            "symbols": symbols,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    elif msg_type == MessageType.CHAT:
        # Broadcast chat message to all connected clients
        chat_message = {
            "type": MessageType.CHAT,
            "from": client_id,
            "message": message.get("message", ""),
            "timestamp": datetime.utcnow().isoformat()
        }
        await manager.broadcast(json.dumps(chat_message))
    
    else:
        # Echo back for unknown message types
        await websocket.send_json({
            "type": MessageType.ERROR,
            "message": f"Unknown message type: {msg_type}",
            "timestamp": datetime.utcnow().isoformat()
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