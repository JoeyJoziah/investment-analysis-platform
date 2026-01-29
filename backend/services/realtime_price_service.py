"""
Real-time Price Service for Portfolio Updates

Integrates with Finnhub WebSocket API for real-time price data and
provides bulk price fetching for portfolio symbols.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Set, Any
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
import aiohttp
from enum import Enum

from backend.config.settings import settings
from backend.utils.cache import get_redis
from backend.repositories import price_repository
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class PriceUpdateType(str, Enum):
    TRADE = "trade"
    QUOTE = "quote"
    ERROR = "error"


@dataclass
class PriceUpdate:
    """Represents a real-time price update"""
    symbol: str
    price: float
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    timestamp: datetime
    volume: Optional[int] = None
    change: Optional[float] = None
    change_percent: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    open: Optional[float] = None
    close: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class FinnhubWebSocketClient:
    """
    Manages WebSocket connection to Finnhub for real-time price data.
    Handles reconnection, subscription management, and message parsing.
    """

    def __init__(self, api_key: str):
        """
        Initialize Finnhub WebSocket client.

        Args:
            api_key: Finnhub API key for authentication
        """
        self.api_key = api_key
        self.ws_url = "wss://ws.finnhub.io?token=" + api_key
        self.websocket = None
        self.subscriptions: Set[str] = set()
        self.price_callbacks: Dict[str, List[callable]] = {}
        self.error_callbacks: List[callable] = []
        self.connection_active = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.reconnect_delay = 5  # Start with 5 seconds
        self._receive_task: Optional[asyncio.Task] = None

    async def connect(self):
        """Establish WebSocket connection to Finnhub"""
        try:
            logger.info("Connecting to Finnhub WebSocket...")

            async with aiohttp.ClientSession() as session:
                self.websocket = await session.ws_connect(
                    self.ws_url,
                    heartbeat=30,  # Send heartbeat every 30 seconds
                    timeout=aiohttp.ClientTimeout(total=60)
                )

                self.connection_active = True
                self.reconnect_attempts = 0
                logger.info("Connected to Finnhub WebSocket")

                # Start receive loop
                self._receive_task = asyncio.create_task(self._receive_loop())

        except Exception as e:
            logger.error(f"Failed to connect to Finnhub: {e}")
            await self._handle_connection_error()

    async def _receive_loop(self):
        """Main receive loop for WebSocket messages"""
        try:
            async for msg in self.websocket:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    await self._handle_message(msg.data)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {msg}")
                    break
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    break
        except asyncio.CancelledError:
            logger.info("WebSocket receive loop cancelled")
        except Exception as e:
            logger.error(f"Error in receive loop: {e}")
            await self._handle_connection_error()

    async def _handle_message(self, data: str):
        """Parse and handle incoming WebSocket message"""
        try:
            message = json.loads(data)

            # Handle trade messages
            if 'type' in message and message['type'] == 'trade':
                for trade in message.get('data', []):
                    symbol = trade.get('s')
                    if symbol and symbol in self.subscriptions:
                        update = self._parse_trade_update(trade)
                        await self._invoke_callbacks(symbol, update)

            # Handle quote messages
            elif 'type' in message and message['type'] == 'quote':
                quote = message.get('data', {})
                symbol = quote.get('s')
                if symbol and symbol in self.subscriptions:
                    update = self._parse_quote_update(quote)
                    await self._invoke_callbacks(symbol, update)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse WebSocket message: {e}")
        except Exception as e:
            logger.error(f"Error handling message: {e}")

    def _parse_trade_update(self, trade: Dict) -> PriceUpdate:
        """Parse trade data from Finnhub"""
        symbol = trade.get('s', '')
        price = trade.get('p', 0.0)
        timestamp_ms = trade.get('t', 0)
        volume = trade.get('v', 0)

        timestamp = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)

        return PriceUpdate(
            symbol=symbol,
            price=float(price),
            bid=float(trade.get('bp', price)),
            ask=float(trade.get('ap', price)),
            bid_size=int(trade.get('bv', 0)),
            ask_size=int(trade.get('av', 0)),
            timestamp=timestamp,
            volume=volume,
            close=float(price)
        )

    def _parse_quote_update(self, quote: Dict) -> PriceUpdate:
        """Parse quote data from Finnhub"""
        symbol = quote.get('s', '')
        bid = float(quote.get('b', 0.0))
        ask = float(quote.get('a', 0.0))
        mid_price = (bid + ask) / 2 if bid and ask else 0.0

        timestamp_ms = quote.get('t', int(datetime.now(timezone.utc).timestamp() * 1000))
        timestamp = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)

        return PriceUpdate(
            symbol=symbol,
            price=float(mid_price or quote.get('c', 0.0)),
            bid=bid,
            ask=ask,
            bid_size=int(quote.get('bv', 0)),
            ask_size=int(quote.get('av', 0)),
            timestamp=timestamp,
            high=float(quote.get('h', 0.0)),
            low=float(quote.get('l', 0.0)),
            open=float(quote.get('o', 0.0)),
            close=float(quote.get('c', 0.0))
        )

    async def _invoke_callbacks(self, symbol: str, update: PriceUpdate):
        """Invoke registered callbacks for a symbol's price update"""
        if symbol in self.price_callbacks:
            for callback in self.price_callbacks[symbol]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(update)
                    else:
                        callback(update)
                except Exception as e:
                    logger.error(f"Error invoking callback for {symbol}: {e}")

    async def subscribe(self, symbol: str, callback: Optional[callable] = None):
        """
        Subscribe to price updates for a symbol.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            callback: Optional callback function to invoke on updates
        """
        if symbol not in self.subscriptions:
            self.subscriptions.add(symbol)

            # Send subscription message
            if self.websocket and self.connection_active:
                try:
                    await self.websocket.send_json({
                        'type': 'subscribe',
                        'symbol': symbol
                    })
                    logger.debug(f"Subscribed to {symbol}")
                except Exception as e:
                    logger.error(f"Failed to subscribe to {symbol}: {e}")

        # Register callback
        if callback:
            if symbol not in self.price_callbacks:
                self.price_callbacks[symbol] = []
            self.price_callbacks[symbol].append(callback)

    async def unsubscribe(self, symbol: str):
        """Unsubscribe from price updates for a symbol"""
        if symbol in self.subscriptions:
            self.subscriptions.discard(symbol)

            if self.websocket and self.connection_active:
                try:
                    await self.websocket.send_json({
                        'type': 'unsubscribe',
                        'symbol': symbol
                    })
                    logger.debug(f"Unsubscribed from {symbol}")
                except Exception as e:
                    logger.error(f"Failed to unsubscribe from {symbol}: {e}")

        # Remove callbacks
        if symbol in self.price_callbacks:
            del self.price_callbacks[symbol]

    async def _handle_connection_error(self):
        """Handle connection errors and attempt reconnection"""
        self.connection_active = False

        if self.reconnect_attempts < self.max_reconnect_attempts:
            self.reconnect_attempts += 1
            delay = min(self.reconnect_delay * (2 ** self.reconnect_attempts), 300)  # Max 5 min
            logger.warning(f"Reconnecting in {delay} seconds (attempt {self.reconnect_attempts})")

            await asyncio.sleep(delay)
            await self.connect()
        else:
            logger.error("Max reconnection attempts reached")
            for callback in self.error_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback("Max reconnection attempts reached")
                    else:
                        callback("Max reconnection attempts reached")
                except Exception as e:
                    logger.error(f"Error invoking error callback: {e}")

    async def disconnect(self):
        """Close WebSocket connection"""
        if self._receive_task:
            self._receive_task.cancel()

        if self.websocket:
            await self.websocket.close()

        self.connection_active = False
        logger.info("Disconnected from Finnhub WebSocket")


class RealtimePriceService:
    """
    Service for managing real-time price updates for portfolio positions.

    Integrates with Finnhub WebSocket for live data and provides fallback
    to database prices for symbols not subscribed or during outages.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the real-time price service.

        Args:
            api_key: Finnhub API key (optional, uses settings if not provided)
        """
        self.api_key = api_key or settings.FINNHUB_API_KEY
        self.ws_client: Optional[FinnhubWebSocketClient] = None
        self.price_cache: Dict[str, PriceUpdate] = {}
        self.redis_client = None
        self.initialized = False

    async def initialize(self):
        """Initialize the service and connect to WebSocket"""
        if not self.api_key:
            logger.warning("No Finnhub API key configured, falling back to database prices")
            self.initialized = True
            return

        try:
            self.redis_client = await get_redis()
        except Exception as e:
            logger.warning(f"Redis not available: {e}")

        self.ws_client = FinnhubWebSocketClient(self.api_key)
        await self.ws_client.connect()
        self.initialized = True

    async def get_latest_price(self, symbol: str, db: Optional[AsyncSession] = None) -> Optional[PriceUpdate]:
        """
        Get latest price for a symbol.

        First checks in-memory cache, then WebSocket subscription,
        then database fallback.

        Args:
            symbol: Stock symbol
            db: Database session for fallback

        Returns:
            PriceUpdate with latest data or None if unavailable
        """
        # Check in-memory cache
        if symbol in self.price_cache:
            return self.price_cache[symbol]

        # Check Redis cache
        if self.redis_client:
            try:
                cached = await self.redis_client.get(f"price:{symbol}")
                if cached:
                    data = json.loads(cached)
                    data['timestamp'] = datetime.fromisoformat(data['timestamp'])
                    return PriceUpdate(**data)
            except Exception as e:
                logger.debug(f"Error reading Redis price cache: {e}")

        # Try database as fallback
        if db:
            try:
                latest_price = await price_repository.get_latest_price(symbol, session=db)
                if latest_price:
                    return PriceUpdate(
                        symbol=symbol,
                        price=float(latest_price.close),
                        bid=float(latest_price.close * 0.999),
                        ask=float(latest_price.close * 1.001),
                        bid_size=100,
                        ask_size=100,
                        timestamp=latest_price.timestamp,
                        volume=latest_price.volume,
                        high=float(latest_price.high),
                        low=float(latest_price.low),
                        open=float(latest_price.open),
                        close=float(latest_price.close)
                    )
            except Exception as e:
                logger.error(f"Error fetching price from database: {e}")

        return None

    async def get_latest_prices_bulk(
        self,
        symbols: List[str],
        db: Optional[AsyncSession] = None
    ) -> Dict[str, PriceUpdate]:
        """
        Get latest prices for multiple symbols.

        Efficiently fetches prices from all available sources.

        Args:
            symbols: List of stock symbols
            db: Database session for fallback

        Returns:
            Dictionary mapping symbols to PriceUpdate objects
        """
        prices: Dict[str, PriceUpdate] = {}

        # Fetch from cache/WebSocket first
        for symbol in symbols:
            price = await self.get_latest_price(symbol, db)
            if price:
                prices[symbol] = price

        return prices

    async def subscribe_to_symbol(
        self,
        symbol: str,
        callback: callable,
        db: Optional[AsyncSession] = None
    ):
        """
        Subscribe to real-time updates for a symbol.

        Args:
            symbol: Stock symbol
            callback: Async function to call on price updates
            db: Database session for initial price
        """
        if not self.initialized:
            await self.initialize()

        # Subscribe to WebSocket if available
        if self.ws_client and self.api_key:
            async def ws_callback(update: PriceUpdate):
                # Cache the update
                self.price_cache[symbol] = update

                # Cache in Redis if available
                if self.redis_client:
                    try:
                        await self.redis_client.setex(
                            f"price:{symbol}",
                            300,  # 5 minute TTL
                            json.dumps(update.to_dict())
                        )
                    except Exception as e:
                        logger.debug(f"Error caching price in Redis: {e}")

                # Invoke callback
                await callback(update)

            await self.ws_client.subscribe(symbol, ws_callback)
        else:
            # Fallback to periodic database polling
            async def db_callback():
                while True:
                    try:
                        price = await self.get_latest_price(symbol, db)
                        if price:
                            await callback(price)
                        await asyncio.sleep(10)  # Poll every 10 seconds
                    except asyncio.CancelledError:
                        break
                    except Exception as e:
                        logger.error(f"Error in database callback: {e}")
                        await asyncio.sleep(10)

            asyncio.create_task(db_callback())

    async def unsubscribe_from_symbol(self, symbol: str):
        """Unsubscribe from real-time updates for a symbol"""
        if self.ws_client:
            await self.ws_client.unsubscribe(symbol)

        # Clear cache
        if symbol in self.price_cache:
            del self.price_cache[symbol]

    async def shutdown(self):
        """Shutdown the service and close connections"""
        if self.ws_client:
            await self.ws_client.disconnect()

        self.initialized = False
        logger.info("RealtimePriceService shut down")


# Global service instance
_realtime_price_service: Optional[RealtimePriceService] = None


async def get_realtime_price_service() -> RealtimePriceService:
    """Get or create the global RealtimePriceService instance"""
    global _realtime_price_service

    if _realtime_price_service is None:
        _realtime_price_service = RealtimePriceService()
        await _realtime_price_service.initialize()

    return _realtime_price_service


async def shutdown_realtime_price_service():
    """Shutdown the global RealtimePriceService instance"""
    global _realtime_price_service

    if _realtime_price_service:
        await _realtime_price_service.shutdown()
        _realtime_price_service = None
