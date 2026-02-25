"""
Unit tests for backend/services/realtime_price_service.py

Tests cover:
- PriceUpdate: dataclass construction, to_dict serialization, optional fields
- PriceUpdateType: enum members and values
- FinnhubWebSocketClient: init, subscribe/unsubscribe, message parsing,
  callback invocation, connection error handling, reconnect logic, disconnect
- RealtimePriceService: init, initialize, get_latest_price (cache/redis/db),
  get_latest_prices_bulk, subscribe_to_symbol, unsubscribe_from_symbol, shutdown
- Module-level helpers: get_realtime_price_service, shutdown_realtime_price_service
"""

import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from backend.services.realtime_price_service import (
    FinnhubWebSocketClient,
    PriceUpdate,
    PriceUpdateType,
    RealtimePriceService,
    get_realtime_price_service,
    shutdown_realtime_price_service,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_price_update(**overrides) -> PriceUpdate:
    """Create a PriceUpdate with sensible defaults."""
    defaults = dict(
        symbol="AAPL",
        price=150.0,
        bid=149.90,
        ask=150.10,
        bid_size=100,
        ask_size=200,
        timestamp=datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
    )
    defaults.update(overrides)
    return PriceUpdate(**defaults)


def _make_db_price_row(close=152.5, high=155.0, low=148.0, open_=150.0,
                       volume=1_000_000, timestamp=None):
    """Return a namespace that mimics a price_repository row."""
    return SimpleNamespace(
        close=close,
        high=high,
        low=low,
        open=open_,
        volume=volume,
        timestamp=timestamp or datetime(2025, 6, 1, tzinfo=timezone.utc),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_settings():
    """Patch settings.FINNHUB_API_KEY for service construction."""
    with patch(
        "backend.services.realtime_price_service.settings"
    ) as mock_s:
        mock_s.FINNHUB_API_KEY = "test-api-key-123"
        yield mock_s


@pytest.fixture
def ws_client():
    """Fresh FinnhubWebSocketClient with a fake API key."""
    return FinnhubWebSocketClient(api_key="fake-key")


@pytest.fixture
def service(mock_settings):
    """Fresh RealtimePriceService with mocked settings."""
    return RealtimePriceService(api_key="test-key")


@pytest.fixture
def service_no_key(mock_settings):
    """RealtimePriceService with no API key (fallback mode)."""
    mock_settings.FINNHUB_API_KEY = None
    return RealtimePriceService(api_key=None)


# ===========================================================================
# PriceUpdateType enum
# ===========================================================================

class TestPriceUpdateType:

    def test_trade_value(self):
        assert PriceUpdateType.TRADE == "trade"

    def test_quote_value(self):
        assert PriceUpdateType.QUOTE == "quote"

    def test_error_value(self):
        assert PriceUpdateType.ERROR == "error"

    def test_is_string_enum(self):
        assert isinstance(PriceUpdateType.TRADE, str)


# ===========================================================================
# PriceUpdate dataclass
# ===========================================================================

class TestPriceUpdate:

    def test_construction_required_fields(self):
        update = _make_price_update()
        assert update.symbol == "AAPL"
        assert update.price == 150.0
        assert update.bid == 149.90
        assert update.ask == 150.10
        assert update.bid_size == 100
        assert update.ask_size == 200

    def test_optional_fields_default_none(self):
        update = _make_price_update()
        assert update.volume is None
        assert update.change is None
        assert update.change_percent is None
        assert update.high is None
        assert update.low is None
        assert update.open is None
        assert update.close is None

    def test_optional_fields_set(self):
        update = _make_price_update(
            volume=500_000, change=1.5, change_percent=1.01,
            high=155.0, low=148.0, open=149.0, close=150.0,
        )
        assert update.volume == 500_000
        assert update.change == 1.5
        assert update.change_percent == 1.01

    def test_to_dict_returns_dict(self):
        update = _make_price_update()
        result = update.to_dict()
        assert isinstance(result, dict)

    def test_to_dict_serializes_timestamp_as_iso(self):
        ts = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        update = _make_price_update(timestamp=ts)
        result = update.to_dict()
        assert result["timestamp"] == ts.isoformat()

    def test_to_dict_contains_all_keys(self):
        update = _make_price_update()
        result = update.to_dict()
        expected_keys = {
            "symbol", "price", "bid", "ask", "bid_size", "ask_size",
            "timestamp", "volume", "change", "change_percent",
            "high", "low", "open", "close",
        }
        assert set(result.keys()) == expected_keys

    def test_to_dict_is_json_serializable(self):
        update = _make_price_update(volume=100, high=155.0, low=148.0)
        data = update.to_dict()
        serialized = json.dumps(data)
        assert isinstance(serialized, str)
        roundtripped = json.loads(serialized)
        assert roundtripped["symbol"] == "AAPL"


# ===========================================================================
# FinnhubWebSocketClient -- init
# ===========================================================================

class TestFinnhubWebSocketClientInit:

    def test_default_state(self, ws_client):
        assert ws_client.api_key == "fake-key"
        assert ws_client.ws_url == "wss://ws.finnhub.io?token=fake-key"
        assert ws_client.websocket is None
        assert ws_client.subscriptions == set()
        assert ws_client.price_callbacks == {}
        assert ws_client.error_callbacks == []
        assert ws_client.connection_active is False
        assert ws_client.reconnect_attempts == 0
        assert ws_client.max_reconnect_attempts == 5
        assert ws_client.reconnect_delay == 5
        assert ws_client._receive_task is None


# ===========================================================================
# FinnhubWebSocketClient -- _parse_trade_update
# ===========================================================================

class TestParseTradeUpdate:

    def test_basic_trade_data(self, ws_client):
        trade = {
            "s": "AAPL",
            "p": 150.25,
            "t": 1717243200000,  # 2024-06-01 12:00 UTC
            "v": 500,
            "bp": 150.20,
            "ap": 150.30,
            "bv": 100,
            "av": 200,
        }
        update = ws_client._parse_trade_update(trade)
        assert update.symbol == "AAPL"
        assert update.price == 150.25
        assert update.bid == 150.20
        assert update.ask == 150.30
        assert update.bid_size == 100
        assert update.ask_size == 200
        assert update.volume == 500
        assert update.close == 150.25

    def test_missing_optional_fields_use_defaults(self, ws_client):
        trade = {"s": "TSLA", "p": 200.0, "t": 1717243200000}
        update = ws_client._parse_trade_update(trade)
        assert update.symbol == "TSLA"
        assert update.price == 200.0
        # bid/ask default to price when bp/ap missing
        assert update.bid == 200.0
        assert update.ask == 200.0
        assert update.bid_size == 0
        assert update.ask_size == 0

    def test_empty_trade_uses_safe_defaults(self, ws_client):
        update = ws_client._parse_trade_update({})
        assert update.symbol == ""
        assert update.price == 0.0
        assert update.volume == 0

    def test_timestamp_converted_from_ms(self, ws_client):
        # 2024-01-01 00:00:00 UTC = 1704067200000 ms
        trade = {"s": "X", "p": 10.0, "t": 1704067200000}
        update = ws_client._parse_trade_update(trade)
        assert update.timestamp.year == 2024
        assert update.timestamp.month == 1
        assert update.timestamp.day == 1
        assert update.timestamp.tzinfo == timezone.utc


# ===========================================================================
# FinnhubWebSocketClient -- _parse_quote_update
# ===========================================================================

class TestParseQuoteUpdate:

    def test_basic_quote(self, ws_client):
        quote = {
            "s": "MSFT",
            "b": 300.0,
            "a": 300.50,
            "bv": 50,
            "av": 75,
            "h": 305.0,
            "l": 295.0,
            "o": 298.0,
            "c": 301.0,
            "t": 1717243200000,
        }
        update = ws_client._parse_quote_update(quote)
        assert update.symbol == "MSFT"
        assert update.bid == 300.0
        assert update.ask == 300.50
        # mid price = (300 + 300.50) / 2 = 300.25
        assert update.price == 300.25
        assert update.high == 305.0
        assert update.low == 295.0
        assert update.open == 298.0
        assert update.close == 301.0

    def test_quote_mid_price_when_bid_ask_present(self, ws_client):
        quote = {"s": "X", "b": 10.0, "a": 12.0, "t": 1717243200000}
        update = ws_client._parse_quote_update(quote)
        assert update.price == 11.0  # (10 + 12) / 2

    def test_quote_falls_back_to_close_when_no_bid_ask(self, ws_client):
        quote = {"s": "X", "b": 0.0, "a": 0.0, "c": 99.0, "t": 1717243200000}
        update = ws_client._parse_quote_update(quote)
        # mid_price is 0.0 (bid=0, ask=0), so falls back to c
        assert update.price == 99.0

    def test_quote_empty_data_defaults(self, ws_client):
        update = ws_client._parse_quote_update({})
        assert update.symbol == ""
        assert update.bid == 0.0
        assert update.ask == 0.0
        assert update.high == 0.0
        assert update.low == 0.0

    def test_quote_timestamp_defaults_to_now_when_missing(self, ws_client):
        # When 't' is missing, should use current time
        quote = {"s": "X", "b": 10.0, "a": 12.0}
        update = ws_client._parse_quote_update(quote)
        assert update.timestamp.tzinfo == timezone.utc
        # Timestamp should be very recent (within last minute)
        delta = datetime.now(timezone.utc) - update.timestamp
        assert delta.total_seconds() < 60


# ===========================================================================
# FinnhubWebSocketClient -- subscribe / unsubscribe
# ===========================================================================

class TestSubscribeUnsubscribe:

    @pytest.mark.asyncio
    async def test_subscribe_adds_to_subscriptions(self, ws_client):
        await ws_client.subscribe("AAPL")
        assert "AAPL" in ws_client.subscriptions

    @pytest.mark.asyncio
    async def test_subscribe_with_callback(self, ws_client):
        cb = MagicMock()
        await ws_client.subscribe("AAPL", callback=cb)
        assert "AAPL" in ws_client.price_callbacks
        assert cb in ws_client.price_callbacks["AAPL"]

    @pytest.mark.asyncio
    async def test_subscribe_multiple_callbacks(self, ws_client):
        cb1 = MagicMock()
        cb2 = MagicMock()
        await ws_client.subscribe("AAPL", callback=cb1)
        await ws_client.subscribe("AAPL", callback=cb2)
        assert len(ws_client.price_callbacks["AAPL"]) == 2

    @pytest.mark.asyncio
    async def test_subscribe_sends_json_when_connected(self, ws_client):
        mock_ws = AsyncMock()
        ws_client.websocket = mock_ws
        ws_client.connection_active = True

        await ws_client.subscribe("GOOG")

        mock_ws.send_json.assert_called_once_with({
            "type": "subscribe",
            "symbol": "GOOG",
        })

    @pytest.mark.asyncio
    async def test_subscribe_does_not_send_when_disconnected(self, ws_client):
        mock_ws = AsyncMock()
        ws_client.websocket = mock_ws
        ws_client.connection_active = False

        await ws_client.subscribe("GOOG")

        mock_ws.send_json.assert_not_called()

    @pytest.mark.asyncio
    async def test_subscribe_handles_send_error(self, ws_client):
        mock_ws = AsyncMock()
        mock_ws.send_json.side_effect = ConnectionError("broken pipe")
        ws_client.websocket = mock_ws
        ws_client.connection_active = True

        # Should not raise
        await ws_client.subscribe("AAPL")
        assert "AAPL" in ws_client.subscriptions

    @pytest.mark.asyncio
    async def test_unsubscribe_removes_from_subscriptions(self, ws_client):
        ws_client.subscriptions.add("AAPL")
        ws_client.price_callbacks["AAPL"] = [MagicMock()]

        await ws_client.unsubscribe("AAPL")

        assert "AAPL" not in ws_client.subscriptions
        assert "AAPL" not in ws_client.price_callbacks

    @pytest.mark.asyncio
    async def test_unsubscribe_sends_json_when_connected(self, ws_client):
        mock_ws = AsyncMock()
        ws_client.websocket = mock_ws
        ws_client.connection_active = True
        ws_client.subscriptions.add("AAPL")

        await ws_client.unsubscribe("AAPL")

        mock_ws.send_json.assert_called_once_with({
            "type": "unsubscribe",
            "symbol": "AAPL",
        })

    @pytest.mark.asyncio
    async def test_unsubscribe_nonexistent_symbol_is_noop(self, ws_client):
        # Should not raise
        await ws_client.unsubscribe("NONEXISTENT")

    @pytest.mark.asyncio
    async def test_unsubscribe_handles_send_error(self, ws_client):
        mock_ws = AsyncMock()
        mock_ws.send_json.side_effect = ConnectionError("broken")
        ws_client.websocket = mock_ws
        ws_client.connection_active = True
        ws_client.subscriptions.add("AAPL")

        # Should not raise
        await ws_client.unsubscribe("AAPL")
        assert "AAPL" not in ws_client.subscriptions


# ===========================================================================
# FinnhubWebSocketClient -- _handle_message
# ===========================================================================

class TestHandleMessage:

    @pytest.mark.asyncio
    async def test_trade_message_invokes_callback(self, ws_client):
        cb = AsyncMock()
        ws_client.subscriptions.add("AAPL")
        ws_client.price_callbacks["AAPL"] = [cb]

        msg = json.dumps({
            "type": "trade",
            "data": [{"s": "AAPL", "p": 150.0, "t": 1717243200000, "v": 100}],
        })
        await ws_client._handle_message(msg)

        cb.assert_called_once()
        update_arg = cb.call_args[0][0]
        assert isinstance(update_arg, PriceUpdate)
        assert update_arg.symbol == "AAPL"
        assert update_arg.price == 150.0

    @pytest.mark.asyncio
    async def test_trade_message_ignores_unsubscribed_symbol(self, ws_client):
        cb = AsyncMock()
        ws_client.subscriptions.add("GOOG")
        ws_client.price_callbacks["GOOG"] = [cb]

        msg = json.dumps({
            "type": "trade",
            "data": [{"s": "AAPL", "p": 150.0, "t": 1717243200000}],
        })
        await ws_client._handle_message(msg)

        cb.assert_not_called()

    @pytest.mark.asyncio
    async def test_quote_message_invokes_callback(self, ws_client):
        cb = AsyncMock()
        ws_client.subscriptions.add("MSFT")
        ws_client.price_callbacks["MSFT"] = [cb]

        msg = json.dumps({
            "type": "quote",
            "data": {"s": "MSFT", "b": 300.0, "a": 301.0, "t": 1717243200000},
        })
        await ws_client._handle_message(msg)

        cb.assert_called_once()

    @pytest.mark.asyncio
    async def test_invalid_json_does_not_raise(self, ws_client):
        # Should log error but not raise
        await ws_client._handle_message("not valid json {{{")

    @pytest.mark.asyncio
    async def test_unknown_message_type_is_ignored(self, ws_client):
        msg = json.dumps({"type": "ping", "data": {}})
        # Should not raise
        await ws_client._handle_message(msg)

    @pytest.mark.asyncio
    async def test_trade_with_no_data_field(self, ws_client):
        msg = json.dumps({"type": "trade"})
        # data defaults to [] so no iteration
        await ws_client._handle_message(msg)

    @pytest.mark.asyncio
    async def test_trade_with_missing_symbol_in_data(self, ws_client):
        msg = json.dumps({
            "type": "trade",
            "data": [{"p": 100.0, "t": 1717243200000}],
        })
        # Should not raise -- symbol is None, not in subscriptions
        await ws_client._handle_message(msg)


# ===========================================================================
# FinnhubWebSocketClient -- _invoke_callbacks
# ===========================================================================

class TestInvokeCallbacks:

    @pytest.mark.asyncio
    async def test_invoke_async_callback(self, ws_client):
        cb = AsyncMock()
        ws_client.price_callbacks["AAPL"] = [cb]
        update = _make_price_update()

        await ws_client._invoke_callbacks("AAPL", update)

        cb.assert_called_once_with(update)

    @pytest.mark.asyncio
    async def test_invoke_sync_callback(self, ws_client):
        cb = MagicMock()
        ws_client.price_callbacks["AAPL"] = [cb]
        update = _make_price_update()

        await ws_client._invoke_callbacks("AAPL", update)

        cb.assert_called_once_with(update)

    @pytest.mark.asyncio
    async def test_invoke_no_callbacks_for_symbol(self, ws_client):
        update = _make_price_update()
        # No callbacks registered -- should not raise
        await ws_client._invoke_callbacks("AAPL", update)

    @pytest.mark.asyncio
    async def test_callback_exception_does_not_propagate(self, ws_client):
        cb_bad = AsyncMock(side_effect=ValueError("bad callback"))
        cb_good = AsyncMock()
        ws_client.price_callbacks["AAPL"] = [cb_bad, cb_good]
        update = _make_price_update()

        await ws_client._invoke_callbacks("AAPL", update)

        # Bad callback raised, but good callback still called
        cb_bad.assert_called_once()
        cb_good.assert_called_once()


# ===========================================================================
# FinnhubWebSocketClient -- _handle_connection_error
# ===========================================================================

class TestHandleConnectionError:

    @pytest.mark.asyncio
    async def test_increments_reconnect_attempts(self, ws_client):
        ws_client.max_reconnect_attempts = 1
        with patch.object(ws_client, "connect", new_callable=AsyncMock) as mock_connect:
            with patch("backend.services.realtime_price_service.asyncio.sleep", new_callable=AsyncMock):
                await ws_client._handle_connection_error()

        assert ws_client.reconnect_attempts == 1
        mock_connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_sets_connection_inactive(self, ws_client):
        ws_client.connection_active = True
        ws_client.max_reconnect_attempts = 0  # Exhaust immediately

        await ws_client._handle_connection_error()

        assert ws_client.connection_active is False

    @pytest.mark.asyncio
    async def test_max_reconnect_invokes_error_callbacks(self, ws_client):
        ws_client.reconnect_attempts = 5
        ws_client.max_reconnect_attempts = 5
        cb = AsyncMock()
        ws_client.error_callbacks = [cb]

        await ws_client._handle_connection_error()

        cb.assert_called_once_with("Max reconnection attempts reached")

    @pytest.mark.asyncio
    async def test_max_reconnect_invokes_sync_error_callback(self, ws_client):
        ws_client.reconnect_attempts = 5
        ws_client.max_reconnect_attempts = 5
        cb = MagicMock()
        ws_client.error_callbacks = [cb]

        await ws_client._handle_connection_error()

        cb.assert_called_once_with("Max reconnection attempts reached")

    @pytest.mark.asyncio
    async def test_error_callback_exception_does_not_propagate(self, ws_client):
        ws_client.reconnect_attempts = 5
        ws_client.max_reconnect_attempts = 5
        cb = AsyncMock(side_effect=RuntimeError("callback failed"))
        ws_client.error_callbacks = [cb]

        # Should not raise
        await ws_client._handle_connection_error()

    @pytest.mark.asyncio
    async def test_reconnect_uses_exponential_backoff(self, ws_client):
        ws_client.max_reconnect_attempts = 3
        ws_client.reconnect_attempts = 0
        ws_client.reconnect_delay = 5

        sleep_values = []

        async def capture_sleep(val):
            sleep_values.append(val)

        with patch.object(ws_client, "connect", new_callable=AsyncMock):
            with patch(
                "backend.services.realtime_price_service.asyncio.sleep",
                side_effect=capture_sleep,
            ):
                await ws_client._handle_connection_error()

        # First attempt: delay = min(5 * 2^1, 300) = 10
        assert sleep_values[0] == 10

    @pytest.mark.asyncio
    async def test_reconnect_delay_capped_at_300(self, ws_client):
        ws_client.max_reconnect_attempts = 20
        ws_client.reconnect_attempts = 9
        ws_client.reconnect_delay = 5

        sleep_values = []

        async def capture_sleep(val):
            sleep_values.append(val)

        with patch.object(ws_client, "connect", new_callable=AsyncMock):
            with patch(
                "backend.services.realtime_price_service.asyncio.sleep",
                side_effect=capture_sleep,
            ):
                await ws_client._handle_connection_error()

        # attempt 10: delay = min(5 * 2^10, 300) = min(5120, 300) = 300
        assert sleep_values[0] == 300


# ===========================================================================
# FinnhubWebSocketClient -- disconnect
# ===========================================================================

class TestDisconnect:

    @pytest.mark.asyncio
    async def test_disconnect_closes_websocket(self, ws_client):
        mock_ws = AsyncMock()
        ws_client.websocket = mock_ws
        ws_client.connection_active = True

        await ws_client.disconnect()

        mock_ws.close.assert_called_once()
        assert ws_client.connection_active is False

    @pytest.mark.asyncio
    async def test_disconnect_cancels_receive_task(self, ws_client):
        mock_task = MagicMock()
        ws_client._receive_task = mock_task
        ws_client.websocket = AsyncMock()

        await ws_client.disconnect()

        mock_task.cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_with_no_websocket(self, ws_client):
        # Should not raise
        await ws_client.disconnect()
        assert ws_client.connection_active is False

    @pytest.mark.asyncio
    async def test_disconnect_with_no_receive_task(self, ws_client):
        ws_client.websocket = AsyncMock()
        ws_client._receive_task = None
        # Should not raise
        await ws_client.disconnect()


# ===========================================================================
# RealtimePriceService -- init
# ===========================================================================

class TestRealtimePriceServiceInit:

    def test_uses_provided_api_key(self, mock_settings):
        svc = RealtimePriceService(api_key="custom-key")
        assert svc.api_key == "custom-key"

    def test_falls_back_to_settings_key(self, mock_settings):
        mock_settings.FINNHUB_API_KEY = "settings-key"
        svc = RealtimePriceService(api_key=None)
        assert svc.api_key == "settings-key"

    def test_initial_state(self, service):
        assert service.ws_client is None
        assert service.price_cache == {}
        assert service.redis_client is None
        assert service.initialized is False


# ===========================================================================
# RealtimePriceService -- initialize
# ===========================================================================

class TestRealtimePriceServiceInitialize:

    @pytest.mark.asyncio
    async def test_initialize_without_api_key(self, mock_settings):
        mock_settings.FINNHUB_API_KEY = None
        svc = RealtimePriceService(api_key=None)

        with patch(
            "backend.services.realtime_price_service.get_redis",
            new_callable=AsyncMock,
        ):
            await svc.initialize()

        assert svc.initialized is True
        assert svc.ws_client is None

    @pytest.mark.asyncio
    async def test_initialize_redis_failure_continues(self, service):
        with patch(
            "backend.services.realtime_price_service.get_redis",
            new_callable=AsyncMock,
            side_effect=ConnectionError("Redis down"),
        ), patch(
            "backend.services.realtime_price_service.FinnhubWebSocketClient",
        ) as MockWSClient:
            mock_ws = AsyncMock()
            MockWSClient.return_value = mock_ws

            await service.initialize()

        assert service.initialized is True
        assert service.redis_client is None

    @pytest.mark.asyncio
    async def test_initialize_creates_ws_client(self, service):
        with patch(
            "backend.services.realtime_price_service.get_redis",
            new_callable=AsyncMock,
            return_value=AsyncMock(),
        ), patch(
            "backend.services.realtime_price_service.FinnhubWebSocketClient",
        ) as MockWSClient:
            mock_ws = AsyncMock()
            MockWSClient.return_value = mock_ws

            await service.initialize()

        MockWSClient.assert_called_once_with("test-key")
        mock_ws.connect.assert_called_once()
        assert service.initialized is True


# ===========================================================================
# RealtimePriceService -- get_latest_price
# ===========================================================================

class TestGetLatestPrice:

    @pytest.mark.asyncio
    async def test_returns_from_memory_cache(self, service):
        cached = _make_price_update(symbol="GOOG", price=2800.0)
        service.price_cache["GOOG"] = cached

        result = await service.get_latest_price("GOOG")

        assert result is cached
        assert result.price == 2800.0

    @pytest.mark.asyncio
    async def test_returns_from_redis_cache(self, service):
        ts = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        update_data = _make_price_update(symbol="TSLA", price=700.0, timestamp=ts).to_dict()
        redis_mock = AsyncMock()
        redis_mock.get = AsyncMock(return_value=json.dumps(update_data))
        service.redis_client = redis_mock

        result = await service.get_latest_price("TSLA")

        assert result is not None
        assert result.symbol == "TSLA"
        assert result.price == 700.0

    @pytest.mark.asyncio
    async def test_redis_error_falls_through(self, service):
        redis_mock = AsyncMock()
        redis_mock.get = AsyncMock(side_effect=Exception("Redis error"))
        service.redis_client = redis_mock

        result = await service.get_latest_price("AAPL")
        # No db provided, so should return None
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_from_database_fallback(self, service):
        db_row = _make_db_price_row(close=155.0, high=160.0, low=150.0,
                                     open_=152.0, volume=2_000_000)
        mock_repo = AsyncMock()
        mock_repo.get_latest_price = AsyncMock(return_value=db_row)
        mock_db = AsyncMock()

        with patch(
            "backend.services.realtime_price_service.price_repository",
            mock_repo,
        ):
            result = await service.get_latest_price("AAPL", db=mock_db)

        assert result is not None
        assert result.symbol == "AAPL"
        assert result.price == 155.0
        assert result.volume == 2_000_000
        assert result.high == 160.0
        # bid/ask are derived from close
        assert result.bid == pytest.approx(155.0 * 0.999)
        assert result.ask == pytest.approx(155.0 * 1.001)

    @pytest.mark.asyncio
    async def test_database_returns_none_when_no_record(self, service):
        mock_repo = AsyncMock()
        mock_repo.get_latest_price = AsyncMock(return_value=None)
        mock_db = AsyncMock()

        with patch(
            "backend.services.realtime_price_service.price_repository",
            mock_repo,
        ):
            result = await service.get_latest_price("ZZZZ", db=mock_db)

        assert result is None

    @pytest.mark.asyncio
    async def test_database_error_returns_none(self, service):
        mock_repo = AsyncMock()
        mock_repo.get_latest_price = AsyncMock(side_effect=Exception("DB down"))
        mock_db = AsyncMock()

        with patch(
            "backend.services.realtime_price_service.price_repository",
            mock_repo,
        ):
            result = await service.get_latest_price("AAPL", db=mock_db)

        assert result is None

    @pytest.mark.asyncio
    async def test_no_db_no_cache_returns_none(self, service):
        result = await service.get_latest_price("AAPL")
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_priority_over_redis(self, service):
        """In-memory cache should be checked before Redis."""
        cached = _make_price_update(symbol="AAPL", price=100.0)
        service.price_cache["AAPL"] = cached

        redis_mock = AsyncMock()
        service.redis_client = redis_mock

        result = await service.get_latest_price("AAPL")

        assert result.price == 100.0
        redis_mock.get.assert_not_called()


# ===========================================================================
# RealtimePriceService -- get_latest_prices_bulk
# ===========================================================================

class TestGetLatestPricesBulk:

    @pytest.mark.asyncio
    async def test_returns_prices_for_all_cached_symbols(self, service):
        service.price_cache["AAPL"] = _make_price_update(symbol="AAPL", price=150.0)
        service.price_cache["GOOG"] = _make_price_update(symbol="GOOG", price=2800.0)

        result = await service.get_latest_prices_bulk(["AAPL", "GOOG"])

        assert len(result) == 2
        assert "AAPL" in result
        assert "GOOG" in result

    @pytest.mark.asyncio
    async def test_skips_unavailable_symbols(self, service):
        service.price_cache["AAPL"] = _make_price_update(symbol="AAPL")

        result = await service.get_latest_prices_bulk(["AAPL", "UNKNOWN"])

        assert len(result) == 1
        assert "AAPL" in result
        assert "UNKNOWN" not in result

    @pytest.mark.asyncio
    async def test_empty_symbols_list(self, service):
        result = await service.get_latest_prices_bulk([])
        assert result == {}

    @pytest.mark.asyncio
    async def test_all_symbols_missing(self, service):
        result = await service.get_latest_prices_bulk(["X", "Y", "Z"])
        assert result == {}


# ===========================================================================
# RealtimePriceService -- subscribe_to_symbol
# ===========================================================================

class TestSubscribeToSymbol:

    @pytest.mark.asyncio
    async def test_initializes_if_not_initialized(self, service):
        service.initialized = False
        service.ws_client = AsyncMock()
        service.api_key = "key"

        with patch.object(service, "initialize", new_callable=AsyncMock) as mock_init:
            mock_init.side_effect = lambda: setattr(service, "initialized", True)
            await service.subscribe_to_symbol("AAPL", callback=AsyncMock())

        mock_init.assert_called_once()

    @pytest.mark.asyncio
    async def test_subscribes_via_ws_client(self, service):
        service.initialized = True
        service.ws_client = AsyncMock()
        service.api_key = "key"

        cb = AsyncMock()
        await service.subscribe_to_symbol("AAPL", callback=cb)

        service.ws_client.subscribe.assert_called_once()
        call_args = service.ws_client.subscribe.call_args
        assert call_args[0][0] == "AAPL"


# ===========================================================================
# RealtimePriceService -- unsubscribe_from_symbol
# ===========================================================================

class TestUnsubscribeFromSymbol:

    @pytest.mark.asyncio
    async def test_unsubscribes_via_ws_client(self, service):
        service.ws_client = AsyncMock()
        service.price_cache["AAPL"] = _make_price_update()

        await service.unsubscribe_from_symbol("AAPL")

        service.ws_client.unsubscribe.assert_called_once_with("AAPL")
        assert "AAPL" not in service.price_cache

    @pytest.mark.asyncio
    async def test_clears_cache_entry(self, service):
        service.ws_client = None
        service.price_cache["AAPL"] = _make_price_update()

        await service.unsubscribe_from_symbol("AAPL")

        assert "AAPL" not in service.price_cache

    @pytest.mark.asyncio
    async def test_unsubscribe_no_ws_client(self, service):
        service.ws_client = None
        # Should not raise
        await service.unsubscribe_from_symbol("AAPL")

    @pytest.mark.asyncio
    async def test_unsubscribe_symbol_not_in_cache(self, service):
        service.ws_client = AsyncMock()
        # Should not raise
        await service.unsubscribe_from_symbol("NOTCACHED")


# ===========================================================================
# RealtimePriceService -- shutdown
# ===========================================================================

class TestShutdown:

    @pytest.mark.asyncio
    async def test_shutdown_disconnects_ws_client(self, service):
        service.ws_client = AsyncMock()
        service.initialized = True

        await service.shutdown()

        service.ws_client.disconnect.assert_called_once()
        assert service.initialized is False

    @pytest.mark.asyncio
    async def test_shutdown_without_ws_client(self, service):
        service.ws_client = None
        service.initialized = True

        await service.shutdown()

        assert service.initialized is False


# ===========================================================================
# Module-level helpers
# ===========================================================================

class TestModuleLevelHelpers:

    @pytest.mark.asyncio
    async def test_get_realtime_price_service_creates_singleton(self):
        with patch(
            "backend.services.realtime_price_service._realtime_price_service",
            None,
        ), patch(
            "backend.services.realtime_price_service.RealtimePriceService",
        ) as MockService:
            mock_instance = AsyncMock()
            MockService.return_value = mock_instance

            result = await get_realtime_price_service()

            MockService.assert_called_once()
            mock_instance.initialize.assert_called_once()
            assert result is mock_instance

    @pytest.mark.asyncio
    async def test_get_realtime_price_service_returns_existing(self):
        existing = AsyncMock()
        with patch(
            "backend.services.realtime_price_service._realtime_price_service",
            existing,
        ):
            result = await get_realtime_price_service()
            assert result is existing

    @pytest.mark.asyncio
    async def test_shutdown_realtime_price_service(self):
        mock_svc = AsyncMock()
        with patch(
            "backend.services.realtime_price_service._realtime_price_service",
            mock_svc,
        ):
            await shutdown_realtime_price_service()
            mock_svc.shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_when_no_service_exists(self):
        with patch(
            "backend.services.realtime_price_service._realtime_price_service",
            None,
        ):
            # Should not raise
            await shutdown_realtime_price_service()
