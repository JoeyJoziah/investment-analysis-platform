"""
Unit tests for data ingestion clients.

Tests cover:
- AlphaVantageClient: constructor, quote parsing, daily prices, company overview, error handling
- FinnhubClient: constructor, quote parsing, candles, financials, news, sentiment, error handling
- PolygonClient: constructor, ticker details, aggregates, snapshots, rate limiting, error handling
- SECEdgarClient: constructor, CIK mapping, company facts, filings, number parsing
- SmartDataFetcher: data type routing, source listing, fallback behavior
- RobustAPIClient: async/sync fallback, circuit breaker, client info
"""

import asyncio
import json
import os
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import aiohttp
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_redis():
    """Provide a mock async Redis client used by all data ingestion clients."""
    redis_mock = AsyncMock()
    redis_mock.get = AsyncMock(return_value=None)
    redis_mock.set = AsyncMock(return_value=True)
    redis_mock.setex = AsyncMock(return_value=True)
    redis_mock.exists = AsyncMock(return_value=False)
    redis_mock.ttl = AsyncMock(return_value=-1)
    redis_mock.ping = AsyncMock(return_value=True)
    redis_mock.delete = AsyncMock(return_value=1)
    return redis_mock


@pytest.fixture
def mock_sync_redis():
    """Provide a mock synchronous Redis client used by PolygonClient."""
    redis_mock = MagicMock()
    redis_mock.get = MagicMock(return_value=None)
    redis_mock.setex = MagicMock(return_value=True)
    return redis_mock


@pytest.fixture
def mock_cost_monitor():
    """Provide a mock cost monitor that allows all API calls."""
    monitor = AsyncMock()
    monitor.check_api_limit = AsyncMock(return_value=True)
    monitor.record_api_call = AsyncMock()
    monitor.redis = True  # pretend initialized
    monitor.initialize = AsyncMock()
    return monitor


@pytest.fixture
def mock_settings():
    """Provide mock settings with test API keys."""
    s = MagicMock()
    s.REDIS_URL = "redis://localhost:6379/1"
    s.get_api_key = MagicMock(side_effect=lambda name: f"test_{name}_key")
    return s


# ---------------------------------------------------------------------------
# Sample response data
# ---------------------------------------------------------------------------

ALPHA_VANTAGE_QUOTE_RESPONSE = {
    "Global Quote": {
        "01. symbol": "AAPL",
        "02. open": "149.00",
        "03. high": "152.00",
        "04. low": "148.50",
        "05. price": "150.25",
        "06. volume": "75000000",
        "07. latest trading day": "2024-01-15",
        "08. previous close": "148.10",
        "09. change": "2.15",
        "10. change percent": "1.45%",
    }
}

ALPHA_VANTAGE_DAILY_RESPONSE = {
    "Meta Data": {"2. Symbol": "AAPL"},
    "Time Series (Daily)": {
        "2024-01-15": {
            "1. open": "149.00",
            "2. high": "152.00",
            "3. low": "148.50",
            "4. close": "150.25",
            "5. adjusted close": "150.25",
            "6. volume": "75000000",
            "7. dividend amount": "0.00",
            "8. split coefficient": "1.0",
        },
        "2024-01-14": {
            "1. open": "147.00",
            "2. high": "149.50",
            "3. low": "146.00",
            "4. close": "148.10",
            "5. adjusted close": "148.10",
            "6. volume": "60000000",
            "7. dividend amount": "0.00",
            "8. split coefficient": "1.0",
        },
    },
}

ALPHA_VANTAGE_OVERVIEW_RESPONSE = {
    "Symbol": "AAPL",
    "Name": "Apple Inc",
    "Description": "Tech company",
    "Exchange": "NASDAQ",
    "Currency": "USD",
    "Country": "US",
    "Sector": "Technology",
    "Industry": "Consumer Electronics",
    "MarketCapitalization": "2500000000000",
    "PERatio": "28.5",
    "EPS": "6.15",
    "DividendYield": "0.005",
    "Beta": "1.2",
    "52WeekHigh": "199.62",
    "52WeekLow": "124.17",
}

FINNHUB_QUOTE_RESPONSE = {
    "c": 150.25,
    "d": 2.15,
    "dp": 1.45,
    "h": 152.00,
    "l": 148.50,
    "o": 149.00,
    "pc": 148.10,
    "t": 1705363200,
}

FINNHUB_CANDLE_RESPONSE = {
    "s": "ok",
    "t": [1705276800, 1705363200],
    "o": [147.00, 149.00],
    "h": [149.50, 152.00],
    "l": [146.00, 148.50],
    "c": [148.10, 150.25],
    "v": [60000000, 75000000],
}

FINNHUB_FINANCIALS_RESPONSE = {
    "metric": {
        "peBasicExclExtraTTM": 28.5,
        "peTTM": 29.0,
        "epsExclExtraItemsTTM": 6.15,
        "roeTTM": 150.0,
        "roaTTM": 28.0,
        "grossMarginTTM": 45.0,
        "beta": 1.2,
        "52WeekHigh": 199.62,
        "52WeekLow": 124.17,
    }
}

FINNHUB_NEWS_RESPONSE = [
    {
        "id": 1,
        "headline": "Apple Reports Record Revenue",
        "summary": "Apple Inc reported record quarterly revenue.",
        "source": "Reuters",
        "url": "https://example.com/news/1",
        "datetime": 1705363200,
        "category": "technology",
        "related": "AAPL,MSFT",
        "image": "https://example.com/img.jpg",
    }
]

FINNHUB_SENTIMENT_RESPONSE = {
    "buzz": {
        "articlesInLastWeek": 120,
        "buzz": 1.5,
        "weeklyAverage": 80,
    },
    "sentiment": {
        "bullishPercent": 0.65,
        "bearishPercent": 0.35,
    },
    "sectorAverageBullishPercent": 0.55,
    "sectorAverageNewsScore": 0.6,
    "companyNewsScore": 0.72,
}

POLYGON_TICKER_DETAILS_RESPONSE = {
    "status": "OK",
    "results": {
        "ticker": "AAPL",
        "name": "Apple Inc.",
        "market": "stocks",
        "locale": "us",
        "primary_exchange": "XNAS",
        "type": "CS",
        "active": True,
        "currency_name": "usd",
        "cik": "0000320193",
        "market_cap": 2500000000000,
        "total_employees": 164000,
        "description": "Apple designs iPhones.",
    },
}

POLYGON_AGGREGATES_RESPONSE = {
    "status": "OK",
    "results": [
        {
            "t": 1705276800000,
            "o": 147.00,
            "h": 149.50,
            "l": 146.00,
            "c": 148.10,
            "v": 60000000,
            "vw": 147.80,
            "n": 500000,
        },
        {
            "t": 1705363200000,
            "o": 149.00,
            "h": 152.00,
            "l": 148.50,
            "c": 150.25,
            "v": 75000000,
            "vw": 150.12,
            "n": 600000,
        },
    ],
}


# ============================================================================
# AlphaVantageClient Tests
# ============================================================================


class TestAlphaVantageClient:
    """Tests for AlphaVantageClient."""

    @patch("backend.data_ingestion.base_client.settings")
    @patch("backend.data_ingestion.base_client.CircuitBreaker")
    def test_constructor_sets_provider_and_api_key(self, mock_cb, mock_settings):
        mock_settings.get_api_key = MagicMock(return_value="test_av_key")
        from backend.data_ingestion.alpha_vantage_client import AlphaVantageClient

        client = AlphaVantageClient()
        assert client.provider_name == "alpha_vantage"
        assert client.api_key == "test_av_key"

    @patch("backend.data_ingestion.base_client.settings")
    @patch("backend.data_ingestion.base_client.CircuitBreaker")
    def test_base_url(self, mock_cb, mock_settings):
        mock_settings.get_api_key = MagicMock(return_value="key")
        from backend.data_ingestion.alpha_vantage_client import AlphaVantageClient

        client = AlphaVantageClient()
        assert client.base_url == "https://www.alphavantage.co/query"

    @patch("backend.data_ingestion.base_client.settings")
    @patch("backend.data_ingestion.base_client.CircuitBreaker")
    def test_add_auth_params(self, mock_cb, mock_settings):
        mock_settings.get_api_key = MagicMock(return_value="my_api_key")
        from backend.data_ingestion.alpha_vantage_client import AlphaVantageClient

        client = AlphaVantageClient()
        result = client._add_auth_params({"symbol": "AAPL"})
        assert result["apikey"] == "my_api_key"
        assert result["symbol"] == "AAPL"

    @patch("backend.data_ingestion.base_client.settings")
    @patch("backend.data_ingestion.base_client.CircuitBreaker")
    def test_functions_map_contains_expected_entries(self, mock_cb, mock_settings):
        mock_settings.get_api_key = MagicMock(return_value="key")
        from backend.data_ingestion.alpha_vantage_client import AlphaVantageClient

        client = AlphaVantageClient()
        expected_keys = {"quote", "daily", "intraday", "earnings", "overview", "rsi", "macd"}
        assert expected_keys.issubset(set(client.functions.keys()))
        assert client.functions["quote"] == "GLOBAL_QUOTE"

    @pytest.mark.asyncio
    @patch("backend.data_ingestion.alpha_vantage_client.get_redis")
    @patch("backend.data_ingestion.base_client.settings")
    @patch("backend.data_ingestion.base_client.CircuitBreaker")
    async def test_get_quote_parses_response(self, mock_cb, mock_settings, mock_get_redis, mock_redis):
        mock_settings.get_api_key = MagicMock(return_value="key")
        mock_get_redis.return_value = mock_redis
        from backend.data_ingestion.alpha_vantage_client import AlphaVantageClient

        client = AlphaVantageClient()
        # Bypass cache and _make_request -- mock get_cached_or_fetch to call inner fetch
        async def fake_cached_or_fetch(cache_key, fetch_func, ttl=300):
            return await fetch_func()

        client.get_cached_or_fetch = fake_cached_or_fetch
        client._make_request = AsyncMock(return_value=ALPHA_VANTAGE_QUOTE_RESPONSE)

        result = await client.get_quote("AAPL")

        assert result is not None
        assert result["symbol"] == "AAPL"
        assert result["price"] == 150.25
        assert result["change"] == 2.15
        assert result["change_percent"] == "1.45%"
        assert result["volume"] == 75000000
        assert result["latest_trading_day"] == "2024-01-15"
        assert result["previous_close"] == 148.10

    @pytest.mark.asyncio
    @patch("backend.data_ingestion.alpha_vantage_client.get_redis")
    @patch("backend.data_ingestion.base_client.settings")
    @patch("backend.data_ingestion.base_client.CircuitBreaker")
    async def test_get_quote_returns_none_for_missing_global_quote(self, mock_cb, mock_settings, mock_get_redis, mock_redis):
        mock_settings.get_api_key = MagicMock(return_value="key")
        mock_get_redis.return_value = mock_redis
        from backend.data_ingestion.alpha_vantage_client import AlphaVantageClient

        client = AlphaVantageClient()

        async def fake_cached_or_fetch(cache_key, fetch_func, ttl=300):
            return await fetch_func()

        client.get_cached_or_fetch = fake_cached_or_fetch
        client._make_request = AsyncMock(return_value={"Note": "API call limit reached"})

        result = await client.get_quote("AAPL")
        assert result is None

    @pytest.mark.asyncio
    @patch("backend.data_ingestion.alpha_vantage_client.get_redis")
    @patch("backend.data_ingestion.base_client.settings")
    @patch("backend.data_ingestion.base_client.CircuitBreaker")
    async def test_get_daily_prices_parses_time_series(self, mock_cb, mock_settings, mock_get_redis, mock_redis):
        mock_settings.get_api_key = MagicMock(return_value="key")
        mock_get_redis.return_value = mock_redis
        from backend.data_ingestion.alpha_vantage_client import AlphaVantageClient

        client = AlphaVantageClient()

        async def fake_cached_or_fetch(cache_key, fetch_func, ttl=3600):
            return await fetch_func()

        client.get_cached_or_fetch = fake_cached_or_fetch
        client._make_request = AsyncMock(return_value=ALPHA_VANTAGE_DAILY_RESPONSE)

        result = await client.get_daily_prices("AAPL")

        assert result is not None
        assert result["symbol"] == "AAPL"
        assert len(result["prices"]) == 2
        # Sorted by date descending
        assert result["prices"][0]["date"] == "2024-01-15"
        assert result["prices"][0]["close"] == 150.25
        assert result["prices"][0]["volume"] == 75000000

    @pytest.mark.asyncio
    @patch("backend.data_ingestion.alpha_vantage_client.get_redis")
    @patch("backend.data_ingestion.base_client.settings")
    @patch("backend.data_ingestion.base_client.CircuitBreaker")
    async def test_get_daily_prices_returns_none_on_empty(self, mock_cb, mock_settings, mock_get_redis, mock_redis):
        mock_settings.get_api_key = MagicMock(return_value="key")
        mock_get_redis.return_value = mock_redis
        from backend.data_ingestion.alpha_vantage_client import AlphaVantageClient

        client = AlphaVantageClient()

        async def fake_cached_or_fetch(cache_key, fetch_func, ttl=3600):
            return await fetch_func()

        client.get_cached_or_fetch = fake_cached_or_fetch
        client._make_request = AsyncMock(return_value={})

        result = await client.get_daily_prices("AAPL")
        assert result is None

    @pytest.mark.asyncio
    @patch("backend.data_ingestion.alpha_vantage_client.get_redis")
    @patch("backend.data_ingestion.base_client.settings")
    @patch("backend.data_ingestion.base_client.CircuitBreaker")
    async def test_get_company_overview_parses_response(self, mock_cb, mock_settings, mock_get_redis, mock_redis):
        mock_settings.get_api_key = MagicMock(return_value="key")
        mock_get_redis.return_value = mock_redis
        from backend.data_ingestion.alpha_vantage_client import AlphaVantageClient

        client = AlphaVantageClient()

        async def fake_cached_or_fetch(cache_key, fetch_func, ttl=86400):
            return await fetch_func()

        client.get_cached_or_fetch = fake_cached_or_fetch
        client._make_request = AsyncMock(return_value=ALPHA_VANTAGE_OVERVIEW_RESPONSE)

        result = await client.get_company_overview("AAPL")

        assert result is not None
        assert result["symbol"] == "AAPL"
        assert result["name"] == "Apple Inc"
        assert result["sector"] == "Technology"
        assert result["market_cap"] == 2500000000000
        assert result["pe_ratio"] == 28.5
        assert result["eps"] == 6.15
        assert result["beta"] == 1.2

    @pytest.mark.asyncio
    @patch("backend.data_ingestion.alpha_vantage_client.get_redis")
    @patch("backend.data_ingestion.base_client.settings")
    @patch("backend.data_ingestion.base_client.CircuitBreaker")
    async def test_get_quote_returns_none_when_make_request_returns_none(self, mock_cb, mock_settings, mock_get_redis, mock_redis):
        mock_settings.get_api_key = MagicMock(return_value="key")
        mock_get_redis.return_value = mock_redis
        from backend.data_ingestion.alpha_vantage_client import AlphaVantageClient

        client = AlphaVantageClient()

        async def fake_cached_or_fetch(cache_key, fetch_func, ttl=300):
            return await fetch_func()

        client.get_cached_or_fetch = fake_cached_or_fetch
        client._make_request = AsyncMock(return_value=None)

        result = await client.get_quote("AAPL")
        assert result is None


# ============================================================================
# FinnhubClient Tests
# ============================================================================


class TestFinnhubClient:
    """Tests for FinnhubClient."""

    @patch("backend.data_ingestion.base_client.settings")
    @patch("backend.data_ingestion.base_client.CircuitBreaker")
    def test_constructor_sets_provider_and_websocket_url(self, mock_cb, mock_settings):
        mock_settings.get_api_key = MagicMock(return_value="fh_key")
        from backend.data_ingestion.finnhub_client import FinnhubClient

        client = FinnhubClient()
        assert client.provider_name == "finnhub"
        assert client.api_key == "fh_key"
        assert client.websocket_url == "wss://ws.finnhub.io"

    @patch("backend.data_ingestion.base_client.settings")
    @patch("backend.data_ingestion.base_client.CircuitBreaker")
    def test_base_url(self, mock_cb, mock_settings):
        mock_settings.get_api_key = MagicMock(return_value="key")
        from backend.data_ingestion.finnhub_client import FinnhubClient

        client = FinnhubClient()
        assert client.base_url == "https://finnhub.io/api/v1"

    @patch("backend.data_ingestion.base_client.settings")
    @patch("backend.data_ingestion.base_client.CircuitBreaker")
    def test_add_auth_params_uses_token(self, mock_cb, mock_settings):
        mock_settings.get_api_key = MagicMock(return_value="fh_secret")
        from backend.data_ingestion.finnhub_client import FinnhubClient

        client = FinnhubClient()
        result = client._add_auth_params({"symbol": "AAPL"})
        assert result["token"] == "fh_secret"

    @pytest.mark.asyncio
    @patch("backend.data_ingestion.finnhub_client.get_redis")
    @patch("backend.data_ingestion.base_client.settings")
    @patch("backend.data_ingestion.base_client.CircuitBreaker")
    async def test_get_quote_parses_response(self, mock_cb, mock_settings, mock_get_redis, mock_redis):
        mock_settings.get_api_key = MagicMock(return_value="key")
        mock_get_redis.return_value = mock_redis
        from backend.data_ingestion.finnhub_client import FinnhubClient

        client = FinnhubClient()

        async def fake_cached_or_fetch(cache_key, fetch_func, ttl=60):
            return await fetch_func()

        client.get_cached_or_fetch = fake_cached_or_fetch
        client._make_request = AsyncMock(return_value=FINNHUB_QUOTE_RESPONSE)

        result = await client.get_quote("AAPL")

        assert result is not None
        assert result["symbol"] == "AAPL"
        assert result["current_price"] == 150.25
        assert result["change"] == 2.15
        assert result["percent_change"] == 1.45
        assert result["high"] == 152.00
        assert result["low"] == 148.50
        assert result["open"] == 149.00
        assert result["previous_close"] == 148.10

    @pytest.mark.asyncio
    @patch("backend.data_ingestion.finnhub_client.get_redis")
    @patch("backend.data_ingestion.base_client.settings")
    @patch("backend.data_ingestion.base_client.CircuitBreaker")
    async def test_get_quote_returns_none_when_no_data(self, mock_cb, mock_settings, mock_get_redis, mock_redis):
        mock_settings.get_api_key = MagicMock(return_value="key")
        mock_get_redis.return_value = mock_redis
        from backend.data_ingestion.finnhub_client import FinnhubClient

        client = FinnhubClient()

        async def fake_cached_or_fetch(cache_key, fetch_func, ttl=60):
            return await fetch_func()

        client.get_cached_or_fetch = fake_cached_or_fetch
        client._make_request = AsyncMock(return_value=None)

        result = await client.get_quote("AAPL")
        assert result is None

    @pytest.mark.asyncio
    @patch("backend.data_ingestion.finnhub_client.get_redis")
    @patch("backend.data_ingestion.base_client.settings")
    @patch("backend.data_ingestion.base_client.CircuitBreaker")
    async def test_get_candles_parses_ok_response(self, mock_cb, mock_settings, mock_get_redis, mock_redis):
        mock_settings.get_api_key = MagicMock(return_value="key")
        mock_get_redis.return_value = mock_redis
        from backend.data_ingestion.finnhub_client import FinnhubClient

        client = FinnhubClient()

        async def fake_cached_or_fetch(cache_key, fetch_func, ttl=3600):
            return await fetch_func()

        client.get_cached_or_fetch = fake_cached_or_fetch
        client._make_request = AsyncMock(return_value=FINNHUB_CANDLE_RESPONSE)

        result = await client.get_candles("AAPL", resolution="D")

        assert result is not None
        assert result["symbol"] == "AAPL"
        assert result["resolution"] == "D"
        assert len(result["candles"]) == 2
        assert result["candles"][0]["open"] == 147.00
        assert result["candles"][1]["close"] == 150.25

    @pytest.mark.asyncio
    @patch("backend.data_ingestion.finnhub_client.get_redis")
    @patch("backend.data_ingestion.base_client.settings")
    @patch("backend.data_ingestion.base_client.CircuitBreaker")
    async def test_get_candles_returns_none_when_status_not_ok(self, mock_cb, mock_settings, mock_get_redis, mock_redis):
        mock_settings.get_api_key = MagicMock(return_value="key")
        mock_get_redis.return_value = mock_redis
        from backend.data_ingestion.finnhub_client import FinnhubClient

        client = FinnhubClient()

        async def fake_cached_or_fetch(cache_key, fetch_func, ttl=3600):
            return await fetch_func()

        client.get_cached_or_fetch = fake_cached_or_fetch
        client._make_request = AsyncMock(return_value={"s": "no_data"})

        result = await client.get_candles("AAPL")
        assert result is None

    @pytest.mark.asyncio
    @patch("backend.data_ingestion.finnhub_client.get_redis")
    @patch("backend.data_ingestion.base_client.settings")
    @patch("backend.data_ingestion.base_client.CircuitBreaker")
    async def test_get_basic_financials_parses_metrics(self, mock_cb, mock_settings, mock_get_redis, mock_redis):
        mock_settings.get_api_key = MagicMock(return_value="key")
        mock_get_redis.return_value = mock_redis
        from backend.data_ingestion.finnhub_client import FinnhubClient

        client = FinnhubClient()

        async def fake_cached_or_fetch(cache_key, fetch_func, ttl=21600):
            return await fetch_func()

        client.get_cached_or_fetch = fake_cached_or_fetch
        client._make_request = AsyncMock(return_value=FINNHUB_FINANCIALS_RESPONSE)

        result = await client.get_basic_financials("AAPL")

        assert result is not None
        assert result["symbol"] == "AAPL"
        assert result["pe_ratio"] == 28.5
        assert result["eps_ttm"] == 6.15
        assert result["roe"] == 150.0
        assert result["beta"] == 1.2

    @pytest.mark.asyncio
    @patch("backend.data_ingestion.finnhub_client.get_redis")
    @patch("backend.data_ingestion.base_client.settings")
    @patch("backend.data_ingestion.base_client.CircuitBreaker")
    async def test_get_news_parses_items(self, mock_cb, mock_settings, mock_get_redis, mock_redis):
        mock_settings.get_api_key = MagicMock(return_value="key")
        mock_get_redis.return_value = mock_redis
        from backend.data_ingestion.finnhub_client import FinnhubClient

        client = FinnhubClient()

        async def fake_cached_or_fetch(cache_key, fetch_func, ttl=900):
            return await fetch_func()

        client.get_cached_or_fetch = fake_cached_or_fetch
        client._make_request = AsyncMock(return_value=FINNHUB_NEWS_RESPONSE)

        result = await client.get_news(symbol="AAPL")

        assert result is not None
        assert len(result) == 1
        assert result[0]["headline"] == "Apple Reports Record Revenue"
        assert result[0]["source"] == "Reuters"
        assert result[0]["related"] == ["AAPL", "MSFT"]

    @pytest.mark.asyncio
    @patch("backend.data_ingestion.finnhub_client.get_redis")
    @patch("backend.data_ingestion.base_client.settings")
    @patch("backend.data_ingestion.base_client.CircuitBreaker")
    async def test_get_sentiment_parses_response(self, mock_cb, mock_settings, mock_get_redis, mock_redis):
        mock_settings.get_api_key = MagicMock(return_value="key")
        mock_get_redis.return_value = mock_redis
        from backend.data_ingestion.finnhub_client import FinnhubClient

        client = FinnhubClient()

        async def fake_cached_or_fetch(cache_key, fetch_func, ttl=3600):
            return await fetch_func()

        client.get_cached_or_fetch = fake_cached_or_fetch
        client._make_request = AsyncMock(return_value=FINNHUB_SENTIMENT_RESPONSE)

        result = await client.get_sentiment("AAPL")

        assert result is not None
        assert result["symbol"] == "AAPL"
        assert result["articles_in_last_week"] == 120
        assert result["bullish_percent"] == 0.65
        assert result["bearish_percent"] == 0.35
        assert result["company_news_score"] == 0.72

    @pytest.mark.asyncio
    @patch("backend.data_ingestion.finnhub_client.get_redis")
    @patch("backend.data_ingestion.base_client.settings")
    @patch("backend.data_ingestion.base_client.CircuitBreaker")
    async def test_get_news_returns_none_when_no_data(self, mock_cb, mock_settings, mock_get_redis, mock_redis):
        mock_settings.get_api_key = MagicMock(return_value="key")
        mock_get_redis.return_value = mock_redis
        from backend.data_ingestion.finnhub_client import FinnhubClient

        client = FinnhubClient()

        async def fake_cached_or_fetch(cache_key, fetch_func, ttl=900):
            return await fetch_func()

        client.get_cached_or_fetch = fake_cached_or_fetch
        client._make_request = AsyncMock(return_value=None)

        result = await client.get_news(symbol="AAPL")
        assert result is None

    @pytest.mark.asyncio
    @patch("backend.data_ingestion.finnhub_client.get_redis")
    @patch("backend.data_ingestion.base_client.settings")
    @patch("backend.data_ingestion.base_client.CircuitBreaker")
    async def test_get_recommendations_parses_response(self, mock_cb, mock_settings, mock_get_redis, mock_redis):
        mock_settings.get_api_key = MagicMock(return_value="key")
        mock_get_redis.return_value = mock_redis
        from backend.data_ingestion.finnhub_client import FinnhubClient

        client = FinnhubClient()

        recs_response = [
            {
                "period": "2024-01-01",
                "strongBuy": 10,
                "buy": 15,
                "hold": 8,
                "sell": 2,
                "strongSell": 1,
            }
        ]

        async def fake_cached_or_fetch(cache_key, fetch_func, ttl=86400):
            return await fetch_func()

        client.get_cached_or_fetch = fake_cached_or_fetch
        client._make_request = AsyncMock(return_value=recs_response)

        result = await client.get_recommendations("AAPL")

        assert result is not None
        assert len(result) == 1
        assert result[0]["strong_buy"] == 10
        assert result[0]["buy"] == 15
        assert result[0]["total"] == 36


# ============================================================================
# PolygonClient Tests
# ============================================================================


class TestPolygonClient:
    """Tests for PolygonClient."""

    @patch("backend.data_ingestion.polygon_client.get_redis_client")
    @patch("backend.data_ingestion.polygon_client.CircuitBreaker")
    def test_constructor_reads_api_key_from_env(self, mock_cb, mock_get_redis):
        mock_get_redis.return_value = MagicMock()
        with patch.dict(os.environ, {"POLYGON_API_KEY": "poly_test_key"}):
            from backend.data_ingestion.polygon_client import PolygonClient

            client = PolygonClient()
            assert client.api_key == "poly_test_key"
            assert client.calls_per_minute == 5
            assert client.min_interval == 12.0

    @patch("backend.data_ingestion.polygon_client.get_redis_client")
    @patch("backend.data_ingestion.polygon_client.CircuitBreaker")
    def test_constructor_raises_on_missing_api_key(self, mock_cb, mock_get_redis):
        mock_get_redis.return_value = MagicMock()
        with patch.dict(os.environ, {}, clear=True):
            # Remove POLYGON_API_KEY if present
            env_copy = os.environ.copy()
            env_copy.pop("POLYGON_API_KEY", None)
            with patch.dict(os.environ, env_copy, clear=True):
                from backend.data_ingestion.polygon_client import PolygonClient

                with pytest.raises(ValueError, match="POLYGON_API_KEY not set"):
                    PolygonClient()

    @pytest.mark.asyncio
    @patch("backend.data_ingestion.polygon_client.get_redis_client")
    @patch("backend.data_ingestion.polygon_client.CircuitBreaker")
    async def test_get_ticker_details_parses_ok_response(self, mock_cb, mock_get_redis, mock_sync_redis):
        mock_get_redis.return_value = mock_sync_redis
        with patch.dict(os.environ, {"POLYGON_API_KEY": "test_key"}):
            from backend.data_ingestion.polygon_client import PolygonClient

            client = PolygonClient()

            # Mock the _make_request method directly
            client._make_request = AsyncMock(return_value=POLYGON_TICKER_DETAILS_RESPONSE)

            result = await client.get_ticker_details("AAPL")

            assert result["symbol"] == "AAPL"
            assert result["name"] == "Apple Inc."
            assert result["market"] == "stocks"
            assert result["active"] is True
            assert result["market_cap"] == 2500000000000

    @pytest.mark.asyncio
    @patch("backend.data_ingestion.polygon_client.get_redis_client")
    @patch("backend.data_ingestion.polygon_client.CircuitBreaker")
    async def test_get_ticker_details_returns_empty_on_error_status(self, mock_cb, mock_get_redis, mock_sync_redis):
        mock_get_redis.return_value = mock_sync_redis
        with patch.dict(os.environ, {"POLYGON_API_KEY": "test_key"}):
            from backend.data_ingestion.polygon_client import PolygonClient

            client = PolygonClient()
            client._make_request = AsyncMock(return_value={"status": "NOT_FOUND"})

            result = await client.get_ticker_details("INVALID")
            assert result == {}

    @pytest.mark.asyncio
    @patch("backend.data_ingestion.polygon_client.get_redis_client")
    @patch("backend.data_ingestion.polygon_client.CircuitBreaker")
    async def test_get_aggregates_parses_bars(self, mock_cb, mock_get_redis, mock_sync_redis):
        mock_get_redis.return_value = mock_sync_redis
        with patch.dict(os.environ, {"POLYGON_API_KEY": "test_key"}):
            from backend.data_ingestion.polygon_client import PolygonClient

            client = PolygonClient()
            client._make_request = AsyncMock(return_value=POLYGON_AGGREGATES_RESPONSE)

            result = await client.get_aggregates("AAPL", from_date="2024-01-01", to_date="2024-01-15")

            assert len(result) == 2
            assert result[0]["open"] == 147.00
            assert result[1]["close"] == 150.25
            assert result[1]["vwap"] == 150.12

    @pytest.mark.asyncio
    @patch("backend.data_ingestion.polygon_client.get_redis_client")
    @patch("backend.data_ingestion.polygon_client.CircuitBreaker")
    async def test_get_aggregates_returns_empty_list_on_no_results(self, mock_cb, mock_get_redis, mock_sync_redis):
        mock_get_redis.return_value = mock_sync_redis
        with patch.dict(os.environ, {"POLYGON_API_KEY": "test_key"}):
            from backend.data_ingestion.polygon_client import PolygonClient

            client = PolygonClient()
            client._make_request = AsyncMock(return_value={"status": "OK", "results": None})

            result = await client.get_aggregates("AAPL")
            assert result == []

    @pytest.mark.asyncio
    @patch("backend.data_ingestion.polygon_client.get_redis_client")
    @patch("backend.data_ingestion.polygon_client.CircuitBreaker")
    async def test_get_ticker_details_propagates_exception(self, mock_cb, mock_get_redis, mock_sync_redis):
        mock_get_redis.return_value = mock_sync_redis
        with patch.dict(os.environ, {"POLYGON_API_KEY": "test_key"}):
            from backend.data_ingestion.polygon_client import PolygonClient

            client = PolygonClient()
            client._make_request = AsyncMock(side_effect=aiohttp.ClientError("Connection failed"))

            with pytest.raises(aiohttp.ClientError, match="Connection failed"):
                await client.get_ticker_details("AAPL")

    @pytest.mark.asyncio
    @patch("backend.data_ingestion.polygon_client.get_redis_client")
    @patch("backend.data_ingestion.polygon_client.CircuitBreaker")
    async def test_get_snapshot_parses_ok_response(self, mock_cb, mock_get_redis, mock_sync_redis):
        mock_get_redis.return_value = mock_sync_redis
        with patch.dict(os.environ, {"POLYGON_API_KEY": "test_key"}):
            from backend.data_ingestion.polygon_client import PolygonClient

            client = PolygonClient()
            snapshot_resp = {
                "status": "OK",
                "ticker": {
                    "ticker": "AAPL",
                    "day": {"o": 149.0, "h": 152.0, "l": 148.5, "c": 150.25},
                    "lastQuote": {"p": 150.20},
                    "lastTrade": {"p": 150.25},
                    "min": {},
                    "prevDay": {"c": 148.10},
                    "updated": 1705363200000,
                },
            }
            client._make_request = AsyncMock(return_value=snapshot_resp)

            result = await client.get_snapshot("AAPL")

            assert result["symbol"] == "AAPL"
            assert result["day"]["o"] == 149.0
            assert result["updated"] == 1705363200000

    @pytest.mark.asyncio
    @patch("backend.data_ingestion.polygon_client.get_redis_client")
    @patch("backend.data_ingestion.polygon_client.CircuitBreaker")
    async def test_get_news_parses_articles(self, mock_cb, mock_get_redis, mock_sync_redis):
        mock_get_redis.return_value = mock_sync_redis
        with patch.dict(os.environ, {"POLYGON_API_KEY": "test_key"}):
            from backend.data_ingestion.polygon_client import PolygonClient

            client = PolygonClient()
            news_resp = {
                "status": "OK",
                "results": [
                    {
                        "title": "AAPL earnings beat",
                        "author": "John Doe",
                        "published_utc": "2024-01-15T10:00:00Z",
                        "article_url": "https://example.com/1",
                        "tickers": ["AAPL"],
                        "publisher": {"name": "MarketWatch"},
                        "keywords": ["earnings", "apple"],
                        "description": "Apple beats expectations.",
                    }
                ],
            }
            client._make_request = AsyncMock(return_value=news_resp)

            result = await client.get_news(symbol="AAPL")

            assert len(result) == 1
            assert result[0]["title"] == "AAPL earnings beat"
            assert result[0]["publisher"] == "MarketWatch"
            assert result[0]["tickers"] == ["AAPL"]


# ============================================================================
# SECEdgarClient Tests
# ============================================================================


class TestSECEdgarClient:
    """Tests for SECEdgarClient."""

    @patch("backend.data_ingestion.base_client.settings")
    @patch("backend.data_ingestion.base_client.CircuitBreaker")
    def test_constructor_sets_provider_and_headers(self, mock_cb, mock_settings):
        mock_settings.get_api_key = MagicMock(return_value=None)
        from backend.data_ingestion.sec_edgar_client import SECEdgarClient

        client = SECEdgarClient()
        assert client.provider_name == "sec_edgar"
        assert "User-Agent" in client.headers
        assert "InvestmentAnalysisPlatform" in client.headers["User-Agent"]

    @patch("backend.data_ingestion.base_client.settings")
    @patch("backend.data_ingestion.base_client.CircuitBreaker")
    def test_base_url(self, mock_cb, mock_settings):
        mock_settings.get_api_key = MagicMock(return_value=None)
        from backend.data_ingestion.sec_edgar_client import SECEdgarClient

        client = SECEdgarClient()
        assert client.base_url == "https://data.sec.gov"

    @patch("backend.data_ingestion.base_client.settings")
    @patch("backend.data_ingestion.base_client.CircuitBreaker")
    def test_parse_number_various_inputs(self, mock_cb, mock_settings):
        mock_settings.get_api_key = MagicMock(return_value=None)
        from backend.data_ingestion.sec_edgar_client import SECEdgarClient

        client = SECEdgarClient()

        assert client._parse_number("1,234,567") == 1234567
        assert client._parse_number("$12.50") == 12.50
        assert client._parse_number("(500)") == -500
        assert client._parse_number("") is None
        assert client._parse_number("\u2014") is None  # em dash
        assert client._parse_number("-") is None
        assert client._parse_number("hello") == "hello"

    @pytest.mark.xfail(reason="Flaky: import-chain state issue with Mock.get_text", strict=False)
    @patch("backend.data_ingestion.base_client.settings")
    @patch("backend.data_ingestion.base_client.CircuitBreaker")
    def test_extract_section_returns_none_for_missing_section(self, mock_cb, mock_settings):
        mock_settings.get_api_key = MagicMock(return_value=None)
        from backend.data_ingestion.sec_edgar_client import SECEdgarClient
        from bs4 import BeautifulSoup

        client = SECEdgarClient()
        soup = BeautifulSoup("<html><body>Nothing relevant here.</body></html>", "html.parser")
        result = client._extract_section(soup, ["ITEM 99", "NONEXISTENT"])
        assert result is None

    @pytest.mark.asyncio
    @patch("backend.data_ingestion.sec_edgar_client.get_redis")
    @patch("backend.data_ingestion.base_client.settings")
    @patch("backend.data_ingestion.base_client.CircuitBreaker")
    async def test_get_company_facts_returns_none_for_unknown_ticker(self, mock_cb, mock_settings, mock_get_redis, mock_redis):
        mock_settings.get_api_key = MagicMock(return_value=None)
        mock_get_redis.return_value = mock_redis
        from backend.data_ingestion.sec_edgar_client import SECEdgarClient

        client = SECEdgarClient()
        # Empty CIK map and make get_cik_mapping return empty
        client.cik_map = {}

        async def fake_cached_or_fetch(cache_key, fetch_func, ttl=86400):
            return await fetch_func()

        client.get_cached_or_fetch = fake_cached_or_fetch
        client._make_request = AsyncMock(return_value=[])

        result = await client.get_company_facts("NONEXISTENT")
        assert result is None


# ============================================================================
# SmartDataFetcher Tests
# ============================================================================


class TestSmartDataFetcher:
    """Tests for SmartDataFetcher."""

    def test_constructor_with_defaults(self):
        from backend.data_ingestion.smart_data_fetcher import SmartDataFetcher

        fetcher = SmartDataFetcher()
        assert fetcher.cache_manager is None
        assert fetcher.rate_limiter is None

    def test_constructor_with_custom_params(self):
        from backend.data_ingestion.smart_data_fetcher import SmartDataFetcher

        cache = MagicMock()
        limiter = MagicMock()
        fetcher = SmartDataFetcher(cache_manager=cache, rate_limiter=limiter)
        assert fetcher.cache_manager is cache
        assert fetcher.rate_limiter is limiter

    @pytest.mark.asyncio
    async def test_fetch_stock_data_routes_to_price(self):
        from backend.data_ingestion.smart_data_fetcher import SmartDataFetcher

        fetcher = SmartDataFetcher()
        result = await fetcher.fetch_stock_data("AAPL", "price")
        assert result["ticker"] == "AAPL"
        assert "price" in result
        assert result["source"] == "mock"

    @pytest.mark.asyncio
    async def test_fetch_stock_data_routes_to_fundamentals(self):
        from backend.data_ingestion.smart_data_fetcher import SmartDataFetcher

        fetcher = SmartDataFetcher()
        result = await fetcher.fetch_stock_data("AAPL", "fundamentals")
        assert result["ticker"] == "AAPL"
        assert "pe_ratio" in result

    @pytest.mark.asyncio
    async def test_fetch_stock_data_routes_to_news(self):
        from backend.data_ingestion.smart_data_fetcher import SmartDataFetcher

        fetcher = SmartDataFetcher()
        result = await fetcher.fetch_stock_data("AAPL", "news")
        assert result["ticker"] == "AAPL"
        assert "articles" in result

    @pytest.mark.asyncio
    async def test_fetch_stock_data_falls_back_to_generic(self):
        from backend.data_ingestion.smart_data_fetcher import SmartDataFetcher

        fetcher = SmartDataFetcher()
        result = await fetcher.fetch_stock_data("AAPL", "unknown_type")
        assert result["ticker"] == "AAPL"
        assert "data" in result

    @pytest.mark.asyncio
    async def test_get_available_sources(self):
        from backend.data_ingestion.smart_data_fetcher import SmartDataFetcher

        fetcher = SmartDataFetcher()
        sources = await fetcher.get_available_sources()
        assert "alpha_vantage" in sources
        assert "finnhub" in sources
        assert "polygon" in sources
        assert "sec_edgar" in sources

    @pytest.mark.asyncio
    async def test_get_source_status_all_available(self):
        from backend.data_ingestion.smart_data_fetcher import SmartDataFetcher

        fetcher = SmartDataFetcher()
        status = await fetcher.get_source_status()
        for source, info in status.items():
            assert info["available"] is True
            assert info["rate_limit_remaining"] == 100

    @pytest.mark.asyncio
    async def test_fetch_earnings_data(self):
        from backend.data_ingestion.smart_data_fetcher import SmartDataFetcher

        fetcher = SmartDataFetcher()
        result = await fetcher.fetch_stock_data("MSFT", "earnings")
        assert result["ticker"] == "MSFT"
        assert "eps_history" in result

    @pytest.mark.asyncio
    async def test_fetch_sentiment_data(self):
        from backend.data_ingestion.smart_data_fetcher import SmartDataFetcher

        fetcher = SmartDataFetcher()
        result = await fetcher.fetch_stock_data("TSLA", "sentiment")
        assert result["ticker"] == "TSLA"
        assert result["sentiment_label"] == "neutral"

    @pytest.mark.asyncio
    async def test_get_smart_fetcher_singleton(self):
        from backend.data_ingestion.smart_data_fetcher import get_smart_fetcher, _smart_fetcher
        import backend.data_ingestion.smart_data_fetcher as sdf_module

        # Reset the global
        sdf_module._smart_fetcher = None
        fetcher1 = await get_smart_fetcher()
        fetcher2 = await get_smart_fetcher()
        assert fetcher1 is fetcher2
        # Clean up
        sdf_module._smart_fetcher = None


# ============================================================================
# RobustAPIClient Tests
# ============================================================================


class TestRobustAPIClient:
    """Tests for RobustAPIClient and its concrete subclasses."""

    @patch("backend.data_ingestion.robust_api_client.settings")
    @patch("backend.data_ingestion.robust_api_client.CircuitBreaker")
    def test_robust_finnhub_constructor(self, mock_cb, mock_settings):
        mock_settings.get_api_key = MagicMock(return_value="robust_fh_key")
        from backend.data_ingestion.robust_api_client import RobustFinnhubClient

        client = RobustFinnhubClient("finnhub")
        assert client.provider_name == "finnhub"
        assert client.api_key == "robust_fh_key"
        assert client.base_url == "https://finnhub.io/api/v1"

    @patch("backend.data_ingestion.robust_api_client.settings")
    @patch("backend.data_ingestion.robust_api_client.CircuitBreaker")
    def test_robust_alpha_vantage_constructor(self, mock_cb, mock_settings):
        mock_settings.get_api_key = MagicMock(return_value="robust_av_key")
        from backend.data_ingestion.robust_api_client import RobustAlphaVantageClient

        client = RobustAlphaVantageClient("alpha_vantage")
        assert client.provider_name == "alpha_vantage"
        assert client.base_url == "https://www.alphavantage.co/query"

    @patch("backend.data_ingestion.robust_api_client.settings")
    @patch("backend.data_ingestion.robust_api_client.CircuitBreaker")
    def test_add_auth_params_finnhub(self, mock_cb, mock_settings):
        mock_settings.get_api_key = MagicMock(return_value="fh_token")
        from backend.data_ingestion.robust_api_client import RobustFinnhubClient

        client = RobustFinnhubClient("finnhub")
        result = client._add_auth_params({"symbol": "AAPL"})
        assert result["token"] == "fh_token"

    @patch("backend.data_ingestion.robust_api_client.settings")
    @patch("backend.data_ingestion.robust_api_client.CircuitBreaker")
    def test_add_auth_params_alpha_vantage(self, mock_cb, mock_settings):
        mock_settings.get_api_key = MagicMock(return_value="av_apikey")
        from backend.data_ingestion.robust_api_client import RobustAlphaVantageClient

        client = RobustAlphaVantageClient("alpha_vantage")
        result = client._add_auth_params({"function": "GLOBAL_QUOTE"})
        assert result["apikey"] == "av_apikey"

    @patch("backend.data_ingestion.robust_api_client.settings")
    @patch("backend.data_ingestion.robust_api_client.CircuitBreaker")
    def test_get_client_info(self, mock_cb, mock_settings):
        mock_settings.get_api_key = MagicMock(return_value="key")
        from backend.data_ingestion.robust_api_client import RobustFinnhubClient

        client = RobustFinnhubClient("finnhub")
        info = client.get_client_info()

        assert info["provider"] == "finnhub"
        assert info["has_api_key"] is True
        assert "async_available" in info
        assert "sync_available" in info
        assert info["preferred_mode"] in ("async", "sync")

    @patch("backend.data_ingestion.robust_api_client.settings")
    @patch("backend.data_ingestion.robust_api_client.CircuitBreaker")
    def test_get_client_info_no_api_key(self, mock_cb, mock_settings):
        mock_settings.get_api_key = MagicMock(return_value=None)
        from backend.data_ingestion.robust_api_client import RobustFinnhubClient

        client = RobustFinnhubClient("finnhub", api_key=None)
        info = client.get_client_info()
        assert info["has_api_key"] is False


# ============================================================================
# BaseAPIClient Tests
# ============================================================================


class TestBaseAPIClient:
    """Tests for BaseAPIClient via a concrete subclass."""

    @patch("backend.data_ingestion.base_client.settings")
    @patch("backend.data_ingestion.base_client.CircuitBreaker")
    def test_constructor_initializes_fields(self, mock_cb, mock_settings):
        mock_settings.get_api_key = MagicMock(return_value="base_key")
        from backend.data_ingestion.alpha_vantage_client import AlphaVantageClient

        client = AlphaVantageClient()
        assert client.session is None
        assert client.timeout is not None
        assert client.provider_name == "alpha_vantage"

    @patch("backend.data_ingestion.base_client.settings")
    @patch("backend.data_ingestion.base_client.CircuitBreaker")
    def test_default_add_auth_params_returns_params_unchanged(self, mock_cb, mock_settings):
        mock_settings.get_api_key = MagicMock(return_value="key")
        from backend.data_ingestion.base_client import BaseAPIClient

        # BaseAPIClient._add_auth_params returns params unchanged
        # Test through a subclass that does NOT override it (SEC Edgar does override _make_request, not _add_auth_params)
        from backend.data_ingestion.sec_edgar_client import SECEdgarClient

        client = SECEdgarClient()
        # SEC Edgar does not override _add_auth_params, so default from base applies
        params = {"key": "value"}
        result = client._add_auth_params(params)
        assert result == {"key": "value"}

    @pytest.mark.asyncio
    @patch("backend.data_ingestion.base_client.cost_monitor")
    @patch("backend.data_ingestion.base_client.get_redis")
    @patch("backend.data_ingestion.base_client.settings")
    @patch("backend.data_ingestion.base_client.CircuitBreaker")
    async def test_make_request_internal_returns_none_on_rate_limit(
        self, mock_cb, mock_settings, mock_get_redis, mock_cost_monitor, mock_redis
    ):
        mock_settings.get_api_key = MagicMock(return_value="key")
        mock_get_redis.return_value = mock_redis
        mock_cost_monitor.check_api_limit = AsyncMock(return_value=False)
        from backend.data_ingestion.alpha_vantage_client import AlphaVantageClient

        client = AlphaVantageClient()
        result = await client._make_request_internal("test_endpoint")
        assert result is None

    @pytest.mark.asyncio
    @patch("backend.data_ingestion.base_client.settings")
    @patch("backend.data_ingestion.base_client.CircuitBreaker")
    async def test_batch_request_handles_exceptions_in_items(self, mock_cb, mock_settings):
        mock_settings.get_api_key = MagicMock(return_value="key")
        from backend.data_ingestion.alpha_vantage_client import AlphaVantageClient

        client = AlphaVantageClient()

        call_count = 0

        async def failing_fetch(item):
            nonlocal call_count
            call_count += 1
            if item == "BAD":
                raise ValueError("fetch failed")
            return {"data": item}

        result = await client.batch_request(
            items=["AAPL", "BAD", "MSFT"],
            fetch_func=failing_fetch,
            batch_size=10,
            delay=0.0,
        )

        assert result["AAPL"] == {"data": "AAPL"}
        assert result["BAD"] is None
        assert result["MSFT"] == {"data": "MSFT"}
        assert call_count == 3


# ============================================================================
# Module __init__ Exports Test
# ============================================================================


class TestModuleExports:
    """Verify the data_ingestion package exports expected names."""

    def test_all_exports_accessible(self):
        from backend.data_ingestion import (
            AlphaVantageClient,
            BaseAPIClient,
            FinnhubClient,
            PolygonClient,
            RobustAPIClient,
            SECEdgarClient,
        )

        assert AlphaVantageClient is not None
        assert FinnhubClient is not None
        assert PolygonClient is not None
        assert SECEdgarClient is not None
        assert BaseAPIClient is not None
        assert RobustAPIClient is not None
