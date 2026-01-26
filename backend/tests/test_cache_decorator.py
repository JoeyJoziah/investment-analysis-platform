"""
Tests for the cache_with_ttl decorator in backend/utils/cache.py

Verifies that the cache decorator:
1. Caches async function results in Redis
2. Returns cached values on subsequent calls
3. Generates proper cache keys
4. Handles different argument types (dates, enums, Pydantic models)
5. Falls back gracefully when Redis is unavailable
"""

import pytest
import asyncio
from datetime import date, datetime
from enum import Enum
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from pydantic import BaseModel


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class StockData(BaseModel):
    ticker: str
    price: float
    timestamp: datetime


class TestCacheWithTTL:
    """Test suite for the cache_with_ttl decorator."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        redis = AsyncMock()
        redis.get = AsyncMock(return_value=None)
        redis.setex = AsyncMock()
        return redis

    @pytest.mark.asyncio
    async def test_cache_miss_then_hit(self, mock_redis):
        """Test that cache stores result on miss and returns it on hit."""
        from backend.utils.cache import cache_with_ttl

        call_count = 0

        @cache_with_ttl(ttl=300)
        async def get_stock_price(ticker: str) -> dict:
            nonlocal call_count
            call_count += 1
            return {"ticker": ticker, "price": 100.0}

        # Patch get_redis to return our mock
        with patch('backend.utils.cache.get_redis', return_value=mock_redis):
            # First call - should execute function and cache result
            result1 = await get_stock_price("AAPL")
            assert result1 == {"ticker": "AAPL", "price": 100.0}
            assert call_count == 1

            # Verify setex was called to cache the result
            assert mock_redis.setex.called

            # Simulate cache hit by setting get to return cached value
            mock_redis.get = AsyncMock(
                return_value='{"ticker": "AAPL", "price": 100.0}'
            )

            # Second call - should return cached value
            result2 = await get_stock_price("AAPL")
            assert result2 == {"ticker": "AAPL", "price": 100.0}
            # Function should not have been called again
            assert call_count == 1

    @pytest.mark.asyncio
    async def test_different_args_different_cache_keys(self, mock_redis):
        """Test that different arguments produce different cache keys."""
        from backend.utils.cache import cache_with_ttl

        call_count = 0

        @cache_with_ttl(ttl=300)
        async def get_stock_price(ticker: str) -> dict:
            nonlocal call_count
            call_count += 1
            return {"ticker": ticker, "price": 100.0 + call_count}

        with patch('backend.utils.cache.get_redis', return_value=mock_redis):
            # Call with different tickers
            result1 = await get_stock_price("AAPL")
            result2 = await get_stock_price("GOOGL")

            # Both should execute the function
            assert call_count == 2
            # setex should be called twice with different keys
            assert mock_redis.setex.call_count == 2

            # Verify different cache keys were used
            calls = mock_redis.setex.call_args_list
            key1 = calls[0][0][0]
            key2 = calls[1][0][0]
            assert key1 != key2
            assert "AAPL" in key1 or "get_stock_price" in key1
            assert "GOOGL" in key2 or "get_stock_price" in key2

    @pytest.mark.asyncio
    async def test_handles_date_arguments(self, mock_redis):
        """Test that date arguments are properly serialized in cache key."""
        from backend.utils.cache import cache_with_ttl

        @cache_with_ttl(ttl=300)
        async def get_daily_data(target_date: date) -> dict:
            return {"date": target_date.isoformat(), "value": 42}

        with patch('backend.utils.cache.get_redis', return_value=mock_redis):
            result = await get_daily_data(date(2024, 1, 15))
            assert result == {"date": "2024-01-15", "value": 42}
            assert mock_redis.setex.called

    @pytest.mark.asyncio
    async def test_handles_enum_arguments(self, mock_redis):
        """Test that enum arguments are properly serialized in cache key."""
        from backend.utils.cache import cache_with_ttl

        @cache_with_ttl(ttl=300)
        async def get_recommendations(risk_level: RiskLevel) -> dict:
            return {"risk": risk_level.value, "count": 5}

        with patch('backend.utils.cache.get_redis', return_value=mock_redis):
            result = await get_recommendations(RiskLevel.HIGH)
            assert result == {"risk": "high", "count": 5}
            assert mock_redis.setex.called

    @pytest.mark.asyncio
    async def test_handles_pydantic_model_response(self, mock_redis):
        """Test that Pydantic model responses are properly serialized."""
        from backend.utils.cache import cache_with_ttl

        @cache_with_ttl(ttl=300)
        async def get_stock(ticker: str) -> StockData:
            return StockData(
                ticker=ticker,
                price=150.0,
                timestamp=datetime(2024, 1, 15, 10, 30, 0)
            )

        with patch('backend.utils.cache.get_redis', return_value=mock_redis):
            result = await get_stock("AAPL")
            assert result.ticker == "AAPL"
            assert result.price == 150.0
            assert mock_redis.setex.called

    @pytest.mark.asyncio
    async def test_handles_list_of_pydantic_models(self, mock_redis):
        """Test that lists of Pydantic models are properly serialized."""
        from backend.utils.cache import cache_with_ttl

        @cache_with_ttl(ttl=300)
        async def get_stocks(tickers: List[str]) -> List[StockData]:
            return [
                StockData(ticker=t, price=100.0, timestamp=datetime.now())
                for t in tickers
            ]

        with patch('backend.utils.cache.get_redis', return_value=mock_redis):
            result = await get_stocks(["AAPL", "GOOGL"])
            assert len(result) == 2
            assert mock_redis.setex.called

    @pytest.mark.asyncio
    async def test_excludes_db_session_from_cache_key(self, mock_redis):
        """Test that AsyncSession objects are excluded from cache key."""
        from backend.utils.cache import cache_with_ttl

        # Create a mock that looks like an AsyncSession
        mock_session = MagicMock()
        mock_session.__class__.__name__ = "AsyncSession"

        @cache_with_ttl(ttl=300)
        async def get_data(ticker: str, db: MagicMock) -> dict:
            return {"ticker": ticker}

        with patch('backend.utils.cache.get_redis', return_value=mock_redis):
            result = await get_data("AAPL", mock_session)
            assert result == {"ticker": "AAPL"}

            # Get the cache key used
            cache_key = mock_redis.setex.call_args[0][0]
            # Cache key should contain ticker but not session reference
            assert "AAPL" in cache_key
            assert "AsyncSession" not in cache_key

    @pytest.mark.asyncio
    async def test_fallback_on_redis_error(self):
        """Test that function executes normally when Redis fails."""
        from backend.utils.cache import cache_with_ttl

        call_count = 0

        @cache_with_ttl(ttl=300)
        async def get_data(ticker: str) -> dict:
            nonlocal call_count
            call_count += 1
            return {"ticker": ticker}

        # Create a mock that raises an exception
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(side_effect=Exception("Redis connection failed"))

        with patch('backend.utils.cache.get_redis', return_value=mock_redis):
            # Should still return result despite Redis error
            result = await get_data("AAPL")
            assert result == {"ticker": "AAPL"}
            assert call_count == 1

    @pytest.mark.asyncio
    async def test_custom_prefix(self, mock_redis):
        """Test that custom prefix is used in cache key."""
        from backend.utils.cache import cache_with_ttl

        @cache_with_ttl(ttl=300, prefix="custom.prefix")
        async def get_data(ticker: str) -> dict:
            return {"ticker": ticker}

        with patch('backend.utils.cache.get_redis', return_value=mock_redis):
            await get_data("AAPL")

            cache_key = mock_redis.setex.call_args[0][0]
            assert "custom.prefix" in cache_key

    @pytest.mark.asyncio
    async def test_ttl_is_passed_to_redis(self, mock_redis):
        """Test that the TTL value is correctly passed to Redis setex."""
        from backend.utils.cache import cache_with_ttl

        @cache_with_ttl(ttl=3600)  # 1 hour
        async def get_data() -> dict:
            return {"value": 42}

        with patch('backend.utils.cache.get_redis', return_value=mock_redis):
            await get_data()

            # Verify setex was called with correct TTL
            call_args = mock_redis.setex.call_args[0]
            ttl_arg = call_args[1]
            assert ttl_arg == 3600


class TestCacheKeyGeneration:
    """Test suite for cache key generation logic."""

    def test_key_with_various_types(self):
        """Test cache key generation with various argument types."""
        from backend.utils.cache import get_cache_key

        # Basic types
        key1 = get_cache_key("AAPL", 100, True)
        assert isinstance(key1, str)
        assert len(key1) == 16  # SHA256 hash truncated to 16 chars

        # With kwargs
        key2 = get_cache_key("AAPL", limit=50)
        assert isinstance(key2, str)

        # Different args should produce different keys
        key3 = get_cache_key("GOOGL", 100, True)
        assert key1 != key3
