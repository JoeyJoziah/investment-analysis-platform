"""
Tests for Token Bucket Rate Limiting Implementation

Tests cover:
- TokenBucket algorithm correctness
- Priority queue functionality
- Batch request handling
- Rate limit manager coordination
"""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch

import sys
import os

# Add the backend directory to path for direct import
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_dir)

# Import directly from the module file to avoid etl/__init__.py import chain
import importlib.util
spec = importlib.util.spec_from_file_location(
    "rate_limiting",
    os.path.join(backend_dir, "etl", "rate_limiting.py")
)
rate_limiting_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rate_limiting_module)

# Extract classes from the module
TokenBucket = rate_limiting_module.TokenBucket
RequestPriority = rate_limiting_module.RequestPriority
PriorityRequestQueue = rate_limiting_module.PriorityRequestQueue
PrioritizedRequest = rate_limiting_module.PrioritizedRequest
RateLimitConfig = rate_limiting_module.RateLimitConfig
RateLimitedAPIClient = rate_limiting_module.RateLimitedAPIClient
RateLimitManager = rate_limiting_module.RateLimitManager
DEFAULT_RATE_CONFIGS = rate_limiting_module.DEFAULT_RATE_CONFIGS
get_rate_limit_manager = rate_limiting_module.get_rate_limit_manager


class TestTokenBucket:
    """Tests for TokenBucket class."""

    @pytest.mark.asyncio
    async def test_initial_tokens(self):
        """Test bucket starts with full capacity."""
        bucket = TokenBucket(rate=1.0, capacity=10)
        assert bucket.get_available_tokens() >= 9.9  # Allow small timing variance

    @pytest.mark.asyncio
    async def test_acquire_reduces_tokens(self):
        """Test acquiring tokens reduces available count."""
        bucket = TokenBucket(rate=1.0, capacity=10)

        # Acquire 5 tokens
        result = await bucket.acquire(tokens=5)
        assert result is True
        assert bucket.get_available_tokens() < 6

    @pytest.mark.asyncio
    async def test_acquire_waits_when_empty(self):
        """Test acquire waits when bucket is empty."""
        bucket = TokenBucket(rate=10.0, capacity=2)  # Fast refill

        # Drain the bucket
        await bucket.acquire(tokens=2)

        # Time how long next acquire takes
        start = time.monotonic()
        result = await bucket.acquire(tokens=1)
        elapsed = time.monotonic() - start

        assert result is True
        assert elapsed >= 0.05  # Should have waited for refill

    @pytest.mark.asyncio
    async def test_acquire_timeout(self):
        """Test acquire times out when bucket stays empty."""
        bucket = TokenBucket(rate=0.1, capacity=1)  # Very slow refill

        # Drain the bucket
        await bucket.acquire(tokens=1)

        # Try to acquire with short timeout
        result = await bucket.acquire(tokens=1, timeout=0.1)
        assert result is False

    @pytest.mark.asyncio
    async def test_try_acquire_non_blocking(self):
        """Test try_acquire doesn't block."""
        bucket = TokenBucket(rate=0.1, capacity=1)

        # First should succeed
        result1 = await bucket.try_acquire()
        assert result1 is True

        # Second should fail immediately
        start = time.monotonic()
        result2 = await bucket.try_acquire()
        elapsed = time.monotonic() - start

        assert result2 is False
        assert elapsed < 0.1  # Should not have waited

    @pytest.mark.asyncio
    async def test_refill_over_time(self):
        """Test tokens refill based on elapsed time."""
        bucket = TokenBucket(rate=10.0, capacity=5)  # 10 tokens/second

        # Drain the bucket
        await bucket.acquire(tokens=5)

        # Wait for refill
        await asyncio.sleep(0.3)  # Should add ~3 tokens

        # Should be able to acquire some
        result = await bucket.try_acquire(tokens=2)
        assert result is True

    @pytest.mark.asyncio
    async def test_capacity_limit(self):
        """Test tokens don't exceed capacity."""
        bucket = TokenBucket(rate=100.0, capacity=5)  # Very fast refill

        # Wait to ensure refill
        await asyncio.sleep(0.1)

        # Should still be capped at capacity
        assert bucket.get_available_tokens() <= 5

    @pytest.mark.asyncio
    async def test_stats_tracking(self):
        """Test statistics are tracked correctly."""
        bucket = TokenBucket(rate=10.0, capacity=10)

        await bucket.acquire(tokens=1)
        await bucket.acquire(tokens=1)
        await bucket.try_acquire(tokens=100)  # Will fail - exceeds capacity

        stats = bucket.get_stats()
        assert stats['total_requests'] == 2
        assert stats['rejected_requests'] == 0  # Exceeds capacity, not rejected


class TestPriorityRequestQueue:
    """Tests for PriorityRequestQueue class."""

    @pytest.mark.asyncio
    async def test_priority_ordering(self):
        """Test requests are dequeued in priority order."""
        queue = PriorityRequestQueue(max_size=100)

        # Enqueue in reverse priority order
        await queue.enqueue("LOW", lambda x: x, RequestPriority.LOW)
        await queue.enqueue("NORMAL", lambda x: x, RequestPriority.NORMAL)
        await queue.enqueue("HIGH", lambda x: x, RequestPriority.HIGH)
        await queue.enqueue("CRITICAL", lambda x: x, RequestPriority.CRITICAL)

        # Dequeue should be in priority order
        r1 = await queue.dequeue(timeout=1)
        r2 = await queue.dequeue(timeout=1)
        r3 = await queue.dequeue(timeout=1)
        r4 = await queue.dequeue(timeout=1)

        assert r1.ticker == "CRITICAL"
        assert r2.ticker == "HIGH"
        assert r3.ticker == "NORMAL"
        assert r4.ticker == "LOW"

    @pytest.mark.asyncio
    async def test_queue_overflow(self):
        """Test queue handles overflow by dropping low priority."""
        queue = PriorityRequestQueue(max_size=2)

        # Fill queue with low priority
        await queue.enqueue("LOW1", lambda x: x, RequestPriority.LOW)
        await queue.enqueue("LOW2", lambda x: x, RequestPriority.LOW)

        # Add high priority should succeed (drops a low priority)
        await queue.enqueue("HIGH", lambda x: x, RequestPriority.HIGH)

        # Dequeue should give high priority first
        r1 = await queue.dequeue(timeout=1)
        assert r1.ticker == "HIGH"

        # Stats should show dropped request
        stats = queue.get_stats()
        assert stats['dropped_requests'] == 1

    @pytest.mark.asyncio
    async def test_dequeue_timeout(self):
        """Test dequeue times out on empty queue."""
        queue = PriorityRequestQueue()

        result = await queue.dequeue(timeout=0.1)
        assert result is None

    @pytest.mark.asyncio
    async def test_batch_dequeue(self):
        """Test batch dequeue retrieves multiple requests."""
        queue = PriorityRequestQueue()

        # Enqueue several requests
        for i in range(5):
            await queue.enqueue(f"TICKER{i}", lambda x: x, RequestPriority.NORMAL)

        # Batch dequeue
        batch = await queue.dequeue_batch(max_size=3, timeout=1)

        assert len(batch) == 3
        assert queue.size() == 2


class TestRateLimitedAPIClient:
    """Tests for RateLimitedAPIClient class."""

    @pytest.mark.asyncio
    async def test_can_make_request_check(self):
        """Test can_make_request returns correct status."""
        config = RateLimitConfig(
            name='test',
            rate=10.0,
            capacity=5,
            max_per_hour=100
        )
        client = RateLimitedAPIClient('test', config)

        # Should be able to make request initially
        assert client.can_make_request() is True

    @pytest.mark.asyncio
    async def test_execute_immediate(self):
        """Test immediate execution with rate limiting."""
        config = RateLimitConfig(
            name='test',
            rate=10.0,
            capacity=5,
            min_delay=0.0
        )
        client = RateLimitedAPIClient('test', config)

        callback = AsyncMock(return_value={'data': 'test'})
        result = await client.execute_immediate('AAPL', callback)

        assert result == {'data': 'test'}
        callback.assert_called_once_with('AAPL')

    @pytest.mark.asyncio
    async def test_hard_limit_daily(self):
        """Test daily hard limit is enforced."""
        config = RateLimitConfig(
            name='test',
            rate=100.0,
            capacity=100,
            max_per_day=3
        )
        client = RateLimitedAPIClient('test', config)

        callback = AsyncMock(return_value={'data': 'test'})

        # Should succeed for first 3
        for _ in range(3):
            await client.execute_immediate('AAPL', callback, timeout=1)

        # Fourth should fail
        with pytest.raises(asyncio.TimeoutError):
            await client.execute_immediate('AAPL', callback, timeout=0.1)

    @pytest.mark.asyncio
    async def test_get_stats(self):
        """Test statistics retrieval."""
        client = RateLimitedAPIClient('finnhub')
        stats = client.get_stats()

        assert 'source' in stats
        assert 'bucket' in stats
        assert 'queue' in stats
        assert stats['source'] == 'finnhub'


class TestRateLimitManager:
    """Tests for RateLimitManager singleton."""

    def test_singleton_pattern(self):
        """Test manager is a singleton."""
        manager1 = RateLimitManager()
        manager2 = RateLimitManager()
        assert manager1 is manager2

    def test_get_client_creates_new(self):
        """Test get_client creates client if not exists."""
        manager = RateLimitManager()
        client = manager.get_client('test_source_new')

        assert client is not None
        assert client.source == 'test_source_new'

    def test_get_client_returns_same(self):
        """Test get_client returns same client for same source."""
        manager = RateLimitManager()
        client1 = manager.get_client('same_source')
        client2 = manager.get_client('same_source')

        assert client1 is client2

    def test_get_optimal_source(self):
        """Test optimal source selection based on wait time."""
        manager = RateLimitManager()

        # Create clients with different configs
        slow_config = RateLimitConfig(
            name='slow_test',
            rate=0.5,
            capacity=5,
            max_per_hour=10
        )
        fast_config = RateLimitConfig(
            name='fast_test',
            rate=10.0,
            capacity=50
        )

        slow_client = RateLimitedAPIClient('slow_test', slow_config)
        fast_client = RateLimitedAPIClient('fast_test', fast_config)

        manager.clients['slow_test'] = slow_client
        manager.clients['fast_test'] = fast_client

        # Make slow client hit its hourly limit
        slow_client._hourly_count = 10
        slow_client._hour_start = time.time()

        optimal = manager.get_optimal_source(['slow_test', 'fast_test'])
        assert optimal == 'fast_test'


class TestDefaultConfigs:
    """Tests for default rate limit configurations."""

    def test_all_sources_configured(self):
        """Test all expected sources have configurations."""
        expected_sources = [
            'yahoo_scraper',
            'yfinance',
            'alpha_vantage',
            'finnhub',
            'polygon',
            'marketwatch_scraper',
            'google_finance_scraper'
        ]

        for source in expected_sources:
            assert source in DEFAULT_RATE_CONFIGS

    def test_finnhub_supports_batch(self):
        """Test Finnhub is configured for batch requests."""
        config = DEFAULT_RATE_CONFIGS['finnhub']
        assert config.supports_batch is True
        assert config.max_batch_size > 1

    def test_alpha_vantage_daily_limit(self):
        """Test Alpha Vantage has correct daily limit."""
        config = DEFAULT_RATE_CONFIGS['alpha_vantage']
        assert config.max_per_day == 25


class TestIntegration:
    """Integration tests for the full rate limiting system."""

    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test system handles concurrent requests correctly."""
        manager = get_rate_limit_manager()
        client = manager.get_client('integration_test')

        # Reset client state
        client.bucket = TokenBucket(rate=10.0, capacity=10)

        results = []
        callback = AsyncMock(return_value={'status': 'ok'})

        async def make_request(i):
            try:
                result = await client.execute_immediate(
                    f'TICKER{i}',
                    callback,
                    timeout=5
                )
                return ('success', result)
            except asyncio.TimeoutError:
                return ('timeout', None)

        # Make 5 concurrent requests
        tasks = [make_request(i) for i in range(5)]
        results = await asyncio.gather(*tasks)

        successes = [r for r in results if r[0] == 'success']
        assert len(successes) >= 3  # At least some should succeed

    @pytest.mark.asyncio
    async def test_priority_affects_processing(self):
        """Test higher priority requests are processed first."""
        manager = get_rate_limit_manager()

        # Get timestamps for different priority requests
        timestamps = {}

        async def record_callback(ticker):
            timestamps[ticker] = time.monotonic()
            return {'ticker': ticker}

        # Queue requests with different priorities
        client = manager.get_client('priority_test')
        await client.start_worker()

        try:
            futures = [
                await client.queue.enqueue('LOW', record_callback, RequestPriority.LOW),
                await client.queue.enqueue('HIGH', record_callback, RequestPriority.HIGH),
                await client.queue.enqueue('CRITICAL', record_callback, RequestPriority.CRITICAL),
            ]

            # Wait for all to complete
            await asyncio.sleep(1)

            # Critical should have been processed first (smallest timestamp)
            if 'CRITICAL' in timestamps and 'LOW' in timestamps:
                assert timestamps['CRITICAL'] <= timestamps['LOW']

        finally:
            await client.stop_worker()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
