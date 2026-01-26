"""
Token Bucket Rate Limiting with Priority Queue Support

Implements efficient rate limiting for API calls with:
- Token bucket algorithm for smooth rate limiting
- Priority queue for request ordering
- Batch request support for Finnhub and other APIs
- Async support for non-blocking rate limiting

Target: 70-80% reduction in API overhead
"""

import asyncio
import time
import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Callable, Any, TypeVar, Generic
from collections import deque
import heapq
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

T = TypeVar('T')


class RequestPriority(IntEnum):
    """Priority levels for rate-limited requests"""
    CRITICAL = 0    # High-value stocks, real-time needs
    HIGH = 1        # Top 100 stocks by volume
    NORMAL = 2      # Standard requests
    LOW = 3         # Background/batch requests
    BULK = 4        # Mass data refresh


@dataclass
class RateLimitConfig:
    """Configuration for a rate-limited source"""
    name: str
    rate: float                    # Tokens per second
    capacity: int                  # Max burst capacity
    max_per_hour: Optional[int] = None
    max_per_day: Optional[int] = None
    min_delay: float = 0.0        # Minimum delay between requests
    supports_batch: bool = False   # Whether API supports batch requests
    max_batch_size: int = 1       # Maximum symbols per batch request


# Default configurations for each API source
DEFAULT_RATE_CONFIGS: Dict[str, RateLimitConfig] = {
    'yahoo_scraper': RateLimitConfig(
        name='yahoo_scraper',
        rate=0.083,           # 5 requests per minute
        capacity=10,
        max_per_hour=300,
        min_delay=2.0
    ),
    'yfinance': RateLimitConfig(
        name='yfinance',
        rate=0.028,           # ~1.7 requests per minute
        capacity=5,
        max_per_hour=100,
        min_delay=3.0
    ),
    'alpha_vantage': RateLimitConfig(
        name='alpha_vantage',
        rate=0.00029,         # 25 per day = 0.00029 per second
        capacity=5,
        max_per_day=25,
        min_delay=60.0        # Conservative delay for precious quota
    ),
    'finnhub': RateLimitConfig(
        name='finnhub',
        rate=1.0,             # 60 per minute = 1 per second
        capacity=30,          # Allow burst
        max_per_hour=60,
        min_delay=1.0,
        supports_batch=True,
        max_batch_size=50     # Finnhub supports batch quote requests
    ),
    'polygon': RateLimitConfig(
        name='polygon',
        rate=0.083,           # 5 per minute free tier
        capacity=5,
        max_per_hour=60,      # Conservative for free tier
        min_delay=5.0
    ),
    'marketwatch_scraper': RateLimitConfig(
        name='marketwatch_scraper',
        rate=0.056,           # ~3.3 per minute
        capacity=5,
        max_per_hour=200,
        min_delay=4.0
    ),
    'google_finance_scraper': RateLimitConfig(
        name='google_finance_scraper',
        rate=0.042,           # 2.5 per minute
        capacity=5,
        max_per_hour=150,
        min_delay=5.0
    )
}


class TokenBucket:
    """
    Token Bucket Rate Limiter

    Implements the token bucket algorithm for smooth rate limiting:
    - Tokens are added at a fixed rate
    - Requests consume tokens
    - Requests wait or fail when bucket is empty
    - Supports burst capacity for handling spikes

    Thread-safe and async-compatible.
    """

    def __init__(self, rate: float, capacity: int, initial_tokens: Optional[int] = None):
        """
        Initialize token bucket.

        Args:
            rate: Tokens added per second
            capacity: Maximum tokens (burst capacity)
            initial_tokens: Starting tokens (defaults to capacity)
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = initial_tokens if initial_tokens is not None else capacity
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()

        # Statistics
        self.total_requests = 0
        self.total_wait_time = 0.0
        self.rejected_requests = 0

    async def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_update

        # Add tokens based on elapsed time
        tokens_to_add = elapsed * self.rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_update = now

    async def acquire(self, tokens: int = 1, timeout: Optional[float] = None) -> bool:
        """
        Acquire tokens from the bucket.

        Args:
            tokens: Number of tokens to acquire
            timeout: Maximum time to wait (None = wait forever)

        Returns:
            True if tokens acquired, False if timed out
        """
        if tokens > self.capacity:
            logger.warning(f"Requested {tokens} tokens exceeds capacity {self.capacity}")
            return False

        start_time = time.monotonic()

        async with self._lock:
            await self._refill()

            while self.tokens < tokens:
                if timeout is not None:
                    elapsed = time.monotonic() - start_time
                    if elapsed >= timeout:
                        self.rejected_requests += 1
                        return False

                # Calculate wait time for tokens
                tokens_needed = tokens - self.tokens
                wait_time = tokens_needed / self.rate if self.rate > 0 else 0.1

                # Don't wait longer than remaining timeout
                if timeout is not None:
                    remaining = timeout - (time.monotonic() - start_time)
                    wait_time = min(wait_time, remaining)

                if wait_time > 0:
                    # Release lock while waiting
                    self._lock.release()
                    try:
                        await asyncio.sleep(wait_time)
                    finally:
                        await self._lock.acquire()

                    await self._refill()

            # Consume tokens
            self.tokens -= tokens
            self.total_requests += 1
            self.total_wait_time += time.monotonic() - start_time

            return True

    async def try_acquire(self, tokens: int = 1) -> bool:
        """
        Try to acquire tokens without waiting.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            True if tokens acquired immediately, False otherwise
        """
        return await self.acquire(tokens, timeout=0)

    def get_available_tokens(self) -> float:
        """Get current available tokens (approximate, not thread-safe for reading)."""
        elapsed = time.monotonic() - self.last_update
        tokens = self.tokens + (elapsed * self.rate)
        return min(self.capacity, tokens)

    def get_wait_time(self, tokens: int = 1) -> float:
        """Estimate wait time for acquiring tokens."""
        available = self.get_available_tokens()
        if available >= tokens:
            return 0.0
        return (tokens - available) / self.rate if self.rate > 0 else float('inf')

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about bucket usage."""
        return {
            'rate': self.rate,
            'capacity': self.capacity,
            'current_tokens': self.get_available_tokens(),
            'total_requests': self.total_requests,
            'rejected_requests': self.rejected_requests,
            'total_wait_time': self.total_wait_time,
            'avg_wait_time': self.total_wait_time / max(1, self.total_requests)
        }


@dataclass(order=True)
class PrioritizedRequest:
    """A request in the priority queue."""
    priority: int
    timestamp: float = field(compare=False)
    request_id: str = field(compare=False)
    ticker: str = field(compare=False)
    callback: Callable = field(compare=False)
    future: asyncio.Future = field(compare=False, default=None)


class PriorityRequestQueue:
    """
    Priority queue for rate-limited requests.

    Features:
    - Priority-based ordering
    - Queue overflow handling
    - Request timeout support
    - Batch coalescing for supported APIs
    """

    def __init__(self, max_size: int = 10000, default_timeout: float = 300.0):
        """
        Initialize priority queue.

        Args:
            max_size: Maximum queue size
            default_timeout: Default request timeout in seconds
        """
        self.max_size = max_size
        self.default_timeout = default_timeout
        self._queue: List[PrioritizedRequest] = []
        self._lock = asyncio.Lock()
        self._not_empty = asyncio.Event()
        self._request_counter = 0

        # Statistics
        self.total_enqueued = 0
        self.total_dequeued = 0
        self.dropped_requests = 0

    async def enqueue(
        self,
        ticker: str,
        callback: Callable,
        priority: RequestPriority = RequestPriority.NORMAL,
        timeout: Optional[float] = None
    ) -> asyncio.Future:
        """
        Enqueue a request.

        Args:
            ticker: Stock ticker symbol
            callback: Async function to call when request is processed
            priority: Request priority
            timeout: Request timeout (None uses default)

        Returns:
            Future that will contain the result
        """
        async with self._lock:
            if len(self._queue) >= self.max_size:
                # Drop lowest priority request if at capacity
                if self._queue and self._queue[-1].priority > priority:
                    dropped = heapq.heappop(self._queue)
                    if dropped.future and not dropped.future.done():
                        dropped.future.set_exception(
                            asyncio.TimeoutError("Request dropped due to queue overflow")
                        )
                    self.dropped_requests += 1
                else:
                    raise asyncio.QueueFull(f"Queue full and request priority too low")

            self._request_counter += 1
            request_id = f"{ticker}_{self._request_counter}_{time.time()}"

            loop = asyncio.get_event_loop()
            future = loop.create_future()

            request = PrioritizedRequest(
                priority=priority,
                timestamp=time.time(),
                request_id=request_id,
                ticker=ticker,
                callback=callback,
                future=future
            )

            heapq.heappush(self._queue, request)
            self.total_enqueued += 1
            self._not_empty.set()

            return future

    async def dequeue(self, timeout: Optional[float] = None) -> Optional[PrioritizedRequest]:
        """
        Dequeue highest priority request.

        Args:
            timeout: Maximum time to wait for a request

        Returns:
            PrioritizedRequest or None if timeout
        """
        start_time = time.monotonic()

        while True:
            async with self._lock:
                if self._queue:
                    request = heapq.heappop(self._queue)
                    if not self._queue:
                        self._not_empty.clear()
                    self.total_dequeued += 1
                    return request

            if timeout is not None:
                remaining = timeout - (time.monotonic() - start_time)
                if remaining <= 0:
                    return None

                try:
                    await asyncio.wait_for(self._not_empty.wait(), timeout=remaining)
                except asyncio.TimeoutError:
                    return None
            else:
                await self._not_empty.wait()

    async def dequeue_batch(
        self,
        max_size: int,
        timeout: Optional[float] = None
    ) -> List[PrioritizedRequest]:
        """
        Dequeue multiple requests for batch processing.

        Args:
            max_size: Maximum number of requests to dequeue
            timeout: Maximum time to wait for at least one request

        Returns:
            List of requests
        """
        # Wait for at least one request
        first = await self.dequeue(timeout)
        if first is None:
            return []

        batch = [first]

        # Greedily collect more requests without waiting
        async with self._lock:
            while len(batch) < max_size and self._queue:
                request = heapq.heappop(self._queue)
                batch.append(request)
                self.total_dequeued += 1

            if not self._queue:
                self._not_empty.clear()

        return batch

    def size(self) -> int:
        """Get current queue size."""
        return len(self._queue)

    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        return {
            'current_size': len(self._queue),
            'max_size': self.max_size,
            'total_enqueued': self.total_enqueued,
            'total_dequeued': self.total_dequeued,
            'dropped_requests': self.dropped_requests
        }


class RateLimitedAPIClient:
    """
    Rate-limited API client with priority queue.

    Combines token bucket rate limiting with priority queuing
    for efficient API usage.
    """

    def __init__(
        self,
        source: str,
        config: Optional[RateLimitConfig] = None,
        queue_size: int = 5000
    ):
        """
        Initialize rate-limited client.

        Args:
            source: API source name
            config: Rate limit configuration (uses default if None)
            queue_size: Maximum queue size
        """
        self.source = source
        self.config = config or DEFAULT_RATE_CONFIGS.get(
            source,
            RateLimitConfig(name=source, rate=0.5, capacity=5)
        )

        self.bucket = TokenBucket(
            rate=self.config.rate,
            capacity=self.config.capacity
        )

        self.queue = PriorityRequestQueue(max_size=queue_size)
        self._worker_task: Optional[asyncio.Task] = None
        self._running = False
        self._last_request_time = 0.0

        # Hourly/daily tracking for hard limits
        self._hourly_count = 0
        self._daily_count = 0
        self._hour_start = time.time()
        self._day_start = time.time()

        logger.info(f"Initialized RateLimitedAPIClient for {source} "
                   f"(rate={self.config.rate}/s, capacity={self.config.capacity})")

    def _check_hard_limits(self) -> bool:
        """Check hourly/daily hard limits."""
        now = time.time()

        # Reset hourly counter
        if now - self._hour_start > 3600:
            self._hourly_count = 0
            self._hour_start = now

        # Reset daily counter
        if now - self._day_start > 86400:
            self._daily_count = 0
            self._day_start = now

        # Check limits
        if self.config.max_per_hour and self._hourly_count >= self.config.max_per_hour:
            return False
        if self.config.max_per_day and self._daily_count >= self.config.max_per_day:
            return False

        return True

    def _record_request(self, count: int = 1):
        """Record request for hard limit tracking."""
        self._hourly_count += count
        self._daily_count += count
        self._last_request_time = time.time()

    async def start_worker(self):
        """Start the queue worker."""
        if self._worker_task is not None:
            return

        self._running = True
        self._worker_task = asyncio.create_task(self._process_queue())
        logger.info(f"Started rate limit worker for {self.source}")

    async def stop_worker(self):
        """Stop the queue worker."""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None
        logger.info(f"Stopped rate limit worker for {self.source}")

    async def _process_queue(self):
        """Process queued requests."""
        while self._running:
            try:
                # Check hard limits
                if not self._check_hard_limits():
                    # Wait until next hour/day
                    await asyncio.sleep(60)
                    continue

                # Batch processing for supported APIs
                if self.config.supports_batch:
                    await self._process_batch()
                else:
                    await self._process_single()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in queue worker for {self.source}: {e}")
                await asyncio.sleep(1)

    async def _process_single(self):
        """Process a single request."""
        request = await self.queue.dequeue(timeout=1.0)
        if request is None:
            return

        # Enforce minimum delay
        elapsed = time.time() - self._last_request_time
        if elapsed < self.config.min_delay:
            await asyncio.sleep(self.config.min_delay - elapsed)

        # Acquire token
        if not await self.bucket.acquire(timeout=30):
            if request.future and not request.future.done():
                request.future.set_exception(
                    asyncio.TimeoutError("Rate limit timeout")
                )
            return

        # Execute request
        try:
            result = await request.callback(request.ticker)
            self._record_request()
            if request.future and not request.future.done():
                request.future.set_result(result)
        except Exception as e:
            if request.future and not request.future.done():
                request.future.set_exception(e)

    async def _process_batch(self):
        """Process a batch of requests."""
        requests = await self.queue.dequeue_batch(
            max_size=self.config.max_batch_size,
            timeout=1.0
        )
        if not requests:
            return

        # Enforce minimum delay
        elapsed = time.time() - self._last_request_time
        if elapsed < self.config.min_delay:
            await asyncio.sleep(self.config.min_delay - elapsed)

        # Acquire tokens for batch
        if not await self.bucket.acquire(tokens=len(requests), timeout=30):
            # Put requests back or fail them
            for request in requests:
                if request.future and not request.future.done():
                    request.future.set_exception(
                        asyncio.TimeoutError("Rate limit timeout for batch")
                    )
            return

        # Get all tickers
        tickers = [r.ticker for r in requests]

        # Execute batch callback (assumes first request has batch callback)
        try:
            # The callback should accept a list of tickers for batch processing
            results = await requests[0].callback(tickers)
            self._record_request(len(requests))

            # Distribute results to individual futures
            if isinstance(results, dict):
                for request in requests:
                    result = results.get(request.ticker)
                    if request.future and not request.future.done():
                        request.future.set_result(result)
            elif isinstance(results, list):
                for request, result in zip(requests, results):
                    if request.future and not request.future.done():
                        request.future.set_result(result)
            else:
                # Single result for all
                for request in requests:
                    if request.future and not request.future.done():
                        request.future.set_result(results)

        except Exception as e:
            for request in requests:
                if request.future and not request.future.done():
                    request.future.set_exception(e)

    async def execute(
        self,
        ticker: str,
        callback: Callable,
        priority: RequestPriority = RequestPriority.NORMAL,
        timeout: Optional[float] = 60.0
    ) -> Any:
        """
        Execute a rate-limited request.

        Args:
            ticker: Stock ticker symbol
            callback: Async function to call
            priority: Request priority
            timeout: Maximum time to wait

        Returns:
            Result from callback
        """
        # Start worker if not running
        if self._worker_task is None:
            await self.start_worker()

        future = await self.queue.enqueue(ticker, callback, priority)

        if timeout:
            return await asyncio.wait_for(future, timeout=timeout)
        return await future

    async def execute_immediate(
        self,
        ticker: str,
        callback: Callable,
        timeout: Optional[float] = 30.0
    ) -> Any:
        """
        Execute a request immediately if rate limit allows.

        Args:
            ticker: Stock ticker symbol
            callback: Async function to call
            timeout: Maximum time to wait for rate limit

        Returns:
            Result from callback

        Raises:
            asyncio.TimeoutError if rate limit not available
        """
        if not self._check_hard_limits():
            raise asyncio.TimeoutError("Hard rate limit exceeded")

        # Enforce minimum delay
        elapsed = time.time() - self._last_request_time
        if elapsed < self.config.min_delay:
            await asyncio.sleep(self.config.min_delay - elapsed)

        if not await self.bucket.acquire(timeout=timeout):
            raise asyncio.TimeoutError("Rate limit timeout")

        try:
            result = await callback(ticker)
            self._record_request()
            return result
        except Exception:
            raise

    def can_make_request(self) -> bool:
        """Check if a request can be made immediately."""
        if not self._check_hard_limits():
            return False
        return self.bucket.get_available_tokens() >= 1

    def get_wait_time(self) -> float:
        """Get estimated wait time for next request."""
        if not self._check_hard_limits():
            # Calculate time until next hour/day
            now = time.time()
            if self.config.max_per_hour and self._hourly_count >= self.config.max_per_hour:
                return 3600 - (now - self._hour_start)
            if self.config.max_per_day and self._daily_count >= self.config.max_per_day:
                return 86400 - (now - self._day_start)

        return max(0, self.bucket.get_wait_time())

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            'source': self.source,
            'bucket': self.bucket.get_stats(),
            'queue': self.queue.get_stats(),
            'hourly_count': self._hourly_count,
            'daily_count': self._daily_count,
            'max_per_hour': self.config.max_per_hour,
            'max_per_day': self.config.max_per_day,
            'supports_batch': self.config.supports_batch,
            'max_batch_size': self.config.max_batch_size
        }


class RateLimitManager:
    """
    Centralized rate limit management for all API sources.

    Provides:
    - Single point of access for all rate limiters
    - Cross-source coordination
    - Global statistics
    """

    _instance: Optional['RateLimitManager'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.clients: Dict[str, RateLimitedAPIClient] = {}
        self._lock = asyncio.Lock()
        self._initialized = True

        logger.info("Initialized RateLimitManager")

    def get_client(self, source: str) -> RateLimitedAPIClient:
        """Get or create a rate-limited client for a source."""
        if source not in self.clients:
            config = DEFAULT_RATE_CONFIGS.get(source)
            self.clients[source] = RateLimitedAPIClient(source, config)
        return self.clients[source]

    async def execute(
        self,
        source: str,
        ticker: str,
        callback: Callable,
        priority: RequestPriority = RequestPriority.NORMAL,
        timeout: Optional[float] = 60.0
    ) -> Any:
        """Execute a rate-limited request for a source."""
        client = self.get_client(source)
        return await client.execute(ticker, callback, priority, timeout)

    def can_make_request(self, source: str) -> bool:
        """Check if a request can be made to a source."""
        client = self.get_client(source)
        return client.can_make_request()

    def get_optimal_source(self, sources: List[str]) -> Optional[str]:
        """Get the source with shortest wait time."""
        best_source = None
        best_wait = float('inf')

        for source in sources:
            client = self.get_client(source)
            wait = client.get_wait_time()
            if wait < best_wait:
                best_wait = wait
                best_source = source

        return best_source

    async def start_all_workers(self):
        """Start workers for all clients."""
        for client in self.clients.values():
            await client.start_worker()

    async def stop_all_workers(self):
        """Stop workers for all clients."""
        for client in self.clients.values():
            await client.stop_worker()

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all sources."""
        return {source: client.get_stats() for source, client in self.clients.items()}

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        total_requests = 0
        total_wait_time = 0.0
        total_rejected = 0
        total_dropped = 0

        for client in self.clients.values():
            stats = client.get_stats()
            bucket_stats = stats.get('bucket', {})
            queue_stats = stats.get('queue', {})

            total_requests += bucket_stats.get('total_requests', 0)
            total_wait_time += bucket_stats.get('total_wait_time', 0)
            total_rejected += bucket_stats.get('rejected_requests', 0)
            total_dropped += queue_stats.get('dropped_requests', 0)

        return {
            'active_sources': len(self.clients),
            'total_requests': total_requests,
            'total_wait_time': total_wait_time,
            'total_rejected': total_rejected,
            'total_dropped': total_dropped,
            'avg_wait_time': total_wait_time / max(1, total_requests)
        }


# Convenience context manager for rate-limited requests
@asynccontextmanager
async def rate_limited(source: str, priority: RequestPriority = RequestPriority.NORMAL):
    """
    Context manager for rate-limited operations.

    Usage:
        async with rate_limited('finnhub', RequestPriority.HIGH):
            result = await make_api_call()
    """
    manager = RateLimitManager()
    client = manager.get_client(source)

    await client.bucket.acquire()
    client._record_request()

    try:
        yield client
    finally:
        pass  # Token already consumed


# Module-level convenience functions
_manager: Optional[RateLimitManager] = None


def get_rate_limit_manager() -> RateLimitManager:
    """Get the global rate limit manager."""
    global _manager
    if _manager is None:
        _manager = RateLimitManager()
    return _manager


async def can_make_request(source: str) -> bool:
    """Check if a request can be made to a source."""
    return get_rate_limit_manager().can_make_request(source)


async def execute_rate_limited(
    source: str,
    ticker: str,
    callback: Callable,
    priority: RequestPriority = RequestPriority.NORMAL
) -> Any:
    """Execute a rate-limited request."""
    return await get_rate_limit_manager().execute(source, ticker, callback, priority)
