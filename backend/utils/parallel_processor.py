"""
Parallel Processing Framework for API Calls
Optimizes API data fetching with intelligent parallelization and resource management.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Callable, Tuple, Union, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
from queue import PriorityQueue
import threading

import aiohttp
from aiohttp import ClientSession, ClientTimeout, TCPConnector
import numpy as np
from prometheus_client import Counter, Histogram, Gauge

from backend.utils.distributed_rate_limiter import APIRateLimiter, RateLimitExceeded
from backend.utils.query_cache import QueryResultCache

logger = logging.getLogger(__name__)

# Metrics
api_calls_counter = Counter('parallel_api_calls_total', 'Total parallel API calls', ['provider', 'status'])
api_latency_histogram = Histogram('parallel_api_latency_seconds', 'API call latency', ['provider'])
concurrent_calls_gauge = Gauge('parallel_concurrent_calls', 'Current concurrent API calls', ['provider'])
queue_size_gauge = Gauge('parallel_queue_size', 'Current queue size', ['provider'])


class ProcessingStrategy(Enum):
    """Processing strategies for different scenarios."""
    AGGRESSIVE = "aggressive"  # Maximum parallelization
    BALANCED = "balanced"      # Balanced approach
    CONSERVATIVE = "conservative"  # Careful with rate limits
    ADAPTIVE = "adaptive"      # Learns and adapts


class Priority(Enum):
    """Task priority levels."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BATCH = 5


@dataclass
class APITask:
    """Represents an API call task."""
    id: str
    provider: str
    endpoint: str
    params: Dict[str, Any]
    priority: Priority = Priority.MEDIUM
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=datetime.now)
    callback: Optional[Callable] = None
    
    def __lt__(self, other):
        """For priority queue ordering."""
        return self.priority.value < other.priority.value


@dataclass
class TaskResult:
    """Result of an API task execution."""
    task_id: str
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    latency_ms: float = 0
    retries: int = 0
    cached: bool = False


class ParallelAPIProcessor:
    """
    Manages parallel API calls with intelligent resource allocation.
    """
    
    def __init__(
        self,
        max_concurrent_calls: int = 100,  # Increased from 50
        strategy: ProcessingStrategy = ProcessingStrategy.ADAPTIVE,  # Changed to adaptive
        enable_caching: bool = True,
        enable_rate_limiting: bool = True,
        enable_connection_pooling: bool = True,
        enable_request_batching: bool = True
    ):
        """
        Initialize parallel processor.
        
        Args:
            max_concurrent_calls: Maximum concurrent API calls
            strategy: Processing strategy to use
            enable_caching: Enable result caching
            enable_rate_limiting: Enable rate limiting
        """
        self.max_concurrent_calls = max_concurrent_calls
        self.strategy = strategy
        self.enable_caching = enable_caching
        self.enable_rate_limiting = enable_rate_limiting
        
        # Resource pools
        self._semaphores: Dict[str, asyncio.Semaphore] = {}
        self._sessions: Dict[str, ClientSession] = {}
        
        # Task management
        self._task_queues: Dict[str, PriorityQueue] = {}
        self._active_tasks: Dict[str, Set[str]] = {}
        self._completed_tasks: Dict[str, TaskResult] = {}
        
        # Rate limiting and caching
        self._rate_limiter = APIRateLimiter() if enable_rate_limiting else None
        self._cache = QueryResultCache() if enable_caching else None
        
        # Performance tracking
        self._performance_stats: Dict[str, Dict[str, Any]] = {}
        
        # Adaptive learning
        self._provider_performance: Dict[str, List[float]] = {}
        self._optimal_concurrency: Dict[str, int] = {}
    
    async def initialize(self) -> None:
        """Initialize components."""
        if self._rate_limiter:
            await self._rate_limiter.initialize()
        if self._cache:
            await self._cache.initialize()
        
        logger.info(f"Parallel processor initialized with strategy: {self.strategy.value}")
    
    def _get_provider_concurrency(self, provider: str) -> int:
        """Get optimal concurrency for a provider."""
        if self.strategy == ProcessingStrategy.AGGRESSIVE:
            return min(self.max_concurrent_calls, 100)
        elif self.strategy == ProcessingStrategy.CONSERVATIVE:
            return min(5, self.max_concurrent_calls)
        elif self.strategy == ProcessingStrategy.BALANCED:
            # Provider-specific limits
            limits = {
                'alpha_vantage': 3,  # Very strict rate limit
                'finnhub': 30,       # More lenient
                'polygon': 5,        # Moderate
                'fmp': 10,
                'newsapi': 20,
                'sec': 8
            }
            return min(limits.get(provider, 10), self.max_concurrent_calls)
        else:  # ADAPTIVE
            return self._optimal_concurrency.get(
                provider,
                self._get_provider_concurrency_balanced(provider)
            )
    
    def _get_provider_concurrency_balanced(self, provider: str) -> int:
        """Get balanced concurrency for a provider."""
        limits = {
            'alpha_vantage': 3,
            'finnhub': 30,
            'polygon': 5,
            'fmp': 10,
            'newsapi': 20,
            'sec': 8
        }
        return min(limits.get(provider, 10), self.max_concurrent_calls)
    
    def _get_semaphore(self, provider: str) -> asyncio.Semaphore:
        """Get or create semaphore for provider."""
        if provider not in self._semaphores:
            concurrency = self._get_provider_concurrency(provider)
            self._semaphores[provider] = asyncio.Semaphore(concurrency)
        return self._semaphores[provider]
    
    async def _get_session(self, provider: str) -> ClientSession:
        """Get or create HTTP session for provider."""
        if provider not in self._sessions:
            connector = TCPConnector(
                limit=100,
                limit_per_host=30,
                ttl_dns_cache=300,
                enable_cleanup_closed=True
            )
            
            timeout = ClientTimeout(
                total=30,
                connect=5,
                sock_connect=5,
                sock_read=25
            )
            
            self._sessions[provider] = ClientSession(
                connector=connector,
                timeout=timeout,
                headers={'User-Agent': 'InvestmentAnalysis/1.0'}
            )
        
        return self._sessions[provider]
    
    async def process_batch(
        self,
        tasks: List[APITask],
        timeout: Optional[float] = None
    ) -> List[TaskResult]:
        """
        Process a batch of API tasks in parallel.
        
        Args:
            tasks: List of API tasks to process
            timeout: Overall timeout for batch processing
        
        Returns:
            List of task results
        """
        start_time = time.time()
        results = []
        
        # Group tasks by provider
        provider_tasks: Dict[str, List[APITask]] = {}
        for task in tasks:
            if task.provider not in provider_tasks:
                provider_tasks[task.provider] = []
            provider_tasks[task.provider].append(task)
        
        # Process each provider group
        provider_futures = []
        for provider, provider_task_list in provider_tasks.items():
            future = asyncio.create_task(
                self._process_provider_batch(provider, provider_task_list)
            )
            provider_futures.append(future)
        
        # Wait for all with timeout
        if timeout:
            done, pending = await asyncio.wait(
                provider_futures,
                timeout=timeout,
                return_when=asyncio.ALL_COMPLETED
            )
            
            # Cancel pending tasks
            for task in pending:
                task.cancel()
        else:
            done = await asyncio.gather(*provider_futures, return_exceptions=True)
        
        # Collect results
        for future_result in done:
            if isinstance(future_result, list):
                results.extend(future_result)
            elif isinstance(future_result, Exception):
                logger.error(f"Provider batch processing error: {future_result}")
        
        # Update performance stats
        elapsed = time.time() - start_time
        self._update_performance_stats(len(tasks), len(results), elapsed)
        
        logger.info(
            f"Processed {len(results)}/{len(tasks)} tasks in {elapsed:.2f}s "
            f"({len(tasks)/elapsed:.1f} tasks/sec)"
        )
        
        return results
    
    async def _process_provider_batch(
        self,
        provider: str,
        tasks: List[APITask]
    ) -> List[TaskResult]:
        """Process tasks for a specific provider."""
        results = []
        semaphore = self._get_semaphore(provider)
        
        # Create task coroutines
        task_coroutines = [
            self._process_single_task(task, semaphore)
            for task in tasks
        ]
        
        # Process with controlled concurrency
        task_results = await asyncio.gather(
            *task_coroutines,
            return_exceptions=True
        )
        
        for task, result in zip(tasks, task_results):
            if isinstance(result, TaskResult):
                results.append(result)
            else:
                # Handle exception
                results.append(TaskResult(
                    task_id=task.id,
                    success=False,
                    error=str(result)
                ))
        
        return results
    
    async def _process_single_task(
        self,
        task: APITask,
        semaphore: asyncio.Semaphore
    ) -> TaskResult:
        """Process a single API task."""
        start_time = time.time()
        
        # Check cache first
        if self.enable_caching and self._cache:
            cache_key = f"{task.provider}:{task.endpoint}:{task.params}"
            cached_result = await self._cache.get(cache_key)
            if cached_result:
                return TaskResult(
                    task_id=task.id,
                    success=True,
                    data=cached_result,
                    cached=True,
                    latency_ms=0
                )
        
        async with semaphore:
            concurrent_calls_gauge.labels(provider=task.provider).inc()
            
            try:
                # Check rate limit
                if self.enable_rate_limiting and self._rate_limiter:
                    allowed, details = await self._rate_limiter.check_api_limit(
                        task.provider
                    )
                    
                    if not allowed:
                        retry_after = details.get('retry_after', 60)
                        
                        # Retry if within retry count
                        if task.retry_count < task.max_retries:
                            await asyncio.sleep(min(retry_after, 5))
                            task.retry_count += 1
                            return await self._process_single_task(task, semaphore)
                        else:
                            raise RateLimitExceeded(
                                f"Rate limit exceeded for {task.provider}",
                                retry_after
                            )
                
                # Execute API call
                result = await self._execute_api_call(task)
                
                # Cache successful result
                if self.enable_caching and self._cache and result.success:
                    cache_key = f"{task.provider}:{task.endpoint}:{task.params}"
                    await self._cache.set(
                        cache_key,
                        result.data,
                        ttl=300  # 5 minutes
                    )
                
                # Update metrics
                latency = (time.time() - start_time) * 1000
                api_latency_histogram.labels(provider=task.provider).observe(latency / 1000)
                api_calls_counter.labels(
                    provider=task.provider,
                    status='success' if result.success else 'failed'
                ).inc()
                
                # Learn from performance
                if self.strategy == ProcessingStrategy.ADAPTIVE:
                    await self._update_adaptive_learning(task.provider, latency)
                
                return result
                
            except Exception as e:
                logger.error(f"Task {task.id} failed: {e}")
                
                # Retry logic
                if task.retry_count < task.max_retries:
                    task.retry_count += 1
                    await asyncio.sleep(2 ** task.retry_count)  # Exponential backoff
                    return await self._process_single_task(task, semaphore)
                
                return TaskResult(
                    task_id=task.id,
                    success=False,
                    error=str(e),
                    retries=task.retry_count
                )
                
            finally:
                concurrent_calls_gauge.labels(provider=task.provider).dec()
    
    async def _execute_api_call(self, task: APITask) -> TaskResult:
        """Execute the actual API call."""
        session = await self._get_session(task.provider)
        
        try:
            # Provider-specific URL construction
            url = self._build_url(task.provider, task.endpoint)
            
            async with session.get(url, params=task.params) as response:
                response.raise_for_status()
                data = await response.json()
                
                return TaskResult(
                    task_id=task.id,
                    success=True,
                    data=data,
                    latency_ms=(time.time() - task.created_at.timestamp()) * 1000
                )
                
        except aiohttp.ClientError as e:
            return TaskResult(
                task_id=task.id,
                success=False,
                error=f"HTTP error: {e}"
            )
    
    def _build_url(self, provider: str, endpoint: str) -> str:
        """Build URL for provider and endpoint."""
        base_urls = {
            'alpha_vantage': 'https://www.alphavantage.co/query',
            'finnhub': 'https://finnhub.io/api/v1',
            'polygon': 'https://api.polygon.io/v2',
            'fmp': 'https://financialmodelingprep.com/api/v3',
            'newsapi': 'https://newsapi.org/v2',
            'sec': 'https://data.sec.gov/api'
        }
        
        base_url = base_urls.get(provider, '')
        if not base_url:
            raise ValueError(f"Unknown provider: {provider}")
        
        return f"{base_url}/{endpoint}" if endpoint else base_url
    
    async def _update_adaptive_learning(
        self,
        provider: str,
        latency_ms: float
    ) -> None:
        """Update adaptive learning for optimal concurrency."""
        if provider not in self._provider_performance:
            self._provider_performance[provider] = []
        
        self._provider_performance[provider].append(latency_ms)
        
        # Keep last 100 measurements
        if len(self._provider_performance[provider]) > 100:
            self._provider_performance[provider] = self._provider_performance[provider][-100:]
        
        # Adjust concurrency based on performance
        if len(self._provider_performance[provider]) >= 20:
            avg_latency = np.mean(self._provider_performance[provider])
            std_latency = np.std(self._provider_performance[provider])
            
            current_concurrency = self._get_provider_concurrency(provider)
            
            # If latency is stable and low, increase concurrency
            if std_latency < avg_latency * 0.2 and avg_latency < 1000:
                new_concurrency = min(current_concurrency + 2, self.max_concurrent_calls)
            # If latency is high or unstable, decrease concurrency
            elif avg_latency > 5000 or std_latency > avg_latency * 0.5:
                new_concurrency = max(current_concurrency - 2, 1)
            else:
                new_concurrency = current_concurrency
            
            self._optimal_concurrency[provider] = new_concurrency
            
            # Update semaphore
            if new_concurrency != current_concurrency:
                self._semaphores[provider] = asyncio.Semaphore(new_concurrency)
                logger.info(
                    f"Adjusted {provider} concurrency: {current_concurrency} -> {new_concurrency}"
                )
    
    def _update_performance_stats(
        self,
        total_tasks: int,
        completed_tasks: int,
        elapsed_time: float
    ) -> None:
        """Update performance statistics."""
        if 'batch_count' not in self._performance_stats:
            self._performance_stats = {
                'batch_count': 0,
                'total_tasks': 0,
                'completed_tasks': 0,
                'total_time': 0,
                'avg_throughput': 0
            }
        
        self._performance_stats['batch_count'] += 1
        self._performance_stats['total_tasks'] += total_tasks
        self._performance_stats['completed_tasks'] += completed_tasks
        self._performance_stats['total_time'] += elapsed_time
        
        if self._performance_stats['total_time'] > 0:
            self._performance_stats['avg_throughput'] = (
                self._performance_stats['completed_tasks'] /
                self._performance_stats['total_time']
            )
    
    async def process_stream(
        self,
        task_generator: asyncio.Queue,
        max_batch_size: int = 100,
        batch_timeout: float = 5.0
    ) -> None:
        """
        Process a stream of tasks continuously.
        
        Args:
            task_generator: Queue generating tasks
            max_batch_size: Maximum batch size
            batch_timeout: Timeout for batch collection
        """
        batch = []
        last_process_time = time.time()
        
        while True:
            try:
                # Collect batch
                while len(batch) < max_batch_size:
                    timeout = batch_timeout - (time.time() - last_process_time)
                    
                    if timeout <= 0:
                        break
                    
                    try:
                        task = await asyncio.wait_for(
                            task_generator.get(),
                            timeout=timeout
                        )
                        batch.append(task)
                    except asyncio.TimeoutError:
                        break
                
                # Process batch if we have tasks
                if batch:
                    results = await self.process_batch(batch)
                    
                    # Execute callbacks
                    for task, result in zip(batch, results):
                        if task.callback:
                            await task.callback(result)
                    
                    batch = []
                    last_process_time = time.time()
                
                await asyncio.sleep(0.1)  # Small delay to prevent tight loop
                
            except Exception as e:
                logger.error(f"Stream processing error: {e}")
                await asyncio.sleep(1)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self._performance_stats.copy()
        
        # Add provider-specific stats
        stats['providers'] = {}
        for provider in self._provider_performance:
            if self._provider_performance[provider]:
                stats['providers'][provider] = {
                    'avg_latency_ms': np.mean(self._provider_performance[provider]),
                    'p95_latency_ms': np.percentile(self._provider_performance[provider], 95),
                    'optimal_concurrency': self._optimal_concurrency.get(provider, 'N/A')
                }
        
        return stats
    
    async def shutdown(self) -> None:
        """Properly shutdown processor with timeout and comprehensive cleanup."""
        close_tasks = []
        
        try:
            # Close all HTTP sessions
            for provider, session in self._sessions.items():
                if session and not session.closed:
                    close_tasks.append(session.close())
                    logger.debug(f"Closing session for {provider}")
            
            # Close rate limiter
            if self._rate_limiter:
                close_tasks.append(self._rate_limiter.close())
                logger.debug("Closing rate limiter")
            
            # Close cache
            if self._cache:
                close_tasks.append(self._cache.close())
                logger.debug("Closing cache")
            
            # Wait for all close operations with timeout
            if close_tasks:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*close_tasks, return_exceptions=True),
                        timeout=5.0
                    )
                except asyncio.TimeoutError:
                    logger.error("Shutdown timeout - forcing cleanup")
            
            # Clear data structures
            self._sessions.clear()
            self._semaphores.clear()
            self._task_queues.clear()
            self._active_tasks.clear()
            self._completed_tasks.clear()
            self._performance_stats.clear()
            self._provider_performance.clear()
            self._optimal_concurrency.clear()
            
            logger.info("Parallel processor shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during parallel processor shutdown: {e}")


# Specialized processor for stock data
class StockDataParallelProcessor(ParallelAPIProcessor):
    """
    Specialized parallel processor for stock market data.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._market_hours_aware = True
    
    def create_stock_tasks(
        self,
        symbols: List[str],
        data_types: List[str],
        provider: str,
        priority: Priority = Priority.MEDIUM
    ) -> List[APITask]:
        """
        Create API tasks for stock data fetching.
        
        Args:
            symbols: List of stock symbols
            data_types: Types of data to fetch (price, fundamentals, etc.)
            provider: API provider to use
            priority: Task priority
        
        Returns:
            List of API tasks
        """
        tasks = []
        
        for symbol in symbols:
            for data_type in data_types:
                endpoint = self._get_endpoint(provider, data_type)
                params = self._get_params(provider, symbol, data_type)
                
                task = APITask(
                    id=f"{symbol}_{data_type}_{provider}",
                    provider=provider,
                    endpoint=endpoint,
                    params=params,
                    priority=priority
                )
                
                tasks.append(task)
        
        return tasks
    
    def _get_endpoint(self, provider: str, data_type: str) -> str:
        """Get API endpoint for data type."""
        endpoints = {
            'alpha_vantage': {
                'price': '',
                'fundamentals': '',
                'technical': ''
            },
            'finnhub': {
                'price': 'quote',
                'news': 'company-news',
                'sentiment': 'news-sentiment'
            },
            'polygon': {
                'price': 'aggs/ticker',
                'trades': 'trades',
                'quotes': 'quotes'
            }
        }
        
        return endpoints.get(provider, {}).get(data_type, '')
    
    def _get_params(
        self,
        provider: str,
        symbol: str,
        data_type: str
    ) -> Dict[str, Any]:
        """Get API parameters for request."""
        if provider == 'alpha_vantage':
            return {
                'symbol': symbol,
                'function': self._get_av_function(data_type),
                'apikey': '${ALPHA_VANTAGE_API_KEY}'  # Will be replaced
            }
        elif provider == 'finnhub':
            return {
                'symbol': symbol,
                'token': '${FINNHUB_API_KEY}'  # Will be replaced
            }
        elif provider == 'polygon':
            return {
                'ticker': symbol,
                'apiKey': '${POLYGON_API_KEY}'  # Will be replaced
            }
        
        return {}
    
    def _get_av_function(self, data_type: str) -> str:
        """Get Alpha Vantage function name."""
        functions = {
            'price': 'GLOBAL_QUOTE',
            'daily': 'TIME_SERIES_DAILY',
            'fundamentals': 'OVERVIEW',
            'technical': 'RSI'
        }
        return functions.get(data_type, 'GLOBAL_QUOTE')


# Global processor instance
parallel_processor = ParallelAPIProcessor()
stock_processor = StockDataParallelProcessor()