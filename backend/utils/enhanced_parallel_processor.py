"""
Enhanced Parallel Processing Framework with Optimized Concurrency
Advanced API processing with intelligent resource allocation and performance optimization
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
import weakref

import aiohttp
from aiohttp import ClientSession, ClientTimeout, TCPConnector
import numpy as np
from prometheus_client import Counter, Histogram, Gauge

from backend.utils.distributed_rate_limiter import APIRateLimiter, RateLimitExceeded
from backend.utils.query_cache import QueryResultCache
from backend.utils.memory_manager import get_memory_manager, memory_efficient, BoundedDict

logger = logging.getLogger(__name__)

# Enhanced metrics
api_calls_counter = Counter('enhanced_api_calls_total', 'Total enhanced API calls', ['provider', 'status'])
api_latency_histogram = Histogram('enhanced_api_latency_seconds', 'Enhanced API call latency', ['provider'])
concurrent_calls_gauge = Gauge('enhanced_concurrent_calls', 'Current enhanced concurrent API calls', ['provider'])
queue_size_gauge = Gauge('enhanced_queue_size', 'Current enhanced queue size', ['provider'])
throughput_gauge = Gauge('enhanced_throughput_rps', 'Enhanced requests per second', ['provider'])
success_rate_gauge = Gauge('enhanced_success_rate', 'Enhanced API success rate', ['provider'])


class ProcessingStrategy(Enum):
    """Enhanced processing strategies for different scenarios."""
    AGGRESSIVE = "aggressive"      # Maximum parallelization
    BALANCED = "balanced"          # Balanced approach  
    CONSERVATIVE = "conservative"  # Careful with rate limits
    ADAPTIVE = "adaptive"          # Learns and adapts
    THROUGHPUT_OPTIMIZED = "throughput_optimized"  # Focus on throughput
    LATENCY_OPTIMIZED = "latency_optimized"        # Focus on low latency


class Priority(Enum):
    """Task priority levels with finer granularity."""
    CRITICAL = 1
    URGENT = 2  
    HIGH = 3
    MEDIUM = 4
    LOW = 5
    BATCH = 6
    BACKGROUND = 7


@dataclass
class EnhancedAPITask:
    """Enhanced API call task with more metadata."""
    id: str
    provider: str
    endpoint: str
    params: Dict[str, Any]
    priority: Priority = Priority.MEDIUM
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=datetime.now)
    timeout_ms: Optional[int] = None
    callback: Optional[Callable] = None
    tags: List[str] = field(default_factory=list)
    
    # Resource requirements
    estimated_memory_mb: float = 1.0
    estimated_cpu_usage: float = 0.1
    requires_auth: bool = False
    
    def __lt__(self, other):
        """For priority queue ordering."""
        return (self.priority.value, self.created_at) < (other.priority.value, other.created_at)


@dataclass
class EnhancedTaskResult:
    """Enhanced result with more performance metrics."""
    task_id: str
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    latency_ms: float = 0
    retries: int = 0
    cached: bool = False
    
    # Enhanced metrics
    response_size_bytes: int = 0
    queue_wait_time_ms: float = 0
    processing_time_ms: float = 0
    memory_used_mb: float = 0
    provider_latency_ms: float = 0


@dataclass
class ConnectionPoolConfig:
    """Connection pool configuration."""
    max_connections: int = 200
    max_connections_per_host: int = 50
    ttl_dns_cache: int = 300
    enable_cleanup_closed: bool = True
    keepalive_timeout: int = 30
    client_timeout_total: int = 60
    client_timeout_connect: int = 10


class EnhancedParallelProcessor:
    """
    Enhanced parallel API processor with advanced optimizations.
    """
    
    def __init__(
        self,
        max_concurrent_calls: int = 150,
        strategy: ProcessingStrategy = ProcessingStrategy.ADAPTIVE,
        enable_caching: bool = True,
        enable_rate_limiting: bool = True,
        enable_connection_pooling: bool = True,
        enable_request_batching: bool = True,
        enable_response_compression: bool = True,
        connection_pool_config: Optional[ConnectionPoolConfig] = None
    ):
        """
        Initialize enhanced parallel processor.
        
        Args:
            max_concurrent_calls: Maximum concurrent API calls (increased)
            strategy: Processing strategy to use
            enable_caching: Enable result caching
            enable_rate_limiting: Enable rate limiting
            enable_connection_pooling: Enable advanced connection pooling
            enable_request_batching: Enable request batching
            enable_response_compression: Enable response compression
            connection_pool_config: Connection pool configuration
        """
        self.max_concurrent_calls = max_concurrent_calls
        self.strategy = strategy
        self.enable_caching = enable_caching
        self.enable_rate_limiting = enable_rate_limiting
        self.enable_connection_pooling = enable_connection_pooling
        self.enable_request_batching = enable_request_batching
        self.enable_response_compression = enable_response_compression
        
        self.connection_pool_config = connection_pool_config or ConnectionPoolConfig()
        
        # Enhanced resource pools
        self._semaphores: Dict[str, asyncio.Semaphore] = {}
        self._sessions: Dict[str, ClientSession] = {}
        self._connection_pools: Dict[str, TCPConnector] = {}
        
        # Task management with batching
        self._task_queues: Dict[str, PriorityQueue] = {}
        self._active_tasks: Dict[str, Set[str]] = {}
        self._completed_tasks: BoundedDict = BoundedDict(max_size=10000)
        self._batch_queues: Dict[str, List[EnhancedAPITask]] = {}
        
        # Rate limiting and caching
        self._rate_limiter = APIRateLimiter() if enable_rate_limiting else None
        self._cache = QueryResultCache() if enable_caching else None
        
        # Enhanced performance tracking
        self._performance_stats: Dict[str, Dict[str, Any]] = {}
        self._provider_performance: Dict[str, List[float]] = {}
        self._optimal_concurrency: Dict[str, int] = {}
        self._success_rates: Dict[str, List[float]] = {}
        self._error_counts: Dict[str, int] = {}
        self._throughput_history: Dict[str, List[Tuple[datetime, float]]] = {}
        
        # Memory management
        self._memory_manager = None
        
        # Connection pool optimization
        self._connection_stats: Dict[str, Dict[str, Any]] = {}
        
        # Batch processing optimization
        self._batch_size_limits = {
            'alpha_vantage': 1,    # No batching due to strict limits
            'finnhub': 10,         # Small batches
            'polygon': 20,         # Medium batches
            'fmp': 15,            # Medium batches
            'newsapi': 25,        # Larger batches
            'sec': 30,            # Government data can handle larger batches
            'yahoo': 50,          # Yahoo Finance can handle large batches
            'quandl': 40          # Quandl supports batching
        }
    
    async def initialize(self) -> None:
        """Initialize enhanced components."""
        if self._rate_limiter:
            await self._rate_limiter.initialize()
        if self._cache:
            await self._cache.initialize()
        
        # Initialize memory manager
        self._memory_manager = await get_memory_manager()
        
        # Pre-warm connection pools for common providers
        await self._prewarm_connection_pools()
        
        logger.info(f"Enhanced parallel processor initialized with strategy: {self.strategy.value}")
    
    async def _prewarm_connection_pools(self):
        """Pre-warm connection pools for better performance."""
        common_providers = ['finnhub', 'polygon', 'fmp', 'newsapi', 'yahoo']
        
        for provider in common_providers:
            # Create connection pool
            await self._get_connection_pool(provider)
            # Create session
            await self._get_session(provider)
            
        logger.info(f"Pre-warmed connection pools for {len(common_providers)} providers")
    
    def _get_enhanced_provider_concurrency(self, provider: str) -> int:
        """Get optimal concurrency for a provider with enhanced limits."""
        if self.strategy == ProcessingStrategy.AGGRESSIVE:
            return min(self.max_concurrent_calls, 200)
        elif self.strategy == ProcessingStrategy.CONSERVATIVE:
            return min(10, self.max_concurrent_calls)
        elif self.strategy == ProcessingStrategy.BALANCED:
            # Enhanced provider-specific limits based on real-world testing
            limits = {
                'alpha_vantage': 5,    # Still conservative due to strict limits
                'finnhub': 60,         # Increased significantly
                'polygon': 25,         # Increased moderately
                'fmp': 35,            # Increased significantly
                'newsapi': 45,        # Increased significantly
                'sec': 20,            # Government endpoints, be respectful
                'yahoo': 80,          # Yahoo Finance can handle high loads
                'quandl': 50,         # Quandl is fairly robust
                'iex': 100,           # IEX Cloud is very robust
                'alphavantage_premium': 15,  # Premium tier
            }
            return min(limits.get(provider, 20), self.max_concurrent_calls)
        elif self.strategy == ProcessingStrategy.THROUGHPUT_OPTIMIZED:
            # Aggressive limits for maximum throughput
            limits = {
                'alpha_vantage': 8,
                'finnhub': 100,
                'polygon': 40,
                'fmp': 50,
                'newsapi': 60,
                'sec': 25,
                'yahoo': 120,
                'quandl': 70,
                'iex': 150
            }
            return min(limits.get(provider, 30), self.max_concurrent_calls)
        elif self.strategy == ProcessingStrategy.LATENCY_OPTIMIZED:
            # Lower concurrency but faster response
            limits = {
                'alpha_vantage': 3,
                'finnhub': 20,
                'polygon': 15,
                'fmp': 15,
                'newsapi': 20,
                'sec': 10,
                'yahoo': 30,
                'quandl': 25,
                'iex': 40
            }
            return min(limits.get(provider, 15), self.max_concurrent_calls)
        else:  # ADAPTIVE
            # Use learned optimal values
            base_limits = self._get_enhanced_provider_concurrency_balanced(provider)
            return self._optimal_concurrency.get(provider, base_limits)
    
    def _get_enhanced_provider_concurrency_balanced(self, provider: str) -> int:
        """Get balanced concurrency for a provider."""
        limits = {
            'alpha_vantage': 5,
            'finnhub': 60,
            'polygon': 25,
            'fmp': 35,
            'newsapi': 45,
            'sec': 20,
            'yahoo': 80,
            'quandl': 50,
            'iex': 100
        }
        return min(limits.get(provider, 20), self.max_concurrent_calls)
    
    def _get_semaphore(self, provider: str) -> asyncio.Semaphore:
        """Get or create semaphore for provider."""
        if provider not in self._semaphores:
            concurrency = self._get_enhanced_provider_concurrency(provider)
            self._semaphores[provider] = asyncio.Semaphore(concurrency)
        return self._semaphores[provider]
    
    async def _get_connection_pool(self, provider: str) -> TCPConnector:
        """Get or create optimized connection pool for provider."""
        if provider not in self._connection_pools:
            config = self.connection_pool_config
            
            # Provider-specific optimizations
            if provider == 'alpha_vantage':
                # Conservative settings for Alpha Vantage
                connector = TCPConnector(
                    limit=20,
                    limit_per_host=5,
                    ttl_dns_cache=config.ttl_dns_cache,
                    enable_cleanup_closed=config.enable_cleanup_closed,
                    keepalive_timeout=config.keepalive_timeout
                )
            elif provider in ['yahoo', 'iex', 'finnhub']:
                # Aggressive settings for robust providers
                connector = TCPConnector(
                    limit=config.max_connections,
                    limit_per_host=config.max_connections_per_host,
                    ttl_dns_cache=config.ttl_dns_cache,
                    enable_cleanup_closed=config.enable_cleanup_closed,
                    keepalive_timeout=config.keepalive_timeout
                )
            else:
                # Standard settings
                connector = TCPConnector(
                    limit=config.max_connections // 2,
                    limit_per_host=config.max_connections_per_host // 2,
                    ttl_dns_cache=config.ttl_dns_cache,
                    enable_cleanup_closed=config.enable_cleanup_closed,
                    keepalive_timeout=config.keepalive_timeout
                )
            
            self._connection_pools[provider] = connector
        
        return self._connection_pools[provider]
    
    async def _get_session(self, provider: str) -> ClientSession:
        """Get or create optimized HTTP session for provider."""
        if provider not in self._sessions:
            connector = await self._get_connection_pool(provider)
            config = self.connection_pool_config
            
            # Provider-specific timeout settings
            timeout_configs = {
                'alpha_vantage': ClientTimeout(
                    total=45,      # Alpha Vantage can be slow
                    connect=10,
                    sock_connect=10,
                    sock_read=35
                ),
                'sec': ClientTimeout(
                    total=60,      # SEC can be very slow
                    connect=15,
                    sock_connect=15,
                    sock_read=45
                ),
                'default': ClientTimeout(
                    total=config.client_timeout_total,
                    connect=config.client_timeout_connect,
                    sock_connect=config.client_timeout_connect,
                    sock_read=config.client_timeout_total - config.client_timeout_connect
                )
            }
            
            timeout = timeout_configs.get(provider, timeout_configs['default'])
            
            # Enhanced headers with compression support
            headers = {
                'User-Agent': 'InvestmentAnalysis/2.0 (Enhanced)',
                'Accept': 'application/json'
            }
            
            if self.enable_response_compression:
                headers['Accept-Encoding'] = 'gzip, deflate, br'
            
            self._sessions[provider] = ClientSession(
                connector=connector,
                timeout=timeout,
                headers=headers,
                connector_owner=False  # Don't close connector when session closes
            )
        
        return self._sessions[provider]
    
    @memory_efficient
    async def process_enhanced_batch(
        self,
        tasks: List[EnhancedAPITask],
        timeout: Optional[float] = None,
        enable_streaming: bool = True
    ) -> List[EnhancedTaskResult]:
        """
        Process a batch of API tasks with enhanced parallel processing.
        
        Args:
            tasks: List of enhanced API tasks to process
            timeout: Overall timeout for batch processing
            enable_streaming: Enable streaming processing for better performance
        
        Returns:
            List of enhanced task results
        """
        start_time = time.time()
        results = []
        
        # Sort tasks by priority
        tasks.sort(key=lambda t: (t.priority.value, t.created_at))
        
        # Group tasks by provider for optimal processing
        provider_tasks: Dict[str, List[EnhancedAPITask]] = {}
        for task in tasks:
            if task.provider not in provider_tasks:
                provider_tasks[task.provider] = []
            provider_tasks[task.provider].append(task)
        
        # Enable batching for supported providers
        if self.enable_request_batching:
            provider_tasks = await self._optimize_batching(provider_tasks)
        
        # Process each provider group with enhanced concurrency
        if enable_streaming and len(provider_tasks) > 1:
            results = await self._process_streaming_batches(provider_tasks, timeout)
        else:
            results = await self._process_sequential_batches(provider_tasks, timeout)
        
        # Update performance metrics
        elapsed = time.time() - start_time
        self._update_enhanced_performance_stats(len(tasks), len(results), elapsed)
        
        # Update throughput metrics
        for provider in provider_tasks.keys():
            self._update_throughput_metrics(provider, len(provider_tasks[provider]), elapsed)
        
        logger.info(
            f"Enhanced processing: {len(results)}/{len(tasks)} tasks in {elapsed:.2f}s "
            f"({len(tasks)/elapsed:.1f} tasks/sec)"
        )
        
        return results
    
    async def _optimize_batching(
        self,
        provider_tasks: Dict[str, List[EnhancedAPITask]]
    ) -> Dict[str, List[EnhancedAPITask]]:
        """Optimize task batching based on provider capabilities."""
        optimized_tasks = {}
        
        for provider, tasks in provider_tasks.items():
            batch_size = self._batch_size_limits.get(provider, 1)
            
            if batch_size > 1 and len(tasks) > 1:
                # Create batched tasks
                batched_tasks = []
                for i in range(0, len(tasks), batch_size):
                    batch = tasks[i:i + batch_size]
                    if len(batch) > 1:
                        # Combine tasks into a single batched request
                        batched_task = self._create_batched_task(provider, batch)
                        batched_tasks.append(batched_task)
                    else:
                        batched_tasks.extend(batch)
                optimized_tasks[provider] = batched_tasks
            else:
                optimized_tasks[provider] = tasks
        
        return optimized_tasks
    
    def _create_batched_task(
        self,
        provider: str,
        tasks: List[EnhancedAPITask]
    ) -> EnhancedAPITask:
        """Create a batched task from multiple individual tasks."""
        # Combine parameters for batch request
        batch_params = {
            'batch': True,
            'requests': []
        }
        
        for task in tasks:
            batch_params['requests'].append({
                'endpoint': task.endpoint,
                'params': task.params,
                'task_id': task.id
            })
        
        # Create batched task with highest priority from the batch
        highest_priority = min(task.priority for task in tasks)
        
        return EnhancedAPITask(
            id=f"batch_{provider}_{len(tasks)}_{int(time.time())}",
            provider=provider,
            endpoint='batch',
            params=batch_params,
            priority=highest_priority,
            tags=['batched'] + [tag for task in tasks for tag in task.tags],
            estimated_memory_mb=sum(task.estimated_memory_mb for task in tasks),
            estimated_cpu_usage=max(task.estimated_cpu_usage for task in tasks)
        )
    
    async def _process_streaming_batches(
        self,
        provider_tasks: Dict[str, List[EnhancedAPITask]],
        timeout: Optional[float] = None
    ) -> List[EnhancedTaskResult]:
        """Process provider batches with streaming for better performance."""
        results = []
        
        # Use asyncio.as_completed for streaming results
        provider_futures = {
            asyncio.create_task(
                self._process_provider_batch_enhanced(provider, tasks)
            ): provider
            for provider, tasks in provider_tasks.items()
        }
        
        # Process results as they complete
        if timeout:
            completed_futures = asyncio.as_completed(provider_futures.keys(), timeout=timeout)
        else:
            completed_futures = asyncio.as_completed(provider_futures.keys())
        
        try:
            for future in completed_futures:
                try:
                    provider_results = await future
                    results.extend(provider_results)
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout processing provider batch")
                except Exception as e:
                    logger.error(f"Error processing provider batch: {e}")
        except asyncio.TimeoutError:
            logger.warning("Overall batch processing timeout")
            # Cancel remaining tasks
            for future in provider_futures.keys():
                if not future.done():
                    future.cancel()
        
        return results
    
    async def _process_sequential_batches(
        self,
        provider_tasks: Dict[str, List[EnhancedAPITask]],
        timeout: Optional[float] = None
    ) -> List[EnhancedTaskResult]:
        """Process provider batches sequentially."""
        results = []
        
        provider_futures = [
            self._process_provider_batch_enhanced(provider, tasks)
            for provider, tasks in provider_tasks.items()
        ]
        
        if timeout:
            done, pending = await asyncio.wait(
                provider_futures,
                timeout=timeout,
                return_when=asyncio.ALL_COMPLETED
            )
            
            # Cancel pending tasks
            for task in pending:
                task.cancel()
            
            # Collect results from completed tasks
            for task in done:
                try:
                    provider_results = await task
                    results.extend(provider_results)
                except Exception as e:
                    logger.error(f"Error in provider batch: {e}")
        else:
            batch_results = await asyncio.gather(*provider_futures, return_exceptions=True)
            for result in batch_results:
                if isinstance(result, list):
                    results.extend(result)
                else:
                    logger.error(f"Provider batch error: {result}")
        
        return results
    
    async def _process_provider_batch_enhanced(
        self,
        provider: str,
        tasks: List[EnhancedAPITask]
    ) -> List[EnhancedTaskResult]:
        """Process tasks for a specific provider with enhancements."""
        results = []
        semaphore = self._get_semaphore(provider)
        
        # Process tasks with controlled concurrency
        async def process_task_with_semaphore(task):
            async with semaphore:
                return await self._process_single_task_enhanced(task)
        
        # Create task coroutines
        task_coroutines = [
            process_task_with_semaphore(task)
            for task in tasks
        ]
        
        # Process with enhanced error handling
        task_results = await asyncio.gather(
            *task_coroutines,
            return_exceptions=True
        )
        
        # Process results
        for task, result in zip(tasks, task_results):
            if isinstance(result, EnhancedTaskResult):
                results.append(result)
            else:
                # Handle exception
                error_result = EnhancedTaskResult(
                    task_id=task.id,
                    success=False,
                    error=str(result),
                    latency_ms=0,
                    retries=task.retry_count
                )
                results.append(error_result)
                
                # Track errors
                self._error_counts[provider] = self._error_counts.get(provider, 0) + 1
        
        # Update provider success rate
        success_count = sum(1 for r in results if r.success)
        success_rate = success_count / len(results) if results else 0
        
        if provider not in self._success_rates:
            self._success_rates[provider] = []
        self._success_rates[provider].append(success_rate)
        
        # Keep last 100 measurements
        if len(self._success_rates[provider]) > 100:
            self._success_rates[provider] = self._success_rates[provider][-100:]
        
        return results
    
    async def _process_single_task_enhanced(
        self,
        task: EnhancedAPITask
    ) -> EnhancedTaskResult:
        """Process a single API task with enhancements."""
        start_time = time.time()
        queue_start = task.created_at.timestamp()
        queue_wait_time = (start_time - queue_start) * 1000
        
        # Update concurrent calls metric
        concurrent_calls_gauge.labels(provider=task.provider).inc()
        
        try:
            # Check cache first with enhanced key
            if self.enable_caching and self._cache:
                cache_key = self._generate_enhanced_cache_key(task)
                cached_result = await self._cache.get(cache_key)
                if cached_result:
                    return EnhancedTaskResult(
                        task_id=task.id,
                        success=True,
                        data=cached_result.get('data'),
                        cached=True,
                        latency_ms=0,
                        queue_wait_time_ms=queue_wait_time,
                        response_size_bytes=cached_result.get('size', 0)
                    )
            
            # Check rate limit with enhanced logic
            if self.enable_rate_limiting and self._rate_limiter:
                allowed, details = await self._rate_limiter.check_api_limit(task.provider)
                
                if not allowed:
                    retry_after = details.get('retry_after', 30)
                    
                    if task.retry_count < task.max_retries:
                        await asyncio.sleep(min(retry_after, 5))
                        task.retry_count += 1
                        return await self._process_single_task_enhanced(task)
                    else:
                        raise RateLimitExceeded(
                            f"Rate limit exceeded for {task.provider}",
                            retry_after
                        )
            
            # Execute API call with enhanced monitoring
            result = await self._execute_enhanced_api_call(task)
            
            # Cache successful result with metadata
            if self.enable_caching and self._cache and result.success:
                cache_key = self._generate_enhanced_cache_key(task)
                cache_value = {
                    'data': result.data,
                    'size': result.response_size_bytes,
                    'timestamp': time.time()
                }
                
                # Dynamic TTL based on data type
                ttl = self._calculate_cache_ttl(task)
                await self._cache.set(cache_key, cache_value, ttl=ttl)
            
            # Enhanced metrics
            latency = (time.time() - start_time) * 1000
            result.queue_wait_time_ms = queue_wait_time
            result.processing_time_ms = latency - queue_wait_time
            
            # Update Prometheus metrics
            api_latency_histogram.labels(provider=task.provider).observe(latency / 1000)
            api_calls_counter.labels(
                provider=task.provider,
                status='success' if result.success else 'failed'
            ).inc()
            
            # Adaptive learning
            if self.strategy == ProcessingStrategy.ADAPTIVE:
                await self._update_enhanced_adaptive_learning(task.provider, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Enhanced task {task.id} failed: {e}")
            
            # Enhanced retry logic with exponential backoff
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                backoff_delay = min(2 ** task.retry_count, 30)  # Max 30 seconds
                await asyncio.sleep(backoff_delay)
                return await self._process_single_task_enhanced(task)
            
            return EnhancedTaskResult(
                task_id=task.id,
                success=False,
                error=str(e),
                retries=task.retry_count,
                latency_ms=(time.time() - start_time) * 1000,
                queue_wait_time_ms=queue_wait_time
            )
            
        finally:
            concurrent_calls_gauge.labels(provider=task.provider).dec()
    
    def _generate_enhanced_cache_key(self, task: EnhancedAPITask) -> str:
        """Generate enhanced cache key with better collision avoidance."""
        import hashlib
        
        # Include more factors in cache key
        key_data = f"{task.provider}:{task.endpoint}:{task.params}:{task.tags}"
        
        # Hash for consistent length
        key_hash = hashlib.md5(key_data.encode()).hexdigest()
        
        return f"enhanced_api:{task.provider}:{key_hash}"
    
    def _calculate_cache_ttl(self, task: EnhancedAPITask) -> int:
        """Calculate dynamic cache TTL based on data type and provider."""
        # Base TTL by provider
        base_ttls = {
            'alpha_vantage': 900,    # 15 minutes (limited API calls)
            'finnhub': 300,         # 5 minutes
            'polygon': 300,         # 5 minutes  
            'fmp': 600,            # 10 minutes
            'newsapi': 1800,       # 30 minutes (news doesn't change that fast)
            'sec': 3600,           # 1 hour (regulatory data is stable)
            'yahoo': 300,          # 5 minutes
            'quandl': 1800         # 30 minutes
        }
        
        base_ttl = base_ttls.get(task.provider, 600)
        
        # Adjust based on endpoint type
        if 'historical' in task.endpoint or 'history' in task.endpoint:
            return base_ttl * 4  # Historical data changes less frequently
        elif 'news' in task.endpoint:
            return base_ttl // 2  # News data changes more frequently
        elif 'quote' in task.endpoint or 'price' in task.endpoint:
            return base_ttl // 3  # Price data changes frequently
        
        return base_ttl
    
    async def _execute_enhanced_api_call(self, task: EnhancedAPITask) -> EnhancedTaskResult:
        """Execute the actual API call with enhancements."""
        session = await self._get_session(task.provider)
        request_start = time.time()
        
        try:
            # Build URL with provider-specific logic
            url = self._build_enhanced_url(task.provider, task.endpoint)
            
            # Prepare request parameters
            request_params = task.params.copy()
            
            # Add API keys
            request_params = self._add_api_credentials(task.provider, request_params)
            
            # Make request with enhanced error handling
            async with session.get(
                url, 
                params=request_params,
                timeout=aiohttp.ClientTimeout(total=task.timeout_ms/1000 if task.timeout_ms else None)
            ) as response:
                response.raise_for_status()
                
                # Get response data
                if response.content_type == 'application/json':
                    data = await response.json()
                else:
                    data = await response.text()
                
                # Calculate response size
                response_size = len(await response.read())
                
                provider_latency = (time.time() - request_start) * 1000
                
                return EnhancedTaskResult(
                    task_id=task.id,
                    success=True,
                    data=data,
                    response_size_bytes=response_size,
                    provider_latency_ms=provider_latency
                )
                
        except aiohttp.ClientError as e:
            return EnhancedTaskResult(
                task_id=task.id,
                success=False,
                error=f"HTTP error: {e}",
                provider_latency_ms=(time.time() - request_start) * 1000
            )
        except asyncio.TimeoutError:
            return EnhancedTaskResult(
                task_id=task.id,
                success=False,
                error="Request timeout",
                provider_latency_ms=(time.time() - request_start) * 1000
            )
    
    def _build_enhanced_url(self, provider: str, endpoint: str) -> str:
        """Build URL for provider and endpoint with enhancements."""
        base_urls = {
            'alpha_vantage': 'https://www.alphavantage.co/query',
            'finnhub': 'https://finnhub.io/api/v1',
            'polygon': 'https://api.polygon.io/v2',
            'fmp': 'https://financialmodelingprep.com/api/v3',
            'newsapi': 'https://newsapi.org/v2',
            'sec': 'https://data.sec.gov/api',
            'yahoo': 'https://query1.finance.yahoo.com/v8',
            'quandl': 'https://www.quandl.com/api/v3',
            'iex': 'https://cloud.iexapis.com/stable'
        }
        
        base_url = base_urls.get(provider, '')
        if not base_url:
            raise ValueError(f"Unknown provider: {provider}")
        
        # Handle batch endpoints
        if endpoint == 'batch':
            batch_endpoints = {
                'finnhub': f"{base_url}/batch",
                'polygon': f"{base_url}/batch",
                'fmp': f"{base_url}/batch"
            }
            return batch_endpoints.get(provider, f"{base_url}/{endpoint}")
        
        return f"{base_url}/{endpoint}" if endpoint else base_url
    
    def _add_api_credentials(self, provider: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Add API credentials to request parameters."""
        import os
        
        credential_mapping = {
            'alpha_vantage': {'apikey': os.getenv('ALPHA_VANTAGE_API_KEY')},
            'finnhub': {'token': os.getenv('FINNHUB_API_KEY')},
            'polygon': {'apiKey': os.getenv('POLYGON_API_KEY')},
            'fmp': {'apikey': os.getenv('FMP_API_KEY')},
            'newsapi': {'apiKey': os.getenv('NEWS_API_KEY')},
            'quandl': {'api_key': os.getenv('QUANDL_API_KEY')},
            'iex': {'token': os.getenv('IEX_API_KEY')}
        }
        
        if provider in credential_mapping:
            credentials = credential_mapping[provider]
            for key, value in credentials.items():
                if value:  # Only add if credential exists
                    params[key] = value
        
        return params
    
    async def _update_enhanced_adaptive_learning(
        self,
        provider: str,
        result: EnhancedTaskResult
    ) -> None:
        """Update enhanced adaptive learning for optimal performance."""
        if provider not in self._provider_performance:
            self._provider_performance[provider] = []
        
        # Track multiple performance metrics
        performance_score = self._calculate_performance_score(result)
        self._provider_performance[provider].append(performance_score)
        
        # Keep last 100 measurements
        if len(self._provider_performance[provider]) > 100:
            self._provider_performance[provider] = self._provider_performance[provider][-100:]
        
        # Adjust concurrency based on performance
        if len(self._provider_performance[provider]) >= 20:
            recent_performance = self._provider_performance[provider][-20:]
            avg_performance = np.mean(recent_performance)
            performance_stability = 1 - np.std(recent_performance)
            
            current_concurrency = self._get_enhanced_provider_concurrency(provider)
            
            # Enhanced concurrency adjustment logic
            if avg_performance > 0.8 and performance_stability > 0.7:
                # Good performance and stable, increase concurrency
                new_concurrency = min(
                    current_concurrency + max(2, current_concurrency // 10),
                    self.max_concurrent_calls
                )
            elif avg_performance < 0.5 or performance_stability < 0.4:
                # Poor performance or unstable, decrease concurrency
                new_concurrency = max(
                    current_concurrency - max(2, current_concurrency // 10),
                    1
                )
            else:
                new_concurrency = current_concurrency
            
            # Apply change if significant
            if abs(new_concurrency - current_concurrency) >= 2:
                self._optimal_concurrency[provider] = new_concurrency
                
                # Update semaphore
                self._semaphores[provider] = asyncio.Semaphore(new_concurrency)
                
                logger.info(
                    f"Enhanced adaptive adjustment for {provider}: "
                    f"{current_concurrency} -> {new_concurrency} "
                    f"(perf: {avg_performance:.2f}, stability: {performance_stability:.2f})"
                )
    
    def _calculate_performance_score(self, result: EnhancedTaskResult) -> float:
        """Calculate performance score for adaptive learning."""
        if not result.success:
            return 0.0
        
        # Factors: success, latency, size efficiency
        success_score = 1.0
        
        # Latency score (lower is better)
        target_latency_ms = 1000  # 1 second target
        latency_score = max(0, 1 - (result.provider_latency_ms / target_latency_ms))
        
        # Size efficiency (more data per ms is better)
        if result.provider_latency_ms > 0 and result.response_size_bytes > 0:
            bytes_per_ms = result.response_size_bytes / result.provider_latency_ms
            size_score = min(1.0, bytes_per_ms / 100)  # Normalize to 100 bytes/ms
        else:
            size_score = 0.5
        
        # Queue efficiency (lower queue wait is better)
        queue_score = max(0, 1 - (result.queue_wait_time_ms / 1000))
        
        # Weighted combination
        return (
            success_score * 0.4 +
            latency_score * 0.3 +
            size_score * 0.2 +
            queue_score * 0.1
        )
    
    def _update_enhanced_performance_stats(
        self,
        total_tasks: int,
        completed_tasks: int,
        elapsed_time: float
    ) -> None:
        """Update enhanced performance statistics."""
        if 'batch_count' not in self._performance_stats:
            self._performance_stats = {
                'batch_count': 0,
                'total_tasks': 0,
                'completed_tasks': 0,
                'total_time': 0,
                'avg_throughput': 0,
                'success_rate': 0
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
        
        if self._performance_stats['total_tasks'] > 0:
            self._performance_stats['success_rate'] = (
                self._performance_stats['completed_tasks'] /
                self._performance_stats['total_tasks']
            )
    
    def _update_throughput_metrics(self, provider: str, task_count: int, elapsed_time: float):
        """Update provider-specific throughput metrics."""
        if provider not in self._throughput_history:
            self._throughput_history[provider] = []
        
        throughput_rps = task_count / elapsed_time if elapsed_time > 0 else 0
        timestamp = datetime.utcnow()
        
        self._throughput_history[provider].append((timestamp, throughput_rps))
        
        # Keep last 100 measurements
        if len(self._throughput_history[provider]) > 100:
            self._throughput_history[provider] = self._throughput_history[provider][-100:]
        
        # Update Prometheus gauge
        throughput_gauge.labels(provider=provider).set(throughput_rps)
        
        # Update success rate gauge
        if provider in self._success_rates and self._success_rates[provider]:
            recent_success_rate = np.mean(self._success_rates[provider][-10:])
            success_rate_gauge.labels(provider=provider).set(recent_success_rate)
    
    def get_enhanced_performance_stats(self) -> Dict[str, Any]:
        """Get enhanced performance statistics."""
        stats = self._performance_stats.copy()
        
        # Add provider-specific stats
        stats['providers'] = {}
        for provider in self._provider_performance:
            if self._provider_performance[provider]:
                recent_performance = self._provider_performance[provider][-20:]
                
                stats['providers'][provider] = {
                    'avg_performance_score': np.mean(recent_performance),
                    'performance_stability': 1 - np.std(recent_performance),
                    'optimal_concurrency': self._optimal_concurrency.get(provider, 'N/A'),
                    'current_concurrency': self._get_enhanced_provider_concurrency(provider),
                    'error_count': self._error_counts.get(provider, 0),
                    'success_rate': np.mean(self._success_rates.get(provider, [1.0])[-10:])
                }
                
                if provider in self._throughput_history:
                    recent_throughput = [t[1] for t in self._throughput_history[provider][-10:]]
                    stats['providers'][provider]['avg_throughput_rps'] = np.mean(recent_throughput)
        
        return stats
    
    async def shutdown(self) -> None:
        """Enhanced shutdown with comprehensive cleanup."""
        close_tasks = []
        
        try:
            # Close all HTTP sessions
            for provider, session in self._sessions.items():
                if session and not session.closed:
                    close_tasks.append(session.close())
                    logger.debug(f"Closing enhanced session for {provider}")
            
            # Close connection pools
            for provider, connector in self._connection_pools.items():
                if connector and not connector.closed:
                    close_tasks.append(connector.close())
                    logger.debug(f"Closing connection pool for {provider}")
            
            # Close rate limiter
            if self._rate_limiter:
                close_tasks.append(self._rate_limiter.close())
                logger.debug("Closing enhanced rate limiter")
            
            # Close cache
            if self._cache:
                close_tasks.append(self._cache.close())
                logger.debug("Closing enhanced cache")
            
            # Wait for all close operations with timeout
            if close_tasks:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*close_tasks, return_exceptions=True),
                        timeout=10.0
                    )
                except asyncio.TimeoutError:
                    logger.error("Enhanced shutdown timeout - forcing cleanup")
            
            # Clear all data structures
            self._sessions.clear()
            self._connection_pools.clear()
            self._semaphores.clear()
            self._task_queues.clear()
            self._active_tasks.clear()
            self._completed_tasks.clear()
            self._batch_queues.clear()
            self._performance_stats.clear()
            self._provider_performance.clear()
            self._optimal_concurrency.clear()
            self._success_rates.clear()
            self._error_counts.clear()
            self._throughput_history.clear()
            self._connection_stats.clear()
            
            logger.info("Enhanced parallel processor shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during enhanced parallel processor shutdown: {e}")


# Global enhanced processor instance
enhanced_parallel_processor = EnhancedParallelProcessor()