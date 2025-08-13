"""
Enhanced Unified Integration Layer with structured logging, optimized caching,
specific exception handling, and comprehensive monitoring.
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
import time
from enum import Enum
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session
from sqlalchemy import select, func, and_, or_

# Import new components
from backend.utils.structured_logging import (
    StructuredLogger,
    log_operation,
    generate_correlation_id,
    set_correlation_id
)
from backend.utils.cache_optimization import (
    CacheKeyGenerator,
    CacheNamespace,
    ConsistentHashRing,
    generate_stock_price_key,
    generate_fundamental_key,
    generate_analysis_key
)
from backend.utils.enhanced_exceptions import (
    RateLimitException,
    APITimeoutException,
    APIDataException,
    CacheConnectionException,
    DataQualityException,
    BudgetExceededException,
    ExceptionHandler,
    RecoveryStrategy
)
from backend.monitoring.data_quality_metrics import (
    dq_metrics,
    check_data_quality_with_metrics
)

# Existing imports
from backend.utils.distributed_rate_limiter import APIRateLimiter, RateLimitExceeded
from backend.utils.query_cache import QueryResultCache, CacheStrategy
from backend.utils.parallel_processor import (
    ParallelAPIProcessor, 
    APITask, 
    Priority,
    ProcessingStrategy,
    TaskResult
)
from backend.utils.persistent_cost_monitor import PersistentCostMonitor
from backend.utils.cost_monitor import CostMonitor, SmartDataFetcher
from backend.config.settings import settings
from backend.utils.async_database import get_async_session, async_db_manager
from backend.utils.monitoring import metrics as base_metrics

# Initialize structured logger
logger = StructuredLogger(__name__)

# Initialize exception handler
exception_handler = ExceptionHandler(logger)

# Initialize cache key generator
cache_key_gen = CacheKeyGenerator(version="v2", use_hashing=True)


class StockTier(Enum):
    """Stock priority tiers for efficient processing."""
    CRITICAL = 1    # S&P 500, high volume - Updated hourly
    HIGH = 2        # Mid-cap active - Updated every 4 hours  
    MEDIUM = 3      # Small-cap - Updated every 8 hours
    LOW = 4         # Inactive - Daily updates
    MINIMAL = 5     # Delisted/low activity - Weekly updates


class EnhancedUnifiedDataIngestion:
    """
    Enhanced unified ingestion system with structured logging, optimized caching,
    and comprehensive exception handling.
    """
    
    def __init__(self, db_session: Optional[AsyncSession] = None):
        """
        Initialize enhanced unified ingestion with all components.
        
        Args:
            db_session: Async database session for persistence
        """
        self.db_session = db_session
        self.correlation_id = None
        
        # Initialize components
        self.rate_limiter = APIRateLimiter()
        self.cache = QueryResultCache(
            strategy=CacheStrategy.TTL,
            max_size=10000,
            default_ttl=300
        )
        self.processor = ParallelAPIProcessor(
            max_concurrent_calls=50,
            strategy=ProcessingStrategy.ADAPTIVE,
            enable_caching=True,
            enable_rate_limiting=True
        )
        self.cost_monitor = PersistentCostMonitor(self.db_session)
        self.smart_fetcher = SmartDataFetcher(self.cost_monitor)
        
        # Initialize consistent hashing for distributed caching
        cache_nodes = settings.CACHE_NODES if hasattr(settings, 'CACHE_NODES') else ['redis:6379']
        self.hash_ring = ConsistentHashRing(cache_nodes)
        
        # Stock tiering configuration
        self.stock_tiers: Dict[StockTier, Set[str]] = {
            StockTier.CRITICAL: set(),
            StockTier.HIGH: set(),
            StockTier.MEDIUM: set(),
            StockTier.LOW: set(),
            StockTier.MINIMAL: set()
        }
        
        # Update frequencies (in seconds)
        self.tier_update_frequencies = {
            StockTier.CRITICAL: 3600,      # 1 hour
            StockTier.HIGH: 14400,          # 4 hours
            StockTier.MEDIUM: 28800,        # 8 hours
            StockTier.LOW: 86400,           # 24 hours
            StockTier.MINIMAL: 604800       # 7 days
        }
        
        # Provider allocation by tier
        self.tier_providers = {
            StockTier.CRITICAL: ['finnhub'],
            StockTier.HIGH: ['alpha_vantage'],
            StockTier.MEDIUM: ['polygon'],
            StockTier.LOW: ['cache_only'],
            StockTier.MINIMAL: ['cache_only']
        }
        
        self._initialized = False
        self._last_tier_update = {}
    
    async def initialize(self) -> None:
        """Initialize all components and load stock tiers."""
        if self._initialized:
            return
        
        # Generate correlation ID for initialization
        self.correlation_id = generate_correlation_id()
        set_correlation_id(self.correlation_id)
        
        with log_operation(logger, "initialization", correlation_id=self.correlation_id):
            try:
                # Initialize components
                await self.rate_limiter.initialize()
                await self.cache.initialize()
                await self.processor.initialize()
                await self.cost_monitor.initialize()
                
                # Load stock tiers
                await self._load_stock_tiers()
                
                self._initialized = True
                logger.info(
                    "unified_ingestion_initialized",
                    components={
                        'rate_limiter': 'ready',
                        'cache': 'ready',
                        'processor': 'ready',
                        'cost_monitor': 'ready'
                    }
                )
                
            except Exception as e:
                logger.error(
                    "initialization_failed",
                    exception=e,
                    correlation_id=self.correlation_id
                )
                raise
    
    async def _load_stock_tiers(self) -> None:
        """Load stocks into appropriate tiers based on volume and market cap."""
        with log_operation(logger, "load_stock_tiers"):
            try:
                from backend.models.unified_models import Stock
                
                async with get_async_session() as session:
                    # Get all stocks ordered by market cap and volume
                    result = await session.execute(
                        select(Stock)
                        .order_by(
                            Stock.market_cap.desc().nullslast(),
                            Stock.avg_volume.desc().nullslast()
                        )
                    )
                    all_stocks = result.scalars().all()
                    
                    # Process and assign tiers
                    total_stocks = len(all_stocks)
                    if total_stocks > 0:
                        self._assign_stock_tiers(all_stocks)
                    
                    logger.info(
                        "stock_tiers_loaded",
                        tier_counts={
                            tier.name: len(symbols) 
                            for tier, symbols in self.stock_tiers.items()
                        }
                    )
                    
            except Exception as e:
                logger.error("failed_to_load_stock_tiers", exception=e)
                # Use default minimal set on failure
                self.stock_tiers[StockTier.CRITICAL] = {'AAPL', 'MSFT', 'GOOGL'}
    
    def _assign_stock_tiers(self, stocks: List[Any]) -> None:
        """Assign stocks to tiers based on percentiles."""
        total_stocks = len(stocks)
        
        # Calculate tier boundaries
        tier_sizes = {
            StockTier.CRITICAL: min(100, total_stocks),
            StockTier.HIGH: min(400, max(0, total_stocks - 100)),
            StockTier.MEDIUM: min(300, max(0, total_stocks - 500)),
            StockTier.LOW: min(200, max(0, total_stocks - 800)),
            StockTier.MINIMAL: max(0, total_stocks - 1000)
        }
        
        # Assign stocks to tiers
        idx = 0
        for tier, size in tier_sizes.items():
            if size > 0:
                tier_stocks = stocks[idx:idx + size]
                self.stock_tiers[tier] = set(s.symbol for s in tier_stocks)
                idx += size
    
    async def fetch_stock_data(
        self,
        symbols: List[str],
        data_types: List[str] = ['price', 'fundamentals'],
        force_refresh: bool = False,
        correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Fetch stock data with intelligent orchestration and enhanced error handling.
        
        Args:
            symbols: List of stock symbols
            data_types: Types of data to fetch
            force_refresh: Force refresh bypassing cache
            correlation_id: Correlation ID for request tracking
            
        Returns:
            Dictionary of results by symbol
        """
        # Set correlation ID
        if correlation_id:
            set_correlation_id(correlation_id)
        else:
            correlation_id = generate_correlation_id()
            set_correlation_id(correlation_id)
        
        logger = StructuredLogger(__name__).with_correlation_id(correlation_id)
        
        async with self._fetch_context(logger, symbols, data_types):
            if not self._initialized:
                await self.initialize()
            
            results = {}
            
            try:
                # Check emergency mode first
                if await self.cost_monitor.is_in_emergency_mode():
                    raise BudgetExceededException(
                        current_cost=await self.cost_monitor.get_current_cost(),
                        budget=50.0
                    )
                
                # Group symbols by tier
                tier_groups = self._group_symbols_by_tier(symbols)
                
                # Process each tier with appropriate strategy
                for tier, tier_symbols in tier_groups.items():
                    if not tier_symbols:
                        continue
                    
                    tier_results = await self._process_tier_with_recovery(
                        tier, tier_symbols, data_types, force_refresh, logger
                    )
                    
                    results.update(tier_results)
                
                # Perform data quality checks on results
                await self._validate_results_quality(results, logger)
                
            except BudgetExceededException as e:
                # Handle budget exceeded - use cache only
                logger.warning("budget_exceeded_using_cache", details=e.to_dict())
                results = await self._fetch_cached_only_with_metrics(symbols, data_types)
                
            except Exception as e:
                logger.error("fetch_stock_data_failed", exception=e)
                # Attempt recovery
                recovery_context = {
                    'cache_key': self._build_batch_cache_key(symbols, data_types),
                    'fallback_func': lambda: self._fetch_cached_only_with_metrics(symbols, data_types)
                }
                results = await exception_handler.handle_exception(e, recovery_context)
                
                if not results:
                    raise
            
            return results
    
    @asynccontextmanager
    async def _fetch_context(self, logger: StructuredLogger, symbols: List[str], data_types: List[str]):
        """Context manager for fetch operations with timing and logging."""
        start_time = time.time()
        
        logger.info(
            "fetch_started",
            symbols_count=len(symbols),
            data_types=data_types
        )
        
        try:
            yield
        finally:
            duration_ms = (time.time() - start_time) * 1000
            logger.info(
                "fetch_completed",
                duration_ms=duration_ms,
                symbols_count=len(symbols)
            )
            
            # Record metrics
            base_metrics.track_analysis(
                analysis_type="data_fetch",
                duration=duration_ms / 1000,
                success=True
            )
    
    async def _process_tier_with_recovery(
        self,
        tier: StockTier,
        symbols: List[str],
        data_types: List[str],
        force_refresh: bool,
        logger: StructuredLogger
    ) -> Dict[str, Any]:
        """Process tier with error recovery strategies."""
        results = {}
        
        # Check if tier needs update
        if not self._should_update_tier(tier) and not force_refresh:
            # Use cached data
            results = await self._fetch_cached_only_with_metrics(symbols, data_types)
        else:
            try:
                # Fetch fresh data with tier-specific strategy
                results = await self._fetch_tier_data_enhanced(
                    tier, symbols, data_types, force_refresh, logger
                )
                self._last_tier_update[tier] = datetime.utcnow()
                
            except RateLimitException as e:
                logger.warning(
                    "rate_limit_hit",
                    provider=e.details['provider'],
                    retry_after=e.retry_after
                )
                
                # Record metric
                base_metrics.rate_limit_hits.labels(
                    limit_type="api",
                    resource=e.details['provider']
                ).inc()
                
                # Fallback to cache
                results = await self._fetch_cached_only_with_metrics(symbols, data_types)
                
            except APITimeoutException as e:
                logger.warning("api_timeout", details=e.to_dict())
                
                # Retry with exponential backoff
                await asyncio.sleep(e.retry_after)
                results = await self._fetch_tier_data_enhanced(
                    tier, symbols, data_types, force_refresh, logger
                )
        
        return results
    
    async def _fetch_tier_data_enhanced(
        self,
        tier: StockTier,
        symbols: List[str],
        data_types: List[str],
        force_refresh: bool,
        logger: StructuredLogger
    ) -> Dict[str, Any]:
        """Enhanced fetch with optimized caching and error handling."""
        results = {}
        
        # Get providers for this tier
        providers = self.tier_providers.get(tier, ['finnhub'])
        
        # Generate optimized cache keys
        cache_keys = {}
        for symbol in symbols:
            for data_type in data_types:
                cache_key = self._generate_optimized_cache_key(symbol, data_type, tier)
                cache_keys[(symbol, data_type)] = cache_key
        
        # Try cache first unless force refresh
        if not force_refresh:
            cached_results = await self._get_cached_batch_optimized(cache_keys)
            
            # Filter out symbols that have valid cache
            uncached_items = [
                (s, dt) for (s, dt), key in cache_keys.items()
                if key not in cached_results or self._is_stale(cached_results.get(key))
            ]
            
            # Process cached results
            for (symbol, data_type), cache_key in cache_keys.items():
                if cache_key in cached_results:
                    if symbol not in results:
                        results[symbol] = {}
                    results[symbol][data_type] = cached_results[cache_key]
        else:
            uncached_items = [(s, dt) for s in symbols for dt in data_types]
        
        if not uncached_items:
            return results
        
        # Create tasks for uncached items
        tasks = await self._create_api_tasks(uncached_items, providers, tier)
        
        if tasks:
            # Process tasks in parallel with error handling
            task_results = await self._process_tasks_with_recovery(tasks, logger)
            
            # Process and cache results
            await self._process_and_cache_results(task_results, tasks, tier, results, logger)
        
        return results
    
    def _generate_optimized_cache_key(
        self,
        symbol: str,
        data_type: str,
        tier: StockTier
    ) -> str:
        """Generate optimized cache key using new cache optimization."""
        namespace_map = {
            'price': CacheNamespace.PRICE,
            'fundamentals': CacheNamespace.FUNDAMENTAL,
            'technical': CacheNamespace.TECHNICAL,
            'sentiment': CacheNamespace.SENTIMENT,
            'news': CacheNamespace.SENTIMENT
        }
        
        namespace = namespace_map.get(data_type, CacheNamespace.SYSTEM)
        
        # Include tier in params for different TTLs
        params = {'tier': tier.value}
        
        # Add date partition for time-series data
        date_partition = None
        if data_type in ['price', 'technical']:
            date_partition = datetime.utcnow().date()
        
        return cache_key_gen.generate_key(
            namespace=namespace,
            identifier=symbol,
            params=params,
            date_partition=date_partition,
            ttl_bucket=str(tier.value)  # Group by tier for TTL management
        )
    
    async def _get_cached_batch_optimized(
        self,
        cache_keys: Dict[Tuple[str, str], str]
    ) -> Dict[str, Any]:
        """Get cached data with optimized batch operations."""
        # Get cache node for each key
        key_to_node = {}
        for key in cache_keys.values():
            node = self.hash_ring.get_node(key)
            if node not in key_to_node:
                key_to_node[node] = []
            key_to_node[node].append(key)
        
        # Batch get from each node
        results = {}
        for node, keys in key_to_node.items():
            try:
                node_results = await self.cache.get_batch(keys)
                results.update(node_results)
            except CacheConnectionException as e:
                logger.warning(f"cache_node_unavailable", node=node, error=str(e))
                # Continue with other nodes
                continue
        
        return results
    
    async def _create_api_tasks(
        self,
        uncached_items: List[Tuple[str, str]],
        providers: List[str],
        tier: StockTier
    ) -> List[APITask]:
        """Create API tasks with rate limit checking."""
        tasks = []
        
        for symbol, data_type in uncached_items:
            for provider in providers:
                if provider == 'cache_only':
                    continue
                
                # Check rate limit preemptively
                can_call = await self.rate_limiter.check_api_limit(provider)
                if not can_call[0]:
                    # Try next provider
                    continue
                
                task = APITask(
                    id=f"{symbol}_{data_type}_{provider}_{datetime.utcnow().timestamp()}",
                    provider=provider,
                    endpoint=self._get_endpoint(provider, data_type),
                    params=self._build_params(symbol, data_type, provider),
                    priority=self._tier_to_priority(tier)
                )
                tasks.append(task)
                break  # Use first available provider
        
        return tasks
    
    async def _process_tasks_with_recovery(
        self,
        tasks: List[APITask],
        logger: StructuredLogger
    ) -> List[TaskResult]:
        """Process tasks with error recovery."""
        try:
            task_results = await self.processor.process_batch(tasks, timeout=30)
            return task_results
            
        except Exception as e:
            logger.error("batch_processing_failed", exception=e, task_count=len(tasks))
            
            # Process tasks individually as fallback
            results = []
            for task in tasks:
                try:
                    result = await self.processor.process_single(task)
                    results.append(result)
                except Exception as task_error:
                    logger.warning(
                        "task_failed",
                        task_id=task.id,
                        error=str(task_error)
                    )
                    # Create failed result
                    results.append(TaskResult(
                        task_id=task.id,
                        success=False,
                        data=None,
                        error=str(task_error),
                        latency_ms=0
                    ))
            
            return results
    
    async def _process_and_cache_results(
        self,
        task_results: List[TaskResult],
        tasks: List[APITask],
        tier: StockTier,
        results: Dict[str, Any],
        logger: StructuredLogger
    ):
        """Process task results and update cache."""
        for task, result in zip(tasks, task_results):
            if result.success:
                symbol = self._extract_symbol_from_task(task)
                data_type = self._extract_data_type_from_task(task)
                
                # Validate data quality
                try:
                    if data_type == 'price' and isinstance(result.data, dict):
                        await self._validate_price_data(symbol, result.data)
                except DataQualityException as e:
                    logger.warning(
                        "data_quality_issue",
                        symbol=symbol,
                        quality_score=e.details.get('quality_score'),
                        issues=e.details.get('issues')
                    )
                    # Continue but mark as degraded
                    result.data['_quality_degraded'] = True
                
                # Store result
                if symbol not in results:
                    results[symbol] = {}
                results[symbol][data_type] = result.data
                
                # Cache successful results with optimized key
                cache_key = self._generate_optimized_cache_key(symbol, data_type, tier)
                ttl = self._get_cache_ttl(tier, data_type)
                
                await self.cache.set(cache_key, result.data, ttl=ttl)
                
                # Log successful fetch
                logger.log_api_call(
                    provider=task.provider,
                    endpoint=task.endpoint,
                    duration_ms=result.latency_ms,
                    success=True,
                    symbol=symbol,
                    data_type=data_type
                )
                
                # Record cost
                await self.cost_monitor.record_api_call(
                    provider=task.provider,
                    endpoint=task.endpoint,
                    success=True,
                    response_time_ms=result.latency_ms,
                    estimated_cost=0.0  # Free tier
                )
            else:
                # Log failure
                logger.warning(
                    "api_call_failed",
                    task_id=task.id,
                    provider=task.provider,
                    error=result.error
                )
    
    async def _validate_price_data(self, symbol: str, data: Dict[str, Any]):
        """Validate price data quality."""
        import pandas as pd
        
        # Convert to DataFrame for validation
        df = pd.DataFrame([data])
        
        # Perform quality check with metrics
        quality_result = await check_data_quality_with_metrics(df, symbol, 'price')
        
        if not quality_result.get('valid'):
            quality_score = quality_result.get('quality_score', 0)
            issues = quality_result.get('issues', [])
            
            # Raise exception if critical issues
            critical_issues = [i for i in issues if i.get('severity') == 'critical']
            if critical_issues:
                raise DataQualityException(
                    symbol=symbol,
                    quality_score=quality_score,
                    issues=critical_issues
                )
    
    async def _fetch_cached_only_with_metrics(
        self,
        symbols: List[str],
        data_types: List[str]
    ) -> Dict[str, Any]:
        """Fetch cached data with metrics recording."""
        results = {}
        
        for symbol in symbols:
            symbol_data = {}
            for data_type in data_types:
                # Use any tier for cache key generation
                cache_key = self._generate_optimized_cache_key(
                    symbol, data_type, StockTier.MEDIUM
                )
                
                start_time = time.time()
                cached = await self.cache.get(cache_key)
                duration_ms = (time.time() - start_time) * 1000
                
                # Log cache operation
                logger.log_cache_operation(
                    operation="get",
                    key=cache_key,
                    hit=cached is not None,
                    duration_ms=duration_ms
                )
                
                if cached:
                    symbol_data[data_type] = cached
                    # Record cache hit
                    base_metrics.track_cache_operation(
                        cache_type="redis",
                        operation="get",
                        hit=True
                    )
                else:
                    # Try stale cache
                    stale_key = f"{cache_key}:stale"
                    stale = await self.cache.get(stale_key)
                    if stale:
                        symbol_data[data_type] = stale
                        symbol_data[f"{data_type}_stale"] = True
                        
                        # Record data staleness
                        dq_metrics.record_data_staleness(
                            symbol=symbol,
                            data_type=data_type,
                            last_update=datetime.utcnow() - timedelta(days=1)
                        )
                    else:
                        # Record cache miss
                        base_metrics.track_cache_operation(
                            cache_type="redis",
                            operation="get",
                            hit=False
                        )
            
            if symbol_data:
                results[symbol] = symbol_data
        
        return results
    
    async def _validate_results_quality(
        self,
        results: Dict[str, Any],
        logger: StructuredLogger
    ):
        """Validate quality of fetched results."""
        for symbol, data in results.items():
            if 'price' in data:
                # Check for data quality issues
                quality_issues = []
                
                # Check for missing fields
                required_fields = ['open', 'high', 'low', 'close', 'volume']
                missing_fields = [f for f in required_fields if f not in data['price']]
                if missing_fields:
                    quality_issues.append({
                        'type': 'missing_fields',
                        'fields': missing_fields
                    })
                
                # Check for price consistency
                price_data = data['price']
                if all(f in price_data for f in ['high', 'low', 'close']):
                    if price_data['high'] < price_data['low']:
                        quality_issues.append({
                            'type': 'price_consistency',
                            'issue': 'high < low'
                        })
                    if price_data['close'] > price_data['high'] or price_data['close'] < price_data['low']:
                        quality_issues.append({
                            'type': 'price_consistency',
                            'issue': 'close outside range'
                        })
                
                if quality_issues:
                    logger.warning(
                        "data_quality_issues",
                        symbol=symbol,
                        issues=quality_issues
                    )
                    
                    # Record metrics
                    for issue in quality_issues:
                        dq_metrics.record_validation_failure(
                            validation_type=issue['type'],
                            field='price',
                            reason=issue.get('issue', 'unknown'),
                            data_type='price'
                        )
    
    def _build_batch_cache_key(self, symbols: List[str], data_types: List[str]) -> str:
        """Build cache key for batch operation."""
        symbols_hash = hashlib.md5('_'.join(sorted(symbols)).encode()).hexdigest()[:8]
        types_hash = hashlib.md5('_'.join(sorted(data_types)).encode()).hexdigest()[:8]
        return f"batch:{symbols_hash}:{types_hash}"
    
    def _should_update_tier(self, tier: StockTier) -> bool:
        """Check if a tier needs updating based on frequency."""
        if tier not in self._last_tier_update:
            return True
        
        last_update = self._last_tier_update[tier]
        update_frequency = self.tier_update_frequencies[tier]
        
        return (datetime.utcnow() - last_update).total_seconds() >= update_frequency
    
    def _group_symbols_by_tier(self, symbols: List[str]) -> Dict[StockTier, List[str]]:
        """Group symbols by their tier."""
        tier_groups = {tier: [] for tier in StockTier}
        
        for symbol in symbols:
            tier = self.get_stock_tier(symbol)
            tier_groups[tier].append(symbol)
        
        return tier_groups
    
    def get_stock_tier(self, symbol: str) -> StockTier:
        """Get the tier for a given stock symbol."""
        for tier, symbols in self.stock_tiers.items():
            if symbol in symbols:
                return tier
        return StockTier.LOW  # Default tier
    
    def _tier_to_priority(self, tier: StockTier) -> Priority:
        """Convert stock tier to task priority."""
        mapping = {
            StockTier.CRITICAL: Priority.CRITICAL,
            StockTier.HIGH: Priority.HIGH,
            StockTier.MEDIUM: Priority.MEDIUM,
            StockTier.LOW: Priority.LOW,
            StockTier.MINIMAL: Priority.BATCH
        }
        return mapping.get(tier, Priority.MEDIUM)
    
    def _get_endpoint(self, provider: str, data_type: str) -> str:
        """Get API endpoint for provider and data type."""
        endpoints = {
            'finnhub': {
                'price': 'quote',
                'fundamentals': 'stock/profile2',
                'news': 'company-news',
                'sentiment': 'news-sentiment'
            },
            'alpha_vantage': {
                'price': 'GLOBAL_QUOTE',
                'fundamentals': 'OVERVIEW',
                'technical': 'RSI'
            },
            'polygon': {
                'price': 'v2/aggs/ticker',
                'fundamentals': 'v3/reference/tickers'
            }
        }
        
        return endpoints.get(provider, {}).get(data_type, '')
    
    def _build_params(
        self,
        symbol: str,
        data_type: str,
        provider: str
    ) -> Dict[str, Any]:
        """Build API parameters."""
        if provider == 'finnhub':
            return {'symbol': symbol}
        elif provider == 'alpha_vantage':
            function_map = {
                'price': 'GLOBAL_QUOTE',
                'fundamentals': 'OVERVIEW',
                'technical': 'RSI'
            }
            return {
                'symbol': symbol,
                'function': function_map.get(data_type, 'GLOBAL_QUOTE')
            }
        elif provider == 'polygon':
            return {'ticker': symbol}
        
        return {}
    
    def _extract_symbol_from_task(self, task: APITask) -> str:
        """Extract symbol from task."""
        if 'symbol' in task.params:
            return task.params['symbol']
        elif 'ticker' in task.params:
            return task.params['ticker']
        
        # Parse from ID
        parts = task.id.split('_')
        if parts:
            return parts[0]
        
        return 'UNKNOWN'
    
    def _extract_data_type_from_task(self, task: APITask) -> str:
        """Extract data type from task."""
        parts = task.id.split('_')
        if len(parts) > 1:
            return parts[1]
        
        return 'unknown'
    
    def _get_cache_ttl(self, tier: StockTier, data_type: str) -> int:
        """Get cache TTL based on tier and data type."""
        base_ttl = {
            'price': 300,           # 5 minutes
            'fundamentals': 86400,  # 1 day
            'news': 3600,          # 1 hour
            'technical': 900       # 15 minutes
        }.get(data_type, 3600)
        
        # Multiply by tier factor
        tier_multiplier = {
            StockTier.CRITICAL: 1,
            StockTier.HIGH: 2,
            StockTier.MEDIUM: 4,
            StockTier.LOW: 8,
            StockTier.MINIMAL: 24
        }.get(tier, 1)
        
        return base_ttl * tier_multiplier
    
    def _is_stale(self, data: Any) -> bool:
        """Check if data is stale."""
        if isinstance(data, dict):
            if '_stale' in data:
                return data['_stale']
            if '_timestamp' in data:
                data_age = (datetime.utcnow() - datetime.fromisoformat(data['_timestamp'])).total_seconds()
                return data_age > 86400  # Consider stale after 1 day
        return False
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        metrics = {
            'cache': await self.cache.get_stats() if self.cache else {},
            'rate_limiter': await self.rate_limiter.get_usage_stats() if self.rate_limiter else {},
            'processor': self.processor.get_performance_stats() if self.processor else {},
            'cost_monitor': await self.cost_monitor.get_usage_report() if self.cost_monitor else {},
            'stock_tiers': {
                tier.name: len(symbols) 
                for tier, symbols in self.stock_tiers.items()
            },
            'exception_stats': exception_handler.get_error_statistics(),
            'correlation_id': self.correlation_id
        }
        
        return metrics
    
    async def shutdown(self) -> None:
        """Shutdown all components gracefully."""
        logger.info("shutdown_initiated", correlation_id=self.correlation_id)
        
        try:
            shutdown_tasks = []
            
            if self.processor:
                shutdown_tasks.append(self.processor.shutdown())
            if self.cache:
                shutdown_tasks.append(self.cache.close())
            if self.rate_limiter:
                shutdown_tasks.append(self.rate_limiter.close())
            
            if shutdown_tasks:
                await asyncio.gather(*shutdown_tasks, return_exceptions=True)
            
            logger.info("shutdown_complete", correlation_id=self.correlation_id)
            
        except Exception as e:
            logger.error("shutdown_error", exception=e)


# Global enhanced instance
enhanced_unified_ingestion = EnhancedUnifiedDataIngestion()


# Export for backward compatibility
unified_ingestion = enhanced_unified_ingestion