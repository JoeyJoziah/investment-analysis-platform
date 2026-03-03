"""
Cache warming mixin and result type for IntelligentCacheManager.

CacheWarmingResult holds warming statistics.
CacheWarmingMixin provides market-open pre-warming logic that is composed
into IntelligentCacheManager via multiple inheritance.
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CacheWarmingResult:
    """Result of a cache warming operation"""
    total_stocks: int = 0
    successful: int = 0
    failed: int = 0
    skipped: int = 0
    duration_seconds: float = 0.0
    errors: List[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.total_stocks == 0:
            return 0.0
        return self.successful / self.total_stocks


class CacheWarmingMixin:
    """
    Mixin that adds cache warming capability to IntelligentCacheManager.

    Requires the host class to expose:
        self.memory_cache   - MemoryTierCache instance
        self.disk_cache     - DiskTierCache instance
        self.redis_cache    - RedisTierCache instance
        self.bloom_filter   - BloomFilter instance
        self.get(key, category) -> coroutine
        self.set(key, data, category, ttl_hours) -> coroutine
    """

    async def get_top_stocks_by_volume(
        self,
        top_n: int = 500,
        db_url: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get top N stocks ranked by trading volume and market cap.

        This method queries the database to find the most frequently
        accessed stocks based on volume and market capitalization.

        Args:
            top_n: Number of top stocks to return
            db_url: Database connection URL (uses env var if not provided)

        Returns:
            List of stock dictionaries with symbol, name, volume, market_cap
        """
        # Build database URL from environment
        if db_url is None:
            host = os.getenv('POSTGRES_HOST', 'localhost')
            port = os.getenv('POSTGRES_PORT', '5432')
            db = os.getenv('POSTGRES_DB', 'investment_db')
            user = os.getenv('POSTGRES_USER', 'postgres')
            password = os.getenv('POSTGRES_PASSWORD', 'postgres')
            db_url = f"postgresql://{user}:{password}@{host}:{port}/{db}"

        top_stocks = []

        try:
            import asyncpg

            conn = await asyncpg.connect(db_url)

            try:
                # Query stocks ordered by market cap and recent volume
                # This prioritizes large-cap stocks that are frequently traded
                query = """
                    SELECT
                        s.symbol,
                        s.name,
                        s.market_cap,
                        s.sector,
                        COALESCE(
                            (SELECT AVG(ph.volume)
                             FROM price_history ph
                             WHERE ph.stock_id = s.id
                             AND ph.date >= CURRENT_DATE - INTERVAL '30 days'),
                            0
                        ) as avg_volume
                    FROM stocks s
                    WHERE s.is_active = true
                      AND s.is_tradable = true
                    ORDER BY
                        s.market_cap DESC NULLS LAST,
                        avg_volume DESC NULLS LAST
                    LIMIT $1
                """

                rows = await conn.fetch(query, top_n)

                for row in rows:
                    top_stocks.append({
                        'symbol': row['symbol'],
                        'name': row['name'],
                        'market_cap': row['market_cap'],
                        'sector': row['sector'],
                        'avg_volume': float(row['avg_volume'] or 0)
                    })

                logger.info(f"Retrieved {len(top_stocks)} top stocks for cache warming")

            finally:
                await conn.close()

        except ImportError:
            logger.warning("asyncpg not available, falling back to psycopg2")
            # Fallback to synchronous query with psycopg2
            top_stocks = await self._get_top_stocks_sync(top_n, db_url)

        except Exception as e:
            logger.error(f"Error getting top stocks: {e}")
            # Return default high-value stocks as fallback
            top_stocks = self._get_default_top_stocks()

        return top_stocks

    async def _get_top_stocks_sync(
        self,
        top_n: int,
        db_url: str
    ) -> List[Dict[str, Any]]:
        """Synchronous fallback for getting top stocks using psycopg2."""
        import psycopg2

        top_stocks = []

        db_config = {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': os.getenv('POSTGRES_PORT', '5432'),
            'database': os.getenv('POSTGRES_DB', 'investment_db'),
            'user': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD', 'postgres')
        }

        try:
            conn = psycopg2.connect(**db_config)
            cursor = conn.cursor()

            query = """
                SELECT
                    s.symbol,
                    s.name,
                    s.market_cap,
                    s.sector
                FROM stocks s
                WHERE s.is_active = true
                ORDER BY s.market_cap DESC NULLS LAST
                LIMIT %s
            """

            cursor.execute(query, (top_n,))
            rows = cursor.fetchall()

            for row in rows:
                top_stocks.append({
                    'symbol': row[0],
                    'name': row[1],
                    'market_cap': row[2],
                    'sector': row[3],
                    'avg_volume': 0
                })

            cursor.close()
            conn.close()

        except Exception as e:
            logger.error(f"Error in sync top stocks query: {e}")
            top_stocks = self._get_default_top_stocks()

        return top_stocks

    def _get_default_top_stocks(self) -> List[Dict[str, Any]]:
        """Return default list of high-value stocks when database unavailable."""
        # Top 100 stocks by market cap (as of 2025)
        default_symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B',
            'UNH', 'XOM', 'JNJ', 'JPM', 'V', 'PG', 'MA', 'HD', 'CVX', 'MRK',
            'ABBV', 'LLY', 'PEP', 'KO', 'COST', 'AVGO', 'TMO', 'MCD', 'WMT',
            'CSCO', 'ACN', 'ABT', 'DHR', 'PFE', 'BAC', 'CRM', 'NKE', 'ADBE',
            'NFLX', 'ORCL', 'DIS', 'INTC', 'VZ', 'CMCSA', 'TXN', 'PM', 'WFC',
            'BMY', 'UPS', 'RTX', 'HON', 'QCOM', 'NEE', 'COP', 'LOW', 'SPGI',
            'UNP', 'GS', 'MS', 'ELV', 'CAT', 'INTU', 'DE', 'AMD', 'SBUX',
            'IBM', 'BA', 'AXP', 'BLK', 'LMT', 'GILD', 'GE', 'AMT', 'MDLZ',
            'ISRG', 'ADI', 'SYK', 'BKNG', 'TJX', 'CVS', 'PLD', 'VRTX', 'ADP',
            'REGN', 'MMC', 'LRCX', 'CI', 'NOW', 'MO', 'TMUS', 'ZTS', 'PGR',
            'CB', 'SO', 'BSX', 'EOG', 'DUK', 'BDX', 'SCHW', 'FI', 'ATVI', 'SLB'
        ]

        return [
            {'symbol': sym, 'name': sym, 'market_cap': None, 'sector': None, 'avg_volume': 0}
            for sym in default_symbols
        ]

    async def warm_cache_for_market_open(
        self,
        top_n: int = 500,
        data_fetcher: Optional[Callable] = None,
        batch_size: int = 50,
        max_concurrent: int = 10,
        rate_limit_delay: float = 1.0,
        ttl_hours: int = 4
    ) -> CacheWarmingResult:
        """
        Pre-warm cache with top N stocks before market open.

        This method proactively loads frequently accessed stock data
        into the cache before market open to ensure fast response times
        when trading begins.

        Args:
            top_n: Number of top stocks to pre-load (default 500)
            data_fetcher: Optional async function to fetch stock data.
                         Signature: async def fetcher(symbol: str) -> Dict
                         If not provided, uses cached data from lower tiers.
            batch_size: Number of stocks to process in each batch
            max_concurrent: Maximum concurrent fetch operations
            rate_limit_delay: Delay between batches to respect rate limits
            ttl_hours: TTL for warmed cache entries (default 4 hours)

        Returns:
            CacheWarmingResult with warming statistics
        """
        start_time = time.time()
        result = CacheWarmingResult(total_stocks=top_n)

        logger.info(f"Starting cache warming for market open with {top_n} stocks")

        try:
            # Get top stocks to warm
            top_stocks = await self.get_top_stocks_by_volume(top_n)
            result.total_stocks = len(top_stocks)

            if not top_stocks:
                logger.warning("No stocks found for cache warming")
                result.duration_seconds = time.time() - start_time
                return result

            # Process in batches to respect rate limits
            batches = [
                top_stocks[i:i + batch_size]
                for i in range(0, len(top_stocks), batch_size)
            ]

            for batch_idx, batch in enumerate(batches):
                batch_start = time.time()
                logger.info(f"Warming batch {batch_idx + 1}/{len(batches)} "
                            f"({len(batch)} stocks)")

                # Create tasks for concurrent warming
                semaphore = asyncio.Semaphore(max_concurrent)

                async def warm_single_stock(stock: Dict) -> bool:
                    async with semaphore:
                        return await self._warm_stock_data(
                            stock=stock,
                            data_fetcher=data_fetcher,
                            ttl_hours=ttl_hours
                        )

                # Execute batch concurrently
                tasks = [warm_single_stock(stock) for stock in batch]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                # Tally results
                for stock, batch_result in zip(batch, batch_results):
                    if isinstance(batch_result, Exception):
                        result.failed += 1
                        result.errors.append(
                            f"{stock['symbol']}: {str(batch_result)}"
                        )
                    elif batch_result is True:
                        result.successful += 1
                    else:
                        result.skipped += 1

                batch_duration = time.time() - batch_start
                logger.info(
                    f"Batch {batch_idx + 1} completed in {batch_duration:.2f}s "
                    f"(success: {sum(1 for r in batch_results if r is True)})"
                )

                # Rate limit delay between batches (except last batch)
                if batch_idx < len(batches) - 1:
                    await asyncio.sleep(rate_limit_delay)

            result.duration_seconds = time.time() - start_time

            logger.info(
                f"Cache warming completed: {result.successful}/{result.total_stocks} "
                f"stocks warmed in {result.duration_seconds:.2f}s "
                f"(success rate: {result.success_rate:.1%})"
            )

        except Exception as e:
            logger.error(f"Cache warming failed: {e}")
            result.errors.append(f"Global error: {str(e)}")
            result.duration_seconds = time.time() - start_time

        return result

    async def _warm_stock_data(
        self,
        stock: Dict[str, Any],
        data_fetcher: Optional[Callable],
        ttl_hours: int
    ) -> bool:
        """
        Warm cache for a single stock.

        Args:
            stock: Stock info dict with 'symbol' key
            data_fetcher: Optional async function to fetch fresh data
            ttl_hours: TTL for cache entry

        Returns:
            True if successfully warmed, False if skipped/already cached
        """
        symbol = stock['symbol']

        try:
            # Check if already in memory cache (hot)
            cache_key = f"stock:{symbol}:summary"
            existing = await self.get(cache_key, category='stocks')

            if existing is not None:
                # Already in cache, just promote to memory if not there
                logger.debug(f"Stock {symbol} already cached, skipping")
                return False

            # Fetch fresh data if fetcher provided
            if data_fetcher is not None:
                try:
                    data = await data_fetcher(symbol)
                    if data:
                        await self.set(cache_key, data, category='stocks', ttl_hours=ttl_hours)

                        # Also cache price history key
                        price_key = f"stock:{symbol}:prices"
                        if 'prices' in data:
                            await self.set(price_key, data['prices'],
                                           category='stocks', ttl_hours=ttl_hours)

                        # Cache technical indicators
                        indicators_key = f"stock:{symbol}:indicators"
                        if 'indicators' in data:
                            await self.set(indicators_key, data['indicators'],
                                           category='stocks', ttl_hours=ttl_hours)

                        # Cache fundamentals
                        fundamentals_key = f"stock:{symbol}:fundamentals"
                        if 'fundamentals' in data:
                            await self.set(fundamentals_key, data['fundamentals'],
                                           category='stocks', ttl_hours=ttl_hours)

                        return True

                except Exception as e:
                    logger.warning(f"Failed to fetch data for {symbol}: {e}")
                    # Continue to try promoting from lower tiers

            # Try to promote from disk/Redis to memory
            promoted = await self._promote_stock_to_memory(symbol, ttl_hours)
            return promoted

        except Exception as e:
            logger.error(f"Error warming cache for {symbol}: {e}")
            raise

    async def _promote_stock_to_memory(
        self,
        symbol: str,
        ttl_hours: int
    ) -> bool:
        """
        Try to promote stock data from lower cache tiers to memory.

        Args:
            symbol: Stock symbol
            ttl_hours: TTL for memory cache entry

        Returns:
            True if data was promoted, False otherwise
        """
        cache_patterns = [
            f"stock:{symbol}:summary",
            f"stock:{symbol}:prices",
            f"stock:{symbol}:indicators",
            f"stock:{symbol}:fundamentals"
        ]

        promoted_any = False

        for key in cache_patterns:
            # Try disk cache first
            data = self.disk_cache.get(key)
            if data is not None:
                self.memory_cache.set(key, data, ttl_hours * 3600)
                self.bloom_filter.add(key)
                promoted_any = True
                continue

            # Try Redis cache
            if self.redis_cache.is_available():
                data = self.redis_cache.get(key)
                if data is not None:
                    self.memory_cache.set(key, data, ttl_hours * 3600)
                    self.disk_cache.set(key, data, ttl_hours)
                    self.bloom_filter.add(key)
                    promoted_any = True

        return promoted_any

    def get_warming_status(self) -> Dict[str, Any]:
        """
        Get current cache warming status and readiness metrics.

        Returns:
            Dict with cache readiness information
        """
        memory_stats = self.memory_cache.get_stats()
        disk_stats = self.disk_cache.get_stats()

        return {
            'memory_cache': {
                'entries': memory_stats.get('entries', 0),
                'utilization': memory_stats.get('utilization', 0),
                'size_mb': memory_stats.get('size_mb', 0)
            },
            'disk_cache': {
                'entries': disk_stats.get('entries', 0),
                'utilization': disk_stats.get('utilization', 0),
                'size_mb': disk_stats.get('size_mb', 0)
            },
            'bloom_filter': {
                'items_tracked': self.bloom_filter.get_stats().get('items_added', 0)
            },
            'estimated_coverage': self._estimate_stock_coverage()
        }

    def _estimate_stock_coverage(self) -> float:
        """
        Estimate the percentage of top stocks that are cached.

        Returns:
            Coverage percentage (0.0 to 1.0)
        """
        # Check bloom filter for common stock patterns
        test_symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
            'JPM', 'V', 'JNJ', 'WMT', 'PG', 'UNH', 'DIS', 'MA'
        ]

        cached_count = 0
        for symbol in test_symbols:
            key = f"stock:{symbol}:summary"
            if self.bloom_filter.might_contain(key):
                cached_count += 1

        return cached_count / len(test_symbols)
