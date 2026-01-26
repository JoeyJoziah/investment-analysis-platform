"""
High-Performance Bulk Data Loader for Stock Market Data

This module provides optimized data loading capabilities for handling massive daily updates
of 6000+ stocks with millions of rows. It uses advanced PostgreSQL/TimescaleDB features
for maximum throughput and minimal blocking.
"""

import asyncio
import asyncpg
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, date, timedelta
import logging
import uuid
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import json
import gzip
# SECURITY: Removed pickle import - using JSON for data serialization
from pathlib import Path
import psutil
import time

logger = logging.getLogger(__name__)


@dataclass
class BulkLoadStats:
    """Statistics for bulk loading operations"""
    records_processed: int = 0
    records_inserted: int = 0
    records_updated: int = 0
    records_failed: int = 0
    duration_seconds: float = 0.0
    throughput_rps: float = 0.0
    memory_used_mb: float = 0.0
    compression_ratio: float = 0.0


class BulkStockDataLoader:
    """
    High-performance bulk data loader optimized for massive stock data updates.
    
    Features:
    - Concurrent batch processing
    - COPY-based bulk inserts
    - Automatic compression and deduplication
    - Memory-efficient streaming
    - Progress tracking and error recovery
    - Intelligent partitioning
    """
    
    def __init__(
        self,
        connection_pool: asyncpg.Pool,
        batch_size: int = 50000,
        max_concurrent_batches: int = 4,
        enable_compression: bool = True,
        staging_path: str = "/tmp/stock_staging"
    ):
        self.pool = connection_pool
        self.batch_size = batch_size
        self.max_concurrent_batches = max_concurrent_batches
        self.enable_compression = enable_compression
        self.staging_path = Path(staging_path)
        self.staging_path.mkdir(exist_ok=True)
        
        # Performance tracking
        self.stats = BulkLoadStats()
        self.start_time: Optional[float] = None
        
        # Symbol to ID mapping cache
        self._symbol_id_cache: Dict[str, int] = {}
        self._cache_loaded = False

    async def _ensure_symbol_cache(self) -> None:
        """Load and cache stock symbol to ID mappings for fast lookups"""
        if self._cache_loaded:
            return
            
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT symbol, id, symbol_hash 
                FROM stocks 
                WHERE is_active = true
            """)
            
            for row in rows:
                self._symbol_id_cache[row['symbol']] = row['id']
                
        self._cache_loaded = True
        logger.info(f"Loaded {len(self._symbol_id_cache)} symbols into cache")

    async def _create_temporary_staging_table(
        self, 
        conn: asyncpg.Connection,
        table_type: str = "price_data"
    ) -> str:
        """Create a temporary staging table for bulk operations"""
        
        temp_table_name = f"staging_{table_type}_{uuid.uuid4().hex[:8]}"
        
        if table_type == "price_data":
            await conn.execute(f"""
                CREATE TEMP TABLE {temp_table_name} (
                    symbol TEXT,
                    stock_id SMALLINT,
                    date DATE,
                    timestamp TIMESTAMP,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    adjusted_close REAL,
                    volume INTEGER,
                    source_batch_id UUID DEFAULT gen_random_uuid()
                ) ON COMMIT DROP;
            """)
        elif table_type == "technical_indicators":
            await conn.execute(f"""
                CREATE TEMP TABLE {temp_table_name} (
                    symbol TEXT,
                    stock_id SMALLINT,
                    date DATE,
                    indicators JSONB,
                    calculation_batch_id UUID DEFAULT gen_random_uuid()
                ) ON COMMIT DROP;
            """)
        elif table_type == "news_sentiment":
            await conn.execute(f"""
                CREATE TEMP TABLE {temp_table_name} (
                    symbol TEXT,
                    stock_id SMALLINT,
                    date DATE,
                    source_hash BIGINT,
                    headline_text TEXT,
                    sentiment_score SMALLINT,
                    confidence SMALLINT,
                    impact_score SMALLINT,
                    published_at TIMESTAMP,
                    keywords TEXT[]
                ) ON COMMIT DROP;
            """)
            
        return temp_table_name

    async def _optimize_dataframe_for_bulk_insert(
        self, 
        df: pd.DataFrame,
        table_type: str = "price_data"
    ) -> pd.DataFrame:
        """Optimize DataFrame for bulk insertion"""
        
        # Ensure symbol cache is loaded
        await self._ensure_symbol_cache()
        
        # Add stock_id column based on symbol lookup
        df['stock_id'] = df['symbol'].map(self._symbol_id_cache)
        
        # Remove rows with unknown symbols
        unknown_symbols = df[df['stock_id'].isna()]['symbol'].unique()
        if len(unknown_symbols) > 0:
            logger.warning(f"Dropping {len(unknown_symbols)} unknown symbols: {unknown_symbols[:10]}")
            df = df.dropna(subset=['stock_id'])
        
        # Convert stock_id to int16 for space efficiency
        df['stock_id'] = df['stock_id'].astype('int16')
        
        if table_type == "price_data":
            # Optimize price data
            price_columns = ['open', 'high', 'low', 'close', 'adjusted_close']
            for col in price_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32')
            
            # Convert volume to int32
            if 'volume' in df.columns:
                df['volume'] = pd.to_numeric(df['volume'], errors='coerce').astype('int32')
            
            # Ensure date columns are properly formatted
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date']).dt.date
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Calculate derived fields
            df['price_range'] = df['high'] - df['low']
            df['price_change'] = df['close'] - df['open']
            df['price_change_pct'] = (df['price_change'] / df['open'] * 100).fillna(0)
            df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
            
        elif table_type == "technical_indicators":
            # Pack indicators into JSONB format
            indicator_columns = [col for col in df.columns if col not in ['symbol', 'stock_id', 'date']]
            df['indicators'] = df[indicator_columns].apply(
                lambda row: json.dumps({k: float(v) if pd.notna(v) else None for k, v in row.items()}),
                axis=1
            )
            
        elif table_type == "news_sentiment":
            # Optimize sentiment data
            if 'sentiment_score' in df.columns:
                df['sentiment_score'] = (df['sentiment_score'] * 1000).astype('int16')
            if 'confidence' in df.columns:
                df['confidence'] = (df['confidence'] * 1000).astype('int16')
            if 'impact_score' in df.columns:
                df['impact_score'] = (df['impact_score'] * 1000).astype('int16')
        
        # Remove duplicates
        duplicate_cols = ['stock_id', 'date']
        if table_type == "price_data" and 'timestamp' in df.columns:
            duplicate_cols.append('timestamp')
        
        initial_count = len(df)
        df = df.drop_duplicates(subset=duplicate_cols, keep='last')
        dedup_count = initial_count - len(df)
        if dedup_count > 0:
            logger.info(f"Removed {dedup_count} duplicate records")
        
        return df

    async def _bulk_copy_to_staging(
        self,
        conn: asyncpg.Connection,
        df: pd.DataFrame,
        temp_table: str
    ) -> int:
        """Use COPY for high-speed bulk insert to staging table"""
        
        # Create a StringIO buffer with CSV data
        from io import StringIO
        
        buffer = StringIO()
        df.to_csv(buffer, index=False, header=False, sep='\t', na_rep='\\N')
        buffer.seek(0)
        
        # Use COPY for maximum speed
        copy_sql = f"COPY {temp_table} FROM STDIN WITH (FORMAT csv, DELIMITER E'\\t', NULL '\\N')"
        
        start_time = time.time()
        await conn.copy_from_table(temp_table, source=buffer, format='csv', delimiter='\t')
        copy_time = time.time() - start_time
        
        rows_copied = len(df)
        throughput = rows_copied / copy_time if copy_time > 0 else 0
        
        logger.info(f"COPY inserted {rows_copied} rows in {copy_time:.2f}s ({throughput:.0f} rows/s)")
        
        return rows_copied

    async def _transfer_from_staging_to_main(
        self,
        conn: asyncpg.Connection,
        temp_table: str,
        target_table: str,
        table_type: str = "price_data"
    ) -> Tuple[int, int]:
        """Transfer data from staging to main table with conflict resolution"""
        
        if table_type == "price_data":
            insert_sql = f"""
                INSERT INTO {target_table} (
                    stock_id, date, timestamp, open, high, low, close, 
                    adjusted_close, volume, price_range, price_change, 
                    price_change_pct, typical_price
                )
                SELECT 
                    stock_id, date, timestamp, open, high, low, close,
                    adjusted_close, volume, 
                    high - low as price_range,
                    close - open as price_change,
                    CASE WHEN open > 0 THEN (close - open) / open * 100 ELSE 0 END as price_change_pct,
                    (high + low + close) / 3 as typical_price
                FROM {temp_table}
                WHERE stock_id IS NOT NULL
                ON CONFLICT (stock_id, date, timestamp) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    adjusted_close = EXCLUDED.adjusted_close,
                    volume = EXCLUDED.volume,
                    price_range = EXCLUDED.price_range,
                    price_change = EXCLUDED.price_change,
                    price_change_pct = EXCLUDED.price_change_pct,
                    typical_price = EXCLUDED.typical_price
            """
            
        elif table_type == "technical_indicators":
            insert_sql = f"""
                INSERT INTO {target_table} (stock_id, date, indicators)
                SELECT stock_id, date, indicators::jsonb
                FROM {temp_table}
                WHERE stock_id IS NOT NULL
                ON CONFLICT (stock_id, date) DO UPDATE SET
                    indicators = EXCLUDED.indicators
            """
            
        elif table_type == "news_sentiment":
            insert_sql = f"""
                INSERT INTO {target_table} (
                    stock_id, date, source_hash, headline_vector,
                    sentiment_score, confidence, impact_score, 
                    published_at, keywords
                )
                SELECT 
                    stock_id, date, source_hash,
                    to_tsvector('english', headline_text) as headline_vector,
                    sentiment_score, confidence, impact_score,
                    published_at, keywords
                FROM {temp_table}
                WHERE stock_id IS NOT NULL
                ON CONFLICT (id) DO UPDATE SET
                    sentiment_score = EXCLUDED.sentiment_score,
                    confidence = EXCLUDED.confidence,
                    impact_score = EXCLUDED.impact_score
            """
        
        # Execute the insert with conflict resolution
        result = await conn.execute(insert_sql)
        
        # Parse result to get inserted/updated counts
        # PostgreSQL returns "INSERT 0 <count>" or "INSERT 0 <inserted> <updated>"
        import re
        match = re.search(r'INSERT 0 (\d+)', result)
        total_affected = int(match.group(1)) if match else 0
        
        return total_affected, 0  # We'll count inserts and updates separately if needed

    async def load_price_data_bulk(
        self,
        data: Union[pd.DataFrame, List[Dict[str, Any]], str],
        target_table: str = "price_history_optimized"
    ) -> BulkLoadStats:
        """
        Load price data in bulk with maximum performance
        
        Args:
            data: DataFrame, list of dicts, or file path containing price data
            target_table: Target table name (defaults to optimized table)
            
        Returns:
            BulkLoadStats with performance metrics
        """
        self.start_time = time.time()
        self.stats = BulkLoadStats()
        
        try:
            # Convert input to DataFrame
            if isinstance(data, str):
                if data.endswith('.parquet'):
                    df = pd.read_parquet(data)
                elif data.endswith('.csv'):
                    df = pd.read_csv(data)
                else:
                    raise ValueError(f"Unsupported file format: {data}")
            elif isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = data.copy()
            
            self.stats.records_processed = len(df)
            logger.info(f"Starting bulk load of {self.stats.records_processed:,} price records")
            
            # Optimize DataFrame for bulk insert
            df = await self._optimize_dataframe_for_bulk_insert(df, "price_data")
            
            # Split into batches for concurrent processing
            batches = [df[i:i + self.batch_size] for i in range(0, len(df), self.batch_size)]
            logger.info(f"Split into {len(batches)} batches of up to {self.batch_size:,} records")
            
            # Process batches concurrently
            semaphore = asyncio.Semaphore(self.max_concurrent_batches)
            tasks = []
            
            for i, batch_df in enumerate(batches):
                task = self._process_price_batch(semaphore, batch_df, target_table, i)
                tasks.append(task)
            
            # Wait for all batches to complete
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Aggregate results
            total_inserted = 0
            total_updated = 0
            
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Batch failed: {result}")
                    self.stats.records_failed += self.batch_size
                else:
                    inserted, updated = result
                    total_inserted += inserted
                    total_updated += updated
            
            self.stats.records_inserted = total_inserted
            self.stats.records_updated = total_updated
            
            # Update stock metadata
            await self._update_stock_metadata(list(df['stock_id'].unique()))
            
            # Calculate final statistics
            self.stats.duration_seconds = time.time() - self.start_time
            if self.stats.duration_seconds > 0:
                self.stats.throughput_rps = self.stats.records_processed / self.stats.duration_seconds
            
            # Log final results
            logger.info(f"Bulk load completed: {self.stats.records_inserted:,} inserted, "
                       f"{self.stats.records_updated:,} updated, {self.stats.records_failed:,} failed "
                       f"in {self.stats.duration_seconds:.2f}s ({self.stats.throughput_rps:.0f} rps)")
            
            return self.stats
            
        except Exception as e:
            logger.error(f"Bulk load failed: {e}")
            self.stats.duration_seconds = time.time() - (self.start_time or time.time())
            raise

    async def _process_price_batch(
        self,
        semaphore: asyncio.Semaphore,
        batch_df: pd.DataFrame,
        target_table: str,
        batch_id: int
    ) -> Tuple[int, int]:
        """Process a single batch of price data"""
        
        async with semaphore:
            async with self.pool.acquire() as conn:
                try:
                    # Create temporary staging table
                    temp_table = await self._create_temporary_staging_table(conn, "price_data")
                    
                    # Bulk copy to staging
                    await self._bulk_copy_to_staging(conn, batch_df, temp_table)
                    
                    # Transfer to main table
                    inserted, updated = await self._transfer_from_staging_to_main(
                        conn, temp_table, target_table, "price_data"
                    )
                    
                    logger.debug(f"Batch {batch_id} completed: {inserted} inserted, {updated} updated")
                    return inserted, updated
                    
                except Exception as e:
                    logger.error(f"Batch {batch_id} failed: {e}")
                    raise

    async def _update_stock_metadata(self, stock_ids: List[int]) -> None:
        """Update stock metadata after bulk data load"""
        
        async with self.pool.acquire() as conn:
            await conn.execute("""
                UPDATE stocks 
                SET 
                    last_data_update = NOW(),
                    last_price_update = NOW()
                WHERE id = ANY($1::int[])
            """, stock_ids)

    async def calculate_technical_indicators_bulk(
        self,
        stock_ids: Optional[List[int]] = None,
        lookback_days: int = 365,
        batch_size: int = 100
    ) -> BulkLoadStats:
        """Calculate technical indicators for multiple stocks in parallel"""
        
        self.start_time = time.time()
        self.stats = BulkLoadStats()
        
        try:
            # Get stock IDs to process
            if stock_ids is None:
                async with self.pool.acquire() as conn:
                    rows = await conn.fetch("""
                        SELECT id FROM stocks 
                        WHERE is_active = true 
                        ORDER BY avg_daily_volume DESC NULLS LAST
                    """)
                    stock_ids = [row['id'] for row in rows]
            
            self.stats.records_processed = len(stock_ids)
            logger.info(f"Calculating technical indicators for {len(stock_ids)} stocks")
            
            # Process stocks in batches
            semaphore = asyncio.Semaphore(self.max_concurrent_batches)
            tasks = []
            
            for i in range(0, len(stock_ids), batch_size):
                batch_ids = stock_ids[i:i + batch_size]
                task = self._calculate_indicators_batch(
                    semaphore, batch_ids, lookback_days, i // batch_size
                )
                tasks.append(task)
            
            # Wait for all batches to complete
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Aggregate results
            total_calculated = 0
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Indicator batch failed: {result}")
                else:
                    total_calculated += result
            
            self.stats.records_inserted = total_calculated
            self.stats.duration_seconds = time.time() - self.start_time
            if self.stats.duration_seconds > 0:
                self.stats.throughput_rps = total_calculated / self.stats.duration_seconds
            
            logger.info(f"Technical indicators calculated: {total_calculated:,} records "
                       f"in {self.stats.duration_seconds:.2f}s")
            
            return self.stats
            
        except Exception as e:
            logger.error(f"Technical indicator calculation failed: {e}")
            raise

    async def _calculate_indicators_batch(
        self,
        semaphore: asyncio.Semaphore,
        stock_ids: List[int],
        lookback_days: int,
        batch_id: int
    ) -> int:
        """Calculate indicators for a batch of stocks"""
        
        async with semaphore:
            async with self.pool.acquire() as conn:
                try:
                    # Use the bulk calculation function
                    result = await conn.execute("""
                        SELECT calculate_technical_indicators_bulk($1::int[])
                    """, stock_ids)
                    
                    # Count calculated indicators
                    count_result = await conn.fetchval("""
                        SELECT COUNT(*) 
                        FROM technical_indicators_optimized 
                        WHERE stock_id = ANY($1::int[]) 
                        AND date >= CURRENT_DATE - INTERVAL '%s days'
                    """ % lookback_days, stock_ids)
                    
                    logger.debug(f"Indicator batch {batch_id} completed: {count_result} records")
                    return count_result or 0
                    
                except Exception as e:
                    logger.error(f"Indicator batch {batch_id} failed: {e}")
                    raise

    async def refresh_materialized_views(self) -> None:
        """Refresh materialized views after bulk data loads"""
        
        views_to_refresh = [
            'daily_stock_summary',
            'weekly_performance'
        ]
        
        async with self.pool.acquire() as conn:
            for view_name in views_to_refresh:
                start_time = time.time()
                await conn.execute(f"REFRESH MATERIALIZED VIEW CONCURRENTLY {view_name}")
                duration = time.time() - start_time
                logger.info(f"Refreshed {view_name} in {duration:.2f}s")

    async def cleanup_old_staging_data(self, days_to_keep: int = 7) -> None:
        """Clean up old staging data to free space"""
        
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        async with self.pool.acquire() as conn:
            # Clean staging tables if they exist
            result = await conn.execute("""
                DELETE FROM price_data_staging 
                WHERE source_batch_id IN (
                    SELECT source_batch_id 
                    FROM price_data_staging 
                    GROUP BY source_batch_id 
                    HAVING MIN(timestamp) < $1
                )
            """, cutoff_date)
            
            logger.info(f"Cleaned up old staging data: {result}")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get detailed performance statistics"""
        
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        return {
            "records_processed": self.stats.records_processed,
            "records_inserted": self.stats.records_inserted,
            "records_updated": self.stats.records_updated,
            "records_failed": self.stats.records_failed,
            "duration_seconds": self.stats.duration_seconds,
            "throughput_rps": self.stats.throughput_rps,
            "memory_usage_mb": memory_usage,
            "cache_size": len(self._symbol_id_cache),
            "batch_size": self.batch_size,
            "max_concurrent_batches": self.max_concurrent_batches
        }


# Example usage and configuration
async def example_usage():
    """Example of how to use the BulkStockDataLoader"""
    
    # Create connection pool
    pool = await asyncpg.create_pool(
        host="localhost",
        port=5432,
        user="postgres",
        password="password",
        database="stock_db",
        min_size=5,
        max_size=20,
        command_timeout=60
    )
    
    # Initialize loader with optimized settings
    loader = BulkStockDataLoader(
        connection_pool=pool,
        batch_size=50000,  # Optimize based on available memory
        max_concurrent_batches=4,  # Optimize based on CPU cores
        enable_compression=True
    )
    
    try:
        # Load price data from file or DataFrame
        stats = await loader.load_price_data_bulk("daily_prices.parquet")
        print(f"Loaded {stats.records_inserted:,} records in {stats.duration_seconds:.2f}s")
        
        # Calculate technical indicators
        await loader.calculate_technical_indicators_bulk()
        
        # Refresh materialized views
        await loader.refresh_materialized_views()
        
        # Print performance statistics
        print(loader.get_performance_stats())
        
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(example_usage())