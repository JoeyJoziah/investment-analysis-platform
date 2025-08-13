"""
Advanced Database Performance Monitoring
Provides comprehensive PostgreSQL and TimescaleDB monitoring with query analysis.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

import asyncpg
import psutil
from prometheus_client import (
    Histogram, Gauge, Counter, Summary, Info
)

from backend.config.settings import settings
from backend.utils.async_database import async_db_manager

logger = logging.getLogger(__name__)

# Database Performance Metrics
db_query_duration_detailed = Histogram(
    'database_query_duration_detailed_seconds',
    'Detailed database query duration',
    ['query_type', 'table', 'index_used', 'result_size']
)

db_slow_queries = Counter(
    'database_slow_queries_total',
    'Total slow queries (>1s)',
    ['query_type', 'table']
)

db_connection_wait_time = Histogram(
    'database_connection_wait_seconds',
    'Time waiting for database connection'
)

db_deadlocks = Counter(
    'database_deadlocks_total',
    'Database deadlocks detected',
    ['table1', 'table2']
)

db_lock_waits = Histogram(
    'database_lock_wait_seconds',
    'Database lock wait time',
    ['lock_type', 'table']
)

# PostgreSQL specific metrics
pg_cache_hit_ratio = Gauge(
    'postgres_cache_hit_ratio',
    'PostgreSQL buffer cache hit ratio',
    ['database']
)

pg_index_usage = Gauge(
    'postgres_index_usage_ratio',
    'PostgreSQL index usage ratio',
    ['table', 'index']
)

pg_table_size = Gauge(
    'postgres_table_size_bytes',
    'PostgreSQL table size in bytes',
    ['database', 'schema', 'table']
)

pg_index_size = Gauge(
    'postgres_index_size_bytes',
    'PostgreSQL index size in bytes',
    ['database', 'schema', 'table', 'index']
)

pg_vacuum_stats = Gauge(
    'postgres_vacuum_last_run_seconds',
    'Seconds since last vacuum',
    ['table', 'vacuum_type']
)

pg_replication_lag = Gauge(
    'postgres_replication_lag_bytes',
    'PostgreSQL replication lag in bytes',
    ['replica']
)

# TimescaleDB specific metrics
hypertable_compression_ratio = Gauge(
    'timescaledb_compression_ratio',
    'TimescaleDB compression ratio',
    ['hypertable']
)

chunk_statistics = Gauge(
    'timescaledb_chunk_count',
    'Number of chunks per hypertable',
    ['hypertable', 'compressed']
)

continuous_aggregate_lag = Gauge(
    'timescaledb_continuous_aggregate_lag_seconds',
    'TimescaleDB continuous aggregate lag',
    ['aggregate_name']
)

# Query Plan Analysis
query_plan_cache_hits = Counter(
    'database_query_plan_cache_hits_total',
    'Query plan cache hits'
)

query_plan_cache_misses = Counter(
    'database_query_plan_cache_misses_total',
    'Query plan cache misses'
)

expensive_operations = Counter(
    'database_expensive_operations_total',
    'Expensive database operations detected',
    ['operation_type', 'cost_range']
)


class DatabasePerformanceMonitor:
    """
    Comprehensive database performance monitoring.
    """
    
    def __init__(self):
        """Initialize database performance monitor."""
        self.monitoring_enabled = True
        self._collection_interval = 30  # seconds
        self._slow_query_threshold = 1.0  # seconds
        self._monitoring_task: Optional[asyncio.Task] = None
        
        # Query analysis cache
        self._query_stats_cache: Dict[str, Dict] = {}
        self._last_stats_collection = datetime.now()
    
    async def start_monitoring(self) -> None:
        """Start database performance monitoring."""
        if not self._monitoring_task:
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            logger.info("Started database performance monitoring")
    
    async def stop_monitoring(self) -> None:
        """Stop database performance monitoring."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            logger.info("Stopped database performance monitoring")
    
    async def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while True:
            try:
                await self.collect_postgresql_stats()
                await self.collect_timescaledb_stats()
                await self.analyze_slow_queries()
                await self.check_index_usage()
                await self.monitor_locks_and_deadlocks()
                await self.collect_vacuum_stats()
                await asyncio.sleep(self._collection_interval)
            except Exception as e:
                logger.error(f"Error in database monitoring loop: {e}")
                await asyncio.sleep(self._collection_interval)
    
    async def collect_postgresql_stats(self) -> None:
        """Collect PostgreSQL performance statistics."""
        try:
            async with async_db_manager.get_session() as db:
                # Cache hit ratio
                cache_hit_query = """
                SELECT 
                    datname,
                    CASE 
                        WHEN (blks_hit + blks_read) > 0 
                        THEN (blks_hit::float / (blks_hit + blks_read)) * 100 
                        ELSE 0 
                    END as cache_hit_ratio
                FROM pg_stat_database 
                WHERE datname = current_database();
                """
                
                result = await db.execute(cache_hit_query)
                for row in result:
                    pg_cache_hit_ratio.labels(database=row['datname']).set(
                        row['cache_hit_ratio']
                    )
                
                # Table and index sizes
                size_query = """
                SELECT 
                    schemaname,
                    tablename,
                    pg_total_relation_size(schemaname||'.'||tablename) as table_size
                FROM pg_tables 
                WHERE schemaname NOT IN ('information_schema', 'pg_catalog');
                """
                
                result = await db.execute(size_query)
                for row in result:
                    pg_table_size.labels(
                        database=settings.DATABASE_NAME,
                        schema=row['schemaname'],
                        table=row['tablename']
                    ).set(row['table_size'])
                
                # Index sizes and usage
                index_query = """
                SELECT 
                    schemaname,
                    tablename,
                    indexname,
                    pg_relation_size(indexrelid) as index_size,
                    idx_scan,
                    idx_tup_read,
                    idx_tup_fetch
                FROM pg_stat_user_indexes 
                JOIN pg_indexes USING (schemaname, tablename, indexname);
                """
                
                result = await db.execute(index_query)
                for row in result:
                    pg_index_size.labels(
                        database=settings.DATABASE_NAME,
                        schema=row['schemaname'],
                        table=row['tablename'],
                        index=row['indexname']
                    ).set(row['index_size'])
                    
                    # Index usage ratio
                    if row['idx_scan'] and row['idx_tup_read']:
                        usage_ratio = row['idx_tup_fetch'] / row['idx_tup_read'] * 100
                        pg_index_usage.labels(
                            table=row['tablename'],
                            index=row['indexname']
                        ).set(usage_ratio)
        
        except Exception as e:
            logger.error(f"Error collecting PostgreSQL stats: {e}")
    
    async def collect_timescaledb_stats(self) -> None:
        """Collect TimescaleDB specific statistics."""
        try:
            async with async_db_manager.get_session() as db:
                # Hypertable compression stats
                compression_query = """
                SELECT 
                    hypertable_name,
                    COALESCE(compression_stats.total_chunks, 0) as total_chunks,
                    COALESCE(compression_stats.number_compressed_chunks, 0) as compressed_chunks,
                    CASE 
                        WHEN compression_stats.uncompressed_heap_size > 0 
                        THEN compression_stats.compressed_heap_size::float / compression_stats.uncompressed_heap_size
                        ELSE 1 
                    END as compression_ratio
                FROM _timescaledb_catalog.hypertable ht
                LEFT JOIN timescaledb_information.compression_settings cs ON ht.table_name = cs.hypertable_name
                LEFT JOIN timescaledb_information.compression_stats ON ht.table_name = compression_stats.hypertable_name;
                """
                
                try:
                    result = await db.execute(compression_query)
                    for row in result:
                        hypertable_compression_ratio.labels(
                            hypertable=row['hypertable_name']
                        ).set(row['compression_ratio'])
                        
                        chunk_statistics.labels(
                            hypertable=row['hypertable_name'],
                            compressed='true'
                        ).set(row['compressed_chunks'])
                        
                        chunk_statistics.labels(
                            hypertable=row['hypertable_name'],
                            compressed='false'
                        ).set(row['total_chunks'] - row['compressed_chunks'])
                
                except Exception as e:
                    logger.debug(f"TimescaleDB stats not available (likely not installed): {e}")
        
        except Exception as e:
            logger.error(f"Error collecting TimescaleDB stats: {e}")
    
    async def analyze_slow_queries(self) -> None:
        """Analyze slow queries and execution plans."""
        try:
            async with async_db_manager.get_session() as db:
                # Get slow queries from pg_stat_statements
                slow_query_sql = """
                SELECT 
                    query,
                    calls,
                    total_exec_time,
                    mean_exec_time,
                    max_exec_time,
                    rows,
                    shared_blks_hit,
                    shared_blks_read,
                    temp_blks_read,
                    temp_blks_written
                FROM pg_stat_statements 
                WHERE mean_exec_time > $1
                ORDER BY mean_exec_time DESC 
                LIMIT 50;
                """
                
                try:
                    result = await db.execute(slow_query_sql, self._slow_query_threshold * 1000)
                    
                    for row in result:
                        # Extract query type and table
                        query_type, table = self._extract_query_info(row['query'])
                        
                        db_slow_queries.labels(
                            query_type=query_type,
                            table=table
                        ).inc(row['calls'])
                        
                        # Track expensive operations
                        if row['mean_exec_time'] > 5000:  # > 5 seconds
                            cost_range = "very_high"
                        elif row['mean_exec_time'] > 1000:  # > 1 second
                            cost_range = "high"
                        else:
                            cost_range = "medium"
                        
                        expensive_operations.labels(
                            operation_type=query_type,
                            cost_range=cost_range
                        ).inc()
                
                except Exception as e:
                    logger.debug(f"pg_stat_statements not available: {e}")
        
        except Exception as e:
            logger.error(f"Error analyzing slow queries: {e}")
    
    async def check_index_usage(self) -> None:
        """Check index usage and identify unused indexes."""
        try:
            async with async_db_manager.get_session() as db:
                unused_index_query = """
                SELECT 
                    schemaname,
                    tablename,
                    indexname,
                    idx_scan,
                    pg_size_pretty(pg_relation_size(indexrelid)) as size
                FROM pg_stat_user_indexes
                WHERE idx_scan < 10  -- Low usage threshold
                AND pg_relation_size(indexrelid) > 1024 * 1024  -- > 1MB
                ORDER BY pg_relation_size(indexrelid) DESC;
                """
                
                result = await db.execute(unused_index_query)
                unused_count = len(result)
                
                if unused_count > 0:
                    logger.warning(f"Found {unused_count} potentially unused indexes")
                    for row in result:
                        logger.info(
                            f"Low usage index: {row['schemaname']}.{row['tablename']}.{row['indexname']} "
                            f"(scans: {row['idx_scan']}, size: {row['size']})"
                        )
        
        except Exception as e:
            logger.error(f"Error checking index usage: {e}")
    
    async def monitor_locks_and_deadlocks(self) -> None:
        """Monitor database locks and deadlocks."""
        try:
            async with async_db_manager.get_session() as db:
                # Check for current locks
                lock_query = """
                SELECT 
                    pl.locktype,
                    pl.mode,
                    pl.granted,
                    pl.relation::regclass as table_name,
                    pa.query,
                    pa.state,
                    extract(epoch from now() - pa.query_start) as duration
                FROM pg_locks pl
                LEFT JOIN pg_stat_activity pa ON pl.pid = pa.pid
                WHERE NOT pl.granted
                AND pl.relation IS NOT NULL;
                """
                
                result = await db.execute(lock_query)
                
                for row in result:
                    if row['duration'] and row['duration'] > 1.0:  # > 1 second
                        db_lock_waits.labels(
                            lock_type=row['mode'],
                            table=str(row['table_name'])
                        ).observe(row['duration'])
        
        except Exception as e:
            logger.error(f"Error monitoring locks: {e}")
    
    async def collect_vacuum_stats(self) -> None:
        """Collect vacuum and maintenance statistics."""
        try:
            async with async_db_manager.get_session() as db:
                vacuum_query = """
                SELECT 
                    schemaname,
                    tablename,
                    last_vacuum,
                    last_autovacuum,
                    last_analyze,
                    last_autoanalyze,
                    vacuum_count,
                    autovacuum_count,
                    analyze_count,
                    autoanalyze_count,
                    n_tup_ins,
                    n_tup_upd,
                    n_tup_del,
                    n_dead_tup
                FROM pg_stat_user_tables;
                """
                
                result = await db.execute(vacuum_query)
                now = datetime.now()
                
                for row in result:
                    table_name = f"{row['schemaname']}.{row['tablename']}"
                    
                    # Time since last vacuum
                    if row['last_vacuum']:
                        vacuum_age = (now - row['last_vacuum']).total_seconds()
                        pg_vacuum_stats.labels(
                            table=table_name,
                            vacuum_type='manual'
                        ).set(vacuum_age)
                    
                    if row['last_autovacuum']:
                        autovacuum_age = (now - row['last_autovacuum']).total_seconds()
                        pg_vacuum_stats.labels(
                            table=table_name,
                            vacuum_type='auto'
                        ).set(autovacuum_age)
                    
                    # Check for tables needing vacuum
                    if row['n_dead_tup'] > 1000 and row['n_tup_ins'] + row['n_tup_upd'] + row['n_tup_del'] > 0:
                        dead_tuple_ratio = row['n_dead_tup'] / (row['n_tup_ins'] + row['n_tup_upd'] + row['n_tup_del'])
                        if dead_tuple_ratio > 0.2:  # > 20% dead tuples
                            logger.warning(f"Table {table_name} may need vacuum: {dead_tuple_ratio:.2%} dead tuples")
        
        except Exception as e:
            logger.error(f"Error collecting vacuum stats: {e}")
    
    def _extract_query_info(self, query: str) -> tuple:
        """Extract query type and main table from SQL query."""
        query_lower = query.lower().strip()
        
        # Determine query type
        if query_lower.startswith('select'):
            query_type = 'SELECT'
        elif query_lower.startswith('insert'):
            query_type = 'INSERT'
        elif query_lower.startswith('update'):
            query_type = 'UPDATE'
        elif query_lower.startswith('delete'):
            query_type = 'DELETE'
        else:
            query_type = 'OTHER'
        
        # Extract table name (simplified)
        table_name = 'unknown'
        try:
            if 'from' in query_lower:
                parts = query_lower.split('from')[1].split()
                if parts:
                    table_name = parts[0].strip('()').split('.')[0]
            elif 'into' in query_lower:
                parts = query_lower.split('into')[1].split()
                if parts:
                    table_name = parts[0].strip('()').split('.')[0]
            elif 'update' in query_lower:
                parts = query_lower.split('update')[1].split()
                if parts:
                    table_name = parts[0].strip('()').split('.')[0]
        except Exception:
            pass
        
        return query_type, table_name
    
    @asynccontextmanager
    async def track_query(self, operation: str, table: str = "unknown"):
        """Context manager to track query performance."""
        start_time = time.time()
        connection_start = time.time()
        
        try:
            # Track connection wait time
            async with async_db_manager.get_session() as db:
                connection_time = time.time() - connection_start
                db_connection_wait_time.observe(connection_time)
                
                yield db
                
                # Track query duration
                duration = time.time() - start_time
                db_query_duration_detailed.labels(
                    query_type=operation,
                    table=table,
                    index_used="unknown",
                    result_size="unknown"
                ).observe(duration)
                
                # Track slow queries
                if duration > self._slow_query_threshold:
                    db_slow_queries.labels(
                        query_type=operation,
                        table=table
                    ).inc()
        
        except Exception as e:
            logger.error(f"Error in query tracking: {e}")
            raise


# Global database performance monitor
db_performance_monitor = DatabasePerformanceMonitor()


# Integration functions
async def setup_database_monitoring():
    """Setup database performance monitoring."""
    await db_performance_monitor.start_monitoring()
    logger.info("Database performance monitoring started")


async def teardown_database_monitoring():
    """Teardown database performance monitoring."""
    await db_performance_monitor.stop_monitoring()
    logger.info("Database performance monitoring stopped")