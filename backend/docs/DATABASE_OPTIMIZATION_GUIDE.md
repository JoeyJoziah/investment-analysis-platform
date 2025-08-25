# Database Optimization Guide for Massive Stock Data

## Overview

This guide covers the complete database optimization strategy for handling 6000+ stock tickers with daily updates, processing millions of rows efficiently without rate limiting or performance degradation.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Schema Optimizations](#schema-optimizations)
3. [Bulk Loading Strategy](#bulk-loading-strategy)
4. [Compression and Storage](#compression-and-storage)
5. [Query Optimization](#query-optimization)
6. [Maintenance Strategy](#maintenance-strategy)
7. [Configuration Guidelines](#configuration-guidelines)
8. [Performance Monitoring](#performance-monitoring)
9. [Troubleshooting](#troubleshooting)

## Architecture Overview

### TimescaleDB Hypertables

The system uses TimescaleDB for time-series optimization with intelligent partitioning:

- **price_history_optimized**: Weekly chunks for OHLCV data
- **technical_indicators_optimized**: Monthly chunks for indicators
- **news_sentiment_bulk**: Daily chunks for news data

### Data Flow

```
Raw Data → Staging Tables → Bulk Processing → Optimized Tables → Compression → Archive
```

## Schema Optimizations

### 1. Optimized Data Types

```sql
-- Price data uses REAL (4-byte) instead of DECIMAL for better compression
CREATE TABLE price_history_optimized (
    stock_id SMALLINT,          -- 6000 stocks fit in SMALLINT
    date DATE,
    timestamp TIMESTAMP,
    open REAL,                  -- 4-byte float vs 8-byte DECIMAL
    high REAL,
    low REAL,
    close REAL,
    adjusted_close REAL,
    volume INTEGER,             -- 4-byte int sufficient
    price_range REAL,           -- Pre-calculated: high - low
    price_change REAL,          -- Pre-calculated: close - open
    price_change_pct REAL,      -- Pre-calculated percentage
    typical_price REAL,         -- Pre-calculated: (H+L+C)/3
    vwap REAL,                  -- Volume Weighted Average Price
    market_flags SMALLINT       -- Bit flags for market conditions
);
```

### 2. Array Storage for Technical Indicators

```sql
-- Pack multiple related indicators into arrays for better compression
CREATE TABLE technical_indicators_optimized (
    stock_id SMALLINT,
    date DATE,
    sma_values REAL[],          -- [5,10,20,50,100,200]
    ema_values REAL[],          -- [12,26,50]
    rsi_values REAL[],          -- [14,30]
    bollinger_values REAL[],    -- [upper,middle,lower,width]
    macd_values REAL[],         -- [macd,signal,histogram]
    -- Most used indicators as separate columns for fast queries
    rsi_14 REAL,
    sma_20 REAL,
    sma_50 REAL,
    sma_200 REAL
);
```

### 3. Compressed News Storage

```sql
-- Optimize news sentiment with compressed storage
CREATE TABLE news_sentiment_bulk (
    stock_id SMALLINT,
    date DATE,
    source_hash BIGINT,         -- Hash instead of full URL
    headline_vector TSVECTOR,   -- Full-text search optimized
    content_summary VARCHAR(500), -- Truncated content
    sentiment_score SMALLINT,   -- -1000 to 1000 (scaled)
    confidence SMALLINT,        -- 0 to 1000 (scaled)
    impact_score SMALLINT,      -- 0 to 1000 (scaled)
    published_at TIMESTAMP,
    keywords TEXT[]             -- Array for efficient storage
);
```

## Bulk Loading Strategy

### 1. High-Performance Loader

Use the `BulkStockDataLoader` class for maximum throughput:

```python
from backend.utils.bulk_data_loader import BulkStockDataLoader

# Initialize with optimal settings
loader = BulkStockDataLoader(
    connection_pool=pool,
    batch_size=50000,              # Optimize based on RAM
    max_concurrent_batches=4,      # Match CPU cores
    enable_compression=True
)

# Load price data
stats = await loader.load_price_data_bulk(price_dataframe)
print(f"Loaded {stats.records_inserted:,} records in {stats.duration_seconds:.2f}s")
print(f"Throughput: {stats.throughput_rps:.0f} records/second")
```

### 2. Staging Tables for COPY Operations

```sql
-- Use unlogged staging tables for maximum speed
CREATE UNLOGGED TABLE price_data_staging (
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
);
```

### 3. Bulk Insert Process

```python
# Optimal bulk insert workflow
async def bulk_insert_workflow(data_df):
    # 1. Optimize DataFrame
    df = await loader._optimize_dataframe_for_bulk_insert(data_df)
    
    # 2. Use COPY for staging
    await loader._bulk_copy_to_staging(conn, df, temp_table)
    
    # 3. Transfer with conflict resolution
    await loader._transfer_from_staging_to_main(conn, temp_table, main_table)
    
    # 4. Update metadata
    await loader._update_stock_metadata(stock_ids)
```

## Compression and Storage

### 1. TimescaleDB Compression

```sql
-- Ultra-aggressive compression settings
ALTER TABLE price_history_optimized SET (
    timescaledb.compress = true,
    timescaledb.compress_orderby = 'timestamp DESC, stock_id ASC',
    timescaledb.compress_segmentby = 'stock_id',
    timescaledb.compress_chunk_time_interval = '1 day'
);

-- Automatic compression policy
SELECT add_compression_policy('price_history_optimized', INTERVAL '2 hours');
```

### 2. Custom Binary Compression

```sql
-- Function to compress OHLCV into 20 bytes
CREATE OR REPLACE FUNCTION compress_ohlcv(
    open_val REAL, high_val REAL, low_val REAL, close_val REAL, volume_val INTEGER
) RETURNS BYTEA AS $$
DECLARE
    compressed_data BYTEA;
    scale_factor INTEGER := 10000;
BEGIN
    compressed_data := 
        (CAST(open_val * scale_factor AS INTEGER))::BYTEA ||
        (CAST(high_val * scale_factor AS INTEGER))::BYTEA ||
        (CAST(low_val * scale_factor AS INTEGER))::BYTEA ||
        (CAST(close_val * scale_factor AS INTEGER))::BYTEA ||
        volume_val::BYTEA;
    RETURN compressed_data;
END;
$$ LANGUAGE plpgsql IMMUTABLE;
```

### 3. Data Archival

```sql
-- Archive old data with extreme compression
CREATE TABLE price_history_archive (
    stock_id SMALLINT,
    date DATE,
    ohlcv_compressed BYTEA,      -- 20 bytes vs 40+ bytes
    volume_compressed INTEGER,
    monthly_summary JSONB        -- Aggregated monthly data
) PARTITION BY RANGE (date);
```

## Query Optimization

### 1. Intelligent Index Strategy

```sql
-- BRIN indexes for time-series data (minimal size, good performance)
CREATE INDEX CONCURRENTLY idx_price_opt_stock_date 
ON price_history_optimized USING BRIN (stock_id, date);

-- Partial indexes for common conditions
CREATE INDEX CONCURRENTLY idx_price_opt_volume 
ON price_history_optimized (volume) 
WHERE volume > 1000000;

-- Composite indexes for frequent query patterns
CREATE INDEX CONCURRENTLY idx_price_opt_stock_date_close
ON price_history_optimized (stock_id, date, close);

-- GIN indexes for full-text search
CREATE INDEX CONCURRENTLY idx_news_bulk_headlines 
ON news_sentiment_bulk USING GIN (headline_vector);
```

### 2. Materialized Views for Common Queries

```sql
-- Daily aggregated metrics
CREATE MATERIALIZED VIEW daily_stock_summary AS
SELECT 
    stock_id,
    date,
    close,
    volume,
    price_change_pct,
    CASE WHEN prev_close > 0 
         THEN (close - prev_close) / prev_close * 100 
         ELSE 0 END as daily_return,
    volume / NULLIF(avg_volume_20d, 0) as volume_ratio
FROM (
    SELECT 
        stock_id,
        date,
        close,
        volume,
        price_change_pct,
        LAG(close) OVER (PARTITION BY stock_id ORDER BY date) as prev_close,
        AVG(volume) OVER (
            PARTITION BY stock_id ORDER BY date ROWS 20 PRECEDING
        ) as avg_volume_20d
    FROM price_history_optimized
    WHERE date >= CURRENT_DATE - INTERVAL '2 years'
) t
WHERE prev_close IS NOT NULL;
```

### 3. Query Optimizer Usage

```python
from backend.utils.query_optimizer import StockQueryOptimizer

# Analyze performance and get recommendations
optimizer = StockQueryOptimizer(pool)
analysis = await optimizer.analyze_query_performance(hours_lookback=24)

# Create recommended indexes
await optimizer.create_recommended_indexes(
    max_indexes=10,
    simulate_only=False  # Set True to simulate only
)

# Monitor index effectiveness
monitoring = await optimizer.monitor_index_effectiveness()
```

## Maintenance Strategy

### 1. Intelligent Maintenance Scheduling

```python
from backend.utils.database_maintenance import StockDatabaseMaintainer

maintainer = StockDatabaseMaintainer(pool)

# Run comprehensive maintenance
results = await maintainer.run_maintenance_schedule(
    max_concurrent_tasks=2,
    max_duration_hours=4
)

print(f"Reclaimed {results['maintenance_summary']['total_space_reclaimed_mb']:.2f} MB")
```

### 2. Optimized Vacuum Settings

```sql
-- High-frequency vacuum for active tables
ALTER TABLE price_history_optimized SET (
    fillfactor = 95,                          -- Leave 5% free space
    autovacuum_vacuum_scale_factor = 0.01,    -- Vacuum at 1% dead tuples
    autovacuum_analyze_scale_factor = 0.005,  -- Analyze at 0.5% changes
    autovacuum_vacuum_cost_delay = 5,         -- Faster vacuum
    autovacuum_vacuum_cost_limit = 2000       -- Higher I/O limit
);
```

### 3. Automated Maintenance Functions

```sql
-- Daily maintenance routine
CREATE OR REPLACE FUNCTION daily_maintenance() RETURNS VOID AS $$
BEGIN
    -- Compress eligible chunks
    PERFORM compress_chunk(chunk) 
    FROM show_chunks('price_history_optimized') AS chunk
    WHERE range_start < NOW() - INTERVAL '1 day'
    AND NOT is_compressed;
    
    -- Update statistics
    ANALYZE price_history_optimized;
    ANALYZE technical_indicators_optimized;
    
    -- Refresh materialized views
    REFRESH MATERIALIZED VIEW CONCURRENTLY daily_stock_summary;
END;
$$ LANGUAGE plpgsql;
```

## Configuration Guidelines

### 1. PostgreSQL Configuration (postgresql.conf)

```ini
# Memory settings for large datasets
shared_buffers = 4GB                    # 25% of RAM for dedicated server
effective_cache_size = 12GB             # 75% of RAM
work_mem = 256MB                        # For complex queries
maintenance_work_mem = 2GB              # For vacuum, index creation
max_wal_size = 4GB                      # Checkpoint frequency
checkpoint_completion_target = 0.9       # Spread checkpoints

# Parallelism for bulk operations
max_worker_processes = 16
max_parallel_workers = 8
max_parallel_workers_per_gather = 4
max_parallel_maintenance_workers = 4

# Vacuum settings for high-volume inserts
autovacuum_max_workers = 6              # More vacuum processes
autovacuum_naptime = 30s                # Check more frequently
autovacuum_vacuum_cost_delay = 10ms     # Faster vacuum
autovacuum_vacuum_cost_limit = 2000     # Higher I/O limit

# Write-ahead logging
wal_buffers = 64MB
wal_compression = on
wal_level = replica
max_wal_senders = 3

# Connection settings
max_connections = 200
shared_preload_libraries = 'timescaledb,pg_stat_statements'

# Random page cost (SSD optimized)
random_page_cost = 1.1
effective_io_concurrency = 200
```

### 2. TimescaleDB Configuration

```sql
-- Optimize TimescaleDB settings
ALTER SYSTEM SET timescaledb.max_background_workers = 16;
ALTER SYSTEM SET timescaledb.restoring = off;

-- Compression job settings
SELECT alter_job(job_id, config => jsonb_set(config, '{maxruntime}', '3600000'))
FROM timescaledb_information.jobs WHERE proc_name = 'policy_compression';
```

### 3. Operating System Tuning (Linux)

```bash
# Kernel parameters for database performance
echo 'vm.swappiness = 1' >> /etc/sysctl.conf
echo 'vm.dirty_expire_centisecs = 500' >> /etc/sysctl.conf
echo 'vm.dirty_writeback_centisecs = 250' >> /etc/sysctl.conf
echo 'vm.dirty_ratio = 5' >> /etc/sysctl.conf
echo 'vm.dirty_background_ratio = 2' >> /etc/sysctl.conf

# Increase file limits for high connection count
echo '* soft nofile 65536' >> /etc/security/limits.conf
echo '* hard nofile 65536' >> /etc/security/limits.conf

# Optimize I/O scheduler for SSD
echo 'noop' > /sys/block/sda/queue/scheduler  # For SSD
echo 'deadline' > /sys/block/sda/queue/scheduler  # For HDD

# Apply changes
sysctl -p
```

## Performance Monitoring

### 1. Key Metrics to Monitor

```sql
-- Monitor table sizes and compression ratios
SELECT * FROM storage_monitoring ORDER BY total_bytes DESC;

-- Monitor chunk compression status
SELECT * FROM chunk_monitoring WHERE compression_ratio_pct < 50;

-- Monitor query performance
SELECT 
    query,
    calls,
    total_exec_time,
    mean_exec_time,
    rows
FROM pg_stat_statements 
ORDER BY total_exec_time DESC 
LIMIT 10;
```

### 2. Performance Dashboards

Create Grafana dashboards monitoring:

- **Database Size Growth**: Track table and index sizes over time
- **Compression Ratios**: Monitor compression effectiveness
- **Query Performance**: Track slow queries and execution times
- **Insert Throughput**: Monitor bulk loading performance
- **Maintenance Operations**: Track vacuum and analyze performance

### 3. Automated Alerts

Set up alerts for:
- Dead tuple ratio > 20%
- Query execution time > 5 seconds
- Database size growth > 10GB/day
- Failed compression jobs
- Long-running maintenance operations

## Performance Benchmarks

### Expected Throughput

With the optimized schema and bulk loading:

- **Price Data**: 500,000+ records/second
- **Technical Indicators**: 250,000+ records/second  
- **News Sentiment**: 100,000+ records/second

### Storage Efficiency

- **Raw Data**: ~50% compression ratio with TimescaleDB
- **Archive Data**: ~80% compression with custom binary format
- **Index Overhead**: <20% of table size with BRIN indexes

### Query Performance

- **Single stock price lookup**: <1ms
- **Range queries (1 month)**: <10ms
- **Technical indicator queries**: <5ms
- **Cross-table joins**: <50ms
- **Aggregation queries**: <100ms (with materialized views)

## Troubleshooting

### Common Issues and Solutions

#### 1. Slow Bulk Inserts

**Symptoms**: Insert rate drops below 100,000 records/second

**Solutions**:
```sql
-- Check for blocking queries
SELECT * FROM pg_stat_activity WHERE state = 'active';

-- Disable synchronous commit temporarily
SET synchronous_commit = off;

-- Increase checkpoint segments
ALTER SYSTEM SET max_wal_size = '8GB';
```

#### 2. High Dead Tuple Ratio

**Symptoms**: Dead tuple ratio > 20%

**Solutions**:
```sql
-- Immediate vacuum
VACUUM (FULL, ANALYZE) table_name;

-- Adjust autovacuum settings
ALTER TABLE table_name SET (
    autovacuum_vacuum_scale_factor = 0.005
);
```

#### 3. Query Performance Degradation

**Symptoms**: Query times increase significantly

**Solutions**:
```sql
-- Update table statistics
ANALYZE table_name;

-- Check for missing indexes
EXPLAIN (ANALYZE, BUFFERS) your_slow_query;

-- Refresh materialized views
REFRESH MATERIALIZED VIEW CONCURRENTLY view_name;
```

#### 4. Storage Space Issues

**Symptoms**: Disk space growing faster than expected

**Solutions**:
```sql
-- Compress old chunks immediately
SELECT compress_chunk(chunk) FROM show_chunks('table_name');

-- Archive old data
SELECT archive_old_price_data(CURRENT_DATE - INTERVAL '1 year');

-- Clean up unused indexes
DROP INDEX IF EXISTS unused_index_name;
```

## Deployment Checklist

### Before Going Live

- [ ] Run all migrations: 006 (optimization) and 007 (compression)
- [ ] Configure PostgreSQL parameters in postgresql.conf
- [ ] Set up TimescaleDB compression policies  
- [ ] Create materialized views and refresh policies
- [ ] Install and configure monitoring dashboards
- [ ] Set up automated maintenance cron jobs
- [ ] Test bulk loading with sample data
- [ ] Validate query performance benchmarks
- [ ] Configure backup strategy for compressed data
- [ ] Set up alerting for critical metrics

### Post-Deployment

- [ ] Monitor initial bulk load performance
- [ ] Verify compression ratios after first week
- [ ] Check maintenance operation completion
- [ ] Validate query response times
- [ ] Monitor storage growth rates
- [ ] Review and adjust configuration based on actual workload
- [ ] Document any customizations for your specific environment

This optimized database schema and configuration should handle 6000+ daily ticker updates with millions of rows efficiently, maintaining sub-second query performance while minimizing storage footprint through intelligent compression and maintenance strategies.