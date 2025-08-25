"""Advanced compression and storage optimization for massive data sets

Revision ID: 007
Revises: 006
Create Date: 2025-01-19 14:00:00.000000

This migration implements advanced compression strategies, storage optimization,
and data lifecycle management for massive stock data workloads.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text
from datetime import datetime, timedelta

# revision identifiers
revision = '007'
down_revision = '006'
branch_labels = None
depends_on = None


def upgrade():
    """Apply advanced compression and storage optimizations"""
    
    # ============================================================================
    # STEP 1: Advanced TimescaleDB Compression Settings
    # ============================================================================
    
    # Configure compression for optimal storage and query performance
    op.execute("""
        -- Ultra-aggressive compression for price history
        ALTER TABLE price_history_optimized SET (
            timescaledb.compress = true,
            timescaledb.compress_orderby = 'timestamp DESC, stock_id ASC',
            timescaledb.compress_segmentby = 'stock_id',
            timescaledb.compress_chunk_time_interval = '1 day'
        );
    """)
    
    op.execute("""
        -- Optimize compression for technical indicators
        ALTER TABLE technical_indicators_optimized SET (
            timescaledb.compress = true,
            timescaledb.compress_orderby = 'date DESC, stock_id ASC',
            timescaledb.compress_segmentby = 'stock_id',
            timescaledb.compress_chunk_time_interval = '3 days'
        );
    """)
    
    op.execute("""
        -- News sentiment compression
        ALTER TABLE news_sentiment_bulk SET (
            timescaledb.compress = true,
            timescaledb.compress_orderby = 'published_at DESC, stock_id ASC',
            timescaledb.compress_segmentby = 'stock_id',
            timescaledb.compress_chunk_time_interval = '6 hours'
        );
    """)
    
    # ============================================================================
    # STEP 2: Create Compression Policies with Smart Scheduling
    # ============================================================================
    
    # Aggressive compression for older data
    op.execute("""
        SELECT add_compression_policy(
            'price_history_optimized', 
            INTERVAL '2 hours',
            if_not_exists => true
        );
    """)
    
    op.execute("""
        SELECT add_compression_policy(
            'technical_indicators_optimized', 
            INTERVAL '4 hours',
            if_not_exists => true
        );
    """)
    
    op.execute("""
        SELECT add_compression_policy(
            'news_sentiment_bulk', 
            INTERVAL '30 minutes',
            if_not_exists => true
        );
    """)
    
    # ============================================================================
    # STEP 3: Create Data Retention Policies
    # ============================================================================
    
    # Retain price data for 7 years (regulatory requirement)
    op.execute("""
        SELECT add_retention_policy(
            'price_history_optimized',
            INTERVAL '7 years',
            if_not_exists => true
        );
    """)
    
    # Technical indicators - 3 years
    op.execute("""
        SELECT add_retention_policy(
            'technical_indicators_optimized',
            INTERVAL '3 years',
            if_not_exists => true
        );
    """)
    
    # News sentiment - 2 years
    op.execute("""
        SELECT add_retention_policy(
            'news_sentiment_bulk',
            INTERVAL '2 years',
            if_not_exists => true
        );
    """)
    
    # ============================================================================
    # STEP 4: Create Compressed Archive Tables
    # ============================================================================
    
    # Create archive table for very old price data with extreme compression
    op.execute("""
        CREATE TABLE price_history_archive (
            stock_id SMALLINT,
            date DATE,
            -- Pack OHLCV into a single BYTEA field for maximum compression
            ohlcv_compressed BYTEA,
            volume_compressed INTEGER,
            -- Store monthly aggregated data only
            monthly_summary JSONB
        ) PARTITION BY RANGE (date);
    """)
    
    # Create partitions for archive (yearly partitions)
    current_year = datetime.now().year
    for year in range(2020, current_year + 2):
        op.execute(f"""
            CREATE TABLE price_history_archive_y{year} 
            PARTITION OF price_history_archive 
            FOR VALUES FROM ('{year}-01-01') TO ('{year + 1}-01-01');
        """)
    
    # ============================================================================
    # STEP 5: Create Columnar Storage Tables for Analytics
    # ============================================================================
    
    # Create columnar table for analytical queries (if cstore_fdw available)
    try:
        op.execute("""
            CREATE FOREIGN TABLE price_analytics_columnar (
                stock_id SMALLINT,
                date DATE,
                close REAL,
                volume INTEGER,
                price_change_pct REAL,
                volatility REAL
            ) SERVER cstore_server
            OPTIONS (compression 'pglz');
        """)
    except:
        # Fallback to regular table with optimized storage
        op.execute("""
            CREATE TABLE price_analytics_columnar (
                stock_id SMALLINT,
                date DATE,
                close REAL,
                volume INTEGER,
                price_change_pct REAL,
                volatility REAL
            ) WITH (fillfactor = 95, autovacuum_vacuum_scale_factor = 0.01);
        """)
        
        # Create BRIN indexes for columnar analytics
        op.execute("""
            CREATE INDEX idx_price_analytics_date_brin
            ON price_analytics_columnar USING BRIN (date, stock_id);
        """)
    
    # ============================================================================
    # STEP 6: Implement Custom Compression Functions
    # ============================================================================
    
    # Function to compress OHLCV data into binary format
    op.execute("""
        CREATE OR REPLACE FUNCTION compress_ohlcv(
            open_val REAL,
            high_val REAL,
            low_val REAL,
            close_val REAL,
            volume_val INTEGER
        ) RETURNS BYTEA AS $$
        DECLARE
            compressed_data BYTEA;
            scale_factor INTEGER := 10000; -- Scale to preserve 4 decimal places
        BEGIN
            -- Pack OHLCV into 20 bytes (4 bytes each for OHLC scaled as integers, 4 bytes for volume)
            compressed_data := 
                (CAST(open_val * scale_factor AS INTEGER))::BYTEA ||
                (CAST(high_val * scale_factor AS INTEGER))::BYTEA ||
                (CAST(low_val * scale_factor AS INTEGER))::BYTEA ||
                (CAST(close_val * scale_factor AS INTEGER))::BYTEA ||
                volume_val::BYTEA;
                
            RETURN compressed_data;
        END;
        $$ LANGUAGE plpgsql IMMUTABLE;
    """)
    
    # Function to decompress OHLCV data
    op.execute("""
        CREATE OR REPLACE FUNCTION decompress_ohlcv(compressed_data BYTEA)
        RETURNS TABLE (
            open_val REAL,
            high_val REAL,
            low_val REAL,
            close_val REAL,
            volume_val INTEGER
        ) AS $$
        DECLARE
            scale_factor REAL := 10000.0;
        BEGIN
            RETURN QUERY SELECT
                (get_byte(compressed_data, 0) << 24 | get_byte(compressed_data, 1) << 16 | 
                 get_byte(compressed_data, 2) << 8 | get_byte(compressed_data, 3))::INTEGER / scale_factor,
                (get_byte(compressed_data, 4) << 24 | get_byte(compressed_data, 5) << 16 | 
                 get_byte(compressed_data, 6) << 8 | get_byte(compressed_data, 7))::INTEGER / scale_factor,
                (get_byte(compressed_data, 8) << 24 | get_byte(compressed_data, 9) << 16 | 
                 get_byte(compressed_data, 10) << 8 | get_byte(compressed_data, 11))::INTEGER / scale_factor,
                (get_byte(compressed_data, 12) << 24 | get_byte(compressed_data, 13) << 16 | 
                 get_byte(compressed_data, 14) << 8 | get_byte(compressed_data, 15))::INTEGER / scale_factor,
                (get_byte(compressed_data, 16) << 24 | get_byte(compressed_data, 17) << 16 | 
                 get_byte(compressed_data, 18) << 8 | get_byte(compressed_data, 19))::INTEGER;
        END;
        $$ LANGUAGE plpgsql IMMUTABLE;
    """)
    
    # ============================================================================
    # STEP 7: Create Optimized Storage Parameters
    # ============================================================================
    
    # Optimize storage parameters for high-volume tables
    op.execute("""
        ALTER TABLE price_history_optimized SET (
            fillfactor = 95,  -- Leave 5% free space for updates
            autovacuum_vacuum_scale_factor = 0.01,
            autovacuum_analyze_scale_factor = 0.005,
            autovacuum_vacuum_cost_delay = 5,
            autovacuum_vacuum_cost_limit = 2000,
            toast_tuple_target = 8160  -- Optimize TOAST storage
        );
    """)
    
    op.execute("""
        ALTER TABLE technical_indicators_optimized SET (
            fillfactor = 90,
            autovacuum_vacuum_scale_factor = 0.02,
            autovacuum_analyze_scale_factor = 0.01,
            autovacuum_vacuum_cost_delay = 10,
            autovacuum_vacuum_cost_limit = 1500
        );
    """)
    
    # ============================================================================
    # STEP 8: Create Data Archival Functions
    # ============================================================================
    
    # Function to archive old price data with compression
    op.execute("""
        CREATE OR REPLACE FUNCTION archive_old_price_data(cutoff_date DATE)
        RETURNS INTEGER AS $$
        DECLARE
            archived_count INTEGER := 0;
            batch_size INTEGER := 100000;
            min_date DATE;
            max_date DATE;
        BEGIN
            -- Get date range to archive
            SELECT MIN(date), MAX(date) 
            INTO min_date, max_date
            FROM price_history_optimized 
            WHERE date < cutoff_date;
            
            IF min_date IS NULL THEN
                RETURN 0;
            END IF;
            
            -- Archive data in batches
            WHILE min_date <= max_date LOOP
                -- Insert compressed data into archive
                INSERT INTO price_history_archive (
                    stock_id, date, ohlcv_compressed, volume_compressed
                )
                SELECT 
                    stock_id,
                    date,
                    compress_ohlcv(open, high, low, close, volume),
                    volume
                FROM price_history_optimized
                WHERE date >= min_date AND date < min_date + INTERVAL '1 month'
                LIMIT batch_size;
                
                GET DIAGNOSTICS archived_count = ROW_COUNT;
                
                -- Delete archived data from main table
                DELETE FROM price_history_optimized
                WHERE date >= min_date AND date < min_date + INTERVAL '1 month';
                
                min_date := min_date + INTERVAL '1 month';
                
                -- Commit transaction periodically
                COMMIT;
            END LOOP;
            
            RETURN archived_count;
        END;
        $$ LANGUAGE plpgsql;
    """)
    
    # ============================================================================
    # STEP 9: Create Intelligent Data Tiering
    # ============================================================================
    
    # Create function for automatic data tiering based on access patterns
    op.execute("""
        CREATE OR REPLACE FUNCTION implement_data_tiering()
        RETURNS VOID AS $$
        BEGIN
            -- Move rarely accessed data to compressed storage
            -- This would be called by a scheduled job
            
            -- Compress chunks older than 1 day
            PERFORM compress_chunk(chunk) 
            FROM show_chunks('price_history_optimized') AS chunk
            WHERE range_start < NOW() - INTERVAL '1 day'
            AND NOT is_compressed;
            
            -- Move very old data to archive tables
            PERFORM archive_old_price_data(CURRENT_DATE - INTERVAL '2 years');
            
            -- Update statistics
            ANALYZE price_history_optimized;
            ANALYZE technical_indicators_optimized;
            ANALYZE news_sentiment_bulk;
        END;
        $$ LANGUAGE plpgsql;
    """)
    
    # ============================================================================
    # STEP 10: Create Space Monitoring Views
    # ============================================================================
    
    # View to monitor table sizes and compression ratios
    op.execute("""
        CREATE VIEW storage_monitoring AS
        WITH table_sizes AS (
            SELECT 
                schemaname,
                tablename,
                pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as total_size,
                pg_total_relation_size(schemaname||'.'||tablename) as total_bytes,
                pg_size_pretty(pg_relation_size(schemaname||'.'||tablename)) as table_size,
                pg_relation_size(schemaname||'.'||tablename) as table_bytes
            FROM pg_tables 
            WHERE schemaname = 'public' 
            AND tablename LIKE '%optimized' OR tablename LIKE '%bulk'
        ),
        compression_stats AS (
            SELECT 
                hypertable_name as table_name,
                compression_status,
                uncompressed_heap_size,
                compressed_heap_size,
                CASE 
                    WHEN uncompressed_heap_size > 0 
                    THEN round((1.0 - compressed_heap_size::numeric / uncompressed_heap_size) * 100, 2)
                    ELSE 0 
                END as compression_ratio_pct
            FROM timescaledb_information.compressed_chunk_stats
            GROUP BY hypertable_name, compression_status, uncompressed_heap_size, compressed_heap_size
        )
        SELECT 
            ts.tablename,
            ts.total_size,
            ts.table_size,
            COALESCE(cs.compression_ratio_pct, 0) as compression_ratio_pct,
            CASE 
                WHEN cs.compression_status = 'Compressed' THEN 'Compressed'
                ELSE 'Uncompressed'
            END as status
        FROM table_sizes ts
        LEFT JOIN compression_stats cs ON ts.tablename = cs.table_name
        ORDER BY ts.total_bytes DESC;
    """)
    
    # View to monitor chunk status
    op.execute("""
        CREATE VIEW chunk_monitoring AS
        SELECT 
            hypertable_name,
            chunk_name,
            range_start,
            range_end,
            is_compressed,
            pg_size_pretty(total_bytes) as chunk_size,
            pg_size_pretty(compressed_heap_size) as compressed_size,
            CASE 
                WHEN total_bytes > 0 AND compressed_heap_size > 0
                THEN round((1.0 - compressed_heap_size::numeric / total_bytes) * 100, 2)
                ELSE 0 
            END as compression_ratio_pct
        FROM timescaledb_information.chunks c
        LEFT JOIN timescaledb_information.compressed_chunk_stats ccs 
            ON c.chunk_name = ccs.chunk_name
        WHERE c.hypertable_name IN (
            'price_history_optimized',
            'technical_indicators_optimized', 
            'news_sentiment_bulk'
        )
        ORDER BY c.range_start DESC;
    """)
    
    # ============================================================================
    # STEP 11: Schedule Automatic Maintenance
    # ============================================================================
    
    # Create maintenance scheduler (would be called by cron or pg_cron)
    op.execute("""
        CREATE OR REPLACE FUNCTION daily_maintenance()
        RETURNS VOID AS $$
        BEGIN
            -- Perform data tiering
            PERFORM implement_data_tiering();
            
            -- Refresh continuous aggregates
            CALL refresh_continuous_aggregate('daily_stock_metrics', NULL, NULL);
            CALL refresh_continuous_aggregate('weekly_stock_metrics', NULL, NULL);
            
            -- Update table statistics
            ANALYZE price_history_optimized;
            ANALYZE technical_indicators_optimized;
            ANALYZE news_sentiment_bulk;
            
            -- Log maintenance completion
            INSERT INTO system_metrics (metric_type, metric_name, metric_value)
            VALUES ('maintenance', 'daily_maintenance_completed', EXTRACT(EPOCH FROM NOW()));
            
        END;
        $$ LANGUAGE plpgsql;
    """)


def downgrade():
    """Remove compression optimizations"""
    
    # Drop maintenance functions
    op.execute("DROP FUNCTION IF EXISTS daily_maintenance();")
    op.execute("DROP FUNCTION IF EXISTS implement_data_tiering();")
    op.execute("DROP FUNCTION IF EXISTS archive_old_price_data(DATE);")
    
    # Drop compression functions
    op.execute("DROP FUNCTION IF EXISTS compress_ohlcv(REAL, REAL, REAL, REAL, INTEGER);")
    op.execute("DROP FUNCTION IF EXISTS decompress_ohlcv(BYTEA);")
    
    # Drop monitoring views
    op.execute("DROP VIEW IF EXISTS storage_monitoring;")
    op.execute("DROP VIEW IF EXISTS chunk_monitoring;")
    
    # Drop archive tables
    current_year = datetime.now().year
    for year in range(2020, current_year + 2):
        op.execute(f"DROP TABLE IF EXISTS price_history_archive_y{year};")
    op.execute("DROP TABLE IF EXISTS price_history_archive;")
    
    # Drop columnar table
    op.execute("DROP FOREIGN TABLE IF EXISTS price_analytics_columnar;")
    op.execute("DROP TABLE IF EXISTS price_analytics_columnar;")
    
    # Remove compression policies (TimescaleDB will handle cleanup)
    try:
        op.execute("SELECT remove_compression_policy('price_history_optimized');")
        op.execute("SELECT remove_compression_policy('technical_indicators_optimized');")
        op.execute("SELECT remove_compression_policy('news_sentiment_bulk');")
    except:
        pass
    
    # Remove retention policies
    try:
        op.execute("SELECT remove_retention_policy('price_history_optimized');")
        op.execute("SELECT remove_retention_policy('technical_indicators_optimized');")
        op.execute("SELECT remove_retention_policy('news_sentiment_bulk');")
    except:
        pass