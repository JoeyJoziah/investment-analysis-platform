-- Database Optimization Script for Investment Analysis Platform
-- Production-ready indexes, partitioning, and performance optimizations
-- Designed for cost-effective operation under $50/month budget

-- =============================================================================
-- INDEXES FOR STOCK DATA PERFORMANCE
-- =============================================================================

-- Primary stock data indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_stock_data_symbol_date 
ON stock_data (symbol, date DESC) 
WHERE date >= CURRENT_DATE - INTERVAL '2 years';

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_stock_data_date_volume 
ON stock_data (date DESC, volume) 
WHERE volume > 0;

-- Partial index for recent high-volume trading
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_stock_data_recent_active
ON stock_data (symbol, date DESC, volume)
WHERE date >= CURRENT_DATE - INTERVAL '90 days' 
  AND volume > 1000000;

-- Composite index for price analysis
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_stock_data_price_analysis
ON stock_data (symbol, date DESC, close, high, low, open)
INCLUDE (volume, adjusted_close);

-- =============================================================================
-- TIMESCALEDB HYPERTABLES AND COMPRESSION
-- =============================================================================

-- Convert stock_data to hypertable for time-series optimization
SELECT create_hypertable('stock_data', 'date', 
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE);

-- Enable compression on older chunks (older than 30 days)
ALTER TABLE stock_data SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol',
    timescaledb.compress_orderby = 'date DESC'
);

-- Add compression policy
SELECT add_compression_policy('stock_data', INTERVAL '30 days');

-- Add data retention policy (keep 3 years of data)
SELECT add_retention_policy('stock_data', INTERVAL '3 years');

-- =============================================================================
-- ANALYSIS RESULTS CACHING TABLES
-- =============================================================================

-- Create optimized table for cached analysis results
CREATE TABLE IF NOT EXISTS analysis_cache (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    analysis_type VARCHAR(50) NOT NULL,
    analysis_date DATE NOT NULL DEFAULT CURRENT_DATE,
    result_data JSONB NOT NULL,
    confidence_score DECIMAL(5,4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    CONSTRAINT unique_analysis_cache UNIQUE (symbol, analysis_type, analysis_date)
);

-- Indexes for analysis cache
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_analysis_cache_lookup
ON analysis_cache (symbol, analysis_type, analysis_date DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_analysis_cache_expiry
ON analysis_cache (expires_at) WHERE expires_at > NOW();

-- GIN index for JSONB queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_analysis_cache_result_gin
ON analysis_cache USING GIN (result_data);

-- =============================================================================
-- RECOMMENDATION SYSTEM OPTIMIZATION
-- =============================================================================

-- Optimized recommendations table with partitioning
CREATE TABLE IF NOT EXISTS recommendations_partitioned (
    id BIGSERIAL,
    symbol VARCHAR(10) NOT NULL,
    recommendation_type VARCHAR(50) NOT NULL,
    action VARCHAR(20) NOT NULL, -- BUY, SELL, HOLD
    confidence_score DECIMAL(5,4) NOT NULL,
    target_price DECIMAL(10,2),
    stop_loss DECIMAL(10,2),
    reasoning TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    valid_until TIMESTAMP WITH TIME ZONE NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    PRIMARY KEY (id, created_at)
) PARTITION BY RANGE (created_at);

-- Create partitions for recommendations (monthly)
DO $$
DECLARE
    start_date DATE := DATE_TRUNC('month', CURRENT_DATE - INTERVAL '12 months');
    end_date DATE := DATE_TRUNC('month', CURRENT_DATE + INTERVAL '12 months');
    partition_date DATE;
BEGIN
    partition_date := start_date;
    WHILE partition_date < end_date LOOP
        EXECUTE format('
            CREATE TABLE IF NOT EXISTS recommendations_%s
            PARTITION OF recommendations_partitioned
            FOR VALUES FROM (%L) TO (%L)',
            TO_CHAR(partition_date, 'YYYY_MM'),
            partition_date,
            partition_date + INTERVAL '1 month'
        );
        partition_date := partition_date + INTERVAL '1 month';
    END LOOP;
END $$;

-- Indexes for recommendations
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_recommendations_active_symbol
ON recommendations_partitioned (symbol, created_at DESC) 
WHERE is_active = TRUE;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_recommendations_confidence
ON recommendations_partitioned (confidence_score DESC, symbol)
WHERE is_active = TRUE AND confidence_score >= 0.7;

-- =============================================================================
-- API RATE LIMITING AND USAGE TRACKING
-- =============================================================================

-- API usage tracking for cost monitoring
CREATE TABLE IF NOT EXISTS api_usage_log (
    id BIGSERIAL PRIMARY KEY,
    provider VARCHAR(50) NOT NULL,
    endpoint VARCHAR(255) NOT NULL,
    symbol VARCHAR(10),
    request_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    response_time_ms INTEGER,
    success BOOLEAN NOT NULL,
    cost_estimate DECIMAL(10,6) DEFAULT 0,
    rate_limit_remaining INTEGER,
    INDEX (provider, request_timestamp),
    INDEX (request_timestamp) WHERE success = FALSE
);

-- Convert to hypertable for time-series data
SELECT create_hypertable('api_usage_log', 'request_timestamp',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE);

-- Add retention policy (keep 90 days of API usage)
SELECT add_retention_policy('api_usage_log', INTERVAL '90 days');

-- =============================================================================
-- MATERIALIZED VIEWS FOR COMMON QUERIES
-- =============================================================================

-- Latest stock prices materialized view
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_latest_stock_prices AS
SELECT DISTINCT ON (symbol) 
    symbol,
    date,
    open,
    high,
    low,
    close,
    volume,
    adjusted_close,
    market_cap,
    pe_ratio,
    dividend_yield
FROM stock_data
WHERE date >= CURRENT_DATE - INTERVAL '30 days'
ORDER BY symbol, date DESC;

-- Unique index for faster refreshes
CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_latest_stock_prices_symbol
ON mv_latest_stock_prices (symbol);

-- Create refresh function
CREATE OR REPLACE FUNCTION refresh_latest_stock_prices()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_latest_stock_prices;
END;
$$ LANGUAGE plpgsql;

-- Top movers materialized view
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_top_movers AS
WITH price_changes AS (
    SELECT 
        symbol,
        close,
        LAG(close) OVER (PARTITION BY symbol ORDER BY date) AS prev_close,
        date,
        volume
    FROM stock_data
    WHERE date >= CURRENT_DATE - INTERVAL '5 days'
)
SELECT 
    symbol,
    close AS current_price,
    prev_close AS previous_price,
    ROUND(((close - prev_close) / prev_close * 100), 2) AS percent_change,
    volume,
    date
FROM price_changes
WHERE prev_close IS NOT NULL
  AND date = (SELECT MAX(date) FROM stock_data)
ORDER BY ABS((close - prev_close) / prev_close) DESC
LIMIT 100;

-- =============================================================================
-- PERFORMANCE MONITORING VIEWS
-- =============================================================================

-- Database performance monitoring
CREATE OR REPLACE VIEW v_database_performance AS
SELECT 
    schemaname,
    tablename,
    attname,
    inherited,
    null_frac,
    avg_width,
    n_distinct,
    most_common_vals,
    most_common_freqs,
    histogram_bounds,
    correlation
FROM pg_stats
WHERE schemaname = 'public'
ORDER BY schemaname, tablename, attname;

-- Slow query monitoring
CREATE OR REPLACE VIEW v_slow_queries AS
SELECT 
    query,
    calls,
    total_time,
    mean_time,
    min_time,
    max_time,
    stddev_time,
    rows
FROM pg_stat_statements
WHERE mean_time > 1000  -- Queries taking more than 1 second on average
ORDER BY mean_time DESC;

-- =============================================================================
-- VACUUM AND MAINTENANCE AUTOMATION
-- =============================================================================

-- Custom vacuum function for cost optimization
CREATE OR REPLACE FUNCTION optimize_tables()
RETURNS void AS $$
DECLARE
    rec RECORD;
BEGIN
    -- VACUUM ANALYZE high-traffic tables
    FOR rec IN 
        SELECT schemaname, tablename 
        FROM pg_stat_user_tables 
        WHERE n_tup_upd + n_tup_del > 1000
    LOOP
        EXECUTE 'VACUUM ANALYZE ' || quote_ident(rec.schemaname) || '.' || quote_ident(rec.tablename);
    END LOOP;
    
    -- Refresh materialized views
    PERFORM refresh_latest_stock_prices();
    REFRESH MATERIALIZED VIEW mv_top_movers;
    
    -- Update table statistics
    ANALYZE;
    
    RAISE NOTICE 'Database optimization completed at %', NOW();
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- CONNECTION POOLING OPTIMIZATION
-- =============================================================================

-- Set optimal connection pool settings
ALTER SYSTEM SET max_connections = 100;
ALTER SYSTEM SET shared_buffers = '512MB';
ALTER SYSTEM SET effective_cache_size = '1536MB';
ALTER SYSTEM SET maintenance_work_mem = '128MB';
ALTER SYSTEM SET work_mem = '4MB';
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.7;
ALTER SYSTEM SET default_statistics_target = 100;

-- Enable parallel query processing for large tables
ALTER SYSTEM SET max_parallel_workers_per_gather = 2;
ALTER SYSTEM SET max_parallel_workers = 4;
ALTER SYSTEM SET max_worker_processes = 8;

-- =============================================================================
-- SECURITY AND MONITORING
-- =============================================================================

-- Create read-only user for monitoring tools
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'monitoring_user') THEN
        CREATE ROLE monitoring_user WITH LOGIN PASSWORD 'monitoring_password_change_this';
    END IF;
END $$;

-- Grant minimal permissions for monitoring
GRANT CONNECT ON DATABASE investment_db TO monitoring_user;
GRANT USAGE ON SCHEMA public TO monitoring_user;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO monitoring_user;
GRANT SELECT ON ALL SEQUENCES IN SCHEMA public TO monitoring_user;

-- Create audit log for sensitive operations
CREATE TABLE IF NOT EXISTS audit_log (
    id BIGSERIAL PRIMARY KEY,
    user_name TEXT NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    table_name VARCHAR(100),
    record_id BIGINT,
    old_values JSONB,
    new_values JSONB,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for audit queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_log_timestamp
ON audit_log (timestamp DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_log_user_event
ON audit_log (user_name, event_type, timestamp DESC);

-- =============================================================================
-- COST MONITORING FUNCTIONS
-- =============================================================================

-- Function to calculate daily database costs
CREATE OR REPLACE FUNCTION calculate_daily_db_cost()
RETURNS TABLE (
    date DATE,
    storage_gb DECIMAL(10,2),
    queries_executed BIGINT,
    estimated_cost_usd DECIMAL(8,2)
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        CURRENT_DATE::DATE as date,
        ROUND(pg_database_size('investment_db') / (1024^3)::DECIMAL, 2) as storage_gb,
        (SELECT sum(calls) FROM pg_stat_statements)::BIGINT as queries_executed,
        ROUND((pg_database_size('investment_db') / (1024^3)::DECIMAL * 0.023), 2) as estimated_cost_usd;
END;
$$ LANGUAGE plpgsql;

COMMENT ON DATABASE investment_db IS 'Investment Analysis Platform - Production optimized for <$50/month operation';