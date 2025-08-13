-- Database optimization script for Investment Analysis App
-- Implements partitioning, indexing, and performance optimizations
-- Designed for PostgreSQL with TimescaleDB extension

-- =====================================================
-- 1. INSTALL EXTENSIONS
-- =====================================================

-- Enable TimescaleDB for time-series optimization
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Enable pg_stat_statements for query performance monitoring
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Enable btree_gist for advanced indexing
CREATE EXTENSION IF NOT EXISTS btree_gist;

-- =====================================================
-- 2. CONVERT TABLES TO HYPERTABLES (TimescaleDB)
-- =====================================================

-- Convert price_history to hypertable for efficient time-series storage
SELECT create_hypertable(
    'price_history',
    'date',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE,
    migrate_data => TRUE
);

-- Convert technical_indicators to hypertable
SELECT create_hypertable(
    'technical_indicators',
    'date',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE,
    migrate_data => TRUE
);

-- Convert ml_predictions to hypertable
SELECT create_hypertable(
    'ml_predictions',
    'prediction_date',
    chunk_time_interval => INTERVAL '1 week',
    if_not_exists => TRUE,
    migrate_data => TRUE
);

-- =====================================================
-- 3. ADD COMPRESSION POLICIES
-- =====================================================

-- Enable compression on price_history (compress data older than 7 days)
ALTER TABLE price_history SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'stock_id',
    timescaledb.compress_orderby = 'date DESC'
);

SELECT add_compression_policy('price_history', 
    INTERVAL '7 days',
    if_not_exists => TRUE
);

-- Enable compression on technical_indicators
ALTER TABLE technical_indicators SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'stock_id',
    timescaledb.compress_orderby = 'date DESC'
);

SELECT add_compression_policy('technical_indicators', 
    INTERVAL '7 days',
    if_not_exists => TRUE
);

-- =====================================================
-- 4. CREATE OPTIMIZED INDEXES
-- =====================================================

-- Primary lookup indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_stocks_symbol 
    ON stocks(symbol) 
    WHERE is_active = true;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_stocks_market_cap 
    ON stocks(market_cap DESC) 
    WHERE is_active = true;

-- Price history indexes for common queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_price_history_stock_date 
    ON price_history(stock_id, date DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_price_history_date_volume 
    ON price_history(date, volume DESC) 
    WHERE volume > 1000000;

-- Technical indicators composite index
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_technical_composite 
    ON technical_indicators(stock_id, date DESC, rsi, macd);

-- Recommendations index for user queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_recommendations_date_score 
    ON recommendations(recommendation_date DESC, confidence_score DESC) 
    WHERE is_active = true;

-- User portfolio indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_portfolio_user_active 
    ON user_portfolios(user_id, is_active) 
    WHERE is_active = true;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_portfolio_holdings_updated 
    ON portfolio_holdings(portfolio_id, last_updated DESC);

-- =====================================================
-- 5. CREATE MATERIALIZED VIEWS FOR COMMON QUERIES
-- =====================================================

-- Daily stock metrics materialized view
CREATE MATERIALIZED VIEW IF NOT EXISTS daily_stock_metrics AS
SELECT 
    s.id as stock_id,
    s.symbol,
    s.name,
    s.sector,
    s.industry,
    ph.date,
    ph.close as price,
    ph.volume,
    ph.close - LAG(ph.close) OVER (PARTITION BY s.id ORDER BY ph.date) as price_change,
    (ph.close - LAG(ph.close) OVER (PARTITION BY s.id ORDER BY ph.date)) / 
        NULLIF(LAG(ph.close) OVER (PARTITION BY s.id ORDER BY ph.date), 0) * 100 as price_change_pct,
    AVG(ph.close) OVER (PARTITION BY s.id ORDER BY ph.date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as sma_20,
    AVG(ph.close) OVER (PARTITION BY s.id ORDER BY ph.date ROWS BETWEEN 49 PRECEDING AND CURRENT ROW) as sma_50,
    AVG(ph.close) OVER (PARTITION BY s.id ORDER BY ph.date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW) as sma_200,
    AVG(ph.volume) OVER (PARTITION BY s.id ORDER BY ph.date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as avg_volume_20d,
    STDDEV(ph.close) OVER (PARTITION BY s.id ORDER BY ph.date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as volatility_20d
FROM stocks s
JOIN price_history ph ON s.id = ph.stock_id
WHERE s.is_active = true
    AND ph.date >= CURRENT_DATE - INTERVAL '2 years'
WITH DATA;

-- Create index on materialized view
CREATE INDEX idx_daily_metrics_symbol_date 
    ON daily_stock_metrics(symbol, date DESC);

CREATE INDEX idx_daily_metrics_date 
    ON daily_stock_metrics(date DESC);

-- Top movers materialized view (refreshed hourly)
CREATE MATERIALIZED VIEW IF NOT EXISTS top_movers AS
WITH latest_prices AS (
    SELECT DISTINCT ON (stock_id)
        stock_id,
        date,
        close,
        volume
    FROM price_history
    WHERE date >= CURRENT_DATE - INTERVAL '2 days'
    ORDER BY stock_id, date DESC
),
previous_prices AS (
    SELECT DISTINCT ON (stock_id)
        stock_id,
        close as prev_close
    FROM price_history
    WHERE date = (
        SELECT MAX(date) 
        FROM price_history 
        WHERE date < CURRENT_DATE
    )
    ORDER BY stock_id, date DESC
)
SELECT 
    s.symbol,
    s.name,
    s.sector,
    lp.close as current_price,
    pp.prev_close as previous_close,
    (lp.close - pp.prev_close) as price_change,
    (lp.close - pp.prev_close) / NULLIF(pp.prev_close, 0) * 100 as change_pct,
    lp.volume,
    s.market_cap
FROM stocks s
JOIN latest_prices lp ON s.id = lp.stock_id
JOIN previous_prices pp ON s.id = pp.stock_id
WHERE s.is_active = true
ORDER BY ABS((lp.close - pp.prev_close) / NULLIF(pp.prev_close, 0)) DESC
LIMIT 100
WITH DATA;

-- Sector performance materialized view
CREATE MATERIALIZED VIEW IF NOT EXISTS sector_performance AS
SELECT 
    s.sector,
    COUNT(DISTINCT s.id) as stock_count,
    AVG(
        CASE 
            WHEN ph_today.close > 0 AND ph_prev.close > 0 
            THEN (ph_today.close - ph_prev.close) / ph_prev.close * 100
            ELSE 0 
        END
    ) as avg_change_pct,
    SUM(ph_today.volume) as total_volume,
    SUM(s.market_cap) as total_market_cap,
    PERCENTILE_CONT(0.5) WITHIN GROUP (
        ORDER BY (ph_today.close - ph_prev.close) / NULLIF(ph_prev.close, 0) * 100
    ) as median_change_pct
FROM stocks s
JOIN LATERAL (
    SELECT close, volume 
    FROM price_history 
    WHERE stock_id = s.id 
        AND date = CURRENT_DATE
    ORDER BY date DESC 
    LIMIT 1
) ph_today ON true
JOIN LATERAL (
    SELECT close 
    FROM price_history 
    WHERE stock_id = s.id 
        AND date < CURRENT_DATE
    ORDER BY date DESC 
    LIMIT 1
) ph_prev ON true
WHERE s.is_active = true
    AND s.sector IS NOT NULL
GROUP BY s.sector
WITH DATA;

-- =====================================================
-- 6. CREATE CONTINUOUS AGGREGATES (TimescaleDB)
-- =====================================================

-- Hourly OHLCV aggregates for real-time data
CREATE MATERIALIZED VIEW stock_ohlcv_hourly
WITH (timescaledb.continuous) AS
SELECT 
    stock_id,
    time_bucket('1 hour', date) AS hour,
    FIRST(open, date) as open,
    MAX(high) as high,
    MIN(low) as low,
    LAST(close, date) as close,
    SUM(volume) as volume,
    COUNT(*) as tick_count
FROM price_history
GROUP BY stock_id, hour
WITH NO DATA;

-- Add refresh policy for continuous aggregate
SELECT add_continuous_aggregate_policy('stock_ohlcv_hourly',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '30 minutes',
    if_not_exists => TRUE
);

-- Daily aggregates with technical indicators
CREATE MATERIALIZED VIEW stock_daily_aggregates
WITH (timescaledb.continuous) AS
SELECT 
    stock_id,
    time_bucket('1 day', date) AS day,
    AVG(close) as avg_price,
    STDDEV(close) as price_stddev,
    MAX(high) as high,
    MIN(low) as low,
    SUM(volume) as total_volume,
    COUNT(*) as data_points
FROM price_history
GROUP BY stock_id, day
WITH NO DATA;

SELECT add_continuous_aggregate_policy('stock_daily_aggregates',
    start_offset => INTERVAL '2 days',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

-- =====================================================
-- 7. CREATE PARTITIONED TABLES FOR LARGE DATASETS
-- =====================================================

-- Create partitioned table for API call logs
CREATE TABLE IF NOT EXISTS api_call_logs_partitioned (
    id BIGSERIAL,
    api_name VARCHAR(50) NOT NULL,
    endpoint VARCHAR(255) NOT NULL,
    parameters JSONB,
    response_code INTEGER,
    response_time_ms INTEGER,
    cost DECIMAL(10, 6),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id, created_at)
) PARTITION BY RANGE (created_at);

-- Create monthly partitions for API logs
CREATE TABLE IF NOT EXISTS api_call_logs_2025_01 
    PARTITION OF api_call_logs_partitioned 
    FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');

CREATE TABLE IF NOT EXISTS api_call_logs_2025_02 
    PARTITION OF api_call_logs_partitioned 
    FOR VALUES FROM ('2025-02-01') TO ('2025-03-01');

CREATE TABLE IF NOT EXISTS api_call_logs_2025_03 
    PARTITION OF api_call_logs_partitioned 
    FOR VALUES FROM ('2025-03-01') TO ('2025-04-01');

-- Add more partitions as needed...

-- =====================================================
-- 8. OPTIMIZE TABLE SETTINGS
-- =====================================================

-- Optimize autovacuum for high-write tables
ALTER TABLE price_history SET (
    autovacuum_vacuum_scale_factor = 0.01,
    autovacuum_analyze_scale_factor = 0.01,
    autovacuum_vacuum_cost_delay = 10,
    autovacuum_vacuum_cost_limit = 1000
);

ALTER TABLE technical_indicators SET (
    autovacuum_vacuum_scale_factor = 0.02,
    autovacuum_analyze_scale_factor = 0.02
);

-- Set fill factor for tables with frequent updates
ALTER TABLE stocks SET (fillfactor = 90);
ALTER TABLE user_portfolios SET (fillfactor = 85);
ALTER TABLE portfolio_holdings SET (fillfactor = 85);

-- =====================================================
-- 9. CREATE FUNCTIONS FOR COMMON OPERATIONS
-- =====================================================

-- Function to get latest price for a stock
CREATE OR REPLACE FUNCTION get_latest_price(p_symbol VARCHAR)
RETURNS TABLE (
    symbol VARCHAR,
    price DECIMAL,
    change DECIMAL,
    change_pct DECIMAL,
    volume BIGINT,
    last_updated TIMESTAMP
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        s.symbol,
        ph.close as price,
        ph.close - LAG(ph.close) OVER (ORDER BY ph.date) as change,
        (ph.close - LAG(ph.close) OVER (ORDER BY ph.date)) / 
            NULLIF(LAG(ph.close) OVER (ORDER BY ph.date), 0) * 100 as change_pct,
        ph.volume,
        ph.date::timestamp as last_updated
    FROM stocks s
    JOIN price_history ph ON s.id = ph.stock_id
    WHERE s.symbol = p_symbol
    ORDER BY ph.date DESC
    LIMIT 1;
END;
$$ LANGUAGE plpgsql;

-- Function to calculate portfolio performance
CREATE OR REPLACE FUNCTION calculate_portfolio_performance(p_portfolio_id INTEGER)
RETURNS TABLE (
    total_value DECIMAL,
    total_cost DECIMAL,
    total_gain_loss DECIMAL,
    total_gain_loss_pct DECIMAL,
    last_updated TIMESTAMP
) AS $$
BEGIN
    RETURN QUERY
    WITH holdings_value AS (
        SELECT 
            ph.portfolio_id,
            SUM(ph.quantity * lp.close) as current_value,
            SUM(ph.quantity * ph.average_cost) as total_cost
        FROM portfolio_holdings ph
        JOIN LATERAL (
            SELECT close 
            FROM price_history 
            WHERE stock_id = ph.stock_id 
            ORDER BY date DESC 
            LIMIT 1
        ) lp ON true
        WHERE ph.portfolio_id = p_portfolio_id
            AND ph.quantity > 0
        GROUP BY ph.portfolio_id
    )
    SELECT 
        current_value as total_value,
        total_cost,
        current_value - total_cost as total_gain_loss,
        (current_value - total_cost) / NULLIF(total_cost, 0) * 100 as total_gain_loss_pct,
        CURRENT_TIMESTAMP as last_updated
    FROM holdings_value;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- 10. CREATE REFRESH POLICIES
-- =====================================================

-- Refresh materialized views on schedule
CREATE OR REPLACE FUNCTION refresh_materialized_views()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY daily_stock_metrics;
    REFRESH MATERIALIZED VIEW CONCURRENTLY top_movers;
    REFRESH MATERIALIZED VIEW CONCURRENTLY sector_performance;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- 11. PERFORMANCE MONITORING VIEWS
-- =====================================================

-- View for monitoring slow queries
CREATE OR REPLACE VIEW slow_queries AS
SELECT 
    calls,
    total_time,
    mean_time,
    max_time,
    stddev_time,
    query
FROM pg_stat_statements
WHERE mean_time > 100  -- Queries taking more than 100ms on average
ORDER BY mean_time DESC
LIMIT 50;

-- View for monitoring table sizes
CREATE OR REPLACE VIEW table_sizes AS
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS total_size,
    pg_size_pretty(pg_relation_size(schemaname||'.'||tablename)) AS table_size,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename) - 
                   pg_relation_size(schemaname||'.'||tablename)) AS indexes_size
FROM pg_tables
WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- View for monitoring index usage
CREATE OR REPLACE VIEW index_usage AS
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan as index_scans,
    idx_tup_read as tuples_read,
    idx_tup_fetch as tuples_fetched,
    pg_size_pretty(pg_relation_size(indexrelid)) as index_size
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;

-- =====================================================
-- 12. MAINTENANCE SCRIPTS
-- =====================================================

-- Function to perform routine maintenance
CREATE OR REPLACE FUNCTION perform_maintenance()
RETURNS void AS $$
BEGIN
    -- Analyze tables for query planner
    ANALYZE stocks;
    ANALYZE price_history;
    ANALYZE technical_indicators;
    ANALYZE recommendations;
    
    -- Reindex heavily used indexes
    REINDEX INDEX CONCURRENTLY idx_price_history_stock_date;
    REINDEX INDEX CONCURRENTLY idx_stocks_symbol;
    
    -- Clean up old data (keep 2 years of history)
    DELETE FROM price_history 
    WHERE date < CURRENT_DATE - INTERVAL '2 years';
    
    DELETE FROM technical_indicators 
    WHERE date < CURRENT_DATE - INTERVAL '2 years';
    
    -- Vacuum to reclaim space
    VACUUM ANALYZE;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- 13. GRANT PERMISSIONS
-- =====================================================

-- Grant appropriate permissions to application user
-- Replace 'app_user' with your actual application user
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_user WHERE usename = 'app_user') THEN
        GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO app_user;
        GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO app_user;
        GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO app_user;
    END IF;
END
$$;

-- =====================================================
-- 14. FINAL OPTIMIZATIONS
-- =====================================================

-- Update table statistics
ANALYZE;

-- Reset query statistics
SELECT pg_stat_reset();

-- Log completion
DO $$
BEGIN
    RAISE NOTICE 'Database optimization completed successfully at %', NOW();
END
$$;