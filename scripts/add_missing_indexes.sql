-- =============================================================================
-- ADD MISSING DATABASE INDEXES FOR QUERY PERFORMANCE OPTIMIZATION
-- =============================================================================
--
-- This script adds missing indexes identified through query pattern analysis.
-- Expected impact: 50-80% query speedup for common operations.
--
-- Run with: psql -U postgres -d investment_platform -f add_missing_indexes.sql
-- Or within psql: \i add_missing_indexes.sql
--
-- Notes:
-- - Uses CONCURRENTLY to avoid locking tables (requires no transaction)
-- - Uses IF NOT EXISTS to be idempotent (safe to run multiple times)
-- - Partial indexes used where appropriate to reduce index size
-- - Covering indexes used to avoid heap fetches for common queries
-- =============================================================================

-- Enable trigram extension for fuzzy search (required for name search)
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- =============================================================================
-- STOCKS TABLE - Missing indexes for common query patterns
-- =============================================================================

-- Index for market cap ordering (used in get_top_stocks, sector summaries)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_stocks_market_cap_desc
ON stocks (market_cap DESC NULLS LAST)
WHERE is_active = true AND is_tradable = true;

-- Index for symbol search with case-insensitive ILIKE pattern
-- Used heavily in search_stocks() and get_by_symbol()
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_stocks_symbol_upper
ON stocks (upper(symbol));

-- Index for name search (ILIKE patterns) using trigram
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_stocks_name_trgm
ON stocks USING gin (name gin_trgm_ops);

-- Index for exchange_id foreign key (not indexed by default in PostgreSQL)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_stocks_exchange_id
ON stocks (exchange_id);

-- Index for industry_id foreign key
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_stocks_industry_id
ON stocks (industry_id)
WHERE industry_id IS NOT NULL;

-- Index for sector filter queries (tables.py uses sector as string, not FK)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_stocks_sector
ON stocks (sector)
WHERE sector IS NOT NULL;

-- Index for last_price_update queries (data freshness checks)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_stocks_last_price_update
ON stocks (last_price_update DESC NULLS LAST)
WHERE is_active = true;

-- =============================================================================
-- PRICE_HISTORY TABLE - Additional indexes for time-series queries
-- =============================================================================

-- Covering index for common price queries (avoids heap fetches)
-- This is the most impactful index for the N+1 query issues in recommendations.py
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_price_history_covering
ON price_history (stock_id, date DESC)
INCLUDE (open, high, low, close, volume, adjusted_close);

-- Index for recent data queries (last 60-90 days are most accessed)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_price_history_recent
ON price_history (stock_id, date DESC)
WHERE date >= CURRENT_DATE - INTERVAL '90 days';

-- =============================================================================
-- RECOMMENDATIONS TABLE - Indexes for repository query patterns
-- =============================================================================

-- Index for stock_id foreign key (used in JOINs with stocks table)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_recommendations_stock_id
ON recommendations (stock_id);

-- Index for valid_until filtering (expire_old_recommendations, active filtering)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_recommendations_valid_until
ON recommendations (valid_until)
WHERE is_active = true;

-- Composite index for get_recommendations_by_type queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_recommendations_type_active
ON recommendations (action, is_active, valid_until DESC)
WHERE is_active = true;

-- Index for recommendation_id UUID lookups
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_recommendations_uuid
ON recommendations (recommendation_id);

-- Index for confidence_score filtering and ordering
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_recommendations_confidence_desc
ON recommendations (confidence_score DESC)
WHERE is_active = true AND valid_until > CURRENT_TIMESTAMP;

-- =============================================================================
-- PORTFOLIOS TABLE - Missing foreign key indexes
-- =============================================================================

-- Index for user_id foreign key (used in get_user_portfolios)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_portfolios_user_id
ON portfolios (user_id);

-- Index for portfolio_id UUID lookups
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_portfolios_uuid
ON portfolios (portfolio_id);

-- Index for is_default lookups (finding default portfolio)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_portfolios_user_default
ON portfolios (user_id, is_default)
WHERE is_default = true;

-- =============================================================================
-- POSITIONS TABLE - Missing foreign key and query indexes
-- =============================================================================

-- Index for stock_id foreign key
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_positions_stock_id
ON positions (stock_id);

-- Composite index for portfolio position queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_positions_portfolio_stock
ON positions (portfolio_id, stock_id);

-- =============================================================================
-- TRANSACTIONS TABLE - Query pattern indexes
-- =============================================================================

-- Index for portfolio transaction history
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_transactions_portfolio_date
ON transactions (portfolio_id, trade_date DESC);

-- Index for stock_id foreign key
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_transactions_stock_id
ON transactions (stock_id);

-- =============================================================================
-- ORDERS TABLE - Missing query pattern indexes
-- =============================================================================

-- Index for user's orders with status filter
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orders_user_status_created
ON orders (user_id, status, created_at DESC);

-- Index for stock_id foreign key
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orders_stock_id
ON orders (stock_id);

-- Index for order_id UUID lookups
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orders_uuid
ON orders (order_id);

-- =============================================================================
-- WATCHLISTS TABLE - Missing indexes
-- =============================================================================

-- Index for stock_id foreign key
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_watchlists_stock_id
ON watchlists (stock_id);

-- Composite index for user watchlist queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_watchlists_user_stock
ON watchlists (user_id, stock_id, name);

-- =============================================================================
-- ALERTS TABLE - Query pattern indexes
-- =============================================================================

-- Index for stock_id foreign key
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_alerts_stock_id
ON alerts (stock_id)
WHERE stock_id IS NOT NULL;

-- Index for active alerts by type
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_alerts_type_active
ON alerts (alert_type, is_active)
WHERE is_active = true;

-- Index for alert_id UUID lookups
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_alerts_uuid
ON alerts (alert_id);

-- =============================================================================
-- FUNDAMENTALS TABLE - Additional query indexes
-- =============================================================================

-- Index for stock_id foreign key with recent data (covering index)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_fundamentals_stock_recent
ON fundamentals (stock_id, period_date DESC)
INCLUDE (pe_ratio, eps, revenue, net_income);

-- =============================================================================
-- TECHNICAL_INDICATORS TABLE - Additional query indexes
-- =============================================================================

-- Covering index for technical analysis queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_technical_covering
ON technical_indicators (stock_id, date DESC)
INCLUDE (rsi_14, macd, macd_signal, sma_20, sma_50, sma_200);

-- =============================================================================
-- NEWS_SENTIMENT TABLE - Query pattern indexes
-- =============================================================================

-- Index for source filtering
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_news_sentiment_source
ON news_sentiment (source, published_at DESC);

-- Index for sentiment label filtering
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_news_sentiment_label
ON news_sentiment (sentiment_label, published_at DESC)
WHERE sentiment_label IS NOT NULL;

-- =============================================================================
-- ML_PREDICTIONS TABLE - Query pattern indexes
-- =============================================================================

-- Index for stock predictions by model
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ml_predictions_stock_model
ON ml_predictions (stock_id, model_name, prediction_date DESC);

-- Index for target date queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ml_predictions_target_date
ON ml_predictions (target_date, stock_id);

-- Index for prediction horizon filtering
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ml_predictions_horizon
ON ml_predictions (prediction_horizon, prediction_date DESC);

-- =============================================================================
-- USER_SESSIONS TABLE - Query pattern indexes
-- =============================================================================

-- Index for session token lookups (authentication)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_sessions_token
ON user_sessions (session_token)
WHERE is_active = true;

-- Index for session expiry checks
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_sessions_expires
ON user_sessions (expires_at)
WHERE is_active = true;

-- =============================================================================
-- API_USAGE TABLE - Cost monitoring indexes
-- =============================================================================

-- Index for daily cost aggregation
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_api_usage_daily_cost
ON api_usage (DATE(timestamp), provider)
INCLUDE (calls_count, estimated_cost, success);

-- =============================================================================
-- AUDIT_LOGS TABLE - Compliance query indexes
-- =============================================================================

-- Index for resource lookups (compliance audits)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_logs_resource
ON audit_logs (resource_type, resource_id, created_at DESC)
WHERE resource_type IS NOT NULL;

-- =============================================================================
-- COST_METRICS TABLE - Budget monitoring indexes
-- =============================================================================

-- Index for date range queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_cost_metrics_date_range
ON cost_metrics (date DESC, provider);

-- =============================================================================
-- RECOMMENDATION_PERFORMANCE TABLE - Analytics indexes
-- =============================================================================

-- Index for recommendation_id foreign key
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_recommendation_performance_rec_id
ON recommendation_performance (recommendation_id);

-- Index for performance analytics
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_recommendation_performance_stats
ON recommendation_performance (target_hit, stop_loss_hit, actual_return)
INCLUDE (max_return, max_drawdown);

-- =============================================================================
-- ANALYZE TABLES TO UPDATE STATISTICS
-- =============================================================================

ANALYZE stocks;
ANALYZE price_history;
ANALYZE recommendations;
ANALYZE portfolios;
ANALYZE positions;
ANALYZE transactions;
ANALYZE orders;
ANALYZE watchlists;
ANALYZE alerts;
ANALYZE fundamentals;
ANALYZE technical_indicators;
ANALYZE news_sentiment;
ANALYZE ml_predictions;
ANALYZE user_sessions;
ANALYZE api_usage;
ANALYZE audit_logs;
ANALYZE cost_metrics;
ANALYZE recommendation_performance;

-- =============================================================================
-- OUTPUT INDEX SUMMARY
-- =============================================================================

SELECT
    schemaname,
    tablename,
    indexname,
    pg_size_pretty(pg_relation_size(schemaname || '.' || indexname)) as index_size
FROM pg_indexes
WHERE schemaname = 'public'
  AND indexname LIKE 'idx_%'
ORDER BY tablename, indexname;
