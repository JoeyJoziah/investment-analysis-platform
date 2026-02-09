-- Performance Indexes for N+1 Query Optimization
-- Run with: psql -U <user> -d <database> -f backend/migrations/add_performance_indexes.sql

BEGIN;

-- Stock symbol index (should already exist from unified_models)
CREATE INDEX IF NOT EXISTS idx_stocks_symbol ON stocks(symbol);
CREATE INDEX IF NOT EXISTS idx_stocks_is_active_tradable ON stocks(is_active, is_tradable) WHERE is_active = true AND is_tradable = true;

-- Portfolio indexes
CREATE INDEX IF NOT EXISTS idx_portfolios_user_id ON portfolios(user_id);
CREATE INDEX IF NOT EXISTS idx_portfolios_user_id_is_default ON portfolios(user_id, is_default) WHERE is_default = true;

-- Position indexes
CREATE INDEX IF NOT EXISTS idx_positions_portfolio_id_stock_id ON positions(portfolio_id, stock_id);

-- Watchlist indexes
CREATE INDEX IF NOT EXISTS idx_watchlists_user_id_stock_id ON watchlists(user_id, stock_id);

-- Recommendation indexes
CREATE INDEX IF NOT EXISTS idx_recommendations_stock_id_is_active ON recommendations(stock_id, is_active) WHERE is_active = true;
CREATE INDEX IF NOT EXISTS idx_recommendations_is_active_valid_until ON recommendations(is_active, valid_until) WHERE is_active = true;

-- Price history indexes (should already exist from unified_models)
CREATE INDEX IF NOT EXISTS idx_price_history_stock_id_date_desc ON price_history(stock_id, date DESC);

-- Transaction indexes
CREATE INDEX IF NOT EXISTS idx_transactions_portfolio_id_trade_date ON transactions(portfolio_id, trade_date DESC);

-- Order indexes
CREATE INDEX IF NOT EXISTS idx_orders_user_id_status_created ON orders(user_id, status, created_at DESC);

-- Alert indexes
CREATE INDEX IF NOT EXISTS idx_alerts_user_id_is_active_stock_id ON alerts(user_id, is_active, stock_id) WHERE is_active = true;

-- Analyze tables after creating indexes
ANALYZE stocks;
ANALYZE portfolios;
ANALYZE positions;
ANALYZE watchlists;
ANALYZE recommendations;
ANALYZE price_history;
ANALYZE transactions;
ANALYZE orders;
ANALYZE alerts;

COMMIT;

-- Display created indexes
SELECT
    schemaname,
    tablename,
    indexname,
    indexdef
FROM pg_indexes
WHERE schemaname = 'public'
    AND indexname LIKE 'idx_%'
ORDER BY tablename, indexname;
