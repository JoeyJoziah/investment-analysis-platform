"""
Add performance indexes for N+1 query optimization

This migration adds critical indexes that were missing:
- stocks.symbol (already exists in unified_models but ensure it's created)
- portfolios.user_id (composite index for filtering and sorting)
- watchlists.user_id, stock_id (composite index for lookups)
- positions.portfolio_id, stock_id (composite index for joins)
- recommendations.stock_id (for filtering and joins)
- price_history.stock_id, date (for historical queries)

Run this migration with:
    alembic upgrade head

Or apply directly with psql:
    psql -U <user> -d <database> -f backend/migrations/add_performance_indexes.sql
"""

# SQL for creating indexes
SQL_CREATE_INDEXES = """
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
"""

SQL_DROP_INDEXES = """
-- Drop indexes (for rollback)
DROP INDEX IF EXISTS idx_stocks_is_active_tradable;
DROP INDEX IF EXISTS idx_portfolios_user_id;
DROP INDEX IF EXISTS idx_portfolios_user_id_is_default;
DROP INDEX IF EXISTS idx_positions_portfolio_id_stock_id;
DROP INDEX IF EXISTS idx_watchlists_user_id_stock_id;
DROP INDEX IF EXISTS idx_recommendations_stock_id_is_active;
DROP INDEX IF EXISTS idx_recommendations_is_active_valid_until;
DROP INDEX IF EXISTS idx_price_history_stock_id_date_desc;
DROP INDEX IF EXISTS idx_transactions_portfolio_id_trade_date;
DROP INDEX IF EXISTS idx_orders_user_id_status_created;
DROP INDEX IF EXISTS idx_alerts_user_id_is_active_stock_id;
"""


def upgrade():
    """Apply performance indexes"""
    import logging
    from sqlalchemy import text
    from backend.config.database import engine

    logger = logging.getLogger(__name__)

    try:
        with engine.connect() as conn:
            # Execute each index creation
            logger.info("Creating performance indexes...")
            conn.execute(text(SQL_CREATE_INDEXES))
            conn.commit()
            logger.info("Performance indexes created successfully")
    except Exception as e:
        logger.error(f"Error creating performance indexes: {e}")
        raise


def downgrade():
    """Remove performance indexes"""
    import logging
    from sqlalchemy import text
    from backend.config.database import engine

    logger = logging.getLogger(__name__)

    try:
        with engine.connect() as conn:
            logger.info("Dropping performance indexes...")
            conn.execute(text(SQL_DROP_INDEXES))
            conn.commit()
            logger.info("Performance indexes dropped successfully")
    except Exception as e:
        logger.error(f"Error dropping performance indexes: {e}")
        raise


if __name__ == "__main__":
    """Run migration directly"""
    import sys
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    if len(sys.argv) > 1 and sys.argv[1] == "down":
        logger.info("Running downgrade...")
        downgrade()
    else:
        logger.info("Running upgrade...")
        upgrade()

    logger.info("Migration complete")
