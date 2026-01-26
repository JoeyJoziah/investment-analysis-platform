"""Add missing database indexes to improve query performance

Revision ID: 008_add_missing_query_indexes
Revises: 007_advanced_compression_optimization
Create Date: 2026-01-26

This migration addresses missing indexes identified through query pattern analysis:

1. Foreign key columns that lack indexes (causing slow JOINs)
2. Columns frequently used in WHERE clauses
3. Columns used in ORDER BY that aren't covered by existing indexes
4. Composite indexes for common multi-column queries
5. Partial indexes for common filter conditions

Expected impact: 50-80% query speedup for common operations.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text


# revision identifiers, used by Alembic.
revision = '008'
down_revision = '007'
branch_labels = None
depends_on = None


def upgrade():
    """Add missing indexes identified through query pattern analysis"""

    # ==========================================================================
    # STOCKS TABLE - Missing indexes for common query patterns
    # ==========================================================================

    # Index for market cap ordering (used in get_top_stocks, sector summaries)
    # The model has idx_stock_market_cap but it's not a composite index for filtering
    op.execute(text("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_stocks_market_cap_desc
        ON stocks (market_cap DESC NULLS LAST)
        WHERE is_active = true AND is_tradable = true;
    """))

    # Index for symbol search with case-insensitive ILIKE pattern
    # Used heavily in search_stocks() and get_by_symbol()
    op.execute(text("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_stocks_symbol_upper
        ON stocks (upper(symbol));
    """))

    # Index for name search (ILIKE patterns)
    op.execute(text("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_stocks_name_trgm
        ON stocks USING gin (name gin_trgm_ops);
    """))

    # Index for exchange_id foreign key (not indexed by default in PostgreSQL)
    op.execute(text("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_stocks_exchange_id
        ON stocks (exchange_id);
    """))

    # Index for industry_id foreign key
    op.execute(text("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_stocks_industry_id
        ON stocks (industry_id)
        WHERE industry_id IS NOT NULL;
    """))

    # Index for sector filter queries
    op.execute(text("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_stocks_sector
        ON stocks (sector)
        WHERE sector IS NOT NULL;
    """))

    # Index for last_price_update queries (data freshness checks)
    op.execute(text("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_stocks_last_price_update
        ON stocks (last_price_update DESC NULLS LAST)
        WHERE is_active = true;
    """))

    # ==========================================================================
    # PRICE_HISTORY TABLE - Additional indexes for time-series queries
    # ==========================================================================

    # Covering index for common price queries (avoids heap fetches)
    op.execute(text("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_price_history_covering
        ON price_history (stock_id, date DESC)
        INCLUDE (open, high, low, close, volume, adjusted_close);
    """))

    # Index for recent data queries (last 60-90 days are most accessed)
    op.execute(text("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_price_history_recent
        ON price_history (stock_id, date DESC)
        WHERE date >= CURRENT_DATE - INTERVAL '90 days';
    """))

    # ==========================================================================
    # RECOMMENDATIONS TABLE - Indexes for repository query patterns
    # ==========================================================================

    # Index for stock_id foreign key (used in JOINs with stocks table)
    op.execute(text("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_recommendations_stock_id
        ON recommendations (stock_id);
    """))

    # Index for valid_until filtering (expire_old_recommendations, active filtering)
    op.execute(text("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_recommendations_valid_until
        ON recommendations (valid_until)
        WHERE is_active = true;
    """))

    # Composite index for get_recommendations_by_type queries
    op.execute(text("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_recommendations_type_active
        ON recommendations (action, is_active, valid_until DESC)
        WHERE is_active = true;
    """))

    # Index for recommendation_id UUID lookups
    op.execute(text("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_recommendations_uuid
        ON recommendations (recommendation_id);
    """))

    # Index for confidence_score filtering and ordering
    op.execute(text("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_recommendations_confidence_desc
        ON recommendations (confidence_score DESC)
        WHERE is_active = true AND valid_until > CURRENT_TIMESTAMP;
    """))

    # ==========================================================================
    # PORTFOLIOS TABLE - Missing foreign key indexes
    # ==========================================================================

    # Index for user_id foreign key (used in get_user_portfolios)
    op.execute(text("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_portfolios_user_id
        ON portfolios (user_id);
    """))

    # Index for portfolio_id UUID lookups
    op.execute(text("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_portfolios_uuid
        ON portfolios (portfolio_id);
    """))

    # Index for is_default lookups (finding default portfolio)
    op.execute(text("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_portfolios_user_default
        ON portfolios (user_id, is_default)
        WHERE is_default = true;
    """))

    # ==========================================================================
    # POSITIONS TABLE - Missing foreign key and query indexes
    # ==========================================================================

    # Index for stock_id foreign key
    op.execute(text("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_positions_stock_id
        ON positions (stock_id);
    """))

    # Composite index for portfolio position queries
    op.execute(text("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_positions_portfolio_stock
        ON positions (portfolio_id, stock_id);
    """))

    # ==========================================================================
    # TRANSACTIONS TABLE - Query pattern indexes
    # ==========================================================================

    # Index for portfolio transaction history
    op.execute(text("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_transactions_portfolio_date
        ON transactions (portfolio_id, trade_date DESC);
    """))

    # Index for stock_id foreign key
    op.execute(text("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_transactions_stock_id
        ON transactions (stock_id);
    """))

    # ==========================================================================
    # ORDERS TABLE - Missing query pattern indexes
    # ==========================================================================

    # Index for user's orders with status filter
    op.execute(text("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orders_user_status_created
        ON orders (user_id, status, created_at DESC);
    """))

    # Index for stock_id foreign key
    op.execute(text("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orders_stock_id
        ON orders (stock_id);
    """))

    # Index for order_id UUID lookups
    op.execute(text("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orders_uuid
        ON orders (order_id);
    """))

    # ==========================================================================
    # WATCHLISTS TABLE - Missing indexes
    # ==========================================================================

    # Index for stock_id foreign key
    op.execute(text("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_watchlists_stock_id
        ON watchlists (stock_id);
    """))

    # Composite index for user watchlist queries
    op.execute(text("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_watchlists_user_stock
        ON watchlists (user_id, stock_id, name);
    """))

    # ==========================================================================
    # ALERTS TABLE - Query pattern indexes
    # ==========================================================================

    # Index for stock_id foreign key
    op.execute(text("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_alerts_stock_id
        ON alerts (stock_id)
        WHERE stock_id IS NOT NULL;
    """))

    # Index for active alerts by type
    op.execute(text("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_alerts_type_active
        ON alerts (alert_type, is_active)
        WHERE is_active = true;
    """))

    # Index for alert_id UUID lookups
    op.execute(text("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_alerts_uuid
        ON alerts (alert_id);
    """))

    # ==========================================================================
    # FUNDAMENTALS TABLE - Additional query indexes
    # ==========================================================================

    # Index for stock_id foreign key with recent data
    op.execute(text("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_fundamentals_stock_recent
        ON fundamentals (stock_id, period_date DESC)
        INCLUDE (pe_ratio, eps, revenue, net_income);
    """))

    # ==========================================================================
    # TECHNICAL_INDICATORS TABLE - Additional query indexes
    # ==========================================================================

    # Covering index for technical analysis queries
    op.execute(text("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_technical_covering
        ON technical_indicators (stock_id, date DESC)
        INCLUDE (rsi_14, macd, macd_signal, sma_20, sma_50, sma_200);
    """))

    # ==========================================================================
    # NEWS_SENTIMENT TABLE - Query pattern indexes
    # ==========================================================================

    # Index for source filtering
    op.execute(text("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_news_sentiment_source
        ON news_sentiment (source, published_at DESC);
    """))

    # Index for sentiment label filtering
    op.execute(text("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_news_sentiment_label
        ON news_sentiment (sentiment_label, published_at DESC)
        WHERE sentiment_label IS NOT NULL;
    """))

    # ==========================================================================
    # ML_PREDICTIONS TABLE - Query pattern indexes
    # ==========================================================================

    # Index for stock predictions by model
    op.execute(text("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ml_predictions_stock_model
        ON ml_predictions (stock_id, model_name, prediction_date DESC);
    """))

    # Index for target date queries
    op.execute(text("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ml_predictions_target_date
        ON ml_predictions (target_date, stock_id);
    """))

    # Index for prediction horizon filtering
    op.execute(text("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ml_predictions_horizon
        ON ml_predictions (prediction_horizon, prediction_date DESC);
    """))

    # ==========================================================================
    # USER_SESSIONS TABLE - Query pattern indexes
    # ==========================================================================

    # Index for session token lookups (authentication)
    op.execute(text("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_sessions_token
        ON user_sessions (session_token)
        WHERE is_active = true;
    """))

    # Index for session expiry checks
    op.execute(text("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_sessions_expires
        ON user_sessions (expires_at)
        WHERE is_active = true;
    """))

    # ==========================================================================
    # API_USAGE TABLE - Cost monitoring indexes
    # ==========================================================================

    # Index for daily cost aggregation
    op.execute(text("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_api_usage_daily_cost
        ON api_usage (DATE(timestamp), provider)
        INCLUDE (calls_count, estimated_cost, success);
    """))

    # ==========================================================================
    # AUDIT_LOGS TABLE - Compliance query indexes
    # ==========================================================================

    # Index for resource lookups (compliance audits)
    op.execute(text("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_logs_resource
        ON audit_logs (resource_type, resource_id, created_at DESC)
        WHERE resource_type IS NOT NULL;
    """))

    # ==========================================================================
    # COST_METRICS TABLE - Budget monitoring indexes
    # ==========================================================================

    # Index for date range queries
    op.execute(text("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_cost_metrics_date_range
        ON cost_metrics (date DESC, provider);
    """))

    # ==========================================================================
    # RECOMMENDATION_PERFORMANCE TABLE - Analytics indexes
    # ==========================================================================

    # Index for recommendation_id foreign key
    op.execute(text("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_recommendation_performance_rec_id
        ON recommendation_performance (recommendation_id);
    """))

    # Index for performance analytics
    op.execute(text("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_recommendation_performance_stats
        ON recommendation_performance (target_hit, stop_loss_hit, actual_return)
        INCLUDE (max_return, max_drawdown);
    """))

    # ==========================================================================
    # Enable trigram extension for fuzzy search (if not exists)
    # ==========================================================================
    op.execute(text("""
        CREATE EXTENSION IF NOT EXISTS pg_trgm;
    """))


def downgrade():
    """Remove the indexes added in this migration"""

    # Stocks table indexes
    op.execute(text("DROP INDEX CONCURRENTLY IF EXISTS idx_stocks_market_cap_desc;"))
    op.execute(text("DROP INDEX CONCURRENTLY IF EXISTS idx_stocks_symbol_upper;"))
    op.execute(text("DROP INDEX CONCURRENTLY IF EXISTS idx_stocks_name_trgm;"))
    op.execute(text("DROP INDEX CONCURRENTLY IF EXISTS idx_stocks_exchange_id;"))
    op.execute(text("DROP INDEX CONCURRENTLY IF EXISTS idx_stocks_industry_id;"))
    op.execute(text("DROP INDEX CONCURRENTLY IF EXISTS idx_stocks_sector;"))
    op.execute(text("DROP INDEX CONCURRENTLY IF EXISTS idx_stocks_last_price_update;"))

    # Price history table indexes
    op.execute(text("DROP INDEX CONCURRENTLY IF EXISTS idx_price_history_covering;"))
    op.execute(text("DROP INDEX CONCURRENTLY IF EXISTS idx_price_history_recent;"))

    # Recommendations table indexes
    op.execute(text("DROP INDEX CONCURRENTLY IF EXISTS idx_recommendations_stock_id;"))
    op.execute(text("DROP INDEX CONCURRENTLY IF EXISTS idx_recommendations_valid_until;"))
    op.execute(text("DROP INDEX CONCURRENTLY IF EXISTS idx_recommendations_type_active;"))
    op.execute(text("DROP INDEX CONCURRENTLY IF EXISTS idx_recommendations_uuid;"))
    op.execute(text("DROP INDEX CONCURRENTLY IF EXISTS idx_recommendations_confidence_desc;"))

    # Portfolios table indexes
    op.execute(text("DROP INDEX CONCURRENTLY IF EXISTS idx_portfolios_user_id;"))
    op.execute(text("DROP INDEX CONCURRENTLY IF EXISTS idx_portfolios_uuid;"))
    op.execute(text("DROP INDEX CONCURRENTLY IF EXISTS idx_portfolios_user_default;"))

    # Positions table indexes
    op.execute(text("DROP INDEX CONCURRENTLY IF EXISTS idx_positions_stock_id;"))
    op.execute(text("DROP INDEX CONCURRENTLY IF EXISTS idx_positions_portfolio_stock;"))

    # Transactions table indexes
    op.execute(text("DROP INDEX CONCURRENTLY IF EXISTS idx_transactions_portfolio_date;"))
    op.execute(text("DROP INDEX CONCURRENTLY IF EXISTS idx_transactions_stock_id;"))

    # Orders table indexes
    op.execute(text("DROP INDEX CONCURRENTLY IF EXISTS idx_orders_user_status_created;"))
    op.execute(text("DROP INDEX CONCURRENTLY IF EXISTS idx_orders_stock_id;"))
    op.execute(text("DROP INDEX CONCURRENTLY IF EXISTS idx_orders_uuid;"))

    # Watchlists table indexes
    op.execute(text("DROP INDEX CONCURRENTLY IF EXISTS idx_watchlists_stock_id;"))
    op.execute(text("DROP INDEX CONCURRENTLY IF EXISTS idx_watchlists_user_stock;"))

    # Alerts table indexes
    op.execute(text("DROP INDEX CONCURRENTLY IF EXISTS idx_alerts_stock_id;"))
    op.execute(text("DROP INDEX CONCURRENTLY IF EXISTS idx_alerts_type_active;"))
    op.execute(text("DROP INDEX CONCURRENTLY IF EXISTS idx_alerts_uuid;"))

    # Fundamentals table indexes
    op.execute(text("DROP INDEX CONCURRENTLY IF EXISTS idx_fundamentals_stock_recent;"))

    # Technical indicators table indexes
    op.execute(text("DROP INDEX CONCURRENTLY IF EXISTS idx_technical_covering;"))

    # News sentiment table indexes
    op.execute(text("DROP INDEX CONCURRENTLY IF EXISTS idx_news_sentiment_source;"))
    op.execute(text("DROP INDEX CONCURRENTLY IF EXISTS idx_news_sentiment_label;"))

    # ML predictions table indexes
    op.execute(text("DROP INDEX CONCURRENTLY IF EXISTS idx_ml_predictions_stock_model;"))
    op.execute(text("DROP INDEX CONCURRENTLY IF EXISTS idx_ml_predictions_target_date;"))
    op.execute(text("DROP INDEX CONCURRENTLY IF EXISTS idx_ml_predictions_horizon;"))

    # User sessions table indexes
    op.execute(text("DROP INDEX CONCURRENTLY IF EXISTS idx_user_sessions_token;"))
    op.execute(text("DROP INDEX CONCURRENTLY IF EXISTS idx_user_sessions_expires;"))

    # API usage table indexes
    op.execute(text("DROP INDEX CONCURRENTLY IF EXISTS idx_api_usage_daily_cost;"))

    # Audit logs table indexes
    op.execute(text("DROP INDEX CONCURRENTLY IF EXISTS idx_audit_logs_resource;"))

    # Cost metrics table indexes
    op.execute(text("DROP INDEX CONCURRENTLY IF EXISTS idx_cost_metrics_date_range;"))

    # Recommendation performance table indexes
    op.execute(text("DROP INDEX CONCURRENTLY IF EXISTS idx_recommendation_performance_rec_id;"))
    op.execute(text("DROP INDEX CONCURRENTLY IF EXISTS idx_recommendation_performance_stats;"))
