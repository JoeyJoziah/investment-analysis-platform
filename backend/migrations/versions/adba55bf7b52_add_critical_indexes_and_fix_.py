"""add_critical_indexes_and_fix_relationships

Revision ID: adba55bf7b52
Revises: a20ad12e7a8d
Create Date: 2026-02-08 17:37:30.296167

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'adba55bf7b52'
down_revision = 'a20ad12e7a8d'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add indexes to Stock table foreign keys
    op.create_index('idx_stocks_exchange_id', 'stocks', ['exchange_id'])
    op.create_index('idx_stocks_sector_id', 'stocks', ['sector_id'])
    op.create_index('idx_stocks_industry_id', 'stocks', ['industry_id'])

    # Add indexes to Portfolio table
    op.create_index('idx_portfolios_user_id', 'portfolios', ['user_id'])

    # Add indexes to Position table
    op.create_index('idx_positions_portfolio_id', 'positions', ['portfolio_id'])
    op.create_index('idx_positions_stock_id', 'positions', ['stock_id'])

    # Add indexes to Transaction table
    op.create_index('idx_transactions_portfolio_id', 'transactions', ['portfolio_id'])
    op.create_index('idx_transactions_stock_id', 'transactions', ['stock_id'])

    # Add indexes to Watchlist table
    op.create_index('idx_watchlists_user_id', 'watchlists', ['user_id'])
    op.create_index('idx_watchlists_stock_id', 'watchlists', ['stock_id'])

    # Add indexes to Recommendation table
    op.create_index('idx_recommendations_stock_id', 'recommendations', ['stock_id'])
    op.create_index('idx_recommendations_created_at', 'recommendations', ['created_at'])

    # Add indexes to PriceHistory table
    op.create_index('idx_price_history_stock_id', 'price_history', ['stock_id'])
    op.create_index('idx_price_history_date', 'price_history', ['date'])

    # Add indexes to Exchange, Sector, Industry tables
    op.create_index('idx_exchanges_code', 'exchanges', ['code'])
    op.create_index('idx_sectors_name', 'sectors', ['name'])
    op.create_index('idx_industries_name', 'industries', ['name'])
    op.create_index('idx_industries_sector_id', 'industries', ['sector_id'])


def downgrade() -> None:
    # Drop indexes in reverse order
    op.drop_index('idx_industries_sector_id', 'industries')
    op.drop_index('idx_industries_name', 'industries')
    op.drop_index('idx_sectors_name', 'sectors')
    op.drop_index('idx_exchanges_code', 'exchanges')

    op.drop_index('idx_price_history_date', 'price_history')
    op.drop_index('idx_price_history_stock_id', 'price_history')

    op.drop_index('idx_recommendations_created_at', 'recommendations')
    op.drop_index('idx_recommendations_stock_id', 'recommendations')

    op.drop_index('idx_watchlists_stock_id', 'watchlists')
    op.drop_index('idx_watchlists_user_id', 'watchlists')

    op.drop_index('idx_transactions_stock_id', 'transactions')
    op.drop_index('idx_transactions_portfolio_id', 'transactions')

    op.drop_index('idx_positions_stock_id', 'positions')
    op.drop_index('idx_positions_portfolio_id', 'positions')

    op.drop_index('idx_portfolios_user_id', 'portfolios')

    op.drop_index('idx_stocks_industry_id', 'stocks')
    op.drop_index('idx_stocks_sector_id', 'stocks')
    op.drop_index('idx_stocks_exchange_id', 'stocks')