"""Add adjusted_close column to price_history table

Revision ID: 003_add_adjusted_close
Revises: 002_partitioning
Create Date: 2025-08-12

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import DECIMAL


# revision identifiers, used by Alembic.
revision = '003_add_adjusted_close'
down_revision = '002_partitioning'
branch_labels = None
depends_on = None


def upgrade():
    """Add adjusted_close column to price_history table"""
    
    # Add the adjusted_close column
    op.add_column('price_history', 
                  sa.Column('adjusted_close', DECIMAL(10, 4), nullable=True))
    
    # Create index for adjusted_close queries
    op.create_index(
        'idx_price_history_adjusted_close',
        'price_history',
        ['stock_id', 'date', 'adjusted_close'],
        postgresql_concurrently=True
    )
    
    # Add additional useful price-related columns while we're at it
    op.add_column('price_history', 
                  sa.Column('intraday_volatility', sa.Float, nullable=True))
    
    op.add_column('price_history', 
                  sa.Column('typical_price', DECIMAL(10, 4), nullable=True))
    
    op.add_column('price_history', 
                  sa.Column('vwap', DECIMAL(10, 4), nullable=True))


def downgrade():
    """Remove the added columns"""
    
    # Drop the index first
    op.drop_index('idx_price_history_adjusted_close', table_name='price_history')
    
    # Drop the columns
    op.drop_column('price_history', 'adjusted_close')
    op.drop_column('price_history', 'intraday_volatility')
    op.drop_column('price_history', 'typical_price')
    op.drop_column('price_history', 'vwap')