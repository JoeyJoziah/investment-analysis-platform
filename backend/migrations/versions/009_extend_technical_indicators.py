"""Extend technical indicators table with additional columns

Revision ID: 009
Revises: 008
Create Date: 2025-01-26

This migration adds additional indicator columns to support comprehensive
technical analysis including shorter-period SMAs and Bollinger middle band.

Added columns:
- sma_5: 5-period Simple Moving Average
- sma_10: 10-period Simple Moving Average
- bollinger_middle: Middle Bollinger Band (same as SMA20)

These additions support the optimized indicator calculation that uses
PostgreSQL window functions for efficient batch processing of 6000+ stocks.
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers
revision = '009'
down_revision = '008'
branch_labels = None
depends_on = None


def upgrade():
    """Add extended technical indicator columns"""

    # Check if columns exist before adding (idempotent operation)
    connection = op.get_bind()

    # Add columns to technical_indicators table
    existing_columns = get_existing_columns(connection, 'technical_indicators')

    if 'sma_5' not in existing_columns:
        op.add_column('technical_indicators',
            sa.Column('sma_5', sa.DECIMAL(10, 4), nullable=True))

    if 'sma_10' not in existing_columns:
        op.add_column('technical_indicators',
            sa.Column('sma_10', sa.DECIMAL(10, 4), nullable=True))

    if 'bollinger_middle' not in existing_columns:
        op.add_column('technical_indicators',
            sa.Column('bollinger_middle', sa.DECIMAL(10, 4), nullable=True))

    # Also check/add to market_data.technical_indicators if schema exists
    try:
        md_existing_columns = get_existing_columns(connection, 'technical_indicators', 'market_data')

        if 'sma_5' not in md_existing_columns:
            op.execute("""
                ALTER TABLE market_data.technical_indicators
                ADD COLUMN IF NOT EXISTS sma_5 DECIMAL(10, 4);
            """)

        if 'sma_10' not in md_existing_columns:
            op.execute("""
                ALTER TABLE market_data.technical_indicators
                ADD COLUMN IF NOT EXISTS sma_10 DECIMAL(10, 4);
            """)

        if 'bollinger_middle' not in md_existing_columns:
            op.execute("""
                ALTER TABLE market_data.technical_indicators
                ADD COLUMN IF NOT EXISTS bollinger_middle DECIMAL(10, 4);
            """)
    except Exception:
        # market_data schema may not exist in all deployments
        pass

    # Check/add to technical_indicators_optimized if it exists
    try:
        opt_existing_columns = get_existing_columns(connection, 'technical_indicators_optimized')

        if 'sma_5' not in opt_existing_columns:
            op.execute("""
                ALTER TABLE technical_indicators_optimized
                ADD COLUMN IF NOT EXISTS sma_5 REAL;
            """)

        if 'sma_10' not in opt_existing_columns:
            op.execute("""
                ALTER TABLE technical_indicators_optimized
                ADD COLUMN IF NOT EXISTS sma_10 REAL;
            """)

        if 'bollinger_middle' not in opt_existing_columns:
            op.execute("""
                ALTER TABLE technical_indicators_optimized
                ADD COLUMN IF NOT EXISTS bollinger_middle REAL;
            """)
    except Exception:
        # technical_indicators_optimized may not exist in all deployments
        pass

    # Create index on frequently filtered columns if not exists
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_tech_indicators_sma_5_10
        ON technical_indicators (stock_id, sma_5, sma_10)
        WHERE sma_5 IS NOT NULL AND sma_10 IS NOT NULL;
    """)


def get_existing_columns(connection, table_name, schema='public'):
    """Get list of existing columns in a table"""
    result = connection.execute(sa.text(f"""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = :schema AND table_name = :table
    """), {'schema': schema, 'table': table_name})
    return [row[0] for row in result]


def downgrade():
    """Remove extended technical indicator columns"""

    # Remove from public.technical_indicators
    op.drop_column('technical_indicators', 'sma_5', if_exists=True)
    op.drop_column('technical_indicators', 'sma_10', if_exists=True)
    op.drop_column('technical_indicators', 'bollinger_middle', if_exists=True)

    # Remove index
    op.execute("""
        DROP INDEX IF EXISTS idx_tech_indicators_sma_5_10;
    """)

    # Try to remove from market_data schema if exists
    try:
        op.execute("""
            ALTER TABLE market_data.technical_indicators
            DROP COLUMN IF EXISTS sma_5;
        """)
        op.execute("""
            ALTER TABLE market_data.technical_indicators
            DROP COLUMN IF EXISTS sma_10;
        """)
        op.execute("""
            ALTER TABLE market_data.technical_indicators
            DROP COLUMN IF EXISTS bollinger_middle;
        """)
    except Exception:
        pass

    # Try to remove from optimized table if exists
    try:
        op.execute("""
            ALTER TABLE technical_indicators_optimized
            DROP COLUMN IF EXISTS sma_5;
        """)
        op.execute("""
            ALTER TABLE technical_indicators_optimized
            DROP COLUMN IF EXISTS sma_10;
        """)
        op.execute("""
            ALTER TABLE technical_indicators_optimized
            DROP COLUMN IF EXISTS bollinger_middle;
        """)
    except Exception:
        pass
