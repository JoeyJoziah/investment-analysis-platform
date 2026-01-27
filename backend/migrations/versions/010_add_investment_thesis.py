"""Add investment thesis table

Revision ID: 010
Revises: 009
Create Date: 2026-01-27

This migration adds the investment_thesis table for storing comprehensive
investment analysis and documentation. Supports:
- User-scoped thesis management
- Stock-specific investment documentation
- Version tracking for thesis updates
- Comprehensive fields for business, financial, and risk analysis
- Full markdown content storage
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text


# revision identifiers
revision = '010'
down_revision = '009'
branch_labels = None
depends_on = None


def upgrade():
    """Create investment_thesis table and indexes"""

    # Check if table exists before creating (idempotent operation)
    connection = op.get_bind()
    inspector = sa.inspect(connection)

    if 'investment_thesis' not in inspector.get_table_names():
        op.create_table(
            'investment_thesis',
            sa.Column('id', sa.Integer(), primary_key=True, index=True),
            sa.Column('user_id', sa.Integer(), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True),
            sa.Column('stock_id', sa.Integer(), sa.ForeignKey('stocks.id', ondelete='CASCADE'), nullable=False, index=True),

            # Core thesis fields
            sa.Column('investment_objective', sa.Text(), nullable=False, comment='Primary investment goal'),
            sa.Column('time_horizon', sa.String(50), nullable=False, comment='Expected holding period'),
            sa.Column('target_price', sa.DECIMAL(10, 2), nullable=True, comment='Target price based on valuation'),

            # Business analysis
            sa.Column('business_model', sa.Text(), nullable=True, comment='How the company makes money'),
            sa.Column('competitive_advantages', sa.Text(), nullable=True, comment='Moats and competitive positioning'),

            # Financial analysis
            sa.Column('financial_health', sa.Text(), nullable=True, comment='Balance sheet and cash flow analysis'),

            # Growth and risk
            sa.Column('growth_drivers', sa.Text(), nullable=True, comment='Key factors driving future growth'),
            sa.Column('risks', sa.Text(), nullable=True, comment='Risk assessment and mitigation'),

            # Valuation
            sa.Column('valuation_rationale', sa.Text(), nullable=True, comment='Valuation methodology'),

            # Exit strategy
            sa.Column('exit_strategy', sa.Text(), nullable=True, comment='Conditions for selling'),

            # Full content
            sa.Column('content', sa.Text(), nullable=True, comment='Complete thesis in markdown'),

            # Versioning
            sa.Column('version', sa.Integer(), default=1, nullable=False, comment='Version number'),

            # Metadata
            sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=False),
            sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
        )

        # Create indexes for performance
        op.create_index(
            'idx_thesis_user_stock',
            'investment_thesis',
            ['user_id', 'stock_id']
        )

        op.create_index(
            'idx_thesis_updated_at',
            'investment_thesis',
            ['updated_at']
        )

        op.create_index(
            'idx_thesis_user_updated',
            'investment_thesis',
            ['user_id', 'updated_at']
        )

        print("✅ Created investment_thesis table with indexes")
    else:
        print("⚠️  investment_thesis table already exists, skipping creation")


def downgrade():
    """Drop investment_thesis table and indexes"""

    # Drop indexes first
    op.execute("""
        DROP INDEX IF EXISTS idx_thesis_user_stock;
    """)
    op.execute("""
        DROP INDEX IF EXISTS idx_thesis_updated_at;
    """)
    op.execute("""
        DROP INDEX IF EXISTS idx_thesis_user_updated;
    """)

    # Drop table
    op.drop_table('investment_thesis')

    print("✅ Dropped investment_thesis table and indexes")
