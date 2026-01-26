"""add_updated_at_to_alerts_table

Revision ID: c849a2ab3b24
Revises: 007
Create Date: 2026-01-25 03:23:34.580229

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'c849a2ab3b24'
down_revision = '007'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add updated_at column to alerts table"""
    # Add updated_at column with default value
    op.add_column(
        'alerts',
        sa.Column(
            'updated_at',
            sa.DateTime(),
            nullable=False,
            server_default=sa.func.now()
        )
    )

    # Create or replace trigger function for auto-updating updated_at
    op.execute("""
        CREATE OR REPLACE FUNCTION update_alerts_updated_at()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)

    # Create trigger
    op.execute("""
        DROP TRIGGER IF EXISTS trigger_alerts_updated_at ON alerts;
        CREATE TRIGGER trigger_alerts_updated_at
        BEFORE UPDATE ON alerts
        FOR EACH ROW
        EXECUTE FUNCTION update_alerts_updated_at();
    """)


def downgrade() -> None:
    """Remove updated_at column from alerts table"""
    # Drop trigger
    op.execute("DROP TRIGGER IF EXISTS trigger_alerts_updated_at ON alerts")

    # Drop trigger function
    op.execute("DROP FUNCTION IF EXISTS update_alerts_updated_at()")

    # Drop column
    op.drop_column('alerts', 'updated_at')