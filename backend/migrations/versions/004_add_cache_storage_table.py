"""Add cache storage table for L3 caching

Revision ID: 004
Revises: 003
Create Date: 2025-01-19 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '004'
down_revision = '003'
branch_labels = None
depends_on = None


def upgrade():
    """Create cache storage table for L3 database caching"""
    
    # Create cache_storage table
    op.create_table(
        'cache_storage',
        sa.Column('cache_key', sa.String(255), primary_key=True, nullable=False),
        sa.Column('data', postgresql.BYTEA, nullable=False),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.func.now()),
        sa.Column('access_count', sa.Integer, nullable=False, default=0),
        sa.Column('data_size', sa.Integer, nullable=False, default=0),
    )
    
    # Create indexes for performance
    op.create_index('idx_cache_storage_expires_at', 'cache_storage', ['expires_at'])
    op.create_index('idx_cache_storage_created_at', 'cache_storage', ['created_at'])
    op.create_index('idx_cache_storage_access_count', 'cache_storage', ['access_count'])
    
    # Create partial index for non-expired entries (most common query)
    op.execute("""
        CREATE INDEX idx_cache_storage_active 
        ON cache_storage (cache_key) 
        WHERE expires_at IS NULL OR expires_at > NOW()
    """)
    
    # Create function to auto-update updated_at
    op.execute("""
        CREATE OR REPLACE FUNCTION update_cache_storage_updated_at()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            NEW.access_count = OLD.access_count + 1;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)
    
    # Create trigger for updated_at
    op.execute("""
        CREATE TRIGGER trigger_cache_storage_updated_at
        BEFORE UPDATE ON cache_storage
        FOR EACH ROW
        EXECUTE FUNCTION update_cache_storage_updated_at();
    """)
    
    # Create function to calculate data size
    op.execute("""
        CREATE OR REPLACE FUNCTION update_cache_storage_data_size()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.data_size = LENGTH(NEW.data);
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)
    
    # Create trigger for data size calculation
    op.execute("""
        CREATE TRIGGER trigger_cache_storage_data_size
        BEFORE INSERT OR UPDATE ON cache_storage
        FOR EACH ROW
        EXECUTE FUNCTION update_cache_storage_data_size();
    """)


def downgrade():
    """Drop cache storage table and related objects"""
    
    # Drop triggers
    op.execute("DROP TRIGGER IF EXISTS trigger_cache_storage_updated_at ON cache_storage")
    op.execute("DROP TRIGGER IF EXISTS trigger_cache_storage_data_size ON cache_storage")
    
    # Drop functions
    op.execute("DROP FUNCTION IF EXISTS update_cache_storage_updated_at()")
    op.execute("DROP FUNCTION IF EXISTS update_cache_storage_data_size()")
    
    # Drop indexes (automatically dropped with table, but explicit for clarity)
    op.drop_index('idx_cache_storage_active', 'cache_storage')
    op.drop_index('idx_cache_storage_access_count', 'cache_storage')
    op.drop_index('idx_cache_storage_created_at', 'cache_storage')
    op.drop_index('idx_cache_storage_expires_at', 'cache_storage')
    
    # Drop table
    op.drop_table('cache_storage')