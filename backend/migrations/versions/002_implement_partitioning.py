"""Implement time-based partitioning for high-volume tables

Revision ID: 002_partitioning
Revises: 001_critical_indexes
Create Date: 2025-08-07

"""
import logging

from alembic import op
import sqlalchemy as sa
from sqlalchemy import text
from datetime import datetime, timedelta

# F-07-016 (PRD audit 2026-04 §3 G4 Phase 2 follow-up): every fallback path
# in this migration previously swallowed exceptions silently
# (`except Exception: pass`). Surface them via a module logger so operators
# can see why a TimescaleDB / hypertable / partitioning step fell through
# to the manual-partitioning fallback.
logger = logging.getLogger("alembic.migration.002_partitioning")


# revision identifiers, used by Alembic.
revision = '002_partitioning'
down_revision = '001_critical_indexes'
branch_labels = None
depends_on = None


def upgrade():
    """Implement time-based partitioning"""
    
    # Enable TimescaleDB extension if not already enabled
    try:
        op.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")
    except Exception as exc:
        # Fallback to manual partitioning if TimescaleDB is not available.
        # Log the exception so operators can distinguish "TimescaleDB not
        # installed" from a permissions / connection issue.
        logger.warning("TimescaleDB extension unavailable, falling back to manual partitioning: %s", exc)
    
    # Convert price_history to hypertable (TimescaleDB) or implement manual partitioning
    try:
        # Try TimescaleDB approach first
        op.execute("""
            SELECT create_hypertable(
                'price_history', 
                'date',
                chunk_time_interval => INTERVAL '1 month',
                if_not_exists => TRUE
            );
        """)
        
        # Set compression policy
        op.execute("""
            ALTER TABLE price_history SET (
                timescaledb.compress,
                timescaledb.compress_orderby = 'date DESC, stock_id',
                timescaledb.compress_segmentby = 'stock_id'
            );
        """)
        
        # Add compression policy - compress chunks older than 7 days
        op.execute("""
            SELECT add_compression_policy('price_history', INTERVAL '7 days');
        """)
        
    except Exception as exc:
        # Fallback to manual partitioning
        logger.warning("create_hypertable(price_history) failed, falling back to manual partitioning: %s", exc)
        implement_manual_partitioning_price_history()
    
    # Convert technical_indicators to hypertable
    try:
        op.execute("""
            SELECT create_hypertable(
                'technical_indicators', 
                'date',
                chunk_time_interval => INTERVAL '1 month',
                if_not_exists => TRUE
            );
        """)
        
        # Set compression policy for technical indicators
        op.execute("""
            ALTER TABLE technical_indicators SET (
                timescaledb.compress,
                timescaledb.compress_orderby = 'date DESC, stock_id',
                timescaledb.compress_segmentby = 'stock_id'
            );
        """)
        
        op.execute("""
            SELECT add_compression_policy('technical_indicators', INTERVAL '7 days');
        """)
        
    except Exception as exc:
        logger.warning("create_hypertable(technical_indicators) failed, falling back to manual partitioning: %s", exc)
        implement_manual_partitioning_technical_indicators()
    
    # Create continuous aggregates for common queries
    try:
        # Daily stock metrics aggregate
        op.execute("""
            CREATE MATERIALIZED VIEW daily_stock_metrics
            WITH (timescaledb.continuous) AS
            SELECT 
                stock_id,
                time_bucket(INTERVAL '1 day', date) AS day,
                FIRST(open, date) as open,
                MAX(high) as high,
                MIN(low) as low,
                LAST(close, date) as close,
                SUM(volume) as volume,
                AVG(close) as avg_price,
                STDDEV(close) as price_volatility
            FROM price_history
            GROUP BY stock_id, day;
        """)
        
        # Weekly aggregates
        op.execute("""
            CREATE MATERIALIZED VIEW weekly_stock_metrics
            WITH (timescaledb.continuous) AS
            SELECT 
                stock_id,
                time_bucket(INTERVAL '1 week', date) AS week,
                FIRST(open, date) as open,
                MAX(high) as high,
                MIN(low) as low,
                LAST(close, date) as close,
                SUM(volume) as volume,
                AVG(close) as avg_price,
                STDDEV(close) as price_volatility
            FROM price_history
            GROUP BY stock_id, week;
        """)
        
        # Add refresh policies
        op.execute("""
            SELECT add_continuous_aggregate_policy('daily_stock_metrics',
                start_offset => INTERVAL '3 days',
                end_offset => INTERVAL '1 hour',
                schedule_interval => INTERVAL '1 hour');
        """)
        
        op.execute("""
            SELECT add_continuous_aggregate_policy('weekly_stock_metrics',
                start_offset => INTERVAL '1 week',
                end_offset => INTERVAL '1 day',
                schedule_interval => INTERVAL '1 day');
        """)
        
    except Exception as exc:
        # Create regular materialized views if TimescaleDB is not available
        logger.warning("TimescaleDB continuous aggregates failed, falling back to regular materialized views: %s", exc)
        create_regular_materialized_views()
    
    # Create retention policy - keep raw data for 5 years
    try:
        op.execute("""
            SELECT add_retention_policy('price_history', INTERVAL '5 years');
        """)
        
        op.execute("""
            SELECT add_retention_policy('technical_indicators', INTERVAL '2 years');
        """)
    except Exception as exc:
        logger.warning("add_retention_policy() failed (TimescaleDB feature unavailable?): %s", exc)
    
    # Optimize autovacuum for high-write tables
    op.execute("""
        ALTER TABLE price_history SET (
            autovacuum_vacuum_scale_factor = 0.02,
            autovacuum_analyze_scale_factor = 0.01,
            autovacuum_vacuum_cost_delay = 10,
            autovacuum_vacuum_cost_limit = 1000
        );
    """)
    
    op.execute("""
        ALTER TABLE technical_indicators SET (
            autovacuum_vacuum_scale_factor = 0.02,
            autovacuum_analyze_scale_factor = 0.01,
            autovacuum_vacuum_cost_delay = 10,
            autovacuum_vacuum_cost_limit = 1000
        );
    """)


def implement_manual_partitioning_price_history():
    """Implement manual partitioning for price_history table"""
    
    # Create parent table (if not exists)
    current_date = datetime.now()
    
    # Create partitions for current month and next 3 months
    for i in range(4):
        partition_date = current_date + timedelta(days=30*i)
        partition_name = f"price_history_y{partition_date.year}m{partition_date.month:02d}"
        start_date = partition_date.replace(day=1)
        
        if partition_date.month == 12:
            end_date = partition_date.replace(year=partition_date.year+1, month=1, day=1)
        else:
            end_date = partition_date.replace(month=partition_date.month+1, day=1)
        
        op.execute(f"""
            CREATE TABLE IF NOT EXISTS {partition_name} 
            PARTITION OF price_history 
            FOR VALUES FROM ('{start_date.strftime('%Y-%m-%d')}') 
                      TO ('{end_date.strftime('%Y-%m-%d')}');
        """)


def implement_manual_partitioning_technical_indicators():
    """Implement manual partitioning for technical_indicators table"""
    
    current_date = datetime.now()
    
    # Create partitions for current month and next 3 months
    for i in range(4):
        partition_date = current_date + timedelta(days=30*i)
        partition_name = f"technical_indicators_y{partition_date.year}m{partition_date.month:02d}"
        start_date = partition_date.replace(day=1)
        
        if partition_date.month == 12:
            end_date = partition_date.replace(year=partition_date.year+1, month=1, day=1)
        else:
            end_date = partition_date.replace(month=partition_date.month+1, day=1)
        
        op.execute(f"""
            CREATE TABLE IF NOT EXISTS {partition_name} 
            PARTITION OF technical_indicators 
            FOR VALUES FROM ('{start_date.strftime('%Y-%m-%d')}') 
                      TO ('{end_date.strftime('%Y-%m-%d')}');
        """)


def create_regular_materialized_views():
    """Create regular materialized views as fallback"""
    
    op.execute("""
        CREATE MATERIALIZED VIEW daily_stock_metrics AS
        SELECT 
            stock_id,
            DATE(date) AS day,
            (array_agg(open ORDER BY date ASC))[1] as open,
            MAX(high) as high,
            MIN(low) as low,
            (array_agg(close ORDER BY date DESC))[1] as close,
            SUM(volume) as volume,
            AVG(close) as avg_price,
            STDDEV(close) as price_volatility,
            COUNT(*) as data_points
        FROM price_history
        GROUP BY stock_id, DATE(date);
    """)
    
    op.execute("""
        CREATE UNIQUE INDEX ON daily_stock_metrics (stock_id, day);
    """)
    
    op.execute("""
        CREATE MATERIALIZED VIEW weekly_stock_metrics AS
        SELECT 
            stock_id,
            DATE_TRUNC('week', date)::date AS week,
            (array_agg(open ORDER BY date ASC))[1] as open,
            MAX(high) as high,
            MIN(low) as low,
            (array_agg(close ORDER BY date DESC))[1] as close,
            SUM(volume) as volume,
            AVG(close) as avg_price,
            STDDEV(close) as price_volatility,
            COUNT(*) as data_points
        FROM price_history
        GROUP BY stock_id, DATE_TRUNC('week', date);
    """)
    
    op.execute("""
        CREATE UNIQUE INDEX ON weekly_stock_metrics (stock_id, week);
    """)


def downgrade():
    """Remove partitioning and aggregates"""
    
    # Drop materialized views
    try:
        op.execute("DROP MATERIALIZED VIEW IF EXISTS daily_stock_metrics CASCADE;")
        op.execute("DROP MATERIALIZED VIEW IF EXISTS weekly_stock_metrics CASCADE;")
    except Exception as exc:
        logger.warning("DROP MATERIALIZED VIEW failed during downgrade (views may not exist): %s", exc)
    
    # Remove TimescaleDB features if they exist
    try:
        op.execute("SELECT remove_compression_policy('price_history');")
        op.execute("SELECT remove_compression_policy('technical_indicators');")
        op.execute("SELECT remove_retention_policy('price_history');")
        op.execute("SELECT remove_retention_policy('technical_indicators');")
    except Exception as exc:
        logger.warning("remove_compression_policy/remove_retention_policy failed during downgrade (TimescaleDB may be absent): %s", exc)
    
    # Reset autovacuum settings to defaults
    op.execute("""
        ALTER TABLE price_history RESET (
            autovacuum_vacuum_scale_factor,
            autovacuum_analyze_scale_factor,
            autovacuum_vacuum_cost_delay,
            autovacuum_vacuum_cost_limit
        );
    """)
    
    op.execute("""
        ALTER TABLE technical_indicators RESET (
            autovacuum_vacuum_scale_factor,
            autovacuum_analyze_scale_factor,
            autovacuum_vacuum_cost_delay,
            autovacuum_vacuum_cost_limit
        );
    """)