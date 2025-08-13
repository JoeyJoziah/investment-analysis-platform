"""
TimescaleDB initialization and optimization for the investment analysis database.
Configures hypertables, compression, and continuous aggregates for optimal performance.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, Connection
from sqlalchemy.exc import SQLAlchemyError
import os

logger = logging.getLogger(__name__)


class TimescaleDBInitializer:
    """Initialize and configure TimescaleDB for time-series data optimization."""
    
    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize TimescaleDB manager.
        
        Args:
            database_url: PostgreSQL connection string
        """
        self.database_url = database_url or os.getenv(
            'DATABASE_URL',
            'postgresql://postgres:password@localhost:5432/investment_db'
        )
        self.engine = create_engine(self.database_url)
    
    def initialize_timescaledb(self) -> bool:
        """
        Initialize TimescaleDB extension and configure hypertables.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with self.engine.begin() as conn:  # Use transaction context
                # Enable TimescaleDB extension
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE"))
                
                logger.info("TimescaleDB extension enabled successfully")
                
                # Configure hypertables
                self._create_hypertables(conn)
                
                # Set up compression policies
                self._setup_compression_policies(conn)
                
                # Create continuous aggregates
                self._create_continuous_aggregates(conn)
                
                # Configure retention policies
                self._setup_retention_policies(conn)
                
                # Create optimized indexes
                self._create_optimized_indexes(conn)
                
                # Transaction will automatically commit if no errors
                return True
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to initialize TimescaleDB: {e}")
            # Transaction automatically rolls back on exception
            return False
    
    def _create_hypertables(self, conn):
        """Create hypertables for time-series data."""
        
        # Price History Hypertable
        try:
            conn.execute(text("""
                SELECT create_hypertable(
                    'price_history',
                    'date',
                    chunk_time_interval => INTERVAL '1 month',
                    if_not_exists => TRUE
                )
            """))
            logger.info("Created hypertable for price_history")
        except Exception as e:
            logger.warning(f"price_history hypertable may already exist: {e}")
        
        # Technical Indicators Hypertable
        try:
            conn.execute(text("""
                SELECT create_hypertable(
                    'technical_indicators',
                    'date',
                    chunk_time_interval => INTERVAL '1 month',
                    if_not_exists => TRUE
                )
            """))
            logger.info("Created hypertable for technical_indicators")
        except Exception as e:
            logger.warning(f"technical_indicators hypertable may already exist: {e}")
        
        # Market Metrics Hypertable (for market-wide metrics)
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS market_metrics (
                time TIMESTAMPTZ NOT NULL,
                metric_name VARCHAR(100) NOT NULL,
                metric_value FLOAT,
                metadata JSONB,
                PRIMARY KEY (time, metric_name)
            )
        """))
        
        try:
            conn.execute(text("""
                SELECT create_hypertable(
                    'market_metrics',
                    'time',
                    chunk_time_interval => INTERVAL '1 week',
                    if_not_exists => TRUE
                )
            """))
            logger.info("Created hypertable for market_metrics")
        except Exception as e:
            logger.warning(f"market_metrics hypertable may already exist: {e}")
        
        # Removed conn.commit() - transaction is managed by context manager
    
    def _setup_compression_policies(self, conn):
        """
        Set up compression policies for hypertables.
        Compression can reduce storage by 90%+ for time-series data.
        """
        
        # Enable compression on price_history
        conn.execute(text("""
            ALTER TABLE price_history SET (
                timescaledb.compress,
                timescaledb.compress_segmentby = 'stock_id',
                timescaledb.compress_orderby = 'date DESC'
            )
        """))
        
        # Add compression policy - compress data older than 7 days
        try:
            conn.execute(text("""
                SELECT add_compression_policy(
                    'price_history',
                    INTERVAL '7 days',
                    if_not_exists => TRUE
                )
            """))
            logger.info("Added compression policy for price_history")
        except Exception as e:
            logger.warning(f"Compression policy may already exist: {e}")
        
        # Enable compression on technical_indicators
        conn.execute(text("""
            ALTER TABLE technical_indicators SET (
                timescaledb.compress,
                timescaledb.compress_segmentby = 'stock_id',
                timescaledb.compress_orderby = 'date DESC'
            )
        """))
        
        # Add compression policy for technical indicators
        try:
            conn.execute(text("""
                SELECT add_compression_policy(
                    'technical_indicators',
                    INTERVAL '14 days',
                    if_not_exists => TRUE
                )
            """))
            logger.info("Added compression policy for technical_indicators")
        except Exception as e:
            logger.warning(f"Compression policy may already exist: {e}")
        
        # Removed conn.commit() - transaction is managed by context manager
    
    def _create_continuous_aggregates(self, conn):
        """
        Create continuous aggregates for common queries.
        These are automatically updated materialized views.
        """
        
        # Daily stock summary aggregate
        conn.execute(text("""
            CREATE MATERIALIZED VIEW IF NOT EXISTS daily_stock_summary
            WITH (timescaledb.continuous) AS
            SELECT
                stock_id,
                time_bucket('1 day', date) AS day,
                AVG(close) AS avg_close,
                MAX(high) AS max_high,
                MIN(low) AS min_low,
                SUM(volume) AS total_volume,
                FIRST(open, date) AS open,
                LAST(close, date) AS close
            FROM price_history
            GROUP BY stock_id, day
            WITH NO DATA
        """))
        
        # Refresh the aggregate with historical data
        try:
            conn.execute(text("""
                SELECT add_continuous_aggregate_policy(
                    'daily_stock_summary',
                    start_offset => INTERVAL '3 days',
                    end_offset => INTERVAL '1 hour',
                    schedule_interval => INTERVAL '1 hour',
                    if_not_exists => TRUE
                )
            """))
            logger.info("Created continuous aggregate: daily_stock_summary")
        except Exception as e:
            logger.warning(f"Continuous aggregate policy may already exist: {e}")
        
        # Weekly stock performance aggregate
        conn.execute(text("""
            CREATE MATERIALIZED VIEW IF NOT EXISTS weekly_stock_performance
            WITH (timescaledb.continuous) AS
            SELECT
                stock_id,
                time_bucket('1 week', date) AS week,
                AVG(close) AS avg_close,
                MAX(high) AS week_high,
                MIN(low) AS week_low,
                SUM(volume) AS week_volume,
                FIRST(open, date) AS week_open,
                LAST(close, date) AS week_close,
                (LAST(close, date) - FIRST(open, date)) / FIRST(open, date) * 100 AS week_return
            FROM price_history
            GROUP BY stock_id, week
            WITH NO DATA
        """))
        
        # Monthly aggregate for long-term analysis
        conn.execute(text("""
            CREATE MATERIALIZED VIEW IF NOT EXISTS monthly_stock_metrics
            WITH (timescaledb.continuous) AS
            SELECT
                stock_id,
                time_bucket('1 month', date) AS month,
                AVG(close) AS avg_close,
                STDDEV(close) AS volatility,
                MAX(high) AS month_high,
                MIN(low) AS month_low,
                SUM(volume) AS month_volume,
                COUNT(*) AS trading_days
            FROM price_history
            GROUP BY stock_id, month
            WITH NO DATA
        """))
        
        # Removed conn.commit() - transaction is managed by context manager
    
    def _setup_retention_policies(self, conn):
        """
        Set up data retention policies to automatically remove old data.
        Keeps storage costs under control.
        """
        
        # Keep detailed price data for 2 years
        try:
            conn.execute(text("""
                SELECT add_retention_policy(
                    'price_history',
                    INTERVAL '2 years',
                    if_not_exists => TRUE
                )
            """))
            logger.info("Added retention policy for price_history")
        except Exception as e:
            logger.warning(f"Retention policy may already exist: {e}")
        
        # Keep technical indicators for 1 year
        try:
            conn.execute(text("""
                SELECT add_retention_policy(
                    'technical_indicators',
                    INTERVAL '1 year',
                    if_not_exists => TRUE
                )
            """))
            logger.info("Added retention policy for technical_indicators")
        except Exception as e:
            logger.warning(f"Retention policy may already exist: {e}")
        
        # Keep market metrics for 6 months
        try:
            conn.execute(text("""
                SELECT add_retention_policy(
                    'market_metrics',
                    INTERVAL '6 months',
                    if_not_exists => TRUE
                )
            """))
            logger.info("Added retention policy for market_metrics")
        except Exception as e:
            logger.warning(f"Retention policy may already exist: {e}")
        
        # Removed conn.commit() - transaction is managed by context manager
    
    def _create_optimized_indexes(self, conn):
        """Create optimized indexes for common query patterns."""
        
        # Composite index for price queries
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_price_history_stock_date_desc
            ON price_history (stock_id, date DESC)
            WHERE date >= CURRENT_DATE - INTERVAL '90 days'
        """))
        
        # Index for volume-based queries
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_price_history_volume
            ON price_history (stock_id, volume DESC)
            WHERE volume > 1000000
        """))
        
        # Index for technical indicators
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_technical_indicators_stock_date
            ON technical_indicators (stock_id, date DESC)
            WHERE date >= CURRENT_DATE - INTERVAL '30 days'
        """))
        
        # Index for RSI-based queries (oversold/overbought)
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_technical_indicators_rsi
            ON technical_indicators (rsi)
            WHERE rsi IS NOT NULL AND (rsi < 30 OR rsi > 70)
        """))
        
        # Removed conn.commit() - transaction is managed by context manager
        logger.info("Created optimized indexes")
    
    def get_compression_stats(self) -> Dict:
        """Get compression statistics for hypertables."""
        with self.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT
                    hypertable_name,
                    pg_size_pretty(before_compression_total_bytes) AS before_compression,
                    pg_size_pretty(after_compression_total_bytes) AS after_compression,
                    ROUND(
                        100 * (1 - after_compression_total_bytes::float / 
                        NULLIF(before_compression_total_bytes, 0)),
                        2
                    ) AS compression_ratio_pct
                FROM timescaledb_information.compression_stats
            """))
            
            stats = []
            for row in result:
                stats.append({
                    'table': row[0],
                    'before': row[1],
                    'after': row[2],
                    'compression_ratio': row[3]
                })
            
            return {'compression_stats': stats}
    
    def optimize_chunks(self):
        """Manually trigger chunk optimization and recompression."""
        with self.engine.connect() as conn:
            # Recompress old chunks that may have been modified
            conn.execute(text("""
                CALL timescaledb.compress_chunk(c)
                FROM timescaledb_information.chunks c
                WHERE c.hypertable_name = 'price_history'
                AND c.is_compressed = false
                AND c.range_end < NOW() - INTERVAL '7 days'
            """))
            
            # Run VACUUM ANALYZE on hypertables
            conn.execute(text("VACUUM ANALYZE price_history"))
            conn.execute(text("VACUUM ANALYZE technical_indicators"))
            
            # Removed conn.commit() - transaction is managed by context manager
            logger.info("Optimized and recompressed chunks")
    
    def create_fast_lookup_functions(self):
        """Create optimized SQL functions for common operations."""
        with self.engine.connect() as conn:
            # Function to get latest price for a stock
            conn.execute(text("""
                CREATE OR REPLACE FUNCTION get_latest_price(p_stock_id INTEGER)
                RETURNS TABLE (
                    date DATE,
                    close NUMERIC,
                    volume BIGINT
                ) AS $$
                BEGIN
                    RETURN QUERY
                    SELECT ph.date, ph.close, ph.volume
                    FROM price_history ph
                    WHERE ph.stock_id = p_stock_id
                    ORDER BY ph.date DESC
                    LIMIT 1;
                END;
                $$ LANGUAGE plpgsql;
            """))
            
            # Function to calculate moving average
            conn.execute(text("""
                CREATE OR REPLACE FUNCTION calculate_sma(
                    p_stock_id INTEGER,
                    p_period INTEGER,
                    p_date DATE DEFAULT CURRENT_DATE
                )
                RETURNS NUMERIC AS $$
                DECLARE
                    v_sma NUMERIC;
                BEGIN
                    SELECT AVG(close) INTO v_sma
                    FROM (
                        SELECT close
                        FROM price_history
                        WHERE stock_id = p_stock_id
                        AND date <= p_date
                        ORDER BY date DESC
                        LIMIT p_period
                    ) AS recent_prices;
                    
                    RETURN v_sma;
                END;
                $$ LANGUAGE plpgsql;
            """))
            
            # Removed conn.commit() - transaction is managed by context manager
            logger.info("Created optimized lookup functions")


def initialize_timescaledb() -> bool:
    """Main function to initialize TimescaleDB."""
    initializer = TimescaleDBInitializer()
    
    logger.info("Starting TimescaleDB initialization...")
    
    if initializer.initialize_timescaledb():
        logger.info("TimescaleDB initialization completed successfully")
        
        # Create fast lookup functions
        initializer.create_fast_lookup_functions()
        
        # Get and log compression stats
        stats = initializer.get_compression_stats()
        logger.info(f"Compression stats: {stats}")
        
        # Optimize chunks
        initializer.optimize_chunks()
        
        return True
    else:
        logger.error("TimescaleDB initialization failed")
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    initialize_timescaledb()