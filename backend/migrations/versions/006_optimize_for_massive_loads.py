"""Optimize database for massive daily stock data loads

Revision ID: 006
Revises: 005
Create Date: 2025-01-19 12:00:00.000000

This migration optimizes the database schema for handling 6000+ tickers with daily updates
including bulk insert optimizations, advanced partitioning, and storage compression.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text
from sqlalchemy.dialects import postgresql
from datetime import datetime, timedelta

# revision identifiers
revision = '006'
down_revision = '005'
branch_labels = None
depends_on = None


def upgrade():
    """Apply optimizations for massive data loads"""
    
    # ============================================================================
    # STEP 1: Enable Required Extensions
    # ============================================================================
    
    # Enable TimescaleDB if not already enabled
    op.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_stat_statements;")
    op.execute("CREATE EXTENSION IF NOT EXISTS btree_gin;")
    op.execute("CREATE EXTENSION IF NOT EXISTS btree_gist;")
    
    # ============================================================================
    # STEP 2: Optimize Stock Master Table
    # ============================================================================
    
    # Add columns for faster lookups
    op.add_column('stocks', sa.Column('symbol_hash', sa.BigInteger, nullable=True))
    op.add_column('stocks', sa.Column('is_sp500', sa.Boolean, default=False))
    op.add_column('stocks', sa.Column('is_nasdaq100', sa.Boolean, default=False))
    op.add_column('stocks', sa.Column('is_dow30', sa.Boolean, default=False))
    op.add_column('stocks', sa.Column('avg_daily_volume', sa.BigInteger))
    op.add_column('stocks', sa.Column('last_data_update', sa.DateTime))
    
    # Create hash values for ultra-fast symbol lookups
    op.execute("""
        UPDATE stocks SET symbol_hash = hashtext(symbol) WHERE symbol_hash IS NULL;
    """)
    
    # Create optimized indexes for stock lookups
    op.create_index(
        'idx_stocks_symbol_hash',
        'stocks',
        ['symbol_hash'],
        unique=True,
        postgresql_concurrently=True
    )
    
    op.create_index(
        'idx_stocks_active_volume',
        'stocks',
        ['is_active', 'avg_daily_volume'],
        postgresql_where=text('is_active = true AND avg_daily_volume > 1000000'),
        postgresql_concurrently=True
    )
    
    # ============================================================================
    # STEP 3: Create Optimized Price History Tables
    # ============================================================================
    
    # Create new optimized price history table
    op.create_table(
        'price_history_optimized',
        sa.Column('stock_id', sa.SmallInteger, nullable=False),  # Smaller int for 6000 stocks
        sa.Column('date', sa.Date, nullable=False),
        sa.Column('timestamp', sa.DateTime, nullable=False),  # For intraday data
        
        # Use REAL (4-byte) instead of DECIMAL for better compression/speed
        sa.Column('open', sa.REAL, nullable=False),
        sa.Column('high', sa.REAL, nullable=False),
        sa.Column('low', sa.REAL, nullable=False),
        sa.Column('close', sa.REAL, nullable=False),
        sa.Column('adjusted_close', sa.REAL),
        sa.Column('volume', sa.Integer, nullable=False),  # 4-byte int sufficient for volume
        
        # Pre-calculated fields for faster queries
        sa.Column('price_range', sa.REAL),  # high - low
        sa.Column('price_change', sa.REAL),  # close - open
        sa.Column('price_change_pct', sa.REAL),  # (close - open) / open * 100
        sa.Column('typical_price', sa.REAL),  # (high + low + close) / 3
        sa.Column('vwap', sa.REAL),  # Volume Weighted Average Price
        
        # Bit flags for market conditions (saves space)
        sa.Column('market_flags', sa.SmallInteger, default=0),  # Bitmap for various flags
        
        sa.PrimaryKey('stock_id', 'date', 'timestamp'),
    )
    
    # Convert to TimescaleDB hypertable
    op.execute("""
        SELECT create_hypertable(
            'price_history_optimized', 
            'timestamp',
            chunk_time_interval => INTERVAL '1 week',
            create_default_indexes => FALSE,
            if_not_exists => TRUE
        );
    """)
    
    # ============================================================================
    # STEP 4: Create Optimized Technical Indicators Table
    # ============================================================================
    
    op.create_table(
        'technical_indicators_optimized',
        sa.Column('stock_id', sa.SmallInteger, nullable=False),
        sa.Column('date', sa.Date, nullable=False),
        
        # Pack multiple indicators into arrays for better compression
        sa.Column('sma_values', sa.ARRAY(sa.REAL), nullable=True),  # [5,10,20,50,100,200]
        sa.Column('ema_values', sa.ARRAY(sa.REAL), nullable=True),  # [12,26,50]
        sa.Column('rsi_values', sa.ARRAY(sa.REAL), nullable=True),  # [14,30]
        sa.Column('bollinger_values', sa.ARRAY(sa.REAL), nullable=True),  # [upper,middle,lower,width]
        sa.Column('macd_values', sa.ARRAY(sa.REAL), nullable=True),  # [macd,signal,histogram]
        sa.Column('stochastic_values', sa.ARRAY(sa.REAL), nullable=True),  # [k,d]
        sa.Column('volume_indicators', sa.ARRAY(sa.REAL), nullable=True),  # [obv,ad_line,mfi]
        sa.Column('trend_indicators', sa.ARRAY(sa.REAL), nullable=True),  # [adx,plus_di,minus_di]
        
        # Most commonly used indicators as separate columns for fast filtering
        sa.Column('rsi_14', sa.REAL),
        sa.Column('sma_20', sa.REAL),
        sa.Column('sma_50', sa.REAL),
        sa.Column('sma_200', sa.REAL),
        
        # Signal flags packed into bits
        sa.Column('signal_flags', sa.Integer, default=0),  # Bull/bear signals as bits
        
        sa.PrimaryKey('stock_id', 'date'),
    )
    
    # Convert to hypertable
    op.execute("""
        SELECT create_hypertable(
            'technical_indicators_optimized', 
            'date',
            chunk_time_interval => INTERVAL '1 month',
            create_default_indexes => FALSE,
            if_not_exists => TRUE
        );
    """)
    
    # ============================================================================
    # STEP 5: Create Bulk News Sentiment Table
    # ============================================================================
    
    op.create_table(
        'news_sentiment_bulk',
        sa.Column('id', sa.BigInteger, primary_key=True),
        sa.Column('stock_id', sa.SmallInteger, nullable=False),
        sa.Column('date', sa.Date, nullable=False),
        sa.Column('source_hash', sa.BigInteger, nullable=False),  # Hash of source + url
        
        # Compressed text storage
        sa.Column('headline_vector', postgresql.TSVECTOR),  # For full-text search
        sa.Column('content_summary', sa.String(500)),  # Truncated content
        
        # Sentiment as packed values
        sa.Column('sentiment_score', sa.SmallInteger),  # -1000 to 1000 (scaled)
        sa.Column('confidence', sa.SmallInteger),  # 0 to 1000 (scaled)
        sa.Column('impact_score', sa.SmallInteger),  # 0 to 1000 (scaled)
        
        # Timestamps
        sa.Column('published_at', sa.DateTime, nullable=False),
        sa.Column('scraped_at', sa.DateTime, default=sa.func.now()),
        
        # Keywords as array for efficient storage
        sa.Column('keywords', sa.ARRAY(sa.String(50)), nullable=True),
    )
    
    # Make it a hypertable
    op.execute("""
        SELECT create_hypertable(
            'news_sentiment_bulk', 
            'published_at',
            chunk_time_interval => INTERVAL '1 week',
            create_default_indexes => FALSE,
            if_not_exists => TRUE
        );
    """)
    
    # ============================================================================
    # STEP 6: Create Fundamentals Staging Table
    # ============================================================================
    
    op.create_table(
        'fundamentals_staging',
        sa.Column('stock_id', sa.SmallInteger, nullable=False),
        sa.Column('period_date', sa.Date, nullable=False),
        sa.Column('period_type', sa.String(1), nullable=False),  # Q or A
        
        # Financial data as JSONB for flexibility and compression
        sa.Column('income_statement', postgresql.JSONB),
        sa.Column('balance_sheet', postgresql.JSONB),
        sa.Column('cash_flow', postgresql.JSONB),
        sa.Column('ratios', postgresql.JSONB),
        
        # Most important ratios as separate columns for fast queries
        sa.Column('pe_ratio', sa.REAL),
        sa.Column('pb_ratio', sa.REAL),
        sa.Column('roe', sa.REAL),
        sa.Column('debt_to_equity', sa.REAL),
        
        sa.Column('last_updated', sa.DateTime, default=sa.func.now()),
        
        sa.PrimaryKey('stock_id', 'period_date', 'period_type'),
    )
    
    # ============================================================================
    # STEP 7: Create Bulk Insert Staging Tables
    # ============================================================================
    
    # Staging table for bulk price data inserts (unlogged for speed)
    op.execute("""
        CREATE UNLOGGED TABLE price_data_staging (
            stock_id SMALLINT,
            date DATE,
            timestamp TIMESTAMP,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            adjusted_close REAL,
            volume INTEGER,
            source_batch_id UUID DEFAULT gen_random_uuid()
        );
    """)
    
    # Staging table for technical indicators
    op.execute("""
        CREATE UNLOGGED TABLE technical_staging (
            stock_id SMALLINT,
            date DATE,
            indicators JSONB,
            calculation_batch_id UUID DEFAULT gen_random_uuid()
        );
    """)
    
    # ============================================================================
    # STEP 8: Create Optimized Indexes
    # ============================================================================
    
    # Price history indexes
    op.execute("""
        CREATE INDEX CONCURRENTLY idx_price_opt_stock_date 
        ON price_history_optimized USING BRIN (stock_id, date);
    """)
    
    op.execute("""
        CREATE INDEX CONCURRENTLY idx_price_opt_volume 
        ON price_history_optimized (volume) 
        WHERE volume > 1000000;
    """)
    
    op.execute("""
        CREATE INDEX CONCURRENTLY idx_price_opt_change 
        ON price_history_optimized (price_change_pct) 
        WHERE abs(price_change_pct) > 5.0;
    """)
    
    # Technical indicators indexes
    op.execute("""
        CREATE INDEX CONCURRENTLY idx_tech_opt_rsi 
        ON technical_indicators_optimized (rsi_14) 
        WHERE rsi_14 < 30 OR rsi_14 > 70;
    """)
    
    op.execute("""
        CREATE INDEX CONCURRENTLY idx_tech_opt_sma_cross 
        ON technical_indicators_optimized (stock_id, sma_20, sma_50) 
        WHERE sma_20 IS NOT NULL AND sma_50 IS NOT NULL;
    """)
    
    # News sentiment indexes
    op.execute("""
        CREATE INDEX CONCURRENTLY idx_news_bulk_sentiment 
        ON news_sentiment_bulk (sentiment_score, impact_score) 
        WHERE abs(sentiment_score) > 500;
    """)
    
    op.execute("""
        CREATE INDEX CONCURRENTLY idx_news_bulk_headlines 
        ON news_sentiment_bulk USING GIN (headline_vector);
    """)
    
    # ============================================================================
    # STEP 9: Create Materialized Views for Common Queries
    # ============================================================================
    
    # Daily aggregated stock metrics
    op.execute("""
        CREATE MATERIALIZED VIEW daily_stock_summary AS
        WITH daily_stats AS (
            SELECT 
                stock_id,
                date,
                close,
                volume,
                price_change_pct,
                LAG(close) OVER (PARTITION BY stock_id ORDER BY date) as prev_close,
                AVG(volume) OVER (PARTITION BY stock_id ORDER BY date ROWS 20 PRECEDING) as avg_volume_20d
            FROM price_history_optimized
            WHERE date >= CURRENT_DATE - INTERVAL '2 years'
        )
        SELECT 
            stock_id,
            date,
            close,
            volume,
            price_change_pct,
            CASE WHEN prev_close > 0 THEN (close - prev_close) / prev_close * 100 ELSE 0 END as daily_return,
            CASE WHEN avg_volume_20d > 0 THEN volume / avg_volume_20d ELSE 1 END as volume_ratio
        FROM daily_stats
        WHERE prev_close IS NOT NULL;
    """)
    
    op.execute("""
        CREATE UNIQUE INDEX idx_daily_summary_stock_date 
        ON daily_stock_summary (stock_id, date);
    """)
    
    # Weekly performance summary
    op.execute("""
        CREATE MATERIALIZED VIEW weekly_performance AS
        SELECT 
            stock_id,
            DATE_TRUNC('week', date) as week,
            FIRST_VALUE(open) OVER w as week_open,
            MAX(high) OVER w as week_high,
            MIN(low) OVER w as week_low,
            LAST_VALUE(close) OVER w as week_close,
            SUM(volume) OVER w as week_volume
        FROM price_history_optimized
        WHERE date >= CURRENT_DATE - INTERVAL '1 year'
        WINDOW w AS (
            PARTITION BY stock_id, DATE_TRUNC('week', date) 
            ORDER BY date 
            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
        );
    """)
    
    # ============================================================================
    # STEP 10: Set Up Compression Policies
    # ============================================================================
    
    # Enable compression on hypertables
    op.execute("""
        ALTER TABLE price_history_optimized SET (
            timescaledb.compress,
            timescaledb.compress_orderby = 'timestamp DESC',
            timescaledb.compress_segmentby = 'stock_id',
            timescaledb.compress_chunk_time_interval = '7 days'
        );
    """)
    
    op.execute("""
        ALTER TABLE technical_indicators_optimized SET (
            timescaledb.compress,
            timescaledb.compress_orderby = 'date DESC',
            timescaledb.compress_segmentby = 'stock_id',
            timescaledb.compress_chunk_time_interval = '14 days'
        );
    """)
    
    op.execute("""
        ALTER TABLE news_sentiment_bulk SET (
            timescaledb.compress,
            timescaledb.compress_orderby = 'published_at DESC',
            timescaledb.compress_segmentby = 'stock_id',
            timescaledb.compress_chunk_time_interval = '3 days'
        );
    """)
    
    # Add compression policies
    op.execute("""
        SELECT add_compression_policy('price_history_optimized', INTERVAL '3 days');
    """)
    
    op.execute("""
        SELECT add_compression_policy('technical_indicators_optimized', INTERVAL '7 days');
    """)
    
    op.execute("""
        SELECT add_compression_policy('news_sentiment_bulk', INTERVAL '1 day');
    """)
    
    # ============================================================================
    # STEP 11: Create Bulk Insert Functions
    # ============================================================================
    
    # Function for bulk price data insert
    op.execute("""
        CREATE OR REPLACE FUNCTION bulk_insert_price_data()
        RETURNS VOID AS $$
        BEGIN
            -- Insert from staging to main table
            INSERT INTO price_history_optimized (
                stock_id, date, timestamp, open, high, low, close, 
                adjusted_close, volume, price_range, price_change, 
                price_change_pct, typical_price
            )
            SELECT 
                stock_id, 
                date, 
                timestamp, 
                open, 
                high, 
                low, 
                close,
                adjusted_close,
                volume,
                high - low,
                close - open,
                CASE WHEN open > 0 THEN (close - open) / open * 100 ELSE 0 END,
                (high + low + close) / 3
            FROM price_data_staging
            ON CONFLICT (stock_id, date, timestamp) DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                adjusted_close = EXCLUDED.adjusted_close,
                volume = EXCLUDED.volume,
                price_range = EXCLUDED.price_range,
                price_change = EXCLUDED.price_change,
                price_change_pct = EXCLUDED.price_change_pct,
                typical_price = EXCLUDED.typical_price;
            
            -- Clear staging table
            TRUNCATE price_data_staging;
            
            -- Update stock metadata
            UPDATE stocks SET 
                last_data_update = NOW(),
                last_price_update = NOW()
            WHERE id IN (
                SELECT DISTINCT stock_id FROM price_data_staging
            );
        END;
        $$ LANGUAGE plpgsql;
    """)
    
    # Function for calculating technical indicators in bulk
    op.execute("""
        CREATE OR REPLACE FUNCTION calculate_technical_indicators_bulk(stock_ids INTEGER[])
        RETURNS VOID AS $$
        DECLARE
            stock_id INTEGER;
        BEGIN
            FOREACH stock_id IN ARRAY stock_ids LOOP
                INSERT INTO technical_indicators_optimized (
                    stock_id, date, sma_20, sma_50, sma_200, rsi_14
                )
                SELECT 
                    stock_id,
                    date,
                    AVG(close) OVER (ORDER BY date ROWS 19 PRECEDING) as sma_20,
                    AVG(close) OVER (ORDER BY date ROWS 49 PRECEDING) as sma_50,
                    AVG(close) OVER (ORDER BY date ROWS 199 PRECEDING) as sma_200,
                    -- Simplified RSI calculation
                    CASE 
                        WHEN AVG(CASE WHEN price_change > 0 THEN price_change ELSE 0 END) 
                             OVER (ORDER BY date ROWS 13 PRECEDING) = 0 
                        THEN 50
                        ELSE 100 - (100 / (1 + 
                            AVG(CASE WHEN price_change > 0 THEN price_change ELSE 0 END) 
                            OVER (ORDER BY date ROWS 13 PRECEDING) /
                            NULLIF(AVG(CASE WHEN price_change < 0 THEN ABS(price_change) ELSE 0 END) 
                                   OVER (ORDER BY date ROWS 13 PRECEDING), 0)
                        ))
                    END as rsi_14
                FROM price_history_optimized 
                WHERE stock_id = stock_id AND date >= CURRENT_DATE - INTERVAL '1 year'
                ON CONFLICT (stock_id, date) DO UPDATE SET
                    sma_20 = EXCLUDED.sma_20,
                    sma_50 = EXCLUDED.sma_50,
                    sma_200 = EXCLUDED.sma_200,
                    rsi_14 = EXCLUDED.rsi_14;
            END LOOP;
        END;
        $$ LANGUAGE plpgsql;
    """)
    
    # ============================================================================
    # STEP 12: Set Database Parameters for Performance
    # ============================================================================
    
    # These would typically be set in postgresql.conf, but we can set some session-level
    op.execute("SET work_mem = '256MB';")
    op.execute("SET maintenance_work_mem = '1GB';")
    op.execute("SET effective_cache_size = '4GB';")
    op.execute("SET random_page_cost = 1.1;")  # For SSD storage


def downgrade():
    """Remove optimizations"""
    
    # Drop functions
    op.execute("DROP FUNCTION IF EXISTS bulk_insert_price_data();")
    op.execute("DROP FUNCTION IF EXISTS calculate_technical_indicators_bulk(INTEGER[]);")
    
    # Drop materialized views
    op.execute("DROP MATERIALIZED VIEW IF EXISTS daily_stock_summary CASCADE;")
    op.execute("DROP MATERIALIZED VIEW IF EXISTS weekly_performance CASCADE;")
    
    # Drop staging tables
    op.execute("DROP TABLE IF EXISTS price_data_staging;")
    op.execute("DROP TABLE IF EXISTS technical_staging;")
    
    # Drop optimized tables
    op.execute("DROP TABLE IF EXISTS fundamentals_staging;")
    op.execute("DROP TABLE IF EXISTS news_sentiment_bulk;")
    op.execute("DROP TABLE IF EXISTS technical_indicators_optimized;")
    op.execute("DROP TABLE IF EXISTS price_history_optimized;")
    
    # Remove columns from stocks table
    op.drop_column('stocks', 'last_data_update')
    op.drop_column('stocks', 'avg_daily_volume')
    op.drop_column('stocks', 'is_dow30')
    op.drop_column('stocks', 'is_nasdaq100')
    op.drop_column('stocks', 'is_sp500')
    op.drop_column('stocks', 'symbol_hash')