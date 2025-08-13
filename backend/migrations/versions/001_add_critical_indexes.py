"""Add critical database indexes for performance optimization

Revision ID: 001_critical_indexes
Revises: 
Create Date: 2025-08-07

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text


# revision identifiers, used by Alembic.
revision = '001_critical_indexes'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    """Add critical indexes for performance optimization"""
    
    # Price History Table Indexes
    # Composite index for stock_id, date (most common query pattern)
    op.create_index(
        'idx_price_history_stock_date_desc',
        'price_history',
        ['stock_id', sa.text('date DESC')],
        postgresql_concurrently=True
    )
    
    # Index for date range queries
    op.create_index(
        'idx_price_history_date_range',
        'price_history',
        ['date'],
        postgresql_concurrently=True
    )
    
    # Index for volume analysis
    op.create_index(
        'idx_price_history_volume',
        'price_history',
        ['volume'],
        postgresql_where=sa.text('volume > 0'),
        postgresql_concurrently=True
    )
    
    # Composite index for OHLC queries
    op.create_index(
        'idx_price_history_ohlc',
        'price_history',
        ['stock_id', 'date', 'close'],
        postgresql_concurrently=True
    )
    
    # Technical Indicators Table Indexes
    # Primary composite index
    op.create_index(
        'idx_technical_stock_date_desc',
        'technical_indicators',
        ['stock_id', sa.text('date DESC')],
        postgresql_concurrently=True
    )
    
    # Index for RSI queries
    op.create_index(
        'idx_technical_rsi',
        'technical_indicators',
        ['stock_id', 'rsi_14'],
        postgresql_where=sa.text('rsi_14 IS NOT NULL'),
        postgresql_concurrently=True
    )
    
    # Index for moving averages
    op.create_index(
        'idx_technical_sma',
        'technical_indicators',
        ['stock_id', 'date', 'sma_20', 'sma_50'],
        postgresql_concurrently=True
    )
    
    # Index for MACD signals
    op.create_index(
        'idx_technical_macd',
        'technical_indicators',
        ['stock_id', 'macd', 'macd_signal'],
        postgresql_where=sa.text('macd IS NOT NULL AND macd_signal IS NOT NULL'),
        postgresql_concurrently=True
    )
    
    # Recommendations Table Indexes
    # Active recommendations by confidence
    op.create_index(
        'idx_recommendations_active_confidence',
        'recommendations',
        ['is_active', sa.text('confidence_score DESC'), sa.text('created_at DESC')],
        postgresql_where=sa.text('is_active = true'),
        postgresql_concurrently=True
    )
    
    # Recommendations by action and date
    op.create_index(
        'idx_recommendations_action_date',
        'recommendations',
        ['action', sa.text('created_at DESC')],
        postgresql_concurrently=True
    )
    
    # Stock-specific recommendations
    op.create_index(
        'idx_recommendations_stock_active',
        'recommendations',
        ['stock_id', 'is_active', sa.text('created_at DESC')],
        postgresql_concurrently=True
    )
    
    # Performance tracking index
    op.create_index(
        'idx_recommendations_performance',
        'recommendations',
        ['outcome', 'actual_return'],
        postgresql_where=sa.text('outcome IS NOT NULL'),
        postgresql_concurrently=True
    )
    
    # Priority recommendations
    op.create_index(
        'idx_recommendations_priority',
        'recommendations',
        [sa.text('priority ASC'), sa.text('confidence_score DESC')],
        postgresql_where=sa.text('is_active = true AND priority <= 3'),
        postgresql_concurrently=True
    )
    
    # News Sentiment Table Indexes
    # Stock and date index
    op.create_index(
        'idx_news_sentiment_stock_date',
        'news_sentiment',
        ['stock_id', sa.text('published_at DESC')],
        postgresql_concurrently=True
    )
    
    # Sentiment score index
    op.create_index(
        'idx_news_sentiment_score',
        'news_sentiment',
        ['sentiment_score', 'confidence'],
        postgresql_where=sa.text('confidence > 0.7'),
        postgresql_concurrently=True
    )
    
    # Fundamentals Table Indexes  
    # Stock and period index
    op.create_index(
        'idx_fundamentals_stock_period',
        'fundamentals',
        ['stock_id', sa.text('period_date DESC'), 'period_type'],
        postgresql_concurrently=True
    )
    
    # Valuation ratios index
    op.create_index(
        'idx_fundamentals_ratios',
        'fundamentals',
        ['pe_ratio', 'pb_ratio', 'ps_ratio'],
        postgresql_where=sa.text('pe_ratio > 0 AND pe_ratio < 100'),
        postgresql_concurrently=True
    )
    
    # Stock Table Indexes
    # Active stocks index
    op.create_index(
        'idx_stocks_active',
        'stocks',
        ['is_active', 'is_tradeable', 'market_cap'],
        postgresql_where=sa.text('is_active = true AND is_tradeable = true'),
        postgresql_concurrently=True
    )
    
    # Sector/Industry index
    op.create_index(
        'idx_stocks_sector_industry',
        'stocks',
        ['sector_id', 'industry_id', 'market_cap'],
        postgresql_concurrently=True
    )
    
    # API Usage Table Indexes
    # Provider and timestamp index
    op.create_index(
        'idx_api_usage_provider_time',
        'api_usage',
        ['provider', sa.text('timestamp DESC')],
        postgresql_concurrently=True
    )
    
    # Cost monitoring index
    op.create_index(
        'idx_api_usage_cost',
        'api_usage',
        [sa.text('timestamp::date'), 'estimated_cost'],
        postgresql_where=sa.text('estimated_cost > 0'),
        postgresql_concurrently=True
    )


def downgrade():
    """Remove the critical indexes"""
    
    # Price History indexes
    op.drop_index('idx_price_history_stock_date_desc', table_name='price_history')
    op.drop_index('idx_price_history_date_range', table_name='price_history')
    op.drop_index('idx_price_history_volume', table_name='price_history')
    op.drop_index('idx_price_history_ohlc', table_name='price_history')
    
    # Technical Indicators indexes
    op.drop_index('idx_technical_stock_date_desc', table_name='technical_indicators')
    op.drop_index('idx_technical_rsi', table_name='technical_indicators')
    op.drop_index('idx_technical_sma', table_name='technical_indicators')
    op.drop_index('idx_technical_macd', table_name='technical_indicators')
    
    # Recommendations indexes
    op.drop_index('idx_recommendations_active_confidence', table_name='recommendations')
    op.drop_index('idx_recommendations_action_date', table_name='recommendations')
    op.drop_index('idx_recommendations_stock_active', table_name='recommendations')
    op.drop_index('idx_recommendations_performance', table_name='recommendations')
    op.drop_index('idx_recommendations_priority', table_name='recommendations')
    
    # News Sentiment indexes
    op.drop_index('idx_news_sentiment_stock_date', table_name='news_sentiment')
    op.drop_index('idx_news_sentiment_score', table_name='news_sentiment')
    
    # Fundamentals indexes
    op.drop_index('idx_fundamentals_stock_period', table_name='fundamentals')
    op.drop_index('idx_fundamentals_ratios', table_name='fundamentals')
    
    # Stock indexes
    op.drop_index('idx_stocks_active', table_name='stocks')
    op.drop_index('idx_stocks_sector_industry', table_name='stocks')
    
    # API Usage indexes
    op.drop_index('idx_api_usage_provider_time', table_name='api_usage')
    op.drop_index('idx_api_usage_cost', table_name='api_usage')