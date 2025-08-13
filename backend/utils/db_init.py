#!/usr/bin/env python3
"""
Database Initialization Script
Creates all tables, indexes, and loads initial data
"""

import os
import sys
import asyncio
import logging
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from backend.models.unified_models import Base, Stock, Exchange, Sector, Industry, CostMetrics, TechnicalIndicators
from backend.config.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatabaseInitializer:
    """Initialize database with schema and initial data"""
    
    def __init__(self):
        self.engine = create_engine(settings.DATABASE_URL)
        
    def create_database(self):
        """Create database if it doesn't exist"""
        try:
            # Connect to default postgres database
            default_db_url = settings.DATABASE_URL.rsplit('/', 1)[0] + '/postgres'
            default_engine = create_engine(default_db_url)
            
            with default_engine.connect() as conn:
                # Check if database exists
                result = conn.execute(
                    text("SELECT 1 FROM pg_database WHERE datname = 'investment_db'")
                )
                
                if not result.fetchone():
                    # Create database
                    conn.execute(text("COMMIT"))  # Close any transaction
                    conn.execute(text("CREATE DATABASE investment_db"))
                    logger.info("Created database: investment_db")
                else:
                    logger.info("Database already exists: investment_db")
                    
        except Exception as e:
            logger.error(f"Error creating database: {e}")
            raise
            
    def create_extensions(self):
        """Create required PostgreSQL extensions"""
        try:
            with self.engine.connect() as conn:
                extensions = [
                    'uuid-ossp',  # For UUID generation
                    'pg_trgm',    # For text search
                    'btree_gin'   # For complex indexes
                ]
                
                for ext in extensions:
                    try:
                        conn.execute(text(f"CREATE EXTENSION IF NOT EXISTS \"{ext}\""))
                        conn.commit()
                        logger.info(f"Created extension: {ext}")
                    except Exception as e:
                        logger.warning(f"Extension {ext} might already exist: {e}")
                        
        except Exception as e:
            logger.error(f"Error creating extensions: {e}")
            raise
            
    def create_tables(self):
        """Create all database tables"""
        try:
            # Create all tables
            Base.metadata.create_all(bind=self.engine)
            logger.info("Created all database tables")
            
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            raise
            
    def create_indexes(self):
        """Create custom indexes for performance"""
        indexes = [
            # Price history indexes
            "CREATE INDEX IF NOT EXISTS idx_price_history_ticker_date ON price_history(stock_id, date DESC)",
            "CREATE INDEX IF NOT EXISTS idx_price_history_date ON price_history(date)",
            
            # Technical indicators indexes
            "CREATE INDEX IF NOT EXISTS idx_technical_indicators_ticker_date ON technical_indicators(stock_id, date DESC)",
            
            # News sentiment indexes
            "CREATE INDEX IF NOT EXISTS idx_news_sentiment_ticker_time ON news_sentiment(stock_id, published_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_news_sentiment_sentiment ON news_sentiment(sentiment_score)",
            
            # ML predictions indexes
            "CREATE INDEX IF NOT EXISTS idx_ml_predictions_ticker_time ON ml_predictions(stock_id, created_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_ml_predictions_horizon ON ml_predictions(prediction_horizon)",
            
            # Recommendations indexes
            "CREATE INDEX IF NOT EXISTS idx_recommendations_active ON recommendations(is_active, created_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_recommendations_ticker ON recommendations(stock_id, created_at DESC)",
            
            # Text search indexes
            "CREATE INDEX IF NOT EXISTS idx_stocks_search ON stocks USING gin(to_tsvector('english', ticker || ' ' || name))",
            
            # Composite indexes for common queries
            "CREATE INDEX IF NOT EXISTS idx_stocks_sector_industry ON stocks(sector_id, industry_id)",
            "CREATE INDEX IF NOT EXISTS idx_stocks_market_cap ON stocks(market_cap DESC) WHERE is_active = true",
        ]
        
        try:
            with self.engine.connect() as conn:
                for index_sql in indexes:
                    try:
                        conn.execute(text(index_sql))
                        conn.commit()
                        logger.info(f"Created index: {index_sql.split('idx_')[1].split(' ')[0]}")
                    except Exception as e:
                        logger.warning(f"Index might already exist: {e}")
                        
        except Exception as e:
            logger.error(f"Error creating indexes: {e}")
            raise
            
    def load_initial_data(self):
        """Load initial reference data"""
        try:
            with Session(self.engine) as session:
                # Load exchanges
                exchanges = [
                    {"code": "NYSE", "name": "New York Stock Exchange", "timezone": "America/New_York"},
                    {"code": "NASDAQ", "name": "NASDAQ Stock Market", "timezone": "America/New_York"},
                    {"code": "AMEX", "name": "NYSE American", "timezone": "America/New_York"}
                ]
                
                for exchange_data in exchanges:
                    exchange = session.query(Exchange).filter_by(code=exchange_data["code"]).first()
                    if not exchange:
                        exchange = Exchange(**exchange_data)
                        session.add(exchange)
                        logger.info(f"Added exchange: {exchange_data['name']}")
                        
                # Load sectors
                sectors = [
                    "Technology", "Healthcare", "Financials", "Consumer Discretionary",
                    "Communication Services", "Industrials", "Consumer Staples",
                    "Energy", "Utilities", "Real Estate", "Materials"
                ]
                
                for sector_name in sectors:
                    sector = session.query(Sector).filter_by(name=sector_name).first()
                    if not sector:
                        sector = Sector(name=sector_name)
                        session.add(sector)
                        logger.info(f"Added sector: {sector_name}")
                        
                # Load industries (sample)
                industries = [
                    ("Software", "Technology"),
                    ("Semiconductors", "Technology"),
                    ("Banks", "Financials"),
                    ("Pharmaceuticals", "Healthcare"),
                    ("Retail", "Consumer Discretionary"),
                    ("Oil & Gas", "Energy"),
                    ("REITs", "Real Estate")
                ]
                
                # First commit sectors to get their IDs
                session.commit()
                
                for industry_name, sector_name in industries:
                    industry = session.query(Industry).filter_by(name=industry_name).first()
                    if not industry:
                        sector = session.query(Sector).filter_by(name=sector_name).first()
                        if sector:
                            industry = Industry(name=industry_name, sector_id=sector.id)
                            session.add(industry)
                            logger.info(f"Added industry: {industry_name}")
                            
                session.commit()
                logger.info("Initial data loaded successfully")
                
        except Exception as e:
            logger.error(f"Error loading initial data: {e}")
            raise
            
    def create_functions(self):
        """Create custom PostgreSQL functions"""
        functions = [
            # Function to calculate price change percentage
            """
            CREATE OR REPLACE FUNCTION calculate_price_change(
                current_price DECIMAL,
                previous_price DECIMAL
            ) RETURNS DECIMAL AS $$
            BEGIN
                IF previous_price = 0 OR previous_price IS NULL THEN
                    RETURN 0;
                END IF;
                RETURN ((current_price - previous_price) / previous_price) * 100;
            END;
            $$ LANGUAGE plpgsql;
            """,
            
            # Function to get latest price for a stock
            """
            CREATE OR REPLACE FUNCTION get_latest_price(stock_id_param INTEGER)
            RETURNS DECIMAL AS $$
            DECLARE
                latest_price DECIMAL;
            BEGIN
                SELECT close INTO latest_price
                FROM price_history
                WHERE stock_id = stock_id_param
                ORDER BY date DESC
                LIMIT 1;
                
                RETURN COALESCE(latest_price, 0);
            END;
            $$ LANGUAGE plpgsql;
            """,
            
            # Function to calculate moving average
            """
            CREATE OR REPLACE FUNCTION calculate_sma(
                stock_id_param INTEGER,
                days INTEGER
            ) RETURNS DECIMAL AS $$
            DECLARE
                avg_price DECIMAL;
            BEGIN
                SELECT AVG(close) INTO avg_price
                FROM (
                    SELECT close
                    FROM price_history
                    WHERE stock_id = stock_id_param
                    ORDER BY date DESC
                    LIMIT days
                ) subquery;
                
                RETURN COALESCE(avg_price, 0);
            END;
            $$ LANGUAGE plpgsql;
            """
        ]
        
        try:
            with self.engine.connect() as conn:
                for func_sql in functions:
                    try:
                        conn.execute(text(func_sql))
                        conn.commit()
                        logger.info("Created custom function")
                    except Exception as e:
                        logger.warning(f"Function might already exist: {e}")
                        
        except Exception as e:
            logger.error(f"Error creating functions: {e}")
            raise
            
    def create_views(self):
        """Create database views for common queries"""
        views = [
            # Stock overview view
            """
            CREATE OR REPLACE VIEW stock_overview AS
            SELECT 
                s.id,
                s.ticker,
                s.name,
                s.market_cap,
                e.name as exchange,
                sec.name as sector,
                ind.name as industry,
                ph.close as current_price,
                calculate_price_change(ph.close, ph_prev.close) as daily_change,
                s.is_active
            FROM stocks s
            LEFT JOIN exchanges e ON s.exchange_id = e.id
            LEFT JOIN sectors sec ON s.sector_id = sec.id
            LEFT JOIN industries ind ON s.industry_id = ind.id
            LEFT JOIN LATERAL (
                SELECT close, date
                FROM price_history
                WHERE stock_id = s.id
                ORDER BY date DESC
                LIMIT 1
            ) ph ON true
            LEFT JOIN LATERAL (
                SELECT close
                FROM price_history
                WHERE stock_id = s.id AND date < ph.date
                ORDER BY date DESC
                LIMIT 1
            ) ph_prev ON true;
            """,
            
            # Active recommendations view
            """
            CREATE OR REPLACE VIEW active_recommendations AS
            SELECT 
                r.*,
                s.ticker,
                s.name as stock_name,
                sec.name as sector
            FROM recommendations r
            JOIN stocks s ON r.stock_id = s.id
            LEFT JOIN sectors sec ON s.sector_id = sec.id
            WHERE r.is_active = true
            ORDER BY r.created_at DESC;
            """
        ]
        
        try:
            with self.engine.connect() as conn:
                for view_sql in views:
                    try:
                        conn.execute(text(view_sql))
                        conn.commit()
                        logger.info("Created view")
                    except Exception as e:
                        logger.warning(f"View might already exist: {e}")
                        
        except Exception as e:
            logger.error(f"Error creating views: {e}")
            raise
            
    def verify_setup(self):
        """Verify database setup is complete"""
        try:
            with self.engine.connect() as conn:
                # Check tables
                result = conn.execute(text("""
                    SELECT COUNT(*) 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                """))
                table_count = result.scalar()
                
                # Check exchanges
                result = conn.execute(text("SELECT COUNT(*) FROM exchanges"))
                exchange_count = result.scalar()
                
                # Check sectors
                result = conn.execute(text("SELECT COUNT(*) FROM sectors"))
                sector_count = result.scalar()
                
                logger.info(f"Setup verification:")
                logger.info(f"  - Tables created: {table_count}")
                logger.info(f"  - Exchanges loaded: {exchange_count}")
                logger.info(f"  - Sectors loaded: {sector_count}")
                
                return table_count > 0 and exchange_count > 0 and sector_count > 0
                
        except Exception as e:
            logger.error(f"Error verifying setup: {e}")
            return False
            
    def initialize(self):
        """Run complete initialization"""
        logger.info("Starting database initialization...")
        
        try:
            # Create database
            self.create_database()
            
            # Create extensions
            self.create_extensions()
            
            # Create tables
            self.create_tables()
            
            # Create indexes
            self.create_indexes()
            
            # Create functions
            self.create_functions()
            
            # Create views
            self.create_views()
            
            # Load initial data
            self.load_initial_data()
            
            # Verify setup
            if self.verify_setup():
                logger.info("✅ Database initialization completed successfully!")
                return True
            else:
                logger.error("❌ Database initialization verification failed")
                return False
                
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            return False


def main():
    """Main initialization function"""
    initializer = DatabaseInitializer()
    success = initializer.initialize()
    
    if success:
        print("\n✅ Database initialization completed successfully!")
        print("\nDatabase is ready for use.")
    else:
        print("\n❌ Database initialization failed. Please check the logs.")
        sys.exit(1)


if __name__ == "__main__":
    main()