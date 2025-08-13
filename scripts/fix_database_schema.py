#!/usr/bin/env python3
"""
Database Schema Fix Script
Addresses all identified database schema issues and inconsistencies

This script:
1. Fixes the column mismatch issues (ticker vs symbol)
2. Ensures exchanges table has 'code' column
3. Creates missing tables with proper schema
4. Migrates existing data safely
5. Validates schema consistency
"""

import sys
import os
import logging
from pathlib import Path
from datetime import datetime
import traceback

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, text, inspect, MetaData, Table, Column, Integer, String, Boolean, DateTime, Float
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from backend.models.consolidated_models import Base, Exchange, Sector, Industry, Stock
from backend.config.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatabaseSchemaFixer:
    """
    Comprehensive database schema fix utility
    """
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = create_engine(database_url)
        self.inspector = inspect(self.engine)
        
    def check_current_schema(self) -> dict:
        """Analyze current database schema and identify issues"""
        logger.info("Analyzing current database schema...")
        
        schema_status = {
            'existing_tables': [],
            'missing_tables': [],
            'column_issues': [],
            'needs_migration': False
        }
        
        try:
            # Get existing tables
            existing_tables = self.inspector.get_table_names()
            schema_status['existing_tables'] = existing_tables
            
            # Check for required tables
            required_tables = ['exchanges', 'sectors', 'industries', 'stocks', 'price_history']
            missing_tables = [table for table in required_tables if table not in existing_tables]
            schema_status['missing_tables'] = missing_tables
            
            # Check exchanges table
            if 'exchanges' in existing_tables:
                exchanges_columns = [col['name'] for col in self.inspector.get_columns('exchanges')]
                if 'code' not in exchanges_columns:
                    schema_status['column_issues'].append("exchanges table missing 'code' column")
                    schema_status['needs_migration'] = True
            else:
                schema_status['missing_tables'].append('exchanges')
                schema_status['needs_migration'] = True
            
            # Check stocks table
            if 'stocks' in existing_tables:
                stocks_columns = [col['name'] for col in self.inspector.get_columns('stocks')]
                if 'ticker' not in stocks_columns and 'symbol' not in stocks_columns:
                    schema_status['column_issues'].append("stocks table missing ticker/symbol column")
                    schema_status['needs_migration'] = True
                elif 'symbol' in stocks_columns and 'ticker' not in stocks_columns:
                    schema_status['column_issues'].append("stocks table uses 'symbol' instead of 'ticker'")
                    schema_status['needs_migration'] = True
            else:
                schema_status['missing_tables'].append('stocks')
                schema_status['needs_migration'] = True
            
            # Set migration flag if issues found
            if missing_tables or schema_status['column_issues']:
                schema_status['needs_migration'] = True
            
            logger.info(f"Schema analysis complete: {len(existing_tables)} tables found, {len(missing_tables)} missing")
            
        except Exception as e:
            logger.error(f"Error analyzing schema: {e}")
            schema_status['error'] = str(e)
        
        return schema_status
    
    def backup_existing_data(self) -> bool:
        """Create backup of existing data before migration"""
        logger.info("Creating data backup before migration...")
        
        try:
            backup_tables = ['exchanges', 'sectors', 'industries', 'stocks']
            backup_data = {}
            
            with self.engine.connect() as conn:
                for table_name in backup_tables:
                    if table_name in self.inspector.get_table_names():
                        try:
                            result = conn.execute(text(f"SELECT * FROM {table_name}"))
                            backup_data[table_name] = result.fetchall()
                            logger.info(f"Backed up {len(backup_data[table_name])} rows from {table_name}")
                        except Exception as e:
                            logger.warning(f"Could not backup {table_name}: {e}")
            
            # Store backup data for restoration if needed
            self.backup_data = backup_data
            logger.info("Data backup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return False
    
    def fix_exchanges_table(self) -> bool:
        """Fix exchanges table schema and ensure 'code' column exists"""
        logger.info("Fixing exchanges table...")
        
        try:
            with self.engine.connect() as conn:
                # Check if exchanges table exists
                if 'exchanges' not in self.inspector.get_table_names():
                    # Create exchanges table from scratch
                    logger.info("Creating exchanges table...")
                    conn.execute(text("""
                        CREATE TABLE exchanges (
                            id SERIAL PRIMARY KEY,
                            code VARCHAR(10) UNIQUE NOT NULL,
                            name VARCHAR(100) NOT NULL,
                            timezone VARCHAR(50) DEFAULT 'America/New_York',
                            country VARCHAR(2) DEFAULT 'US',
                            currency VARCHAR(3) DEFAULT 'USD',
                            market_open VARCHAR(5) DEFAULT '09:30',
                            market_close VARCHAR(5) DEFAULT '16:00'
                        )
                    """))
                    conn.commit()
                    
                    # Insert default exchanges
                    conn.execute(text("""
                        INSERT INTO exchanges (code, name, timezone) VALUES
                        ('NYSE', 'New York Stock Exchange', 'America/New_York'),
                        ('NASDAQ', 'NASDAQ Stock Market', 'America/New_York'),
                        ('AMEX', 'NYSE American', 'America/New_York')
                    """))
                    conn.commit()
                    logger.info("Created exchanges table with default data")
                    
                else:
                    # Check if 'code' column exists
                    columns = [col['name'] for col in self.inspector.get_columns('exchanges')]
                    if 'code' not in columns:
                        logger.info("Adding 'code' column to exchanges table...")
                        conn.execute(text("ALTER TABLE exchanges ADD COLUMN code VARCHAR(10)"))
                        conn.commit()
                        
                        # Update existing records if any
                        conn.execute(text("""
                            UPDATE exchanges SET code = 
                                CASE 
                                    WHEN name ILIKE '%nasdaq%' THEN 'NASDAQ'
                                    WHEN name ILIKE '%nyse%' OR name ILIKE '%new york%' THEN 'NYSE'
                                    WHEN name ILIKE '%amex%' OR name ILIKE '%american%' THEN 'AMEX'
                                    ELSE 'UNKNOWN'
                                END
                            WHERE code IS NULL
                        """))
                        conn.commit()
                        
                        # Add unique constraint
                        conn.execute(text("ALTER TABLE exchanges ADD CONSTRAINT uq_exchanges_code UNIQUE (code)"))
                        conn.commit()
                        logger.info("Added 'code' column and updated existing data")
            
            return True
            
        except Exception as e:
            logger.error(f"Error fixing exchanges table: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def fix_stocks_table(self) -> bool:
        """Fix stocks table to use 'ticker' field consistently"""
        logger.info("Fixing stocks table...")
        
        try:
            with self.engine.connect() as conn:
                if 'stocks' not in self.inspector.get_table_names():
                    # Create stocks table from scratch using consolidated model
                    logger.info("Creating stocks table with proper schema...")
                    conn.execute(text("""
                        CREATE TABLE stocks (
                            id SERIAL PRIMARY KEY,
                            ticker VARCHAR(10) UNIQUE NOT NULL,
                            name VARCHAR(255) NOT NULL,
                            exchange_id INTEGER REFERENCES exchanges(id),
                            sector_id INTEGER REFERENCES sectors(id),
                            industry_id INTEGER REFERENCES industries(id),
                            asset_type VARCHAR(20) DEFAULT 'stock',
                            market_cap FLOAT,
                            is_active BOOLEAN DEFAULT true NOT NULL,
                            is_tradeable BOOLEAN DEFAULT true NOT NULL,
                            is_delisted BOOLEAN DEFAULT false NOT NULL,
                            data_quality_score FLOAT DEFAULT 100.0,
                            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            last_price_update TIMESTAMP
                        )
                    """))
                    conn.commit()
                    
                    # Create indexes
                    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_stocks_ticker ON stocks(ticker)"))
                    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_stocks_active_tradeable ON stocks(is_active, is_tradeable)"))
                    conn.commit()
                    logger.info("Created stocks table with proper schema")
                    
                else:
                    # Check current column structure
                    columns = [col['name'] for col in self.inspector.get_columns('stocks')]
                    
                    if 'symbol' in columns and 'ticker' not in columns:
                        logger.info("Renaming 'symbol' column to 'ticker'...")
                        conn.execute(text("ALTER TABLE stocks RENAME COLUMN symbol TO ticker"))
                        conn.commit()
                        logger.info("Renamed 'symbol' to 'ticker'")
                    
                    elif 'ticker' not in columns and 'symbol' not in columns:
                        logger.error("Stocks table missing both 'ticker' and 'symbol' columns")
                        return False
                    
                    # Add missing columns if needed
                    if 'data_quality_score' not in columns:
                        conn.execute(text("ALTER TABLE stocks ADD COLUMN data_quality_score FLOAT DEFAULT 100.0"))
                        conn.commit()
                        logger.info("Added data_quality_score column")
                    
                    if 'is_delisted' not in columns:
                        conn.execute(text("ALTER TABLE stocks ADD COLUMN is_delisted BOOLEAN DEFAULT false"))
                        conn.commit()
                        logger.info("Added is_delisted column")
            
            return True
            
        except Exception as e:
            logger.error(f"Error fixing stocks table: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def create_missing_tables(self) -> bool:
        """Create any missing tables using the consolidated model"""
        logger.info("Creating missing database tables...")
        
        try:
            # Use the consolidated model to create all tables
            Base.metadata.create_all(bind=self.engine, checkfirst=True)
            logger.info("All missing tables created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error creating missing tables: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def validate_schema_fix(self) -> bool:
        """Validate that schema fixes were successful"""
        logger.info("Validating schema fixes...")
        
        try:
            # Re-analyze schema
            updated_status = self.check_current_schema()
            
            # Check that critical issues are resolved
            if updated_status['missing_tables']:
                logger.error(f"Still have missing tables: {updated_status['missing_tables']}")
                return False
            
            if updated_status['column_issues']:
                logger.error(f"Still have column issues: {updated_status['column_issues']}")
                return False
            
            # Test basic operations
            with self.engine.connect() as conn:
                # Test exchanges query that was failing
                result = conn.execute(text("SELECT id FROM exchanges WHERE code = 'NASDAQ'"))
                nasdaq_row = result.fetchone()
                if not nasdaq_row:
                    logger.error("NASDAQ exchange not found after fix")
                    return False
                
                # Test stocks table structure
                result = conn.execute(text("SELECT ticker FROM stocks LIMIT 1"))
                # This should not raise an error
                
                logger.info("Schema validation passed successfully")
                return True
                
        except Exception as e:
            logger.error(f"Schema validation failed: {e}")
            return False
    
    def fix_all_schema_issues(self) -> bool:
        """Run complete schema fix process"""
        logger.info("Starting comprehensive database schema fix...")
        
        try:
            # 1. Analyze current state
            schema_status = self.check_current_schema()
            
            if not schema_status['needs_migration']:
                logger.info("No schema migration needed")
                return True
            
            logger.info(f"Schema issues found: {schema_status}")
            
            # 2. Create backup
            if not self.backup_existing_data():
                logger.error("Failed to create backup - aborting migration")
                return False
            
            # 3. Fix exchanges table
            if not self.fix_exchanges_table():
                logger.error("Failed to fix exchanges table")
                return False
            
            # 4. Create missing tables (sectors, industries, etc.)
            if not self.create_missing_tables():
                logger.error("Failed to create missing tables")
                return False
            
            # 5. Fix stocks table
            if not self.fix_stocks_table():
                logger.error("Failed to fix stocks table")
                return False
            
            # 6. Validate fixes
            if not self.validate_schema_fix():
                logger.error("Schema validation failed after fixes")
                return False
            
            logger.info("✅ All database schema issues fixed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Critical error in schema fix: {e}")
            logger.error(traceback.format_exc())
            return False

def main():
    """Main function to run database schema fixes"""
    logger.info("Database Schema Fix Tool Started")
    
    try:
        # Get database URL from settings or environment
        database_url = getattr(settings, 'DATABASE_URL', os.getenv('DATABASE_URL'))
        
        if not database_url:
            logger.error("DATABASE_URL not found in settings or environment")
            return 1
        
        # Initialize fixer
        fixer = DatabaseSchemaFixer(database_url)
        
        # Run complete fix process
        success = fixer.fix_all_schema_issues()
        
        if success:
            print("\n✅ Database schema fix completed successfully!")
            print("\nNext steps:")
            print("1. Test database operations with the fixed schema")
            print("2. Run your stock loading script to verify the fix")
            print("3. Monitor logs for any remaining issues")
            return 0
        else:
            print("\n❌ Database schema fix failed!")
            print("Check the logs above for detailed error information")
            return 1
            
    except Exception as e:
        logger.error(f"Critical error in main: {e}")
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())