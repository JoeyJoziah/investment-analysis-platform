#!/usr/bin/env python3
"""
Database Connectivity Verification Script
Verifies PostgreSQL/TimescaleDB connection and initializes database if needed.
"""

import os
import sys
import time
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from dotenv import load_dotenv
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

class DatabaseVerifier:
    def __init__(self):
        self.db_config = {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': int(os.getenv('POSTGRES_PORT', 5432)),
            'database': os.getenv('POSTGRES_DB', 'investment_db'),
            'user': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD', '9v1g^OV9XUwzUP6cEgCYgNOE')
        }
        
    def check_connection(self, max_retries=5, retry_delay=5):
        """Check database connection with retries"""
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempting database connection (attempt {attempt + 1}/{max_retries})...")
                conn = psycopg2.connect(**self.db_config)
                conn.close()
                logger.info("✅ Database connection successful!")
                return True
            except psycopg2.OperationalError as e:
                logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
        
        logger.error("❌ Failed to connect to database after all retries")
        return False
    
    def create_database_if_not_exists(self):
        """Create database if it doesn't exist"""
        try:
            # Connect to postgres database first
            admin_config = self.db_config.copy()
            admin_config['database'] = 'postgres'
            
            conn = psycopg2.connect(**admin_config)
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()
            
            # Check if database exists
            cursor.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s",
                (self.db_config['database'],)
            )
            exists = cursor.fetchone()
            
            if not exists:
                logger.info(f"Creating database '{self.db_config['database']}'...")
                cursor.execute(f"CREATE DATABASE {self.db_config['database']}")
                logger.info("✅ Database created successfully!")
            else:
                logger.info(f"✅ Database '{self.db_config['database']}' already exists")
            
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Error creating database: {e}")
            return False
    
    def enable_timescaledb(self):
        """Enable TimescaleDB extension"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Check if TimescaleDB is already enabled
            cursor.execute(
                "SELECT 1 FROM pg_extension WHERE extname = 'timescaledb'"
            )
            exists = cursor.fetchone()
            
            if not exists:
                logger.info("Enabling TimescaleDB extension...")
                cursor.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE")
                conn.commit()
                logger.info("✅ TimescaleDB extension enabled!")
            else:
                logger.info("✅ TimescaleDB extension already enabled")
            
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Error enabling TimescaleDB: {e}")
            return False
    
    def verify_tables(self):
        """Verify that essential tables exist"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Get list of tables
            cursor.execute("""
                SELECT tablename 
                FROM pg_tables 
                WHERE schemaname = 'public'
                ORDER BY tablename
            """)
            tables = [row[0] for row in cursor.fetchall()]
            
            if tables:
                logger.info(f"✅ Found {len(tables)} tables in database:")
                for table in tables[:10]:  # Show first 10 tables
                    logger.info(f"  - {table}")
                if len(tables) > 10:
                    logger.info(f"  ... and {len(tables) - 10} more")
            else:
                logger.warning("⚠️ No tables found in database. Run migrations to create schema.")
            
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Error verifying tables: {e}")
            return False
    
    def test_query(self):
        """Run a simple test query"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute("SELECT version()")
            version = cursor.fetchone()[0]
            logger.info(f"✅ Database version: {version[:50]}...")
            
            cursor.execute("SELECT current_database(), current_user")
            db_info = cursor.fetchone()
            logger.info(f"✅ Connected to database '{db_info[0]}' as user '{db_info[1]}'")
            
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Error running test query: {e}")
            return False
    
    def run_verification(self):
        """Run complete database verification"""
        logger.info("=" * 60)
        logger.info("Starting Database Verification")
        logger.info("=" * 60)
        
        # Check if running in Docker
        if os.path.exists('/.dockerenv'):
            logger.info("Running inside Docker container")
            self.db_config['host'] = 'postgres'  # Use Docker service name
        
        logger.info(f"Database Host: {self.db_config['host']}")
        logger.info(f"Database Port: {self.db_config['port']}")
        logger.info(f"Database Name: {self.db_config['database']}")
        logger.info(f"Database User: {self.db_config['user']}")
        logger.info("-" * 60)
        
        steps = [
            ("Creating database if not exists", self.create_database_if_not_exists),
            ("Checking database connection", self.check_connection),
            ("Enabling TimescaleDB", self.enable_timescaledb),
            ("Running test query", self.test_query),
            ("Verifying tables", self.verify_tables),
        ]
        
        all_passed = True
        for step_name, step_func in steps:
            logger.info(f"\n{step_name}...")
            if not step_func():
                all_passed = False
                logger.error(f"❌ {step_name} failed!")
                break
        
        logger.info("\n" + "=" * 60)
        if all_passed:
            logger.info("✅ DATABASE VERIFICATION SUCCESSFUL!")
            logger.info("All database checks passed.")
        else:
            logger.error("❌ DATABASE VERIFICATION FAILED!")
            logger.error("Please check your database configuration and ensure PostgreSQL is running.")
            logger.info("\nTroubleshooting tips:")
            logger.info("1. Ensure Docker services are running: docker-compose up -d postgres")
            logger.info("2. Check Docker logs: docker-compose logs postgres")
            logger.info("3. Verify .env file has correct database credentials")
            logger.info("4. Try connecting manually: psql -h localhost -U postgres -d investment_db")
        
        logger.info("=" * 60)
        return all_passed


def main():
    verifier = DatabaseVerifier()
    success = verifier.run_verification()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()