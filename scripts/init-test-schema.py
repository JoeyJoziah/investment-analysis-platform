#!/usr/bin/env python3
"""
Initialize Test Database Schema
Creates all tables directly from SQLAlchemy models, bypassing migrations
"""

import asyncio
import os
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy.ext.asyncio import create_async_engine
from backend.models.unified_models import Base

async def init_test_schema():
    """Create all tables in test database"""

    # Test database URL
    database_url = "postgresql+asyncpg://postgres:CEP4j9ZHgd352ONsrj8VgKRCwoOR8Yp@localhost:5432/investment_db_test"

    print("=" * 60)
    print("Initializing Test Database Schema")
    print("=" * 60)
    print(f"\nDatabase: {database_url.split('@')[1]}")
    print(f"Creating {len(Base.metadata.tables)} tables...")
    print()

    # Create engine
    engine = create_async_engine(database_url, echo=False)

    try:
        # Drop all existing tables (clean slate)
        print("[1/3] Dropping existing tables...")
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        print("✓ Existing tables dropped")
        print()

        # Create all tables from models
        print("[2/3] Creating tables from models...")
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        print("✓ All tables created")
        print()

        # Verify tables
        print("[3/3] Verifying schema...")
        async with engine.begin() as conn:
            result = await conn.execute(text("""
                SELECT tablename
                FROM pg_tables
                WHERE schemaname = 'public'
                ORDER BY tablename
            """))
            tables = [row[0] for row in result]

        print(f"✓ Schema verified - {len(tables)} tables created:")
        for table in tables[:10]:  # Show first 10
            print(f"  - {table}")
        if len(tables) > 10:
            print(f"  ... and {len(tables) - 10} more")
        print()

        print("=" * 60)
        print("✓ Test Database Schema Ready")
        print("=" * 60)
        print()
        print("You can now run tests:")
        print("  pytest backend/tests/ -v")
        print()

    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)
    finally:
        await engine.dispose()

if __name__ == "__main__":
    # Import text for SQL queries
    from sqlalchemy import text
    asyncio.run(init_test_schema())
