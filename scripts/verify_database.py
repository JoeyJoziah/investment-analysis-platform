#!/usr/bin/env python3
"""
Simple database verification script to check if schema is correct
"""

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, text, inspect
from backend.config.settings import settings

def verify_database():
    """Verify database schema is correct for stock loading"""
    
    print("🔍 Database Schema Verification")
    print("=" * 50)
    
    # Connect to database
    database_url = settings.DATABASE_URL
    engine = create_engine(database_url)
    inspector = inspect(engine)
    
    # Get all tables
    tables = inspector.get_table_names()
    print(f"\n✅ Found {len(tables)} tables in database")
    
    # Check critical tables
    required_tables = ['exchanges', 'sectors', 'industries', 'stocks']
    missing_tables = []
    
    for table in required_tables:
        if table in tables:
            print(f"  ✅ {table} table exists")
        else:
            print(f"  ❌ {table} table MISSING")
            missing_tables.append(table)
    
    if missing_tables:
        print(f"\n❌ Missing tables: {missing_tables}")
        return False
    
    # Check exchanges table has 'code' column
    print("\n📋 Checking exchanges table structure...")
    exchanges_columns = {col['name']: col['type'] for col in inspector.get_columns('exchanges')}
    
    if 'code' in exchanges_columns:
        print(f"  ✅ 'code' column exists in exchanges table")
    else:
        print(f"  ❌ 'code' column MISSING in exchanges table")
        print(f"     Available columns: {list(exchanges_columns.keys())}")
        return False
    
    # Check stocks table has 'ticker' column
    print("\n📋 Checking stocks table structure...")
    stocks_columns = {col['name']: col['type'] for col in inspector.get_columns('stocks')}
    
    if 'ticker' in stocks_columns:
        print(f"  ✅ 'ticker' column exists in stocks table")
    elif 'symbol' in stocks_columns:
        print(f"  ⚠️  stocks table uses 'symbol' instead of 'ticker'")
        return False
    else:
        print(f"  ❌ Neither 'ticker' nor 'symbol' column found in stocks table")
        return False
    
    # Test actual query that was failing
    print("\n🧪 Testing critical queries...")
    with engine.connect() as conn:
        try:
            # Test the exact query that was failing
            result = conn.execute(text("SELECT id FROM exchanges WHERE code = :code"), {'code': 'NASDAQ'})
            print(f"  ✅ Can query exchanges by 'code' column")
        except Exception as e:
            print(f"  ❌ Cannot query exchanges by 'code': {e}")
            return False
        
        try:
            # Test ticker query
            result = conn.execute(text("SELECT id FROM stocks WHERE ticker = :ticker LIMIT 1"), {'ticker': 'TEST'})
            print(f"  ✅ Can query stocks by 'ticker' column")
        except Exception as e:
            print(f"  ❌ Cannot query stocks by 'ticker': {e}")
            return False
    
    print("\n" + "=" * 50)
    print("✅ Database schema is READY for stock loading!")
    print("\nThe database has:")
    print("  • exchanges table with 'code' column")
    print("  • stocks table with 'ticker' column")
    print("  • All required tables present")
    print("\nYou can now run the background_loader_enhanced.py script!")
    
    return True

if __name__ == "__main__":
    success = verify_database()
    sys.exit(0 if success else 1)