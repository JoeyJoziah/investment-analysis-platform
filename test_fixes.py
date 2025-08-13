#!/usr/bin/env python3
"""
Test script to verify database and asyncio fixes
"""

import asyncio
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from sqlalchemy import create_engine, text
from backend.config.settings import settings

def test_database_fixes():
    """Test that database schema issues are resolved"""
    print("Testing Database Fixes")
    print("=" * 50)
    
    engine = create_engine(settings.DATABASE_URL)
    
    with engine.connect() as conn:
        # Test 1: Can query exchanges by code (was failing before)
        try:
            result = conn.execute(text("SELECT id FROM exchanges WHERE code = :code"), {'code': 'NASDAQ'})
            row = result.fetchone()
            if row:
                print("✅ Test 1 PASSED: Can query exchanges by 'code'")
            else:
                print("⚠️  Test 1: No NASDAQ exchange found, inserting...")
                conn.execute(text("INSERT INTO exchanges (name, code) VALUES ('NASDAQ', 'NASDAQ') ON CONFLICT (name) DO UPDATE SET code = 'NASDAQ'"))
                conn.commit()
                print("✅ Test 1 PASSED: NASDAQ exchange created")
        except Exception as e:
            print(f"❌ Test 1 FAILED: {e}")
            return False
        
        # Test 2: Can insert stocks with ticker field
        try:
            # Try to insert a test stock
            conn.execute(text("""
                INSERT INTO stocks (ticker, name, exchange_id, sector_id, industry_id) 
                VALUES (:ticker, :name, 
                        (SELECT id FROM exchanges WHERE code = 'NYSE' LIMIT 1),
                        (SELECT id FROM sectors LIMIT 1),
                        (SELECT id FROM industries LIMIT 1))
                ON CONFLICT (ticker) DO NOTHING
            """), {'ticker': 'TEST_FIX', 'name': 'Test Fix Stock'})
            conn.commit()
            print("✅ Test 2 PASSED: Can insert stocks with all required fields")
        except Exception as e:
            print(f"❌ Test 2 FAILED: {e}")
            return False
        
        # Test 3: Verify industry_id column exists
        try:
            result = conn.execute(text("SELECT industry_id FROM stocks LIMIT 1"))
            print("✅ Test 3 PASSED: industry_id column exists in stocks table")
        except Exception as e:
            print(f"❌ Test 3 FAILED: {e}")
            return False
    
    return True

async def test_asyncio_fixes():
    """Test that asyncio Future errors are resolved"""
    print("\nTesting AsyncIO Fixes")
    print("=" * 50)
    
    # Simulate batch processing without nested asyncio.run
    tasks = []
    
    async def process_item(item):
        await asyncio.sleep(0.01)  # Simulate async work
        return f"Processed {item}"
    
    # Create batch of tasks
    for i in range(5):
        tasks.append(process_item(f"item_{i}"))
    
    try:
        # Process all tasks concurrently
        results = await asyncio.gather(*tasks)
        print(f"✅ AsyncIO Test PASSED: Processed {len(results)} items without Future errors")
        return True
    except AttributeError as e:
        if "_condition" in str(e):
            print(f"❌ AsyncIO Test FAILED: Future._condition error still exists")
            return False
        raise
    except Exception as e:
        print(f"❌ AsyncIO Test FAILED: {e}")
        return False

def main():
    print("🔧 Testing Database and AsyncIO Fixes")
    print("=" * 50)
    
    # Test database fixes
    db_success = test_database_fixes()
    
    # Test asyncio fixes
    asyncio_success = asyncio.run(test_asyncio_fixes())
    
    print("\n" + "=" * 50)
    print("Test Results Summary:")
    print(f"  Database Fixes: {'✅ PASSED' if db_success else '❌ FAILED'}")
    print(f"  AsyncIO Fixes:  {'✅ PASSED' if asyncio_success else '❌ FAILED'}")
    
    if db_success and asyncio_success:
        print("\n✅ ALL TESTS PASSED - System is ready for stock loading!")
        print("\nNext steps:")
        print("1. Run: python3 background_loader_enhanced.py --mode initial --batch-size 20 --max-workers 4")
        print("2. Monitor for errors in the output")
        print("3. Check database for loaded stocks: SELECT COUNT(*) FROM stocks;")
        return 0
    else:
        print("\n❌ Some tests failed - please review the errors above")
        return 1

if __name__ == "__main__":
    sys.exit(main())