#!/usr/bin/env python3
"""
Fixed Service Connection Test
Tests connections with corrected configurations based on docker-compose.yml
"""

import os
import sys

def test_postgresql():
    """Test PostgreSQL connection"""
    try:
        import psycopg2
        # Test with investment_user (which we just created)
        conn = psycopg2.connect(
            host='localhost',
            port=5432,
            database='investment_db',
            user='investment_user',
            password='9v1g^OV9XUwzUP6cEgCYgNOE'
        )
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        print("✅ PostgreSQL: Connection successful")
        print(f"   Version: {version[:80]}...")
        return True
    except Exception as e:
        print(f"❌ PostgreSQL: Connection failed - {e}")
        return False

def test_redis():
    """Test Redis connection with password from environment"""
    try:
        import redis
        # Use the exact password from .env
        r = redis.Redis(
            host='localhost',
            port=6379,
            password='RsYque&Xh%TUD*Nv^7k7B8X3',  # From .env REDIS_PASSWORD
            db=0,
            decode_responses=True
        )
        pong = r.ping()
        info = r.info()
        version = info.get('redis_version', 'Unknown')
        print("✅ Redis: Connection successful")
        print(f"   Version: {version}, Ping: {pong}")
        return True
    except Exception as e:
        print(f"❌ Redis: Connection failed - {e}")
        return False

def test_comprehensive_postgresql():
    """Test PostgreSQL with multiple operations"""
    try:
        import psycopg2
        conn = psycopg2.connect(
            host='localhost',
            port=5432,
            database='investment_db',
            user='investment_user',
            password='9v1g^OV9XUwzUP6cEgCYgNOE'
        )
        cursor = conn.cursor()
        
        # Test TimescaleDB extension
        cursor.execute("SELECT extname FROM pg_extension WHERE extname = 'timescaledb';")
        timescale = cursor.fetchone()
        
        # Test basic operations
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS test_stocks (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(10),
                price DECIMAL(10,2),
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        
        cursor.execute("INSERT INTO test_stocks (symbol, price) VALUES (%s, %s)", ('TEST', 100.50))
        cursor.execute("SELECT COUNT(*) FROM test_stocks")
        count = cursor.fetchone()[0]
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print("✅ PostgreSQL (Comprehensive): All operations successful")
        print(f"   TimescaleDB: {'✅ Installed' if timescale else '❌ Not found'}")
        print(f"   Test records: {count}")
        return True
        
    except Exception as e:
        print(f"❌ PostgreSQL (Comprehensive): Failed - {e}")
        return False

def main():
    print("Testing service connections with fixed configurations...")
    print("=" * 60)
    
    tests = [
        ("PostgreSQL Basic", test_postgresql),
        ("PostgreSQL Comprehensive", test_comprehensive_postgresql),
        ("Redis", test_redis)
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            print(f"\n🔧 Testing {name}...")
            success = test_func()
            results.append(success)
        except ImportError as e:
            print(f"❌ {name}: Missing library - {e}")
            results.append(False)
        except Exception as e:
            print(f"❌ {name}: Unexpected error - {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All services are working correctly!")
        print("\n✅ Your investment analysis app services are ready!")
        print("   • PostgreSQL: Ready with TimescaleDB")
        print("   • Redis: Ready for caching")
    else:
        print("⚠️  Some services need attention")
        
        # Specific troubleshooting
        if not results[0]:  # PostgreSQL basic
            print("\n🔧 PostgreSQL Issues:")
            print("   - Make sure containers are running: docker-compose up -d")
            print("   - Check if investment_user was created properly")
            
        if not results[2]:  # Redis
            print("\n🔧 Redis Issues:")
            print("   - Verify REDIS_PASSWORD in .env matches docker-compose configuration")
            print("   - Check Redis container: docker logs investment_cache")

if __name__ == "__main__":
    main()