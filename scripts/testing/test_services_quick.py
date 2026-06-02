#!/usr/bin/env python3
"""
Quick Service Connection Test
Simple script to test database and service connections
"""

import os
import sys

def test_postgresql():
    """Test PostgreSQL connection"""
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
        cursor.execute("SELECT 1;")
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        print("✅ PostgreSQL: Connection successful")
        return True
    except Exception as e:
        print(f"❌ PostgreSQL: Connection failed - {e}")
        return False

def test_redis():
    """Test Redis connection"""
    try:
        import redis
        r = redis.Redis(
            host='localhost',
            port=6379,
            password='RsYque&Xh%TUD*Nv^7k7B8X3',
            db=0
        )
        r.ping()
        print("✅ Redis: Connection successful")
        return True
    except Exception as e:
        print(f"❌ Redis: Connection failed - {e}")
        return False

def main():
    print("Testing service connections...")
    print("=" * 40)
    
    tests = [test_postgresql, test_redis]
    results = []
    
    for test in tests:
        results.append(test())
    
    print("=" * 40)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} services connected successfully")
    
    if passed == total:
        print("🎉 All services are working!")
    else:
        print("⚠️  Some services failed - check Docker containers")

if __name__ == "__main__":
    main()