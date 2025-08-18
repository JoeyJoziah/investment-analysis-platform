#!/usr/bin/env python3
"""
Corrected Service Connection Test
Uses the actual passwords that are working in the containers
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
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        
        # Test TimescaleDB
        cursor.execute("SELECT extname FROM pg_extension WHERE extname = 'timescaledb';")
        timescale = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        print("✅ PostgreSQL: Connection successful")
        print(f"   Version: {version.split(',')[0]}")
        print(f"   TimescaleDB: {'✅ Available' if timescale else '⚠️  Not installed'}")
        return True
    except Exception as e:
        print(f"❌ PostgreSQL: Connection failed - {e}")
        return False

def test_redis():
    """Test Redis connection with actual working password"""
    try:
        import redis
        # Use the truncated password that actually works
        r = redis.Redis(
            host='localhost',
            port=6379,
            password='RsYque',  # Actual password in container (truncated due to &)
            db=0,
            decode_responses=True
        )
        pong = r.ping()
        info = r.info()
        version = info.get('redis_version', 'Unknown')
        memory_used = info.get('used_memory_human', 'Unknown')
        
        # Test basic operations
        test_key = "connection_test"
        r.set(test_key, "test_value", ex=60)
        value = r.get(test_key)
        r.delete(test_key)
        
        print("✅ Redis: Connection successful")
        print(f"   Version: {version}")
        print(f"   Memory used: {memory_used}")
        print(f"   Test operation: {'✅ Passed' if value == 'test_value' else '❌ Failed'}")
        return True
    except Exception as e:
        print(f"❌ Redis: Connection failed - {e}")
        return False

def test_elasticsearch():
    """Test Elasticsearch connection (no auth required)"""
    try:
        from elasticsearch import Elasticsearch
        es = Elasticsearch(
            [{'host': 'localhost', 'port': 9200, 'scheme': 'http'}],
            verify_certs=False,
            timeout=10
        )
        
        health = es.cluster.health()
        
        # Test basic indexing
        test_doc = {
            'timestamp': '2025-08-08T19:30:00',
            'test': True,
            'message': 'Connection test'
        }
        
        # Index test document
        response = es.index(index='test_index', body=test_doc)
        
        # Search for document
        search_response = es.search(
            index='test_index', 
            body={'query': {'match': {'test': True}}}
        )
        
        # Clean up
        es.indices.delete(index='test_index', ignore=[400, 404])
        
        print("✅ Elasticsearch: Connection successful")
        print(f"   Cluster status: {health.get('status', 'unknown')}")
        print(f"   Cluster name: {health.get('cluster_name', 'unknown')}")
        print(f"   Nodes: {health.get('number_of_nodes', 0)}")
        print(f"   Test indexing: {'✅ Passed' if response.get('result') == 'created' else '❌ Failed'}")
        return True
    except Exception as e:
        print(f"❌ Elasticsearch: Connection failed - {e}")
        return False

def test_comprehensive_database():
    """Test comprehensive database operations"""
    try:
        import psycopg2
        from datetime import datetime
        
        conn = psycopg2.connect(
            host='localhost',
            port=5432,
            database='investment_db',
            user='investment_user',
            password='9v1g^OV9XUwzUP6cEgCYgNOE'
        )
        cursor = conn.cursor()
        
        # Create test table with time-series structure
        cursor.execute("""
            DROP TABLE IF EXISTS test_stock_prices;
            CREATE TABLE test_stock_prices (
                time TIMESTAMPTZ NOT NULL,
                symbol VARCHAR(10) NOT NULL,
                price DECIMAL(10,2) NOT NULL,
                volume BIGINT NOT NULL DEFAULT 0
            );
        """)
        
        # Try to create TimescaleDB hypertable
        try:
            cursor.execute("SELECT create_hypertable('test_stock_prices', 'time', if_not_exists => TRUE);")
            timescale_working = True
        except:
            timescale_working = False
        
        # Insert test data
        test_data = [
            ('2025-08-08 10:00:00', 'AAPL', 150.00, 1000000),
            ('2025-08-08 10:01:00', 'AAPL', 150.50, 1100000),
            ('2025-08-08 10:00:00', 'GOOGL', 2800.00, 500000)
        ]
        
        cursor.executemany(
            "INSERT INTO test_stock_prices (time, symbol, price, volume) VALUES (%s, %s, %s, %s)",
            test_data
        )
        
        # Query test data
        cursor.execute("SELECT symbol, COUNT(*), AVG(price) FROM test_stock_prices GROUP BY symbol ORDER BY symbol")
        results = cursor.fetchall()
        
        # Clean up
        cursor.execute("DROP TABLE test_stock_prices")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print("✅ Database (Comprehensive): All operations successful")
        print(f"   TimescaleDB: {'✅ Working' if timescale_working else '⚠️  Standard PostgreSQL'}")
        print(f"   Test data: {len(test_data)} records inserted")
        print(f"   Query results: {len(results)} stock symbols processed")
        
        return True
        
    except Exception as e:
        print(f"❌ Database (Comprehensive): Failed - {e}")
        return False

def show_connection_info():
    """Show connection information for reference"""
    print("\n📋 CONNECTION INFORMATION")
    print("=" * 50)
    print("PostgreSQL:")
    print("  Host: localhost:5432")
    print("  Database: investment_db") 
    print("  User: investment_user")
    print("  Password: 9v1g^OV9XUwzUP6cEgCYgNOE")
    print("")
    print("Redis:")
    print("  Host: localhost:6379")
    print("  Password: RsYque (Note: truncated from original due to shell interpretation)")
    print("  DB: 0")
    print("")
    print("Elasticsearch:")
    print("  Host: localhost:9200")
    print("  Authentication: None (security disabled)")
    print("=" * 50)

def main():
    print("🔧 INVESTMENT ANALYSIS APP - SERVICE CONNECTION TEST")
    print("=" * 65)
    
    tests = [
        ("PostgreSQL", test_postgresql),
        ("Redis", test_redis),
        ("Elasticsearch", test_elasticsearch),
        ("Database Comprehensive", test_comprehensive_database)
    ]
    
    results = []
    
    for name, test_func in tests:
        print(f"\n🧪 Testing {name}...")
        try:
            success = test_func()
            results.append(success)
        except ImportError as e:
            print(f"❌ {name}: Missing library - {e}")
            print(f"   Install with: pip install {e.name}")
            results.append(False)
        except Exception as e:
            print(f"❌ {name}: Unexpected error - {e}")
            results.append(False)
    
    # Results summary
    print("\n" + "=" * 65)
    print("🏆 TEST RESULTS SUMMARY")
    print("=" * 65)
    
    passed = sum(results)
    total = len(results)
    
    test_names = ["PostgreSQL", "Redis", "Elasticsearch", "Comprehensive DB"]
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{name}: {status}")
    
    print(f"\nOVERALL: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 EXCELLENT! All database and service connections are working perfectly!")
        print("\n✅ Your investment analysis app is ready for:")
        print("   • Data storage and time-series operations (PostgreSQL + TimescaleDB)")
        print("   • High-performance caching (Redis)")
        print("   • Full-text search and analytics (Elasticsearch)")
        print("\n🚀 You can now start the full application stack!")
        
    else:
        print(f"\n⚠️  {total - passed} service(s) need attention")
        
        if not results[0]:  # PostgreSQL
            print("\n🔧 PostgreSQL troubleshooting:")
            print("   • Ensure Docker containers are running: docker-compose up -d")
            print("   • Check logs: docker logs investment_db")
            
        if not results[1]:  # Redis
            print("\n🔧 Redis troubleshooting:")  
            print("   • Check container: docker logs investment_cache")
            print("   • Try restarting Redis: docker-compose restart redis")
            
        if not results[2]:  # Elasticsearch
            print("\n🔧 Elasticsearch troubleshooting:")
            print("   • Elasticsearch takes time to start (wait 2-3 minutes)")
            print("   • Check logs: docker logs investment_search")
            print("   • Increase memory if needed: docker stats")
    
    # Show connection info for reference
    show_connection_info()
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)