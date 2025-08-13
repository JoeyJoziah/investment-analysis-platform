#!/usr/bin/env python3
"""
Fixed Docker Container Connection Test
Tests connections using corrected Docker service names and passwords
"""

import os
import sys

def test_postgresql_docker():
    """Test PostgreSQL connection using Docker service name"""
    try:
        import psycopg2
        conn = psycopg2.connect(
            host='postgres',  # Docker service name
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
        print("✅ PostgreSQL (Docker): Connection successful")
        print(f"   Version: {version.split(',')[0]}")
        return True
    except Exception as e:
        print(f"❌ PostgreSQL (Docker): Connection failed - {e}")
        return False

def test_redis_docker():
    """Test Redis connection using Docker service name and correct password"""
    try:
        import redis
        r = redis.Redis(
            host='redis',  # Docker service name
            port=6379,
            password='RsYque',  # Actual working password (truncated)
            db=0,
            decode_responses=True
        )
        pong = r.ping()
        info = r.info()
        version = info.get('redis_version', 'Unknown')
        print("✅ Redis (Docker): Connection successful")
        print(f"   Version: {version}, Ping: {pong}")
        return True
    except Exception as e:
        print(f"❌ Redis (Docker): Connection failed - {e}")
        return False

def test_elasticsearch_docker():
    """Test Elasticsearch connection using Docker service name"""
    try:
        from elasticsearch import Elasticsearch
        es = Elasticsearch(
            [{'host': 'elasticsearch', 'port': 9200, 'scheme': 'http'}],  # Docker service name
            verify_certs=False,
            timeout=15
        )
        health = es.cluster.health()
        print("✅ Elasticsearch (Docker): Connection successful")
        print(f"   Status: {health.get('status')}, Nodes: {health.get('number_of_nodes')}")
        return True
    except Exception as e:
        print(f"❌ Elasticsearch (Docker): Connection failed - {e}")
        return False

def check_environment():
    """Check if we're running inside a Docker container"""
    try:
        # Check for .dockerenv file (common indicator)
        if os.path.exists('/.dockerenv'):
            return "Inside Docker container"
        
        # Check cgroup (another indicator)
        with open('/proc/1/cgroup', 'r') as f:
            content = f.read()
            if 'docker' in content or 'containerd' in content:
                return "Inside Docker container"
        
        return "Outside Docker container"
    except:
        return "Environment unknown"

def show_docker_configuration():
    """Show Docker networking configuration"""
    print("\n🐳 DOCKER CONFIGURATION")
    print("=" * 40)
    print("Service Names (for container-to-container communication):")
    print("  PostgreSQL: postgres:5432")
    print("  Redis: redis:6379")  
    print("  Elasticsearch: elasticsearch:9200")
    print("")
    print("Host Access (from outside Docker):")
    print("  PostgreSQL: localhost:5432")
    print("  Redis: localhost:6379")
    print("  Elasticsearch: localhost:9200")
    print("=" * 40)

def main():
    env_status = check_environment()
    print(f"🌍 Environment: {env_status}")
    print("🐳 Testing Docker service connections...")
    print("=" * 55)
    
    tests = [
        ("PostgreSQL", test_postgresql_docker),
        ("Redis", test_redis_docker), 
        ("Elasticsearch", test_elasticsearch_docker)
    ]
    
    results = []
    
    for name, test_func in tests:
        print(f"\n🧪 Testing {name}...")
        try:
            success = test_func()
            results.append(success)
        except ImportError as e:
            print(f"❌ {name}: Missing library - {e}")
            results.append(False)
        except Exception as e:
            print(f"❌ {name}: Unexpected error - {e}")
            results.append(False)
    
    print("\n" + "=" * 55)
    passed = sum(results)
    total = len(results)
    print(f"📊 RESULTS: {passed}/{total} Docker services connected successfully")
    
    if passed == total:
        print("\n🎉 All Docker services are working perfectly!")
        print("✅ Container-to-container communication is functional")
    else:
        print(f"\n⚠️  {total - passed} Docker service(s) failed")
        print("💡 This is expected if running outside Docker containers")
        print("   Use test_services_corrected.py for host-based testing")
    
    show_docker_configuration()
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)