#!/usr/bin/env python3
"""
Docker Container Connection Test
Tests connections using Docker service names (for container-to-container communication)
"""

import os
import sys

def test_postgresql_docker():
    """Test PostgreSQL connection using Docker service name"""
    try:
        import psycopg2
        # Use 'postgres' service name instead of localhost
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
        print(f"   Version: {version[:80]}...")
        return True
    except Exception as e:
        print(f"❌ PostgreSQL (Docker): Connection failed - {e}")
        return False

def test_redis_docker():
    """Test Redis connection using Docker service name"""
    try:
        import redis
        # Use 'redis' service name instead of localhost
        r = redis.Redis(
            host='redis',  # Docker service name
            port=6379,
            password='RsYque&Xh%TUD*Nv^7k7B8X3',
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
        # Use 'elasticsearch' service name instead of localhost
        es = Elasticsearch(
            [{'host': 'elasticsearch', 'port': 9200, 'scheme': 'http'}],  # Docker service name
            http_auth=('elastic', '4Bx+UM1CdSiEbMlQueRVvda+A4fLzCRsyuHUHbv5wMw='),
            verify_certs=False,
            timeout=10
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

def main():
    env_status = check_environment()
    print(f"Environment: {env_status}")
    print("Testing Docker service connections...")
    print("=" * 50)
    
    tests = [
        ("PostgreSQL", test_postgresql_docker),
        ("Redis", test_redis_docker), 
        ("Elasticsearch", test_elasticsearch_docker)
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            success = test_func()
            results.append(success)
        except ImportError as e:
            print(f"❌ {name}: Missing library - {e}")
            results.append(False)
        except Exception as e:
            print(f"❌ {name}: Unexpected error - {e}")
            results.append(False)
    
    print("=" * 50)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} Docker services connected successfully")
    
    if passed == total:
        print("🎉 All Docker services are working!")
    else:
        print("⚠️  Some Docker services failed")
        print("Try running: docker-compose up -d")

if __name__ == "__main__":
    main()