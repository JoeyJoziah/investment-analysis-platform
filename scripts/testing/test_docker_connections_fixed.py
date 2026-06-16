#!/usr/bin/env python3
"""
Fixed Docker Container Connection Test
Tests connections using corrected Docker service names and passwords
"""

import os
import sys


def _require_secret(env_var: str, non_prod_fallback: str) -> str:
    """Require a secret in production; allow a marked non-prod fallback otherwise.

    Mirrors backend.security.secrets_manager's production-vs-non-prod handling.
    """
    value = os.getenv(env_var)
    if value:
        return value
    if os.getenv('ENVIRONMENT', 'development').lower() == 'production':
        raise RuntimeError(
            f"{env_var} must be set in production (no hardcoded fallback allowed)."
        )
    return non_prod_fallback


def test_postgresql_docker():
    """Test PostgreSQL connection using Docker service name"""
    try:
        import psycopg2
        conn = psycopg2.connect(
            host=os.getenv('POSTGRES_HOST', 'postgres'),  # Docker service name
            port=int(os.getenv('POSTGRES_PORT', 5432)),
            database=os.getenv('POSTGRES_DB', 'investment_db'),
            user=os.getenv('POSTGRES_USER', 'investment_user'),
            # NON-PROD FALLBACK ONLY — production must set POSTGRES_PASSWORD.
            password=_require_secret('POSTGRES_PASSWORD', 'CHANGE_ME_LOCAL_DEV')
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
            host=os.getenv('REDIS_HOST', 'redis'),  # Docker service name
            port=int(os.getenv('REDIS_PORT', 6379)),
            # NON-PROD FALLBACK ONLY — production must set REDIS_PASSWORD.
            password=_require_secret('REDIS_PASSWORD', 'CHANGE_ME_LOCAL_DEV'),
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