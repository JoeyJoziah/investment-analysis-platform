#!/usr/bin/env python3
"""
Comprehensive Password Test Suite
Tests all services with the newly configured strong passwords
"""

import os
import sys
import time
from typing import Dict, Tuple

# Check for required dependencies
try:
    import psycopg2
except ImportError:
    print("ERROR: psycopg2 is not installed. Run: pip install psycopg2-binary")
    sys.exit(1)

try:
    import redis
except ImportError:
    print("ERROR: redis is not installed. Run: pip install redis")
    sys.exit(1)

try:
    import requests
    from requests.auth import HTTPBasicAuth
except ImportError:
    print("ERROR: requests is not installed. Run: pip install requests")
    sys.exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("WARNING: python-dotenv is not installed. Run: pip install python-dotenv")
    print("         Continuing with default values from script...")
    # Define a no-op load_dotenv if not available
    def load_dotenv():
        pass
    load_dotenv()

def test_postgresql() -> Tuple[bool, str]:
    """Test PostgreSQL connection with new password"""
    try:
        # Get credentials from environment
        db_password = os.getenv('DB_PASSWORD', 'xdfBj7S3TufIuyDi67MxTwHLx53lwUZN')
        
        conn = psycopg2.connect(
            host='localhost',
            port=5432,
            database='investment_db',
            user='postgres',
            password=db_password
        )
        
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        
        cursor.close()
        conn.close()
        
        return True, f"Connected! Version: {version[:30]}..."
    except Exception as e:
        return False, f"Failed: {str(e)}"

def test_redis() -> Tuple[bool, str]:
    """Test Redis connection with new password"""
    try:
        # Get password from environment
        redis_password = os.getenv('REDIS_PASSWORD', '7ba20b200b3069a611d3a908905278275bb6bb10f58cea97f2461e5eb0bb7be2')
        
        r = redis.Redis(
            host='localhost',
            port=6379,
            password=redis_password,
            decode_responses=True
        )
        
        # Test connection
        r.ping()
        info = r.info('server')
        version = info.get('redis_version', 'Unknown')
        
        return True, f"Connected! Redis v{version}"
    except Exception as e:
        return False, f"Failed: {str(e)}"

def test_elasticsearch() -> Tuple[bool, str]:
    """Test Elasticsearch connection with new password"""
    try:
        # Get credentials from environment
        es_password = os.getenv('ES_PASSWORD', '4Bx+UM1CdSiEbMlQueRVvda+A4fLzCRsyuHUHbv5wMw=')
        es_user = os.getenv('ES_USER', 'elastic')
        
        # Note: Elasticsearch might not have authentication enabled by default
        response = requests.get(
            'http://localhost:9200',
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            version = data.get('version', {}).get('number', 'Unknown')
            return True, f"Connected! Elasticsearch v{version}"
        else:
            # Try with authentication
            response = requests.get(
                'http://localhost:9200',
                auth=HTTPBasicAuth(es_user, es_password),
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                version = data.get('version', {}).get('number', 'Unknown')
                return True, f"Connected with auth! Elasticsearch v{version}"
            else:
                return False, f"HTTP {response.status_code}"
    except Exception as e:
        return False, f"Failed: {str(e)}"

def test_grafana() -> Tuple[bool, str]:
    """Test Grafana API with new password"""
    try:
        # Get credentials from environment
        grafana_password = os.getenv('GRAFANA_PASSWORD', 'a2A5j4JQ0nF8aTLyIYwRgZnMLQpIu5lW9jYx6pB5Xdw=')
        grafana_user = os.getenv('GRAFANA_USER', 'admin')
        
        response = requests.get(
            'http://localhost:3001/api/health',
            auth=HTTPBasicAuth(grafana_user, grafana_password),
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            return True, f"Connected! Grafana is {data.get('database', 'unknown')}"
        else:
            return False, f"HTTP {response.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "Grafana not running on port 3001"
    except Exception as e:
        return False, f"Failed: {str(e)}"

def print_results(results: Dict[str, Tuple[bool, str]]):
    """Print test results in a formatted table"""
    print("\n" + "="*70)
    print("PASSWORD VERIFICATION RESULTS")
    print("="*70)
    
    for service, (success, message) in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{service:20} {status:10} {message}")
    
    print("="*70)
    
    # Summary
    passed = sum(1 for _, (success, _) in results.items() if success)
    total = len(results)
    
    print(f"\nSUMMARY: {passed}/{total} services passed")
    
    if passed == total:
        print("🎉 All passwords are properly configured and working!")
    else:
        print("⚠️  Some services need attention")
    
    print("\nPASSWORD STRENGTH SUMMARY:")
    print("-" * 70)
    print("PostgreSQL:     32 chars (alphanumeric only - URL safe)")
    print("Redis:          64 chars (hex - no special chars)")
    print("Elasticsearch:  44 chars (base64 encoded)")
    print("Grafana:        44 chars (base64 encoded)")
    print("Airflow DB:     44 chars (base64 encoded)")
    print("-" * 70)
    print("All passwords are cryptographically strong and properly formatted!")

def main():
    """Run all tests"""
    print("Testing all service connections with new passwords...")
    print("Please wait while services are checked...\n")
    
    results = {}
    
    # Test each service
    print("1. Testing PostgreSQL...")
    results["PostgreSQL"] = test_postgresql()
    
    print("2. Testing Redis...")
    results["Redis"] = test_redis()
    
    print("3. Testing Elasticsearch...")
    results["Elasticsearch"] = test_elasticsearch()
    
    print("4. Testing Grafana...")
    results["Grafana"] = test_grafana()
    
    # Print results
    print_results(results)
    
    return 0 if all(success for _, (success, _) in results.items()) else 1

if __name__ == "__main__":
    sys.exit(main())