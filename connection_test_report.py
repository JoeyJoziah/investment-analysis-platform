#!/usr/bin/env python3
"""
Comprehensive Connection Test Report
Tests all database and service connections and provides detailed report
"""

import os
import sys
import subprocess
import json
from datetime import datetime

def run_command(cmd, timeout=10):
    """Run a command and return output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"

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
        cursor.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';")
        table_count = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        return True, {'version': version, 'tables': table_count}
    except Exception as e:
        return False, {'error': str(e)}

def test_redis():
    """Test Redis connection"""
    try:
        import redis
        r = redis.Redis(
            host='localhost',
            port=6379,
            password='RsYque',
            db=0,
            decode_responses=True
        )
        r.ping()
        info = r.info()
        return True, {
            'version': info.get('redis_version'),
            'memory': info.get('used_memory_human'),
            'clients': info.get('connected_clients')
        }
    except Exception as e:
        return False, {'error': str(e)}

def test_elasticsearch():
    """Test Elasticsearch connection"""
    try:
        from elasticsearch import Elasticsearch
        es = Elasticsearch(
            [{'host': 'localhost', 'port': 9200, 'scheme': 'http'}],
            verify_certs=False,
            timeout=10
        )
        health = es.cluster.health()
        return True, {
            'status': health.get('status'),
            'cluster_name': health.get('cluster_name'),
            'nodes': health.get('number_of_nodes')
        }
    except Exception as e:
        return False, {'error': str(e)}

def get_docker_status():
    """Get Docker container status"""
    code, stdout, stderr = run_command("docker ps --format json")
    if code != 0:
        return []
    
    containers = []
    for line in stdout.strip().split('\n'):
        if line.strip():
            try:
                container = json.loads(line)
                if any(name in container.get('Names', '') for name in ['investment_db', 'investment_cache', 'investment_search']):
                    containers.append({
                        'name': container.get('Names'),
                        'status': container.get('Status'),
                        'ports': container.get('Ports'),
                        'image': container.get('Image')
                    })
            except:
                pass
    
    return containers

def main():
    """Main test function"""
    print("🔍 INVESTMENT ANALYSIS APP - COMPREHENSIVE CONNECTION REPORT")
    print("=" * 70)
    print(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Check Docker containers
    print("\n🐳 DOCKER CONTAINER STATUS")
    print("-" * 35)
    containers = get_docker_status()
    if containers:
        for container in containers:
            print(f"✅ {container['name']}: {container['status']}")
    else:
        print("❌ No relevant Docker containers found or Docker not accessible")
    
    # Test services
    services = [
        ('PostgreSQL', test_postgresql),
        ('Redis', test_redis),
        ('Elasticsearch', test_elasticsearch)
    ]
    
    results = {}
    
    for service_name, test_func in services:
        print(f"\n🧪 TESTING {service_name.upper()}")
        print("-" * 30)
        
        try:
            success, details = test_func()
            results[service_name] = {'success': success, 'details': details}
            
            if success:
                print(f"✅ Status: Connected successfully")
                for key, value in details.items():
                    if key != 'error':
                        print(f"   {key.title()}: {value}")
            else:
                print(f"❌ Status: Connection failed")
                print(f"   Error: {details.get('error', 'Unknown error')}")
                
        except ImportError as e:
            print(f"❌ Status: Missing library ({e.name})")
            results[service_name] = {'success': False, 'details': {'error': f'Missing library: {e.name}'}}
        except Exception as e:
            print(f"❌ Status: Unexpected error")
            print(f"   Error: {str(e)}")
            results[service_name] = {'success': False, 'details': {'error': str(e)}}
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 FINAL SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for r in results.values() if r['success'])
    total = len(results)
    
    for service, result in results.items():
        status = "✅ WORKING" if result['success'] else "❌ FAILED"
        print(f"{service:15}: {status}")
    
    print(f"\nOverall Success Rate: {passed}/{total} ({passed/total*100:.1f}%)")
    
    # Connection details for working services
    if any(r['success'] for r in results.values()):
        print("\n🔗 WORKING CONNECTION DETAILS")
        print("-" * 35)
        
        if results.get('PostgreSQL', {}).get('success'):
            print("PostgreSQL:")
            print("  URL: postgresql://investment_user:9v1g^OV9XUwzUP6cEgCYgNOE@localhost:5432/investment_db")
            print("  Host: localhost:5432")
        
        if results.get('Redis', {}).get('success'):
            print("Redis:")
            print("  URL: redis://:RsYque@localhost:6379/0")
            print("  Note: Password truncated due to shell interpretation of special characters")
        
        if results.get('Elasticsearch', {}).get('success'):
            print("Elasticsearch:")
            print("  URL: http://localhost:9200")
            print("  Authentication: Disabled (xpack.security.enabled=false)")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS")
    print("-" * 20)
    
    if passed == total:
        print("🎉 EXCELLENT! All services are working perfectly.")
        print("✅ Your investment analysis app is ready for development.")
        print("🚀 You can now run: docker-compose up -d to start all services")
        
    else:
        print("⚠️  Some services need attention:")
        
        if not results.get('PostgreSQL', {}).get('success'):
            print("  • PostgreSQL: Check if container is running and user exists")
            print("    - Run: docker-compose up -d postgres")
            print("    - Check: docker logs investment_db")
            
        if not results.get('Redis', {}).get('success'):
            print("  • Redis: Check password configuration")
            print("    - The password contains special characters that may be shell-escaped")
            print("    - Check: docker logs investment_cache")
            
        if not results.get('Elasticsearch', {}).get('success'):
            print("  • Elasticsearch: Service may still be starting")
            print("    - Elasticsearch takes 1-2 minutes to fully initialize")
            print("    - Check: docker logs investment_search")
    
    print("\n" + "=" * 70)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)