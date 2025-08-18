#!/usr/bin/env python3
"""
Comprehensive Connection Testing Script
Tests all database and service connections with configured passwords
"""

import os
import sys
import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import traceback

# Import required libraries
try:
    import psycopg2
    import redis
    from elasticsearch import Elasticsearch
    import asyncpg
    from sqlalchemy import create_engine, text
    from sqlalchemy.exc import SQLAlchemyError
except ImportError as e:
    print(f"Missing required library: {e}")
    print("Install with: pip install psycopg2-binary redis elasticsearch asyncpg sqlalchemy")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ConnectionTester:
    """Comprehensive connection testing for all services"""
    
    def __init__(self):
        # PostgreSQL credentials
        self.db_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'investment_db',
            'user': 'postgres',
            'password': '9v1g^OV9XUwzUP6cEgCYgNOE'
        }
        
        # PostgreSQL URL for SQLAlchemy
        self.database_url = f"postgresql://{self.db_config['user']}:{self.db_config['password']}@{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
        
        # Redis credentials (password truncated due to shell escaping in docker-compose)
        self.redis_config = {
            'host': 'localhost',
            'port': 6379,
            'password': 'RsYque',  # Actual password being used (truncated at &)
            'db': 0
        }
        
        # Elasticsearch credentials
        self.es_config = {
            'host': 'localhost',
            'port': 9200,
            'username': None,  # No authentication enabled in docker-compose
            'password': None
        }
        
        # Results storage
        self.results = {}
    
    def test_postgresql_psycopg2(self) -> Dict[str, Any]:
        """Test PostgreSQL connection using psycopg2"""
        test_name = "PostgreSQL (psycopg2)"
        logger.info(f"Testing {test_name}...")
        
        try:
            # Create connection
            conn = psycopg2.connect(
                host=self.db_config['host'],
                port=self.db_config['port'],
                database=self.db_config['database'],
                user=self.db_config['user'],
                password=self.db_config['password']
            )
            
            # Test basic query
            cursor = conn.cursor()
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]
            
            # Test write operation
            cursor.execute("CREATE TABLE IF NOT EXISTS test_connection (id SERIAL PRIMARY KEY, test_time TIMESTAMP DEFAULT NOW());")
            cursor.execute("INSERT INTO test_connection DEFAULT VALUES;")
            cursor.execute("SELECT COUNT(*) FROM test_connection;")
            count = cursor.fetchone()[0]
            
            conn.commit()
            cursor.close()
            conn.close()
            
            return {
                'status': 'SUCCESS',
                'version': version,
                'test_records': count,
                'message': 'Connection successful, read/write operations working'
            }
            
        except Exception as e:
            return {
                'status': 'FAILED',
                'error': str(e),
                'traceback': traceback.format_exc(),
                'message': 'Failed to connect to PostgreSQL'
            }
    
    def test_postgresql_sqlalchemy(self) -> Dict[str, Any]:
        """Test PostgreSQL connection using SQLAlchemy"""
        test_name = "PostgreSQL (SQLAlchemy)"
        logger.info(f"Testing {test_name}...")
        
        try:
            # Create engine
            engine = create_engine(self.database_url)
            
            # Test connection
            with engine.connect() as conn:
                result = conn.execute(text("SELECT version();"))
                version = result.fetchone()[0]
                
                # Test write operation
                conn.execute(text("CREATE TABLE IF NOT EXISTS test_sqlalchemy (id SERIAL PRIMARY KEY, test_time TIMESTAMP DEFAULT NOW());"))
                conn.execute(text("INSERT INTO test_sqlalchemy DEFAULT VALUES;"))
                result = conn.execute(text("SELECT COUNT(*) FROM test_sqlalchemy;"))
                count = result.fetchone()[0]
                conn.commit()
            
            engine.dispose()
            
            return {
                'status': 'SUCCESS',
                'version': version,
                'test_records': count,
                'message': 'SQLAlchemy connection successful'
            }
            
        except Exception as e:
            return {
                'status': 'FAILED',
                'error': str(e),
                'traceback': traceback.format_exc(),
                'message': 'Failed to connect via SQLAlchemy'
            }
    
    async def test_postgresql_asyncpg(self) -> Dict[str, Any]:
        """Test PostgreSQL connection using asyncpg"""
        test_name = "PostgreSQL (asyncpg)"
        logger.info(f"Testing {test_name}...")
        
        try:
            # Create async connection
            conn = await asyncpg.connect(
                host=self.db_config['host'],
                port=self.db_config['port'],
                database=self.db_config['database'],
                user=self.db_config['user'],
                password=self.db_config['password']
            )
            
            # Test basic query
            version = await conn.fetchval("SELECT version();")
            
            # Test write operation
            await conn.execute("CREATE TABLE IF NOT EXISTS test_asyncpg (id SERIAL PRIMARY KEY, test_time TIMESTAMP DEFAULT NOW());")
            await conn.execute("INSERT INTO test_asyncpg DEFAULT VALUES;")
            count = await conn.fetchval("SELECT COUNT(*) FROM test_asyncpg;")
            
            await conn.close()
            
            return {
                'status': 'SUCCESS',
                'version': version,
                'test_records': count,
                'message': 'Async PostgreSQL connection successful'
            }
            
        except Exception as e:
            return {
                'status': 'FAILED',
                'error': str(e),
                'traceback': traceback.format_exc(),
                'message': 'Failed to connect via asyncpg'
            }
    
    def test_redis(self) -> Dict[str, Any]:
        """Test Redis connection"""
        test_name = "Redis"
        logger.info(f"Testing {test_name}...")
        
        try:
            # Create Redis connection
            r = redis.Redis(
                host=self.redis_config['host'],
                port=self.redis_config['port'],
                password=self.redis_config['password'],
                db=self.redis_config['db'],
                decode_responses=True
            )
            
            # Test ping
            pong = r.ping()
            
            # Test basic operations
            test_key = f"test_connection_{datetime.now().timestamp()}"
            r.set(test_key, "test_value", ex=60)  # Expires in 60 seconds
            value = r.get(test_key)
            
            # Get server info
            info = r.info()
            redis_version = info.get('redis_version', 'Unknown')
            memory_used = info.get('used_memory_human', 'Unknown')
            
            # Test key operations
            r.delete(test_key)
            
            return {
                'status': 'SUCCESS',
                'ping': pong,
                'test_value': value,
                'redis_version': redis_version,
                'memory_used': memory_used,
                'message': 'Redis connection successful, read/write operations working'
            }
            
        except Exception as e:
            return {
                'status': 'FAILED',
                'error': str(e),
                'traceback': traceback.format_exc(),
                'message': 'Failed to connect to Redis'
            }
    
    def test_elasticsearch(self) -> Dict[str, Any]:
        """Test Elasticsearch connection"""
        test_name = "Elasticsearch"
        logger.info(f"Testing {test_name}...")
        
        try:
            # Create Elasticsearch connection
            if self.es_config['username'] and self.es_config['password']:
                es = Elasticsearch(
                    [{'host': self.es_config['host'], 'port': self.es_config['port'], 'scheme': 'http'}],
                    http_auth=(self.es_config['username'], self.es_config['password']),
                    verify_certs=False,
                    http_compress=True,
                    timeout=30
                )
            else:
                es = Elasticsearch(
                    [{'host': self.es_config['host'], 'port': self.es_config['port'], 'scheme': 'http'}],
                    verify_certs=False,
                    http_compress=True,
                    timeout=30
                )
            
            # Test cluster health
            health = es.cluster.health()
            
            # Test index operations
            test_index = f"test_connection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create test document
            test_doc = {
                'timestamp': datetime.now().isoformat(),
                'test': 'connection_test',
                'message': 'This is a test document'
            }
            
            # Index document
            response = es.index(index=test_index, body=test_doc)
            doc_id = response['_id']
            
            # Search for document
            search_response = es.search(index=test_index, body={'query': {'match_all': {}}})
            
            # Clean up - delete test index
            es.indices.delete(index=test_index, ignore=[400, 404])
            
            return {
                'status': 'SUCCESS',
                'cluster_health': health.get('status', 'Unknown'),
                'version': health.get('cluster_name', 'Unknown'),
                'doc_created': response.get('result') == 'created',
                'search_hits': search_response.get('hits', {}).get('total', {}).get('value', 0),
                'message': 'Elasticsearch connection successful, index/search operations working'
            }
            
        except Exception as e:
            return {
                'status': 'FAILED',
                'error': str(e),
                'traceback': traceback.format_exc(),
                'message': 'Failed to connect to Elasticsearch'
            }
    
    def test_docker_containers(self) -> Dict[str, Any]:
        """Test Docker container connectivity"""
        test_name = "Docker Containers"
        logger.info(f"Testing {test_name}...")
        
        try:
            import subprocess
            
            # Check if Docker is running
            result = subprocess.run(['docker', 'ps'], capture_output=True, text=True)
            if result.returncode != 0:
                return {
                    'status': 'FAILED',
                    'error': 'Docker not available or not running',
                    'message': 'Cannot test Docker container connections'
                }
            
            # Get running containers
            containers = []
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            for line in lines:
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 2:
                        containers.append({
                            'id': parts[0],
                            'image': parts[1],
                            'status': ' '.join(parts[4:])
                        })
            
            # Test specific service containers
            service_tests = {}
            
            # Test PostgreSQL container connection
            if any('postgres' in c['image'].lower() for c in containers):
                try:
                    pg_result = self.test_postgresql_psycopg2()
                    service_tests['postgres_container'] = pg_result['status'] == 'SUCCESS'
                except:
                    service_tests['postgres_container'] = False
            
            # Test Redis container connection
            if any('redis' in c['image'].lower() for c in containers):
                try:
                    redis_result = self.test_redis()
                    service_tests['redis_container'] = redis_result['status'] == 'SUCCESS'
                except:
                    service_tests['redis_container'] = False
            
            return {
                'status': 'SUCCESS',
                'containers_running': len(containers),
                'containers': containers,
                'service_connectivity': service_tests,
                'message': f'Found {len(containers)} running containers'
            }
            
        except Exception as e:
            return {
                'status': 'FAILED',
                'error': str(e),
                'traceback': traceback.format_exc(),
                'message': 'Failed to check Docker containers'
            }
    
    async def run_all_tests(self) -> Dict[str, Dict[str, Any]]:
        """Run all connection tests"""
        logger.info("Starting comprehensive connection tests...")
        
        # Synchronous tests
        tests = [
            ('postgresql_psycopg2', self.test_postgresql_psycopg2),
            ('postgresql_sqlalchemy', self.test_postgresql_sqlalchemy),
            ('redis', self.test_redis),
            ('elasticsearch', self.test_elasticsearch),
            ('docker_containers', self.test_docker_containers),
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            try:
                results[test_name] = test_func()
            except Exception as e:
                results[test_name] = {
                    'status': 'FAILED',
                    'error': str(e),
                    'message': f'Unexpected error in {test_name} test'
                }
        
        # Asynchronous test
        try:
            results['postgresql_asyncpg'] = await self.test_postgresql_asyncpg()
        except Exception as e:
            results['postgresql_asyncpg'] = {
                'status': 'FAILED',
                'error': str(e),
                'message': 'Unexpected error in asyncpg test'
            }
        
        return results
    
    def print_results(self, results: Dict[str, Dict[str, Any]]):
        """Print formatted test results"""
        print("\n" + "="*80)
        print("CONNECTION TEST RESULTS")
        print("="*80)
        
        success_count = 0
        total_count = len(results)
        
        for test_name, result in results.items():
            status = result.get('status', 'UNKNOWN')
            if status == 'SUCCESS':
                print(f"✅ {test_name.replace('_', ' ').title()}: {status}")
                success_count += 1
            else:
                print(f"❌ {test_name.replace('_', ' ').title()}: {status}")
            
            # Print key details
            if status == 'SUCCESS':
                if 'version' in result:
                    print(f"   Version: {result['version'][:100]}...")
                if 'message' in result:
                    print(f"   Details: {result['message']}")
            else:
                if 'error' in result:
                    print(f"   Error: {result['error']}")
                if 'message' in result:
                    print(f"   Details: {result['message']}")
            print()
        
        print("="*80)
        print(f"SUMMARY: {success_count}/{total_count} tests passed")
        print("="*80)
        
        # Specific recommendations
        if success_count < total_count:
            print("\nRECOMMENDATIONS:")
            
            if results.get('postgresql_psycopg2', {}).get('status') != 'SUCCESS':
                print("- PostgreSQL: Check if PostgreSQL service is running")
                print("  - Run: docker-compose up postgres -d")
                print("  - Verify credentials match .env file")
            
            if results.get('redis', {}).get('status') != 'SUCCESS':
                print("- Redis: Check if Redis service is running")
                print("  - Run: docker-compose up redis -d")
                print("  - Verify Redis password configuration")
            
            if results.get('elasticsearch', {}).get('status') != 'SUCCESS':
                print("- Elasticsearch: Check if Elasticsearch service is running")
                print("  - Run: docker-compose up elasticsearch -d")
                print("  - Verify Elasticsearch credentials")
        else:
            print("\n🎉 All connection tests passed! Your services are configured correctly.")

def main():
    """Main function to run connection tests"""
    tester = ConnectionTester()
    
    try:
        # Run tests
        results = asyncio.run(tester.run_all_tests())
        
        # Print results
        tester.print_results(results)
        
        # Return appropriate exit code
        failed_tests = [name for name, result in results.items() if result.get('status') != 'SUCCESS']
        if failed_tests:
            print(f"\nFailed tests: {', '.join(failed_tests)}")
            sys.exit(1)
        else:
            print("\n✅ All connection tests passed successfully!")
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error running tests: {e}")
        print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()