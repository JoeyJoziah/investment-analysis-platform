#!/usr/bin/env python3
"""
Comprehensive API Connection Testing Script
Tests all financial data API connections with robust fallback mechanisms
"""

import asyncio
import logging
import sys
import os
import traceback
from typing import Dict, Any, Optional, Union, List
from datetime import datetime
import json

# Add backend to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class APIConnectionTester:
    """Comprehensive API connection testing with fallback mechanisms"""
    
    def __init__(self):
        self.results = {}
        self.api_keys = self._load_api_keys()
        
    def _load_api_keys(self) -> Dict[str, str]:
        """Load API keys from environment or .env file"""
        api_keys = {}
        
        # Try to load from .env file
        env_file = '.env'
        if os.path.exists(env_file):
            with open(env_file, 'r') as f:
                for line in f:
                    if '=' in line and not line.strip().startswith('#'):
                        key, value = line.strip().split('=', 1)
                        if 'API_KEY' in key:
                            api_keys[key] = value
        
        # Override with environment variables
        for key in ['FINNHUB_API_KEY', 'ALPHA_VANTAGE_API_KEY', 'POLYGON_API_KEY', 'NEWS_API_KEY']:
            if key in os.environ:
                api_keys[key] = os.environ[key]
        
        return api_keys
    
    async def test_aiohttp_availability(self) -> Dict[str, Any]:
        """Test if aiohttp is available and working"""
        test_name = "aiohttp Availability"
        logger.info(f"Testing {test_name}...")
        
        try:
            import aiohttp
            
            # Test basic aiohttp functionality
            async with aiohttp.ClientSession() as session:
                # Test with httpbin.org for basic connectivity
                async with session.get('https://httpbin.org/status/200', timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        return {
                            'status': 'SUCCESS',
                            'version': aiohttp.__version__,
                            'message': f'aiohttp {aiohttp.__version__} is working properly'
                        }
                    else:
                        return {
                            'status': 'FAILED',
                            'version': aiohttp.__version__,
                            'error': f'HTTP request failed with status {response.status}',
                            'message': 'aiohttp module loads but HTTP requests fail'
                        }
        
        except ImportError as e:
            return {
                'status': 'FAILED',
                'error': str(e),
                'message': 'aiohttp module not found - install with: pip install aiohttp'
            }
        except Exception as e:
            return {
                'status': 'FAILED',
                'error': str(e),
                'traceback': traceback.format_exc(),
                'message': 'aiohttp available but failed to make HTTP request'
            }
    
    async def test_requests_fallback(self) -> Dict[str, Any]:
        """Test requests library as fallback"""
        test_name = "Requests Fallback"
        logger.info(f"Testing {test_name}...")
        
        try:
            import requests
            
            # Test basic requests functionality
            response = requests.get('https://httpbin.org/status/200', timeout=10)
            if response.status_code == 200:
                return {
                    'status': 'SUCCESS',
                    'version': requests.__version__,
                    'message': f'requests {requests.__version__} is working as fallback'
                }
            else:
                return {
                    'status': 'FAILED',
                    'version': requests.__version__,
                    'error': f'HTTP request failed with status {response.status_code}',
                    'message': 'requests module loads but HTTP requests fail'
                }
        
        except ImportError as e:
            return {
                'status': 'FAILED',
                'error': str(e),
                'message': 'requests module not found - install with: pip install requests'
            }
        except Exception as e:
            return {
                'status': 'FAILED',
                'error': str(e),
                'traceback': traceback.format_exc(),
                'message': 'requests available but failed to make HTTP request'
            }
    
    async def test_finnhub_api(self) -> Dict[str, Any]:
        """Test Finnhub API connection"""
        test_name = "Finnhub API"
        logger.info(f"Testing {test_name}...")
        
        try:
            from backend.data_ingestion.finnhub_client import FinnhubClient
            
            client = FinnhubClient()
            
            # Check if API key is available
            if not client.api_key:
                return {
                    'status': 'SKIPPED',
                    'message': 'No Finnhub API key found in environment (FINNHUB_API_KEY)',
                    'recommendation': 'Get free API key from https://finnhub.io/'
                }
            
            # Test connection with basic quote request
            async with client:
                result = await client.get_quote('AAPL')
                
                if result and 'current_price' in result:
                    return {
                        'status': 'SUCCESS',
                        'sample_data': {
                            'symbol': result.get('symbol'),
                            'current_price': result.get('current_price'),
                            'timestamp': result.get('timestamp')
                        },
                        'message': 'Finnhub API connection successful - quote data retrieved'
                    }
                else:
                    return {
                        'status': 'FAILED',
                        'error': 'No data returned or invalid response format',
                        'message': 'API key may be invalid or rate limited'
                    }
        
        except ImportError as e:
            return {
                'status': 'FAILED',
                'error': str(e),
                'message': 'Failed to import FinnhubClient - check backend.data_ingestion module'
            }
        except Exception as e:
            return {
                'status': 'FAILED',
                'error': str(e),
                'traceback': traceback.format_exc(),
                'message': 'Finnhub API test failed'
            }
    
    async def test_alpha_vantage_api(self) -> Dict[str, Any]:
        """Test Alpha Vantage API connection"""
        test_name = "Alpha Vantage API"
        logger.info(f"Testing {test_name}...")
        
        try:
            from backend.data_ingestion.alpha_vantage_client import AlphaVantageClient
            
            client = AlphaVantageClient()
            
            # Check if API key is available
            if not client.api_key:
                return {
                    'status': 'SKIPPED',
                    'message': 'No Alpha Vantage API key found in environment (ALPHA_VANTAGE_API_KEY)',
                    'recommendation': 'Get free API key from https://www.alphavantage.co/support/#api-key'
                }
            
            # Test connection with basic quote request
            async with client:
                result = await client.get_quote('AAPL')
                
                if result and 'price' in result:
                    return {
                        'status': 'SUCCESS',
                        'sample_data': {
                            'symbol': result.get('symbol'),
                            'price': result.get('price'),
                            'timestamp': result.get('timestamp')
                        },
                        'message': 'Alpha Vantage API connection successful - quote data retrieved'
                    }
                else:
                    return {
                        'status': 'FAILED',
                        'error': 'No data returned or invalid response format',
                        'message': 'API key may be invalid or rate limited (25 calls/day limit)'
                    }
        
        except ImportError as e:
            return {
                'status': 'FAILED',
                'error': str(e),
                'message': 'Failed to import AlphaVantageClient - check backend.data_ingestion module'
            }
        except Exception as e:
            return {
                'status': 'FAILED',
                'error': str(e),
                'traceback': traceback.format_exc(),
                'message': 'Alpha Vantage API test failed'
            }
    
    async def test_polygon_api(self) -> Dict[str, Any]:
        """Test Polygon.io API connection"""
        test_name = "Polygon API"
        logger.info(f"Testing {test_name}...")
        
        try:
            from backend.data_ingestion.polygon_client import PolygonClient
            
            client = PolygonClient()
            
            # Check if API key is available
            if not client.api_key:
                return {
                    'status': 'SKIPPED',
                    'message': 'No Polygon API key found in environment (POLYGON_API_KEY)',
                    'recommendation': 'Get free API key from https://polygon.io/'
                }
            
            # Test connection with basic quote request
            async with client:
                result = await client.get_quote('AAPL')
                
                if result:
                    return {
                        'status': 'SUCCESS',
                        'sample_data': result,
                        'message': 'Polygon API connection successful - quote data retrieved'
                    }
                else:
                    return {
                        'status': 'FAILED',
                        'error': 'No data returned or invalid response format',
                        'message': 'API key may be invalid or rate limited (5 calls/minute on free tier)'
                    }
        
        except ImportError as e:
            return {
                'status': 'FAILED',
                'error': str(e),
                'message': 'Failed to import PolygonClient - check backend.data_ingestion module'
            }
        except Exception as e:
            return {
                'status': 'FAILED',
                'error': str(e),
                'traceback': traceback.format_exc(),
                'message': 'Polygon API test failed'
            }
    
    async def test_base_client_functionality(self) -> Dict[str, Any]:
        """Test BaseAPIClient functionality"""
        test_name = "BaseAPIClient Functionality"
        logger.info(f"Testing {test_name}...")
        
        try:
            from backend.data_ingestion.base_client import BaseAPIClient
            
            # Test that BaseAPIClient can be imported and used
            class TestClient(BaseAPIClient):
                def _get_base_url(self) -> str:
                    return "https://httpbin.org"
                
                async def test_request(self):
                    return await self._make_request("status/200")
            
            client = TestClient("test_provider")
            
            async with client:
                result = await client.test_request()
                
                if result is not None:
                    return {
                        'status': 'SUCCESS',
                        'message': 'BaseAPIClient working - HTTP requests successful'
                    }
                else:
                    return {
                        'status': 'FAILED',
                        'message': 'BaseAPIClient HTTP request returned None'
                    }
        
        except ImportError as e:
            return {
                'status': 'FAILED',
                'error': str(e),
                'message': 'Failed to import BaseAPIClient - check backend.data_ingestion module'
            }
        except Exception as e:
            return {
                'status': 'FAILED',
                'error': str(e),
                'traceback': traceback.format_exc(),
                'message': 'BaseAPIClient test failed'
            }
    
    async def test_redis_connection(self) -> Dict[str, Any]:
        """Test Redis connection for caching"""
        test_name = "Redis Connection (for caching)"
        logger.info(f"Testing {test_name}...")
        
        try:
            from backend.utils.cache import get_redis
            
            redis = await get_redis()
            
            # Test basic operations
            test_key = f"api_test_{datetime.now().timestamp()}"
            await redis.set(test_key, "test_value", ex=60)
            value = await redis.get(test_key)
            await redis.delete(test_key)
            
            if value == "test_value":
                return {
                    'status': 'SUCCESS',
                    'message': 'Redis connection successful - caching will work for API clients'
                }
            else:
                return {
                    'status': 'FAILED',
                    'message': 'Redis operations failed - API caching may not work'
                }
        
        except ImportError as e:
            return {
                'status': 'FAILED',
                'error': str(e),
                'message': 'Failed to import Redis cache module'
            }
        except Exception as e:
            return {
                'status': 'FAILED',
                'error': str(e),
                'traceback': traceback.format_exc(),
                'message': 'Redis connection test failed'
            }
    
    async def test_cost_monitor(self) -> Dict[str, Any]:
        """Test cost monitoring system"""
        test_name = "Cost Monitor System"
        logger.info(f"Testing {test_name}...")
        
        try:
            from backend.utils.cost_monitor import cost_monitor
            
            # Initialize the cost monitor (sets up Redis connection)
            await cost_monitor.initialize()
            
            # Test cost monitoring functionality
            provider = "test_provider"
            endpoint = "test_endpoint"
            
            # Test API limit check
            can_call = await cost_monitor.check_api_limit(provider, endpoint)
            
            # Record a test API call
            await cost_monitor.record_api_call(
                provider=provider,
                endpoint=endpoint,
                success=True,
                response_time_ms=100
            )
            
            return {
                'status': 'SUCCESS',
                'can_make_calls': can_call,
                'message': 'Cost monitoring system working - API usage will be tracked'
            }
        
        except ImportError as e:
            return {
                'status': 'FAILED',
                'error': str(e),
                'message': 'Failed to import cost monitor module'
            }
        except Exception as e:
            return {
                'status': 'FAILED',
                'error': str(e),
                'traceback': traceback.format_exc(),
                'message': 'Cost monitor test failed'
            }
    
    async def run_all_tests(self) -> Dict[str, Dict[str, Any]]:
        """Run all API connection tests"""
        logger.info("Starting comprehensive API connection tests...")
        
        tests = [
            ('aiohttp_availability', self.test_aiohttp_availability),
            ('requests_fallback', self.test_requests_fallback),
            ('base_client_functionality', self.test_base_client_functionality),
            ('redis_connection', self.test_redis_connection),
            ('cost_monitor', self.test_cost_monitor),
            ('finnhub_api', self.test_finnhub_api),
            ('alpha_vantage_api', self.test_alpha_vantage_api),
            ('polygon_api', self.test_polygon_api),
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            try:
                logger.info(f"Running {test_name}...")
                results[test_name] = await test_func()
            except Exception as e:
                results[test_name] = {
                    'status': 'FAILED',
                    'error': str(e),
                    'traceback': traceback.format_exc(),
                    'message': f'Unexpected error in {test_name} test'
                }
        
        return results
    
    def print_results(self, results: Dict[str, Dict[str, Any]]):
        """Print formatted test results"""
        print("\n" + "="*80)
        print("API CONNECTION TEST RESULTS")
        print("="*80)
        
        success_count = 0
        failed_count = 0
        skipped_count = 0
        total_count = len(results)
        
        for test_name, result in results.items():
            status = result.get('status', 'UNKNOWN')
            display_name = test_name.replace('_', ' ').title()
            
            if status == 'SUCCESS':
                print(f"✅ {display_name}: {status}")
                success_count += 1
            elif status == 'SKIPPED':
                print(f"⚠️  {display_name}: {status}")
                skipped_count += 1
            else:
                print(f"❌ {display_name}: {status}")
                failed_count += 1
            
            # Print key details
            if 'message' in result:
                print(f"   {result['message']}")
            
            if status == 'SUCCESS' and 'sample_data' in result:
                print(f"   Sample: {json.dumps(result['sample_data'], indent=6)[:100]}...")
            
            if status == 'FAILED' and 'error' in result:
                print(f"   Error: {result['error']}")
            
            if status == 'SKIPPED' and 'recommendation' in result:
                print(f"   💡 {result['recommendation']}")
            
            print()
        
        print("="*80)
        print(f"SUMMARY: {success_count} passed, {failed_count} failed, {skipped_count} skipped (Total: {total_count})")
        print("="*80)
        
        # Specific recommendations
        if failed_count > 0:
            print("\nTROUBLESHOoting RECOMMENDATIONS:")
            
            if results.get('aiohttp_availability', {}).get('status') == 'FAILED':
                print("🔧 aiohttp Issue:")
                print("   - Install: pip install aiohttp")
                print("   - Check Python version compatibility")
            
            if results.get('redis_connection', {}).get('status') == 'FAILED':
                print("🔧 Redis Issue:")
                print("   - Start Redis: docker-compose up redis -d")
                print("   - Check Redis configuration in .env")
            
            if any('api' in name and results[name]['status'] == 'FAILED' for name in results):
                print("🔧 API Issues:")
                print("   - Check internet connectivity")
                print("   - Verify API keys in .env file")
                print("   - Check API rate limits")
        
        elif skipped_count == 0:
            print("\n🎉 All API connection tests passed! Your API clients are working perfectly.")
        
        else:
            print(f"\n✅ Core functionality working. {skipped_count} tests skipped due to missing API keys.")


def main():
    """Main function to run API connection tests"""
    print("Investment Analysis Platform - API Connection Tester")
    print("Testing all financial data API connections with robust fallback mechanisms")
    print("-" * 80)
    
    tester = APIConnectionTester()
    
    try:
        # Run tests
        results = asyncio.run(tester.run_all_tests())
        
        # Print results
        tester.print_results(results)
        
        # Return appropriate exit code
        failed_tests = [name for name, result in results.items() if result.get('status') == 'FAILED']
        if failed_tests:
            print(f"\nFailed tests: {', '.join(failed_tests)}")
            sys.exit(1)
        else:
            print("\n✅ API connection testing completed successfully!")
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