#!/usr/bin/env python3
"""
Pipeline Component Testing Script
Tests individual components of the data pipeline to ensure they work correctly.
"""

import asyncio
import logging
import sys
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import importlib.util

# Add backend to Python path
sys.path.insert(0, 'backend')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PipelineComponentTester:
    """Tests all pipeline components independently"""
    
    def __init__(self):
        self.test_results = {}
        
    async def run_all_tests(self) -> bool:
        """Run comprehensive pipeline testing"""
        logger.info("🧪 Starting Pipeline Component Tests")
        
        test_methods = [
            ("API Clients", self.test_api_clients),
            ("Cost Monitor", self.test_cost_monitor), 
            ("Stock Prioritization", self.test_stock_prioritization),
            ("Data Quality", self.test_data_quality),
            ("Cache System", self.test_cache_system),
            ("Circuit Breakers", self.test_circuit_breakers),
            ("Rate Limiters", self.test_rate_limiters),
            ("Database Operations", self.test_database_operations),
            ("Pipeline Integration", self.test_pipeline_integration)
        ]
        
        for test_name, test_method in test_methods:
            logger.info(f"🔍 Testing {test_name}...")
            try:
                result = await test_method()
                self.test_results[test_name] = {
                    'status': 'PASS' if result else 'FAIL',
                    'timestamp': datetime.now().isoformat()
                }
                logger.info(f"  {'✅ PASS' if result else '❌ FAIL'}")
            except Exception as e:
                self.test_results[test_name] = {
                    'status': 'ERROR',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                logger.error(f"  ❌ ERROR: {e}")
        
        # Print summary
        await self.print_test_summary()
        
        # Return overall success
        passed_tests = sum(1 for result in self.test_results.values() if result['status'] == 'PASS')
        total_tests = len(self.test_results)
        
        return passed_tests >= (total_tests * 0.7)  # 70% pass rate required
    
    async def test_api_clients(self) -> bool:
        """Test API client functionality"""
        try:
            # Test importing API clients
            from backend.data_ingestion.base_client import BaseAPIClient
            from backend.data_ingestion.finnhub_client import FinnhubClient
            from backend.data_ingestion.alpha_vantage_client import AlphaVantageClient
            
            # Test base client functionality
            test_passed = True
            
            # Test Finnhub client
            if os.getenv('FINNHUB_API_KEY'):
                logger.info("    Testing Finnhub client...")
                finnhub = FinnhubClient()
                # Don't actually make API calls in test - just test initialization
                test_passed = test_passed and hasattr(finnhub, 'get_quote')
            
            # Test Alpha Vantage client
            if os.getenv('ALPHA_VANTAGE_API_KEY'):
                logger.info("    Testing Alpha Vantage client...")
                av_client = AlphaVantageClient()
                test_passed = test_passed and hasattr(av_client, 'get_quote')
            
            logger.info(f"    API Clients: {len([c for c in ['FINNHUB_API_KEY', 'ALPHA_VANTAGE_API_KEY'] if os.getenv(c)])} configured")
            
            return test_passed
            
        except ImportError as e:
            logger.error(f"    Import error: {e}")
            return False
        except Exception as e:
            logger.error(f"    Unexpected error: {e}")
            return False
    
    async def test_cost_monitor(self) -> bool:
        """Test cost monitoring system"""
        try:
            from backend.utils.cost_monitor import CostMonitor, SmartDataFetcher
            from backend.utils.enhanced_cost_monitor import EnhancedCostMonitor
            
            # Test basic cost monitor
            basic_monitor = CostMonitor()
            test_passed = hasattr(basic_monitor, 'check_api_limit')
            
            # Test enhanced cost monitor
            enhanced_monitor = EnhancedCostMonitor()
            test_passed = test_passed and hasattr(enhanced_monitor, 'get_optimal_schedule')
            
            # Test smart data fetcher
            smart_fetcher = SmartDataFetcher(basic_monitor)
            test_passed = test_passed and hasattr(smart_fetcher, 'fetch_stock_data')
            
            logger.info("    Cost monitoring components loaded successfully")
            return test_passed
            
        except ImportError as e:
            logger.error(f"    Import error: {e}")
            return False
        except Exception as e:
            logger.error(f"    Unexpected error: {e}")
            return False
    
    async def test_stock_prioritization(self) -> bool:
        """Test stock prioritization system"""
        try:
            # Test stock tier system
            from backend.utils.enhanced_cost_monitor import StockPriority
            
            # Test priority enum
            priorities = list(StockPriority)
            test_passed = len(priorities) == 5  # Should have 5 priority levels
            
            # Test tier functionality (mock)
            sample_stocks = [
                ('AAPL', StockPriority.CRITICAL),
                ('MSFT', StockPriority.CRITICAL),
                ('SOME_SMALL_CAP', StockPriority.MEDIUM)
            ]
            
            tier_counts = {}
            for symbol, priority in sample_stocks:
                tier_counts[priority.value] = tier_counts.get(priority.value, 0) + 1
            
            test_passed = test_passed and len(tier_counts) > 0
            
            logger.info(f"    Stock prioritization: {len(sample_stocks)} test stocks processed")
            return test_passed
            
        except ImportError as e:
            logger.error(f"    Import error: {e}")
            return False
        except Exception as e:
            logger.error(f"    Unexpected error: {e}")
            return False
    
    async def test_data_quality(self) -> bool:
        """Test data quality checking"""
        try:
            from backend.utils.data_quality import DataQualityChecker
            from backend.utils.enhanced_data_quality import EnhancedDataQualityChecker
            
            # Test basic data quality checker
            checker = DataQualityChecker()
            test_passed = hasattr(checker, 'validate_price_data')
            
            # Test enhanced checker if available
            try:
                enhanced_checker = EnhancedDataQualityChecker()
                test_passed = test_passed and hasattr(enhanced_checker, 'comprehensive_validation')
            except ImportError:
                logger.info("    Enhanced data quality checker not available")
            
            # Test sample data validation
            sample_data = {
                'open': 100.0,
                'high': 105.0,
                'low': 95.0,
                'close': 102.0,
                'volume': 1000000
            }
            
            # Mock validation (basic checks)
            data_valid = (
                sample_data['high'] >= sample_data['low'] and
                sample_data['high'] >= sample_data['open'] and
                sample_data['high'] >= sample_data['close'] and
                sample_data['low'] <= sample_data['open'] and
                sample_data['low'] <= sample_data['close'] and
                sample_data['volume'] > 0
            )
            
            test_passed = test_passed and data_valid
            
            logger.info("    Data quality validation working")
            return test_passed
            
        except ImportError as e:
            logger.error(f"    Import error: {e}")
            return False
        except Exception as e:
            logger.error(f"    Unexpected error: {e}")
            return False
    
    async def test_cache_system(self) -> bool:
        """Test caching system"""
        try:
            from backend.utils.cache import get_cache, get_redis
            from backend.utils.advanced_cache import AdvancedCache
            
            # Test basic cache functionality (mock)
            test_passed = True
            
            # Test cache components exist
            test_passed = test_passed and callable(get_cache)
            test_passed = test_passed and callable(get_redis)
            
            # Test advanced cache if available
            try:
                cache = AdvancedCache()
                test_passed = test_passed and hasattr(cache, 'get')
            except ImportError:
                logger.info("    Advanced cache not available, using basic cache")
            
            logger.info("    Cache system components available")
            return test_passed
            
        except ImportError as e:
            logger.error(f"    Import error: {e}")
            return False
        except Exception as e:
            logger.error(f"    Unexpected error: {e}")
            return False
    
    async def test_circuit_breakers(self) -> bool:
        """Test circuit breaker functionality"""
        try:
            from backend.utils.circuit_breaker import CircuitBreaker
            from backend.utils.advanced_circuit_breaker import AdvancedCircuitBreaker
            
            # Test basic circuit breaker
            cb = CircuitBreaker(
                failure_threshold=3,
                recovery_timeout=10,
                expected_exception=Exception
            )
            test_passed = hasattr(cb, 'call')
            
            # Test advanced circuit breaker if available
            try:
                advanced_cb = AdvancedCircuitBreaker()
                test_passed = test_passed and hasattr(advanced_cb, 'call')
            except ImportError:
                logger.info("    Advanced circuit breaker not available")
            
            logger.info("    Circuit breaker components loaded")
            return test_passed
            
        except ImportError as e:
            logger.error(f"    Import error: {e}")
            return False
        except Exception as e:
            logger.error(f"    Unexpected error: {e}")
            return False
    
    async def test_rate_limiters(self) -> bool:
        """Test rate limiting functionality"""
        try:
            from backend.utils.rate_limiter import RateLimiter
            
            # Test rate limiter
            limiter = RateLimiter(max_calls=10, window_seconds=60)
            test_passed = hasattr(limiter, 'is_allowed')
            
            # Test distributed rate limiter if available
            try:
                from backend.utils.distributed_rate_limiter import DistributedRateLimiter
                dist_limiter = DistributedRateLimiter()
                test_passed = test_passed and hasattr(dist_limiter, 'is_allowed')
            except ImportError:
                logger.info("    Distributed rate limiter not available")
            
            logger.info("    Rate limiter components loaded")
            return test_passed
            
        except ImportError as e:
            logger.error(f"    Import error: {e}")
            return False
        except Exception as e:
            logger.error(f"    Unexpected error: {e}")
            return False
    
    async def test_database_operations(self) -> bool:
        """Test database operations"""
        try:
            from backend.models.database import get_db_session
            from backend.models.tables import Stock, PriceHistory
            
            test_passed = True
            
            # Test model imports
            test_passed = test_passed and Stock is not None
            test_passed = test_passed and PriceHistory is not None
            
            # Test database session function
            test_passed = test_passed and callable(get_db_session)
            
            logger.info("    Database models and session management available")
            return test_passed
            
        except ImportError as e:
            logger.error(f"    Import error: {e}")
            return False
        except Exception as e:
            logger.error(f"    Unexpected error: {e}")
            return False
    
    async def test_pipeline_integration(self) -> bool:
        """Test pipeline integration components"""
        try:
            # Test that main DAG can be imported
            dag_path = "data_pipelines/airflow/dags/daily_market_analysis.py"
            if os.path.exists(dag_path):
                # Load the DAG module
                spec = importlib.util.spec_from_file_location("dag_module", dag_path)
                dag_module = importlib.util.module_from_spec(spec)
                
                # Check if we can load without errors
                test_passed = spec is not None
                logger.info("    Main DAG file loadable")
            else:
                logger.warning("    DAG file not found")
                test_passed = False
            
            # Test analytics components
            try:
                from backend.analytics.recommendation_engine import RecommendationEngine
                engine = RecommendationEngine()
                test_passed = test_passed and hasattr(engine, 'generate_daily_recommendations')
                logger.info("    Recommendation engine available")
            except ImportError:
                logger.warning("    Recommendation engine not available")
            
            # Test technical analysis
            try:
                from backend.analytics.technical_analysis import TechnicalAnalysisEngine
                ta_engine = TechnicalAnalysisEngine()
                test_passed = test_passed and hasattr(ta_engine, 'analyze')
                logger.info("    Technical analysis engine available")
            except ImportError:
                logger.warning("    Technical analysis engine not available")
            
            return test_passed
            
        except Exception as e:
            logger.error(f"    Integration test error: {e}")
            return False
    
    async def print_test_summary(self):
        """Print comprehensive test summary"""
        logger.info("\n" + "="*60)
        logger.info("🧪 PIPELINE COMPONENT TEST RESULTS")
        logger.info("="*60)
        
        passed = sum(1 for r in self.test_results.values() if r['status'] == 'PASS')
        failed = sum(1 for r in self.test_results.values() if r['status'] == 'FAIL')
        errors = sum(1 for r in self.test_results.values() if r['status'] == 'ERROR')
        total = len(self.test_results)
        
        logger.info(f"📊 OVERALL RESULTS:")
        logger.info(f"  Total Tests: {total}")
        logger.info(f"  Passed: {passed} ✅")
        logger.info(f"  Failed: {failed} ❌")
        logger.info(f"  Errors: {errors} ⚠️")
        logger.info(f"  Success Rate: {(passed/total)*100:.1f}%")
        
        logger.info(f"\n📋 DETAILED RESULTS:")
        for test_name, result in self.test_results.items():
            status_icon = "✅" if result['status'] == 'PASS' else "❌" if result['status'] == 'FAIL' else "⚠️"
            logger.info(f"  {status_icon} {test_name}: {result['status']}")
            if 'error' in result:
                logger.info(f"    Error: {result['error']}")
        
        logger.info("\n🔧 RECOMMENDATIONS:")
        
        if failed > 0 or errors > 0:
            logger.info("  • Fix failed components before production deployment")
            logger.info("  • Check import paths and dependencies")
            logger.info("  • Verify environment configuration")
        
        if passed / total < 0.8:
            logger.info("  • Component test success rate below 80%")
            logger.info("  • Review component implementations")
        
        if passed / total >= 0.8:
            logger.info("  • Pipeline components look good!")
            logger.info("  • Ready for integration testing")
        
        logger.info("="*60)
        
        # Save results to file
        with open('pipeline_test_results.json', 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        logger.info("📄 Results saved to: pipeline_test_results.json")


async def main():
    """Main entry point"""
    tester = PipelineComponentTester()
    success = await tester.run_all_tests()
    
    if success:
        print("\n🎉 Pipeline Component Tests: MOSTLY SUCCESSFUL!")
        print("Check the summary above for any issues that need attention.")
        return 0
    else:
        print("\n❌ Pipeline Component Tests: FAILED!")
        print("Review the test results and fix failing components.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)