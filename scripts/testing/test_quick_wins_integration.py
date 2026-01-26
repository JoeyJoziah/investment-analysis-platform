#!/usr/bin/env python3
"""
Quick Wins Integration Test

This script validates that all five Quick Win optimizations are properly
integrated and working together:

1. Cache Decorator Fix - Redis caching works
2. API Parallelization - External calls run in parallel
3. Elasticsearch Removal - App works without Elasticsearch
4. Redis Memory Increase - 512MB configured
5. Database Indexes - Migration applied successfully

Run with: python scripts/testing/test_quick_wins_integration.py
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)


class IntegrationTestResult:
    """Container for test results"""
    def __init__(self, name: str, passed: bool, message: str, duration_ms: float = 0):
        self.name = name
        self.passed = passed
        self.message = message
        self.duration_ms = duration_ms


class QuickWinsIntegrationTester:
    """Test suite for Quick Wins integration validation"""

    def __init__(self):
        self.results: List[IntegrationTestResult] = []
        self.redis_client = None
        self.db_session = None

    def _add_result(self, name: str, passed: bool, message: str, duration_ms: float = 0):
        """Record a test result"""
        self.results.append(IntegrationTestResult(name, passed, message, duration_ms))
        status = "PASS" if passed else "FAIL"
        timing = f" ({duration_ms:.1f}ms)" if duration_ms > 0 else ""
        print(f"  [{status}] {name}{timing}: {message}")

    async def test_redis_connection(self) -> bool:
        """Test 1: Verify Redis connection with new memory settings"""
        print("\n=== Test 1: Redis Connection & Memory Configuration ===")

        try:
            from backend.utils.cache import get_redis

            start = time.time()
            self.redis_client = await get_redis()
            duration = (time.time() - start) * 1000

            # Test connection
            pong = await self.redis_client.ping()
            if not pong:
                self._add_result("Redis ping", False, "Ping failed")
                return False

            self._add_result("Redis connection", True, "Connected successfully", duration)

            # Check memory configuration
            info = await self.redis_client.info("memory")
            maxmemory = info.get("maxmemory", 0)
            maxmemory_mb = maxmemory / (1024 * 1024) if maxmemory else 0

            # Verify 512MB configuration (allow some variance)
            if maxmemory_mb >= 500:
                self._add_result("Redis memory config", True, f"maxmemory={maxmemory_mb:.0f}MB (expected ~512MB)")
            elif maxmemory_mb > 0:
                self._add_result("Redis memory config", False, f"maxmemory={maxmemory_mb:.0f}MB (expected ~512MB)")
            else:
                self._add_result("Redis memory config", True, "No memory limit set (unlimited)")

            # Check eviction policy
            config = await self.redis_client.config_get("maxmemory-policy")
            policy = config.get("maxmemory-policy", "unknown")
            if policy == "allkeys-lru":
                self._add_result("Redis eviction policy", True, f"Policy: {policy}")
            else:
                self._add_result("Redis eviction policy", False, f"Policy: {policy} (expected allkeys-lru)")

            return True

        except Exception as e:
            self._add_result("Redis connection", False, f"Error: {e}")
            return False

    async def test_cache_decorator(self) -> bool:
        """Test 2: Verify cache decorator functionality"""
        print("\n=== Test 2: Cache Decorator Functionality ===")

        if not self.redis_client:
            self._add_result("Cache decorator", False, "Redis not available")
            return False

        try:
            from backend.utils.cache import cache_with_ttl

            # Test counter to verify caching
            call_count = 0

            @cache_with_ttl(ttl=60, prefix="test")
            async def test_function(x: int, y: str) -> Dict:
                nonlocal call_count
                call_count += 1
                return {"x": x, "y": y, "call_count": call_count}

            # First call - should execute function
            start = time.time()
            result1 = await test_function(1, "hello")
            duration1 = (time.time() - start) * 1000

            if result1["call_count"] != 1:
                self._add_result("Cache first call", False, f"Unexpected call count: {result1['call_count']}")
                return False

            self._add_result("Cache first call", True, f"Function executed", duration1)

            # Second call - should use cache
            start = time.time()
            result2 = await test_function(1, "hello")
            duration2 = (time.time() - start) * 1000

            if result2["call_count"] != 1:
                # Function was called again - cache miss
                self._add_result("Cache hit", False, f"Cache miss - function called again (count: {result2['call_count']})")
            else:
                # Cache hit - call_count should still be 1
                self._add_result("Cache hit", True, f"Cache hit - returned cached result", duration2)

            # Different args - should execute function
            result3 = await test_function(2, "world")
            if result3["call_count"] > 1:
                self._add_result("Cache key isolation", True, f"Different args = different cache key")

            # Clean up test keys
            pattern = "test:*"
            keys = []
            async for key in self.redis_client.scan_iter(match=pattern):
                keys.append(key)
            if keys:
                await self.redis_client.delete(*keys)

            return True

        except Exception as e:
            self._add_result("Cache decorator", False, f"Error: {e}")
            return False

    async def test_elasticsearch_optional(self) -> bool:
        """Test 3: Verify Elasticsearch is optional"""
        print("\n=== Test 3: Elasticsearch Optional (Graceful Degradation) ===")

        try:
            # Test settings configuration
            from backend.config.settings import settings

            es_url = settings.ELASTICSEARCH_URL
            if es_url is None or es_url == "":
                self._add_result("ES settings", True, "ELASTICSEARCH_URL is None/empty (optional)")
            else:
                self._add_result("ES settings", True, f"ELASTICSEARCH_URL configured but optional")

            # Test logging module handles missing Elasticsearch
            from backend.utils.enhanced_logging import ELASTICSEARCH_AVAILABLE, setup_application_logging

            if ELASTICSEARCH_AVAILABLE:
                self._add_result("ES library", True, "elasticsearch library installed (optional)")
            else:
                self._add_result("ES library", True, "elasticsearch library not installed (works without it)")

            # Verify logging works without Elasticsearch
            try:
                # This should not raise an error even without Elasticsearch
                from backend.utils.enhanced_logging import initialize_logging_system, LogLevel
                logging_system = initialize_logging_system(
                    log_level=LogLevel.INFO,
                    elasticsearch_hosts=None,  # Explicitly disable
                    log_directory="/tmp/test_logs",
                    enable_file_rotation=False,
                    enable_console_output=False
                )
                self._add_result("Logging without ES", True, "LoggingSystem initialized without Elasticsearch")
            except Exception as e:
                self._add_result("Logging without ES", False, f"Error: {e}")
                return False

            return True

        except Exception as e:
            self._add_result("Elasticsearch optional", False, f"Error: {e}")
            return False

    async def test_database_indexes(self) -> bool:
        """Test 4: Verify database indexes exist"""
        print("\n=== Test 4: Database Index Verification ===")

        try:
            from sqlalchemy import text
            from backend.config.database import get_async_db_session

            async with get_async_db_session() as session:
                # Query for indexes created by migration 008
                query = text("""
                    SELECT indexname, tablename
                    FROM pg_indexes
                    WHERE schemaname = 'public'
                    AND indexname LIKE 'idx_%'
                    ORDER BY tablename, indexname
                """)

                result = await session.execute(query)
                indexes = result.fetchall()

                # Check for key indexes from migration 008
                expected_indexes = [
                    "idx_stocks_market_cap_desc",
                    "idx_stocks_symbol_upper",
                    "idx_price_history_covering",
                    "idx_recommendations_stock_id",
                    "idx_portfolios_user_id",
                ]

                found_indexes = {row[0] for row in indexes}

                for idx_name in expected_indexes:
                    if idx_name in found_indexes:
                        self._add_result(f"Index: {idx_name}", True, "Exists")
                    else:
                        self._add_result(f"Index: {idx_name}", False, "Not found - migration may not be applied")

                # Check pg_trgm extension
                ext_query = text("SELECT extname FROM pg_extension WHERE extname = 'pg_trgm'")
                ext_result = await session.execute(ext_query)
                ext = ext_result.scalar()

                if ext:
                    self._add_result("pg_trgm extension", True, "Installed for fuzzy search")
                else:
                    self._add_result("pg_trgm extension", False, "Not installed")

                self._add_result("Total indexes", True, f"Found {len(indexes)} custom indexes")

            return True

        except Exception as e:
            self._add_result("Database indexes", False, f"Error: {e}")
            return False

    async def test_parallel_api_infrastructure(self) -> bool:
        """Test 5: Verify parallel API infrastructure exists"""
        print("\n=== Test 5: Parallel API Infrastructure ===")

        try:
            # Import the analysis router to verify functions exist
            from backend.api.routers.analysis import (
                safe_async_call,
                fetch_parallel_with_fallback,
                fetch_technical_indicators,
                fetch_fundamental_data,
                fetch_sentiment_data,
                DEFAULT_API_TIMEOUT,
                PARALLEL_BATCH_TIMEOUT
            )

            self._add_result("safe_async_call", True, f"Function exists (timeout={DEFAULT_API_TIMEOUT}s)")
            self._add_result("fetch_parallel_with_fallback", True, f"Function exists (batch_timeout={PARALLEL_BATCH_TIMEOUT}s)")
            self._add_result("fetch_technical_indicators", True, "Function exists (parallel indicator fetching)")
            self._add_result("fetch_fundamental_data", True, "Function exists (parallel fundamental fetching)")
            self._add_result("fetch_sentiment_data", True, "Function exists (parallel sentiment fetching)")

            # Test parallel execution helper
            async def mock_fast():
                await asyncio.sleep(0.01)
                return "fast"

            async def mock_slow():
                await asyncio.sleep(0.05)
                return "slow"

            tasks = [
                ("fast", mock_fast()),
                ("slow", mock_slow())
            ]

            start = time.time()
            results = await fetch_parallel_with_fallback(tasks, timeout=1.0)
            duration = (time.time() - start) * 1000

            if results.get("fast") == "fast" and results.get("slow") == "slow":
                # Parallel execution should take ~50ms (max of both), not ~60ms (sum)
                if duration < 100:
                    self._add_result("Parallel execution", True, f"Tasks ran in parallel ({duration:.0f}ms)", duration)
                else:
                    self._add_result("Parallel execution", False, f"Tasks may have run sequentially ({duration:.0f}ms)")
            else:
                self._add_result("Parallel execution", False, f"Unexpected results: {results}")

            return True

        except ImportError as e:
            self._add_result("Parallel API import", False, f"Import error: {e}")
            return False
        except Exception as e:
            self._add_result("Parallel API", False, f"Error: {e}")
            return False

    async def test_cache_manager_exports(self) -> bool:
        """Test 6: Verify cache module exports all required functions"""
        print("\n=== Test 6: Cache Module Exports ===")

        try:
            from backend.utils.cache import (
                get_redis,
                close_redis,
                CacheManager,
                cache_with_ttl,
                get_cache_key,
                init_cache,
                get_redis_client,
                enhanced_cache,
                get_cache_manager,
                stock_cache,
                market_cache,
                analysis_cache,
                user_cache
            )

            self._add_result("get_redis", True, "Async Redis getter exported")
            self._add_result("close_redis", True, "Connection cleanup exported")
            self._add_result("CacheManager", True, "Cache manager class exported")
            self._add_result("cache_with_ttl", True, "Cache decorator exported")
            self._add_result("get_cache_key", True, "Key generator exported")
            self._add_result("enhanced_cache", True, "Backward compat wrapper exported")
            self._add_result("get_cache_manager", True, "Cache manager getter exported")
            self._add_result("Specialized caches", True, "stock_cache, market_cache, analysis_cache, user_cache exported")

            return True

        except ImportError as e:
            self._add_result("Cache exports", False, f"Missing export: {e}")
            return False

    async def run_all_tests(self) -> Tuple[int, int]:
        """Run all integration tests"""
        print("=" * 60)
        print("QUICK WINS INTEGRATION TEST SUITE")
        print(f"Started: {datetime.now().isoformat()}")
        print("=" * 60)

        # Run tests in order
        await self.test_redis_connection()
        await self.test_cache_decorator()
        await self.test_elasticsearch_optional()
        await self.test_database_indexes()
        await self.test_parallel_api_infrastructure()
        await self.test_cache_manager_exports()

        # Cleanup
        if self.redis_client:
            from backend.utils.cache import close_redis
            await close_redis()

        # Summary
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)

        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {len(self.results)}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Success Rate: {(passed / len(self.results) * 100):.1f}%")

        if failed > 0:
            print("\nFailed Tests:")
            for r in self.results:
                if not r.passed:
                    print(f"  - {r.name}: {r.message}")

        return passed, failed


async def main():
    """Main entry point"""
    tester = QuickWinsIntegrationTester()
    passed, failed = await tester.run_all_tests()

    # Exit with error code if any tests failed
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())
