"""
Performance and Load Testing Suite

This module provides comprehensive performance and load tests for handling
6,000+ stocks with proper resource management and budget constraints.
"""

import pytest
import asyncio
import time
import psutil
import threading
from datetime import datetime, timedelta, date
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, AsyncMock
from concurrent.futures import ThreadPoolExecutor, as_completed
import aiohttp
import resource
import gc
import sys
from memory_profiler import profile
from dataclasses import dataclass
import json

# Import application modules
from backend.analytics.recommendation_engine import RecommendationEngine
from backend.data_ingestion.alpha_vantage_client import AlphaVantageClient
from backend.data_ingestion.finnhub_client import FinnhubClient
from backend.data_ingestion.polygon_client import PolygonClient
from backend.utils.cost_monitor import EnhancedCostMonitor
from backend.utils.cache import CacheManager
from backend.utils.parallel_processor import ParallelStockProcessor
from backend.repositories.stock_repository import StockRepository
from backend.ml.model_manager import ModelManager
from backend.tasks.analysis_tasks import analyze_stock_task
from backend.tests.fixtures.comprehensive_mock_fixtures import (
    mock_external_apis, AlphaVantageMocks, FinnhubMocks, PolygonMocks
)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    total_time: float
    avg_time_per_stock: float
    peak_memory_mb: float
    cpu_usage_percent: float
    api_calls_made: int
    cache_hit_rate: float
    error_rate: float
    throughput_stocks_per_second: float
    concurrent_connections: int


class PerformanceMonitor:
    """Monitor system performance during tests"""
    
    def __init__(self):
        self.start_time = None
        self.initial_memory = None
        self.peak_memory = 0
        self.cpu_samples = []
        self.api_calls = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.errors = 0
        self.monitoring = False
        self._monitor_thread = None
    
    def start_monitoring(self):
        """Start performance monitoring"""
        self.start_time = time.time()
        self.initial_memory = self._get_memory_usage()
        self.monitoring = True
        
        # Start background monitoring thread
        self._monitor_thread = threading.Thread(target=self._monitor_loop)
        self._monitor_thread.start()
    
    def stop_monitoring(self) -> PerformanceMetrics:
        """Stop monitoring and return metrics"""
        self.monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join()
        
        total_time = time.time() - self.start_time
        avg_cpu = np.mean(self.cpu_samples) if self.cpu_samples else 0
        
        total_cache_ops = self.cache_hits + self.cache_misses
        cache_hit_rate = self.cache_hits / max(total_cache_ops, 1)
        
        return PerformanceMetrics(
            total_time=total_time,
            avg_time_per_stock=0,  # Will be calculated by caller
            peak_memory_mb=self.peak_memory,
            cpu_usage_percent=avg_cpu,
            api_calls_made=self.api_calls,
            cache_hit_rate=cache_hit_rate,
            error_rate=0,  # Will be calculated by caller
            throughput_stocks_per_second=0,  # Will be calculated by caller
            concurrent_connections=0  # Will be set by caller
        )
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            # Monitor memory
            current_memory = self._get_memory_usage()
            self.peak_memory = max(self.peak_memory, current_memory)
            
            # Monitor CPU
            cpu_percent = psutil.cpu_percent()
            self.cpu_samples.append(cpu_percent)
            
            time.sleep(0.1)  # Sample every 100ms
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def record_api_call(self):
        """Record an API call"""
        self.api_calls += 1
    
    def record_cache_hit(self):
        """Record a cache hit"""
        self.cache_hits += 1
    
    def record_cache_miss(self):
        """Record a cache miss"""
        self.cache_misses += 1
    
    def record_error(self):
        """Record an error"""
        self.errors += 1


class TestLargeScaleProcessing:
    """Test processing large numbers of stocks"""
    
    @pytest.fixture
    def large_stock_list(self):
        """Generate list of 6000+ stocks for testing"""
        # Mix of real tickers and synthetic ones
        real_tickers = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX',
            'AMD', 'CRM', 'ADBE', 'PYPL', 'INTC', 'CSCO', 'ORCL', 'IBM',
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'V', 'MA',
            'JNJ', 'PFE', 'UNH', 'ABBV', 'TMO', 'DHR', 'ABT', 'BMY',
            'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PXD', 'KMI', 'OKE'
        ]
        
        # Generate synthetic tickers to reach 6000+
        synthetic_tickers = [f'SYN{i:04d}' for i in range(6000)]
        
        all_tickers = real_tickers + synthetic_tickers
        return all_tickers[:6500]  # Slightly over 6000 for stress testing
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_6000_stock_analysis_performance(self, large_stock_list, 
                                                  mock_external_apis):
        """Test analyzing 6000+ stocks within performance constraints"""
        
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        # Setup recommendation engine with optimizations
        rec_engine = RecommendationEngine()
        rec_engine.alpha_vantage_client = mock_external_apis['alpha_vantage']
        rec_engine.finnhub_client = mock_external_apis['finnhub']
        rec_engine.polygon_client = mock_external_apis['polygon']
        
        # Mock cost monitor to allow processing
        cost_monitor = Mock(spec=EnhancedCostMonitor)
        cost_monitor.should_use_cached_data.return_value = False
        cost_monitor.record_api_call.side_effect = monitor.record_api_call
        cost_monitor.get_remaining_budget.return_value = 45.0
        rec_engine.cost_monitor = cost_monitor
        
        # Setup caching with monitoring
        cache_manager = Mock()
        cache_manager.get.side_effect = lambda key: None  # Force cache miss initially
        cache_manager.set.return_value = None
        rec_engine.cache_manager = cache_manager
        
        # Test with parallel processing
        processor = ParallelStockProcessor(
            max_concurrent=100,  # Limit concurrent connections
            batch_size=50,
            rate_limit_per_second=100
        )
        
        start_time = time.time()
        
        # Process stocks in batches to manage memory
        batch_size = 500
        all_results = []
        
        for i in range(0, len(large_stock_list), batch_size):
            batch = large_stock_list[i:i + batch_size]
            
            # Process batch
            batch_results = await processor.process_stocks_batch(
                batch, rec_engine.analyze_stock
            )
            
            all_results.extend(batch_results)
            
            # Force garbage collection between batches
            gc.collect()
            
            print(f"Processed batch {i//batch_size + 1}: {len(batch)} stocks")
        
        total_time = time.time() - start_time
        
        # Stop monitoring and get metrics
        metrics = monitor.stop_monitoring()
        metrics.total_time = total_time
        metrics.avg_time_per_stock = total_time / len(large_stock_list)
        metrics.throughput_stocks_per_second = len(large_stock_list) / total_time
        metrics.concurrent_connections = processor.max_concurrent
        
        # Calculate error rate
        successful_results = [r for r in all_results if r is not None]
        metrics.error_rate = (len(large_stock_list) - len(successful_results)) / len(large_stock_list)
        
        # Performance assertions
        assert len(all_results) == len(large_stock_list), "Not all stocks were processed"
        assert metrics.error_rate < 0.05, f"Error rate too high: {metrics.error_rate}"
        assert metrics.peak_memory_mb < 2048, f"Memory usage too high: {metrics.peak_memory_mb}MB"
        assert metrics.total_time < 3600, f"Processing took too long: {metrics.total_time}s"  # 1 hour limit
        assert metrics.throughput_stocks_per_second > 2, f"Throughput too low: {metrics.throughput_stocks_per_second} stocks/s"
        
        print(f"Performance Metrics:")
        print(f"  Total time: {metrics.total_time:.2f}s")
        print(f"  Avg time per stock: {metrics.avg_time_per_stock:.3f}s")
        print(f"  Peak memory: {metrics.peak_memory_mb:.1f}MB")
        print(f"  Throughput: {metrics.throughput_stocks_per_second:.2f} stocks/s")
        print(f"  Error rate: {metrics.error_rate:.3f}")
        print(f"  API calls: {metrics.api_calls_made}")
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_memory_efficiency_large_scale(self, large_stock_list):
        """Test memory efficiency when processing large numbers of stocks"""
        
        # Monitor memory usage throughout the test
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        max_memory = initial_memory
        memory_samples = [initial_memory]
        
        def monitor_memory():
            nonlocal max_memory
            while True:
                current = psutil.Process().memory_info().rss / 1024 / 1024
                memory_samples.append(current)
                max_memory = max(max_memory, current)
                time.sleep(0.1)
        
        # Start memory monitoring
        monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
        monitor_thread.start()
        
        # Setup lightweight processing
        processed_count = 0
        
        # Process stocks in small batches to test memory management
        batch_size = 100
        
        for i in range(0, min(2000, len(large_stock_list)), batch_size):  # Test with 2000 stocks
            batch = large_stock_list[i:i + batch_size]
            
            # Simulate lightweight processing
            batch_data = []
            for ticker in batch:
                # Create minimal data structure
                data = {
                    'ticker': ticker,
                    'score': np.random.random(),
                    'processed_at': datetime.now()
                }
                batch_data.append(data)
                processed_count += 1
            
            # Clear batch data to free memory
            del batch_data
            
            # Force garbage collection
            gc.collect()
            
            # Check memory growth
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_growth = current_memory - initial_memory
            
            # Memory growth should be reasonable
            assert memory_growth < 500, f"Memory growth too high: {memory_growth}MB after {processed_count} stocks"
        
        # Final memory check
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        total_growth = final_memory - initial_memory
        memory_per_stock = total_growth / processed_count if processed_count > 0 else 0
        
        print(f"Memory efficiency metrics:")
        print(f"  Initial memory: {initial_memory:.1f}MB")
        print(f"  Final memory: {final_memory:.1f}MB")
        print(f"  Peak memory: {max_memory:.1f}MB")
        print(f"  Total growth: {total_growth:.1f}MB")
        print(f"  Memory per stock: {memory_per_stock:.3f}MB")
        
        # Assertions
        assert total_growth < 1000, f"Total memory growth too high: {total_growth}MB"
        assert memory_per_stock < 0.1, f"Memory per stock too high: {memory_per_stock}MB"


class TestDatabasePerformance:
    """Test database performance with large datasets"""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_bulk_stock_operations(self, db_session):
        """Test bulk database operations performance"""
        
        from backend.repositories.stock_repository import StockRepository
        
        stock_repo = StockRepository(db_session)
        
        # Generate large dataset
        stock_data_list = []
        for i in range(10000):  # 10k stocks for testing
            stock_data_list.append({
                'ticker': f'BULK{i:05d}',
                'company_name': f'Bulk Test Company {i}',
                'sector': 'Technology',
                'industry': 'Software',
                'market_cap': np.random.uniform(1e9, 100e9),
                'shares_outstanding': np.random.uniform(1e6, 1e9),
                'is_active': True
            })
        
        # Test bulk insert performance
        start_time = time.time()
        created_stocks = await stock_repo.bulk_create_stocks(stock_data_list)
        insert_time = time.time() - start_time
        
        assert len(created_stocks) == 10000
        assert insert_time < 30.0, f"Bulk insert took too long: {insert_time}s"
        
        insert_rate = len(stock_data_list) / insert_time
        print(f"Bulk insert rate: {insert_rate:.0f} stocks/second")
        
        # Test bulk query performance
        start_time = time.time()
        retrieved_stocks = await stock_repo.get_active_stocks(limit=10000)
        query_time = time.time() - start_time
        
        assert len(retrieved_stocks) == 10000
        assert query_time < 5.0, f"Bulk query took too long: {query_time}s"
        
        query_rate = len(retrieved_stocks) / query_time
        print(f"Bulk query rate: {query_rate:.0f} stocks/second")
        
        # Test filtered query performance
        start_time = time.time()
        tech_stocks = await stock_repo.get_stocks_by_sector('Technology', limit=10000)
        filter_time = time.time() - start_time
        
        assert len(tech_stocks) == 10000
        assert filter_time < 3.0, f"Filtered query took too long: {filter_time}s"
        
        # Test pagination performance
        start_time = time.time()
        
        all_paginated = []
        page_size = 100
        for page in range(100):  # 100 pages of 100 stocks each
            paginated_stocks = await stock_repo.get_active_stocks(
                offset=page * page_size,
                limit=page_size
            )
            all_paginated.extend(paginated_stocks)
        
        pagination_time = time.time() - start_time
        
        assert len(all_paginated) == 10000
        assert pagination_time < 10.0, f"Pagination took too long: {pagination_time}s"
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_price_history_performance(self, db_session):
        """Test price history operations performance"""
        
        from backend.repositories.price_repository import PriceRepository
        from backend.repositories.stock_repository import StockRepository
        
        stock_repo = StockRepository(db_session)
        price_repo = PriceRepository(db_session)
        
        # Create test stock
        stock_data = {
            'ticker': 'PRICETEST',
            'company_name': 'Price Test Company',
            'sector': 'Technology',
            'is_active': True
        }
        stock = await stock_repo.create_stock(stock_data)
        
        # Generate large price history dataset (5 years of daily data)
        dates = pd.date_range(end=date.today(), periods=1825, freq='D')  # 5 years
        price_records = []
        
        base_price = 100.0
        current_price = base_price
        
        for date_val in dates:
            # Generate realistic price movement
            daily_return = np.random.normal(0.0005, 0.02)
            current_price *= (1 + daily_return)
            
            open_price = current_price * (1 + np.random.normal(0, 0.005))
            high_price = max(open_price, current_price) * (1 + abs(np.random.normal(0, 0.01)))
            low_price = min(open_price, current_price) * (1 - abs(np.random.normal(0, 0.01)))
            volume = int(np.random.lognormal(14, 0.5))
            
            price_records.append({
                'stock_id': stock.id,
                'ticker': 'PRICETEST',
                'date': date_val.date(),
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(current_price, 2),
                'volume': volume,
                'adj_close': round(current_price, 2)
            })
        
        # Test bulk price insert performance
        start_time = time.time()
        
        # Insert in batches to avoid memory issues
        batch_size = 500
        for i in range(0, len(price_records), batch_size):
            batch = price_records[i:i + batch_size]
            await price_repo.bulk_create_price_records(batch)
        
        insert_time = time.time() - start_time
        
        assert insert_time < 60.0, f"Price history insert took too long: {insert_time}s"
        
        insert_rate = len(price_records) / insert_time
        print(f"Price history insert rate: {insert_rate:.0f} records/second")
        
        # Test price history query performance
        start_time = time.time()
        recent_prices = await price_repo.get_price_history(stock.id, days=252)  # 1 year
        query_time = time.time() - start_time
        
        assert len(recent_prices) == 252
        assert query_time < 1.0, f"Price history query took too long: {query_time}s"
        
        # Test aggregate query performance  
        start_time = time.time()
        monthly_aggregates = await price_repo.get_monthly_aggregates(stock.id, months=12)
        agg_time = time.time() - start_time
        
        assert len(monthly_aggregates) == 12
        assert agg_time < 2.0, f"Monthly aggregates took too long: {agg_time}s"


class TestCachePerformance:
    """Test caching system performance"""
    
    @pytest.mark.performance
    def test_cache_throughput(self, test_redis):
        """Test cache throughput performance"""
        
        cache_manager = CacheManager(redis_client=test_redis.client)
        
        # Test write performance
        num_keys = 10000
        start_time = time.time()
        
        for i in range(num_keys):
            cache_key = f"perf_test:{i}"
            cache_data = {
                'ticker': f'TEST{i:04d}',
                'price': 100 + i * 0.1,
                'volume': 1000000 + i * 1000,
                'timestamp': datetime.now().isoformat()
            }
            cache_manager.set(cache_key, cache_data, ttl=3600)
        
        write_time = time.time() - start_time
        write_rate = num_keys / write_time
        
        print(f"Cache write rate: {write_rate:.0f} ops/second")
        assert write_rate > 1000, f"Cache write rate too low: {write_rate} ops/s"
        
        # Test read performance
        start_time = time.time()
        
        for i in range(num_keys):
            cache_key = f"perf_test:{i}"
            cached_data = cache_manager.get(cache_key)
            assert cached_data is not None
        
        read_time = time.time() - start_time
        read_rate = num_keys / read_time
        
        print(f"Cache read rate: {read_rate:.0f} ops/second")
        assert read_rate > 2000, f"Cache read rate too low: {read_rate} ops/s"
        
        # Test mixed workload
        start_time = time.time()
        
        for i in range(num_keys):
            if i % 3 == 0:  # 33% writes
                cache_key = f"mixed_test:{i}"
                cache_data = {'value': i}
                cache_manager.set(cache_key, cache_data, ttl=3600)
            else:  # 67% reads
                cache_key = f"perf_test:{i % 1000}"  # Read from existing keys
                cache_manager.get(cache_key)
        
        mixed_time = time.time() - start_time
        mixed_rate = num_keys / mixed_time
        
        print(f"Cache mixed workload rate: {mixed_rate:.0f} ops/second")
        assert mixed_rate > 1500, f"Mixed workload rate too low: {mixed_rate} ops/s"
    
    @pytest.mark.performance
    def test_cache_memory_efficiency(self, test_redis):
        """Test cache memory efficiency"""
        
        cache_manager = CacheManager(redis_client=test_redis.client)
        
        # Get initial memory usage
        initial_memory = test_redis.client.memory_usage("non_existent_key") or 0
        
        # Store large amount of data
        num_entries = 100000
        entry_size_bytes = 1024  # 1KB per entry
        
        for i in range(num_entries):
            cache_key = f"memory_test:{i:06d}"
            # Create 1KB data entry
            cache_data = {
                'id': i,
                'data': 'x' * (entry_size_bytes - 100),  # Account for JSON overhead
                'timestamp': datetime.now().isoformat()
            }
            cache_manager.set(cache_key, cache_data, ttl=3600)
            
            # Sample memory usage periodically
            if i % 10000 == 0 and i > 0:
                current_memory = test_redis.client.info('memory')['used_memory']
                print(f"After {i} entries: {current_memory / 1024 / 1024:.1f}MB")
        
        # Check final memory usage
        final_memory = test_redis.client.info('memory')['used_memory']
        memory_per_entry = (final_memory - initial_memory) / num_entries
        
        print(f"Memory per cache entry: {memory_per_entry:.0f} bytes")
        
        # Should be reasonably efficient (allowing for Redis overhead)
        assert memory_per_entry < entry_size_bytes * 1.5, \
            f"Cache memory overhead too high: {memory_per_entry} bytes per entry"


class TestAPIRateLimitingPerformance:
    """Test API rate limiting performance"""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_rate_limited_throughput(self, mock_external_apis):
        """Test throughput with API rate limiting"""
        
        from backend.data_ingestion.alpha_vantage_client import AlphaVantageClient
        
        # Setup client with strict rate limiting
        client = AlphaVantageClient(api_key="test")
        client._session = mock_external_apis['alpha_vantage']
        
        # Override rate limits for testing
        client.calls_per_minute = 60  # 1 call per second
        client.calls_per_day = 500
        
        # Test sustained throughput
        tickers = [f'TEST{i:03d}' for i in range(100)]
        
        start_time = time.time()
        
        # Process with rate limiting
        tasks = []
        for ticker in tickers:
            task = client.get_daily_prices(ticker)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        total_time = time.time() - start_time
        successful_calls = len([r for r in results if not isinstance(r, Exception)])
        
        print(f"Rate limited throughput: {successful_calls / total_time:.2f} calls/second")
        print(f"Success rate: {successful_calls / len(tickers):.3f}")
        
        # Should respect rate limits while maintaining reasonable throughput
        assert successful_calls == len(tickers), "Some API calls failed"
        assert total_time >= (len(tickers) - 1), "Rate limiting not enforced properly"
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_cost_budget_enforcement_performance(self):
        """Test cost budget enforcement doesn't significantly impact performance"""
        
        from backend.utils.enhanced_cost_monitor import EnhancedCostMonitor
        
        # Setup cost monitor
        cost_monitor = EnhancedCostMonitor(monthly_budget=50.0)
        
        # Simulate many API calls with budget checking
        num_calls = 10000
        start_time = time.time()
        
        for i in range(num_calls):
            # Check budget before each call (realistic usage)
            can_make_call = not cost_monitor.should_use_cached_data('test_provider')
            
            if can_make_call:
                # Record API call
                cost_monitor.record_api_call('test_provider', cost=0.001)
            
            # Periodic budget checks
            if i % 100 == 0:
                alerts = cost_monitor.check_budget_alerts()
                usage = cost_monitor.get_current_usage()
        
        budget_check_time = time.time() - start_time
        calls_per_second = num_calls / budget_check_time
        
        print(f"Budget checking throughput: {calls_per_second:.0f} calls/second")
        
        # Budget checking should be fast
        assert calls_per_second > 5000, f"Budget checking too slow: {calls_per_second} calls/s"
        assert budget_check_time < 5.0, f"Budget checking took too long: {budget_check_time}s"


class TestConcurrencyPerformance:
    """Test concurrent processing performance"""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_stock_analysis(self, mock_external_apis):
        """Test concurrent stock analysis performance"""
        
        from backend.analytics.recommendation_engine import RecommendationEngine
        
        # Setup recommendation engine
        rec_engine = RecommendationEngine()
        rec_engine.alpha_vantage_client = mock_external_apis['alpha_vantage']
        rec_engine.finnhub_client = mock_external_apis['finnhub']
        
        # Test different concurrency levels
        concurrency_levels = [1, 5, 10, 20, 50, 100]
        test_stocks = [f'CONC{i:03d}' for i in range(100)]
        
        results = {}
        
        for concurrency in concurrency_levels:
            # Limit concurrent tasks
            semaphore = asyncio.Semaphore(concurrency)
            
            async def analyze_with_semaphore(ticker):
                async with semaphore:
                    return await rec_engine.analyze_stock(ticker)
            
            start_time = time.time()
            
            tasks = [analyze_with_semaphore(ticker) for ticker in test_stocks]
            analysis_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            total_time = time.time() - start_time
            successful_analyses = len([r for r in analysis_results if not isinstance(r, Exception)])
            throughput = successful_analyses / total_time
            
            results[concurrency] = {
                'time': total_time,
                'throughput': throughput,
                'success_rate': successful_analyses / len(test_stocks)
            }
            
            print(f"Concurrency {concurrency:2d}: {throughput:.1f} analyses/s, "
                  f"{results[concurrency]['success_rate']:.3f} success rate")
        
        # Find optimal concurrency level
        optimal_concurrency = max(results.keys(), 
                                key=lambda k: results[k]['throughput'])
        
        print(f"Optimal concurrency level: {optimal_concurrency}")
        
        # Verify performance improves with concurrency (up to a point)
        assert results[10]['throughput'] > results[1]['throughput'], \
            "Concurrency should improve throughput"
        
        # Verify we don't degrade too much at high concurrency
        assert results[100]['success_rate'] > 0.9, \
            "High concurrency should maintain good success rate"
    
    @pytest.mark.performance
    def test_thread_pool_performance(self):
        """Test thread pool performance for CPU-bound tasks"""
        
        def cpu_intensive_task(n):
            """Simulate CPU-intensive calculation"""
            result = 0
            for i in range(n):
                result += i ** 0.5
            return result
        
        # Test different thread pool sizes
        pool_sizes = [1, 2, 4, 8, 16]
        task_count = 100
        task_size = 100000
        
        for pool_size in pool_sizes:
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=pool_size) as executor:
                futures = [executor.submit(cpu_intensive_task, task_size) 
                          for _ in range(task_count)]
                
                results = [future.result() for future in as_completed(futures)]
            
            total_time = time.time() - start_time
            throughput = task_count / total_time
            
            print(f"Thread pool size {pool_size:2d}: {throughput:.1f} tasks/s")
            
            # Verify all tasks completed
            assert len(results) == task_count
        
        # Multi-threading should provide some improvement
        # (though limited by Python GIL for CPU-bound tasks)


class TestResourceUtilization:
    """Test system resource utilization"""
    
    @pytest.mark.performance
    def test_cpu_utilization_efficiency(self):
        """Test CPU utilization efficiency"""
        
        def monitor_cpu_usage(duration_seconds=10):
            """Monitor CPU usage over time"""
            cpu_samples = []
            start_time = time.time()
            
            while time.time() - start_time < duration_seconds:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                cpu_samples.append(cpu_percent)
            
            return cpu_samples
        
        # Baseline CPU usage
        baseline_cpu = monitor_cpu_usage(2)
        baseline_avg = np.mean(baseline_cpu)
        
        print(f"Baseline CPU usage: {baseline_avg:.1f}%")
        
        # CPU usage during processing
        import threading
        
        def cpu_intensive_work():
            # Simulate stock analysis work
            for _ in range(1000):
                data = np.random.random(1000)
                np.mean(data)
                np.std(data)
                np.corrcoef(data[:-1], data[1:])[0, 1]
        
        # Start background work
        work_thread = threading.Thread(target=cpu_intensive_work)
        work_thread.start()
        
        # Monitor CPU during work
        work_cpu = monitor_cpu_usage(5)
        work_avg = np.mean(work_cpu)
        
        work_thread.join()
        
        print(f"CPU usage during work: {work_avg:.1f}%")
        
        # CPU should be utilized but not saturated
        cpu_increase = work_avg - baseline_avg
        assert cpu_increase > 5, f"CPU utilization too low: {cpu_increase:.1f}% increase"
        assert work_avg < 90, f"CPU utilization too high: {work_avg:.1f}%"
    
    @pytest.mark.performance
    def test_memory_growth_pattern(self):
        """Test memory growth patterns during processing"""
        
        import gc
        
        # Initial memory
        gc.collect()  # Clean start
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        memory_samples = [initial_memory]
        
        # Simulate processing batches
        batch_size = 1000
        num_batches = 10
        
        for batch in range(num_batches):
            # Create batch data
            batch_data = []
            for i in range(batch_size):
                item = {
                    'id': batch * batch_size + i,
                    'data': np.random.random(100).tolist(),
                    'metadata': {'processed': True, 'batch': batch}
                }
                batch_data.append(item)
            
            # Process batch (simulate)
            processed_count = len(batch_data)
            
            # Clean up batch
            del batch_data
            gc.collect()
            
            # Sample memory
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_samples.append(current_memory)
            
            print(f"Batch {batch + 1}: {current_memory:.1f}MB "
                  f"(+{current_memory - initial_memory:.1f}MB)")
        
        final_memory = memory_samples[-1]
        max_memory = max(memory_samples)
        memory_growth = final_memory - initial_memory
        peak_growth = max_memory - initial_memory
        
        print(f"Memory growth: {memory_growth:.1f}MB final, {peak_growth:.1f}MB peak")
        
        # Memory growth should be reasonable
        items_processed = num_batches * batch_size
        memory_per_item = memory_growth / items_processed * 1024 * 1024  # bytes
        
        assert memory_per_item < 1000, \
            f"Memory per item too high: {memory_per_item:.0f} bytes"
        assert memory_growth < 200, \
            f"Total memory growth too high: {memory_growth:.1f}MB"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "performance"])