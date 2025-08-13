"""
Database performance testing and validation utilities
"""

import time
import random
import statistics
from typing import List, Dict, Any, Callable, Optional
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from sqlalchemy.orm import Session
from sqlalchemy import text
from dataclasses import dataclass

from backend.utils.database import get_db_sync
from backend.utils.optimized_queries import OptimizedQueryManager
from backend.utils.database_monitoring import DatabaseMonitor

logger = logging.getLogger(__name__)


@dataclass
class PerformanceResult:
    """Performance test result"""
    test_name: str
    duration_ms: float
    success: bool
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class BenchmarkResult:
    """Benchmark test result"""
    test_name: str
    iterations: int
    avg_duration_ms: float
    min_duration_ms: float
    max_duration_ms: float
    std_dev_ms: float
    success_rate: float
    throughput_ops_per_sec: float


class DatabasePerformanceTester:
    """Comprehensive database performance testing"""
    
    def __init__(self):
        self.results: List[PerformanceResult] = []
        self.benchmarks: List[BenchmarkResult] = []
    
    def run_single_query_test(
        self, 
        test_name: str, 
        query_func: Callable[[], Any], 
        expected_max_duration_ms: float = 1000
    ) -> PerformanceResult:
        """Run a single query performance test"""
        
        start_time = time.time()
        success = True
        error = None
        
        try:
            result = query_func()
            duration_ms = (time.time() - start_time) * 1000
            
            metadata = {
                'expected_max_ms': expected_max_duration_ms,
                'within_threshold': duration_ms <= expected_max_duration_ms,
                'result_count': len(result) if hasattr(result, '__len__') else 1
            }
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            success = False
            error = str(e)
            metadata = {'expected_max_ms': expected_max_duration_ms}
        
        result = PerformanceResult(
            test_name=test_name,
            duration_ms=duration_ms,
            success=success,
            error=error,
            metadata=metadata
        )
        
        self.results.append(result)
        return result
    
    def run_benchmark_test(
        self, 
        test_name: str, 
        query_func: Callable[[], Any], 
        iterations: int = 100,
        max_workers: int = 5
    ) -> BenchmarkResult:
        """Run benchmark test with multiple iterations"""
        
        durations = []
        successes = 0
        
        def run_iteration():
            start_time = time.time()
            try:
                query_func()
                duration = (time.time() - start_time) * 1000
                return duration, True
            except Exception as e:
                duration = (time.time() - start_time) * 1000
                logger.error(f"Benchmark iteration failed: {e}")
                return duration, False
        
        # Run iterations with thread pool
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_iteration = {
                executor.submit(run_iteration): i 
                for i in range(iterations)
            }
            
            for future in as_completed(future_to_iteration):
                duration, success = future.result()
                durations.append(duration)
                if success:
                    successes += 1
        
        # Calculate statistics
        avg_duration = statistics.mean(durations)
        min_duration = min(durations)
        max_duration = max(durations)
        std_dev = statistics.stdev(durations) if len(durations) > 1 else 0
        success_rate = successes / iterations
        throughput = iterations / (sum(durations) / 1000) if sum(durations) > 0 else 0
        
        benchmark = BenchmarkResult(
            test_name=test_name,
            iterations=iterations,
            avg_duration_ms=avg_duration,
            min_duration_ms=min_duration,
            max_duration_ms=max_duration,
            std_dev_ms=std_dev,
            success_rate=success_rate,
            throughput_ops_per_sec=throughput
        )
        
        self.benchmarks.append(benchmark)
        return benchmark
    
    def test_index_performance(self) -> List[PerformanceResult]:
        """Test performance of database indexes"""
        
        index_tests = []
        
        def test_stock_lookup():
            """Test stock lookup by ticker (should use index)"""
            db = get_db_sync()
            try:
                result = db.execute(
                    text("SELECT * FROM stocks WHERE ticker = 'AAPL'")
                ).fetchall()
                return result
            finally:
                db.close()
        
        def test_price_history_range():
            """Test price history date range query (should use index)"""
            db = get_db_sync()
            try:
                result = db.execute(
                    text("""
                        SELECT * FROM price_history 
                        WHERE stock_id = 1 
                        AND date >= CURRENT_DATE - INTERVAL '30 days'
                        ORDER BY date DESC
                        LIMIT 30
                    """)
                ).fetchall()
                return result
            finally:
                db.close()
        
        def test_recommendations_active():
            """Test active recommendations query (should use index)"""
            db = get_db_sync()
            try:
                result = db.execute(
                    text("""
                        SELECT * FROM recommendations 
                        WHERE is_active = true 
                        AND confidence_score > 0.7
                        ORDER BY confidence_score DESC
                        LIMIT 20
                    """)
                ).fetchall()
                return result
            finally:
                db.close()
        
        def test_technical_indicators_latest():
            """Test latest technical indicators (should use index)"""
            db = get_db_sync()
            try:
                result = db.execute(
                    text("""
                        SELECT DISTINCT ON (stock_id) *
                        FROM technical_indicators 
                        WHERE stock_id IN (1, 2, 3, 4, 5)
                        ORDER BY stock_id, date DESC
                    """)
                ).fetchall()
                return result
            finally:
                db.close()
        
        # Run index performance tests
        index_tests.extend([
            self.run_single_query_test("Stock Ticker Lookup", test_stock_lookup, 50),
            self.run_single_query_test("Price History Range", test_price_history_range, 100),
            self.run_single_query_test("Active Recommendations", test_recommendations_active, 100),
            self.run_single_query_test("Latest Technical Indicators", test_technical_indicators_latest, 200)
        ])
        
        return index_tests
    
    def test_optimized_queries(self) -> List[PerformanceResult]:
        """Test performance of optimized query patterns"""
        
        optimized_tests = []
        
        def test_batch_stock_data():
            """Test batch stock data retrieval"""
            db = get_db_sync()
            try:
                query_manager = OptimizedQueryManager(db)
                result = query_manager.get_stocks_with_latest_data(
                    stock_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    include_price=True,
                    include_technical=True,
                    include_fundamentals=True
                )
                return result
            finally:
                db.close()
        
        def test_recommendations_with_context():
            """Test recommendations with context query"""
            db = get_db_sync()
            try:
                query_manager = OptimizedQueryManager(db)
                result = query_manager.get_recommendations_with_context(
                    limit=20,
                    min_confidence=0.6
                )
                return result
            finally:
                db.close()
        
        def test_sector_analysis():
            """Test sector analysis batch query"""
            db = get_db_sync()
            try:
                query_manager = OptimizedQueryManager(db)
                result = query_manager.get_sector_analysis_batch([1, 2, 3])
                return result
            finally:
                db.close()
        
        optimized_tests.extend([
            self.run_single_query_test("Batch Stock Data", test_batch_stock_data, 500),
            self.run_single_query_test("Recommendations with Context", test_recommendations_with_context, 300),
            self.run_single_query_test("Sector Analysis Batch", test_sector_analysis, 800)
        ])
        
        return optimized_tests
    
    def test_connection_pool_performance(self) -> List[BenchmarkResult]:
        """Test connection pool performance under load"""
        
        def concurrent_query():
            """Simulate concurrent database query"""
            db = get_db_sync()
            try:
                # Simple query to test connection acquisition
                result = db.execute(text("SELECT COUNT(*) FROM stocks")).fetchone()
                return result[0]
            finally:
                db.close()
        
        def batch_queries():
            """Simulate batch of queries"""
            db = get_db_sync()
            try:
                results = []
                for i in range(5):
                    result = db.execute(
                        text("SELECT ticker FROM stocks WHERE is_active = true LIMIT 10")
                    ).fetchall()
                    results.append(result)
                return results
            finally:
                db.close()
        
        pool_benchmarks = [
            self.run_benchmark_test("Concurrent Simple Queries", concurrent_query, 50, 10),
            self.run_benchmark_test("Batch Queries", batch_queries, 20, 5)
        ]
        
        return pool_benchmarks
    
    def test_large_dataset_queries(self) -> List[PerformanceResult]:
        """Test queries on large datasets"""
        
        large_data_tests = []
        
        def test_full_table_aggregation():
            """Test full table aggregation performance"""
            db = get_db_sync()
            try:
                result = db.execute(
                    text("""
                        SELECT 
                            stock_id,
                            COUNT(*) as record_count,
                            AVG(close) as avg_price,
                            MAX(high) as max_high,
                            MIN(low) as min_low
                        FROM price_history 
                        WHERE date >= CURRENT_DATE - INTERVAL '90 days'
                        GROUP BY stock_id
                        ORDER BY avg_price DESC
                        LIMIT 100
                    """)
                ).fetchall()
                return result
            finally:
                db.close()
        
        def test_complex_join_query():
            """Test complex join query performance"""
            db = get_db_sync()
            try:
                result = db.execute(
                    text("""
                        SELECT 
                            s.ticker,
                            s.name,
                            ph.close,
                            ti.rsi_14,
                            r.action,
                            r.confidence_score
                        FROM stocks s
                        LEFT JOIN price_history ph ON s.id = ph.stock_id
                        LEFT JOIN technical_indicators ti ON s.id = ti.stock_id AND ti.date = ph.date
                        LEFT JOIN recommendations r ON s.id = r.stock_id AND r.is_active = true
                        WHERE s.is_active = true 
                          AND ph.date >= CURRENT_DATE - INTERVAL '7 days'
                        ORDER BY ph.date DESC, s.ticker
                        LIMIT 1000
                    """)
                ).fetchall()
                return result
            finally:
                db.close()
        
        large_data_tests.extend([
            self.run_single_query_test("Full Table Aggregation", test_full_table_aggregation, 2000),
            self.run_single_query_test("Complex Join Query", test_complex_join_query, 3000)
        ])
        
        return large_data_tests
    
    def validate_partition_performance(self) -> List[PerformanceResult]:
        """Validate performance improvements from partitioning"""
        
        partition_tests = []
        
        def test_date_range_partition_pruning():
            """Test partition pruning on date range queries"""
            db = get_db_sync()
            try:
                # This should benefit from partition pruning
                result = db.execute(
                    text("""
                        EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
                        SELECT * FROM price_history 
                        WHERE date >= CURRENT_DATE - INTERVAL '7 days'
                        AND stock_id = 1
                    """)
                ).fetchone()
                
                # Check if partition pruning occurred
                explain_data = result[0][0] if result else {}
                return explain_data
            finally:
                db.close()
        
        def test_old_data_query():
            """Test query on older data (should use different partitions)"""
            db = get_db_sync()
            try:
                result = db.execute(
                    text("""
                        SELECT COUNT(*) 
                        FROM price_history 
                        WHERE date BETWEEN CURRENT_DATE - INTERVAL '60 days' 
                                     AND CURRENT_DATE - INTERVAL '30 days'
                    """)
                ).fetchone()
                return result[0]
            finally:
                db.close()
        
        partition_tests.extend([
            self.run_single_query_test("Partition Pruning Test", test_date_range_partition_pruning, 100),
            self.run_single_query_test("Old Data Query", test_old_data_query, 500)
        ])
        
        return partition_tests
    
    def run_comprehensive_performance_test(self) -> Dict[str, Any]:
        """Run comprehensive performance test suite"""
        
        logger.info("Starting comprehensive database performance test...")
        start_time = time.time()
        
        # Clear previous results
        self.results = []
        self.benchmarks = []
        
        try:
            # Run all test categories
            logger.info("Testing index performance...")
            index_results = self.test_index_performance()
            
            logger.info("Testing optimized queries...")
            optimized_results = self.test_optimized_queries()
            
            logger.info("Testing connection pool performance...")
            pool_benchmarks = self.test_connection_pool_performance()
            
            logger.info("Testing large dataset queries...")
            large_data_results = self.test_large_dataset_queries()
            
            logger.info("Validating partition performance...")
            partition_results = self.validate_partition_performance()
            
            # Compile summary
            total_duration = (time.time() - start_time)
            
            summary = {
                'test_completed_at': datetime.now().isoformat(),
                'total_test_duration_seconds': total_duration,
                'total_tests_run': len(self.results),
                'total_benchmarks_run': len(self.benchmarks),
                'success_rate': sum(1 for r in self.results if r.success) / len(self.results) if self.results else 0,
                'avg_query_duration_ms': statistics.mean([r.duration_ms for r in self.results if r.success]) if self.results else 0,
                'failed_tests': [r.test_name for r in self.results if not r.success],
                'slow_tests': [
                    {'name': r.test_name, 'duration_ms': r.duration_ms} 
                    for r in self.results 
                    if r.success and r.metadata and not r.metadata.get('within_threshold', True)
                ],
                'performance_summary': {
                    'index_tests': {
                        'count': len(index_results),
                        'avg_duration_ms': statistics.mean([r.duration_ms for r in index_results if r.success]) if index_results else 0,
                        'success_rate': sum(1 for r in index_results if r.success) / len(index_results) if index_results else 0
                    },
                    'optimized_queries': {
                        'count': len(optimized_results),
                        'avg_duration_ms': statistics.mean([r.duration_ms for r in optimized_results if r.success]) if optimized_results else 0,
                        'success_rate': sum(1 for r in optimized_results if r.success) / len(optimized_results) if optimized_results else 0
                    },
                    'connection_pool': {
                        'benchmarks_count': len(pool_benchmarks),
                        'avg_throughput_ops_per_sec': statistics.mean([b.throughput_ops_per_sec for b in pool_benchmarks]) if pool_benchmarks else 0
                    },
                    'large_datasets': {
                        'count': len(large_data_results),
                        'avg_duration_ms': statistics.mean([r.duration_ms for r in large_data_results if r.success]) if large_data_results else 0
                    }
                },
                'detailed_results': [
                    {
                        'test_name': r.test_name,
                        'duration_ms': r.duration_ms,
                        'success': r.success,
                        'error': r.error,
                        'within_threshold': r.metadata.get('within_threshold', True) if r.metadata else True
                    }
                    for r in self.results
                ],
                'benchmark_results': [
                    {
                        'test_name': b.test_name,
                        'iterations': b.iterations,
                        'avg_duration_ms': b.avg_duration_ms,
                        'throughput_ops_per_sec': b.throughput_ops_per_sec,
                        'success_rate': b.success_rate
                    }
                    for b in self.benchmarks
                ]
            }
            
            logger.info("Performance testing completed successfully")
            return summary
            
        except Exception as e:
            logger.error(f"Performance testing failed: {e}")
            return {
                'error': str(e),
                'test_completed_at': datetime.now().isoformat(),
                'total_test_duration_seconds': time.time() - start_time
            }


def run_performance_validation() -> Dict[str, Any]:
    """Convenience function to run performance validation"""
    
    tester = DatabasePerformanceTester()
    return tester.run_comprehensive_performance_test()


def validate_database_optimizations() -> Dict[str, Any]:
    """Validate that database optimizations are working correctly"""
    
    logger.info("Validating database optimizations...")
    
    validation_results = {
        'timestamp': datetime.now().isoformat(),
        'validations': {}
    }
    
    db = get_db_sync()
    try:
        monitor = DatabaseMonitor(db)
        
        # Check connection pool status
        from backend.utils.database import get_connection_pool_status
        pool_status = get_connection_pool_status()
        validation_results['validations']['connection_pool'] = {
            'pool_size': pool_status['size'],
            'utilization_percent': pool_status['utilization_percent'],
            'status': 'OK' if pool_status['utilization_percent'] < 80 else 'WARNING'
        }
        
        # Check table statistics
        table_stats = monitor.get_table_statistics()
        validation_results['validations']['table_health'] = {}
        
        for table in ['price_history', 'technical_indicators', 'recommendations']:
            table_stat = next((t for t in table_stats if table in t.table_name), None)
            if table_stat:
                validation_results['validations']['table_health'][table] = {
                    'row_count': table_stat.row_count,
                    'sequential_scans': table_stat.sequential_scans,
                    'index_scans': table_stat.index_scans,
                    'dead_tuples': table_stat.dead_tuples,
                    'status': 'OK' if table_stat.index_scans > table_stat.sequential_scans else 'WARNING'
                }
        
        # Check index usage
        index_usage = monitor.get_index_usage_statistics()
        unused_indexes = [idx for idx in index_usage if idx['usage_category'] == 'UNUSED']
        validation_results['validations']['indexes'] = {
            'total_indexes': len(index_usage),
            'unused_indexes': len(unused_indexes),
            'unused_index_names': [idx['index'] for idx in unused_indexes]
        }
        
        # Run quick performance test
        tester = DatabasePerformanceTester()
        quick_results = tester.test_index_performance()
        validation_results['validations']['performance'] = {
            'avg_query_time_ms': statistics.mean([r.duration_ms for r in quick_results if r.success]) if quick_results else 0,
            'all_tests_passed': all(r.success for r in quick_results),
            'within_thresholds': all(
                r.metadata.get('within_threshold', True) 
                for r in quick_results 
                if r.metadata
            )
        }
        
        logger.info("Database optimization validation completed")
        
    except Exception as e:
        validation_results['error'] = str(e)
        logger.error(f"Validation failed: {e}")
    
    finally:
        db.close()
    
    return validation_results


if __name__ == "__main__":
    # Run performance tests when script is executed directly
    results = run_performance_validation()
    print(json.dumps(results, indent=2))