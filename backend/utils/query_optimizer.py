"""
Advanced Query Optimizer for Stock Market Database

This module provides intelligent query optimization, index management,
and performance monitoring for massive stock data workloads.
"""

import asyncio
import asyncpg
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import re
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)


@dataclass
class QueryStats:
    """Query performance statistics"""
    query_id: str
    query_text: str
    execution_count: int
    total_time_ms: float
    avg_time_ms: float
    max_time_ms: float
    min_time_ms: float
    rows_returned: int
    cache_hits: int
    cache_misses: int
    index_usage: Dict[str, int]
    optimization_suggestions: List[str]


@dataclass
class IndexRecommendation:
    """Index creation recommendation"""
    table_name: str
    columns: List[str]
    index_type: str  # btree, brin, gin, gist, hash
    condition: Optional[str]
    estimated_benefit: float
    estimated_size_mb: float
    priority: int  # 1-10


class StockQueryOptimizer:
    """
    Advanced query optimizer for stock market data with intelligent
    index management and performance monitoring.
    """
    
    def __init__(self, connection_pool: asyncpg.Pool):
        self.pool = connection_pool
        self.query_cache: Dict[str, QueryStats] = {}
        self.slow_query_threshold_ms = 1000
        self.index_recommendations: List[IndexRecommendation] = []
        
        # Common query patterns for stock data
        self.stock_query_patterns = {
            'price_lookup': r'SELECT.*FROM.*price_history.*WHERE.*stock_id.*AND.*date',
            'range_query': r'SELECT.*FROM.*price_history.*WHERE.*date.*BETWEEN',
            'technical_analysis': r'SELECT.*FROM.*technical_indicators.*WHERE.*stock_id',
            'volume_analysis': r'SELECT.*FROM.*price_history.*WHERE.*volume',
            'sentiment_lookup': r'SELECT.*FROM.*news_sentiment.*WHERE.*stock_id',
            'cross_table_join': r'SELECT.*FROM.*stocks.*JOIN.*price_history',
            'aggregation_query': r'SELECT.*AVG\(|SUM\(|COUNT\(|MAX\(|MIN\(',
            'time_series_window': r'SELECT.*OVER.*PARTITION.*ORDER.*BY.*date'
        }

    async def analyze_query_performance(self, hours_lookback: int = 24) -> Dict[str, Any]:
        """Analyze query performance from pg_stat_statements"""
        
        async with self.pool.acquire() as conn:
            # Enable pg_stat_statements if not already enabled
            try:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS pg_stat_statements;")
            except:
                logger.warning("pg_stat_statements extension not available")
                return {}
            
            # Get query statistics
            query_stats = await conn.fetch("""
                SELECT 
                    queryid,
                    query,
                    calls,
                    total_exec_time as total_time,
                    mean_exec_time as avg_time,
                    max_exec_time as max_time,
                    min_exec_time as min_time,
                    rows,
                    shared_blks_hit,
                    shared_blks_read,
                    shared_blks_dirtied,
                    shared_blks_written,
                    local_blks_hit,
                    local_blks_read
                FROM pg_stat_statements 
                WHERE query NOT LIKE '%pg_stat_statements%'
                AND calls > 1
                ORDER BY total_exec_time DESC
                LIMIT 100
            """)
            
            analyzed_queries = []
            for row in query_stats:
                query_analysis = await self._analyze_individual_query(conn, row)
                analyzed_queries.append(query_analysis)
            
            return {
                'total_queries_analyzed': len(analyzed_queries),
                'slow_queries': [q for q in analyzed_queries if q.avg_time_ms > self.slow_query_threshold_ms],
                'most_frequent': sorted(analyzed_queries, key=lambda x: x.execution_count, reverse=True)[:10],
                'slowest_queries': sorted(analyzed_queries, key=lambda x: x.avg_time_ms, reverse=True)[:10],
                'index_recommendations': await self._generate_index_recommendations(analyzed_queries),
                'cache_performance': self._analyze_cache_performance(analyzed_queries)
            }

    async def _analyze_individual_query(self, conn: asyncpg.Connection, row: dict) -> QueryStats:
        """Analyze performance of an individual query"""
        
        query_text = row['query'].strip()
        query_id = str(row['queryid'])
        
        # Classify query type
        query_type = self._classify_query(query_text)
        
        # Analyze index usage for this query
        index_usage = await self._analyze_index_usage(conn, query_text)
        
        # Generate optimization suggestions
        suggestions = self._generate_optimization_suggestions(query_text, row, index_usage)
        
        # Calculate cache performance
        cache_hits = row['shared_blks_hit'] + row['local_blks_hit']
        cache_misses = row['shared_blks_read'] + row['local_blks_read']
        
        return QueryStats(
            query_id=query_id,
            query_text=query_text,
            execution_count=row['calls'],
            total_time_ms=row['total_time'],
            avg_time_ms=row['avg_time'],
            max_time_ms=row['max_time'],
            min_time_ms=row['min_time'],
            rows_returned=row['rows'],
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            index_usage=index_usage,
            optimization_suggestions=suggestions
        )

    def _classify_query(self, query_text: str) -> str:
        """Classify query based on common patterns"""
        
        query_lower = query_text.lower()
        
        for pattern_name, pattern in self.stock_query_patterns.items():
            if re.search(pattern, query_lower, re.IGNORECASE):
                return pattern_name
        
        return 'unknown'

    async def _analyze_index_usage(self, conn: asyncpg.Connection, query_text: str) -> Dict[str, int]:
        """Analyze index usage for a specific query using EXPLAIN"""
        
        try:
            # Get query plan
            plan_result = await conn.fetchval(f"EXPLAIN (FORMAT JSON) {query_text}")
            plan = json.loads(plan_result)[0]['Plan']
            
            index_usage = {}
            self._extract_index_usage_from_plan(plan, index_usage)
            
            return index_usage
            
        except Exception as e:
            logger.debug(f"Could not analyze query plan: {e}")
            return {}

    def _extract_index_usage_from_plan(self, plan: dict, index_usage: Dict[str, int]):
        """Recursively extract index usage from query plan"""
        
        if 'Index Name' in plan:
            index_name = plan['Index Name']
            index_usage[index_name] = index_usage.get(index_name, 0) + 1
        
        if 'Plans' in plan:
            for subplan in plan['Plans']:
                self._extract_index_usage_from_plan(subplan, index_usage)

    def _generate_optimization_suggestions(
        self, 
        query_text: str, 
        stats: dict, 
        index_usage: Dict[str, int]
    ) -> List[str]:
        """Generate optimization suggestions for a query"""
        
        suggestions = []
        query_lower = query_text.lower()
        
        # Check for missing indexes
        if 'where' in query_lower and not index_usage:
            suggestions.append("Consider adding indexes on WHERE clause columns")
        
        # Check for sequential scans on large tables
        if stats['rows'] > 100000 and stats['avg_time'] > 1000:
            if 'price_history' in query_lower:
                suggestions.append("Large table scan detected - consider partitioning or better indexes")
        
        # Check for inefficient JOINs
        if 'join' in query_lower and stats['avg_time'] > 500:
            suggestions.append("JOIN operation may benefit from better indexing on join columns")
        
        # Check for ORDER BY without index
        if 'order by' in query_lower and stats['avg_time'] > 200:
            suggestions.append("ORDER BY clause may benefit from covering index")
        
        # Check for aggregation queries
        if any(agg in query_lower for agg in ['sum(', 'avg(', 'count(', 'max(', 'min(']):
            if stats['avg_time'] > 1000:
                suggestions.append("Consider materialized views for frequently used aggregations")
        
        # Check for date range queries
        if 'date' in query_lower and 'between' in query_lower:
            suggestions.append("Consider BRIN indexes for date range queries on large tables")
        
        # Check cache efficiency
        cache_hit_ratio = stats.get('shared_blks_hit', 0) / max(
            stats.get('shared_blks_hit', 0) + stats.get('shared_blks_read', 0), 1
        )
        if cache_hit_ratio < 0.8:
            suggestions.append(f"Low cache hit ratio ({cache_hit_ratio:.2%}) - consider increasing shared_buffers")
        
        return suggestions

    async def _generate_index_recommendations(self, queries: List[QueryStats]) -> List[IndexRecommendation]:
        """Generate index recommendations based on query analysis"""
        
        recommendations = []
        
        # Analyze WHERE clause columns
        where_columns = defaultdict(int)
        join_columns = defaultdict(int)
        order_columns = defaultdict(int)
        
        for query in queries:
            query_text = query.query_text.lower()
            
            # Extract WHERE clause patterns
            self._extract_where_patterns(query_text, where_columns, query.execution_count)
            
            # Extract JOIN patterns
            self._extract_join_patterns(query_text, join_columns, query.execution_count)
            
            # Extract ORDER BY patterns
            self._extract_order_patterns(query_text, order_columns, query.execution_count)
        
        # Generate recommendations for frequently used columns
        for column_pattern, frequency in where_columns.items():
            if frequency > 10:  # Frequently used
                table, column = column_pattern.split('.')
                rec = IndexRecommendation(
                    table_name=table,
                    columns=[column],
                    index_type='btree',
                    condition=None,
                    estimated_benefit=frequency * 0.1,
                    estimated_size_mb=self._estimate_index_size(table, [column]),
                    priority=min(10, frequency // 5)
                )
                recommendations.append(rec)
        
        # Special recommendations for stock data
        recommendations.extend(self._generate_stock_specific_recommendations())
        
        return sorted(recommendations, key=lambda x: x.priority, reverse=True)

    def _extract_where_patterns(self, query_text: str, where_columns: dict, frequency: int):
        """Extract WHERE clause column patterns"""
        
        # Simple pattern matching for WHERE clauses
        patterns = [
            r'where\s+(\w+)\.(\w+)\s*[=<>]',
            r'where\s+(\w+)\s*[=<>]',
            r'and\s+(\w+)\.(\w+)\s*[=<>]',
            r'and\s+(\w+)\s*[=<>]'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, query_text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    if len(match) == 2:
                        column_key = f"{match[0]}.{match[1]}"
                    else:
                        column_key = f"unknown.{match[0]}"
                else:
                    column_key = f"unknown.{match}"
                where_columns[column_key] += frequency

    def _extract_join_patterns(self, query_text: str, join_columns: dict, frequency: int):
        """Extract JOIN column patterns"""
        
        join_pattern = r'join\s+(\w+).*?on\s+(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)'
        matches = re.findall(join_pattern, query_text, re.IGNORECASE)
        
        for match in matches:
            left_col = f"{match[1]}.{match[2]}"
            right_col = f"{match[3]}.{match[4]}"
            join_columns[left_col] += frequency
            join_columns[right_col] += frequency

    def _extract_order_patterns(self, query_text: str, order_columns: dict, frequency: int):
        """Extract ORDER BY column patterns"""
        
        order_pattern = r'order\s+by\s+(\w+)\.?(\w+)?'
        matches = re.findall(order_pattern, query_text, re.IGNORECASE)
        
        for match in matches:
            if match[1]:  # table.column
                column_key = f"{match[0]}.{match[1]}"
            else:  # just column
                column_key = f"unknown.{match[0]}"
            order_columns[column_key] += frequency

    def _generate_stock_specific_recommendations(self) -> List[IndexRecommendation]:
        """Generate recommendations specific to stock market data"""
        
        return [
            # BRIN indexes for time-series data
            IndexRecommendation(
                table_name="price_history_optimized",
                columns=["date", "stock_id"],
                index_type="brin",
                condition=None,
                estimated_benefit=8.0,
                estimated_size_mb=50.0,
                priority=9
            ),
            
            # Composite index for common price queries
            IndexRecommendation(
                table_name="price_history_optimized",
                columns=["stock_id", "date", "close"],
                index_type="btree",
                condition=None,
                estimated_benefit=7.5,
                estimated_size_mb=200.0,
                priority=8
            ),
            
            # Volume-based partial index
            IndexRecommendation(
                table_name="price_history_optimized",
                columns=["volume", "stock_id"],
                index_type="btree",
                condition="volume > 1000000",
                estimated_benefit=6.0,
                estimated_size_mb=100.0,
                priority=7
            ),
            
            # Technical indicators composite index
            IndexRecommendation(
                table_name="technical_indicators_optimized",
                columns=["stock_id", "date", "rsi_14"],
                index_type="btree",
                condition="rsi_14 IS NOT NULL",
                estimated_benefit=6.5,
                estimated_size_mb=80.0,
                priority=7
            ),
            
            # News sentiment GIN index for full-text search
            IndexRecommendation(
                table_name="news_sentiment_bulk",
                columns=["headline_vector"],
                index_type="gin",
                condition=None,
                estimated_benefit=5.0,
                estimated_size_mb=150.0,
                priority=6
            )
        ]

    def _estimate_index_size(self, table_name: str, columns: List[str]) -> float:
        """Estimate index size in MB (simplified calculation)"""
        
        # Rough estimates based on typical stock data
        size_estimates = {
            'stock_id': 4,  # SMALLINT
            'date': 4,      # DATE  
            'timestamp': 8,  # TIMESTAMP
            'close': 4,      # REAL
            'volume': 4,     # INTEGER
            'rsi_14': 4,     # REAL
        }
        
        row_estimate = 6000 * 365 * 5  # 6000 stocks, 5 years of data
        
        total_column_size = sum(size_estimates.get(col, 8) for col in columns)
        index_overhead = total_column_size * 1.5  # B-tree overhead
        
        return (row_estimate * index_overhead) / (1024 * 1024)  # Convert to MB

    def _analyze_cache_performance(self, queries: List[QueryStats]) -> Dict[str, Any]:
        """Analyze overall cache performance"""
        
        total_hits = sum(q.cache_hits for q in queries)
        total_misses = sum(q.cache_misses for q in queries)
        total_requests = total_hits + total_misses
        
        if total_requests == 0:
            return {'cache_hit_ratio': 0.0, 'recommendation': 'No cache data available'}
        
        hit_ratio = total_hits / total_requests
        
        recommendation = ""
        if hit_ratio < 0.8:
            recommendation = "Consider increasing shared_buffers and effective_cache_size"
        elif hit_ratio > 0.95:
            recommendation = "Excellent cache performance"
        else:
            recommendation = "Good cache performance"
        
        return {
            'cache_hit_ratio': hit_ratio,
            'total_cache_requests': total_requests,
            'recommendation': recommendation
        }

    async def create_recommended_indexes(
        self, 
        recommendations: Optional[List[IndexRecommendation]] = None,
        max_indexes: int = 10,
        simulate_only: bool = True
    ) -> Dict[str, Any]:
        """Create indexes based on recommendations"""
        
        if recommendations is None:
            analysis = await self.analyze_query_performance()
            recommendations = analysis['index_recommendations']
        
        # Sort by priority and take top recommendations
        top_recommendations = sorted(
            recommendations, 
            key=lambda x: x.priority, 
            reverse=True
        )[:max_indexes]
        
        results = {
            'created_indexes': [],
            'failed_indexes': [],
            'total_estimated_size_mb': 0.0,
            'simulation_mode': simulate_only
        }
        
        async with self.pool.acquire() as conn:
            for rec in top_recommendations:
                try:
                    index_name = self._generate_index_name(rec)
                    create_sql = self._generate_create_index_sql(rec, index_name)
                    
                    if simulate_only:
                        logger.info(f"SIMULATION: Would create index: {create_sql}")
                        results['created_indexes'].append({
                            'index_name': index_name,
                            'sql': create_sql,
                            'estimated_size_mb': rec.estimated_size_mb
                        })
                    else:
                        # Create index with CONCURRENTLY to avoid blocking
                        await conn.execute(create_sql)
                        logger.info(f"Created index: {index_name}")
                        results['created_indexes'].append({
                            'index_name': index_name,
                            'sql': create_sql,
                            'estimated_size_mb': rec.estimated_size_mb
                        })
                    
                    results['total_estimated_size_mb'] += rec.estimated_size_mb
                    
                except Exception as e:
                    logger.error(f"Failed to create index {index_name}: {e}")
                    results['failed_indexes'].append({
                        'recommendation': rec,
                        'error': str(e)
                    })
        
        return results

    def _generate_index_name(self, rec: IndexRecommendation) -> str:
        """Generate a descriptive index name"""
        
        columns_str = "_".join(rec.columns)
        condition_str = "_partial" if rec.condition else ""
        
        return f"idx_{rec.table_name}_{columns_str}_{rec.index_type}{condition_str}"

    def _generate_create_index_sql(self, rec: IndexRecommendation, index_name: str) -> str:
        """Generate CREATE INDEX SQL statement"""
        
        columns_str = ", ".join(rec.columns)
        
        sql = f"CREATE INDEX CONCURRENTLY {index_name} ON {rec.table_name}"
        
        if rec.index_type != 'btree':
            sql += f" USING {rec.index_type}"
        
        sql += f" ({columns_str})"
        
        if rec.condition:
            sql += f" WHERE {rec.condition}"
        
        return sql

    async def monitor_index_effectiveness(
        self, 
        hours_lookback: int = 24
    ) -> Dict[str, Any]:
        """Monitor the effectiveness of existing indexes"""
        
        async with self.pool.acquire() as conn:
            # Get index usage statistics
            index_stats = await conn.fetch("""
                SELECT 
                    schemaname,
                    tablename,
                    indexname,
                    idx_scan,
                    idx_tup_read,
                    idx_tup_fetch,
                    pg_size_pretty(pg_relation_size(indexrelid)) as index_size
                FROM pg_stat_user_indexes 
                WHERE schemaname = 'public'
                AND tablename IN (
                    'price_history_optimized',
                    'technical_indicators_optimized',
                    'news_sentiment_bulk',
                    'stocks'
                )
                ORDER BY idx_scan DESC
            """)
            
            # Identify unused indexes
            unused_indexes = [
                idx for idx in index_stats 
                if idx['idx_scan'] == 0 and not idx['indexname'].endswith('_pkey')
            ]
            
            # Calculate index efficiency
            efficient_indexes = []
            for idx in index_stats:
                if idx['idx_scan'] > 0:
                    efficiency = idx['idx_tup_fetch'] / idx['idx_scan'] if idx['idx_scan'] > 0 else 0
                    efficient_indexes.append({
                        'index_name': idx['indexname'],
                        'table_name': idx['tablename'],
                        'scans': idx['idx_scan'],
                        'efficiency_ratio': efficiency,
                        'size': idx['index_size']
                    })
            
            return {
                'total_indexes': len(index_stats),
                'unused_indexes': unused_indexes,
                'most_used_indexes': sorted(index_stats, key=lambda x: x['idx_scan'], reverse=True)[:10],
                'index_efficiency': sorted(efficient_indexes, key=lambda x: x['efficiency_ratio'], reverse=True),
                'cleanup_recommendations': [
                    f"DROP INDEX {idx['indexname']}" for idx in unused_indexes[:5]
                ]
            }

    async def optimize_table_statistics(self) -> Dict[str, Any]:
        """Update table statistics for better query planning"""
        
        tables_to_analyze = [
            'price_history_optimized',
            'technical_indicators_optimized', 
            'news_sentiment_bulk',
            'stocks',
            'fundamentals'
        ]
        
        results = {
            'analyzed_tables': [],
            'analysis_times': {}
        }
        
        async with self.pool.acquire() as conn:
            for table in tables_to_analyze:
                try:
                    start_time = datetime.now()
                    await conn.execute(f"ANALYZE {table}")
                    duration = (datetime.now() - start_time).total_seconds()
                    
                    results['analyzed_tables'].append(table)
                    results['analysis_times'][table] = duration
                    
                    logger.info(f"Analyzed {table} in {duration:.2f}s")
                    
                except Exception as e:
                    logger.error(f"Failed to analyze {table}: {e}")
        
        return results


# Example usage
async def example_optimization_workflow():
    """Example of complete optimization workflow"""
    
    # Create connection pool
    pool = await asyncpg.create_pool(
        host="localhost",
        port=5432,
        user="postgres", 
        password="password",
        database="stock_db",
        min_size=2,
        max_size=10
    )
    
    optimizer = StockQueryOptimizer(pool)
    
    try:
        # 1. Analyze current query performance
        print("Analyzing query performance...")
        performance_analysis = await optimizer.analyze_query_performance(hours_lookback=24)
        
        # 2. Create recommended indexes (simulation mode)
        print("Generating index recommendations...")
        index_results = await optimizer.create_recommended_indexes(
            max_indexes=5,
            simulate_only=True  # Change to False to actually create indexes
        )
        
        # 3. Monitor existing index effectiveness
        print("Monitoring index effectiveness...")
        index_monitoring = await optimizer.monitor_index_effectiveness()
        
        # 4. Update table statistics
        print("Updating table statistics...")
        stats_results = await optimizer.optimize_table_statistics()
        
        # Print summary
        print(f"\n=== OPTIMIZATION SUMMARY ===")
        print(f"Slow queries found: {len(performance_analysis['slow_queries'])}")
        print(f"Index recommendations: {len(performance_analysis['index_recommendations'])}")
        print(f"Unused indexes: {len(index_monitoring['unused_indexes'])}")
        print(f"Tables analyzed: {len(stats_results['analyzed_tables'])}")
        
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(example_optimization_workflow())