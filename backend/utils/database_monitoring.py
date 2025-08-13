"""
Database monitoring and performance analysis utilities
"""

from sqlalchemy.orm import Session
from sqlalchemy import text, func
from typing import Dict, List, Any, Optional
import logging
import time
import json
from datetime import datetime, timedelta
from dataclasses import dataclass
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class SlowQuery:
    """Represents a slow query record"""
    query: str
    duration_ms: float
    calls: int
    avg_duration_ms: float
    total_duration_ms: float
    timestamp: datetime


@dataclass
class TableStats:
    """Represents table statistics"""
    table_name: str
    table_size: str
    index_size: str
    total_size: str
    row_count: int
    sequential_scans: int
    index_scans: int
    dead_tuples: int
    live_tuples: int


class DatabaseMonitor:
    """Comprehensive database monitoring and analysis"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def enable_slow_query_logging(self, threshold_ms: int = 1000):
        """Enable slow query logging with specified threshold"""
        
        try:
            # Enable pg_stat_statements extension for query tracking
            self.db.execute(text("CREATE EXTENSION IF NOT EXISTS pg_stat_statements;"))
            
            # Set slow query logging parameters
            self.db.execute(text(f"SET log_min_duration_statement = {threshold_ms};"))
            self.db.execute(text("SET log_statement = 'all';"))
            self.db.execute(text("SET log_duration = on;"))
            self.db.execute(text("SET log_line_prefix = '%t [%p]: [%l-1] user=%u,db=%d,app=%a,client=%h ';"))
            
            self.db.commit()
            logger.info(f"Slow query logging enabled with threshold {threshold_ms}ms")
            
        except Exception as e:
            logger.error(f"Failed to enable slow query logging: {e}")
            self.db.rollback()
    
    def get_slow_queries(self, limit: int = 20) -> List[SlowQuery]:
        """Get slowest queries from pg_stat_statements"""
        
        query = text("""
            SELECT 
                query,
                mean_exec_time as avg_duration_ms,
                calls,
                total_exec_time as total_duration_ms,
                max_exec_time as max_duration_ms,
                stddev_exec_time as stddev_duration_ms,
                rows,
                100.0 * shared_blks_hit / nullif(shared_blks_hit + shared_blks_read, 0) AS hit_percent
            FROM pg_stat_statements 
            WHERE query NOT LIKE '%pg_stat_statements%'
            ORDER BY mean_exec_time DESC
            LIMIT :limit
        """)
        
        try:
            result = self.db.execute(query, {'limit': limit}).fetchall()
            
            slow_queries = []
            for row in result:
                slow_query = SlowQuery(
                    query=row.query[:500] + "..." if len(row.query) > 500 else row.query,
                    duration_ms=float(row.avg_duration_ms),
                    calls=int(row.calls),
                    avg_duration_ms=float(row.avg_duration_ms),
                    total_duration_ms=float(row.total_duration_ms),
                    timestamp=datetime.utcnow()
                )
                slow_queries.append(slow_query)
            
            return slow_queries
            
        except Exception as e:
            logger.error(f"Failed to get slow queries: {e}")
            return []
    
    def get_table_statistics(self) -> List[TableStats]:
        """Get comprehensive table statistics"""
        
        query = text("""
            SELECT 
                schemaname,
                tablename,
                pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as total_size,
                pg_size_pretty(pg_relation_size(schemaname||'.'||tablename)) as table_size,
                pg_size_pretty(pg_indexes_size(schemaname||'.'||tablename)) as index_size,
                n_live_tup as row_count,
                seq_scan as sequential_scans,
                seq_tup_read,
                idx_scan as index_scans,
                idx_tup_fetch,
                n_dead_tup as dead_tuples,
                n_live_tup as live_tuples,
                last_vacuum,
                last_autovacuum,
                last_analyze,
                last_autoanalyze
            FROM pg_stat_user_tables pst
            WHERE schemaname = 'public'
            ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
        """)
        
        try:
            result = self.db.execute(query).fetchall()
            
            table_stats = []
            for row in result:
                stats = TableStats(
                    table_name=f"{row.schemaname}.{row.tablename}",
                    table_size=row.table_size,
                    index_size=row.index_size,
                    total_size=row.total_size,
                    row_count=int(row.row_count or 0),
                    sequential_scans=int(row.sequential_scans or 0),
                    index_scans=int(row.index_scans or 0),
                    dead_tuples=int(row.dead_tuples or 0),
                    live_tuples=int(row.live_tuples or 0)
                )
                table_stats.append(stats)
            
            return table_stats
            
        except Exception as e:
            logger.error(f"Failed to get table statistics: {e}")
            return []
    
    def get_index_usage_statistics(self) -> List[Dict[str, Any]]:
        """Get index usage statistics to identify unused indexes"""
        
        query = text("""
            SELECT 
                schemaname,
                tablename,
                indexname,
                idx_scan as index_scans,
                idx_tup_read as tuples_read,
                idx_tup_fetch as tuples_fetched,
                pg_size_pretty(pg_relation_size(indexrelid)) as index_size,
                CASE 
                    WHEN idx_scan = 0 THEN 'UNUSED'
                    WHEN idx_scan < 10 THEN 'LOW_USAGE'
                    WHEN idx_scan < 100 THEN 'MEDIUM_USAGE'
                    ELSE 'HIGH_USAGE'
                END as usage_category
            FROM pg_stat_user_indexes psi
            WHERE schemaname = 'public'
            ORDER BY idx_scan DESC, pg_relation_size(indexrelid) DESC
        """)
        
        try:
            result = self.db.execute(query).fetchall()
            
            return [
                {
                    'schema': row.schemaname,
                    'table': row.tablename,
                    'index': row.indexname,
                    'scans': int(row.index_scans or 0),
                    'tuples_read': int(row.tuples_read or 0),
                    'tuples_fetched': int(row.tuples_fetched or 0),
                    'size': row.index_size,
                    'usage_category': row.usage_category
                }
                for row in result
            ]
            
        except Exception as e:
            logger.error(f"Failed to get index usage statistics: {e}")
            return []
    
    def analyze_query_performance(self, query: str) -> Dict[str, Any]:
        """Analyze performance of a specific query using EXPLAIN ANALYZE"""
        
        explain_query = text(f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {query}")
        
        try:
            start_time = time.time()
            result = self.db.execute(explain_query).fetchone()
            execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            explain_data = result[0][0] if result else {}
            
            return {
                'query': query,
                'execution_time_ms': execution_time,
                'explain_plan': explain_data,
                'analyzed_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze query performance: {e}")
            return {
                'query': query,
                'error': str(e),
                'analyzed_at': datetime.utcnow().isoformat()
            }
    
    def get_database_size_info(self) -> Dict[str, Any]:
        """Get database size and space usage information"""
        
        query = text("""
            SELECT 
                pg_database.datname as database_name,
                pg_size_pretty(pg_database_size(pg_database.datname)) as database_size,
                pg_database_size(pg_database.datname) as database_size_bytes
            FROM pg_database
            WHERE datname = current_database()
        """)
        
        tablespace_query = text("""
            SELECT 
                spcname as tablespace_name,
                pg_size_pretty(pg_tablespace_size(spcname)) as tablespace_size
            FROM pg_tablespace
        """)
        
        try:
            db_result = self.db.execute(query).fetchone()
            tablespace_result = self.db.execute(tablespace_query).fetchall()
            
            return {
                'database': {
                    'name': db_result.database_name,
                    'size': db_result.database_size,
                    'size_bytes': int(db_result.database_size_bytes)
                },
                'tablespaces': [
                    {
                        'name': row.tablespace_name,
                        'size': row.tablespace_size
                    }
                    for row in tablespace_result
                ]
            }
            
        except Exception as e:
            logger.error(f"Failed to get database size info: {e}")
            return {'error': str(e)}
    
    def get_connection_statistics(self) -> Dict[str, Any]:
        """Get database connection statistics"""
        
        query = text("""
            SELECT 
                count(*) as total_connections,
                count(*) FILTER (WHERE state = 'active') as active_connections,
                count(*) FILTER (WHERE state = 'idle') as idle_connections,
                count(*) FILTER (WHERE state = 'idle in transaction') as idle_in_transaction,
                max(extract(epoch from (now() - backend_start))) as longest_connection_seconds,
                max(extract(epoch from (now() - query_start))) as longest_query_seconds,
                max(extract(epoch from (now() - state_change))) as longest_idle_seconds
            FROM pg_stat_activity 
            WHERE pid != pg_backend_pid()
        """)
        
        try:
            result = self.db.execute(query).fetchone()
            
            return {
                'total_connections': int(result.total_connections or 0),
                'active_connections': int(result.active_connections or 0),
                'idle_connections': int(result.idle_connections or 0),
                'idle_in_transaction': int(result.idle_in_transaction or 0),
                'longest_connection_seconds': float(result.longest_connection_seconds or 0),
                'longest_query_seconds': float(result.longest_query_seconds or 0),
                'longest_idle_seconds': float(result.longest_idle_seconds or 0)
            }
            
        except Exception as e:
            logger.error(f"Failed to get connection statistics: {e}")
            return {'error': str(e)}
    
    def get_lock_information(self) -> List[Dict[str, Any]]:
        """Get information about database locks"""
        
        query = text("""
            SELECT 
                pg_locks.mode,
                pg_locks.locktype,
                pg_locks.relation::regclass as relation,
                pg_locks.granted,
                pg_stat_activity.pid,
                pg_stat_activity.query,
                pg_stat_activity.state,
                extract(epoch from (now() - pg_stat_activity.query_start)) as query_duration_seconds
            FROM pg_locks
            JOIN pg_stat_activity ON pg_locks.pid = pg_stat_activity.pid
            WHERE pg_locks.granted = false
            ORDER BY query_duration_seconds DESC
        """)
        
        try:
            result = self.db.execute(query).fetchall()
            
            return [
                {
                    'mode': row.mode,
                    'lock_type': row.locktype,
                    'relation': str(row.relation) if row.relation else None,
                    'granted': row.granted,
                    'pid': row.pid,
                    'query': row.query[:200] + "..." if row.query and len(row.query) > 200 else row.query,
                    'state': row.state,
                    'duration_seconds': float(row.query_duration_seconds or 0)
                }
                for row in result
            ]
            
        except Exception as e:
            logger.error(f"Failed to get lock information: {e}")
            return []
    
    def check_table_health(self, table_name: str) -> Dict[str, Any]:
        """Check health of a specific table"""
        
        health_query = text("""
            SELECT 
                schemaname,
                tablename,
                n_live_tup as live_tuples,
                n_dead_tup as dead_tuples,
                CASE 
                    WHEN n_live_tup > 0 THEN (n_dead_tup::float / n_live_tup::float) * 100
                    ELSE 0
                END as dead_tuple_percentage,
                last_vacuum,
                last_autovacuum,
                last_analyze,
                last_autoanalyze,
                vacuum_count,
                autovacuum_count,
                analyze_count,
                autoanalyze_count,
                seq_scan,
                seq_tup_read,
                idx_scan,
                idx_tup_fetch
            FROM pg_stat_user_tables 
            WHERE tablename = :table_name
        """)
        
        try:
            result = self.db.execute(health_query, {'table_name': table_name}).fetchone()
            
            if not result:
                return {'error': f'Table {table_name} not found'}
            
            health_score = 100
            warnings = []
            
            # Check dead tuple percentage
            dead_pct = float(result.dead_tuple_percentage or 0)
            if dead_pct > 20:
                health_score -= 30
                warnings.append(f'High dead tuple percentage: {dead_pct:.1f}%')
            elif dead_pct > 10:
                health_score -= 15
                warnings.append(f'Moderate dead tuple percentage: {dead_pct:.1f}%')
            
            # Check vacuum frequency
            if not result.last_vacuum and not result.last_autovacuum:
                health_score -= 25
                warnings.append('Table has never been vacuumed')
            elif result.last_autovacuum:
                last_vacuum = result.last_autovacuum
                if last_vacuum < datetime.utcnow() - timedelta(days=7):
                    health_score -= 15
                    warnings.append('Table not vacuumed in over a week')
            
            # Check analyze frequency
            if not result.last_analyze and not result.last_autoanalyze:
                health_score -= 20
                warnings.append('Table statistics never analyzed')
            
            # Check sequential scan ratio
            total_scans = (result.seq_scan or 0) + (result.idx_scan or 0)
            if total_scans > 0:
                seq_scan_ratio = (result.seq_scan or 0) / total_scans * 100
                if seq_scan_ratio > 50:
                    health_score -= 20
                    warnings.append(f'High sequential scan ratio: {seq_scan_ratio:.1f}%')
            
            return {
                'table': table_name,
                'health_score': max(0, health_score),
                'live_tuples': int(result.live_tuples or 0),
                'dead_tuples': int(result.dead_tuples or 0),
                'dead_tuple_percentage': dead_pct,
                'last_vacuum': result.last_vacuum,
                'last_autovacuum': result.last_autovacuum,
                'last_analyze': result.last_analyze,
                'last_autoanalyze': result.last_autoanalyze,
                'sequential_scans': int(result.seq_scan or 0),
                'index_scans': int(result.idx_scan or 0),
                'warnings': warnings
            }
            
        except Exception as e:
            logger.error(f"Failed to check table health: {e}")
            return {'error': str(e)}
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive database performance report"""
        
        try:
            report = {
                'generated_at': datetime.utcnow().isoformat(),
                'database_size': self.get_database_size_info(),
                'connection_stats': self.get_connection_statistics(),
                'slow_queries': [
                    {
                        'query': sq.query,
                        'avg_duration_ms': sq.avg_duration_ms,
                        'calls': sq.calls,
                        'total_duration_ms': sq.total_duration_ms
                    }
                    for sq in self.get_slow_queries(10)
                ],
                'table_stats': [
                    {
                        'table': ts.table_name,
                        'total_size': ts.total_size,
                        'row_count': ts.row_count,
                        'sequential_scans': ts.sequential_scans,
                        'index_scans': ts.index_scans,
                        'dead_tuples': ts.dead_tuples
                    }
                    for ts in self.get_table_statistics()[:10]
                ],
                'index_usage': self.get_index_usage_statistics()[:20],
                'locks': self.get_lock_information(),
                'table_health': {}
            }
            
            # Check health of main tables
            main_tables = ['price_history', 'technical_indicators', 'recommendations', 'stocks']
            for table in main_tables:
                report['table_health'][table] = self.check_table_health(table)
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate performance report: {e}")
            return {'error': str(e)}
    
    @contextmanager
    def query_timer(self, query_name: str):
        """Context manager to time query execution"""
        
        start_time = time.time()
        try:
            yield
        finally:
            duration = (time.time() - start_time) * 1000
            logger.info(f"Query '{query_name}' executed in {duration:.2f}ms")
            
            # Log slow queries
            if duration > 1000:
                logger.warning(f"Slow query detected: '{query_name}' took {duration:.2f}ms")


# Utility functions for monitoring
def setup_database_monitoring(db: Session, slow_query_threshold_ms: int = 1000):
    """Setup database monitoring with specified parameters"""
    
    monitor = DatabaseMonitor(db)
    monitor.enable_slow_query_logging(slow_query_threshold_ms)
    
    return monitor


def log_query_performance(db: Session, query: str, query_name: str = "unknown"):
    """Decorator to log query performance"""
    
    monitor = DatabaseMonitor(db)
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            with monitor.query_timer(query_name):
                return func(*args, **kwargs)
        return wrapper
    
    return decorator