"""
Advanced Database Maintenance System for Stock Market Data

This module provides comprehensive maintenance strategies for high-volume
stock data including vacuum optimization, index maintenance, and performance
monitoring specifically designed for massive daily data loads.
"""

import asyncio
import asyncpg
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import psutil
from enum import Enum
import statistics

logger = logging.getLogger(__name__)


class MaintenanceType(Enum):
    """Types of maintenance operations"""
    VACUUM = "vacuum"
    ANALYZE = "analyze" 
    REINDEX = "reindex"
    COMPRESSION = "compression"
    CLEANUP = "cleanup"
    STATISTICS = "statistics"


@dataclass
class MaintenanceTask:
    """Individual maintenance task"""
    task_id: str
    task_type: MaintenanceType
    target_table: str
    priority: int  # 1-10
    estimated_duration_minutes: int
    last_run: Optional[datetime]
    next_run: datetime
    parameters: Dict[str, Any]
    dependencies: List[str]  # Task IDs that must complete first


@dataclass
class MaintenanceResult:
    """Result of a maintenance operation"""
    task_id: str
    task_type: MaintenanceType
    target_table: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    success: bool
    rows_affected: int
    size_before_mb: float
    size_after_mb: float
    space_reclaimed_mb: float
    error_message: Optional[str]
    metrics: Dict[str, Any]


class StockDatabaseMaintainer:
    """
    Comprehensive database maintenance system optimized for
    massive stock data workloads with intelligent scheduling
    and performance-aware operations.
    """
    
    def __init__(self, connection_pool: asyncpg.Pool):
        self.pool = connection_pool
        self.maintenance_history: List[MaintenanceResult] = []
        self.active_tasks: Dict[str, MaintenanceTask] = {}
        
        # Configuration for stock-specific maintenance
        self.table_configs = {
            'price_history_optimized': {
                'vacuum_threshold': 50000,  # More frequent vacuum
                'analyze_threshold': 100000,
                'vacuum_scale_factor': 0.01,
                'analyze_scale_factor': 0.005,
                'fillfactor': 95,
                'maintenance_work_mem': '1GB'
            },
            'technical_indicators_optimized': {
                'vacuum_threshold': 25000,
                'analyze_threshold': 50000,
                'vacuum_scale_factor': 0.02,
                'analyze_scale_factor': 0.01,
                'fillfactor': 90,
                'maintenance_work_mem': '512MB'
            },
            'news_sentiment_bulk': {
                'vacuum_threshold': 10000,
                'analyze_threshold': 20000,
                'vacuum_scale_factor': 0.05,
                'analyze_scale_factor': 0.02,
                'fillfactor': 85,
                'maintenance_work_mem': '256MB'
            }
        }

    async def create_maintenance_schedule(self) -> List[MaintenanceTask]:
        """Create intelligent maintenance schedule based on table activity"""
        
        tasks = []
        
        # Get table statistics
        table_stats = await self._get_table_statistics()
        
        for table_name, stats in table_stats.items():
            if table_name in self.table_configs:
                config = self.table_configs[table_name]
                
                # Create vacuum task
                vacuum_task = MaintenanceTask(
                    task_id=f"vacuum_{table_name}_{datetime.now().strftime('%Y%m%d_%H%M')}",
                    task_type=MaintenanceType.VACUUM,
                    target_table=table_name,
                    priority=self._calculate_vacuum_priority(stats, config),
                    estimated_duration_minutes=self._estimate_vacuum_duration(stats),
                    last_run=stats.get('last_vacuum'),
                    next_run=self._calculate_next_vacuum_time(stats, config),
                    parameters={
                        'vacuum_type': 'full' if stats['dead_tup_ratio'] > 0.2 else 'standard',
                        'analyze': True,
                        'maintenance_work_mem': config['maintenance_work_mem']
                    },
                    dependencies=[]
                )
                tasks.append(vacuum_task)
                
                # Create analyze task
                analyze_task = MaintenanceTask(
                    task_id=f"analyze_{table_name}_{datetime.now().strftime('%Y%m%d_%H%M')}",
                    task_type=MaintenanceType.ANALYZE,
                    target_table=table_name,
                    priority=self._calculate_analyze_priority(stats, config),
                    estimated_duration_minutes=self._estimate_analyze_duration(stats),
                    last_run=stats.get('last_analyze'),
                    next_run=self._calculate_next_analyze_time(stats, config),
                    parameters={
                        'sample_size': 'default',
                        'statistics_target': 1000 if stats['n_tup_ins'] > 1000000 else 100
                    },
                    dependencies=[]
                )
                tasks.append(analyze_task)
                
                # Create compression task for TimescaleDB tables
                if self._is_timescaledb_table(table_name):
                    compression_task = MaintenanceTask(
                        task_id=f"compress_{table_name}_{datetime.now().strftime('%Y%m%d_%H%M')}",
                        task_type=MaintenanceType.COMPRESSION,
                        target_table=table_name,
                        priority=8,
                        estimated_duration_minutes=self._estimate_compression_duration(stats),
                        last_run=None,
                        next_run=datetime.now() + timedelta(hours=2),
                        parameters={
                            'compress_older_than': '4 hours',
                            'recompress_enabled': True
                        },
                        dependencies=[vacuum_task.task_id]
                    )
                    tasks.append(compression_task)
        
        # Add index maintenance tasks
        index_tasks = await self._create_index_maintenance_tasks()
        tasks.extend(index_tasks)
        
        # Add cleanup tasks
        cleanup_tasks = self._create_cleanup_tasks()
        tasks.extend(cleanup_tasks)
        
        return sorted(tasks, key=lambda x: (x.priority, x.next_run), reverse=True)

    async def _get_table_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get comprehensive table statistics for maintenance planning"""
        
        async with self.pool.acquire() as conn:
            stats_query = """
                SELECT 
                    schemaname,
                    tablename,
                    n_tup_ins,
                    n_tup_upd,
                    n_tup_del,
                    n_live_tup,
                    n_dead_tup,
                    last_vacuum,
                    last_autovacuum,
                    last_analyze,
                    last_autoanalyze,
                    vacuum_count,
                    autovacuum_count,
                    analyze_count,
                    autoanalyze_count,
                    pg_total_relation_size(schemaname||'.'||tablename) as total_size_bytes,
                    pg_relation_size(schemaname||'.'||tablename) as table_size_bytes
                FROM pg_stat_user_tables 
                WHERE schemaname = 'public'
                AND tablename IN (
                    'price_history_optimized',
                    'technical_indicators_optimized', 
                    'news_sentiment_bulk',
                    'stocks',
                    'fundamentals'
                )
            """
            
            rows = await conn.fetch(stats_query)
            
            table_stats = {}
            for row in rows:
                stats = dict(row)
                
                # Calculate derived metrics
                total_tuples = stats['n_live_tup'] + stats['n_dead_tup']
                stats['dead_tup_ratio'] = stats['n_dead_tup'] / max(total_tuples, 1)
                stats['table_size_mb'] = stats['table_size_bytes'] / (1024 * 1024)
                stats['total_size_mb'] = stats['total_size_bytes'] / (1024 * 1024)
                
                # Calculate activity score (for prioritization)
                recent_activity = (
                    stats['n_tup_ins'] + stats['n_tup_upd'] + stats['n_tup_del']
                ) / max(stats['n_live_tup'], 1)
                stats['activity_score'] = min(recent_activity, 10.0)
                
                table_stats[stats['tablename']] = stats
            
            return table_stats

    def _calculate_vacuum_priority(self, stats: Dict[str, Any], config: Dict[str, Any]) -> int:
        """Calculate vacuum priority based on table statistics"""
        
        priority = 5  # Base priority
        
        # High dead tuple ratio increases priority
        if stats['dead_tup_ratio'] > 0.2:
            priority += 3
        elif stats['dead_tup_ratio'] > 0.1:
            priority += 2
        elif stats['dead_tup_ratio'] > 0.05:
            priority += 1
        
        # Large tables with any dead tuples get higher priority
        if stats['table_size_mb'] > 1000 and stats['n_dead_tup'] > 1000:
            priority += 2
        
        # Tables with high activity get higher priority
        if stats['activity_score'] > 5:
            priority += 1
        
        # Time since last vacuum
        if stats['last_vacuum']:
            hours_since_vacuum = (
                datetime.now() - stats['last_vacuum']
            ).total_seconds() / 3600
            
            if hours_since_vacuum > 48:
                priority += 2
            elif hours_since_vacuum > 24:
                priority += 1
        else:
            priority += 3  # Never vacuumed
        
        return min(priority, 10)

    def _calculate_analyze_priority(self, stats: Dict[str, Any], config: Dict[str, Any]) -> int:
        """Calculate analyze priority based on data changes"""
        
        priority = 3  # Lower base priority than vacuum
        
        # Calculate change ratio since last analyze
        total_changes = stats['n_tup_ins'] + stats['n_tup_upd'] + stats['n_tup_del']
        change_ratio = total_changes / max(stats['n_live_tup'], 1)
        
        if change_ratio > 0.3:
            priority += 3
        elif change_ratio > 0.1:
            priority += 2
        elif change_ratio > 0.05:
            priority += 1
        
        # Time since last analyze
        if stats['last_analyze']:
            hours_since_analyze = (
                datetime.now() - stats['last_analyze']
            ).total_seconds() / 3600
            
            if hours_since_analyze > 72:
                priority += 2
            elif hours_since_analyze > 48:
                priority += 1
        else:
            priority += 2
        
        return min(priority, 10)

    def _calculate_next_vacuum_time(self, stats: Dict[str, Any], config: Dict[str, Any]) -> datetime:
        """Calculate when next vacuum should run"""
        
        base_interval_hours = 12  # Base interval for high-volume tables
        
        # Adjust based on activity
        if stats['activity_score'] > 8:
            interval_hours = 6
        elif stats['activity_score'] > 5:
            interval_hours = 8
        elif stats['activity_score'] > 2:
            interval_hours = 12
        else:
            interval_hours = 24
        
        # Adjust based on dead tuple ratio
        if stats['dead_tup_ratio'] > 0.1:
            interval_hours = max(interval_hours // 2, 2)
        
        return datetime.now() + timedelta(hours=interval_hours)

    def _calculate_next_analyze_time(self, stats: Dict[str, Any], config: Dict[str, Any]) -> datetime:
        """Calculate when next analyze should run"""
        
        # Analyze less frequently than vacuum
        base_interval_hours = 24
        
        # Adjust based on data changes
        total_changes = stats['n_tup_ins'] + stats['n_tup_upd'] + stats['n_tup_del']
        if total_changes > 1000000:  # High change volume
            interval_hours = 12
        elif total_changes > 100000:
            interval_hours = 18
        else:
            interval_hours = 24
        
        return datetime.now() + timedelta(hours=interval_hours)

    def _estimate_vacuum_duration(self, stats: Dict[str, Any]) -> int:
        """Estimate vacuum duration in minutes"""
        
        # Base estimate: 1 minute per GB, adjusted for dead tuples
        base_minutes = max(stats['table_size_mb'] / 1024, 1)
        
        # Adjust for dead tuple ratio
        dead_tuple_multiplier = 1 + (stats['dead_tup_ratio'] * 2)
        
        estimated_minutes = base_minutes * dead_tuple_multiplier
        
        return min(int(estimated_minutes), 120)  # Cap at 2 hours

    def _estimate_analyze_duration(self, stats: Dict[str, Any]) -> int:
        """Estimate analyze duration in minutes"""
        
        # Analyze is generally faster than vacuum
        base_minutes = max(stats['table_size_mb'] / 5120, 1)  # 1 min per 5GB
        
        return min(int(base_minutes), 30)  # Cap at 30 minutes

    def _estimate_compression_duration(self, stats: Dict[str, Any]) -> int:
        """Estimate compression duration in minutes"""
        
        # Compression time depends on data volume and existing compression
        base_minutes = max(stats['table_size_mb'] / 2048, 2)  # 1 min per 2GB
        
        return min(int(base_minutes), 60)  # Cap at 1 hour

    def _is_timescaledb_table(self, table_name: str) -> bool:
        """Check if table is a TimescaleDB hypertable"""
        return table_name in [
            'price_history_optimized',
            'technical_indicators_optimized',
            'news_sentiment_bulk'
        ]

    async def _create_index_maintenance_tasks(self) -> List[MaintenanceTask]:
        """Create tasks for index maintenance"""
        
        tasks = []
        
        # Get index statistics
        async with self.pool.acquire() as conn:
            index_stats = await conn.fetch("""
                SELECT 
                    schemaname,
                    tablename,
                    indexname,
                    idx_scan,
                    idx_tup_read,
                    idx_tup_fetch,
                    pg_relation_size(indexrelid) as index_size_bytes
                FROM pg_stat_user_indexes 
                WHERE schemaname = 'public'
                AND tablename IN (
                    'price_history_optimized',
                    'technical_indicators_optimized',
                    'news_sentiment_bulk'
                )
                AND idx_scan > 0  -- Only used indexes
            """)
            
            for idx in index_stats:
                # Create reindex task for large, heavily used indexes
                if idx['index_size_bytes'] > 100 * 1024 * 1024:  # > 100MB
                    reindex_task = MaintenanceTask(
                        task_id=f"reindex_{idx['indexname']}_{datetime.now().strftime('%Y%m%d_%H%M')}",
                        task_type=MaintenanceType.REINDEX,
                        target_table=idx['tablename'],
                        priority=4,
                        estimated_duration_minutes=max(idx['index_size_bytes'] // (50 * 1024 * 1024), 5),
                        last_run=None,
                        next_run=datetime.now() + timedelta(days=7),  # Weekly reindex
                        parameters={
                            'index_name': idx['indexname'],
                            'concurrent': True
                        },
                        dependencies=[]
                    )
                    tasks.append(reindex_task)
        
        return tasks

    def _create_cleanup_tasks(self) -> List[MaintenanceTask]:
        """Create cleanup tasks for old data and temporary objects"""
        
        tasks = []
        
        # Cleanup old staging data
        cleanup_staging = MaintenanceTask(
            task_id=f"cleanup_staging_{datetime.now().strftime('%Y%m%d_%H%M')}",
            task_type=MaintenanceType.CLEANUP,
            target_table='staging_tables',
            priority=3,
            estimated_duration_minutes=10,
            last_run=None,
            next_run=datetime.now() + timedelta(hours=6),
            parameters={
                'cleanup_type': 'staging_data',
                'retention_hours': 24
            },
            dependencies=[]
        )
        tasks.append(cleanup_staging)
        
        # Cleanup old log entries
        cleanup_logs = MaintenanceTask(
            task_id=f"cleanup_logs_{datetime.now().strftime('%Y%m%d_%H%M')}",
            task_type=MaintenanceType.CLEANUP,
            target_table='system_metrics',
            priority=2,
            estimated_duration_minutes=5,
            last_run=None,
            next_run=datetime.now() + timedelta(days=1),
            parameters={
                'cleanup_type': 'old_logs',
                'retention_days': 30
            },
            dependencies=[]
        )
        tasks.append(cleanup_logs)
        
        return tasks

    async def execute_maintenance_task(self, task: MaintenanceTask) -> MaintenanceResult:
        """Execute a single maintenance task with comprehensive monitoring"""
        
        start_time = datetime.now()
        size_before_mb = 0.0
        size_after_mb = 0.0
        
        try:
            # Get initial size
            size_before_mb = await self._get_table_size_mb(task.target_table)
            
            # Execute the task based on type
            if task.task_type == MaintenanceType.VACUUM:
                result = await self._execute_vacuum_task(task)
            elif task.task_type == MaintenanceType.ANALYZE:
                result = await self._execute_analyze_task(task)
            elif task.task_type == MaintenanceType.REINDEX:
                result = await self._execute_reindex_task(task)
            elif task.task_type == MaintenanceType.COMPRESSION:
                result = await self._execute_compression_task(task)
            elif task.task_type == MaintenanceType.CLEANUP:
                result = await self._execute_cleanup_task(task)
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")
            
            # Get final size
            size_after_mb = await self._get_table_size_mb(task.target_table)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            maintenance_result = MaintenanceResult(
                task_id=task.task_id,
                task_type=task.task_type,
                target_table=task.target_table,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                success=True,
                rows_affected=result.get('rows_affected', 0),
                size_before_mb=size_before_mb,
                size_after_mb=size_after_mb,
                space_reclaimed_mb=max(0, size_before_mb - size_after_mb),
                error_message=None,
                metrics=result.get('metrics', {})
            )
            
            logger.info(f"Completed task {task.task_id} in {duration:.2f}s, "
                       f"reclaimed {maintenance_result.space_reclaimed_mb:.2f}MB")
            
            return maintenance_result
            
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.error(f"Task {task.task_id} failed: {e}")
            
            return MaintenanceResult(
                task_id=task.task_id,
                task_type=task.task_type,
                target_table=task.target_table,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                success=False,
                rows_affected=0,
                size_before_mb=size_before_mb,
                size_after_mb=size_before_mb,  # No change on failure
                space_reclaimed_mb=0.0,
                error_message=str(e),
                metrics={}
            )

    async def _get_table_size_mb(self, table_name: str) -> float:
        """Get table size in MB"""
        
        async with self.pool.acquire() as conn:
            try:
                size_bytes = await conn.fetchval("""
                    SELECT pg_total_relation_size($1)
                """, table_name)
                return (size_bytes or 0) / (1024 * 1024)
            except:
                return 0.0

    async def _execute_vacuum_task(self, task: MaintenanceTask) -> Dict[str, Any]:
        """Execute vacuum operation"""
        
        config = self.table_configs.get(task.target_table, {})
        
        async with self.pool.acquire() as conn:
            # Set maintenance work memory
            maintenance_work_mem = task.parameters.get(
                'maintenance_work_mem', 
                config.get('maintenance_work_mem', '256MB')
            )
            await conn.execute(f"SET maintenance_work_mem = '{maintenance_work_mem}'")
            
            # Execute vacuum
            vacuum_type = task.parameters.get('vacuum_type', 'standard')
            analyze_flag = task.parameters.get('analyze', True)
            
            if vacuum_type == 'full':
                vacuum_sql = f"VACUUM (FULL, ANALYZE) {task.target_table}"
            else:
                analyze_part = ", ANALYZE" if analyze_flag else ""
                vacuum_sql = f"VACUUM (VERBOSE{analyze_part}) {task.target_table}"
            
            # Execute and capture output
            start_time = datetime.now()
            await conn.execute(vacuum_sql)
            duration = (datetime.now() - start_time).total_seconds()
            
            # Get post-vacuum statistics
            stats = await conn.fetchrow("""
                SELECT n_dead_tup, n_live_tup, last_vacuum
                FROM pg_stat_user_tables 
                WHERE tablename = $1
            """, task.target_table)
            
            return {
                'rows_affected': stats['n_live_tup'] if stats else 0,
                'metrics': {
                    'vacuum_type': vacuum_type,
                    'duration_seconds': duration,
                    'dead_tuples_removed': stats['n_dead_tup'] if stats else 0,
                    'maintenance_work_mem': maintenance_work_mem
                }
            }

    async def _execute_analyze_task(self, task: MaintenanceTask) -> Dict[str, Any]:
        """Execute analyze operation"""
        
        async with self.pool.acquire() as conn:
            # Set statistics target if specified
            stats_target = task.parameters.get('statistics_target')
            if stats_target:
                await conn.execute(f"""
                    ALTER TABLE {task.target_table} 
                    ALTER COLUMN date SET STATISTICS {stats_target}
                """)
            
            # Execute analyze
            start_time = datetime.now()
            await conn.execute(f"ANALYZE (VERBOSE) {task.target_table}")
            duration = (datetime.now() - start_time).total_seconds()
            
            # Get table statistics
            stats = await conn.fetchrow("""
                SELECT n_live_tup, last_analyze
                FROM pg_stat_user_tables 
                WHERE tablename = $1
            """, task.target_table)
            
            return {
                'rows_affected': stats['n_live_tup'] if stats else 0,
                'metrics': {
                    'duration_seconds': duration,
                    'statistics_target': stats_target or 'default'
                }
            }

    async def _execute_reindex_task(self, task: MaintenanceTask) -> Dict[str, Any]:
        """Execute reindex operation"""
        
        async with self.pool.acquire() as conn:
            index_name = task.parameters['index_name']
            concurrent = task.parameters.get('concurrent', True)
            
            # Get index size before
            size_before = await conn.fetchval("""
                SELECT pg_relation_size($1)
            """, index_name)
            
            # Execute reindex
            reindex_sql = f"REINDEX INDEX {'CONCURRENTLY' if concurrent else ''} {index_name}"
            
            start_time = datetime.now()
            await conn.execute(reindex_sql)
            duration = (datetime.now() - start_time).total_seconds()
            
            # Get index size after
            size_after = await conn.fetchval("""
                SELECT pg_relation_size($1)
            """, index_name)
            
            return {
                'rows_affected': 0,
                'metrics': {
                    'index_name': index_name,
                    'duration_seconds': duration,
                    'size_before_mb': (size_before or 0) / (1024 * 1024),
                    'size_after_mb': (size_after or 0) / (1024 * 1024),
                    'concurrent': concurrent
                }
            }

    async def _execute_compression_task(self, task: MaintenanceTask) -> Dict[str, Any]:
        """Execute TimescaleDB compression task"""
        
        async with self.pool.acquire() as conn:
            compress_older_than = task.parameters.get('compress_older_than', '4 hours')
            
            # Compress eligible chunks
            start_time = datetime.now()
            
            result = await conn.fetch(f"""
                SELECT compress_chunk(chunk)
                FROM show_chunks('{task.target_table}') AS chunk
                WHERE range_start < NOW() - INTERVAL '{compress_older_than}'
                AND NOT is_compressed
                LIMIT 10
            """)
            
            duration = (datetime.now() - start_time).total_seconds()
            chunks_compressed = len(result)
            
            return {
                'rows_affected': chunks_compressed,
                'metrics': {
                    'chunks_compressed': chunks_compressed,
                    'duration_seconds': duration,
                    'compress_older_than': compress_older_than
                }
            }

    async def _execute_cleanup_task(self, task: MaintenanceTask) -> Dict[str, Any]:
        """Execute cleanup task"""
        
        cleanup_type = task.parameters['cleanup_type']
        rows_deleted = 0
        
        async with self.pool.acquire() as conn:
            if cleanup_type == 'staging_data':
                retention_hours = task.parameters.get('retention_hours', 24)
                
                # Clean up staging tables
                result = await conn.execute("""
                    DELETE FROM price_data_staging 
                    WHERE source_batch_id IN (
                        SELECT source_batch_id 
                        FROM price_data_staging 
                        GROUP BY source_batch_id 
                        HAVING MIN(date) < CURRENT_DATE - INTERVAL '%s hours'
                    )
                """ % retention_hours)
                
                rows_deleted = int(result.split()[-1]) if result.split() else 0
                
            elif cleanup_type == 'old_logs':
                retention_days = task.parameters.get('retention_days', 30)
                
                result = await conn.execute("""
                    DELETE FROM system_metrics 
                    WHERE timestamp < NOW() - INTERVAL '%s days'
                """ % retention_days)
                
                rows_deleted = int(result.split()[-1]) if result.split() else 0
            
            return {
                'rows_affected': rows_deleted,
                'metrics': {
                    'cleanup_type': cleanup_type,
                    'rows_deleted': rows_deleted
                }
            }

    async def run_maintenance_schedule(
        self, 
        max_concurrent_tasks: int = 2,
        max_duration_hours: int = 4
    ) -> Dict[str, Any]:
        """Run the complete maintenance schedule with intelligent task management"""
        
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=max_duration_hours)
        
        # Create maintenance schedule
        tasks = await self.create_maintenance_schedule()
        logger.info(f"Created maintenance schedule with {len(tasks)} tasks")
        
        # Track results
        completed_tasks = []
        failed_tasks = []
        skipped_tasks = []
        
        # Create semaphore for concurrent task limiting
        semaphore = asyncio.Semaphore(max_concurrent_tasks)
        
        async def execute_task_with_semaphore(task: MaintenanceTask):
            async with semaphore:
                return await self.execute_maintenance_task(task)
        
        # Execute tasks by priority and dependencies
        remaining_tasks = tasks.copy()
        
        while remaining_tasks and datetime.now() < end_time:
            # Find tasks that can be executed (dependencies met)
            ready_tasks = []
            for task in remaining_tasks:
                if all(dep_id in [t.task_id for t in completed_tasks] for dep_id in task.dependencies):
                    ready_tasks.append(task)
            
            if not ready_tasks:
                logger.warning("No ready tasks found, breaking execution loop")
                skipped_tasks.extend(remaining_tasks)
                break
            
            # Take highest priority tasks up to concurrent limit
            tasks_to_execute = ready_tasks[:max_concurrent_tasks]
            
            # Execute tasks concurrently
            task_futures = [execute_task_with_semaphore(task) for task in tasks_to_execute]
            results = await asyncio.gather(*task_futures, return_exceptions=True)
            
            # Process results
            for task, result in zip(tasks_to_execute, results):
                remaining_tasks.remove(task)
                
                if isinstance(result, Exception):
                    failed_tasks.append({
                        'task': task,
                        'error': str(result)
                    })
                    logger.error(f"Task {task.task_id} failed: {result}")
                else:
                    completed_tasks.append(result)
                    logger.info(f"Task {task.task_id} completed successfully")
        
        # Calculate summary statistics
        total_duration = (datetime.now() - start_time).total_seconds()
        total_space_reclaimed = sum(t.space_reclaimed_mb for t in completed_tasks)
        
        return {
            'maintenance_summary': {
                'start_time': start_time,
                'end_time': datetime.now(),
                'total_duration_seconds': total_duration,
                'total_tasks_scheduled': len(tasks),
                'completed_tasks': len(completed_tasks),
                'failed_tasks': len(failed_tasks),
                'skipped_tasks': len(skipped_tasks),
                'total_space_reclaimed_mb': total_space_reclaimed
            },
            'completed_tasks': [asdict(t) for t in completed_tasks],
            'failed_tasks': failed_tasks,
            'skipped_tasks': [asdict(t) for t in skipped_tasks],
            'performance_metrics': {
                'avg_task_duration': statistics.mean([t.duration_seconds for t in completed_tasks]) if completed_tasks else 0,
                'total_space_reclaimed_gb': total_space_reclaimed / 1024,
                'tasks_per_hour': len(completed_tasks) / (total_duration / 3600) if total_duration > 0 else 0
            }
        }

    async def get_maintenance_recommendations(self) -> Dict[str, Any]:
        """Get intelligent maintenance recommendations based on current database state"""
        
        table_stats = await self._get_table_statistics()
        
        recommendations = {
            'urgent_actions': [],
            'scheduled_actions': [],
            'optimization_opportunities': [],
            'configuration_suggestions': []
        }
        
        for table_name, stats in table_stats.items():
            # Urgent actions
            if stats['dead_tup_ratio'] > 0.3:
                recommendations['urgent_actions'].append({
                    'action': 'VACUUM FULL',
                    'table': table_name,
                    'reason': f"High dead tuple ratio: {stats['dead_tup_ratio']:.2%}",
                    'priority': 10
                })
            
            if stats['table_size_mb'] > 10000 and not stats['last_analyze']:
                recommendations['urgent_actions'].append({
                    'action': 'ANALYZE',
                    'table': table_name,
                    'reason': 'Large table never analyzed',
                    'priority': 8
                })
            
            # Scheduled actions
            if stats['activity_score'] > 5:
                recommendations['scheduled_actions'].append({
                    'action': 'Increase vacuum frequency',
                    'table': table_name,
                    'reason': f"High activity score: {stats['activity_score']:.1f}",
                    'suggested_interval': '6 hours'
                })
            
            # Optimization opportunities
            if stats['table_size_mb'] > 1000 and stats['dead_tup_ratio'] < 0.02:
                recommendations['optimization_opportunities'].append({
                    'opportunity': 'Reduce vacuum frequency',
                    'table': table_name,
                    'reason': 'Low dead tuple ratio on large table',
                    'potential_saving': 'CPU and I/O resources'
                })
        
        # Configuration suggestions
        total_size_gb = sum(s['table_size_mb'] for s in table_stats.values()) / 1024
        if total_size_gb > 100:
            recommendations['configuration_suggestions'].extend([
                {
                    'parameter': 'maintenance_work_mem',
                    'suggested_value': '2GB',
                    'reason': 'Large database size warrants increased maintenance memory'
                },
                {
                    'parameter': 'autovacuum_max_workers',
                    'suggested_value': '6',
                    'reason': 'High-volume tables benefit from more vacuum workers'
                }
            ])
        
        return recommendations


# Example usage and testing
async def example_maintenance_workflow():
    """Example of complete maintenance workflow"""
    
    # Create connection pool
    pool = await asyncpg.create_pool(
        host="localhost",
        port=5432,
        user="postgres",
        password="password",
        database="stock_db",
        min_size=3,
        max_size=10
    )
    
    maintainer = StockDatabaseMaintainer(pool)
    
    try:
        # Get maintenance recommendations
        recommendations = await maintainer.get_maintenance_recommendations()
        print("Maintenance Recommendations:")
        print(json.dumps(recommendations, indent=2, default=str))
        
        # Run maintenance schedule
        results = await maintainer.run_maintenance_schedule(
            max_concurrent_tasks=2,
            max_duration_hours=2
        )
        
        print(f"\nMaintenance completed:")
        print(f"- Completed tasks: {results['maintenance_summary']['completed_tasks']}")
        print(f"- Space reclaimed: {results['maintenance_summary']['total_space_reclaimed_mb']:.2f} MB")
        print(f"- Duration: {results['maintenance_summary']['total_duration_seconds']:.2f} seconds")
        
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(example_maintenance_workflow())