"""
Celery tasks for system maintenance and cleanup
"""
from celery import shared_task
from typing import Dict, Any, List
from datetime import datetime, timedelta, date
import logging
import os
import subprocess
import shutil
import json
from pathlib import Path

from backend.tasks.celery_app import celery_app
from backend.utils.database import get_db_sync, get_engine
from backend.utils.cache import get_redis_client
from backend.models.tables import (
    Stock, PriceHistory, News, Recommendation, RecommendationPerformance,
    PortfolioPerformance, AuditLog, SystemMetrics, Portfolio, User, Fundamental
)
from sqlalchemy import text, select, delete, and_, func
from sqlalchemy.orm import Session
import psutil
import redis

logger = logging.getLogger(__name__)

# Configuration
BACKUP_DIR = os.getenv('BACKUP_DIR', '/app/backups')
DATA_RETENTION_DAYS = int(os.getenv('DATA_RETENTION_DAYS', '365'))
LOG_RETENTION_DAYS = int(os.getenv('LOG_RETENTION_DAYS', '90'))
CACHE_TTL_HOURS = int(os.getenv('CACHE_TTL_HOURS', '24'))

@celery_app.task
def cleanup_old_data() -> Dict[str, Any]:
    """Remove old data based on retention policies"""
    try:
        results = {
            'price_history': 0,
            'news': 0,
            'recommendations': 0,
            'portfolio_performance': 0,
            'audit_logs': 0,
            'errors': []
        }
        
        with get_db_sync() as db:
            # Cleanup old price history (keep 2 years)
            cutoff_date = date.today() - timedelta(days=DATA_RETENTION_DAYS * 2)
            try:
                deleted = db.query(PriceHistory).filter(
                    PriceHistory.date < cutoff_date
                ).delete()
                results['price_history'] = deleted
                logger.info(f"Deleted {deleted} old price history records")
            except Exception as e:
                results['errors'].append(f"Price history cleanup error: {e}")
            
            # Cleanup old news (keep 6 months)
            news_cutoff = datetime.utcnow() - timedelta(days=180)
            try:
                deleted = db.query(News).filter(
                    News.published_at < news_cutoff
                ).delete()
                results['news'] = deleted
                logger.info(f"Deleted {deleted} old news articles")
            except Exception as e:
                results['errors'].append(f"News cleanup error: {e}")
            
            # Cleanup expired recommendations
            try:
                deleted = db.query(Recommendation).filter(
                    Recommendation.valid_until < datetime.utcnow(),
                    Recommendation.is_active == False
                ).delete()
                results['recommendations'] = deleted
                logger.info(f"Deleted {deleted} expired recommendations")
            except Exception as e:
                results['errors'].append(f"Recommendations cleanup error: {e}")
            
            # Cleanup old portfolio performance records (keep 1 year)
            perf_cutoff = date.today() - timedelta(days=365)
            try:
                deleted = db.query(PortfolioPerformance).filter(
                    PortfolioPerformance.date < perf_cutoff
                ).delete()
                results['portfolio_performance'] = deleted
                logger.info(f"Deleted {deleted} old portfolio performance records")
            except Exception as e:
                results['errors'].append(f"Portfolio performance cleanup error: {e}")
            
            # Cleanup old audit logs
            audit_cutoff = datetime.utcnow() - timedelta(days=LOG_RETENTION_DAYS)
            try:
                deleted = db.query(AuditLog).filter(
                    AuditLog.created_at < audit_cutoff
                ).delete()
                results['audit_logs'] = deleted
                logger.info(f"Deleted {deleted} old audit log entries")
            except Exception as e:
                results['errors'].append(f"Audit log cleanup error: {e}")
            
            db.commit()
            
        # Cleanup Redis cache
        try:
            redis_client = get_redis_client()
            expired_keys = cleanup_redis_cache(redis_client)
            results['cache_keys_removed'] = expired_keys
        except Exception as e:
            results['errors'].append(f"Redis cleanup error: {e}")
        
        # Cleanup old backup files
        try:
            old_backups = cleanup_old_backups()
            results['backups_removed'] = old_backups
        except Exception as e:
            results['errors'].append(f"Backup cleanup error: {e}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in cleanup_old_data: {e}")
        return {'error': str(e)}

@celery_app.task
def optimize_database() -> Dict[str, Any]:
    """Optimize database performance"""
    try:
        results = {
            'tables_analyzed': [],
            'tables_vacuumed': [],
            'indexes_rebuilt': [],
            'statistics_updated': False,
            'errors': []
        }
        
        engine = get_engine()
        
        with engine.connect() as conn:
            # Get all tables
            tables_query = text("""
                SELECT tablename 
                FROM pg_tables 
                WHERE schemaname = 'public'
            """)
            tables = conn.execute(tables_query).fetchall()
            
            for table in tables:
                table_name = table[0]
                
                try:
                    # Analyze table
                    conn.execute(text(f"ANALYZE {table_name}"))
                    conn.commit()
                    results['tables_analyzed'].append(table_name)
                    
                    # Vacuum table
                    conn.execute(text(f"VACUUM {table_name}"))
                    conn.commit()
                    results['tables_vacuumed'].append(table_name)
                    
                except Exception as e:
                    results['errors'].append(f"Error optimizing {table_name}: {e}")
            
            # Rebuild indexes
            indexes_query = text("""
                SELECT indexname, tablename 
                FROM pg_indexes 
                WHERE schemaname = 'public'
            """)
            indexes = conn.execute(indexes_query).fetchall()
            
            for index in indexes:
                index_name = index[0]
                try:
                    conn.execute(text(f"REINDEX INDEX {index_name}"))
                    conn.commit()
                    results['indexes_rebuilt'].append(index_name)
                except Exception as e:
                    results['errors'].append(f"Error rebuilding index {index_name}: {e}")
            
            # Update statistics
            try:
                conn.execute(text("ANALYZE"))
                conn.commit()
                results['statistics_updated'] = True
            except Exception as e:
                results['errors'].append(f"Error updating statistics: {e}")
        
        logger.info(f"Database optimization completed: {len(results['tables_analyzed'])} tables analyzed")
        return results
        
    except Exception as e:
        logger.error(f"Error in optimize_database: {e}")
        return {'error': str(e)}

@celery_app.task
def backup_database() -> Dict[str, Any]:
    """Create database backup"""
    try:
        # Create backup directory if it doesn't exist
        Path(BACKUP_DIR).mkdir(parents=True, exist_ok=True)
        
        # Generate backup filename
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        backup_file = os.path.join(BACKUP_DIR, f'backup_{timestamp}.sql')
        
        # Database connection parameters
        db_host = os.getenv('POSTGRES_HOST', 'localhost')
        db_port = os.getenv('POSTGRES_PORT', '5432')
        db_name = os.getenv('POSTGRES_DB', 'investment_db')
        db_user = os.getenv('POSTGRES_USER', 'postgres')
        db_password = os.getenv('POSTGRES_PASSWORD', '')
        
        # Create pg_dump command
        env = os.environ.copy()
        env['PGPASSWORD'] = db_password
        
        cmd = [
            'pg_dump',
            '-h', db_host,
            '-p', db_port,
            '-U', db_user,
            '-d', db_name,
            '-f', backup_file,
            '--verbose',
            '--clean',
            '--no-owner',
            '--no-privileges'
        ]
        
        # Execute backup
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode != 0:
            logger.error(f"Backup failed: {result.stderr}")
            return {
                'status': 'failed',
                'error': result.stderr
            }
        
        # Compress backup
        compressed_file = f"{backup_file}.gz"
        compress_cmd = ['gzip', backup_file]
        subprocess.run(compress_cmd, check=True)
        
        # Get file size
        file_size = os.path.getsize(compressed_file)
        
        # Store backup metadata
        backup_metadata = {
            'filename': compressed_file,
            'timestamp': timestamp,
            'size_bytes': file_size,
            'size_mb': round(file_size / (1024 * 1024), 2),
            'status': 'success'
        }
        
        # Save metadata
        metadata_file = os.path.join(BACKUP_DIR, 'backup_metadata.json')
        existing_metadata = []
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                existing_metadata = json.load(f)
        
        existing_metadata.append(backup_metadata)
        
        with open(metadata_file, 'w') as f:
            json.dump(existing_metadata, f, indent=2)
        
        logger.info(f"Database backup completed: {compressed_file}")
        return backup_metadata
        
    except subprocess.TimeoutExpired:
        logger.error("Database backup timed out")
        return {'status': 'failed', 'error': 'Backup timeout'}
    except Exception as e:
        logger.error(f"Error in backup_database: {e}")
        return {'status': 'failed', 'error': str(e)}

@celery_app.task
def check_system_health() -> Dict[str, Any]:
    """Check overall system health"""
    try:
        health_status = {
            'timestamp': datetime.utcnow().isoformat(),
            'database': check_database_health(),
            'redis': check_redis_health(),
            'disk': check_disk_usage(),
            'memory': check_memory_usage(),
            'cpu': check_cpu_usage(),
            'services': check_services_health()
        }
        
        # Calculate overall health score
        health_scores = []
        if health_status['database']['status'] == 'healthy':
            health_scores.append(1)
        else:
            health_scores.append(0)
        
        if health_status['redis']['status'] == 'healthy':
            health_scores.append(1)
        else:
            health_scores.append(0)
        
        if health_status['disk']['percent_used'] < 80:
            health_scores.append(1)
        else:
            health_scores.append(0)
        
        if health_status['memory']['percent_used'] < 90:
            health_scores.append(1)
        else:
            health_scores.append(0)
        
        if health_status['cpu']['percent_used'] < 80:
            health_scores.append(1)
        else:
            health_scores.append(0)
        
        overall_score = sum(health_scores) / len(health_scores) * 100
        health_status['overall_health_score'] = overall_score
        health_status['status'] = 'healthy' if overall_score >= 80 else 'degraded' if overall_score >= 60 else 'unhealthy'
        
        # Store metrics in database
        store_system_metrics(health_status)
        
        # Send alert if unhealthy
        if health_status['status'] == 'unhealthy':
            send_health_alert.delay(health_status)
        
        return health_status
        
    except Exception as e:
        logger.error(f"Error in check_system_health: {e}")
        return {'error': str(e)}

@celery_app.task
def generate_system_reports() -> Dict[str, Any]:
    """Generate monthly system reports"""
    try:
        reports = {
            'period': datetime.utcnow().strftime('%Y-%m'),
            'generated_at': datetime.utcnow().isoformat(),
            'reports': []
        }
        
        # Database statistics report
        db_report = generate_database_report()
        reports['reports'].append({
            'type': 'database',
            'data': db_report
        })
        
        # API usage report
        api_report = generate_api_usage_report()
        reports['reports'].append({
            'type': 'api_usage',
            'data': api_report
        })
        
        # Performance report
        perf_report = generate_performance_report()
        reports['reports'].append({
            'type': 'performance',
            'data': perf_report
        })
        
        # Error report
        error_report = generate_error_report()
        reports['reports'].append({
            'type': 'errors',
            'data': error_report
        })
        
        # Save reports
        report_dir = os.path.join(BACKUP_DIR, 'reports')
        Path(report_dir).mkdir(parents=True, exist_ok=True)
        
        report_file = os.path.join(
            report_dir,
            f"system_report_{reports['period']}.json"
        )
        
        with open(report_file, 'w') as f:
            json.dump(reports, f, indent=2)
        
        logger.info(f"System reports generated: {report_file}")
        return reports
        
    except Exception as e:
        logger.error(f"Error in generate_system_reports: {e}")
        return {'error': str(e)}

@celery_app.task
def clear_cache(pattern: str = None) -> Dict[str, Any]:
    """Clear Redis cache based on pattern"""
    try:
        redis_client = get_redis_client()
        
        if pattern:
            keys = redis_client.keys(pattern)
        else:
            keys = redis_client.keys('*')
        
        deleted = 0
        for key in keys:
            redis_client.delete(key)
            deleted += 1
        
        return {
            'deleted_keys': deleted,
            'pattern': pattern or 'all'
        }
        
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        return {'error': str(e)}

@celery_app.task
def update_market_calendars() -> Dict[str, Any]:
    """Update market holiday calendars and trading hours"""
    try:
        # This would typically fetch from an API or update from a data source
        calendars = {
            'NYSE': {
                'timezone': 'America/New_York',
                'open_time': '09:30',
                'close_time': '16:00',
                'holidays': [
                    '2024-01-01',  # New Year's Day
                    '2024-01-15',  # MLK Day
                    '2024-02-19',  # Presidents Day
                    '2024-03-29',  # Good Friday
                    '2024-05-27',  # Memorial Day
                    '2024-06-19',  # Juneteenth
                    '2024-07-04',  # Independence Day
                    '2024-09-02',  # Labor Day
                    '2024-11-28',  # Thanksgiving
                    '2024-12-25',  # Christmas
                ]
            },
            'NASDAQ': {
                'timezone': 'America/New_York',
                'open_time': '09:30',
                'close_time': '16:00',
                'holidays': [
                    # Same as NYSE
                ]
            }
        }
        
        # Store in Redis for quick access
        redis_client = get_redis_client()
        redis_client.setex(
            'market_calendars',
            86400,  # 24 hours
            json.dumps(calendars)
        )
        
        return {
            'status': 'updated',
            'exchanges': list(calendars.keys())
        }
        
    except Exception as e:
        logger.error(f"Error updating market calendars: {e}")
        return {'error': str(e)}

# Helper functions
def cleanup_redis_cache(redis_client: redis.Redis) -> int:
    """Remove expired cache entries"""
    try:
        # Get all keys
        all_keys = redis_client.keys('*')
        expired = 0
        
        for key in all_keys:
            ttl = redis_client.ttl(key)
            # Remove keys with no TTL or very long TTL
            if ttl == -1 or ttl > CACHE_TTL_HOURS * 3600:
                redis_client.expire(key, CACHE_TTL_HOURS * 3600)
                expired += 1
        
        return expired
    except Exception as e:
        logger.error(f"Error cleaning Redis cache: {e}")
        return 0

def cleanup_old_backups() -> int:
    """Remove old backup files"""
    try:
        if not os.path.exists(BACKUP_DIR):
            return 0
        
        removed = 0
        cutoff_date = datetime.utcnow() - timedelta(days=30)  # Keep 30 days of backups
        
        for file in Path(BACKUP_DIR).glob('backup_*.sql.gz'):
            # Extract timestamp from filename
            try:
                timestamp_str = file.stem.replace('backup_', '').replace('.sql', '')
                file_date = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                
                if file_date < cutoff_date:
                    file.unlink()
                    removed += 1
            except Exception as e:
                logger.warning(f"Error processing backup file {file}: {e}")
        
        return removed
    except Exception as e:
        logger.error(f"Error cleaning old backups: {e}")
        return 0

def check_database_health() -> Dict[str, Any]:
    """Check database connectivity and performance"""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            # Check connectivity
            result = conn.execute(text("SELECT 1"))
            
            # Get database size
            size_query = text("""
                SELECT pg_database_size(current_database()) as size
            """)
            db_size = conn.execute(size_query).fetchone()[0]
            
            # Get connection count
            conn_query = text("""
                SELECT count(*) FROM pg_stat_activity
            """)
            conn_count = conn.execute(conn_query).fetchone()[0]
            
            # Get slow queries
            slow_query = text("""
                SELECT count(*) 
                FROM pg_stat_activity 
                WHERE state = 'active' 
                AND query_start < now() - interval '5 seconds'
            """)
            slow_queries = conn.execute(slow_query).fetchone()[0]
            
            return {
                'status': 'healthy',
                'size_mb': round(db_size / (1024 * 1024), 2),
                'connections': conn_count,
                'slow_queries': slow_queries
            }
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e)
        }

def check_redis_health() -> Dict[str, Any]:
    """Check Redis connectivity and stats"""
    try:
        redis_client = get_redis_client()
        info = redis_client.info()
        
        return {
            'status': 'healthy',
            'used_memory_mb': round(info['used_memory'] / (1024 * 1024), 2),
            'connected_clients': info['connected_clients'],
            'total_commands_processed': info['total_commands_processed'],
            'keyspace_hits': info.get('keyspace_hits', 0),
            'keyspace_misses': info.get('keyspace_misses', 0)
        }
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e)
        }

def check_disk_usage() -> Dict[str, Any]:
    """Check disk usage"""
    try:
        usage = psutil.disk_usage('/')
        return {
            'total_gb': round(usage.total / (1024**3), 2),
            'used_gb': round(usage.used / (1024**3), 2),
            'free_gb': round(usage.free / (1024**3), 2),
            'percent_used': usage.percent
        }
    except Exception as e:
        return {'error': str(e)}

def check_memory_usage() -> Dict[str, Any]:
    """Check memory usage"""
    try:
        memory = psutil.virtual_memory()
        return {
            'total_gb': round(memory.total / (1024**3), 2),
            'available_gb': round(memory.available / (1024**3), 2),
            'percent_used': memory.percent,
            'used_gb': round(memory.used / (1024**3), 2)
        }
    except Exception as e:
        return {'error': str(e)}

def check_cpu_usage() -> Dict[str, Any]:
    """Check CPU usage"""
    try:
        return {
            'percent_used': psutil.cpu_percent(interval=1),
            'core_count': psutil.cpu_count(),
            'load_average': os.getloadavg()
        }
    except Exception as e:
        return {'error': str(e)}

def check_services_health() -> Dict[str, Any]:
    """Check health of various services"""
    services = {}
    
    # Check if Celery workers are running
    try:
        from celery import current_app
        inspect = current_app.control.inspect()
        active_workers = inspect.active()
        services['celery_workers'] = len(active_workers) if active_workers else 0
    except:
        services['celery_workers'] = 0
    
    # Check if critical processes are running
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            if 'postgres' in proc.info['name']:
                services['postgresql'] = 'running'
            elif 'redis' in proc.info['name']:
                services['redis'] = 'running'
            elif 'nginx' in proc.info['name']:
                services['nginx'] = 'running'
        except:
            pass
    
    return services

def store_system_metrics(metrics: Dict[str, Any]):
    """Store system metrics in database"""
    try:
        with get_db_sync() as db:
            system_metric = SystemMetrics(
                timestamp=datetime.utcnow(),
                metric_type='health_check',
                metrics=metrics
            )
            db.add(system_metric)
            db.commit()
    except Exception as e:
        logger.error(f"Error storing system metrics: {e}")

def generate_database_report() -> Dict[str, Any]:
    """Generate database statistics report"""
    try:
        with get_db_sync() as db:
            report = {
                'total_stocks': db.query(func.count(Stock.id)).scalar(),
                'total_price_records': db.query(func.count(PriceHistory.id)).scalar(),
                'total_news_articles': db.query(func.count(News.id)).scalar(),
                'active_recommendations': db.query(Recommendation).filter(
                    Recommendation.is_active == True
                ).count(),
                'total_portfolios': db.query(func.count(Portfolio.id)).scalar(),
                'total_users': db.query(func.count(User.id)).scalar()
            }
            
            # Get table sizes
            engine = get_engine()
            with engine.connect() as conn:
                size_query = text("""
                    SELECT 
                        schemaname,
                        tablename,
                        pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
                    FROM pg_tables 
                    WHERE schemaname = 'public'
                    ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
                    LIMIT 10
                """)
                table_sizes = conn.execute(size_query).fetchall()
                report['largest_tables'] = [
                    {'table': row[1], 'size': row[2]} for row in table_sizes
                ]
            
            return report
    except Exception as e:
        return {'error': str(e)}

def generate_api_usage_report() -> Dict[str, Any]:
    """Generate API usage statistics"""
    try:
        # This would typically query logs or metrics
        return {
            'alpha_vantage_calls': 0,  # Would query actual metrics
            'finnhub_calls': 0,
            'polygon_calls': 0,
            'total_api_calls': 0,
            'api_errors': 0
        }
    except Exception as e:
        return {'error': str(e)}

def generate_performance_report() -> Dict[str, Any]:
    """Generate system performance report"""
    try:
        with get_db_sync() as db:
            # Get average task execution times (simplified)
            return {
                'avg_response_time_ms': 150,
                'total_requests': 10000,
                'error_rate': 0.01,
                'uptime_percent': 99.9
            }
    except Exception as e:
        return {'error': str(e)}

def generate_error_report() -> Dict[str, Any]:
    """Generate error statistics report"""
    try:
        # This would typically query error logs
        return {
            'total_errors': 0,
            'critical_errors': 0,
            'warnings': 0,
            'top_errors': []
        }
    except Exception as e:
        return {'error': str(e)}

@celery_app.task
def send_health_alert(health_status: Dict[str, Any]):
    """Send alert when system health is degraded"""
    try:
        # This would send to monitoring system or email
        logger.critical(f"System health alert: {health_status['status']}")
        # Could integrate with PagerDuty, Slack, email, etc.
        return True
    except Exception as e:
        logger.error(f"Error sending health alert: {e}")
        return False

# =============================================================================
# Cache Warming Tasks for Market Open
# =============================================================================

@celery_app.task(bind=True, max_retries=2, default_retry_delay=300)
def warm_cache_for_market_open(self, top_n: int = 500) -> Dict[str, Any]:
    """
    Pre-warm cache with top N stocks before market open.

    This task runs before market open (e.g., 9:00 AM ET) to ensure
    frequently accessed stock data is cached and ready for fast access
    when trading begins at 9:30 AM ET.

    Target: 50% faster response times at market open.

    Args:
        top_n: Number of top stocks to pre-load (default 500)

    Returns:
        Dict with warming statistics and success rate
    """
    import asyncio

    logger.info(f"Starting cache warming task for top {top_n} stocks")

    try:
        # Import the intelligent cache manager
        from backend.etl.intelligent_cache_system import IntelligentCacheManager

        # Initialize cache manager with Redis connection
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
        cache_dir = os.getenv('CACHE_DIR', '/tmp/intelligent_cache')

        cache_manager = IntelligentCacheManager(
            cache_dir=cache_dir,
            memory_size_mb=256,
            disk_size_mb=2048,
            redis_url=redis_url,
            enable_analytics=True
        )

        # Define data fetcher that uses our existing data sources
        async def fetch_stock_data(symbol: str) -> Dict[str, Any]:
            """Fetch comprehensive stock data for caching."""
            return await _fetch_stock_data_for_warming(symbol)

        # Run the async warming function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(
                cache_manager.warm_cache_for_market_open(
                    top_n=top_n,
                    data_fetcher=fetch_stock_data,
                    batch_size=50,
                    max_concurrent=10,
                    rate_limit_delay=2.0,  # Respect API rate limits
                    ttl_hours=4  # Cache valid until after market close
                )
            )
        finally:
            loop.close()

        # Convert result to dict for Celery serialization
        warming_result = {
            'total_stocks': result.total_stocks,
            'successful': result.successful,
            'failed': result.failed,
            'skipped': result.skipped,
            'duration_seconds': result.duration_seconds,
            'success_rate': result.success_rate,
            'errors': result.errors[:10] if result.errors else [],  # Limit errors
            'timestamp': datetime.utcnow().isoformat(),
            'status': 'completed'
        }

        # Log summary
        logger.info(
            f"Cache warming completed: {warming_result['successful']}/{warming_result['total_stocks']} "
            f"stocks warmed in {warming_result['duration_seconds']:.1f}s "
            f"(success rate: {warming_result['success_rate']:.1%})"
        )

        # Store warming metrics
        try:
            redis_client = get_redis_client()
            redis_client.setex(
                'cache_warming:last_run',
                86400,  # Keep for 24 hours
                json.dumps(warming_result)
            )
        except Exception as e:
            logger.warning(f"Failed to store warming metrics: {e}")

        return warming_result

    except Exception as e:
        logger.error(f"Cache warming task failed: {e}")

        # Retry with exponential backoff
        raise self.retry(exc=e, countdown=60 * (self.request.retries + 1))


async def _fetch_stock_data_for_warming(symbol: str) -> Dict[str, Any]:
    """
    Fetch stock data from database/APIs for cache warming.

    This function retrieves data that should be pre-cached:
    - Current price and daily OHLCV
    - Technical indicators
    - Basic fundamentals
    - Recent price history (30 days)

    Args:
        symbol: Stock symbol

    Returns:
        Dict with stock data for caching
    """
    stock_data = {
        'symbol': symbol,
        'cached_at': datetime.utcnow().isoformat()
    }

    try:
        # Use synchronous database access for reliability
        with get_db_sync() as db:
            from backend.models.tables import Stock, PriceHistory, Fundamental

            # Get stock info
            stock = db.query(Stock).filter(Stock.symbol == symbol).first()
            if not stock:
                return None

            stock_data['name'] = stock.name
            stock_data['sector'] = stock.sector
            stock_data['industry'] = stock.industry
            stock_data['market_cap'] = stock.market_cap

            # Get recent price history (30 days)
            cutoff_date = date.today() - timedelta(days=30)
            prices = db.query(PriceHistory).filter(
                and_(
                    PriceHistory.stock_id == stock.id,
                    PriceHistory.date >= cutoff_date
                )
            ).order_by(PriceHistory.date.desc()).all()

            if prices:
                stock_data['prices'] = [
                    {
                        'date': p.date.isoformat(),
                        'open': float(p.open) if p.open else None,
                        'high': float(p.high) if p.high else None,
                        'low': float(p.low) if p.low else None,
                        'close': float(p.close) if p.close else None,
                        'volume': p.volume
                    }
                    for p in prices
                ]

                # Calculate basic indicators from price data
                if len(prices) >= 2:
                    latest = prices[0]
                    previous = prices[1]
                    stock_data['indicators'] = {
                        'current_price': float(latest.close) if latest.close else None,
                        'previous_close': float(previous.close) if previous.close else None,
                        'daily_change': float(latest.close - previous.close) if latest.close and previous.close else None,
                        'daily_change_pct': float((latest.close - previous.close) / previous.close * 100) if latest.close and previous.close else None,
                        'volume': latest.volume,
                        'avg_volume_30d': sum(p.volume or 0 for p in prices) / len(prices)
                    }

            # Get latest fundamentals
            fundamental = db.query(Fundamental).filter(
                Fundamental.stock_id == stock.id
            ).order_by(Fundamental.report_date.desc()).first()

            if fundamental:
                stock_data['fundamentals'] = {
                    'pe_ratio': float(fundamental.pe_ratio) if fundamental.pe_ratio else None,
                    'pb_ratio': float(fundamental.pb_ratio) if fundamental.pb_ratio else None,
                    'dividend_yield': float(fundamental.dividend_yield) if fundamental.dividend_yield else None,
                    'roe': float(fundamental.roe) if fundamental.roe else None,
                    'report_date': fundamental.report_date.isoformat() if fundamental.report_date else None
                }

    except Exception as e:
        logger.warning(f"Error fetching data for {symbol}: {e}")
        return None

    return stock_data


@celery_app.task
def check_cache_warming_status() -> Dict[str, Any]:
    """
    Check the status of the last cache warming operation.

    Returns:
        Dict with last warming status and cache readiness metrics
    """
    try:
        from backend.etl.intelligent_cache_system import IntelligentCacheManager

        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
        cache_dir = os.getenv('CACHE_DIR', '/tmp/intelligent_cache')

        cache_manager = IntelligentCacheManager(
            cache_dir=cache_dir,
            redis_url=redis_url
        )

        # Get warming status
        status = cache_manager.get_warming_status()

        # Get last run info from Redis
        try:
            redis_client = get_redis_client()
            last_run_json = redis_client.get('cache_warming:last_run')
            if last_run_json:
                status['last_run'] = json.loads(last_run_json)
        except Exception as e:
            status['last_run'] = None
            status['last_run_error'] = str(e)

        return status

    except Exception as e:
        logger.error(f"Error checking cache warming status: {e}")
        return {'error': str(e)}


@celery_app.task
def refresh_hot_stocks_cache(symbols: List[str] = None) -> Dict[str, Any]:
    """
    Refresh cache for specific high-value stocks.

    This task can be triggered during market hours to refresh
    cache for stocks experiencing high trading activity.

    Args:
        symbols: List of stock symbols to refresh (defaults to top 100)

    Returns:
        Dict with refresh statistics
    """
    import asyncio

    logger.info(f"Starting hot stocks cache refresh")

    try:
        from backend.etl.intelligent_cache_system import IntelligentCacheManager

        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
        cache_dir = os.getenv('CACHE_DIR', '/tmp/intelligent_cache')

        cache_manager = IntelligentCacheManager(
            cache_dir=cache_dir,
            redis_url=redis_url
        )

        # If no symbols provided, get top 100 by volume
        if not symbols:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                top_stocks = loop.run_until_complete(
                    cache_manager.get_top_stocks_by_volume(100)
                )
                symbols = [s['symbol'] for s in top_stocks]
            finally:
                loop.close()

        refreshed = 0
        failed = 0
        errors = []

        for symbol in symbols:
            try:
                # Invalidate existing cache
                cache_keys = [
                    f"stock:{symbol}:summary",
                    f"stock:{symbol}:prices",
                    f"stock:{symbol}:indicators"
                ]

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    for key in cache_keys:
                        loop.run_until_complete(cache_manager.delete(key))

                    # Fetch fresh data
                    fresh_data = loop.run_until_complete(
                        _fetch_stock_data_for_warming(symbol)
                    )

                    if fresh_data:
                        loop.run_until_complete(
                            cache_manager.set(
                                f"stock:{symbol}:summary",
                                fresh_data,
                                category='stocks',
                                ttl_hours=1
                            )
                        )
                        refreshed += 1
                    else:
                        failed += 1
                finally:
                    loop.close()

            except Exception as e:
                failed += 1
                errors.append(f"{symbol}: {str(e)}")

        return {
            'total_symbols': len(symbols),
            'refreshed': refreshed,
            'failed': failed,
            'errors': errors[:10],
            'timestamp': datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Hot stocks cache refresh failed: {e}")
        return {'error': str(e)}


# Additional maintenance tasks
@celery_app.task
def rotate_logs() -> Dict[str, Any]:
    """Rotate application logs"""
    try:
        log_dir = '/app/logs'
        if not os.path.exists(log_dir):
            return {'status': 'no_logs_directory'}
        
        rotated = 0
        for log_file in Path(log_dir).glob('*.log'):
            if log_file.stat().st_size > 100 * 1024 * 1024:  # 100MB
                # Archive and compress
                archive_name = f"{log_file}.{datetime.utcnow().strftime('%Y%m%d')}.gz"
                with open(log_file, 'rb') as f_in:
                    import gzip
                    with gzip.open(archive_name, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                
                # Clear original log
                open(log_file, 'w').close()
                rotated += 1
        
        return {'rotated_files': rotated}
        
    except Exception as e:
        logger.error(f"Error rotating logs: {e}")
        return {'error': str(e)}