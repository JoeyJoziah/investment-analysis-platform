"""Async Database connection utilities - Replacement for deprecated synchronous version"""

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from typing import Generator
import logging

from backend.config.settings import settings
from backend.models.unified_models import Base
from backend.config.database import (
    db_manager, 
    initialize_database, 
    cleanup_database,
    get_async_db_session,
    get_db_session
)

logger = logging.getLogger(__name__)

# Legacy synchronous engine for backwards compatibility
# DEPRECATED: Use async database operations instead
engine = create_engine(
    settings.DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,  # Reduced for legacy use
    max_overflow=20,
    pool_timeout=30,
    pool_recycle=1800,
    echo=settings.DEBUG,
)

# Legacy session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)


def get_db() -> Generator[Session, None, None]:
    """
    DEPRECATED: Use get_async_db_session() instead.
    Legacy dependency to get synchronous database session.
    """
    logger.warning("Using deprecated synchronous database session. Migrate to async.")
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def init_db():
    """Initialize database by creating all tables"""
    try:
        # Initialize the async database manager
        await initialize_database()
        
        # Create all tables using async engine
        async with db_manager._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        logger.info("Database tables created successfully")
        
        # Run any initialization scripts if db_init exists
        try:
            from backend.utils.db_init import initialize_database_data
            await initialize_database_data()
        except ImportError:
            logger.info("No db_init module found, skipping data initialization")
        
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise


async def close_db():
    """Close database connections"""
    try:
        # Close async database connections
        await cleanup_database()
        
        # Close legacy sync engine
        engine.dispose()
        logger.info("Database connections closed")
    except Exception as e:
        logger.error(f"Error closing database: {e}")


def get_db_sync() -> Session:
    """
    DEPRECATED: Use async database operations instead.
    Get synchronous database session for legacy Celery tasks
    """
    logger.warning("Using deprecated synchronous database session in Celery task. Migrate to async.")
    db = SessionLocal()
    return db


def get_engine():
    """
    DEPRECATED: Access async engine through db_manager instead.
    Get legacy database engine
    """
    logger.warning("Using deprecated synchronous engine. Use async database manager.")
    return engine


def get_connection_pool_status() -> dict:
    """Get current connection pool status - Legacy version"""
    pool = engine.pool
    return {
        "size": pool.size(),
        "checked_in": pool.checkedin(),
        "checked_out": pool.checkedout(),
        "overflow": pool.overflow(),
        "invalid": pool.invalid(),
        "total_connections": pool.size() + pool.overflow(),
        "utilization_percent": round((pool.checkedout() / (pool.size() + pool.overflow())) * 100, 2) if (pool.size() + pool.overflow()) > 0 else 0
    }


def log_connection_pool_metrics():
    """Log connection pool metrics for monitoring - Legacy version"""
    status = get_connection_pool_status()
    logger.info(f"Legacy connection pool status: {status}")
    
    # Alert if utilization is high
    if status['utilization_percent'] > 80:
        logger.warning(f"High legacy connection pool utilization: {status['utilization_percent']}%")
    
    return status


async def check_database_connection() -> bool:
    """Check if database is accessible using async connection"""
    try:
        health_status = await db_manager.health_check()
        return health_status.get("status") == "healthy"
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False


async def monitor_connection_pool():
    """Monitor async connection pool health"""
    import asyncio
    while True:
        try:
            status = await db_manager.get_connection_pool_status()
            logger.info(f"Async connection pool status: {status}")
            
            # Check for potential issues
            metrics = status.get('metrics', {})
            if metrics.get('connection_errors', 0) > 0:
                logger.warning(f"Connection errors detected: {metrics['connection_errors']}")
            
            if metrics.get('deadlocks_detected', 0) > 0:
                logger.warning(f"Deadlocks detected: {metrics['deadlocks_detected']}")
                
        except Exception as e:
            logger.error(f"Error monitoring async connection pool: {e}")
        
        # Monitor every 30 seconds
        await asyncio.sleep(30)


# New async utilities
async def get_database_stats():
    """Get comprehensive database statistics"""
    try:
        async with get_db_session() as session:
            # Get table sizes
            table_stats = await session.execute(text("""
                SELECT 
                    schemaname,
                    tablename,
                    attname,
                    n_distinct,
                    correlation
                FROM pg_stats 
                WHERE schemaname = 'public'
                ORDER BY schemaname, tablename;
            """))
            
            # Get connection stats
            connection_stats = await session.execute(text("""
                SELECT 
                    state,
                    count(*) as count
                FROM pg_stat_activity 
                WHERE datname = current_database()
                GROUP BY state;
            """))
            
            return {
                "table_stats": [dict(row) for row in table_stats],
                "connection_stats": [dict(row) for row in connection_stats],
                "pool_status": await db_manager.get_connection_pool_status()
            }
            
    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        return None


async def optimize_database():
    """Run database optimization tasks"""
    try:
        async with get_db_session() as session:
            # Update table statistics
            await session.execute(text("ANALYZE;"))
            
            # Vacuum analyze on specific tables (non-blocking)
            critical_tables = [
                'stocks', 'price_history', 'technical_indicators',
                'recommendations', 'fundamentals'
            ]
            
            for table in critical_tables:
                await session.execute(text(f"VACUUM ANALYZE {table};"))
            
            logger.info("Database optimization completed")
            return True
            
    except Exception as e:
        logger.error(f"Error optimizing database: {e}")
        return False