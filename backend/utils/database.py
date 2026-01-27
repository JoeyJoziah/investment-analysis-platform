"""
Async Database connection utilities - Replacement for deprecated synchronous version

This module provides both async (preferred) and sync (legacy/Celery) database operations.

Migration Guide:
----------------
- get_db() -> get_async_db_session() or get_db_session()
- get_db_sync() -> Use async patterns where possible, keep for Celery tasks
- get_engine() -> db_manager._engine (async engine)
- get_connection_pool_status() -> get_async_connection_pool_status()
- log_connection_pool_metrics() -> log_async_connection_pool_metrics()

Example async usage:
    async with get_db_session() as session:
        result = await session.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()
"""

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from typing import Generator, Optional, Dict, Any
from contextlib import contextmanager
import logging
import warnings
import asyncio

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


def _emit_deprecation_warning(old_func: str, new_func: str) -> None:
    """Emit a deprecation warning for legacy sync functions."""
    warnings.warn(
        f"{old_func}() is deprecated and will be removed in a future version. "
        f"Use {new_func}() instead for async operations.",
        DeprecationWarning,
        stacklevel=3
    )

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
    This function is maintained for backward compatibility with FastAPI Depends()
    that haven't been migrated to async.

    Migration:
        # Old (sync):
        @app.get("/items/")
        def get_items(db: Session = Depends(get_db)):
            return db.query(Item).all()

        # New (async):
        @app.get("/items/")
        async def get_items(db: AsyncSession = Depends(get_async_db_session)):
            result = await db.execute(select(Item))
            return result.scalars().all()
    """
    _emit_deprecation_warning("get_db", "get_async_db_session")
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


@contextmanager
def get_db_sync() -> Generator[Session, None, None]:
    """
    DEPRECATED: Use async database operations where possible.

    Get synchronous database session for legacy Celery tasks.
    This function is maintained for Celery tasks which don't natively support async.

    Note: Now returns a context manager for proper resource cleanup.

    Usage:
        # Celery task (sync required)
        @celery.task
        def process_data():
            with get_db_sync() as db:
                items = db.query(Item).all()
                # Process items...

    For new async code, use:
        async with get_db_session() as session:
            result = await session.execute(select(Item))
            items = result.scalars().all()
    """
    _emit_deprecation_warning("get_db_sync", "get_db_session")
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_engine():
    """
    DEPRECATED: Access async engine through db_manager instead.

    Get legacy synchronous database engine.
    This is maintained for scripts and tools that require direct engine access.

    Migration:
        # Old (sync):
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        # New (async):
        async with db_manager._engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
    """
    _emit_deprecation_warning("get_engine", "db_manager._engine")
    return engine


def get_connection_pool_status() -> Dict[str, Any]:
    """
    DEPRECATED: Use get_async_connection_pool_status() instead.

    Get current connection pool status for the legacy synchronous engine.

    Returns:
        dict: Pool metrics including size, utilization, and connection counts.
    """
    _emit_deprecation_warning("get_connection_pool_status", "get_async_connection_pool_status")
    pool = engine.pool
    total = pool.size() + pool.overflow()
    return {
        "size": pool.size(),
        "checked_in": pool.checkedin(),
        "checked_out": pool.checkedout(),
        "overflow": pool.overflow(),
        "invalid": pool.invalid(),
        "total_connections": total,
        "utilization_percent": round((pool.checkedout() / total) * 100, 2) if total > 0 else 0,
        "engine_type": "sync_legacy"
    }


def log_connection_pool_metrics() -> Dict[str, Any]:
    """
    DEPRECATED: Use log_async_connection_pool_metrics() instead.

    Log connection pool metrics for monitoring - Legacy version.

    Returns:
        dict: The pool status that was logged.
    """
    _emit_deprecation_warning("log_connection_pool_metrics", "log_async_connection_pool_metrics")
    status = get_connection_pool_status()
    logger.info(f"Legacy sync connection pool status: {status}")

    # Alert if utilization is high
    if status['utilization_percent'] > 80:
        logger.warning(f"High legacy connection pool utilization: {status['utilization_percent']}%")

    return status


# =============================================================================
# ASYNC ALTERNATIVES (Preferred)
# =============================================================================


async def get_async_connection_pool_status() -> Dict[str, Any]:
    """
    Get current connection pool status for the async engine.

    This is the preferred method for checking pool health in async code.

    Returns:
        dict: Comprehensive pool metrics from the async database manager.

    Example:
        status = await get_async_connection_pool_status()
        if status['metrics']['connection_errors'] > 0:
            logger.warning("Connection errors detected")
    """
    return await db_manager.get_connection_pool_status()


async def log_async_connection_pool_metrics() -> Dict[str, Any]:
    """
    Log async connection pool metrics for monitoring.

    This is the preferred method for logging pool metrics in async code.

    Returns:
        dict: The pool status that was logged.

    Example:
        # In a periodic health check task
        status = await log_async_connection_pool_metrics()
        if status.get('connection_health', {}).get('status') != 'healthy':
            await send_alert("Database connection issues detected")
    """
    status = await db_manager.get_connection_pool_status()
    logger.info(f"Async connection pool status: {status}")

    # Check for potential issues
    metrics = status.get('metrics', {})
    if metrics.get('connection_errors', 0) > 0:
        logger.warning(f"Connection errors detected: {metrics['connection_errors']}")

    if metrics.get('deadlocks_detected', 0) > 0:
        logger.warning(f"Deadlocks detected: {metrics['deadlocks_detected']}")

    health = status.get('connection_health', {})
    if health.get('status') != 'healthy':
        logger.warning(f"Connection health status: {health.get('status')}")

    return status


async def get_async_engine():
    """
    Get the async database engine.

    This provides access to the async engine for advanced operations
    like running DDL or batch operations.

    Returns:
        AsyncEngine: The async SQLAlchemy engine.

    Example:
        engine = await get_async_engine()
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    """
    if not db_manager._initialized:
        await db_manager.initialize()
    return db_manager._engine


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


async def optimize_database() -> bool:
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


# =============================================================================
# ASYNC TRANSACTION UTILITIES
# =============================================================================


async def execute_in_transaction(operation, *args, **kwargs) -> Any:
    """
    Execute an async operation within a managed transaction.

    This provides automatic commit on success and rollback on failure,
    with retry support for transient errors like deadlocks.

    Args:
        operation: Async callable that takes a session as first argument
        *args, **kwargs: Additional arguments passed to the operation

    Returns:
        The result of the operation

    Example:
        async def transfer_funds(session, from_id, to_id, amount):
            from_account = await session.get(Account, from_id)
            to_account = await session.get(Account, to_id)
            from_account.balance -= amount
            to_account.balance += amount
            return {"from": from_id, "to": to_id, "amount": amount}

        result = await execute_in_transaction(transfer_funds, 1, 2, 100.00)
    """
    return await db_manager.execute_with_retry(
        _execute_operation_in_session,
        operation,
        *args,
        **kwargs
    )


async def _execute_operation_in_session(operation, *args, **kwargs) -> Any:
    """Helper to execute an operation within a session context."""
    async with get_db_session() as session:
        return await operation(session, *args, **kwargs)


async def bulk_upsert(
    model_class,
    data: list,
    conflict_strategy: str = "update",
    batch_size: int = 1000
) -> int:
    """
    Perform bulk insert/update with conflict handling.

    This is the preferred method for bulk data operations.

    Args:
        model_class: SQLAlchemy model class
        data: List of dictionaries to insert/update
        conflict_strategy: "ignore", "update", or "error"
        batch_size: Number of records per batch

    Returns:
        Number of records affected

    Example:
        affected = await bulk_upsert(
            Stock,
            [{"symbol": "AAPL", "name": "Apple Inc"}, ...],
            conflict_strategy="update"
        )
    """
    return await db_manager.bulk_insert_with_conflict_handling(
        model_class,
        data,
        conflict_strategy=conflict_strategy,
        batch_size=batch_size
    )


# =============================================================================
# CONNECTION CLEANUP UTILITIES
# =============================================================================


async def cleanup_all_connections() -> Dict[str, bool]:
    """
    Clean up all database connections (both async and legacy sync).

    This should be called during application shutdown.

    Returns:
        dict: Status of cleanup operations

    Example:
        # In FastAPI shutdown event
        @app.on_event("shutdown")
        async def shutdown():
            await cleanup_all_connections()
    """
    results = {"async": False, "sync": False}

    # Cleanup async connections
    try:
        await cleanup_database()
        results["async"] = True
        logger.info("Async database connections cleaned up")
    except Exception as e:
        logger.error(f"Error cleaning up async connections: {e}")

    # Cleanup legacy sync connections
    try:
        engine.dispose()
        results["sync"] = True
        logger.info("Legacy sync database connections cleaned up")
    except Exception as e:
        logger.error(f"Error cleaning up sync connections: {e}")

    return results


def cleanup_sync_connections() -> bool:
    """
    Clean up legacy synchronous connections only.

    This can be called from sync contexts like Celery shutdown.

    Returns:
        bool: True if cleanup was successful
    """
    try:
        engine.dispose()
        logger.info("Legacy sync database connections cleaned up")
        return True
    except Exception as e:
        logger.error(f"Error cleaning up sync connections: {e}")
        return False


# =============================================================================
# HEALTH CHECK UTILITIES
# =============================================================================


async def get_full_health_status() -> Dict[str, Any]:
    """
    Get comprehensive health status of all database connections.

    Returns:
        dict: Combined health status of async and sync engines

    Example:
        health = await get_full_health_status()
        if health['overall_status'] != 'healthy':
            await send_alert(f"Database issues: {health}")
    """
    async_health = await db_manager.health_check()

    # Check sync engine health
    sync_healthy = False
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            sync_healthy = True
    except Exception as e:
        logger.warning(f"Sync engine health check failed: {e}")

    sync_pool_status = get_connection_pool_status()

    overall_healthy = (
        async_health.get("status") == "healthy" and
        sync_healthy
    )

    return {
        "overall_status": "healthy" if overall_healthy else "degraded",
        "async_engine": async_health,
        "sync_engine": {
            "status": "healthy" if sync_healthy else "unhealthy",
            "pool_status": sync_pool_status
        }
    }


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Async (preferred)
    "get_db_session",
    "get_async_db_session",
    "get_async_connection_pool_status",
    "log_async_connection_pool_metrics",
    "get_async_engine",
    "check_database_connection",
    "monitor_connection_pool",
    "get_database_stats",
    "optimize_database",
    "execute_in_transaction",
    "bulk_upsert",
    "cleanup_all_connections",
    "get_full_health_status",
    # Initialization
    "init_db",
    "close_db",
    "initialize_database",
    "cleanup_database",
    # Database manager
    "db_manager",
    # Legacy sync (deprecated)
    "get_db",
    "get_db_sync",
    "get_engine",
    "get_connection_pool_status",
    "log_connection_pool_metrics",
    "cleanup_sync_connections",
    "engine",
    "SessionLocal",
]