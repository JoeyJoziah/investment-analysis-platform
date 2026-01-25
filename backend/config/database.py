"""
Async Database Configuration
Comprehensive async database setup with connection pooling, transaction management, and monitoring.
"""

import os
import asyncio
import logging
from typing import Optional, AsyncGenerator, Dict, Any, Union
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine
)
from sqlalchemy.pool import NullPool, AsyncAdaptedQueuePool
from sqlalchemy import text, event
from sqlalchemy.exc import SQLAlchemyError, DisconnectionError, TimeoutError
import asyncpg

from backend.config.settings import settings

logger = logging.getLogger(__name__)


class TransactionIsolationLevel(Enum):
    """Transaction isolation levels for PostgreSQL"""
    READ_UNCOMMITTED = "READ UNCOMMITTED"
    READ_COMMITTED = "READ COMMITTED"
    REPEATABLE_READ = "REPEATABLE READ"
    SERIALIZABLE = "SERIALIZABLE"


@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    url: str
    pool_size: int = 20
    max_overflow: int = 40
    pool_timeout: int = 30
    pool_recycle: int = 1800  # 30 minutes
    pool_pre_ping: bool = True
    echo: bool = False
    statement_cache_size: int = 1000
    prepared_statement_cache_size: int = 100
    isolation_level: TransactionIsolationLevel = TransactionIsolationLevel.READ_COMMITTED


class AsyncDatabaseManager:
    """
    Comprehensive async database manager with advanced features:
    - Connection pooling with health monitoring
    - Transaction management with isolation levels
    - Deadlock detection and retry mechanisms
    - Performance monitoring and metrics
    - Concurrent processing safety
    """
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        """
        Initialize the async database manager.
        
        Args:
            config: Database configuration. If None, uses default settings.
        """
        self.config = config or self._create_default_config()
        self._engine: Optional[AsyncEngine] = None
        self._sessionmaker: Optional[async_sessionmaker] = None
        self._initialized = False
        
        # Monitoring metrics
        self._metrics = {
            'total_connections': 0,
            'active_connections': 0,
            'connection_errors': 0,
            'transaction_rollbacks': 0,
            'deadlocks_detected': 0,
            'retry_attempts': 0,
            'slow_queries': 0
        }
        
        # Connection health monitoring
        self._connection_health = {
            'last_check': None,
            'consecutive_failures': 0,
            'status': 'unknown'
        }
    
    def _create_default_config(self) -> DatabaseConfig:
        """Create default database configuration based on environment."""
        # Convert PostgreSQL URL to async format
        database_url = settings.DATABASE_URL
        if database_url.startswith('postgresql://'):
            database_url = database_url.replace('postgresql://', 'postgresql+asyncpg://')
        elif database_url.startswith('postgres://'):
            database_url = database_url.replace('postgres://', 'postgresql+asyncpg://')
        
        # Environment-specific settings
        if settings.ENVIRONMENT == "production":
            pool_size = 50
            max_overflow = 100
        elif settings.ENVIRONMENT == "testing":
            pool_size = 5
            max_overflow = 10
        else:
            pool_size = 20
            max_overflow = 40
        
        return DatabaseConfig(
            url=database_url,
            pool_size=pool_size,
            max_overflow=max_overflow,
            echo=settings.DEBUG and settings.ENVIRONMENT != "production"
        )
    
    async def initialize(self) -> None:
        """Initialize the async database engine and session factory."""
        if self._initialized:
            logger.warning("Database manager already initialized")
            return
        
        try:
            # Create async engine with optimized settings
            connect_args = {
                "server_settings": {
                    "application_name": "investment_analysis_app",
                    "jit": "off",  # Disable JIT for predictable performance
                },
                "command_timeout": 60,
                "statement_cache_size": 0,  # Disable prepared statement caching for compatibility
            }
            
            # Choose pool class based on environment
            if settings.ENVIRONMENT == "testing":
                poolclass = NullPool
            else:
                poolclass = AsyncAdaptedQueuePool
            
            self._engine = create_async_engine(
                self.config.url,
                poolclass=poolclass,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle,
                pool_pre_ping=self.config.pool_pre_ping,
                echo=self.config.echo,
                future=True,
                query_cache_size=self.config.statement_cache_size,
                connect_args=connect_args,
                isolation_level=self.config.isolation_level.value
            )
            
            # Set up event listeners for monitoring
            self._setup_event_listeners()
            
            # Create async session factory
            self._sessionmaker = async_sessionmaker(
                self._engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=False,
                autocommit=False
            )
            
            # Test connection
            await self._test_connection()
            
            self._initialized = True
            logger.info("Async database manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize async database: {e}")
            await self.cleanup()
            raise
    
    def _prepared_statement_name_func(self) -> str:
        """Generate unique prepared statement names to prevent conflicts."""
        task = asyncio.current_task()
        task_id = id(task) if task else 0
        return f"stmt_{task_id}_{asyncio.get_event_loop().time()}"
    
    def _setup_event_listeners(self) -> None:
        """Set up SQLAlchemy event listeners for monitoring."""
        if not self._engine:
            return
        
        @event.listens_for(self._engine.sync_engine, "connect")
        def on_connect(dbapi_conn, connection_record):
            """Handle new database connections."""
            self._metrics['total_connections'] += 1
            logger.debug("New database connection established")
        
        @event.listens_for(self._engine.sync_engine, "checkout")
        def on_checkout(dbapi_conn, connection_record, connection_proxy):
            """Handle connection checkout from pool."""
            self._metrics['active_connections'] += 1
        
        @event.listens_for(self._engine.sync_engine, "checkin")
        def on_checkin(dbapi_conn, connection_record):
            """Handle connection checkin to pool."""
            self._metrics['active_connections'] -= 1
        
        @event.listens_for(self._engine.sync_engine, "invalidate")
        def on_invalidate(dbapi_conn, connection_record, exception):
            """Handle connection invalidation."""
            self._metrics['connection_errors'] += 1
            logger.warning(f"Database connection invalidated: {exception}")
    
    async def _test_connection(self) -> None:
        """Test database connection health."""
        try:
            async with self._engine.begin() as conn:
                result = await conn.execute(text("SELECT 1"))
                result.fetchone()  # No await needed - result is already resolved
            
            self._connection_health['status'] = 'healthy'
            self._connection_health['consecutive_failures'] = 0
            logger.debug("Database connection test successful")
            
        except Exception as e:
            self._connection_health['status'] = 'unhealthy'
            self._connection_health['consecutive_failures'] += 1
            logger.error(f"Database connection test failed: {e}")
            raise
    
    @asynccontextmanager
    async def get_session(
        self,
        isolation_level: Optional[TransactionIsolationLevel] = None,
        readonly: bool = False
    ) -> AsyncGenerator[AsyncSession, None]:
        """
        Get async database session with comprehensive error handling.
        
        Args:
            isolation_level: Transaction isolation level override
            readonly: Whether this is a read-only session
        
        Yields:
            AsyncSession: Database session with automatic cleanup
        """
        if not self._initialized:
            await self.initialize()
        
        session = self._sessionmaker()
        
        try:
            # Set isolation level if specified
            if isolation_level:
                await session.execute(
                    text(f"SET TRANSACTION ISOLATION LEVEL {isolation_level.value}")
                )
            
            # Set read-only mode if requested
            if readonly:
                await session.execute(text("SET TRANSACTION READ ONLY"))
            
            yield session
            await session.commit()
            
        except asyncpg.exceptions.SerializationError as e:
            # Handle serialization failures (deadlocks)
            await session.rollback()
            self._metrics['deadlocks_detected'] += 1
            logger.warning(f"Serialization failure detected: {e}")
            raise

        except asyncpg.exceptions.DeadlockDetectedError as e:
            # Handle explicit deadlocks
            await session.rollback()
            self._metrics['deadlocks_detected'] += 1
            logger.warning(f"Deadlock detected: {e}")
            raise

        except (DisconnectionError, TimeoutError) as e:
            # Handle connection issues
            await session.rollback()
            self._metrics['connection_errors'] += 1
            logger.error(f"Connection error in session: {e}")
            raise

        except Exception as e:
            # Handle all other errors
            await session.rollback()
            self._metrics['transaction_rollbacks'] += 1
            logger.error(f"Session error: {e}")
            raise
            
        finally:
            await session.close()
    
    async def execute_with_retry(
        self,
        operation,
        *args,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        **kwargs
    ) -> Any:
        """
        Execute database operation with retry logic for transient failures.
        
        Args:
            operation: Async function to execute
            max_retries: Maximum number of retry attempts
            backoff_factor: Exponential backoff factor
            *args, **kwargs: Arguments to pass to operation
        
        Returns:
            Result of the operation
        """
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                result = await operation(*args, **kwargs)
                if attempt > 0:
                    logger.info(f"Operation succeeded after {attempt} retries")
                return result
                
            except (
                asyncpg.exceptions.SerializationError,
                asyncpg.exceptions.DeadlockDetectedError,
                DisconnectionError,
                TimeoutError
            ) as e:
                last_exception = e
                self._metrics['retry_attempts'] += 1
                
                if attempt < max_retries:
                    wait_time = backoff_factor * (2 ** attempt)
                    logger.warning(
                        f"Operation failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                        f"Retrying in {wait_time:.2f} seconds..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Operation failed after {max_retries + 1} attempts")
                    break
            
            except Exception as e:
                # Non-retryable error
                logger.error(f"Non-retryable error in operation: {e}")
                raise
        
        # All retries exhausted
        raise last_exception
    
    async def bulk_insert_with_conflict_handling(
        self,
        model_class,
        data: list[dict],
        conflict_strategy: str = "update",
        batch_size: int = 1000
    ) -> int:
        """
        Perform bulk insert with comprehensive conflict handling.
        
        Args:
            model_class: SQLAlchemy model class
            data: List of dictionaries to insert
            conflict_strategy: "ignore", "update", or "error"
            batch_size: Number of records per batch
        
        Returns:
            Number of records affected
        """
        if not data:
            return 0
        
        total_affected = 0
        
        async def _bulk_insert_batch(batch_data: list[dict]) -> int:
            async with self.get_session() as session:
                from sqlalchemy.dialects.postgresql import insert
                
                stmt = insert(model_class).values(batch_data)
                
                if conflict_strategy == "ignore":
                    stmt = stmt.on_conflict_do_nothing()
                elif conflict_strategy == "update":
                    # Update all non-primary key columns
                    update_columns = {
                        c.name: stmt.excluded[c.name]
                        for c in model_class.__table__.columns
                        if not c.primary_key
                    }
                    if update_columns:
                        stmt = stmt.on_conflict_do_update(
                            index_elements=[c for c in model_class.__table__.primary_key],
                            set_=update_columns
                        )
                elif conflict_strategy != "error":
                    raise ValueError(f"Invalid conflict strategy: {conflict_strategy}")
                
                result = await session.execute(stmt)
                return result.rowcount
        
        # Process data in batches
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            batch_affected = await self.execute_with_retry(_bulk_insert_batch, batch)
            total_affected += batch_affected
        
        logger.info(f"Bulk insert completed: {total_affected} records affected")
        return total_affected
    
    async def get_connection_pool_status(self) -> Dict[str, Any]:
        """Get detailed connection pool status."""
        if not self._engine:
            return {"status": "not_initialized"}
        
        pool = self._engine.pool
        
        return {
            "pool_size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "invalid": pool.invalid(),
            "metrics": self._metrics.copy(),
            "connection_health": self._connection_health.copy()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive database health check."""
        try:
            start_time = asyncio.get_event_loop().time()
            
            async with self.get_session() as session:
                # Test basic connectivity
                await session.execute(text("SELECT 1"))
                
                # Test table access
                await session.execute(text("SELECT COUNT(*) FROM information_schema.tables"))
                
                # Test write capability (if not readonly)
                await session.execute(text("SELECT pg_is_in_recovery()"))
            
            response_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            pool_status = await self.get_connection_pool_status()
            
            return {
                "status": "healthy",
                "response_time_ms": round(response_time, 2),
                "pool_status": pool_status,
                "database_version": await self._get_database_version()
            }
            
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "pool_status": await self.get_connection_pool_status()
            }
    
    async def _get_database_version(self) -> str:
        """Get PostgreSQL version."""
        try:
            async with self.get_session() as session:
                result = await session.execute(text("SELECT version()"))
                version = await result.scalar()
                return version
        except Exception:
            return "unknown"
    
    async def cleanup(self) -> None:
        """Clean up database connections and resources."""
        try:
            if self._engine:
                await self._engine.dispose()
                logger.info("Database engine disposed")
            
            self._engine = None
            self._sessionmaker = None
            self._initialized = False
            
        except Exception as e:
            logger.error(f"Error during database cleanup: {e}")


# Global instance
db_manager = AsyncDatabaseManager()


# FastAPI dependency
async def get_async_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency for getting async database session.
    
    Usage:
        @app.get("/items/")
        async def get_items(db: AsyncSession = Depends(get_async_db_session)):
            result = await db.execute(select(Item))
            return result.scalars().all()
    """
    async with db_manager.get_session() as session:
        yield session


# Context manager for standalone usage
@asynccontextmanager
async def get_db_session(
    isolation_level: Optional[TransactionIsolationLevel] = None,
    readonly: bool = False
) -> AsyncGenerator[AsyncSession, None]:
    """
    Context manager for getting async database session outside of FastAPI.
    
    Args:
        isolation_level: Transaction isolation level
        readonly: Whether this is a read-only session
    
    Usage:
        async with get_db_session() as session:
            result = await session.execute(select(Stock))
            stocks = result.scalars().all()
    """
    async with db_manager.get_session(isolation_level=isolation_level, readonly=readonly) as session:
        yield session


# Utility functions
async def initialize_database() -> None:
    """Initialize the database system."""
    await db_manager.initialize()


async def cleanup_database() -> None:
    """Clean up database resources."""
    await db_manager.cleanup()


async def get_database_health() -> Dict[str, Any]:
    """Get database health status."""
    return await db_manager.health_check()