"""
Async Database Utilities
Provides async database connections and session management for high-performance operations.
"""

from typing import AsyncGenerator, Optional, Any, Dict
import logging
import asyncio
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncEngine,
    create_async_engine,
    async_sessionmaker
)
from sqlalchemy.pool import NullPool, QueuePool
from sqlalchemy import text, select
from sqlalchemy.orm import selectinload, joinedload
import asyncpg

from backend.config.settings import settings
from backend.models.unified_models import Base

logger = logging.getLogger(__name__)


class AsyncDatabaseManager:
    """
    Manages async database connections with proper pooling and session lifecycle.
    """
    
    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize async database manager.
        
        Args:
            database_url: Async database URL (postgresql+asyncpg://...)
        """
        self.database_url = database_url or settings.database_url_async
        self._engine: Optional[AsyncEngine] = None
        self._sessionmaker: Optional[async_sessionmaker] = None
        
        # Performance metrics
        self._metrics = {
            'connections_created': 0,
            'connections_closed': 0,
            'queries_executed': 0,
            'errors': 0
        }
    
    async def initialize(self) -> None:
        """Initialize async engine and session maker."""
        try:
            # Determine pool class based on environment
            if settings.ENVIRONMENT == "production":
                poolclass = QueuePool
                pool_size = 50
                max_overflow = 100
            else:
                poolclass = QueuePool
                pool_size = 20
                max_overflow = 40
            
            # Create async engine with optimized settings
            self._engine = create_async_engine(
                self.database_url,
                poolclass=poolclass,
                pool_size=pool_size,
                max_overflow=max_overflow,
                pool_timeout=30,
                pool_recycle=1800,  # Recycle connections after 30 minutes
                pool_pre_ping=True,  # Verify connections before using
                echo=settings.DEBUG,
                future=True,
                query_cache_size=1200,  # Cache parsed SQL statements
                connect_args={
                    "server_settings": {
                        "application_name": "investment_analysis_app",
                        "jit": "off"  # Disable JIT for more predictable performance
                    },
                    "command_timeout": 60,
                    # Statement cache size of 100 enables prepared statement reuse for 10-15% faster repeated queries
                    # Using unique statement names per task prevents conflicts in connection pooling
                    "prepared_statement_cache_size": 100,
                    "prepared_statement_name_func": lambda: f"stmt_{id(asyncio.current_task())}"
                }
            )
            
            # Create session maker
            self._sessionmaker = async_sessionmaker(
                self._engine,
                class_=AsyncSession,
                expire_on_commit=False,  # Don't expire objects after commit
                autoflush=False,  # Don't auto-flush before queries
                autocommit=False
            )
            
            # Test connection
            async with self._engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
            
            logger.info("Async database manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize async database: {e}")
            raise
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get async database session with proper lifecycle management.
        
        Yields:
            AsyncSession: Database session
        """
        if not self._sessionmaker:
            await self.initialize()
        
        async with self._sessionmaker() as session:
            try:
                self._metrics['connections_created'] += 1
                yield session
                await session.commit()
            except Exception as e:
                await session.rollback()
                self._metrics['errors'] += 1
                logger.error(f"Database session error: {e}")
                raise
            finally:
                await session.close()
                self._metrics['connections_closed'] += 1
    
    async def execute_query(
        self,
        query: Any,
        params: Optional[Dict] = None
    ) -> Any:
        """
        Execute a query with automatic session management.
        
        Args:
            query: SQLAlchemy query or raw SQL
            params: Query parameters
        
        Returns:
            Query result
        """
        async with self.get_session() as session:
            self._metrics['queries_executed'] += 1
            
            if isinstance(query, str):
                result = await session.execute(text(query), params or {})
            else:
                result = await session.execute(query)
            
            return result
    
    async def bulk_insert(
        self,
        model: type,
        data: list[dict],
        on_conflict_update: bool = True
    ) -> int:
        """
        Perform bulk insert with conflict handling.
        
        Args:
            model: SQLAlchemy model class
            data: List of dictionaries to insert
            on_conflict_update: Update on conflict if True
        
        Returns:
            Number of rows affected
        """
        if not data:
            return 0
        
        async with self.get_session() as session:
            try:
                if on_conflict_update:
                    # Use PostgreSQL's ON CONFLICT for upsert
                    from sqlalchemy.dialects.postgresql import insert
                    
                    stmt = insert(model).values(data)
                    
                    # Get all columns except primary key for update
                    update_columns = {
                        c.name: c for c in model.__table__.columns
                        if not c.primary_key
                    }
                    
                    stmt = stmt.on_conflict_do_update(
                        index_elements=[c for c in model.__table__.primary_key],
                        set_=update_columns
                    )
                else:
                    # Simple bulk insert
                    stmt = insert(model).values(data)
                
                result = await session.execute(stmt)
                await session.commit()
                
                return result.rowcount
                
            except Exception as e:
                logger.error(f"Bulk insert error: {e}")
                await session.rollback()
                raise
    
    async def create_tables(self) -> None:
        """Create all database tables."""
        if not self._engine:
            await self.initialize()
        
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created successfully")
    
    async def drop_tables(self) -> None:
        """Drop all database tables (use with caution)."""
        if not self._engine:
            await self.initialize()
        
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            logger.warning("All database tables dropped")
    
    async def check_connection(self) -> bool:
        """Check if database is accessible."""
        try:
            async with self.get_session() as session:
                await session.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Database connection check failed: {e}")
            return False
    
    def get_metrics(self) -> Dict[str, int]:
        """Get database operation metrics."""
        return self._metrics.copy()
    
    async def close(self) -> None:
        """Close database connections."""
        if self._engine:
            await self._engine.dispose()
            logger.info("Async database connections closed")


# Global async database manager instance
async_db_manager = AsyncDatabaseManager()


# Dependency for FastAPI
async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency to get async database session for FastAPI.
    
    Usage:
        @app.get("/items")
        async def get_items(db: AsyncSession = Depends(get_async_db)):
            result = await db.execute(select(Item))
            return result.scalars().all()
    """
    async with async_db_manager.get_session() as session:
        yield session


# Context manager for standalone usage
@asynccontextmanager
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Get async session for standalone usage.
    
    Usage:
        async with get_async_session() as session:
            result = await session.execute(select(Stock))
            stocks = result.scalars().all()
    """
    async with async_db_manager.get_session() as session:
        yield session


# Utility functions for common operations
async def fetch_stocks_async(
    limit: Optional[int] = None,
    offset: int = 0,
    tier: Optional[str] = None
) -> list:
    """
    Fetch stocks asynchronously with filtering.
    
    Args:
        limit: Maximum number of stocks to fetch
        offset: Offset for pagination
        tier: Stock tier filter
    
    Returns:
        List of stock objects
    """
    from backend.models.unified_models import Stock
    
    async with get_async_session() as session:
        query = select(Stock)
        
        if tier:
            query = query.where(Stock.tier == tier)
        
        query = query.offset(offset)
        
        if limit:
            query = query.limit(limit)
        
        result = await session.execute(query)
        return result.scalars().all()


async def update_stock_metrics_async(
    symbol: str,
    metrics: Dict[str, Any]
) -> bool:
    """
    Update stock metrics asynchronously.
    
    Args:
        symbol: Stock symbol
        metrics: Dictionary of metrics to update
    
    Returns:
        True if successful
    """
    from backend.models.unified_models import Stock
    
    async with get_async_session() as session:
        try:
            # Fetch stock
            result = await session.execute(
                select(Stock).where(Stock.symbol == symbol)
            )
            stock = result.scalar_one_or_none()
            
            if not stock:
                logger.warning(f"Stock {symbol} not found")
                return False
            
            # Update metrics
            for key, value in metrics.items():
                if hasattr(stock, key):
                    setattr(stock, key, value)
            
            await session.commit()
            return True
            
        except Exception as e:
            logger.error(f"Error updating stock metrics for {symbol}: {e}")
            await session.rollback()
            return False


# Initialize on module import
def init_async_db():
    """Initialize async database synchronously (for startup scripts)."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(async_db_manager.initialize())
    loop.close()