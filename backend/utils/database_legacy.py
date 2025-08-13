"""Database connection utilities"""

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from typing import Generator
import logging

from backend.config.settings import settings
from backend.models.unified_models import Base

logger = logging.getLogger(__name__)

# Create engine with optimized connection pooling
engine = create_engine(
    settings.DATABASE_URL,
    poolclass=QueuePool,
    pool_size=15,  # Increased from default 8 to 15
    max_overflow=30,  # Reduced overflow to maintain better control
    pool_timeout=60,  # Increased timeout for high-load scenarios
    pool_recycle=3600,  # Recycle connections after 1 hour
    pool_pre_ping=True,  # Validate connections before use
    pool_reset_on_return='commit',  # Reset connections on return
    echo=settings.DEBUG,  # Log SQL queries in debug mode
    # Performance optimizations
    connect_args={
        "options": "-c statement_timeout=30000 -c idle_in_transaction_session_timeout=60000"
    }
)

# Create session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)


def get_db() -> Generator[Session, None, None]:
    """
    Dependency to get database session.
    Usage in FastAPI endpoints:
    
    @app.get("/items")
    def get_items(db: Session = Depends(get_db)):
        return db.query(Item).all()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def init_db():
    """Initialize database by creating all tables"""
    try:
        # Create all tables
        Base.metadata.create_all(bind=engine)
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
        engine.dispose()
        logger.info("Database connections closed")
    except Exception as e:
        logger.error(f"Error closing database: {e}")


def get_db_sync() -> Session:
    """Get synchronous database session for Celery tasks"""
    db = SessionLocal()
    return db


def get_engine():
    """Get database engine"""
    return engine


def get_connection_pool_status() -> dict:
    """Get current connection pool status"""
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
    """Log connection pool metrics for monitoring"""
    status = get_connection_pool_status()
    logger.info(f"Connection pool status: {status}")
    
    # Alert if utilization is high
    if status['utilization_percent'] > 80:
        logger.warning(f"High connection pool utilization: {status['utilization_percent']}%")
    
    return status


async def check_database_connection() -> bool:
    """Check if database is accessible and log pool status"""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        
        # Log pool status during health check
        log_connection_pool_metrics()
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        # Log pool status on failure
        status = get_connection_pool_status()
        logger.error(f"Pool status during failure: {status}")
        return False


async def monitor_connection_pool():
    """Continuous monitoring of connection pool health"""
    import asyncio
    while True:
        try:
            status = log_connection_pool_metrics()
            
            # Check for potential issues
            if status['invalid'] > 0:
                logger.warning(f"Invalid connections detected: {status['invalid']}")
            
            if status['overflow'] > status['size']:
                logger.warning(f"High overflow usage: {status['overflow']} overflow connections active")
                
        except Exception as e:
            logger.error(f"Error monitoring connection pool: {e}")
        
        # Monitor every 30 seconds
        await asyncio.sleep(30)