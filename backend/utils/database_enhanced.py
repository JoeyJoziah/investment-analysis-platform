"""
Enhanced Database Connection with Optimized Pooling
Provides robust connection pooling, monitoring, and error handling.
"""

from typing import Generator, Optional, Dict, Any
from contextlib import contextmanager
import logging
import time
from datetime import datetime

from sqlalchemy import create_engine, text, event, pool
from sqlalchemy.orm import sessionmaker, Session, scoped_session
from sqlalchemy.pool import QueuePool, NullPool, StaticPool
from sqlalchemy.exc import SQLAlchemyError, OperationalError, DisconnectionError
from prometheus_client import Counter, Histogram, Gauge
import psutil

from backend.config.settings import settings
from backend.models.unified_models import Base

logger = logging.getLogger(__name__)

# Metrics for monitoring
db_connection_counter = Counter('db_connections_total', 'Total database connections', ['status'])
db_query_histogram = Histogram('db_query_duration_seconds', 'Database query duration', ['operation'])
db_pool_size_gauge = Gauge('db_pool_size', 'Current database pool size')
db_pool_checked_out_gauge = Gauge('db_pool_checked_out', 'Checked out connections')
db_pool_overflow_gauge = Gauge('db_pool_overflow', 'Overflow connections')


class DatabaseConfig:
    """Database configuration with environment-based settings."""
    
    def __init__(self):
        self.url = settings.DATABASE_URL
        self.environment = settings.ENVIRONMENT.lower()
        
        # Connection pool settings based on environment
        if self.environment == 'production':
            self.pool_size = 50
            self.max_overflow = 100
            self.pool_timeout = 30
            self.pool_recycle = 1800  # 30 minutes
            self.pool_pre_ping = True
            self.echo = False
        elif self.environment == 'development':
            self.pool_size = 10
            self.max_overflow = 20
            self.pool_timeout = 10
            self.pool_recycle = 3600  # 1 hour
            self.pool_pre_ping = True
            self.echo = settings.DEBUG
        else:  # test
            self.pool_size = 5
            self.max_overflow = 10
            self.pool_timeout = 5
            self.pool_recycle = 3600
            self.pool_pre_ping = False
            self.echo = False


class EnhancedDatabaseManager:
    """Enhanced database manager with monitoring and error recovery."""
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or DatabaseConfig()
        self._engine = None
        self._session_factory = None
        self._scoped_session = None
        self._initialize()
    
    def _initialize(self):
        """Initialize database engine and session factory."""
        try:
            # Create engine with optimized pooling
            self._engine = create_engine(
                self.config.url,
                poolclass=QueuePool,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle,
                pool_pre_ping=self.config.pool_pre_ping,
                echo=self.config.echo,
                # Additional optimizations
                connect_args={
                    "connect_timeout": 10,
                    "options": "-c statement_timeout=60000",  # 60 second statement timeout
                    "keepalives": 1,
                    "keepalives_idle": 30,
                    "keepalives_interval": 10,
                    "keepalives_count": 5,
                }
            )
            
            # Set up event listeners for monitoring
            self._setup_event_listeners()
            
            # Create session factory
            self._session_factory = sessionmaker(
                bind=self._engine,
                autocommit=False,
                autoflush=False,
                expire_on_commit=False  # Prevent lazy loading issues
            )
            
            # Create scoped session for thread safety
            self._scoped_session = scoped_session(self._session_factory)
            
            logger.info(f"Database engine initialized with pool_size={self.config.pool_size}")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def _setup_event_listeners(self):
        """Set up SQLAlchemy event listeners for monitoring."""
        
        @event.listens_for(self._engine, "connect")
        def receive_connect(dbapi_conn, connection_record):
            """Track new connections."""
            connection_record.info['connect_time'] = time.time()
            db_connection_counter.labels(status='created').inc()
            logger.debug("New database connection created")
        
        @event.listens_for(self._engine, "checkout")
        def receive_checkout(dbapi_conn, connection_record, connection_proxy):
            """Track connection checkouts."""
            db_pool_checked_out_gauge.inc()
            checkout_time = time.time()
            connection_record.info['checkout_time'] = checkout_time
        
        @event.listens_for(self._engine, "checkin")
        def receive_checkin(dbapi_conn, connection_record):
            """Track connection checkins."""
            db_pool_checked_out_gauge.dec()
            if 'checkout_time' in connection_record.info:
                duration = time.time() - connection_record.info['checkout_time']
                db_query_histogram.labels(operation='connection_hold').observe(duration)
        
        @event.listens_for(pool.Pool, "connect")
        def receive_pool_connect(dbapi_conn, connection_record):
            """Set up connection for optimal performance."""
            with dbapi_conn.cursor() as cursor:
                # PostgreSQL optimizations
                cursor.execute("SET jit = 'off'")  # Disable JIT for consistent performance
                cursor.execute("SET random_page_cost = 1.1")  # Optimize for SSD
                cursor.execute("SET effective_io_concurrency = 200")  # For SSD
    
    @property
    def engine(self):
        """Get database engine."""
        return self._engine
    
    @property
    def session_factory(self):
        """Get session factory."""
        return self._session_factory
    
    def get_session(self) -> Session:
        """Get a new database session."""
        session = self._session_factory()
        db_connection_counter.labels(status='session_created').inc()
        return session
    
    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """
        Provide a transactional scope for database operations.
        
        Usage:
            with db_manager.session_scope() as session:
                session.query(Model).all()
        """
        session = self.get_session()
        try:
            yield session
            session.commit()
            db_connection_counter.labels(status='commit').inc()
        except OperationalError as e:
            session.rollback()
            db_connection_counter.labels(status='rollback_operational').inc()
            logger.error(f"Database operational error: {e}")
            
            # Attempt reconnection for connection errors
            if "server closed the connection" in str(e):
                logger.info("Attempting to reconnect to database...")
                self._engine.dispose()
                self._initialize()
            raise
        except Exception as e:
            session.rollback()
            db_connection_counter.labels(status='rollback_error').inc()
            logger.error(f"Database error: {e}")
            raise
        finally:
            session.close()
            db_connection_counter.labels(status='session_closed').inc()
    
    def execute_with_retry(
        self, 
        func, 
        max_retries: int = 3, 
        retry_delay: float = 1.0
    ) -> Any:
        """
        Execute a database operation with retry logic.
        
        Args:
            func: Function that takes a session as argument
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        last_error = None
        
        for attempt in range(max_retries):
            try:
                with self.session_scope() as session:
                    return func(session)
            except (OperationalError, DisconnectionError) as e:
                last_error = e
                if attempt < max_retries - 1:
                    logger.warning(f"Database operation failed, retrying... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                    self._engine.dispose()  # Reset connection pool
                else:
                    logger.error(f"Database operation failed after {max_retries} attempts")
        
        raise last_error
    
    def get_pool_status(self) -> Dict:
        """Get current connection pool status."""
        pool = self._engine.pool
        return {
            'size': pool.size(),
            'checked_out': pool.checked_out(),
            'overflow': pool.overflow(),
            'total': pool.size() + pool.overflow(),
            'status': 'healthy' if pool.size() > 0 else 'unhealthy'
        }
    
    def check_connection(self) -> bool:
        """Check if database is accessible."""
        try:
            with self._engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                return result.scalar() == 1
        except Exception as e:
            logger.error(f"Database connection check failed: {e}")
            return False
    
    def optimize_for_bulk_operations(self, session: Session):
        """Optimize session for bulk operations."""
        # Disable autoflush for bulk operations
        session.autoflush = False
        
        # Execute PostgreSQL optimizations
        session.execute(text("SET synchronous_commit = OFF"))
        session.execute(text("SET work_mem = '256MB'"))
        session.execute(text("SET maintenance_work_mem = '512MB'"))
    
    def dispose(self):
        """Dispose of the connection pool."""
        if self._engine:
            self._engine.dispose()
            logger.info("Database connection pool disposed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.dispose()


# Global database manager instance
db_manager = EnhancedDatabaseManager()


# FastAPI dependency
def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency for database sessions.
    
    Usage:
        @app.get("/items")
        def get_items(db: Session = Depends(get_db)):
            return db.query(Item).all()
    """
    with db_manager.session_scope() as session:
        yield session


# Celery task helper
def get_db_session() -> Session:
    """Get database session for Celery tasks or background jobs."""
    return db_manager.get_session()


# Context manager for standalone scripts
@contextmanager
def db_session() -> Generator[Session, None, None]:
    """
    Context manager for database operations in standalone scripts.
    
    Usage:
        with db_session() as session:
            results = session.query(Model).all()
    """
    with db_manager.session_scope() as session:
        yield session


# Health check function
async def check_database_health() -> Dict:
    """
    Comprehensive database health check.
    
    Returns:
        Dict with health status and metrics
    """
    start_time = time.time()
    
    try:
        # Check basic connectivity
        is_connected = db_manager.check_connection()
        
        # Get pool status
        pool_status = db_manager.get_pool_status()
        
        # Check system resources
        memory_usage = psutil.virtual_memory().percent
        
        # Run diagnostic query
        with db_manager.session_scope() as session:
            result = session.execute(text("""
                SELECT 
                    pg_database_size(current_database()) as db_size,
                    count(*) as connection_count
                FROM pg_stat_activity
                WHERE datname = current_database()
            """)).first()
            
            db_size = result.db_size if result else 0
            active_connections = result.connection_count if result else 0
        
        response_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return {
            'status': 'healthy' if is_connected else 'unhealthy',
            'connected': is_connected,
            'response_time_ms': round(response_time, 2),
            'pool_status': pool_status,
            'active_connections': active_connections,
            'database_size_bytes': db_size,
            'memory_usage_percent': memory_usage,
            'timestamp': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {
            'status': 'unhealthy',
            'connected': False,
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }


# Initialize database tables
async def init_db():
    """Initialize database by creating all tables."""
    try:
        Base.metadata.create_all(bind=db_manager.engine)
        logger.info("Database tables created successfully")
        
        # Initialize TimescaleDB if available
        try:
            from backend.utils.db_timescale_init import initialize_timescaledb
            if initialize_timescaledb():
                logger.info("TimescaleDB initialized successfully")
        except ImportError:
            logger.info("TimescaleDB initialization module not found")
        
        return True
        
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        return False