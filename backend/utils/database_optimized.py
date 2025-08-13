"""
Optimized Database Configuration for Budget Constraints
Implements connection pooling with proper sizing for $50/month infrastructure
"""

import os
import logging
from typing import Optional, Dict, Any
from contextlib import contextmanager
from sqlalchemy import create_engine, event, pool
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool, QueuePool
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


class OptimizedDatabaseConfig:
    """
    Database configuration optimized for budget constraints.
    
    Key optimizations:
    - Reduced connection pool sizes
    - Aggressive connection recycling
    - Connection health checks
    - Graceful degradation under load
    """
    
    def __init__(self):
        self.environment = os.getenv('ENVIRONMENT', 'production')
        self.database_url = os.getenv('DATABASE_URL')
        
        # Connection pool settings based on environment and budget
        if self.environment == 'production':
            # Production: Conservative settings for budget
            self.pool_size = int(os.getenv('DB_POOL_SIZE', '8'))
            self.max_overflow = int(os.getenv('DB_MAX_OVERFLOW', '4'))
            self.pool_timeout = 30
            self.pool_recycle = 1800  # 30 minutes
            self.pool_pre_ping = True
            self.echo = False
        elif self.environment == 'development':
            # Development: Minimal resources
            self.pool_size = int(os.getenv('DB_POOL_SIZE', '5'))
            self.max_overflow = int(os.getenv('DB_MAX_OVERFLOW', '2'))
            self.pool_timeout = 20
            self.pool_recycle = 3600  # 1 hour
            self.pool_pre_ping = True
            self.echo = True
        else:  # test
            # Test: Minimal pooling
            self.pool_size = 2
            self.max_overflow = 1
            self.pool_timeout = 10
            self.pool_recycle = 3600
            self.pool_pre_ping = False
            self.echo = False
    
    def calculate_optimal_pool_size(self, max_connections: int = 50) -> Dict[str, int]:
        """
        Calculate optimal pool size based on available connections.
        
        PostgreSQL max_connections = 50 (budget constraint)
        Distribution:
        - Backend API: 2 replicas * 8 connections = 16
        - Celery Worker: 1 replica * 5 connections = 5
        - Airflow: 5 connections
        - Admin/Maintenance: 5 connections
        - Buffer: 19 connections
        
        Args:
            max_connections: Total PostgreSQL connections available
            
        Returns:
            Dict with recommended pool sizes per service
        """
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        
        # Formula: connections = ((core_count * 2) + effective_spindle_count)
        # For cloud/SSD, effective_spindle_count = 1
        optimal_per_instance = min((cpu_count * 2) + 1, 10)
        
        return {
            'backend_api': min(optimal_per_instance, 8),
            'celery_worker': min(optimal_per_instance // 2, 5),
            'airflow': min(optimal_per_instance // 2, 5),
            'total_required': 26,  # Sum of above * replicas
            'available': max_connections,
            'buffer': max_connections - 26
        }
    
    def create_engine(self, **kwargs) -> Engine:
        """
        Create SQLAlchemy engine with optimized settings.
        
        Returns:
            Configured SQLAlchemy engine
        """
        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable not set")
        
        # Merge provided kwargs with defaults
        engine_args = {
            'pool_size': self.pool_size,
            'max_overflow': self.max_overflow,
            'pool_timeout': self.pool_timeout,
            'pool_recycle': self.pool_recycle,
            'pool_pre_ping': self.pool_pre_ping,
            'echo': self.echo,
            'echo_pool': self.echo,
            'pool_class': QueuePool,
            'connect_args': {
                'connect_timeout': 10,
                'application_name': f'investment_{self.environment}',
                'options': '-c statement_timeout=30000'  # 30 second statement timeout
            }
        }
        engine_args.update(kwargs)
        
        engine = create_engine(self.database_url, **engine_args)
        
        # Add connection pool logging
        @event.listens_for(engine, "connect")
        def receive_connect(dbapi_conn, connection_record):
            connection_record.info['connect_time'] = os.times()[4]
            logger.debug(f"Connection checked out from pool: {id(dbapi_conn)}")
        
        @event.listens_for(engine, "checkout")
        def receive_checkout(dbapi_conn, connection_record, connection_proxy):
            # Calculate connection age
            connect_time = connection_record.info.get('connect_time', 0)
            age = os.times()[4] - connect_time
            if age > self.pool_recycle:
                logger.info(f"Recycling old connection: age={age:.1f}s")
                connection_proxy.invalidate()
        
        # Add pool size monitoring
        @event.listens_for(engine, "close")
        def receive_close(dbapi_conn, connection_record):
            logger.debug(f"Connection returned to pool: {id(dbapi_conn)}")
        
        logger.info(
            f"Database engine created: pool_size={self.pool_size}, "
            f"max_overflow={self.max_overflow}, environment={self.environment}"
        )
        
        return engine


class DatabaseConnection:
    """
    Singleton database connection manager with optimized pooling.
    """
    
    _instance = None
    _engine = None
    _sessionmaker = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._engine is None:
            config = OptimizedDatabaseConfig()
            self._engine = config.create_engine()
            self._sessionmaker = sessionmaker(
                bind=self._engine,
                expire_on_commit=False,
                autoflush=False
            )
            
            # Validate connection
            try:
                with self._engine.connect() as conn:
                    conn.execute("SELECT 1")
                logger.info("Database connection validated successfully")
            except Exception as e:
                logger.error(f"Database connection validation failed: {e}")
                raise
    
    @property
    def engine(self) -> Engine:
        """Get the database engine."""
        return self._engine
    
    @property
    def session_factory(self) -> sessionmaker:
        """Get the session factory."""
        return self._sessionmaker
    
    @contextmanager
    def get_session(self) -> Session:
        """
        Get a database session with automatic cleanup.
        
        Usage:
            with db.get_session() as session:
                # Use session
                pass
        """
        session = self._sessionmaker()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def get_pool_status(self) -> Dict[str, Any]:
        """
        Get current connection pool status.
        
        Returns:
            Dictionary with pool metrics
        """
        pool = self._engine.pool
        return {
            'size': pool.size(),
            'checked_in': pool.checkedin(),
            'checked_out': pool.checkedout(),
            'overflow': pool.overflow(),
            'total': pool.size() + pool.overflow(),
            'max_overflow': pool._max_overflow,
            'timeout': pool._timeout
        }
    
    def close(self):
        """Close all database connections."""
        if self._engine:
            self._engine.dispose()
            logger.info("Database connections closed")
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()


# Global database instance
db = DatabaseConnection()


def get_db_session() -> Session:
    """
    Dependency for FastAPI to get database session.
    
    Usage:
        @app.get("/")
        def read_root(db: Session = Depends(get_db_session)):
            pass
    """
    with db.get_session() as session:
        yield session


def init_database():
    """Initialize database connection on startup."""
    try:
        db_instance = DatabaseConnection()
        status = db_instance.get_pool_status()
        logger.info(f"Database initialized with pool status: {status}")
        return db_instance
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


def close_database():
    """Close database connections on shutdown."""
    if db._engine:
        db.close()


# Health check query
def check_database_health() -> Dict[str, Any]:
    """
    Check database health and connection pool status.
    
    Returns:
        Health status dictionary
    """
    try:
        # Test query
        with db.get_session() as session:
            result = session.execute("SELECT version(), current_database(), pg_size_pretty(pg_database_size(current_database()))")
            version, database, size = result.fetchone()
        
        # Get pool status
        pool_status = db.get_pool_status()
        
        # Check for connection starvation
        utilization = pool_status['checked_out'] / (pool_status['size'] + pool_status['max_overflow'])
        health = 'healthy' if utilization < 0.8 else 'degraded' if utilization < 0.95 else 'critical'
        
        return {
            'status': health,
            'database': database,
            'version': version,
            'size': size,
            'pool': pool_status,
            'utilization_percent': round(utilization * 100, 1)
        }
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {
            'status': 'unhealthy',
            'error': str(e)
        }


if __name__ == "__main__":
    # Test configuration
    config = OptimizedDatabaseConfig()
    optimal = config.calculate_optimal_pool_size()
    print(f"Optimal pool sizes: {optimal}")
    
    # Test connection
    db_instance = init_database()
    health = check_database_health()
    print(f"Database health: {health}")
    
    close_database()