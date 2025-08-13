"""
Database Read Replica Management
Provides intelligent routing of read queries to replicas and writes to primary.
"""

import asyncio
import logging
import random
from typing import Optional, List, Dict, Any, AsyncGenerator
from contextlib import asynccontextmanager
from enum import Enum

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncEngine,
    create_async_engine,
    async_sessionmaker
)
from sqlalchemy.pool import QueuePool
from sqlalchemy import text, select
from sqlalchemy.exc import OperationalError, DBAPIError

from backend.config.settings import settings

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of database queries."""
    READ = "read"
    WRITE = "write"
    READ_WRITE = "read_write"


class ReplicaHealth(Enum):
    """Health status of a replica."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class DatabaseReplica:
    """Represents a database replica with health monitoring."""
    
    def __init__(
        self,
        name: str,
        url: str,
        is_primary: bool = False,
        weight: int = 1
    ):
        """
        Initialize database replica.
        
        Args:
            name: Replica name
            url: Database connection URL
            is_primary: True if this is the primary database
            weight: Weight for load balancing (higher = more traffic)
        """
        self.name = name
        self.url = url
        self.is_primary = is_primary
        self.weight = weight
        self.health = ReplicaHealth.HEALTHY
        
        self._engine: Optional[AsyncEngine] = None
        self._sessionmaker: Optional[async_sessionmaker] = None
        
        # Health metrics
        self._metrics = {
            'queries_executed': 0,
            'errors': 0,
            'avg_latency_ms': 0,
            'health_checks_passed': 0,
            'health_checks_failed': 0
        }
    
    async def initialize(self) -> None:
        """Initialize database connection."""
        try:
            self._engine = create_async_engine(
                self.url,
                poolclass=QueuePool,
                pool_size=20 if self.is_primary else 15,
                max_overflow=40 if self.is_primary else 30,
                pool_timeout=30,
                pool_recycle=1800,
                pool_pre_ping=True,
                echo=False,
                future=True
            )
            
            self._sessionmaker = async_sessionmaker(
                self._engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=False,
                autocommit=False
            )
            
            # Test connection
            async with self._engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
            
            self.health = ReplicaHealth.HEALTHY
            logger.info(f"Initialized replica {self.name} (primary={self.is_primary})")
            
        except Exception as e:
            self.health = ReplicaHealth.UNHEALTHY
            logger.error(f"Failed to initialize replica {self.name}: {e}")
            raise
    
    async def health_check(self) -> bool:
        """
        Perform health check on replica.
        
        Returns:
            True if healthy
        """
        if not self._engine:
            self.health = ReplicaHealth.UNHEALTHY
            return False
        
        try:
            async with self._engine.begin() as conn:
                # Check basic connectivity
                result = await conn.execute(text("SELECT 1"))
                
                # Check replication lag for replicas
                if not self.is_primary:
                    lag_query = text("""
                        SELECT 
                            EXTRACT(EPOCH FROM (now() - pg_last_xact_replay_timestamp()))::int 
                        AS replication_lag_seconds
                    """)
                    lag_result = await conn.execute(lag_query)
                    lag_seconds = lag_result.scalar()
                    
                    # Consider unhealthy if lag > 30 seconds
                    if lag_seconds and lag_seconds > 30:
                        self.health = ReplicaHealth.DEGRADED
                        logger.warning(f"Replica {self.name} has high lag: {lag_seconds}s")
                    else:
                        self.health = ReplicaHealth.HEALTHY
                else:
                    self.health = ReplicaHealth.HEALTHY
            
            self._metrics['health_checks_passed'] += 1
            return True
            
        except Exception as e:
            self.health = ReplicaHealth.UNHEALTHY
            self._metrics['health_checks_failed'] += 1
            logger.error(f"Health check failed for {self.name}: {e}")
            return False
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get database session."""
        if not self._sessionmaker:
            await self.initialize()
        
        async with self._sessionmaker() as session:
            try:
                self._metrics['queries_executed'] += 1
                yield session
                await session.commit()
            except Exception as e:
                await session.rollback()
                self._metrics['errors'] += 1
                raise
            finally:
                await session.close()
    
    async def close(self) -> None:
        """Close database connections."""
        if self._engine:
            await self._engine.dispose()
            logger.info(f"Closed connections for replica {self.name}")


class ReadReplicaManager:
    """
    Manages read replicas with intelligent query routing.
    """
    
    def __init__(self):
        """Initialize read replica manager."""
        self.primary: Optional[DatabaseReplica] = None
        self.replicas: List[DatabaseReplica] = []
        self._initialized = False
        
        # Load balancing state
        self._current_replica_index = 0
        self._replica_weights: List[int] = []
        
        # Health monitoring
        self._health_check_interval = 30  # seconds
        self._health_check_task: Optional[asyncio.Task] = None
    
    async def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize replica manager with configuration.
        
        Args:
            config: Optional configuration override
        """
        if self._initialized:
            return
        
        try:
            # Default configuration
            if not config:
                config = self._get_default_config()
            
            # Initialize primary
            primary_config = config['primary']
            self.primary = DatabaseReplica(
                name=primary_config['name'],
                url=primary_config['url'],
                is_primary=True
            )
            await self.primary.initialize()
            
            # Initialize replicas
            for replica_config in config.get('replicas', []):
                replica = DatabaseReplica(
                    name=replica_config['name'],
                    url=replica_config['url'],
                    is_primary=False,
                    weight=replica_config.get('weight', 1)
                )
                
                try:
                    await replica.initialize()
                    self.replicas.append(replica)
                    self._replica_weights.append(replica.weight)
                except Exception as e:
                    logger.error(f"Failed to initialize replica {replica.name}: {e}")
            
            # Start health monitoring
            self._health_check_task = asyncio.create_task(self._monitor_health())
            
            self._initialized = True
            logger.info(f"Read replica manager initialized with {len(self.replicas)} replicas")
            
        except Exception as e:
            logger.error(f"Failed to initialize read replica manager: {e}")
            raise
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default replica configuration."""
        base_url = settings.DATABASE_URL
        
        # Parse base URL to create replica URLs
        # Format: postgresql://user:pass@host:port/db
        if 'postgresql://' in base_url:
            base_url = base_url.replace('postgresql://', 'postgresql+asyncpg://')
        
        return {
            'primary': {
                'name': 'primary',
                'url': base_url
            },
            'replicas': [
                {
                    'name': 'replica1',
                    'url': base_url.replace(':5432', ':5433'),
                    'weight': 1
                },
                {
                    'name': 'replica2',
                    'url': base_url.replace(':5432', ':5434'),
                    'weight': 1
                }
            ]
        }
    
    async def _monitor_health(self) -> None:
        """Monitor replica health in background."""
        while True:
            try:
                await asyncio.sleep(self._health_check_interval)
                
                # Check primary
                if self.primary:
                    await self.primary.health_check()
                
                # Check replicas
                for replica in self.replicas:
                    await replica.health_check()
                
                # Log health status
                healthy_replicas = [
                    r for r in self.replicas 
                    if r.health == ReplicaHealth.HEALTHY
                ]
                logger.debug(f"Health check: {len(healthy_replicas)}/{len(self.replicas)} replicas healthy")
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
    
    def _get_healthy_replicas(self) -> List[DatabaseReplica]:
        """Get list of healthy replicas."""
        return [
            r for r in self.replicas
            if r.health in [ReplicaHealth.HEALTHY, ReplicaHealth.DEGRADED]
        ]
    
    def _select_replica(self, strategy: str = "weighted_round_robin") -> Optional[DatabaseReplica]:
        """
        Select a replica for read query.
        
        Args:
            strategy: Selection strategy
        
        Returns:
            Selected replica or None
        """
        healthy_replicas = self._get_healthy_replicas()
        
        if not healthy_replicas:
            return None
        
        if strategy == "random":
            return random.choice(healthy_replicas)
        
        elif strategy == "weighted_round_robin":
            # Weighted round-robin selection
            total_weight = sum(r.weight for r in healthy_replicas)
            
            if total_weight == 0:
                return healthy_replicas[0] if healthy_replicas else None
            
            # Select based on weights
            target = self._current_replica_index % total_weight
            cumulative = 0
            
            for replica in healthy_replicas:
                cumulative += replica.weight
                if target < cumulative:
                    self._current_replica_index += 1
                    return replica
            
            return healthy_replicas[0]
        
        elif strategy == "least_connections":
            # Select replica with least active connections
            # This would require tracking active connections
            return min(
                healthy_replicas,
                key=lambda r: r._metrics.get('queries_executed', 0)
            )
        
        return healthy_replicas[0] if healthy_replicas else None
    
    @asynccontextmanager
    async def get_session(
        self,
        query_type: QueryType = QueryType.READ
    ) -> AsyncGenerator[AsyncSession, None]:
        """
        Get database session routed to appropriate database.
        
        Args:
            query_type: Type of query (read/write)
        
        Yields:
            Database session
        """
        if not self._initialized:
            await self.initialize()
        
        # Route writes to primary
        if query_type in [QueryType.WRITE, QueryType.READ_WRITE]:
            if not self.primary:
                raise RuntimeError("No primary database available")
            
            async with self.primary.get_session() as session:
                yield session
        
        # Route reads to replicas (or primary if no replicas)
        else:
            replica = self._select_replica()
            
            # Fallback to primary if no healthy replicas
            if not replica:
                logger.warning("No healthy replicas, falling back to primary")
                replica = self.primary
            
            if not replica:
                raise RuntimeError("No database available")
            
            async with replica.get_session() as session:
                yield session
    
    async def execute_read(
        self,
        query: Any,
        max_retries: int = 3
    ) -> Any:
        """
        Execute read query with automatic failover.
        
        Args:
            query: SQLAlchemy query
            max_retries: Maximum retry attempts
        
        Returns:
            Query result
        """
        last_error = None
        
        for attempt in range(max_retries):
            try:
                async with self.get_session(QueryType.READ) as session:
                    result = await session.execute(query)
                    return result
                    
            except (OperationalError, DBAPIError) as e:
                last_error = e
                logger.warning(f"Read query failed (attempt {attempt + 1}): {e}")
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.5 * (2 ** attempt))  # Exponential backoff
        
        raise last_error
    
    async def execute_write(
        self,
        query: Any
    ) -> Any:
        """
        Execute write query on primary.
        
        Args:
            query: SQLAlchemy query
        
        Returns:
            Query result
        """
        async with self.get_session(QueryType.WRITE) as session:
            result = await session.execute(query)
            await session.commit()
            return result
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get replica manager metrics."""
        metrics = {
            'primary': self.primary._metrics if self.primary else None,
            'replicas': {}
        }
        
        for replica in self.replicas:
            metrics['replicas'][replica.name] = {
                'health': replica.health.value,
                'metrics': replica._metrics
            }
        
        # Overall stats
        healthy_count = len(self._get_healthy_replicas())
        metrics['summary'] = {
            'total_replicas': len(self.replicas),
            'healthy_replicas': healthy_count,
            'degraded_replicas': len([
                r for r in self.replicas 
                if r.health == ReplicaHealth.DEGRADED
            ]),
            'unhealthy_replicas': len([
                r for r in self.replicas 
                if r.health == ReplicaHealth.UNHEALTHY
            ])
        }
        
        return metrics
    
    async def close(self) -> None:
        """Close all database connections."""
        # Cancel health monitoring
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # Close all connections
        if self.primary:
            await self.primary.close()
        
        for replica in self.replicas:
            await replica.close()
        
        logger.info("Read replica manager closed")


# Global read replica manager
replica_manager = ReadReplicaManager()


# FastAPI dependency
async def get_read_session() -> AsyncGenerator[AsyncSession, None]:
    """Get read-only database session."""
    async with replica_manager.get_session(QueryType.READ) as session:
        yield session


async def get_write_session() -> AsyncGenerator[AsyncSession, None]:
    """Get write database session."""
    async with replica_manager.get_session(QueryType.WRITE) as session:
        yield session