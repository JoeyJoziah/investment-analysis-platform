"""
Async Locking Mechanisms
Provides comprehensive locking mechanisms for concurrent database access and processing safety.
"""

import asyncio
import logging
from typing import Dict, Optional, Any, Set
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import weakref

logger = logging.getLogger(__name__)


class LockType(Enum):
    """Types of locks available"""
    READ = "read"
    WRITE = "write"
    EXCLUSIVE = "exclusive"


@dataclass
class LockInfo:
    """Information about a lock"""
    lock_type: LockType
    resource_id: str
    acquired_at: datetime
    task_id: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class AsyncResourceLockManager:
    """
    Manages locks for database resources to prevent race conditions and ensure data consistency.
    Supports read/write locks and exclusive locks with deadlock detection.
    """
    
    def __init__(self, deadlock_timeout: float = 30.0):
        """
        Initialize the lock manager.
        
        Args:
            deadlock_timeout: Maximum time to wait for a lock before declaring deadlock
        """
        self.deadlock_timeout = deadlock_timeout
        
        # Resource locks: resource_id -> set of LockInfo
        self._locks: Dict[str, Set[LockInfo]] = {}
        
        # Task to locks mapping for deadlock detection
        self._task_locks: Dict[int, Set[str]] = {}
        
        # Lock queues for waiting tasks
        self._lock_queues: Dict[str, asyncio.Queue] = {}
        
        # Main lock for thread safety
        self._main_lock = asyncio.Lock()
        
        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._start_cleanup_task()
    
    def _start_cleanup_task(self):
        """Start background task for lock cleanup"""
        async def cleanup_expired_locks():
            while True:
                try:
                    await asyncio.sleep(60)  # Check every minute
                    await self._cleanup_expired_locks()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in lock cleanup task: {e}")
        
        self._cleanup_task = asyncio.create_task(cleanup_expired_locks())
    
    async def _cleanup_expired_locks(self):
        """Clean up expired locks"""
        async with self._main_lock:
            current_time = datetime.utcnow()
            expired_resources = []
            
            for resource_id, locks in self._locks.items():
                expired_locks = []
                
                for lock_info in locks:
                    if current_time - lock_info.acquired_at > timedelta(seconds=self.deadlock_timeout * 2):
                        expired_locks.append(lock_info)
                
                for expired_lock in expired_locks:
                    locks.discard(expired_lock)
                    if expired_lock.task_id:
                        self._task_locks.get(expired_lock.task_id, set()).discard(resource_id)
                
                if not locks:
                    expired_resources.append(resource_id)
            
            for resource_id in expired_resources:
                del self._locks[resource_id]
                if resource_id in self._lock_queues:
                    del self._lock_queues[resource_id]
            
            if expired_resources:
                logger.info(f"Cleaned up {len(expired_resources)} expired lock resources")
    
    def _get_current_task_id(self) -> Optional[int]:
        """Get current task ID"""
        task = asyncio.current_task()
        return id(task) if task else None
    
    async def _can_acquire_lock(self, resource_id: str, lock_type: LockType) -> bool:
        """Check if a lock can be acquired"""
        existing_locks = self._locks.get(resource_id, set())
        
        if not existing_locks:
            return True
        
        current_task_id = self._get_current_task_id()
        
        # Check if current task already holds a compatible lock
        for lock_info in existing_locks:
            if lock_info.task_id == current_task_id:
                # Same task, check compatibility
                if lock_type == LockType.READ and lock_info.lock_type == LockType.READ:
                    return True
                elif lock_type == LockType.WRITE or lock_info.lock_type == LockType.WRITE:
                    return False  # Cannot upgrade/mix read/write
                elif lock_type == LockType.EXCLUSIVE or lock_info.lock_type == LockType.EXCLUSIVE:
                    return lock_type == lock_info.lock_type  # Only same exclusive lock
        
        # Check compatibility with other tasks' locks
        if lock_type == LockType.READ:
            # Read locks are compatible with other read locks
            return all(lock_info.lock_type == LockType.READ for lock_info in existing_locks)
        
        elif lock_type == LockType.WRITE:
            # Write locks are not compatible with any other locks
            return False
        
        elif lock_type == LockType.EXCLUSIVE:
            # Exclusive locks are not compatible with any other locks
            return False
        
        return False
    
    async def _detect_deadlock(self, resource_id: str, requesting_task_id: int) -> bool:
        """Detect potential deadlock scenario"""
        # Simple deadlock detection: check if any task holding this resource
        # is waiting for a resource that the requesting task holds
        
        current_locks = self._locks.get(resource_id, set())
        requesting_task_resources = self._task_locks.get(requesting_task_id, set())
        
        for lock_info in current_locks:
            if lock_info.task_id and lock_info.task_id != requesting_task_id:
                # Check if this task is waiting for any resource the requesting task holds
                for held_resource in requesting_task_resources:
                    if held_resource != resource_id:  # Don't check the same resource
                        held_locks = self._locks.get(held_resource, set())
                        if any(l.task_id == requesting_task_id for l in held_locks):
                            # Potential deadlock detected
                            logger.warning(
                                f"Potential deadlock detected: Task {requesting_task_id} "
                                f"requesting {resource_id} while holding {held_resource}, "
                                f"but task {lock_info.task_id} holds {resource_id}"
                            )
                            return True
        
        return False
    
    @asynccontextmanager
    async def acquire_lock(
        self,
        resource_id: str,
        lock_type: LockType = LockType.READ,
        timeout: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Acquire a lock on a resource.
        
        Args:
            resource_id: Unique identifier for the resource
            lock_type: Type of lock to acquire
            timeout: Maximum time to wait for lock (uses deadlock_timeout if None)
            metadata: Optional metadata about the lock
        
        Yields:
            Lock context manager
        """
        if timeout is None:
            timeout = self.deadlock_timeout
        
        task_id = self._get_current_task_id()
        lock_acquired = False
        
        try:
            # Wait for lock availability
            start_time = datetime.utcnow()
            
            while True:
                async with self._main_lock:
                    # Check for deadlock
                    if task_id and await self._detect_deadlock(resource_id, task_id):
                        raise asyncio.TimeoutError(
                            f"Deadlock detected while acquiring {lock_type.value} lock on {resource_id}"
                        )
                    
                    # Check if we can acquire the lock
                    if await self._can_acquire_lock(resource_id, lock_type):
                        # Acquire the lock
                        lock_info = LockInfo(
                            lock_type=lock_type,
                            resource_id=resource_id,
                            acquired_at=datetime.utcnow(),
                            task_id=task_id,
                            metadata=metadata
                        )
                        
                        if resource_id not in self._locks:
                            self._locks[resource_id] = set()
                        
                        self._locks[resource_id].add(lock_info)
                        
                        if task_id:
                            if task_id not in self._task_locks:
                                self._task_locks[task_id] = set()
                            self._task_locks[task_id].add(resource_id)
                        
                        lock_acquired = True
                        logger.debug(
                            f"Acquired {lock_type.value} lock on {resource_id} "
                            f"by task {task_id}"
                        )
                        break
                
                # Check timeout
                if datetime.utcnow() - start_time > timedelta(seconds=timeout):
                    raise asyncio.TimeoutError(
                        f"Timeout waiting for {lock_type.value} lock on {resource_id}"
                    )
                
                # Wait a bit before retrying
                await asyncio.sleep(0.1)
            
            # Yield control while holding the lock
            yield
            
        finally:
            if lock_acquired:
                # Release the lock
                async with self._main_lock:
                    resource_locks = self._locks.get(resource_id, set())
                    
                    # Find and remove the lock
                    lock_to_remove = None
                    for lock_info in resource_locks:
                        if (lock_info.lock_type == lock_type and 
                            lock_info.task_id == task_id):
                            lock_to_remove = lock_info
                            break
                    
                    if lock_to_remove:
                        resource_locks.discard(lock_to_remove)
                        
                        # Clean up empty resource
                        if not resource_locks:
                            del self._locks[resource_id]
                        
                        # Clean up task locks
                        if task_id and task_id in self._task_locks:
                            self._task_locks[task_id].discard(resource_id)
                            if not self._task_locks[task_id]:
                                del self._task_locks[task_id]
                        
                        logger.debug(
                            f"Released {lock_type.value} lock on {resource_id} "
                            f"by task {task_id}"
                        )
    
    async def get_lock_status(self, resource_id: Optional[str] = None) -> Dict[str, Any]:
        """Get status of locks"""
        async with self._main_lock:
            if resource_id:
                locks = self._locks.get(resource_id, set())
                return {
                    'resource_id': resource_id,
                    'lock_count': len(locks),
                    'locks': [
                        {
                            'lock_type': lock.lock_type.value,
                            'task_id': lock.task_id,
                            'acquired_at': lock.acquired_at.isoformat(),
                            'metadata': lock.metadata
                        }
                        for lock in locks
                    ]
                }
            else:
                return {
                    'total_resources': len(self._locks),
                    'total_locks': sum(len(locks) for locks in self._locks.values()),
                    'resources': list(self._locks.keys()),
                    'tasks': list(self._task_locks.keys())
                }
    
    async def force_release_locks(self, task_id: int):
        """Force release all locks held by a specific task"""
        async with self._main_lock:
            if task_id not in self._task_locks:
                return
            
            resources_to_release = list(self._task_locks[task_id])
            
            for resource_id in resources_to_release:
                resource_locks = self._locks.get(resource_id, set())
                
                locks_to_remove = [
                    lock for lock in resource_locks 
                    if lock.task_id == task_id
                ]
                
                for lock in locks_to_remove:
                    resource_locks.discard(lock)
                
                if not resource_locks:
                    del self._locks[resource_id]
            
            del self._task_locks[task_id]
            
            logger.warning(f"Force released all locks for task {task_id}")
    
    async def shutdown(self):
        """Shutdown the lock manager"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Clear all locks
        async with self._main_lock:
            self._locks.clear()
            self._task_locks.clear()
            self._lock_queues.clear()
        
        logger.info("AsyncResourceLockManager shutdown complete")


# Global lock manager instance
resource_lock_manager = AsyncResourceLockManager()


# Convenience functions for common use cases
@asynccontextmanager
async def stock_read_lock(symbol: str, timeout: Optional[float] = None):
    """Acquire read lock for stock data"""
    async with resource_lock_manager.acquire_lock(
        f"stock:{symbol.upper()}",
        LockType.READ,
        timeout=timeout,
        metadata={'resource_type': 'stock', 'symbol': symbol}
    ):
        yield


@asynccontextmanager
async def stock_write_lock(symbol: str, timeout: Optional[float] = None):
    """Acquire write lock for stock data"""
    async with resource_lock_manager.acquire_lock(
        f"stock:{symbol.upper()}",
        LockType.WRITE,
        timeout=timeout,
        metadata={'resource_type': 'stock', 'symbol': symbol}
    ):
        yield


@asynccontextmanager
async def portfolio_lock(user_id: int, portfolio_id: int, timeout: Optional[float] = None):
    """Acquire exclusive lock for portfolio operations"""
    async with resource_lock_manager.acquire_lock(
        f"portfolio:{user_id}:{portfolio_id}",
        LockType.EXCLUSIVE,
        timeout=timeout,
        metadata={'resource_type': 'portfolio', 'user_id': user_id, 'portfolio_id': portfolio_id}
    ):
        yield


@asynccontextmanager
async def user_data_lock(user_id: int, timeout: Optional[float] = None):
    """Acquire exclusive lock for user data operations"""
    async with resource_lock_manager.acquire_lock(
        f"user:{user_id}",
        LockType.EXCLUSIVE,
        timeout=timeout,
        metadata={'resource_type': 'user', 'user_id': user_id}
    ):
        yield


# Database-specific locking decorators
def with_stock_lock(lock_type: LockType = LockType.READ):
    """Decorator for functions that need stock-level locking"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Try to extract symbol from arguments
            symbol = None
            if args and hasattr(args[0], '__name__') and 'symbol' in kwargs:
                symbol = kwargs['symbol']
            elif len(args) > 1 and isinstance(args[1], str):
                symbol = args[1]
            
            if not symbol:
                # Fall back to no locking if we can't determine the symbol
                return await func(*args, **kwargs)
            
            async with resource_lock_manager.acquire_lock(
                f"stock:{symbol.upper()}",
                lock_type,
                metadata={'function': func.__name__, 'resource_type': 'stock'}
            ):
                return await func(*args, **kwargs)
        
        return wrapper
    return decorator


async def cleanup_locks():
    """Clean up all locks (for testing or shutdown)"""
    await resource_lock_manager.shutdown()