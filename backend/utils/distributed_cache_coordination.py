"""
Distributed Cache Coordination System
Advanced coordination for distributed caching across multiple nodes with intelligent
invalidation policies, consistency management, and conflict resolution.
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import threading
import hashlib
from concurrent.futures import ThreadPoolExecutor

import redis.asyncio as redis
from redis.exceptions import RedisError, LockError

from backend.utils.redis_resilience import ResilientRedisClient, CircuitState
from backend.utils.monitoring import metrics

logger = logging.getLogger(__name__)


class ConsistencyLevel(Enum):
    """Cache consistency levels."""
    EVENTUAL = "eventual"      # Best performance, eventual consistency
    STRONG = "strong"         # Strong consistency, higher latency
    BOUNDED_STALENESS = "bounded_staleness"  # Bounded staleness guarantees


class InvalidationType(Enum):
    """Cache invalidation types."""
    IMMEDIATE = "immediate"    # Invalidate immediately
    LAZY = "lazy"             # Invalidate on next access
    TTL_BASED = "ttl_based"   # Let TTL handle invalidation
    EVENT_DRIVEN = "event_driven"  # Invalidate based on events


class ConflictResolution(Enum):
    """Conflict resolution strategies."""
    LAST_WRITE_WINS = "last_write_wins"
    TIMESTAMP_ORDERING = "timestamp_ordering"
    VERSION_VECTOR = "version_vector"
    CUSTOM_MERGE = "custom_merge"


@dataclass
class CacheEntry:
    """Enhanced cache entry with distributed metadata."""
    key: str
    value: Any
    version: int
    timestamp: float
    node_id: str
    ttl: int
    consistency_level: ConsistencyLevel
    tags: Set[str] = field(default_factory=set)
    dependencies: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InvalidationEvent:
    """Cache invalidation event."""
    event_id: str
    invalidation_type: InvalidationType
    patterns: List[str]
    timestamp: float
    node_id: str
    reason: str
    priority: int = 1  # 1=highest, 5=lowest
    batch_id: Optional[str] = None


class DistributedLockManager:
    """
    Distributed lock manager for coordinating cache operations.
    """
    
    def __init__(self, redis_client: ResilientRedisClient):
        self.redis_client = redis_client
        self.locks = {}  # local lock tracking
        self.lock_timeout = 30  # seconds
        self._local_lock = threading.RLock()
    
    async def acquire_lock(
        self,
        resource: str,
        timeout: int = None,
        blocking: bool = True
    ) -> Optional[str]:
        """
        Acquire distributed lock for a resource.
        
        Returns:
            Lock token if successful, None if failed
        """
        timeout = timeout or self.lock_timeout
        lock_key = f"lock:{resource}"
        lock_token = str(uuid.uuid4())
        
        try:
            # Try to acquire lock
            if blocking:
                # Blocking acquisition with timeout
                end_time = time.time() + timeout
                while time.time() < end_time:
                    success = await self.redis_client.set(
                        lock_key, lock_token, ex=timeout
                    )
                    
                    if success:
                        with self._local_lock:
                            self.locks[resource] = {
                                'token': lock_token,
                                'acquired_at': time.time(),
                                'timeout': timeout
                            }
                        
                        logger.debug(f"Acquired lock for {resource}")
                        return lock_token
                    
                    await asyncio.sleep(0.1)  # Wait before retry
                
                return None  # Failed to acquire
                
            else:
                # Non-blocking acquisition
                success = await self.redis_client.set(
                    lock_key, lock_token, ex=timeout
                )
                
                if success:
                    with self._local_lock:
                        self.locks[resource] = {
                            'token': lock_token,
                            'acquired_at': time.time(),
                            'timeout': timeout
                        }
                    
                    return lock_token
                
                return None
        
        except Exception as e:
            logger.error(f"Failed to acquire lock for {resource}: {e}")
            return None
    
    async def release_lock(self, resource: str, token: str) -> bool:
        """Release distributed lock."""
        lock_key = f"lock:{resource}"
        
        try:
            # Lua script for atomic lock release
            lua_script = """
                if redis.call("GET", KEYS[1]) == ARGV[1] then
                    return redis.call("DEL", KEYS[1])
                else
                    return 0
                end
            """
            
            # Execute script
            result = await self.redis_client._redis_client.eval(
                lua_script, 1, lock_key, token
            )
            
            if result:
                with self._local_lock:
                    if resource in self.locks:
                        del self.locks[resource]
                
                logger.debug(f"Released lock for {resource}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to release lock for {resource}: {e}")
            return False
    
    async def extend_lock(self, resource: str, token: str, extend_seconds: int) -> bool:
        """Extend lock timeout."""
        lock_key = f"lock:{resource}"
        
        try:
            # Lua script for atomic lock extension
            lua_script = """
                if redis.call("GET", KEYS[1]) == ARGV[1] then
                    return redis.call("EXPIRE", KEYS[1], ARGV[2])
                else
                    return 0
                end
            """
            
            result = await self.redis_client._redis_client.eval(
                lua_script, 1, lock_key, token, extend_seconds
            )
            
            if result:
                with self._local_lock:
                    if resource in self.locks:
                        self.locks[resource]['timeout'] = extend_seconds
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to extend lock for {resource}: {e}")
            return False
    
    async def cleanup_expired_locks(self):
        """Clean up expired local lock references."""
        current_time = time.time()
        expired_resources = []
        
        with self._local_lock:
            for resource, lock_info in self.locks.items():
                acquired_at = lock_info['acquired_at']
                timeout = lock_info['timeout']
                
                if current_time - acquired_at > timeout:
                    expired_resources.append(resource)
        
        # Remove expired locks
        for resource in expired_resources:
            with self._local_lock:
                if resource in self.locks:
                    del self.locks[resource]
            
            logger.debug(f"Cleaned up expired lock reference for {resource}")


class InvalidationCoordinator:
    """
    Coordinates cache invalidation across distributed nodes.
    """
    
    def __init__(self, redis_client: ResilientRedisClient, node_id: str):
        self.redis_client = redis_client
        self.node_id = node_id
        self.invalidation_channel = "cache_invalidation"
        self.pending_invalidations = deque()
        self.processed_events = set()  # Prevent duplicate processing
        self.event_handlers: Dict[str, Callable] = {}
        
        # Metrics
        self.metrics = {
            'events_sent': 0,
            'events_received': 0,
            'events_processed': 0,
            'events_failed': 0
        }
        
        # Background tasks
        self.subscriber_task: Optional[asyncio.Task] = None
        self.processor_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start invalidation coordination."""
        # Start subscriber task
        self.subscriber_task = asyncio.create_task(self._invalidation_subscriber())
        
        # Start processor task
        self.processor_task = asyncio.create_task(self._invalidation_processor())
        
        logger.info(f"Invalidation coordinator started for node {self.node_id}")
    
    async def stop(self):
        """Stop invalidation coordination."""
        if self.subscriber_task:
            self.subscriber_task.cancel()
        
        if self.processor_task:
            self.processor_task.cancel()
        
        logger.info("Invalidation coordinator stopped")
    
    async def invalidate_pattern(
        self,
        patterns: List[str],
        invalidation_type: InvalidationType = InvalidationType.IMMEDIATE,
        reason: str = "manual",
        priority: int = 1,
        batch_id: Optional[str] = None
    ) -> bool:
        """
        Send invalidation event for patterns.
        """
        try:
            event = InvalidationEvent(
                event_id=str(uuid.uuid4()),
                invalidation_type=invalidation_type,
                patterns=patterns,
                timestamp=time.time(),
                node_id=self.node_id,
                reason=reason,
                priority=priority,
                batch_id=batch_id
            )
            
            # Serialize and publish event
            event_data = {
                'event_id': event.event_id,
                'invalidation_type': event.invalidation_type.value,
                'patterns': event.patterns,
                'timestamp': event.timestamp,
                'node_id': event.node_id,
                'reason': event.reason,
                'priority': event.priority,
                'batch_id': event.batch_id
            }
            
            # Publish to Redis channel
            await self.redis_client._redis_client.publish(
                self.invalidation_channel,
                json.dumps(event_data)
            )
            
            self.metrics['events_sent'] += 1
            logger.debug(f"Sent invalidation event for patterns: {patterns}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send invalidation event: {e}")
            return False
    
    async def invalidate_by_tags(
        self,
        tags: List[str],
        invalidation_type: InvalidationType = InvalidationType.IMMEDIATE
    ) -> bool:
        """Invalidate entries by tags."""
        try:
            # Get keys associated with tags
            patterns = []
            for tag in tags:
                tag_key = f"tag:{tag}"
                tagged_keys = await self.redis_client.smembers(tag_key)
                patterns.extend(tagged_keys)
            
            if patterns:
                return await self.invalidate_pattern(
                    patterns,
                    invalidation_type,
                    reason=f"tag_invalidation:{','.join(tags)}"
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to invalidate by tags: {e}")
            return False
    
    async def _invalidation_subscriber(self):
        """Subscribe to invalidation events."""
        try:
            # Create subscriber
            pubsub = self.redis_client._redis_client.pubsub()
            await pubsub.subscribe(self.invalidation_channel)
            
            logger.info(f"Subscribed to invalidation channel: {self.invalidation_channel}")
            
            async for message in pubsub.listen():
                if message['type'] == 'message':
                    try:
                        event_data = json.loads(message['data'])
                        
                        # Skip events from this node
                        if event_data['node_id'] == self.node_id:
                            continue
                        
                        # Create event object
                        event = InvalidationEvent(
                            event_id=event_data['event_id'],
                            invalidation_type=InvalidationType(event_data['invalidation_type']),
                            patterns=event_data['patterns'],
                            timestamp=event_data['timestamp'],
                            node_id=event_data['node_id'],
                            reason=event_data['reason'],
                            priority=event_data['priority'],
                            batch_id=event_data.get('batch_id')
                        )
                        
                        # Queue for processing
                        self.pending_invalidations.append(event)
                        self.metrics['events_received'] += 1
                        
                    except Exception as e:
                        logger.error(f"Failed to process invalidation message: {e}")
                        self.metrics['events_failed'] += 1
        
        except asyncio.CancelledError:
            logger.info("Invalidation subscriber cancelled")
        except Exception as e:
            logger.error(f"Invalidation subscriber error: {e}")
    
    async def _invalidation_processor(self):
        """Process pending invalidation events."""
        try:
            while True:
                if not self.pending_invalidations:
                    await asyncio.sleep(0.1)
                    continue
                
                # Get next event (priority order)
                event = None
                if len(self.pending_invalidations) > 1:
                    # Sort by priority (1=highest)
                    sorted_events = sorted(
                        self.pending_invalidations,
                        key=lambda e: (e.priority, e.timestamp)
                    )
                    event = sorted_events[0]
                    self.pending_invalidations.remove(event)
                else:
                    event = self.pending_invalidations.popleft()
                
                # Check for duplicate
                if event.event_id in self.processed_events:
                    continue
                
                # Process event
                try:
                    await self._process_invalidation_event(event)
                    self.processed_events.add(event.event_id)
                    self.metrics['events_processed'] += 1
                    
                    # Limit processed events set size
                    if len(self.processed_events) > 10000:
                        # Remove oldest half
                        processed_list = list(self.processed_events)
                        self.processed_events = set(processed_list[-5000:])
                
                except Exception as e:
                    logger.error(f"Failed to process invalidation event {event.event_id}: {e}")
                    self.metrics['events_failed'] += 1
        
        except asyncio.CancelledError:
            logger.info("Invalidation processor cancelled")
        except Exception as e:
            logger.error(f"Invalidation processor error: {e}")
    
    async def _process_invalidation_event(self, event: InvalidationEvent):
        """Process a single invalidation event."""
        logger.debug(f"Processing invalidation event: {event.event_id}")
        
        # Handle different invalidation types
        if event.invalidation_type == InvalidationType.IMMEDIATE:
            await self._immediate_invalidation(event.patterns)
        elif event.invalidation_type == InvalidationType.LAZY:
            await self._lazy_invalidation(event.patterns)
        elif event.invalidation_type == InvalidationType.TTL_BASED:
            await self._ttl_based_invalidation(event.patterns)
        elif event.invalidation_type == InvalidationType.EVENT_DRIVEN:
            await self._event_driven_invalidation(event.patterns, event.reason)
        
        # Call custom handlers
        for handler in self.event_handlers.values():
            try:
                await handler(event)
            except Exception as e:
                logger.error(f"Custom invalidation handler failed: {e}")
    
    async def _immediate_invalidation(self, patterns: List[str]):
        """Perform immediate invalidation."""
        for pattern in patterns:
            try:
                if '*' in pattern or '?' in pattern:
                    # Pattern-based deletion
                    keys = await self.redis_client._redis_client.keys(pattern)
                    if keys:
                        await self.redis_client._redis_client.delete(*keys)
                else:
                    # Direct key deletion
                    await self.redis_client._redis_client.delete(pattern)
            
            except Exception as e:
                logger.error(f"Failed to invalidate pattern {pattern}: {e}")
    
    async def _lazy_invalidation(self, patterns: List[str]):
        """Mark entries for lazy invalidation."""
        for pattern in patterns:
            try:
                # Mark entries as invalid (they'll be checked on access)
                invalid_marker_key = f"invalid:{pattern}"
                await self.redis_client.set(invalid_marker_key, "1", ex=3600)  # 1 hour marker
            
            except Exception as e:
                logger.error(f"Failed to mark pattern {pattern} for lazy invalidation: {e}")
    
    async def _ttl_based_invalidation(self, patterns: List[str]):
        """Reduce TTL for TTL-based invalidation."""
        for pattern in patterns:
            try:
                if '*' in pattern or '?' in pattern:
                    keys = await self.redis_client._redis_client.keys(pattern)
                    for key in keys:
                        await self.redis_client.expire(key, 1)  # Expire in 1 second
                else:
                    await self.redis_client.expire(pattern, 1)
            
            except Exception as e:
                logger.error(f"Failed TTL-based invalidation for {pattern}: {e}")
    
    async def _event_driven_invalidation(self, patterns: List[str], reason: str):
        """Handle event-driven invalidation."""
        # This could trigger more complex invalidation logic based on the reason
        await self._immediate_invalidation(patterns)
    
    def register_handler(self, name: str, handler: Callable):
        """Register custom invalidation handler."""
        self.event_handlers[name] = handler
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get invalidation metrics."""
        return self.metrics.copy()


class ConsistencyManager:
    """
    Manages cache consistency across distributed nodes.
    """
    
    def __init__(self, redis_client: ResilientRedisClient, node_id: str):
        self.redis_client = redis_client
        self.node_id = node_id
        self.version_vectors: Dict[str, Dict[str, int]] = {}
        self.conflict_resolver = ConflictResolver()
        
        # Consistency tracking
        self.consistency_violations = defaultdict(int)
        self.reconciliation_events = []
    
    async def read_with_consistency(
        self,
        key: str,
        consistency_level: ConsistencyLevel
    ) -> Tuple[Optional[Any], Dict[str, Any]]:
        """
        Read value with specified consistency level.
        """
        if consistency_level == ConsistencyLevel.EVENTUAL:
            return await self._eventual_read(key)
        elif consistency_level == ConsistencyLevel.STRONG:
            return await self._strong_read(key)
        elif consistency_level == ConsistencyLevel.BOUNDED_STALENESS:
            return await self._bounded_staleness_read(key)
        else:
            return await self._eventual_read(key)
    
    async def write_with_consistency(
        self,
        key: str,
        value: Any,
        consistency_level: ConsistencyLevel,
        version: Optional[int] = None
    ) -> bool:
        """
        Write value with specified consistency level.
        """
        if consistency_level == ConsistencyLevel.STRONG:
            return await self._strong_write(key, value, version)
        else:
            return await self._eventual_write(key, value, version)
    
    async def _eventual_read(self, key: str) -> Tuple[Optional[Any], Dict[str, Any]]:
        """Eventual consistency read - fastest but may return stale data."""
        try:
            data = await self.redis_client.get(key)
            metadata = {
                'consistency_level': 'eventual',
                'timestamp': time.time(),
                'node_id': self.node_id
            }
            
            return data, metadata
            
        except Exception as e:
            logger.error(f"Eventual read failed for {key}: {e}")
            return None, {}
    
    async def _strong_read(self, key: str) -> Tuple[Optional[Any], Dict[str, Any]]:
        """Strong consistency read - ensures latest value."""
        try:
            # Use distributed lock for strong consistency
            lock_manager = DistributedLockManager(self.redis_client)
            lock_token = await lock_manager.acquire_lock(f"read:{key}", timeout=5)
            
            if not lock_token:
                raise Exception("Failed to acquire read lock for strong consistency")
            
            try:
                # Read value
                data = await self.redis_client.get(key)
                
                # Get version information
                version_key = f"version:{key}"
                version_data = await self.redis_client.get(version_key)
                
                metadata = {
                    'consistency_level': 'strong',
                    'timestamp': time.time(),
                    'node_id': self.node_id,
                    'version': version_data or 0
                }
                
                return data, metadata
                
            finally:
                await lock_manager.release_lock(f"read:{key}", lock_token)
        
        except Exception as e:
            logger.error(f"Strong read failed for {key}: {e}")
            return None, {}
    
    async def _bounded_staleness_read(
        self,
        key: str,
        max_staleness_seconds: int = 300
    ) -> Tuple[Optional[Any], Dict[str, Any]]:
        """Bounded staleness read - ensures data isn't too old."""
        try:
            # Get data and timestamp
            data = await self.redis_client.get(key)
            timestamp_key = f"ts:{key}"
            timestamp_data = await self.redis_client.get(timestamp_key)
            
            if data is None:
                return None, {}
            
            # Check staleness
            if timestamp_data:
                try:
                    write_timestamp = float(timestamp_data)
                    staleness = time.time() - write_timestamp
                    
                    if staleness > max_staleness_seconds:
                        # Data too stale, try to refresh
                        logger.warning(f"Data for {key} is stale ({staleness:.1f}s)")
                        # Could trigger refresh here
                        
                        metadata = {
                            'consistency_level': 'bounded_staleness',
                            'timestamp': time.time(),
                            'node_id': self.node_id,
                            'staleness_seconds': staleness,
                            'stale': True
                        }
                    else:
                        metadata = {
                            'consistency_level': 'bounded_staleness',
                            'timestamp': time.time(),
                            'node_id': self.node_id,
                            'staleness_seconds': staleness,
                            'stale': False
                        }
                    
                except ValueError:
                    metadata = {'error': 'invalid_timestamp'}
            else:
                metadata = {
                    'consistency_level': 'bounded_staleness',
                    'timestamp': time.time(),
                    'node_id': self.node_id,
                    'staleness_unknown': True
                }
            
            return data, metadata
            
        except Exception as e:
            logger.error(f"Bounded staleness read failed for {key}: {e}")
            return None, {}
    
    async def _strong_write(
        self,
        key: str,
        value: Any,
        version: Optional[int] = None
    ) -> bool:
        """Strong consistency write."""
        try:
            lock_manager = DistributedLockManager(self.redis_client)
            lock_token = await lock_manager.acquire_lock(f"write:{key}", timeout=10)
            
            if not lock_token:
                raise Exception("Failed to acquire write lock for strong consistency")
            
            try:
                # Get current version
                version_key = f"version:{key}"
                current_version = await self.redis_client.get(version_key)
                current_version = int(current_version or 0)
                
                # Check version conflict
                if version is not None and version < current_version:
                    logger.warning(f"Version conflict for {key}: {version} < {current_version}")
                    return False
                
                # Increment version
                new_version = current_version + 1
                
                # Write value, version, and timestamp atomically
                pipeline = self.redis_client._redis_client.pipeline()
                pipeline.set(key, value)
                pipeline.set(version_key, new_version)
                pipeline.set(f"ts:{key}", time.time())
                await pipeline.execute()
                
                logger.debug(f"Strong write successful for {key}, version: {new_version}")
                return True
                
            finally:
                await lock_manager.release_lock(f"write:{key}", lock_token)
        
        except Exception as e:
            logger.error(f"Strong write failed for {key}: {e}")
            return False
    
    async def _eventual_write(
        self,
        key: str,
        value: Any,
        version: Optional[int] = None
    ) -> bool:
        """Eventual consistency write."""
        try:
            # Simple write with timestamp
            pipeline = self.redis_client._redis_client.pipeline()
            pipeline.set(key, value)
            pipeline.set(f"ts:{key}", time.time())
            await pipeline.execute()
            
            return True
            
        except Exception as e:
            logger.error(f"Eventual write failed for {key}: {e}")
            return False
    
    async def detect_conflicts(self, keys: List[str]) -> List[Dict[str, Any]]:
        """Detect consistency conflicts for given keys."""
        conflicts = []
        
        for key in keys:
            try:
                # Get all replicas info (simplified)
                version_key = f"version:{key}"
                timestamp_key = f"ts:{key}"
                
                version = await self.redis_client.get(version_key)
                timestamp = await self.redis_client.get(timestamp_key)
                
                # Check if we have conflict markers
                conflict_key = f"conflict:{key}"
                conflict_info = await self.redis_client.get(conflict_key)
                
                if conflict_info:
                    conflicts.append({
                        'key': key,
                        'type': 'version_conflict',
                        'version': version,
                        'timestamp': timestamp,
                        'conflict_info': conflict_info
                    })
            
            except Exception as e:
                logger.error(f"Failed to check conflicts for {key}: {e}")
        
        return conflicts
    
    async def resolve_conflicts(self, conflicts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Resolve detected conflicts."""
        resolution_results = {
            'resolved': 0,
            'failed': 0,
            'strategies_used': defaultdict(int)
        }
        
        for conflict in conflicts:
            try:
                strategy = ConflictResolution.LAST_WRITE_WINS  # Default strategy
                
                success = await self.conflict_resolver.resolve_conflict(
                    conflict, strategy
                )
                
                if success:
                    resolution_results['resolved'] += 1
                else:
                    resolution_results['failed'] += 1
                
                resolution_results['strategies_used'][strategy.value] += 1
                
            except Exception as e:
                logger.error(f"Failed to resolve conflict for {conflict['key']}: {e}")
                resolution_results['failed'] += 1
        
        return dict(resolution_results)


class ConflictResolver:
    """Handles conflict resolution strategies."""
    
    async def resolve_conflict(
        self,
        conflict: Dict[str, Any],
        strategy: ConflictResolution
    ) -> bool:
        """Resolve conflict using specified strategy."""
        try:
            if strategy == ConflictResolution.LAST_WRITE_WINS:
                return await self._last_write_wins(conflict)
            elif strategy == ConflictResolution.TIMESTAMP_ORDERING:
                return await self._timestamp_ordering(conflict)
            else:
                # Default to last write wins
                return await self._last_write_wins(conflict)
                
        except Exception as e:
            logger.error(f"Conflict resolution failed: {e}")
            return False
    
    async def _last_write_wins(self, conflict: Dict[str, Any]) -> bool:
        """Resolve using last write wins strategy."""
        # Implementation would choose the version with latest timestamp
        return True
    
    async def _timestamp_ordering(self, conflict: Dict[str, Any]) -> bool:
        """Resolve using timestamp ordering."""
        # Implementation would order writes by timestamp
        return True


class DistributedCacheCoordinator:
    """
    Main coordinator for distributed caching operations.
    """
    
    def __init__(self, redis_client: ResilientRedisClient):
        self.redis_client = redis_client
        self.node_id = f"node_{uuid.uuid4().hex[:8]}"
        
        # Components
        self.lock_manager = DistributedLockManager(redis_client)
        self.invalidation_coordinator = InvalidationCoordinator(redis_client, self.node_id)
        self.consistency_manager = ConsistencyManager(redis_client, self.node_id)
        
        # Node health tracking
        self.node_registry = NodeRegistry(redis_client, self.node_id)
        
        # Metrics
        self.coordination_metrics = {
            'nodes_active': 0,
            'locks_acquired': 0,
            'locks_failed': 0,
            'invalidations_sent': 0,
            'invalidations_received': 0,
            'conflicts_detected': 0,
            'conflicts_resolved': 0
        }
    
    async def start(self):
        """Start distributed cache coordination."""
        # Register this node
        await self.node_registry.register_node()
        
        # Start invalidation coordination
        await self.invalidation_coordinator.start()
        
        # Start background tasks
        asyncio.create_task(self._health_monitoring())
        asyncio.create_task(self._metrics_collection())
        
        logger.info(f"Distributed cache coordinator started for node {self.node_id}")
    
    async def stop(self):
        """Stop distributed cache coordination."""
        await self.invalidation_coordinator.stop()
        await self.node_registry.unregister_node()
        
        logger.info("Distributed cache coordinator stopped")
    
    async def coordinated_set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        consistency_level: ConsistencyLevel = ConsistencyLevel.EVENTUAL,
        invalidate_pattern: Optional[str] = None
    ) -> bool:
        """Set value with coordination."""
        try:
            # Write with consistency
            success = await self.consistency_manager.write_with_consistency(
                key, value, consistency_level
            )
            
            if success and ttl:
                await self.redis_client.expire(key, ttl)
            
            # Send invalidation if pattern specified
            if success and invalidate_pattern:
                await self.invalidation_coordinator.invalidate_pattern(
                    [invalidate_pattern],
                    InvalidationType.IMMEDIATE,
                    reason="coordinated_set"
                )
            
            return success
            
        except Exception as e:
            logger.error(f"Coordinated set failed for {key}: {e}")
            return False
    
    async def coordinated_get(
        self,
        key: str,
        consistency_level: ConsistencyLevel = ConsistencyLevel.EVENTUAL
    ) -> Tuple[Optional[Any], Dict[str, Any]]:
        """Get value with coordination."""
        try:
            return await self.consistency_manager.read_with_consistency(
                key, consistency_level
            )
            
        except Exception as e:
            logger.error(f"Coordinated get failed for {key}: {e}")
            return None, {}
    
    async def coordinated_delete(
        self,
        keys: List[str],
        invalidation_type: InvalidationType = InvalidationType.IMMEDIATE
    ) -> bool:
        """Delete keys with coordination."""
        try:
            # Delete locally
            deleted = await self.redis_client.delete(*keys)
            
            # Send invalidation to other nodes
            await self.invalidation_coordinator.invalidate_pattern(
                keys,
                invalidation_type,
                reason="coordinated_delete"
            )
            
            return deleted > 0
            
        except Exception as e:
            logger.error(f"Coordinated delete failed: {e}")
            return False
    
    async def batch_invalidate(
        self,
        patterns: List[str],
        invalidation_type: InvalidationType = InvalidationType.IMMEDIATE,
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """Batch invalidate patterns."""
        results = {
            'batches_sent': 0,
            'total_patterns': len(patterns),
            'failed_batches': 0
        }
        
        # Process in batches
        batch_id = str(uuid.uuid4())
        
        for i in range(0, len(patterns), batch_size):
            batch = patterns[i:i + batch_size]
            
            try:
                success = await self.invalidation_coordinator.invalidate_pattern(
                    batch,
                    invalidation_type,
                    reason="batch_invalidate",
                    batch_id=batch_id
                )
                
                if success:
                    results['batches_sent'] += 1
                else:
                    results['failed_batches'] += 1
                    
            except Exception as e:
                logger.error(f"Batch invalidation failed for batch {i//batch_size}: {e}")
                results['failed_batches'] += 1
        
        return results
    
    async def health_check(self) -> Dict[str, Any]:
        """Get health status of distributed cache coordination."""
        try:
            # Check Redis connectivity
            redis_healthy = await self.redis_client.ping()
            
            # Get active nodes
            active_nodes = await self.node_registry.get_active_nodes()
            
            # Get component metrics
            invalidation_metrics = self.invalidation_coordinator.get_metrics()
            
            return {
                'node_id': self.node_id,
                'redis_healthy': redis_healthy,
                'active_nodes': len(active_nodes),
                'invalidation_metrics': invalidation_metrics,
                'coordination_metrics': self.coordination_metrics,
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'node_id': self.node_id,
                'healthy': False,
                'error': str(e),
                'timestamp': time.time()
            }
    
    async def _health_monitoring(self):
        """Monitor health of distributed cache system."""
        while True:
            try:
                # Update node registry
                await self.node_registry.heartbeat()
                
                # Clean up expired locks
                await self.lock_manager.cleanup_expired_locks()
                
                # Detect conflicts
                # This would be more sophisticated in a real implementation
                
                await asyncio.sleep(30)  # Every 30 seconds
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _metrics_collection(self):
        """Collect and update metrics."""
        while True:
            try:
                # Update coordination metrics
                active_nodes = await self.node_registry.get_active_nodes()
                self.coordination_metrics['nodes_active'] = len(active_nodes)
                
                # Update component metrics
                invalidation_metrics = self.invalidation_coordinator.get_metrics()
                self.coordination_metrics.update({
                    'invalidations_sent': invalidation_metrics['events_sent'],
                    'invalidations_received': invalidation_metrics['events_received']
                })
                
                await asyncio.sleep(60)  # Every minute
                
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(120)


class NodeRegistry:
    """Registry for tracking active cache nodes."""
    
    def __init__(self, redis_client: ResilientRedisClient, node_id: str):
        self.redis_client = redis_client
        self.node_id = node_id
        self.registry_key = "cache_nodes"
        self.heartbeat_interval = 30  # seconds
    
    async def register_node(self):
        """Register this node in the registry."""
        node_info = {
            'node_id': self.node_id,
            'registered_at': time.time(),
            'last_heartbeat': time.time()
        }
        
        await self.redis_client.hset(
            self.registry_key,
            self.node_id,
            json.dumps(node_info)
        )
        
        logger.info(f"Node {self.node_id} registered")
    
    async def unregister_node(self):
        """Unregister this node from the registry."""
        await self.redis_client._redis_client.hdel(self.registry_key, self.node_id)
        logger.info(f"Node {self.node_id} unregistered")
    
    async def heartbeat(self):
        """Send heartbeat to maintain registration."""
        node_info = {
            'node_id': self.node_id,
            'last_heartbeat': time.time()
        }
        
        await self.redis_client.hset(
            self.registry_key,
            self.node_id,
            json.dumps(node_info)
        )
    
    async def get_active_nodes(self) -> List[Dict[str, Any]]:
        """Get list of active nodes."""
        try:
            node_data = await self.redis_client._redis_client.hgetall(self.registry_key)
            active_nodes = []
            current_time = time.time()
            
            for node_id, info_json in node_data.items():
                try:
                    node_info = json.loads(info_json)
                    last_heartbeat = node_info.get('last_heartbeat', 0)
                    
                    # Consider node active if heartbeat within last 2 minutes
                    if current_time - last_heartbeat < 120:
                        active_nodes.append(node_info)
                
                except json.JSONDecodeError:
                    continue
            
            return active_nodes
            
        except Exception as e:
            logger.error(f"Failed to get active nodes: {e}")
            return []


# Global distributed cache coordinator
distributed_cache_coordinator: Optional[DistributedCacheCoordinator] = None


async def initialize_distributed_coordination(redis_client: ResilientRedisClient):
    """Initialize distributed cache coordination."""
    global distributed_cache_coordinator
    
    distributed_cache_coordinator = DistributedCacheCoordinator(redis_client)
    await distributed_cache_coordinator.start()
    
    return distributed_cache_coordinator