"""
Redis Cluster Optimization and Memory Management
Advanced Redis clustering implementation with intelligent memory optimization
for high-performance financial data processing with 6000+ stocks.
"""

import asyncio
import logging
import time
import json
import hashlib
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import threading
import redis
import redis.asyncio as aioredis
from redis.cluster import RedisCluster
from redis.exceptions import RedisClusterException, RedisError
import redis.asyncio as aioredis_cluster

logger = logging.getLogger(__name__)


class ClusterRole(Enum):
    """Redis cluster node roles."""
    MASTER = "master"
    SLAVE = "slave"
    UNKNOWN = "unknown"


class ClusterState(Enum):
    """Redis cluster states."""
    OK = "ok"
    FAIL = "fail" 
    PARTIAL = "partial"


class MemoryOptimizationStrategy(Enum):
    """Memory optimization strategies."""
    AGGRESSIVE = "aggressive"       # Maximum memory optimization
    BALANCED = "balanced"          # Balance performance and memory
    PERFORMANCE = "performance"    # Prioritize performance over memory


@dataclass
class ClusterNode:
    """Redis cluster node information."""
    node_id: str
    host: str
    port: int
    role: ClusterRole
    master_id: Optional[str] = None
    slots: Set[int] = field(default_factory=set)
    flags: Set[str] = field(default_factory=set)
    ping_sent: int = 0
    pong_recv: int = 0
    config_epoch: int = 0
    connected: bool = True
    
    # Performance metrics
    memory_usage_mb: float = 0.0
    cpu_usage: float = 0.0
    ops_per_sec: int = 0
    hit_rate: float = 0.0
    latency_ms: float = 0.0


@dataclass
class ClusterSlotRange:
    """Redis cluster slot range."""
    start: int
    end: int
    master: ClusterNode
    replicas: List[ClusterNode] = field(default_factory=list)


class RedisClusterManager:
    """
    Manages Redis cluster operations with financial data optimizations.
    """
    
    def __init__(self, startup_nodes: List[Dict[str, Any]]):
        """
        Initialize Redis cluster manager.
        
        Args:
            startup_nodes: List of {'host': str, 'port': int} dictionaries
        """
        self.startup_nodes = startup_nodes
        self.cluster_client: Optional[RedisCluster] = None
        self.async_cluster_client: Optional[aioredis_cluster.RedisCluster] = None
        
        # Cluster state
        self.nodes: Dict[str, ClusterNode] = {}
        self.slot_map: Dict[int, ClusterNode] = {}  # slot -> master node
        self.cluster_state = ClusterState.UNKNOWN
        
        # Performance tracking
        self.performance_history = deque(maxlen=1000)
        self.rebalance_history = []
        
        # Configuration
        self.max_memory_per_node_mb = 8192  # 8GB default
        self.rebalance_threshold = 0.2      # 20% imbalance triggers rebalance
        self.health_check_interval = 30     # seconds
        
        # Memory optimization
        self.memory_optimizer = RedisMemoryOptimizer()
        
        # Background tasks
        self.monitoring_task: Optional[asyncio.Task] = None
        self.rebalancing_task: Optional[asyncio.Task] = None
    
    async def initialize(self) -> bool:
        """Initialize Redis cluster connections."""
        try:
            # Initialize synchronous cluster client
            self.cluster_client = RedisCluster(
                startup_nodes=self.startup_nodes,
                decode_responses=True,
                skip_full_coverage_check=True,
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Initialize asynchronous cluster client
            self.async_cluster_client = aioredis_cluster.RedisCluster(
                startup_nodes=self.startup_nodes,
                decode_responses=True,
                skip_full_coverage_check=True,
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True
            )
            
            # Discover cluster topology
            await self.discover_cluster_topology()
            
            # Start background monitoring
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            self.rebalancing_task = asyncio.create_task(self._rebalancing_loop())
            
            logger.info(f"Redis cluster initialized with {len(self.nodes)} nodes")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis cluster: {e}")
            return False
    
    async def discover_cluster_topology(self):
        """Discover and map cluster topology."""
        try:
            # Get cluster nodes information
            cluster_info = await self.async_cluster_client.cluster_nodes()
            
            # Parse node information
            self.nodes.clear()
            self.slot_map.clear()
            
            for line in cluster_info.split('\n'):
                if not line.strip():
                    continue
                
                parts = line.split()
                if len(parts) < 8:
                    continue
                
                node_id = parts[0]
                host_port = parts[1].split(':')[0] + ':' + parts[1].split(':')[1]
                host, port = host_port.rsplit(':', 1)
                flags = set(parts[2].split(','))
                master_id = parts[3] if parts[3] != '-' else None
                ping_sent = int(parts[4])
                pong_recv = int(parts[5])
                config_epoch = int(parts[6])
                link_state = parts[7]
                
                # Determine role
                role = ClusterRole.MASTER if 'master' in flags else ClusterRole.SLAVE
                
                # Parse slot ranges (for masters)
                slots = set()
                if len(parts) > 8 and role == ClusterRole.MASTER:
                    for slot_range in parts[8:]:
                        if '-' in slot_range:
                            start, end = map(int, slot_range.split('-'))
                            slots.update(range(start, end + 1))
                        else:
                            slots.add(int(slot_range))
                
                # Create node
                node = ClusterNode(
                    node_id=node_id,
                    host=host,
                    port=int(port),
                    role=role,
                    master_id=master_id,
                    slots=slots,
                    flags=flags,
                    ping_sent=ping_sent,
                    pong_recv=pong_recv,
                    config_epoch=config_epoch,
                    connected=(link_state == 'connected')
                )
                
                self.nodes[node_id] = node
                
                # Build slot map (masters only)
                if role == ClusterRole.MASTER:
                    for slot in slots:
                        self.slot_map[slot] = node
            
            # Update cluster state
            await self._update_cluster_state()
            
            logger.info(f"Discovered cluster topology: {len([n for n in self.nodes.values() if n.role == ClusterRole.MASTER])} masters, "
                       f"{len([n for n in self.nodes.values() if n.role == ClusterRole.SLAVE])} slaves")
            
        except Exception as e:
            logger.error(f"Failed to discover cluster topology: {e}")
            raise
    
    async def _update_cluster_state(self):
        """Update overall cluster state."""
        try:
            cluster_info = await self.async_cluster_client.cluster_info()
            
            # Parse cluster state
            for line in cluster_info.split('\r\n'):
                if line.startswith('cluster_state:'):
                    state_str = line.split(':')[1]
                    if state_str == 'ok':
                        self.cluster_state = ClusterState.OK
                    elif state_str == 'fail':
                        self.cluster_state = ClusterState.FAIL
                    else:
                        self.cluster_state = ClusterState.PARTIAL
                    break
            
        except Exception as e:
            logger.error(f"Failed to update cluster state: {e}")
            self.cluster_state = ClusterState.UNKNOWN
    
    async def get_cluster_health(self) -> Dict[str, Any]:
        """Get comprehensive cluster health information."""
        health = {
            'cluster_state': self.cluster_state.value,
            'total_nodes': len(self.nodes),
            'master_nodes': len([n for n in self.nodes.values() if n.role == ClusterRole.MASTER]),
            'slave_nodes': len([n for n in self.nodes.values() if n.role == ClusterRole.SLAVE]),
            'connected_nodes': len([n for n in self.nodes.values() if n.connected]),
            'slot_coverage': len(self.slot_map),
            'nodes': []
        }
        
        # Get detailed node information
        for node in self.nodes.values():
            node_health = {
                'node_id': node.node_id[:8],  # Short ID
                'address': f"{node.host}:{node.port}",
                'role': node.role.value,
                'connected': node.connected,
                'slots_count': len(node.slots),
                'memory_usage_mb': node.memory_usage_mb,
                'cpu_usage': node.cpu_usage,
                'ops_per_sec': node.ops_per_sec,
                'hit_rate': node.hit_rate,
                'latency_ms': node.latency_ms
            }
            health['nodes'].append(node_health)
        
        return health
    
    async def optimize_cluster_memory(
        self,
        strategy: MemoryOptimizationStrategy = MemoryOptimizationStrategy.BALANCED
    ) -> Dict[str, Any]:
        """Optimize memory usage across cluster nodes."""
        results = {
            'strategy': strategy.value,
            'nodes_optimized': 0,
            'memory_saved_mb': 0.0,
            'optimizations': []
        }
        
        try:
            for node in self.nodes.values():
                if node.role == ClusterRole.MASTER:
                    # Get node-specific client
                    node_client = redis.Redis(
                        host=node.host,
                        port=node.port,
                        decode_responses=True
                    )
                    
                    # Optimize node memory
                    node_optimization = await self.memory_optimizer.optimize_node_memory(
                        node_client, strategy
                    )
                    
                    if node_optimization['memory_saved_mb'] > 0:
                        results['nodes_optimized'] += 1
                        results['memory_saved_mb'] += node_optimization['memory_saved_mb']
                        results['optimizations'].append({
                            'node': f"{node.host}:{node.port}",
                            'optimization': node_optimization
                        })
            
            logger.info(f"Cluster memory optimization completed: {results['memory_saved_mb']:.1f}MB saved across {results['nodes_optimized']} nodes")
            
        except Exception as e:
            logger.error(f"Cluster memory optimization failed: {e}")
            results['error'] = str(e)
        
        return results
    
    async def rebalance_cluster(self, dry_run: bool = True) -> Dict[str, Any]:
        """Rebalance cluster slots for optimal distribution."""
        rebalance_plan = {
            'timestamp': datetime.now().isoformat(),
            'dry_run': dry_run,
            'moves': [],
            'total_moves': 0,
            'estimated_downtime_ms': 0
        }
        
        try:
            # Calculate current slot distribution
            master_nodes = [n for n in self.nodes.values() if n.role == ClusterRole.MASTER]
            if len(master_nodes) < 2:
                return {'error': 'Need at least 2 master nodes for rebalancing'}
            
            target_slots_per_node = 16384 // len(master_nodes)
            slot_moves = []
            
            # Find nodes with too many or too few slots
            overloaded_nodes = []
            underloaded_nodes = []
            
            for node in master_nodes:
                slot_count = len(node.slots)
                if slot_count > target_slots_per_node * (1 + self.rebalance_threshold):
                    excess_slots = slot_count - target_slots_per_node
                    overloaded_nodes.append((node, excess_slots))
                elif slot_count < target_slots_per_node * (1 - self.rebalance_threshold):
                    needed_slots = target_slots_per_node - slot_count
                    underloaded_nodes.append((node, needed_slots))
            
            # Plan slot moves
            for overloaded_node, excess_slots in overloaded_nodes:
                slots_to_move = list(overloaded_node.slots)[:excess_slots]
                
                for underloaded_node, needed_slots in underloaded_nodes:
                    if not slots_to_move:
                        break
                    
                    slots_for_this_node = min(len(slots_to_move), needed_slots)
                    moving_slots = slots_to_move[:slots_for_this_node]
                    
                    slot_moves.append({
                        'from_node': f"{overloaded_node.host}:{overloaded_node.port}",
                        'to_node': f"{underloaded_node.host}:{underloaded_node.port}",
                        'slots': moving_slots,
                        'slot_count': len(moving_slots)
                    })
                    
                    slots_to_move = slots_to_move[slots_for_this_node:]
            
            rebalance_plan['moves'] = slot_moves
            rebalance_plan['total_moves'] = sum(move['slot_count'] for move in slot_moves)
            rebalance_plan['estimated_downtime_ms'] = rebalance_plan['total_moves'] * 10  # ~10ms per slot
            
            # Execute moves if not dry run
            if not dry_run and slot_moves:
                executed_moves = 0
                for move in slot_moves:
                    try:
                        # This would execute actual slot migration
                        # For now, just log the intended move
                        logger.info(f"Would move {move['slot_count']} slots from {move['from_node']} to {move['to_node']}")
                        executed_moves += move['slot_count']
                    except Exception as e:
                        logger.error(f"Failed to execute slot move: {e}")
                        break
                
                rebalance_plan['executed_moves'] = executed_moves
                
                # Record rebalance in history
                self.rebalance_history.append({
                    'timestamp': time.time(),
                    'moves_executed': executed_moves,
                    'total_planned': rebalance_plan['total_moves']
                })
            
            return rebalance_plan
            
        except Exception as e:
            logger.error(f"Cluster rebalancing failed: {e}")
            rebalance_plan['error'] = str(e)
            return rebalance_plan
    
    async def scale_cluster(
        self,
        target_nodes: int,
        node_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Scale cluster by adding or removing nodes."""
        current_masters = len([n for n in self.nodes.values() if n.role == ClusterRole.MASTER])
        
        scaling_plan = {
            'current_masters': current_masters,
            'target_masters': target_nodes,
            'action': 'none',
            'steps': []
        }
        
        if target_nodes > current_masters:
            scaling_plan['action'] = 'scale_up'
            nodes_to_add = target_nodes - current_masters
            
            for i in range(nodes_to_add):
                scaling_plan['steps'].append({
                    'step': f'add_node_{i + 1}',
                    'config': node_config,
                    'estimated_time_minutes': 5
                })
            
            # Add rebalancing step
            scaling_plan['steps'].append({
                'step': 'rebalance_slots',
                'estimated_time_minutes': 10
            })
            
        elif target_nodes < current_masters:
            scaling_plan['action'] = 'scale_down'
            nodes_to_remove = current_masters - target_nodes
            
            # First rebalance to evacuate slots from nodes to be removed
            scaling_plan['steps'].append({
                'step': 'rebalance_before_removal',
                'estimated_time_minutes': 15
            })
            
            for i in range(nodes_to_remove):
                scaling_plan['steps'].append({
                    'step': f'remove_node_{i + 1}',
                    'estimated_time_minutes': 3
                })
        
        return scaling_plan
    
    async def _monitoring_loop(self):
        """Background monitoring loop for cluster health."""
        logger.info("Cluster monitoring started")
        
        while True:
            try:
                # Update node performance metrics
                await self._collect_node_metrics()
                
                # Check cluster health
                health = await self.get_cluster_health()
                
                # Store performance data
                self.performance_history.append({
                    'timestamp': time.time(),
                    'health': health
                })
                
                # Alert on issues
                if health['cluster_state'] != 'ok':
                    logger.warning(f"Cluster state is {health['cluster_state']}")
                
                if health['connected_nodes'] < health['total_nodes']:
                    logger.warning(f"Not all nodes connected: {health['connected_nodes']}/{health['total_nodes']}")
                
                await asyncio.sleep(self.health_check_interval)
                
            except Exception as e:
                logger.error(f"Cluster monitoring error: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _collect_node_metrics(self):
        """Collect performance metrics from all nodes."""
        for node in self.nodes.values():
            if not node.connected:
                continue
            
            try:
                # Create connection to node
                node_client = redis.Redis(
                    host=node.host,
                    port=node.port,
                    decode_responses=True,
                    socket_timeout=5
                )
                
                # Get node info
                info = node_client.info()
                
                # Update node metrics
                node.memory_usage_mb = info.get('used_memory', 0) / (1024 * 1024)
                node.ops_per_sec = info.get('instantaneous_ops_per_sec', 0)
                
                # Calculate hit rate
                hits = info.get('keyspace_hits', 0)
                misses = info.get('keyspace_misses', 0)
                if hits + misses > 0:
                    node.hit_rate = hits / (hits + misses)
                
                # Get latency (simplified)
                start_time = time.time()
                node_client.ping()
                node.latency_ms = (time.time() - start_time) * 1000
                
            except Exception as e:
                logger.debug(f"Failed to collect metrics for node {node.host}:{node.port}: {e}")
                node.connected = False
    
    async def _rebalancing_loop(self):
        """Background loop for automatic rebalancing."""
        logger.info("Cluster rebalancing monitor started")
        
        while True:
            try:
                await asyncio.sleep(3600)  # Check every hour
                
                # Check if rebalancing is needed
                master_nodes = [n for n in self.nodes.values() if n.role == ClusterRole.MASTER]
                if len(master_nodes) < 2:
                    continue
                
                # Calculate slot distribution variance
                slot_counts = [len(node.slots) for node in master_nodes]
                if not slot_counts:
                    continue
                
                avg_slots = sum(slot_counts) / len(slot_counts)
                max_deviation = max(abs(count - avg_slots) / avg_slots for count in slot_counts)
                
                if max_deviation > self.rebalance_threshold:
                    logger.info(f"Slot imbalance detected ({max_deviation:.1%}), planning rebalance")
                    
                    # Plan rebalancing
                    rebalance_plan = await self.rebalance_cluster(dry_run=True)
                    
                    if rebalance_plan.get('total_moves', 0) > 0:
                        logger.info(f"Rebalance plan: {rebalance_plan['total_moves']} slot moves")
                        
                        # Execute rebalancing during low-traffic periods
                        # This would typically check current load before executing
                        current_hour = datetime.now().hour
                        if 2 <= current_hour <= 5:  # Low traffic hours (2-5 AM)
                            logger.info("Executing automatic rebalancing")
                            await self.rebalance_cluster(dry_run=False)
                
            except Exception as e:
                logger.error(f"Rebalancing loop error: {e}")
                await asyncio.sleep(1800)  # Wait 30 minutes on error
    
    async def cleanup(self):
        """Clean up cluster resources."""
        if self.monitoring_task:
            self.monitoring_task.cancel()
        
        if self.rebalancing_task:
            self.rebalancing_task.cancel()
        
        if self.async_cluster_client:
            await self.async_cluster_client.close()
        
        if self.cluster_client:
            self.cluster_client.close()


class RedisMemoryOptimizer:
    """
    Advanced Redis memory optimization for financial data.
    """
    
    def __init__(self):
        self.optimization_history = []
        
    async def optimize_node_memory(
        self,
        redis_client,
        strategy: MemoryOptimizationStrategy = MemoryOptimizationStrategy.BALANCED
    ) -> Dict[str, Any]:
        """Optimize memory for a single Redis node."""
        optimization_result = {
            'strategy': strategy.value,
            'initial_memory_mb': 0.0,
            'final_memory_mb': 0.0,
            'memory_saved_mb': 0.0,
            'optimizations_applied': []
        }
        
        try:
            # Get initial memory usage
            info = redis_client.info('memory')
            initial_memory = info.get('used_memory', 0)
            optimization_result['initial_memory_mb'] = initial_memory / (1024 * 1024)
            
            # Apply optimizations based on strategy
            if strategy == MemoryOptimizationStrategy.AGGRESSIVE:
                optimizations = await self._apply_aggressive_optimizations(redis_client)
            elif strategy == MemoryOptimizationStrategy.PERFORMANCE:
                optimizations = await self._apply_performance_optimizations(redis_client)
            else:  # BALANCED
                optimizations = await self._apply_balanced_optimizations(redis_client)
            
            optimization_result['optimizations_applied'] = optimizations
            
            # Get final memory usage
            info = redis_client.info('memory')
            final_memory = info.get('used_memory', 0)
            optimization_result['final_memory_mb'] = final_memory / (1024 * 1024)
            optimization_result['memory_saved_mb'] = (initial_memory - final_memory) / (1024 * 1024)
            
            # Store in history
            self.optimization_history.append({
                'timestamp': time.time(),
                'strategy': strategy.value,
                'memory_saved_mb': optimization_result['memory_saved_mb']
            })
            
        except Exception as e:
            logger.error(f"Memory optimization failed: {e}")
            optimization_result['error'] = str(e)
        
        return optimization_result
    
    async def _apply_aggressive_optimizations(self, redis_client) -> List[str]:
        """Apply aggressive memory optimizations."""
        optimizations = []
        
        try:
            # Set aggressive memory policy
            redis_client.config_set('maxmemory-policy', 'allkeys-lru')
            optimizations.append('Set aggressive eviction policy')
            
            # Optimize data structures for memory
            redis_client.config_set('hash-max-ziplist-entries', 512)
            redis_client.config_set('hash-max-ziplist-value', 1024)
            redis_client.config_set('list-max-ziplist-size', -1)
            redis_client.config_set('set-max-intset-entries', 512)
            redis_client.config_set('zset-max-ziplist-entries', 128)
            optimizations.append('Optimized data structures for memory')
            
            # Enable aggressive lazy freeing
            redis_client.config_set('lazyfree-lazy-eviction', 'yes')
            redis_client.config_set('lazyfree-lazy-expire', 'yes')
            redis_client.config_set('lazyfree-lazy-server-del', 'yes')
            optimizations.append('Enabled aggressive lazy freeing')
            
            # Clean up expired keys more aggressively
            redis_client.config_set('hz', 100)
            optimizations.append('Increased cleanup frequency')
            
        except Exception as e:
            logger.error(f"Aggressive optimization failed: {e}")
            optimizations.append(f'Error: {e}')
        
        return optimizations
    
    async def _apply_balanced_optimizations(self, redis_client) -> List[str]:
        """Apply balanced memory optimizations."""
        optimizations = []
        
        try:
            # Set balanced memory policy
            redis_client.config_set('maxmemory-policy', 'allkeys-lfu')
            optimizations.append('Set balanced eviction policy')
            
            # Optimize data structures moderately
            redis_client.config_set('hash-max-ziplist-entries', 2048)
            redis_client.config_set('hash-max-ziplist-value', 4096)
            redis_client.config_set('list-max-ziplist-size', -2)
            redis_client.config_set('set-max-intset-entries', 2048)
            redis_client.config_set('zset-max-ziplist-entries', 512)
            optimizations.append('Optimized data structures for balance')
            
            # Enable moderate lazy freeing
            redis_client.config_set('lazyfree-lazy-eviction', 'yes')
            redis_client.config_set('lazyfree-lazy-expire', 'yes')
            optimizations.append('Enabled balanced lazy freeing')
            
            # Moderate cleanup frequency
            redis_client.config_set('hz', 50)
            optimizations.append('Set moderate cleanup frequency')
            
        except Exception as e:
            logger.error(f"Balanced optimization failed: {e}")
            optimizations.append(f'Error: {e}')
        
        return optimizations
    
    async def _apply_performance_optimizations(self, redis_client) -> List[str]:
        """Apply performance-focused optimizations."""
        optimizations = []
        
        try:
            # Set performance-oriented memory policy
            redis_client.config_set('maxmemory-policy', 'volatile-lru')
            optimizations.append('Set performance-oriented eviction policy')
            
            # Optimize data structures for performance
            redis_client.config_set('hash-max-ziplist-entries', 5000)
            redis_client.config_set('hash-max-ziplist-value', 8192)
            redis_client.config_set('list-max-ziplist-size', -3)
            redis_client.config_set('set-max-intset-entries', 5000)
            redis_client.config_set('zset-max-ziplist-entries', 2000)
            optimizations.append('Optimized data structures for performance')
            
            # Minimal lazy freeing for performance
            redis_client.config_set('lazyfree-lazy-eviction', 'no')
            redis_client.config_set('lazyfree-lazy-expire', 'yes')
            optimizations.append('Configured minimal lazy freeing')
            
            # Lower cleanup frequency for better performance
            redis_client.config_set('hz', 10)
            optimizations.append('Set low cleanup frequency for performance')
            
        except Exception as e:
            logger.error(f"Performance optimization failed: {e}")
            optimizations.append(f'Error: {e}')
        
        return optimizations
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization history."""
        if not self.optimization_history:
            return {'message': 'No optimizations performed yet'}
        
        recent_optimizations = self.optimization_history[-10:]  # Last 10
        
        total_memory_saved = sum(opt['memory_saved_mb'] for opt in recent_optimizations)
        strategies_used = list(set(opt['strategy'] for opt in recent_optimizations))
        
        return {
            'total_optimizations': len(self.optimization_history),
            'recent_optimizations': len(recent_optimizations),
            'total_memory_saved_mb': total_memory_saved,
            'strategies_used': strategies_used,
            'last_optimization': datetime.fromtimestamp(recent_optimizations[-1]['timestamp']).isoformat(),
            'average_savings_mb': total_memory_saved / len(recent_optimizations)
        }


class ClusterFinancialDataOptimizer:
    """
    Financial data-specific cluster optimizations.
    """
    
    def __init__(self, cluster_manager: RedisClusterManager):
        self.cluster_manager = cluster_manager
        
    async def optimize_for_stock_data(self) -> Dict[str, Any]:
        """Optimize cluster specifically for stock data workloads."""
        optimizations = {
            'shard_optimization': await self._optimize_stock_sharding(),
            'memory_optimization': await self._optimize_stock_memory_usage(),
            'performance_optimization': await self._optimize_stock_performance()
        }
        
        return optimizations
    
    async def _optimize_stock_sharding(self) -> Dict[str, Any]:
        """Optimize how stocks are sharded across cluster."""
        # Analyze current stock distribution
        stock_distribution = {}
        
        for node_id, node in self.cluster_manager.nodes.items():
            if node.role == ClusterRole.MASTER:
                try:
                    node_client = redis.Redis(host=node.host, port=node.port)
                    
                    # Sample some keys to understand distribution
                    sample_keys = node_client.randomkey() or []
                    if sample_keys:
                        # Count stock-related keys
                        stock_keys = 0
                        for i in range(min(100, len(sample_keys))):
                            key = node_client.randomkey()
                            if key and ':' in key and len(key.split(':')[0]) <= 5:  # Likely stock symbol
                                stock_keys += 1
                        
                        stock_distribution[node_id] = {
                            'estimated_stock_keys': stock_keys,
                            'total_keys': node_client.dbsize()
                        }
                
                except Exception as e:
                    logger.error(f"Failed to analyze node {node_id}: {e}")
        
        return {
            'current_distribution': stock_distribution,
            'recommendations': self._generate_sharding_recommendations(stock_distribution)
        }
    
    async def _optimize_stock_memory_usage(self) -> Dict[str, Any]:
        """Optimize memory usage for stock data."""
        memory_optimizations = []
        
        for node in self.cluster_manager.nodes.values():
            if node.role == ClusterRole.MASTER:
                try:
                    node_client = redis.Redis(host=node.host, port=node.port)
                    
                    # Stock-specific memory optimizations
                    optimizations_applied = []
                    
                    # Optimize hash settings for stock data
                    node_client.config_set('hash-max-ziplist-entries', 10000)  # Good for stock data
                    node_client.config_set('hash-max-ziplist-value', 8192)
                    optimizations_applied.append('Optimized hash settings for stock data')
                    
                    # Optimize sorted sets for time series
                    node_client.config_set('zset-max-ziplist-entries', 5000)
                    optimizations_applied.append('Optimized sorted sets for time series')
                    
                    # Set appropriate eviction for financial data
                    node_client.config_set('maxmemory-policy', 'volatile-lru')  # Keep important data
                    optimizations_applied.append('Set financial data eviction policy')
                    
                    memory_optimizations.append({
                        'node': f"{node.host}:{node.port}",
                        'optimizations': optimizations_applied
                    })
                    
                except Exception as e:
                    logger.error(f"Memory optimization failed for node {node.node_id}: {e}")
        
        return {'node_optimizations': memory_optimizations}
    
    async def _optimize_stock_performance(self) -> Dict[str, Any]:
        """Optimize cluster performance for stock data access patterns."""
        performance_optimizations = []
        
        for node in self.cluster_manager.nodes.values():
            if node.role == ClusterRole.MASTER:
                try:
                    node_client = redis.Redis(host=node.host, port=node.port)
                    
                    optimizations_applied = []
                    
                    # Enable I/O threading for better throughput
                    try:
                        node_client.config_set('io-threads', 4)
                        node_client.config_set('io-threads-do-reads', 'yes')
                        optimizations_applied.append('Enabled I/O threading')
                    except:
                        pass  # Might not be supported in all versions
                    
                    # Optimize for financial data access patterns
                    node_client.config_set('hz', 50)  # Good balance for stock data
                    optimizations_applied.append('Optimized background task frequency')
                    
                    # Enable lazy freeing for better performance
                    node_client.config_set('lazyfree-lazy-server-del', 'yes')
                    optimizations_applied.append('Enabled lazy freeing')
                    
                    performance_optimizations.append({
                        'node': f"{node.host}:{node.port}",
                        'optimizations': optimizations_applied
                    })
                    
                except Exception as e:
                    logger.error(f"Performance optimization failed for node {node.node_id}: {e}")
        
        return {'node_optimizations': performance_optimizations}
    
    def _generate_sharding_recommendations(self, distribution: Dict[str, Any]) -> List[str]:
        """Generate recommendations for better stock data sharding."""
        recommendations = []
        
        if not distribution:
            return ['No distribution data available']
        
        # Analyze distribution uniformity
        key_counts = [node_data.get('total_keys', 0) for node_data in distribution.values()]
        if key_counts:
            avg_keys = sum(key_counts) / len(key_counts)
            max_deviation = max(abs(count - avg_keys) / avg_keys for count in key_counts if avg_keys > 0)
            
            if max_deviation > 0.3:  # 30% deviation
                recommendations.append('Consider rebalancing cluster - uneven key distribution detected')
            
            if max(key_counts) > 1000000:  # 1M keys
                recommendations.append('Consider adding more master nodes - some nodes have high key counts')
        
        return recommendations or ['Distribution appears balanced']


# Global cluster manager instance
redis_cluster_manager: Optional[RedisClusterManager] = None


async def initialize_redis_cluster(startup_nodes: List[Dict[str, Any]]) -> bool:
    """Initialize Redis cluster with optimizations."""
    global redis_cluster_manager
    
    try:
        redis_cluster_manager = RedisClusterManager(startup_nodes)
        success = await redis_cluster_manager.initialize()
        
        if success:
            # Apply financial data optimizations
            financial_optimizer = ClusterFinancialDataOptimizer(redis_cluster_manager)
            await financial_optimizer.optimize_for_stock_data()
            
            logger.info("Redis cluster initialized and optimized for financial data")
        
        return success
        
    except Exception as e:
        logger.error(f"Failed to initialize Redis cluster: {e}")
        return False


async def cleanup_redis_cluster():
    """Cleanup Redis cluster resources."""
    global redis_cluster_manager
    
    if redis_cluster_manager:
        await redis_cluster_manager.cleanup()
        redis_cluster_manager = None