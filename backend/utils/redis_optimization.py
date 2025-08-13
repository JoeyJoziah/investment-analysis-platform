"""
Redis Configuration Optimization for Financial Data Workloads
Specialized Redis configuration and optimization for high-performance
financial data caching with 6000+ stocks processing requirements.
"""

import asyncio
import logging
import time
import json
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import redis.asyncio as aioredis
import redis
from redis.exceptions import RedisError
import yaml

logger = logging.getLogger(__name__)


class RedisWorkloadType(Enum):
    """Redis workload types for optimization."""
    HIGH_FREQUENCY_TRADING = "hft"
    MARKET_DATA = "market_data"
    ANALYTICS = "analytics"
    TIME_SERIES = "time_series"
    MIXED = "mixed"


class MemoryPolicy(Enum):
    """Redis memory eviction policies."""
    ALLKEYS_LRU = "allkeys-lru"
    ALLKEYS_LFU = "allkeys-lfu"
    VOLATILE_LRU = "volatile-lru"
    VOLATILE_LFU = "volatile-lfu"
    ALLKEYS_RANDOM = "allkeys-random"
    VOLATILE_RANDOM = "volatile-random"
    VOLATILE_TTL = "volatile-ttl"
    NO_EVICTION = "noeviction"


@dataclass
class RedisOptimizationConfig:
    """Redis optimization configuration."""
    workload_type: RedisWorkloadType = RedisWorkloadType.MIXED
    max_memory_gb: float = 8.0
    max_clients: int = 10000
    enable_clustering: bool = False
    enable_persistence: bool = True
    enable_compression: bool = True
    
    # Performance settings
    tcp_keepalive: int = 300
    timeout: int = 0  # 0 means no timeout
    tcp_backlog: int = 511
    databases: int = 16
    
    # Memory optimization
    memory_policy: MemoryPolicy = MemoryPolicy.ALLKEYS_LFU
    memory_samples: int = 10
    lazyfree_lazy_eviction: bool = True
    lazyfree_lazy_expire: bool = True
    
    # Persistence settings
    save_intervals: List[Tuple[int, int]] = None  # [(seconds, changes)]
    rdb_compression: bool = True
    rdb_checksum: bool = True
    
    # AOF settings
    appendonly: bool = True
    appendfsync: str = "everysec"  # always, everysec, no
    no_appendfsync_on_rewrite: bool = False
    
    # Financial data specific
    hash_max_ziplist_entries: int = 10000  # Increased for stock data
    hash_max_ziplist_value: int = 8192     # Increased for market data
    set_max_intset_entries: int = 10000    # Increased for stock lists
    zset_max_ziplist_entries: int = 5000   # For time series data
    
    def __post_init__(self):
        """Initialize default save intervals."""
        if self.save_intervals is None:
            # Optimized for financial data
            self.save_intervals = [
                (900, 1),    # 15 minutes if at least 1 change
                (300, 10),   # 5 minutes if at least 10 changes
                (60, 10000)  # 1 minute if at least 10000 changes
            ]


class RedisConfigGenerator:
    """
    Generate optimized Redis configuration for financial workloads.
    """
    
    def __init__(self, config: RedisOptimizationConfig):
        self.config = config
        
    def generate_redis_conf(self) -> Dict[str, Any]:
        """Generate Redis configuration dictionary."""
        redis_config = {}
        
        # Basic server settings
        redis_config.update(self._get_server_config())
        
        # Memory settings
        redis_config.update(self._get_memory_config())
        
        # Network settings
        redis_config.update(self._get_network_config())
        
        # Persistence settings
        redis_config.update(self._get_persistence_config())
        
        # Data structure optimization
        redis_config.update(self._get_data_structure_config())
        
        # Logging and monitoring
        redis_config.update(self._get_logging_config())
        
        # Financial data specific optimizations
        redis_config.update(self._get_financial_optimizations())
        
        return redis_config
    
    def _get_server_config(self) -> Dict[str, Any]:
        """Get server configuration."""
        return {
            'bind': '0.0.0.0',
            'port': 6379,
            'timeout': self.config.timeout,
            'tcp-keepalive': self.config.tcp_keepalive,
            'tcp-backlog': self.config.tcp_backlog,
            'databases': self.config.databases,
            'maxclients': self.config.max_clients,
            
            # Process management
            'daemonize': 'yes',
            'pidfile': '/var/run/redis/redis-server.pid',
            
            # Directory
            'dir': '/var/lib/redis',
            
            # Disable dangerous commands
            'rename-command': [
                'FLUSHDB ""',
                'FLUSHALL ""',
                'DEBUG ""'
            ]
        }
    
    def _get_memory_config(self) -> Dict[str, Any]:
        """Get memory configuration."""
        max_memory_bytes = int(self.config.max_memory_gb * 1024 * 1024 * 1024)
        
        config = {
            'maxmemory': f'{max_memory_bytes}',
            'maxmemory-policy': self.config.memory_policy.value,
            'maxmemory-samples': self.config.memory_samples,
            
            # Lazy freeing (non-blocking deletion)
            'lazyfree-lazy-eviction': 'yes' if self.config.lazyfree_lazy_eviction else 'no',
            'lazyfree-lazy-expire': 'yes' if self.config.lazyfree_lazy_expire else 'no',
            'lazyfree-lazy-server-del': 'yes',
            'replica-lazy-flush': 'yes',
        }
        
        # Memory-specific optimizations for financial data
        if self.config.workload_type in [RedisWorkloadType.HIGH_FREQUENCY_TRADING, RedisWorkloadType.MARKET_DATA]:
            config.update({
                # More aggressive memory management for HFT
                'maxmemory-samples': 10,
                'hash-max-ziplist-entries': 2048,  # Smaller for HFT
                'hash-max-ziplist-value': 2048,
            })
        
        return config
    
    def _get_network_config(self) -> Dict[str, Any]:
        """Get network configuration optimized for financial data."""
        config = {
            # Network optimization
            'tcp-nodelay': 'yes',  # Disable Nagle's algorithm for low latency
            
            # Connection limits
            'maxclients': self.config.max_clients,
            
            # Buffer sizes (optimized for financial data throughput)
            'client-output-buffer-limit': [
                'normal 256mb 128mb 60',      # Normal clients
                'replica 512mb 256mb 60',     # Replica clients  
                'pubsub 64mb 32mb 60'         # Pub/sub clients
            ],
            
            # Timeout settings
            'timeout': self.config.timeout,
            'tcp-keepalive': self.config.tcp_keepalive,
        }
        
        # HFT-specific network optimizations
        if self.config.workload_type == RedisWorkloadType.HIGH_FREQUENCY_TRADING:
            config.update({
                'timeout': 0,  # No timeout for HFT
                'tcp-nodelay': 'yes',
                'tcp-keepalive': 60,  # Shorter keepalive for HFT
            })
        
        return config
    
    def _get_persistence_config(self) -> Dict[str, Any]:
        """Get persistence configuration."""
        config = {}
        
        if self.config.enable_persistence:
            # RDB configuration
            save_rules = []
            for seconds, changes in self.config.save_intervals:
                save_rules.append(f'{seconds} {changes}')
            
            config.update({
                'save': save_rules,
                'rdbcompression': 'yes' if self.config.rdb_compression else 'no',
                'rdbchecksum': 'yes' if self.config.rdb_checksum else 'no',
                'dbfilename': 'financial_data.rdb',
                
                # Background save optimization
                'stop-writes-on-bgsave-error': 'no',  # Continue writes even if save fails
            })
            
            # AOF configuration
            if self.config.appendonly:
                config.update({
                    'appendonly': 'yes',
                    'appendfilename': 'financial_data.aof',
                    'appendfsync': self.config.appendfsync,
                    'no-appendfsync-on-rewrite': 'yes' if self.config.no_appendfsync_on_rewrite else 'no',
                    'auto-aof-rewrite-percentage': 100,
                    'auto-aof-rewrite-min-size': '64mb',
                })
        else:
            config.update({
                'save': '',  # Disable RDB
                'appendonly': 'no'  # Disable AOF
            })
        
        # Workload-specific persistence
        if self.config.workload_type == RedisWorkloadType.HIGH_FREQUENCY_TRADING:
            # Minimal persistence for HFT
            config.update({
                'save': ['3600 1'],  # Only save every hour if any changes
                'appendfsync': 'no'   # No fsync for maximum performance
            })
        elif self.config.workload_type == RedisWorkloadType.TIME_SERIES:
            # More frequent saves for time series
            config.update({
                'save': ['300 10', '60 1000'],  # More frequent saves
                'appendfsync': 'everysec'
            })
        
        return config
    
    def _get_data_structure_config(self) -> Dict[str, Any]:
        """Get data structure optimization configuration."""
        return {
            # Hash optimization (for stock data)
            'hash-max-ziplist-entries': self.config.hash_max_ziplist_entries,
            'hash-max-ziplist-value': self.config.hash_max_ziplist_value,
            
            # List optimization
            'list-max-ziplist-size': -2,    # 8KB per node
            'list-compress-depth': 1,       # Compress all but head/tail
            
            # Set optimization (for stock symbols)
            'set-max-intset-entries': self.config.set_max_intset_entries,
            
            # Sorted set optimization (for time series)
            'zset-max-ziplist-entries': self.config.zset_max_ziplist_entries,
            'zset-max-ziplist-value': 64,
            
            # HyperLogLog optimization
            'hll-sparse-max-bytes': 3000,
            
            # Stream optimization (for market data streams)
            'stream-node-max-bytes': 4096,
            'stream-node-max-entries': 100,
        }
    
    def _get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return {
            'loglevel': 'notice',
            'logfile': '/var/log/redis/redis-server.log',
            'syslog-enabled': 'yes',
            'syslog-ident': 'redis-financial',
            
            # Slow log (important for financial data)
            'slowlog-log-slower-than': 1000,  # 1ms threshold
            'slowlog-max-len': 1000,
            
            # Latency monitoring
            'latency-monitor-threshold': 100,  # 100ms threshold
        }
    
    def _get_financial_optimizations(self) -> Dict[str, Any]:
        """Get financial data specific optimizations."""
        config = {
            # CPU optimizations
            'dynamic-hz': 'yes',  # Adaptive background task frequency
            'hz': 50,  # Higher frequency for financial data
            
            # I/O optimizations
            'io-threads': 4,  # Multi-threaded I/O
            'io-threads-do-reads': 'yes',
            
            # Active rehashing for consistent performance
            'activerehashing': 'yes',
            
            # Jemalloc optimization
            'jemalloc-bg-thread': 'yes',
            
            # Disable protected mode for internal network
            'protected-mode': 'no',
        }
        
        # Workload-specific optimizations
        if self.config.workload_type == RedisWorkloadType.HIGH_FREQUENCY_TRADING:
            config.update({
                'hz': 100,  # Higher frequency for HFT
                'dynamic-hz': 'no',  # Fixed frequency for consistency
                'io-threads': 8,  # More I/O threads for HFT
            })
        
        return config
    
    def generate_redis_conf_file(self, filepath: str) -> bool:
        """Generate Redis configuration file."""
        try:
            config = self.generate_redis_conf()
            
            with open(filepath, 'w') as f:
                f.write("# Redis configuration for financial data workloads\n")
                f.write(f"# Generated on {datetime.now().isoformat()}\n")
                f.write(f"# Workload type: {self.config.workload_type.value}\n\n")
                
                for key, value in config.items():
                    if isinstance(value, list):
                        for item in value:
                            f.write(f"{key} {item}\n")
                    else:
                        f.write(f"{key} {value}\n")
                
                f.write("\n# End of configuration\n")
            
            logger.info(f"Redis configuration written to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to write Redis configuration: {e}")
            return False


class RedisPerformanceTuner:
    """
    Dynamic Redis performance tuning based on workload patterns.
    """
    
    def __init__(self, redis_client):
        self.redis_client = redis_client
        self.performance_history = []
        self.current_config = {}
        self.tuning_enabled = True
        
        # Performance thresholds
        self.latency_threshold_ms = 5.0
        self.throughput_threshold_ops = 10000
        self.memory_usage_threshold = 0.85
        
    async def analyze_performance(self) -> Dict[str, Any]:
        """Analyze current Redis performance."""
        try:
            # Get Redis info
            info = await self.redis_client.info()
            
            # Calculate key metrics
            metrics = {
                'timestamp': time.time(),
                'connected_clients': info.get('connected_clients', 0),
                'used_memory_mb': info.get('used_memory', 0) / (1024 * 1024),
                'used_memory_peak_mb': info.get('used_memory_peak', 0) / (1024 * 1024),
                'total_commands_processed': info.get('total_commands_processed', 0),
                'instantaneous_ops_per_sec': info.get('instantaneous_ops_per_sec', 0),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0),
                'expired_keys': info.get('expired_keys', 0),
                'evicted_keys': info.get('evicted_keys', 0),
            }
            
            # Calculate derived metrics
            total_requests = metrics['keyspace_hits'] + metrics['keyspace_misses']
            if total_requests > 0:
                metrics['hit_rate'] = metrics['keyspace_hits'] / total_requests
            else:
                metrics['hit_rate'] = 0.0
            
            # Get latency information
            try:
                latency_info = await self.redis_client.execute_command('LATENCY', 'LATEST')
                if latency_info:
                    metrics['avg_latency_ms'] = latency_info[0][2] if latency_info[0] else 0
                else:
                    metrics['avg_latency_ms'] = 0
            except:
                metrics['avg_latency_ms'] = 0
            
            # Store in history
            self.performance_history.append(metrics)
            
            # Keep only recent history
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-500:]
            
            return metrics
            
        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
            return {}
    
    async def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get performance optimization recommendations."""
        recommendations = []
        
        if not self.performance_history:
            return recommendations
        
        latest_metrics = self.performance_history[-1]
        
        # Check memory usage
        if 'used_memory_mb' in latest_metrics and 'used_memory_peak_mb' in latest_metrics:
            memory_usage_ratio = latest_metrics['used_memory_mb'] / max(latest_metrics['used_memory_peak_mb'], 1)
            
            if memory_usage_ratio > self.memory_usage_threshold:
                recommendations.append({
                    'type': 'memory_optimization',
                    'priority': 'high',
                    'description': f'Memory usage is high ({memory_usage_ratio:.1%})',
                    'recommendation': 'Consider increasing maxmemory or enabling more aggressive eviction',
                    'config_changes': {
                        'maxmemory-policy': 'allkeys-lfu',
                        'maxmemory-samples': 10
                    }
                })
        
        # Check hit rate
        if latest_metrics.get('hit_rate', 1.0) < 0.8:
            recommendations.append({
                'type': 'cache_efficiency',
                'priority': 'medium',
                'description': f'Cache hit rate is low ({latest_metrics["hit_rate"]:.1%})',
                'recommendation': 'Review caching strategy and TTL values',
                'config_changes': {
                    'maxmemory-policy': 'allkeys-lru'
                }
            })
        
        # Check latency
        if latest_metrics.get('avg_latency_ms', 0) > self.latency_threshold_ms:
            recommendations.append({
                'type': 'latency_optimization',
                'priority': 'high',
                'description': f'Average latency is high ({latest_metrics["avg_latency_ms"]:.1f}ms)',
                'recommendation': 'Enable lazy freeing and optimize data structures',
                'config_changes': {
                    'lazyfree-lazy-eviction': 'yes',
                    'lazyfree-lazy-expire': 'yes',
                    'io-threads': 4
                }
            })
        
        # Check throughput
        if latest_metrics.get('instantaneous_ops_per_sec', 0) < self.throughput_threshold_ops:
            recommendations.append({
                'type': 'throughput_optimization',
                'priority': 'medium',
                'description': f'Throughput is low ({latest_metrics["instantaneous_ops_per_sec"]} ops/sec)',
                'recommendation': 'Increase I/O threads and optimize persistence',
                'config_changes': {
                    'io-threads': 6,
                    'appendfsync': 'everysec'
                }
            })
        
        return recommendations
    
    async def apply_optimization(self, optimization: Dict[str, Any]) -> bool:
        """Apply optimization recommendation."""
        if not self.tuning_enabled:
            return False
        
        try:
            config_changes = optimization.get('config_changes', {})
            
            for config_key, config_value in config_changes.items():
                # Apply configuration change
                await self.redis_client.config_set(config_key, config_value)
                logger.info(f"Applied config change: {config_key} = {config_value}")
            
            # Store applied configuration
            self.current_config.update(config_changes)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply optimization: {e}")
            return False
    
    async def auto_tune(self) -> Dict[str, Any]:
        """Perform automatic performance tuning."""
        results = {
            'optimizations_applied': 0,
            'optimizations_failed': 0,
            'recommendations': []
        }
        
        try:
            # Analyze current performance
            await self.analyze_performance()
            
            # Get recommendations
            recommendations = await self.get_optimization_recommendations()
            results['recommendations'] = recommendations
            
            # Apply high-priority optimizations automatically
            for recommendation in recommendations:
                if recommendation.get('priority') == 'high':
                    success = await self.apply_optimization(recommendation)
                    
                    if success:
                        results['optimizations_applied'] += 1
                    else:
                        results['optimizations_failed'] += 1
            
            logger.info(f"Auto-tuning completed: {results['optimizations_applied']} optimizations applied")
            
        except Exception as e:
            logger.error(f"Auto-tuning failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary over specified hours."""
        if not self.performance_history:
            return {}
        
        # Filter to timeframe
        cutoff_time = time.time() - (hours * 3600)
        recent_metrics = [
            m for m in self.performance_history 
            if m.get('timestamp', 0) >= cutoff_time
        ]
        
        if not recent_metrics:
            return {}
        
        # Calculate summary statistics
        summary = {
            'timeframe_hours': hours,
            'data_points': len(recent_metrics),
            'avg_ops_per_sec': sum(m.get('instantaneous_ops_per_sec', 0) for m in recent_metrics) / len(recent_metrics),
            'avg_hit_rate': sum(m.get('hit_rate', 0) for m in recent_metrics) / len(recent_metrics),
            'avg_latency_ms': sum(m.get('avg_latency_ms', 0) for m in recent_metrics) / len(recent_metrics),
            'peak_memory_mb': max(m.get('used_memory_mb', 0) for m in recent_metrics),
            'avg_memory_mb': sum(m.get('used_memory_mb', 0) for m in recent_metrics) / len(recent_metrics),
        }
        
        return summary


class RedisClusterOptimizer:
    """
    Optimize Redis cluster configuration for financial data.
    """
    
    def __init__(self):
        self.cluster_nodes = []
        self.slot_distribution = {}
        self.replication_factor = 2
        
    def calculate_optimal_cluster_size(
        self,
        total_memory_gb: float,
        expected_keys: int,
        average_key_size_bytes: int
    ) -> Dict[str, Any]:
        """Calculate optimal cluster size for financial data."""
        
        # Estimate total data size
        total_data_size_gb = (expected_keys * average_key_size_bytes) / (1024 ** 3)
        
        # Add overhead for replication and Redis metadata (50%)
        total_size_with_overhead = total_data_size_gb * 1.5 * self.replication_factor
        
        # Calculate number of nodes (with safety margin)
        nodes_needed = max(3, int(total_size_with_overhead / total_memory_gb * 1.2))
        
        # Ensure odd number of master nodes for better failover
        if nodes_needed % 2 == 0:
            nodes_needed += 1
        
        master_nodes = nodes_needed
        replica_nodes = master_nodes * (self.replication_factor - 1)
        total_nodes = master_nodes + replica_nodes
        
        return {
            'recommended_master_nodes': master_nodes,
            'recommended_replica_nodes': replica_nodes,
            'total_nodes': total_nodes,
            'memory_per_node_gb': total_memory_gb,
            'estimated_data_size_gb': total_data_size_gb,
            'cluster_efficiency': total_data_size_gb / (total_nodes * total_memory_gb)
        }
    
    def generate_cluster_config(
        self,
        nodes: List[Dict[str, str]],
        base_config: RedisOptimizationConfig
    ) -> Dict[str, Dict[str, Any]]:
        """Generate cluster configuration for each node."""
        cluster_configs = {}
        
        for i, node in enumerate(nodes):
            node_id = node.get('id', f'node-{i}')
            
            # Base configuration
            config_generator = RedisConfigGenerator(base_config)
            node_config = config_generator.generate_redis_conf()
            
            # Cluster-specific settings
            node_config.update({
                'cluster-enabled': 'yes',
                'cluster-config-file': f'nodes-{node_id}.conf',
                'cluster-node-timeout': 15000,  # 15 seconds
                'cluster-announce-ip': node.get('ip', '127.0.0.1'),
                'cluster-announce-port': int(node.get('port', 6379)),
                'cluster-announce-bus-port': int(node.get('port', 6379)) + 10000,
                
                # Financial data specific cluster settings
                'cluster-require-full-coverage': 'no',  # Allow partial coverage
                'cluster-allow-reads-when-down': 'yes',  # Allow reads during splits
                'cluster-migration-barrier': 1,  # Minimum replicas before migration
            })
            
            cluster_configs[node_id] = node_config
        
        return cluster_configs


# Predefined configurations for different financial workloads
REDIS_CONFIGS = {
    'high_frequency_trading': RedisOptimizationConfig(
        workload_type=RedisWorkloadType.HIGH_FREQUENCY_TRADING,
        max_memory_gb=16.0,
        max_clients=50000,
        enable_persistence=False,  # No persistence for HFT
        memory_policy=MemoryPolicy.ALLKEYS_LRU,
        timeout=0,  # No timeout
        appendonly=False,  # No AOF for maximum performance
        tcp_keepalive=60,
        hash_max_ziplist_entries=512,  # Smaller for HFT
    ),
    
    'market_data': RedisOptimizationConfig(
        workload_type=RedisWorkloadType.MARKET_DATA,
        max_memory_gb=32.0,
        max_clients=20000,
        enable_persistence=True,
        memory_policy=MemoryPolicy.ALLKEYS_LFU,
        appendfsync="everysec",
        hash_max_ziplist_entries=5000,
        zset_max_ziplist_entries=2000,
    ),
    
    'analytics': RedisOptimizationConfig(
        workload_type=RedisWorkloadType.ANALYTICS,
        max_memory_gb=64.0,
        max_clients=5000,
        enable_persistence=True,
        memory_policy=MemoryPolicy.ALLKEYS_LFU,
        appendfsync="always",  # Strong persistence for analytics
        hash_max_ziplist_entries=10000,
        save_intervals=[(300, 10), (60, 1000)],  # More frequent saves
    ),
    
    'time_series': RedisOptimizationConfig(
        workload_type=RedisWorkloadType.TIME_SERIES,
        max_memory_gb=128.0,
        max_clients=10000,
        enable_persistence=True,
        memory_policy=MemoryPolicy.VOLATILE_TTL,  # Good for time series
        zset_max_ziplist_entries=10000,  # Optimized for time series
        hash_max_ziplist_entries=15000,
    )
}


def get_redis_config_for_workload(workload: str) -> RedisOptimizationConfig:
    """Get predefined Redis configuration for workload type."""
    return REDIS_CONFIGS.get(workload, REDIS_CONFIGS['market_data'])


async def optimize_redis_for_financial_data(
    redis_client,
    workload_type: str = 'market_data',
    auto_tune: bool = True
) -> Dict[str, Any]:
    """
    Optimize Redis configuration for financial data workload.
    """
    results = {
        'workload_type': workload_type,
        'optimizations_applied': 0,
        'configuration_generated': False,
        'auto_tuning_enabled': auto_tune
    }
    
    try:
        # Get optimized configuration
        config = get_redis_config_for_workload(workload_type)
        
        # Generate configuration
        config_generator = RedisConfigGenerator(config)
        redis_config = config_generator.generate_redis_conf()
        
        # Write configuration file
        config_file_path = f'/tmp/redis-{workload_type}.conf'
        success = config_generator.generate_redis_conf_file(config_file_path)
        results['configuration_generated'] = success
        results['config_file_path'] = config_file_path
        
        # Initialize performance tuner if auto-tuning enabled
        if auto_tune and redis_client:
            tuner = RedisPerformanceTuner(redis_client)
            tuning_results = await tuner.auto_tune()
            results['tuning_results'] = tuning_results
        
        logger.info(f"Redis optimization completed for {workload_type} workload")
        
    except Exception as e:
        logger.error(f"Redis optimization failed: {e}")
        results['error'] = str(e)
    
    return results


# Global performance tuner instance
redis_performance_tuner = None


async def initialize_redis_optimization(redis_client):
    """Initialize Redis optimization with performance monitoring."""
    global redis_performance_tuner
    
    redis_performance_tuner = RedisPerformanceTuner(redis_client)
    
    # Start background performance monitoring
    asyncio.create_task(performance_monitoring_loop())
    
    logger.info("Redis optimization initialized")
    

async def performance_monitoring_loop():
    """Background loop for Redis performance monitoring."""
    while redis_performance_tuner:
        try:
            await redis_performance_tuner.analyze_performance()
            await asyncio.sleep(60)  # Analyze every minute
        except Exception as e:
            logger.error(f"Performance monitoring error: {e}")
            await asyncio.sleep(300)  # Wait 5 minutes on error