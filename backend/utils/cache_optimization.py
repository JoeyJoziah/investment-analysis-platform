"""
Optimized cache key generation with consistent hashing for distributed caching.
Implements efficient key generation, versioning, and cache partitioning strategies.
"""

import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, date
from enum import Enum


class CacheNamespace(Enum):
    """Cache namespaces for different data types."""
    PRICE = "price"
    FUNDAMENTAL = "fund"
    TECHNICAL = "tech"
    SENTIMENT = "sent"
    ANALYSIS = "anal"
    RECOMMENDATION = "rec"
    PORTFOLIO = "port"
    USER = "user"
    SYSTEM = "sys"


class CacheKeyGenerator:
    """
    Optimized cache key generator with consistent hashing.
    Supports versioning, namespacing, and efficient key distribution.
    """
    
    # Current cache schema version
    CACHE_VERSION = "v2"
    
    # Maximum key length for Redis
    MAX_KEY_LENGTH = 512
    
    # Hash algorithm for consistent hashing
    HASH_ALGORITHM = hashlib.blake2b
    
    def __init__(
        self,
        version: str = CACHE_VERSION,
        use_hashing: bool = True,
        hash_long_keys: bool = True,
        hash_threshold: int = 128
    ):
        """
        Initialize cache key generator.
        
        Args:
            version: Cache version for key invalidation
            use_hashing: Whether to use hashing for keys
            hash_long_keys: Hash keys longer than threshold
            hash_threshold: Length threshold for hashing keys
        """
        self.version = version
        self.use_hashing = use_hashing
        self.hash_long_keys = hash_long_keys
        self.hash_threshold = hash_threshold
        
    def generate_key(
        self,
        namespace: Union[CacheNamespace, str],
        identifier: str,
        params: Optional[Dict[str, Any]] = None,
        date_partition: Optional[Union[datetime, date, str]] = None,
        ttl_bucket: Optional[str] = None
    ) -> str:
        """
        Generate optimized cache key with consistent structure.
        
        Args:
            namespace: Cache namespace (e.g., PRICE, FUNDAMENTAL)
            identifier: Primary identifier (e.g., stock symbol)
            params: Additional parameters for the key
            date_partition: Date for partitioning time-series data
            ttl_bucket: TTL bucket for grouped expiration
            
        Returns:
            Optimized cache key
        """
        # Build key components
        components = []
        
        # Add version
        components.append(self.version)
        
        # Add namespace
        if isinstance(namespace, CacheNamespace):
            components.append(namespace.value)
        else:
            components.append(str(namespace)[:10])  # Limit custom namespace length
        
        # Add date partition if provided
        if date_partition:
            date_str = self._format_date_partition(date_partition)
            components.append(date_str)
        
        # Add TTL bucket if provided
        if ttl_bucket:
            components.append(f"ttl{ttl_bucket}")
        
        # Add identifier (possibly hashed)
        if self.use_hashing and len(identifier) > 32:
            components.append(self._hash_identifier(identifier))
        else:
            components.append(identifier)
        
        # Add params hash if provided
        if params:
            params_hash = self._hash_params(params)
            components.append(params_hash)
        
        # Join components
        key = ":".join(components)
        
        # Hash if too long
        if self.hash_long_keys and len(key) > self.hash_threshold:
            # Keep prefix for debugging, hash the rest
            prefix = ":".join(components[:3])
            key_hash = self._hash_string(key)
            key = f"{prefix}:{key_hash}"
        
        # Ensure key doesn't exceed maximum length
        if len(key) > self.MAX_KEY_LENGTH:
            key = self._truncate_key(key)
        
        return key
    
    def generate_pattern(
        self,
        namespace: Optional[Union[CacheNamespace, str]] = None,
        identifier: Optional[str] = None,
        date_partition: Optional[Union[datetime, date, str]] = None
    ) -> str:
        """
        Generate pattern for cache key matching (e.g., for deletion).
        
        Args:
            namespace: Optional namespace filter
            identifier: Optional identifier filter
            date_partition: Optional date partition filter
            
        Returns:
            Pattern string for Redis SCAN/KEYS operations
        """
        components = [self.version]
        
        if namespace:
            if isinstance(namespace, CacheNamespace):
                components.append(namespace.value)
            else:
                components.append(str(namespace)[:10])
        else:
            components.append("*")
        
        if date_partition:
            date_str = self._format_date_partition(date_partition)
            components.append(date_str)
        else:
            components.append("*")
        
        if identifier:
            if self.use_hashing and len(identifier) > 32:
                components.append(self._hash_identifier(identifier))
            else:
                components.append(identifier)
        else:
            components.append("*")
        
        # Add wildcard for params
        components.append("*")
        
        return ":".join(components)
    
    def generate_batch_keys(
        self,
        namespace: Union[CacheNamespace, str],
        identifiers: List[str],
        params: Optional[Dict[str, Any]] = None,
        date_partition: Optional[Union[datetime, date, str]] = None
    ) -> Dict[str, str]:
        """
        Generate multiple cache keys efficiently.
        
        Args:
            namespace: Cache namespace
            identifiers: List of identifiers
            params: Common parameters for all keys
            date_partition: Common date partition
            
        Returns:
            Dictionary mapping identifiers to cache keys
        """
        keys = {}
        
        # Pre-compute common components
        common_prefix = self._get_common_prefix(namespace, date_partition)
        params_hash = self._hash_params(params) if params else None
        
        for identifier in identifiers:
            # Build key efficiently
            if self.use_hashing and len(identifier) > 32:
                id_component = self._hash_identifier(identifier)
            else:
                id_component = identifier
            
            components = [common_prefix, id_component]
            if params_hash:
                components.append(params_hash)
            
            key = ":".join(components)
            
            # Apply length constraints
            if self.hash_long_keys and len(key) > self.hash_threshold:
                key = self._apply_length_constraint(key, common_prefix)
            
            keys[identifier] = key
        
        return keys
    
    def parse_key(self, key: str) -> Dict[str, Any]:
        """
        Parse cache key to extract components.
        
        Args:
            key: Cache key to parse
            
        Returns:
            Dictionary with parsed components
        """
        components = key.split(":")
        result = {}
        
        if len(components) >= 1:
            result['version'] = components[0]
        
        if len(components) >= 2:
            result['namespace'] = components[1]
        
        if len(components) >= 3:
            # Check if it's a date partition
            if components[2].startswith('20'):  # Assuming years 20xx
                result['date_partition'] = components[2]
                if len(components) >= 4:
                    result['identifier'] = components[3]
            else:
                result['identifier'] = components[2]
        
        return result
    
    def _format_date_partition(
        self,
        date_partition: Union[datetime, date, str]
    ) -> str:
        """Format date partition consistently."""
        if isinstance(date_partition, datetime):
            return date_partition.strftime('%Y%m%d')
        elif isinstance(date_partition, date):
            return date_partition.strftime('%Y%m%d')
        elif isinstance(date_partition, str):
            # Assume already formatted or parse
            return date_partition[:8]  # Take first 8 chars
        else:
            return "00000000"
    
    def _hash_identifier(self, identifier: str) -> str:
        """Hash identifier for consistent length."""
        hash_obj = self.HASH_ALGORITHM(digest_size=16)
        hash_obj.update(identifier.encode('utf-8'))
        return hash_obj.hexdigest()[:32]
    
    def _hash_params(self, params: Dict[str, Any]) -> str:
        """Hash parameters deterministically."""
        # Sort keys for consistent hashing
        sorted_params = json.dumps(params, sort_keys=True, default=str)
        
        hash_obj = self.HASH_ALGORITHM(digest_size=8)
        hash_obj.update(sorted_params.encode('utf-8'))
        return hash_obj.hexdigest()[:16]
    
    def _hash_string(self, s: str) -> str:
        """Hash arbitrary string."""
        hash_obj = self.HASH_ALGORITHM(digest_size=16)
        hash_obj.update(s.encode('utf-8'))
        return hash_obj.hexdigest()
    
    def _get_common_prefix(
        self,
        namespace: Union[CacheNamespace, str],
        date_partition: Optional[Union[datetime, date, str]] = None
    ) -> str:
        """Get common prefix for batch operations."""
        components = [self.version]
        
        if isinstance(namespace, CacheNamespace):
            components.append(namespace.value)
        else:
            components.append(str(namespace)[:10])
        
        if date_partition:
            date_str = self._format_date_partition(date_partition)
            components.append(date_str)
        
        return ":".join(components)
    
    def _apply_length_constraint(self, key: str, prefix: str) -> str:
        """Apply length constraint while preserving prefix."""
        key_hash = self._hash_string(key)
        return f"{prefix}:{key_hash}"
    
    def _truncate_key(self, key: str) -> str:
        """Truncate key to maximum length while preserving uniqueness."""
        if len(key) <= self.MAX_KEY_LENGTH:
            return key
        
        # Keep prefix and suffix, hash middle
        prefix_len = 100
        suffix_len = 20
        
        prefix = key[:prefix_len]
        suffix = key[-suffix_len:]
        middle_hash = self._hash_string(key)[:32]
        
        return f"{prefix}...{middle_hash}...{suffix}"


class ConsistentHashRing:
    """
    Consistent hash ring for distributed cache partitioning.
    Ensures even distribution of keys across cache nodes.
    """
    
    def __init__(self, nodes: List[str], virtual_nodes: int = 150):
        """
        Initialize consistent hash ring.
        
        Args:
            nodes: List of cache node identifiers
            virtual_nodes: Number of virtual nodes per physical node
        """
        self.nodes = nodes
        self.virtual_nodes = virtual_nodes
        self.ring = {}
        self._build_ring()
    
    def _build_ring(self):
        """Build the consistent hash ring."""
        for node in self.nodes:
            for i in range(self.virtual_nodes):
                virtual_key = f"{node}:{i}"
                hash_value = self._hash(virtual_key)
                self.ring[hash_value] = node
        
        self.sorted_keys = sorted(self.ring.keys())
    
    def _hash(self, key: str) -> int:
        """Hash key to integer."""
        hash_obj = hashlib.md5(key.encode('utf-8'))
        return int(hash_obj.hexdigest(), 16)
    
    def get_node(self, key: str) -> str:
        """Get node for given key."""
        if not self.ring:
            return None
        
        hash_value = self._hash(key)
        
        # Find first node with hash >= key hash
        for node_hash in self.sorted_keys:
            if node_hash >= hash_value:
                return self.ring[node_hash]
        
        # Wrap around to first node
        return self.ring[self.sorted_keys[0]]
    
    def add_node(self, node: str):
        """Add node to the ring."""
        self.nodes.append(node)
        for i in range(self.virtual_nodes):
            virtual_key = f"{node}:{i}"
            hash_value = self._hash(virtual_key)
            self.ring[hash_value] = node
        
        self.sorted_keys = sorted(self.ring.keys())
    
    def remove_node(self, node: str):
        """Remove node from the ring."""
        self.nodes.remove(node)
        
        # Remove all virtual nodes
        keys_to_remove = []
        for hash_value, node_name in self.ring.items():
            if node_name == node:
                keys_to_remove.append(hash_value)
        
        for key in keys_to_remove:
            del self.ring[key]
        
        self.sorted_keys = sorted(self.ring.keys())


class CacheKeyOptimizer:
    """
    Optimizer for cache key strategies based on access patterns.
    """
    
    def __init__(self):
        self.access_counts = {}
        self.key_sizes = {}
        self.optimization_suggestions = []
    
    def track_access(self, key: str, size: int = 0):
        """Track key access for optimization."""
        self.access_counts[key] = self.access_counts.get(key, 0) + 1
        if size > 0:
            self.key_sizes[key] = size
    
    def analyze_patterns(self) -> Dict[str, Any]:
        """Analyze access patterns and suggest optimizations."""
        total_accesses = sum(self.access_counts.values())
        
        # Find hot keys
        hot_keys = sorted(
            self.access_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:20]
        
        # Calculate key size statistics
        if self.key_sizes:
            avg_size = sum(self.key_sizes.values()) / len(self.key_sizes)
            max_size = max(self.key_sizes.values())
        else:
            avg_size = 0
            max_size = 0
        
        # Generate suggestions
        suggestions = []
        
        # Check for hot keys
        if hot_keys and hot_keys[0][1] > total_accesses * 0.1:
            suggestions.append({
                'type': 'hot_key',
                'message': f"Key '{hot_keys[0][0]}' accounts for >10% of accesses",
                'recommendation': 'Consider local caching or replication'
            })
        
        # Check for large keys
        large_keys = [k for k, v in self.key_sizes.items() if v > 1024 * 1024]
        if large_keys:
            suggestions.append({
                'type': 'large_keys',
                'message': f"Found {len(large_keys)} keys >1MB",
                'recommendation': 'Consider compression or chunking'
            })
        
        return {
            'total_unique_keys': len(self.access_counts),
            'total_accesses': total_accesses,
            'hot_keys': hot_keys[:10],
            'average_key_size': avg_size,
            'max_key_size': max_size,
            'suggestions': suggestions
        }


# Global cache key generator instance
cache_key_generator = CacheKeyGenerator()


# Helper functions for common use cases
def generate_stock_price_key(
    symbol: str,
    date: Optional[Union[datetime, date, str]] = None,
    interval: str = "1d"
) -> str:
    """Generate cache key for stock price data."""
    params = {'interval': interval}
    return cache_key_generator.generate_key(
        CacheNamespace.PRICE,
        symbol,
        params=params,
        date_partition=date
    )


def generate_fundamental_key(
    symbol: str,
    metric: str,
    period: str = "quarterly"
) -> str:
    """Generate cache key for fundamental data."""
    params = {'metric': metric, 'period': period}
    return cache_key_generator.generate_key(
        CacheNamespace.FUNDAMENTAL,
        symbol,
        params=params
    )


def generate_analysis_key(
    symbol: str,
    analysis_type: str,
    date: Optional[Union[datetime, date, str]] = None
) -> str:
    """Generate cache key for analysis results."""
    params = {'type': analysis_type}
    return cache_key_generator.generate_key(
        CacheNamespace.ANALYSIS,
        symbol,
        params=params,
        date_partition=date
    )