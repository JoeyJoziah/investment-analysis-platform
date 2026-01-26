"""
Cache Hit Rate Optimization System
Advanced strategies to maximize cache hit rates through intelligent key management,
compression, and data locality optimization.
"""

import asyncio
import hashlib
import logging
import lz4.frame
import gzip
import zlib
import json
# SECURITY: Removed pickle import - using JSON for cache serialization
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from datetime import datetime, timedelta
from collections import defaultdict, OrderedDict
from dataclasses import dataclass
from enum import Enum
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading

logger = logging.getLogger(__name__)


class KeyOptimizationStrategy(Enum):
    """Cache key optimization strategies."""
    HIERARCHICAL = "hierarchical"        # ticker:date:type:params
    FLAT_HASHED = "flat_hashed"         # hash of all components
    SEMANTIC = "semantic"               # meaningful key structure
    COMPRESSED = "compressed"           # compressed key representation


class CompressionLevel(Enum):
    """Compression levels for different data types."""
    NONE = 0
    LIGHT = 1      # Fast compression for real-time data
    MEDIUM = 2     # Balanced compression/speed
    HEAVY = 3      # Maximum compression for historical data


@dataclass
class CacheHitMetrics:
    """Cache hit rate metrics."""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    hit_rate: float = 0.0
    average_response_time_ms: float = 0.0
    compression_ratio: float = 0.0
    memory_efficiency: float = 0.0


class IntelligentKeyGenerator:
    """
    Intelligent cache key generator with optimization strategies.
    """
    
    def __init__(self):
        self.key_patterns = defaultdict(int)
        self.access_frequencies = defaultdict(int)
        self.key_performance = {}  # key -> response time
        self.optimization_strategy = KeyOptimizationStrategy.HIERARCHICAL
        self.hash_cache = {}  # Cache for expensive hash calculations
        
        # Key component weights for optimization
        self.component_weights = {
            'ticker': 3.0,     # Most important
            'date': 2.0,       # Time-based grouping
            'data_type': 2.5,  # Data type grouping
            'params': 1.0      # Least important
        }
    
    def generate_optimized_key(
        self,
        ticker: str,
        data_type: str,
        date: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        strategy: Optional[KeyOptimizationStrategy] = None
    ) -> str:
        """
        Generate optimized cache key based on access patterns and strategy.
        """
        strategy = strategy or self.optimization_strategy
        
        # Track access pattern
        pattern = f"{ticker}:{data_type}"
        self.access_frequencies[pattern] += 1
        
        if strategy == KeyOptimizationStrategy.HIERARCHICAL:
            return self._generate_hierarchical_key(ticker, data_type, date, params)
        elif strategy == KeyOptimizationStrategy.FLAT_HASHED:
            return self._generate_hashed_key(ticker, data_type, date, params)
        elif strategy == KeyOptimizationStrategy.SEMANTIC:
            return self._generate_semantic_key(ticker, data_type, date, params)
        elif strategy == KeyOptimizationStrategy.COMPRESSED:
            return self._generate_compressed_key(ticker, data_type, date, params)
        else:
            return self._generate_hierarchical_key(ticker, data_type, date, params)
    
    def _generate_hierarchical_key(
        self,
        ticker: str,
        data_type: str,
        date: Optional[str],
        params: Optional[Dict[str, Any]]
    ) -> str:
        """Generate hierarchical key for better locality."""
        components = []
        
        # Primary grouping by ticker (most important)
        components.append(ticker.upper())
        
        # Secondary grouping by date for time-series data
        if date:
            components.append(self._normalize_date(date))
        
        # Tertiary grouping by data type
        components.append(data_type.lower())
        
        # Parameters as suffix
        if params:
            param_hash = self._hash_params(params)
            components.append(param_hash[:8])  # Short hash for params
        
        return ":".join(components)
    
    def _generate_hashed_key(
        self,
        ticker: str,
        data_type: str,
        date: Optional[str],
        params: Optional[Dict[str, Any]]
    ) -> str:
        """Generate hashed key for consistent length."""
        key_data = f"{ticker}:{data_type}:{date or ''}:{params or ''}"
        
        # Check hash cache first
        if key_data in self.hash_cache:
            return self.hash_cache[key_data]
        
        hash_value = hashlib.blake2b(
            key_data.encode('utf-8'),
            digest_size=16
        ).hexdigest()
        
        self.hash_cache[key_data] = hash_value
        
        # Limit hash cache size
        if len(self.hash_cache) > 10000:
            # Remove oldest entries
            keys_to_remove = list(self.hash_cache.keys())[:1000]
            for key in keys_to_remove:
                del self.hash_cache[key]
        
        return hash_value
    
    def _generate_semantic_key(
        self,
        ticker: str,
        data_type: str,
        date: Optional[str],
        params: Optional[Dict[str, Any]]
    ) -> str:
        """Generate semantic key with meaningful structure."""
        # Use semantic prefixes for better readability and debugging
        prefixes = {
            'price': 'px',
            'fundamental': 'fd',
            'technical': 'ta',
            'news': 'nw',
            'sentiment': 'st'
        }
        
        prefix = prefixes.get(data_type, 'dt')
        
        components = [prefix, ticker.upper()]
        
        if date:
            components.append(self._normalize_date(date))
        
        if params:
            # Add most important parameters to key
            important_params = ['interval', 'period', 'indicator']
            for param in important_params:
                if param in params:
                    components.append(f"{param}:{params[param]}")
        
        return ":".join(components)
    
    def _generate_compressed_key(
        self,
        ticker: str,
        data_type: str,
        date: Optional[str],
        params: Optional[Dict[str, Any]]
    ) -> str:
        """Generate compressed key representation."""
        # Use compressed representation for space efficiency
        full_key = f"{ticker}:{data_type}:{date or ''}:{params or ''}"
        
        # Compress key if it's long
        if len(full_key) > 64:
            compressed = lz4.frame.compress(full_key.encode('utf-8'))
            return hashlib.md5(compressed).hexdigest()
        
        return full_key
    
    def _normalize_date(self, date: str) -> str:
        """Normalize date format for consistent keying."""
        # Handle various date formats
        try:
            if isinstance(date, str):
                if 'T' in date:  # ISO format
                    return date[:10]  # YYYY-MM-DD
                elif len(date) == 8:  # YYYYMMDD
                    return f"{date[:4]}-{date[4:6]}-{date[6:8]}"
            return date
        except Exception:
            return date
    
    def _hash_params(self, params: Dict[str, Any]) -> str:
        """Generate consistent hash for parameters."""
        param_str = json.dumps(params, sort_keys=True, default=str)
        return hashlib.md5(param_str.encode('utf-8')).hexdigest()
    
    def optimize_key_strategy(self) -> KeyOptimizationStrategy:
        """
        Analyze access patterns and optimize key generation strategy.
        """
        # Analyze current performance
        total_accesses = sum(self.access_frequencies.values())
        
        if total_accesses < 1000:
            return KeyOptimizationStrategy.HIERARCHICAL  # Default for low volume
        
        # Calculate locality scores for different strategies
        locality_scores = {}
        
        # Hierarchical strategy score (good for range queries)
        hierarchical_score = self._calculate_hierarchical_locality()
        locality_scores[KeyOptimizationStrategy.HIERARCHICAL] = hierarchical_score
        
        # Hashed strategy score (good for uniform distribution)
        hashed_score = self._calculate_hash_locality()
        locality_scores[KeyOptimizationStrategy.FLAT_HASHED] = hashed_score
        
        # Choose best strategy
        best_strategy = max(locality_scores, key=locality_scores.get)
        
        if best_strategy != self.optimization_strategy:
            logger.info(f"Switching key optimization strategy to {best_strategy}")
            self.optimization_strategy = best_strategy
        
        return best_strategy
    
    def _calculate_hierarchical_locality(self) -> float:
        """Calculate locality score for hierarchical keys."""
        # Score based on ticker grouping efficiency
        ticker_groups = defaultdict(int)
        for pattern in self.access_frequencies:
            ticker = pattern.split(':')[0]
            ticker_groups[ticker] += self.access_frequencies[pattern]
        
        # Higher score for more concentrated access per ticker
        if not ticker_groups:
            return 0.0
        
        total_accesses = sum(ticker_groups.values())
        concentration = sum((count / total_accesses) ** 2 
                          for count in ticker_groups.values())
        
        return concentration * 100  # Scale to 0-100
    
    def _calculate_hash_locality(self) -> float:
        """Calculate locality score for hashed keys."""
        # Hashed keys have uniform distribution (lower locality)
        return 50.0  # Neutral score


class AdaptiveCompressionManager:
    """
    Adaptive compression manager that selects optimal compression
    based on data characteristics and system performance.
    """
    
    def __init__(self):
        self.compression_stats = defaultdict(lambda: {
            'total_size': 0, 'compressed_size': 0, 'time_taken': 0.0, 'count': 0
        })
        self.cpu_threshold = 70.0  # CPU usage threshold for compression
        self.size_threshold = 1024  # Minimum size to compress (bytes)
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Compression algorithms and their characteristics
        self.algorithms = {
            CompressionLevel.LIGHT: {
                'name': 'lz4',
                'compress_func': self._lz4_compress,
                'decompress_func': self._lz4_decompress,
                'speed_ratio': 1.0,
                'compression_ratio': 0.3
            },
            CompressionLevel.MEDIUM: {
                'name': 'zlib',
                'compress_func': self._zlib_compress,
                'decompress_func': self._zlib_decompress,
                'speed_ratio': 0.7,
                'compression_ratio': 0.6
            },
            CompressionLevel.HEAVY: {
                'name': 'gzip',
                'compress_func': self._gzip_compress,
                'decompress_func': self._gzip_decompress,
                'speed_ratio': 0.5,
                'compression_ratio': 0.8
            }
        }
    
    def select_compression_level(
        self,
        data: Any,
        data_type: str,
        urgency: str = 'normal'
    ) -> CompressionLevel:
        """Select optimal compression level based on context."""
        data_size = len(str(data).encode('utf-8'))
        
        # No compression for small data
        if data_size < self.size_threshold:
            return CompressionLevel.NONE
        
        # Check system load
        cpu_usage = self._get_cpu_usage()
        
        # Real-time data needs fast compression
        if urgency == 'urgent' or data_type in ['price', 'quotes']:
            if cpu_usage < self.cpu_threshold:
                return CompressionLevel.LIGHT
            else:
                return CompressionLevel.NONE
        
        # Historical data can use heavy compression
        elif data_type in ['historical', 'fundamentals']:
            if cpu_usage < 80:
                return CompressionLevel.HEAVY
            else:
                return CompressionLevel.MEDIUM
        
        # Default to medium compression
        else:
            if cpu_usage < self.cpu_threshold:
                return CompressionLevel.MEDIUM
            else:
                return CompressionLevel.LIGHT
    
    async def compress_async(
        self,
        data: bytes,
        level: CompressionLevel
    ) -> Tuple[bytes, Dict[str, Any]]:
        """Compress data asynchronously."""
        if level == CompressionLevel.NONE:
            return data, {'compressed': False, 'original_size': len(data)}
        
        start_time = time.time()
        
        # Run compression in thread pool
        loop = asyncio.get_event_loop()
        algorithm = self.algorithms[level]
        
        compressed_data = await loop.run_in_executor(
            self.executor,
            algorithm['compress_func'],
            data
        )
        
        compression_time = time.time() - start_time
        
        # Update stats
        stats = self.compression_stats[algorithm['name']]
        stats['total_size'] += len(data)
        stats['compressed_size'] += len(compressed_data)
        stats['time_taken'] += compression_time
        stats['count'] += 1
        
        metadata = {
            'compressed': True,
            'algorithm': algorithm['name'],
            'original_size': len(data),
            'compressed_size': len(compressed_data),
            'compression_ratio': len(compressed_data) / len(data),
            'compression_time_ms': compression_time * 1000
        }
        
        return compressed_data, metadata
    
    async def decompress_async(
        self,
        data: bytes,
        algorithm: str
    ) -> bytes:
        """Decompress data asynchronously."""
        level = self._get_level_by_algorithm(algorithm)
        if level == CompressionLevel.NONE:
            return data
        
        loop = asyncio.get_event_loop()
        decompress_func = self.algorithms[level]['decompress_func']
        
        return await loop.run_in_executor(
            self.executor,
            decompress_func,
            data
        )
    
    def _lz4_compress(self, data: bytes) -> bytes:
        """LZ4 compression."""
        try:
            return lz4.frame.compress(data)
        except Exception as e:
            logger.error(f"LZ4 compression failed: {e}")
            return data
    
    def _lz4_decompress(self, data: bytes) -> bytes:
        """LZ4 decompression."""
        try:
            return lz4.frame.decompress(data)
        except Exception as e:
            logger.error(f"LZ4 decompression failed: {e}")
            return data
    
    def _zlib_compress(self, data: bytes) -> bytes:
        """ZLIB compression."""
        try:
            return zlib.compress(data, level=6)
        except Exception as e:
            logger.error(f"ZLIB compression failed: {e}")
            return data
    
    def _zlib_decompress(self, data: bytes) -> bytes:
        """ZLIB decompression."""
        try:
            return zlib.decompress(data)
        except Exception as e:
            logger.error(f"ZLIB decompression failed: {e}")
            return data
    
    def _gzip_compress(self, data: bytes) -> bytes:
        """GZIP compression."""
        try:
            return gzip.compress(data, compresslevel=6)
        except Exception as e:
            logger.error(f"GZIP compression failed: {e}")
            return data
    
    def _gzip_decompress(self, data: bytes) -> bytes:
        """GZIP decompression."""
        try:
            return gzip.decompress(data)
        except Exception as e:
            logger.error(f"GZIP decompression failed: {e}")
            return data
    
    def _get_level_by_algorithm(self, algorithm: str) -> CompressionLevel:
        """Get compression level by algorithm name."""
        for level, config in self.algorithms.items():
            if config['name'] == algorithm:
                return level
        return CompressionLevel.NONE
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        try:
            import psutil
            return psutil.cpu_percent(interval=0.1)
        except ImportError:
            return 50.0  # Default assumption
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression statistics."""
        stats = {}
        total_original = 0
        total_compressed = 0
        total_time = 0.0
        total_operations = 0
        
        for algorithm, data in self.compression_stats.items():
            if data['count'] > 0:
                ratio = data['compressed_size'] / data['total_size']
                avg_time = data['time_taken'] / data['count'] * 1000  # ms
                
                stats[algorithm] = {
                    'operations': data['count'],
                    'compression_ratio': ratio,
                    'average_time_ms': avg_time,
                    'total_saved_bytes': data['total_size'] - data['compressed_size']
                }
                
                total_original += data['total_size']
                total_compressed += data['compressed_size']
                total_time += data['time_taken']
                total_operations += data['count']
        
        if total_operations > 0:
            stats['overall'] = {
                'total_operations': total_operations,
                'overall_ratio': total_compressed / total_original,
                'average_time_ms': total_time / total_operations * 1000,
                'total_space_saved_mb': (total_original - total_compressed) / (1024**2)
            }
        
        return stats


class CacheHitOptimizer:
    """
    Main cache hit rate optimization system.
    """
    
    def __init__(self):
        self.key_generator = IntelligentKeyGenerator()
        self.compression_manager = AdaptiveCompressionManager()
        self.hit_metrics = CacheHitMetrics()
        self.locality_optimizer = LocalityOptimizer()
        self._lock = threading.RLock()
        
        # Hit rate tracking
        self.hit_history = deque(maxlen=1000)  # Last 1000 requests
        self.optimization_interval = 300  # 5 minutes
        self.last_optimization = 0
    
    async def get_optimized_cache_entry(
        self,
        ticker: str,
        data_type: str,
        data: Any,
        date: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        urgency: str = 'normal'
    ) -> Tuple[str, bytes, Dict[str, Any]]:
        """
        Get optimized cache entry with best key and compression.
        """
        start_time = time.time()
        
        # Generate optimized key
        cache_key = self.key_generator.generate_optimized_key(
            ticker, data_type, date, params
        )
        
        # Serialize data
        if isinstance(data, (dict, list)):
            data_bytes = json.dumps(data, default=str).encode('utf-8')
        elif isinstance(data, str):
            data_bytes = data.encode('utf-8')
        else:
            data_bytes = str(data).encode('utf-8')
        
        # Select and apply compression
        compression_level = self.compression_manager.select_compression_level(
            data, data_type, urgency
        )
        
        compressed_data, compression_metadata = await self.compression_manager.compress_async(
            data_bytes, compression_level
        )
        
        # Create entry metadata
        metadata = {
            'key_strategy': self.key_generator.optimization_strategy.value,
            'compression': compression_metadata,
            'created_at': datetime.now().isoformat(),
            'data_type': data_type,
            'ticker': ticker,
            'processing_time_ms': (time.time() - start_time) * 1000
        }
        
        return cache_key, compressed_data, metadata
    
    async def track_cache_access(
        self,
        cache_key: str,
        hit: bool,
        response_time_ms: float,
        data_size: int = 0
    ):
        """Track cache access for optimization."""
        with self._lock:
            self.hit_metrics.total_requests += 1
            
            if hit:
                self.hit_metrics.cache_hits += 1
            else:
                self.hit_metrics.cache_misses += 1
            
            # Update hit rate
            self.hit_metrics.hit_rate = (
                self.hit_metrics.cache_hits / self.hit_metrics.total_requests * 100
            )
            
            # Track response time
            total_time = (self.hit_metrics.average_response_time_ms * 
                         (self.hit_metrics.total_requests - 1) + response_time_ms)
            self.hit_metrics.average_response_time_ms = (
                total_time / self.hit_metrics.total_requests
            )
            
            # Add to hit history for trend analysis
            self.hit_history.append({
                'timestamp': time.time(),
                'hit': hit,
                'response_time_ms': response_time_ms,
                'key': cache_key
            })
        
        # Trigger optimization if needed
        current_time = time.time()
        if current_time - self.last_optimization > self.optimization_interval:
            asyncio.create_task(self.optimize_performance())
    
    async def optimize_performance(self):
        """Optimize cache performance based on metrics."""
        self.last_optimization = time.time()
        
        try:
            # Optimize key generation strategy
            self.key_generator.optimize_key_strategy()
            
            # Optimize locality
            await self.locality_optimizer.optimize_data_locality(
                list(self.hit_history)
            )
            
            # Log optimization results
            logger.info(f"Cache optimization completed. Hit rate: {self.hit_metrics.hit_rate:.2f}%")
            
        except Exception as e:
            logger.error(f"Cache optimization failed: {e}")
    
    def get_optimization_metrics(self) -> Dict[str, Any]:
        """Get comprehensive optimization metrics."""
        compression_stats = self.compression_manager.get_compression_stats()
        
        return {
            'hit_metrics': {
                'hit_rate_percent': self.hit_metrics.hit_rate,
                'total_requests': self.hit_metrics.total_requests,
                'cache_hits': self.hit_metrics.cache_hits,
                'cache_misses': self.hit_metrics.cache_misses,
                'average_response_time_ms': self.hit_metrics.average_response_time_ms
            },
            'key_optimization': {
                'current_strategy': self.key_generator.optimization_strategy.value,
                'total_patterns': len(self.key_generator.access_frequencies),
                'most_accessed': dict(sorted(
                    self.key_generator.access_frequencies.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10])
            },
            'compression': compression_stats,
            'locality': self.locality_optimizer.get_stats()
        }


class LocalityOptimizer:
    """
    Optimize data locality for better cache performance.
    """
    
    def __init__(self):
        self.access_patterns = defaultdict(list)
        self.locality_score = 0.0
        self.co_occurrence_matrix = defaultdict(lambda: defaultdict(int))
    
    async def optimize_data_locality(self, access_history: List[Dict[str, Any]]):
        """Optimize data locality based on access patterns."""
        # Analyze temporal locality
        await self._analyze_temporal_locality(access_history)
        
        # Analyze spatial locality
        await self._analyze_spatial_locality(access_history)
        
        # Update locality score
        self.locality_score = self._calculate_locality_score()
    
    async def _analyze_temporal_locality(self, access_history: List[Dict[str, Any]]):
        """Analyze temporal access patterns."""
        # Group accesses by time windows
        time_windows = defaultdict(list)
        
        for access in access_history:
            timestamp = access['timestamp']
            time_bucket = int(timestamp // 300)  # 5-minute buckets
            time_windows[time_bucket].append(access['key'])
        
        # Calculate temporal co-occurrence
        for keys in time_windows.values():
            for i, key1 in enumerate(keys):
                for key2 in keys[i+1:]:
                    self.co_occurrence_matrix[key1][key2] += 1
                    self.co_occurrence_matrix[key2][key1] += 1
    
    async def _analyze_spatial_locality(self, access_history: List[Dict[str, Any]]):
        """Analyze spatial access patterns (related keys)."""
        # Analyze key prefixes and patterns
        key_groups = defaultdict(list)
        
        for access in access_history:
            key = access['key']
            # Group by ticker (first part of hierarchical key)
            if ':' in key:
                prefix = key.split(':')[0]
                key_groups[prefix].append(key)
        
        # Calculate spatial co-occurrence within groups
        for keys in key_groups.values():
            for i, key1 in enumerate(keys):
                for key2 in keys[i+1:]:
                    self.co_occurrence_matrix[key1][key2] += 1
                    self.co_occurrence_matrix[key2][key1] += 1
    
    def _calculate_locality_score(self) -> float:
        """Calculate overall locality score."""
        if not self.co_occurrence_matrix:
            return 0.0
        
        total_pairs = len(self.co_occurrence_matrix)
        total_co_occurrences = sum(
            sum(inner_dict.values()) 
            for inner_dict in self.co_occurrence_matrix.values()
        ) // 2  # Avoid double counting
        
        if total_pairs == 0:
            return 0.0
        
        return (total_co_occurrences / total_pairs) * 10  # Scale to 0-100
    
    def get_stats(self) -> Dict[str, Any]:
        """Get locality optimization statistics."""
        return {
            'locality_score': self.locality_score,
            'co_occurrence_pairs': len(self.co_occurrence_matrix),
            'most_related_keys': self._get_most_related_keys()
        }
    
    def _get_most_related_keys(self) -> List[Tuple[str, str, int]]:
        """Get most co-occurring key pairs."""
        pairs = []
        
        for key1, related in self.co_occurrence_matrix.items():
            for key2, count in related.items():
                if key1 < key2:  # Avoid duplicates
                    pairs.append((key1, key2, count))
        
        return sorted(pairs, key=lambda x: x[2], reverse=True)[:10]


# Global cache hit optimizer
cache_hit_optimizer = CacheHitOptimizer()