"""
Enhanced Cache Configuration System
Optimized for investment analysis application handling 6000+ stocks with intelligent memory management.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import gzip
import zlib
# SECURITY: Removed pickle import - using JSON for cache serialization
from collections import defaultdict
import psutil
import threading

logger = logging.getLogger(__name__)


class StockTier(Enum):
    """Stock priority tiers for optimized caching."""
    CRITICAL = 1    # S&P 500, high volume stocks
    HIGH = 2        # Mid-cap active stocks
    MEDIUM = 3      # Small-cap stocks
    LOW = 4         # Inactive stocks
    MINIMAL = 5     # Delisted/low activity


class CompressionType(Enum):
    """Cache compression types."""
    NONE = "none"
    GZIP = "gzip"
    ZLIB = "zlib"
    LZ4 = "lz4"


@dataclass
class CacheConfiguration:
    """Enhanced cache configuration."""
    # L1 Cache Settings (optimized for 6000+ stocks)
    l1_max_entries: int = 5000
    l1_max_memory_mb: int = 256
    l1_eviction_policy: str = "lru"  # lru, lfu, ttl-aware
    
    # L2 Cache Settings (Redis)
    l2_compression: CompressionType = CompressionType.GZIP
    l2_compression_threshold: int = 1024  # bytes
    l2_serialize_format: str = "json"  # json, pickle, msgpack
    
    # TTL Configuration by data type
    ttl_historical_prices: int = 86400 * 7  # 7 days
    ttl_live_quotes: int = 60  # 1 minute
    ttl_fundamentals: int = 86400  # 1 day
    ttl_technical_analysis: int = 3600  # 1 hour
    ttl_sentiment: int = 1800  # 30 minutes
    ttl_recommendations: int = 1800  # 30 minutes
    
    # Tier-based caching strategy
    tier_l1_ratio: Dict[StockTier, float] = None
    tier_ttl_multipliers: Dict[StockTier, float] = None
    
    # Performance settings
    batch_operations: bool = True
    pipeline_size: int = 100
    compression_cpu_threshold: float = 80.0  # CPU % threshold for compression
    memory_pressure_threshold: float = 85.0  # Memory % threshold
    
    # Predictive caching
    enable_predictive_warming: bool = True
    warmup_market_open_offset: int = 3600  # 1 hour before market open
    warmup_high_frequency_threshold: int = 10  # access frequency threshold
    
    def __post_init__(self):
        """Initialize default tier configurations."""
        if self.tier_l1_ratio is None:
            self.tier_l1_ratio = {
                StockTier.CRITICAL: 0.6,    # 60% of L1 cache for critical stocks
                StockTier.HIGH: 0.25,       # 25% for high priority
                StockTier.MEDIUM: 0.10,     # 10% for medium
                StockTier.LOW: 0.04,        # 4% for low
                StockTier.MINIMAL: 0.01     # 1% for minimal
            }
        
        if self.tier_ttl_multipliers is None:
            self.tier_ttl_multipliers = {
                StockTier.CRITICAL: 1.0,    # Standard TTL
                StockTier.HIGH: 1.2,        # 20% longer
                StockTier.MEDIUM: 1.5,      # 50% longer
                StockTier.LOW: 2.0,         # 2x longer
                StockTier.MINIMAL: 3.0      # 3x longer
            }


class IntelligentCacheManager:
    """
    Intelligent cache manager with tier-based strategies and adaptive behavior.
    """
    
    def __init__(self, config: CacheConfiguration):
        """Initialize intelligent cache manager."""
        self.config = config
        self.access_patterns = defaultdict(list)
        self.tier_mappings: Dict[str, StockTier] = {}
        self.memory_monitor = MemoryMonitor()
        self.compression_stats = CompressionStats()
        self._lock = threading.RLock()
        
        # Tier-based cache partitions
        self.tier_caches = {
            tier: TierCachePartition(
                tier=tier,
                max_entries=int(config.l1_max_entries * config.tier_l1_ratio[tier]),
                config=config
            )
            for tier in StockTier
        }
        
        logger.info(f"Initialized intelligent cache manager with {config.l1_max_entries} L1 entries")
    
    def assign_stock_tier(self, symbol: str, tier: StockTier) -> None:
        """Assign stock to a tier for optimized caching."""
        with self._lock:
            old_tier = self.tier_mappings.get(symbol)
            self.tier_mappings[symbol] = tier
            
            # Migrate cache entry if tier changed
            if old_tier and old_tier != tier:
                self._migrate_cache_entry(symbol, old_tier, tier)
    
    def get_stock_tier(self, symbol: str) -> StockTier:
        """Get stock tier, defaulting to MEDIUM if not assigned."""
        return self.tier_mappings.get(symbol, StockTier.MEDIUM)
    
    def get_optimal_ttl(self, symbol: str, data_type: str) -> int:
        """Get optimal TTL based on stock tier and data type."""
        tier = self.get_stock_tier(symbol)
        base_ttl = self._get_base_ttl(data_type)
        multiplier = self.config.tier_ttl_multipliers[tier]
        
        return int(base_ttl * multiplier)
    
    def _get_base_ttl(self, data_type: str) -> int:
        """Get base TTL for data type."""
        ttl_mapping = {
            'price': self.config.ttl_live_quotes,
            'historical': self.config.ttl_historical_prices,
            'fundamentals': self.config.ttl_fundamentals,
            'technical': self.config.ttl_technical_analysis,
            'sentiment': self.config.ttl_sentiment,
            'recommendations': self.config.ttl_recommendations
        }
        return ttl_mapping.get(data_type, 300)  # Default 5 minutes
    
    def should_compress(self, data: Any, size_bytes: int) -> bool:
        """Determine if data should be compressed."""
        # Don't compress if below threshold
        if size_bytes < self.config.l2_compression_threshold:
            return False
        
        # Check CPU usage
        cpu_usage = psutil.cpu_percent(interval=0.1)
        if cpu_usage > self.config.compression_cpu_threshold:
            return False
        
        return True
    
    def compress_data(self, data: bytes, compression_type: CompressionType) -> bytes:
        """Compress data using specified algorithm."""
        start_time = time.time()
        original_size = len(data)
        
        try:
            if compression_type == CompressionType.GZIP:
                compressed = gzip.compress(data)
            elif compression_type == CompressionType.ZLIB:
                compressed = zlib.compress(data)
            elif compression_type == CompressionType.LZ4:
                try:
                    import lz4.frame
                    compressed = lz4.frame.compress(data)
                except ImportError:
                    # Fallback to gzip if lz4 not available
                    compressed = gzip.compress(data)
            else:
                return data
            
            # Update compression stats
            compression_time = time.time() - start_time
            ratio = len(compressed) / original_size
            
            self.compression_stats.update(
                original_size, len(compressed), compression_time, compression_type
            )
            
            return compressed
            
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            return data
    
    def decompress_data(self, data: bytes, compression_type: CompressionType) -> bytes:
        """Decompress data."""
        try:
            if compression_type == CompressionType.GZIP:
                return gzip.decompress(data)
            elif compression_type == CompressionType.ZLIB:
                return zlib.decompress(data)
            elif compression_type == CompressionType.LZ4:
                try:
                    import lz4.frame
                    return lz4.frame.decompress(data)
                except ImportError:
                    return gzip.decompress(data)
            else:
                return data
        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            return data
    
    def track_access(self, symbol: str, data_type: str, timestamp: float = None) -> None:
        """Track access patterns for predictive caching."""
        if timestamp is None:
            timestamp = time.time()
        
        with self._lock:
            key = f"{symbol}:{data_type}"
            self.access_patterns[key].append(timestamp)
            
            # Keep only recent accesses (last 24 hours)
            cutoff = timestamp - 86400
            self.access_patterns[key] = [
                ts for ts in self.access_patterns[key] if ts > cutoff
            ]
    
    def get_high_frequency_stocks(self, threshold: int = None) -> List[str]:
        """Get stocks with high access frequency for predictive warming."""
        if threshold is None:
            threshold = self.config.warmup_high_frequency_threshold
        
        high_freq_stocks = []
        current_time = time.time()
        
        with self._lock:
            for key, accesses in self.access_patterns.items():
                symbol = key.split(':')[0]
                # Count accesses in last hour
                recent_accesses = [
                    ts for ts in accesses 
                    if current_time - ts < 3600
                ]
                
                if len(recent_accesses) >= threshold:
                    high_freq_stocks.append(symbol)
        
        return list(set(high_freq_stocks))
    
    def _migrate_cache_entry(self, symbol: str, old_tier: StockTier, new_tier: StockTier) -> None:
        """Migrate cache entry between tiers."""
        # This would involve moving cached data between tier partitions
        # Implementation depends on specific cache structure
        logger.info(f"Migrating {symbol} from {old_tier} to {new_tier}")
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        stats = {
            'configuration': {
                'l1_max_entries': self.config.l1_max_entries,
                'l1_max_memory_mb': self.config.l1_max_memory_mb,
                'compression_enabled': self.config.l2_compression != CompressionType.NONE
            },
            'tier_mappings': {tier.name: count for tier, count in 
                             self._count_stocks_by_tier().items()},
            'memory': self.memory_monitor.get_stats(),
            'compression': self.compression_stats.get_stats(),
            'access_patterns': self._get_access_pattern_stats()
        }
        
        # Add tier-specific stats
        for tier, cache_partition in self.tier_caches.items():
            stats[f'tier_{tier.name}'] = cache_partition.get_stats()
        
        return stats
    
    def _count_stocks_by_tier(self) -> Dict[StockTier, int]:
        """Count stocks assigned to each tier."""
        tier_counts = defaultdict(int)
        for tier in self.tier_mappings.values():
            tier_counts[tier] += 1
        return dict(tier_counts)
    
    def _get_access_pattern_stats(self) -> Dict[str, Any]:
        """Get access pattern statistics."""
        total_accesses = sum(len(accesses) for accesses in self.access_patterns.values())
        active_keys = len([k for k, v in self.access_patterns.items() if v])
        
        return {
            'total_accesses': total_accesses,
            'active_keys': active_keys,
            'high_frequency_stocks': len(self.get_high_frequency_stocks())
        }


class TierCachePartition:
    """Cache partition for a specific stock tier."""
    
    def __init__(self, tier: StockTier, max_entries: int, config: CacheConfiguration):
        self.tier = tier
        self.max_entries = max_entries
        self.config = config
        self.cache: Dict[str, Any] = {}
        self.access_times: Dict[str, float] = {}
        self._lock = threading.RLock()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tier-specific cache statistics."""
        with self._lock:
            return {
                'tier': self.tier.name,
                'entries': len(self.cache),
                'max_entries': self.max_entries,
                'utilization': len(self.cache) / self.max_entries if self.max_entries > 0 else 0
            }


class MemoryMonitor:
    """Monitor system memory usage for cache optimization."""
    
    def __init__(self):
        self.last_check = 0
        self.check_interval = 30  # seconds
        self.memory_history = []
    
    def get_memory_pressure(self) -> float:
        """Get current memory pressure as percentage."""
        current_time = time.time()
        if current_time - self.last_check < self.check_interval:
            return self.memory_history[-1] if self.memory_history else 0.0
        
        memory = psutil.virtual_memory()
        pressure = memory.percent
        
        self.memory_history.append(pressure)
        self.last_check = current_time
        
        # Keep only recent history
        if len(self.memory_history) > 100:
            self.memory_history = self.memory_history[-50:]
        
        return pressure
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory monitoring statistics."""
        memory = psutil.virtual_memory()
        return {
            'current_usage_percent': memory.percent,
            'available_gb': memory.available / (1024**3),
            'total_gb': memory.total / (1024**3),
            'average_pressure': sum(self.memory_history) / len(self.memory_history) 
                              if self.memory_history else 0
        }


class CompressionStats:
    """Track compression statistics for optimization."""
    
    def __init__(self):
        self.stats = {
            'total_compressed': 0,
            'total_original_size': 0,
            'total_compressed_size': 0,
            'total_compression_time': 0.0,
            'by_algorithm': defaultdict(lambda: {
                'count': 0, 'original_size': 0, 'compressed_size': 0, 'time': 0.0
            })
        }
        self._lock = threading.Lock()
    
    def update(self, original_size: int, compressed_size: int, 
               compression_time: float, algorithm: CompressionType):
        """Update compression statistics."""
        with self._lock:
            self.stats['total_compressed'] += 1
            self.stats['total_original_size'] += original_size
            self.stats['total_compressed_size'] += compressed_size
            self.stats['total_compression_time'] += compression_time
            
            alg_stats = self.stats['by_algorithm'][algorithm.value]
            alg_stats['count'] += 1
            alg_stats['original_size'] += original_size
            alg_stats['compressed_size'] += compressed_size
            alg_stats['time'] += compression_time
    
    def get_stats(self) -> Dict[str, Any]:
        """Get compression statistics."""
        with self._lock:
            total_original = self.stats['total_original_size']
            total_compressed = self.stats['total_compressed_size']
            
            return {
                'total_operations': self.stats['total_compressed'],
                'overall_compression_ratio': (
                    total_compressed / total_original 
                    if total_original > 0 else 1.0
                ),
                'average_compression_time_ms': (
                    self.stats['total_compression_time'] * 1000 / 
                    self.stats['total_compressed']
                    if self.stats['total_compressed'] > 0 else 0
                ),
                'space_saved_mb': (total_original - total_compressed) / (1024**2),
                'by_algorithm': dict(self.stats['by_algorithm'])
            }


# Global enhanced cache configuration
enhanced_cache_config = CacheConfiguration()
intelligent_cache_manager = IntelligentCacheManager(enhanced_cache_config)