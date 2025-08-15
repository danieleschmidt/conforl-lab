"""Adaptive caching system for performance optimization."""

import time
import pickle
import hashlib
import gzip
import json
import os
from typing import Any, Dict, Optional, Tuple, Union, Callable, List
from collections import OrderedDict, defaultdict
import threading
from pathlib import Path

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    class np:
        @staticmethod
        def array(data):
            return data
        @staticmethod
        def mean(data):
            return sum(data) / len(data) if data else 0

from ..utils.logging import get_logger
from ..utils.concurrency import ThreadSafeDict
from ..utils.errors import ConfoRLError

logger = get_logger(__name__)


class AdaptiveCache:
    """Adaptive caching system with usage pattern learning."""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        """Initialize adaptive cache.
        
        Args:
            max_size: Maximum cache size
            ttl: Time-to-live in seconds
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache = {}
        self.access_times = {}
        self.access_count = {}
        self.hits = 0
        self.misses = 0
    
    def set(self, key: str, value: Any) -> None:
        """Set cache value."""
        import time
        current_time = time.time()
        
        # Evict old entries if at capacity
        if len(self.cache) >= self.max_size:
            self._evict_oldest()
        
        self.cache[key] = value
        self.access_times[key] = current_time
        self.access_count[key] = 0
    
    def get(self, key: str) -> Any:
        """Get cache value."""
        import time
        current_time = time.time()
        
        if key not in self.cache:
            self.misses += 1
            return None
        
        # Check TTL
        if current_time - self.access_times[key] > self.ttl:
            del self.cache[key]
            del self.access_times[key]
            del self.access_count[key]
            self.misses += 1
            return None
        
        # Update access stats
        self.access_times[key] = current_time
        self.access_count[key] += 1
        self.hits += 1
        
        return self.cache[key]
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            'hits': self.hits,
            'misses': self.misses,
            'size': len(self.cache),
            'max_size': self.max_size
        }
    
    def _evict_oldest(self) -> None:
        """Evict oldest cache entry."""
        if not self.access_times:
            return
        
        oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        del self.cache[oldest_key]
        del self.access_times[oldest_key]
        del self.access_count[oldest_key]


class OriginalAdaptiveCache:
    """Adaptive cache that learns access patterns and optimizes accordingly."""
    
    def __init__(
        self,
        max_size: int = 1000,
        ttl: float = 3600.0,  # 1 hour default TTL
        adaptive_ttl: bool = True,
        compression: bool = False
    ):
        """Initialize adaptive cache.
        
        Args:
            max_size: Maximum number of cache entries
            ttl: Time-to-live in seconds
            adaptive_ttl: Whether to adapt TTL based on access patterns
            compression: Whether to compress cached data
        """
        self.max_size = max_size
        self.base_ttl = ttl
        self.adaptive_ttl = adaptive_ttl
        self.compression = compression
        
        # Cache storage
        self._cache = OrderedDict()
        self._metadata = {}
        self._lock = threading.RLock()
        
        # Access pattern tracking
        self._access_counts = defaultdict(int)
        self._access_times = defaultdict(list)
        self._hit_rates = defaultdict(float)
        
        # Performance metrics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
    def _generate_key(self, key: Any) -> str:
        """Generate string key from any hashable object."""
        if isinstance(key, str):
            return key
        
        # Create hash of the key
        key_str = str(key)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _compress_data(self, data: Any) -> bytes:
        """Compress data for storage."""
        if not self.compression:
            return pickle.dumps(data)
        
        import gzip
        pickled_data = pickle.dumps(data)
        return gzip.compress(pickled_data)
    
    def _decompress_data(self, compressed_data: bytes) -> Any:
        """Decompress data from storage."""
        if not self.compression:
            return pickle.loads(compressed_data)
        
        import gzip
        pickled_data = gzip.decompress(compressed_data)
        return pickle.loads(pickled_data)
    
    def _calculate_adaptive_ttl(self, key: str) -> float:
        """Calculate adaptive TTL based on access patterns."""
        if not self.adaptive_ttl:
            return self.base_ttl
        
        access_count = self._access_counts[key]
        access_times = self._access_times[key]
        
        # Higher access count = longer TTL
        frequency_factor = min(2.0, 1.0 + access_count / 100.0)
        
        # Recent access pattern analysis
        if len(access_times) >= 2:
            # Calculate average access interval
            intervals = [
                access_times[i] - access_times[i-1] 
                for i in range(1, len(access_times))
            ]
            avg_interval = np.mean(intervals) if intervals else self.base_ttl
            
            # Predict next access time
            recency_factor = max(0.5, min(2.0, self.base_ttl / avg_interval))
        else:
            recency_factor = 1.0
        
        # Combine factors
        adaptive_ttl = self.base_ttl * frequency_factor * recency_factor
        return min(adaptive_ttl, self.base_ttl * 3)  # Cap at 3x base TTL
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired."""
        if key not in self._metadata:
            return True
        
        metadata = self._metadata[key]
        current_time = time.time()
        return current_time > metadata['expires_at']
    
    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if not self._cache:
            return
        
        # Remove oldest item (LRU in OrderedDict)
        key, _ = self._cache.popitem(last=False)
        
        if key in self._metadata:
            del self._metadata[key]
        
        self.evictions += 1
        logger.debug(f"Evicted cache entry: {key}")
    
    def get(self, key: Any, default: Any = None) -> Any:
        """Get item from cache.
        
        Args:
            key: Cache key
            default: Default value if key not found
            
        Returns:
            Cached value or default
        """
        str_key = self._generate_key(key)
        current_time = time.time()
        
        with self._lock:
            if str_key in self._cache and not self._is_expired(str_key):
                # Cache hit
                value = self._decompress_data(self._cache[str_key])
                
                # Move to end (most recently used)
                self._cache.move_to_end(str_key)
                
                # Update access tracking
                self._access_counts[str_key] += 1
                self._access_times[str_key].append(current_time)
                
                # Keep only recent access times
                if len(self._access_times[str_key]) > 10:
                    self._access_times[str_key] = self._access_times[str_key][-10:]
                
                self.hits += 1
                return value
            
            else:
                # Cache miss or expired
                if str_key in self._cache:
                    # Remove expired entry
                    del self._cache[str_key]
                    if str_key in self._metadata:
                        del self._metadata[str_key]
                
                self.misses += 1
                return default
    
    def put(self, key: Any, value: Any) -> None:
        """Put item in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        str_key = self._generate_key(key)
        current_time = time.time()
        
        with self._lock:
            # Check if we need to evict
            if str_key not in self._cache and len(self._cache) >= self.max_size:
                self._evict_lru()
            
            # Store compressed data
            compressed_value = self._compress_data(value)
            self._cache[str_key] = compressed_value
            
            # Calculate adaptive TTL
            ttl = self._calculate_adaptive_ttl(str_key)
            
            # Store metadata
            self._metadata[str_key] = {
                'created_at': current_time,
                'expires_at': current_time + ttl,
                'ttl': ttl,
                'size': len(compressed_value)
            }
            
            # Move to end (most recently used)
            self._cache.move_to_end(str_key)
            
            logger.debug(f"Cached entry: {str_key} (TTL: {ttl:.1f}s)")
    
    def invalidate(self, key: Any) -> bool:
        """Invalidate cache entry.
        
        Args:
            key: Cache key to invalidate
            
        Returns:
            True if key was in cache
        """
        str_key = self._generate_key(key)
        
        with self._lock:
            if str_key in self._cache:
                del self._cache[str_key]
                if str_key in self._metadata:
                    del self._metadata[str_key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._metadata.clear()
            self._access_counts.clear()
            self._access_times.clear()
            logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
            
            # Calculate total cache size
            total_size = sum(
                metadata.get('size', 0) 
                for metadata in self._metadata.values()
            )
            
            return {
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate,
                'evictions': self.evictions,
                'size': len(self._cache),
                'max_size': self.max_size,
                'total_bytes': total_size,
                'avg_ttl': np.mean([
                    metadata['ttl'] for metadata in self._metadata.values()
                ]) if self._metadata else 0.0
            }
    
    def cleanup_expired(self) -> int:
        """Clean up expired entries.
        
        Returns:
            Number of entries removed
        """
        current_time = time.time()
        expired_keys = []
        
        with self._lock:
            for key, metadata in self._metadata.items():
                if current_time > metadata['expires_at']:
                    expired_keys.append(key)
            
            for key in expired_keys:
                if key in self._cache:
                    del self._cache[key]
                del self._metadata[key]
        
        logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
        return len(expired_keys)


class PerformanceCache:
    """Specialized cache for performance-critical operations."""
    
    def __init__(self, max_size: int = 500):
        """Initialize performance cache.
        
        Args:
            max_size: Maximum number of cache entries
        """
        self.max_size = max_size
        self._cache = {}
        self._access_order = []
        self._lock = threading.Lock()
        
        # Performance tracking
        self.computation_times = defaultdict(list)
        self.cache_savings = 0.0
    
    def cached_computation(
        self,
        key: str,
        computation_func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Perform cached computation.
        
        Args:
            key: Cache key
            computation_func: Function to compute result
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Computation result (cached or fresh)
        """
        cache_key = f"{key}_{hash(str(args))}_{hash(str(sorted(kwargs.items())))}"
        
        with self._lock:
            if cache_key in self._cache:
                # Cache hit - move to end of access order
                self._access_order.remove(cache_key)
                self._access_order.append(cache_key)
                
                # Estimate time saved
                avg_computation_time = np.mean(self.computation_times[key]) if self.computation_times[key] else 0.1
                self.cache_savings += avg_computation_time
                
                return self._cache[cache_key]
        
        # Cache miss - compute result
        start_time = time.time()
        result = computation_func(*args, **kwargs)
        computation_time = time.time() - start_time
        
        # Track computation time
        self.computation_times[key].append(computation_time)
        if len(self.computation_times[key]) > 100:
            self.computation_times[key] = self.computation_times[key][-100:]
        
        # Store in cache
        with self._lock:
            # Evict if necessary
            if len(self._cache) >= self.max_size and cache_key not in self._cache:
                oldest_key = self._access_order.pop(0)
                del self._cache[oldest_key]
            
            self._cache[cache_key] = result
            if cache_key not in self._access_order:
                self._access_order.append(cache_key)
        
        return result
    
    def precompute_batch(
        self,
        keys_and_args: list,
        computation_func: Callable
    ) -> None:
        """Precompute and cache a batch of results.
        
        Args:
            keys_and_args: List of (key, args, kwargs) tuples
            computation_func: Function to compute results
        """
        logger.info(f"Precomputing batch of {len(keys_and_args)} items")
        
        for key, args, kwargs in keys_and_args:
            try:
                self.cached_computation(key, computation_func, *args, **kwargs)
                
            except Exception as e:
                logger.warning(f"Failed to precompute {key}: {e}")
        
        logger.info("Batch precomputation completed")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics.
        
        Returns:
            Dictionary with performance stats
        """
        with self._lock:
            return {
                'cache_size': len(self._cache),
                'max_size': self.max_size,
                'cache_savings_seconds': self.cache_savings,
                'avg_computation_times': {
                    key: np.mean(times) 
                    for key, times in self.computation_times.items()
                }
            }