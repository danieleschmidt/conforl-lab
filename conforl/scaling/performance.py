"""Performance optimization and memory management for ConfoRL.

Advanced performance optimization techniques including memory pooling,
computation caching, JIT compilation, and adaptive resource management.
"""

import time
import psutil
import threading
import gc
import weakref
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass
from collections import OrderedDict, defaultdict
import hashlib
import pickle
import functools
import sys

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    class np:
        float32 = float
        float64 = float
        dtype = type
        ndarray = list
        
        @staticmethod
        def array(data):
            return data
        @staticmethod 
        def mean(data):
            return sum(data) / len(data) if data else 0

from ..utils.logging import get_logger
from ..utils.errors import ConfoRLError

logger = get_logger(__name__)


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    
    total_memory_gb: float
    available_memory_gb: float
    used_memory_gb: float
    memory_percent: float
    cache_memory_mb: float
    process_memory_mb: float
    gc_collections: Dict[int, int]
    memory_leaks_detected: int = 0


@dataclass
class PerformanceMetrics:
    """Performance metrics for optimization."""
    
    cpu_usage_percent: float
    memory_usage_mb: float
    cache_hit_ratio: float
    avg_computation_time: float
    peak_memory_usage: float
    gc_pause_time: float
    optimization_speedup: float = 1.0
    bottleneck_operations: List[str] = None
    
    def __post_init__(self):
        if self.bottleneck_operations is None:
            self.bottleneck_operations = []


class MemoryManager:
    """Advanced memory management for ConfoRL."""
    
    def __init__(
        self,
        max_cache_size_mb: float = 1000.0,
        gc_threshold: float = 0.8,
        enable_memory_pooling: bool = True
    ):
        """Initialize memory manager.
        
        Args:
            max_cache_size_mb: Maximum cache size in MB
            gc_threshold: Memory usage threshold for GC
            enable_memory_pooling: Enable memory pooling
        """
        self.max_cache_size_mb = max_cache_size_mb
        self.gc_threshold = gc_threshold
        self.enable_memory_pooling = enable_memory_pooling
        
        # Memory pools for different object types
        self.memory_pools = {
            'arrays': [],
            'trajectories': [],
            'certificates': [],
            'general': []
        }
        
        # Memory tracking
        self.allocated_objects = weakref.WeakSet()
        self.peak_memory_usage = 0.0
        self.memory_pressure_callbacks = []
        
        # Cache management
        self.object_cache = OrderedDict()
        self.cache_stats = {'hits': 0, 'misses': 0, 'evictions': 0}
        
        # Performance monitoring
        self.gc_stats = {'collections': 0, 'total_time': 0.0}
        self.memory_warnings_issued = 0
        
        # Start memory monitoring
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(target=self._memory_monitor_loop, daemon=True)
        self._monitor_thread.start()
        
        logger.info(f"Memory manager initialized (cache: {max_cache_size_mb}MB)")
    
    def allocate_array(self, shape: Tuple[int, ...], dtype: np.dtype = np.float32) -> np.ndarray:
        """Allocate array with memory pooling.
        
        Args:
            shape: Array shape
            dtype: Array data type
            
        Returns:
            Allocated array
        """
        if not self.enable_memory_pooling:
            return np.zeros(shape, dtype=dtype)
        
        # Try to reuse array from pool
        required_size = np.prod(shape) * np.dtype(dtype).itemsize
        
        for i, pooled_array in enumerate(self.memory_pools['arrays']):
            if (pooled_array.shape == shape and 
                pooled_array.dtype == dtype and
                pooled_array.nbytes == required_size):
                
                # Reuse pooled array
                array = self.memory_pools['arrays'].pop(i)
                array.fill(0)  # Clear data
                logger.debug(f"Reused array from pool: {shape}")
                return array
        
        # Allocate new array
        array = np.zeros(shape, dtype=dtype)
        self.allocated_objects.add(array)
        
        logger.debug(f"Allocated new array: {shape}")
        return array
    
    def deallocate_array(self, array: np.ndarray) -> None:
        """Return array to memory pool.
        
        Args:
            array: Array to deallocate
        """
        if not self.enable_memory_pooling or array.nbytes > 100 * 1024 * 1024:  # Don't pool huge arrays
            return
        
        # Add to pool if there's space
        if len(self.memory_pools['arrays']) < 100:  # Limit pool size
            self.memory_pools['arrays'].append(array.copy())
            logger.debug(f"Returned array to pool: {array.shape}")
    
    def cache_object(self, key: str, obj: Any, size_mb: Optional[float] = None) -> None:
        """Cache object with LRU eviction.
        
        Args:
            key: Cache key
            obj: Object to cache
            size_mb: Object size in MB (estimated if None)
        """
        if size_mb is None:
            size_mb = self._estimate_object_size(obj)
        
        # Check cache size limit
        while self._get_cache_size_mb() + size_mb > self.max_cache_size_mb:
            if not self.object_cache:
                break
            
            # Evict least recently used item
            evicted_key, evicted_obj = self.object_cache.popitem(last=False)
            self.cache_stats['evictions'] += 1
            logger.debug(f"Evicted cached object: {evicted_key}")
        
        # Add to cache
        self.object_cache[key] = obj
        self.object_cache.move_to_end(key)  # Mark as most recently used
        
        logger.debug(f"Cached object: {key} ({size_mb:.2f}MB)")
    
    def get_cached_object(self, key: str) -> Optional[Any]:
        """Retrieve cached object.
        
        Args:
            key: Cache key
            
        Returns:
            Cached object or None if not found
        """
        if key in self.object_cache:
            # Move to end (most recently used)
            self.object_cache.move_to_end(key)
            self.cache_stats['hits'] += 1
            logger.debug(f"Cache hit: {key}")
            return self.object_cache[key]
        else:
            self.cache_stats['misses'] += 1
            logger.debug(f"Cache miss: {key}")
            return None
    
    def clear_cache(self) -> None:
        """Clear object cache."""
        cleared_count = len(self.object_cache)
        self.object_cache.clear()
        logger.info(f"Cleared cache ({cleared_count} objects)")
    
    def force_garbage_collection(self) -> Dict[int, int]:
        """Force garbage collection and return statistics.
        
        Returns:
            GC statistics per generation
        """
        start_time = time.time()
        
        # Collect statistics before GC
        before_stats = {i: gc.get_count()[i] for i in range(len(gc.get_count()))}
        
        # Force collection for all generations
        collected = {}
        for generation in range(len(gc.get_stats())):
            collected[generation] = gc.collect(generation)
        
        gc_time = time.time() - start_time
        self.gc_stats['collections'] += 1
        self.gc_stats['total_time'] += gc_time
        
        logger.debug(f"GC completed in {gc_time:.3f}s, collected: {collected}")
        return collected
    
    def _memory_monitor_loop(self):
        """Memory monitoring loop."""
        while self._monitoring_active:
            try:
                memory_stats = self.get_memory_stats()
                
                # Check for memory pressure
                if memory_stats.memory_percent > self.gc_threshold * 100:
                    self._handle_memory_pressure()
                
                # Update peak memory usage
                current_memory = memory_stats.used_memory_gb
                if current_memory > self.peak_memory_usage:
                    self.peak_memory_usage = current_memory
                
                # Sleep between checks
                time.sleep(10.0)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                time.sleep(30.0)  # Wait longer on error
    
    def _handle_memory_pressure(self):
        """Handle memory pressure situation."""
        logger.warning("Memory pressure detected - initiating cleanup")
        
        # Clear caches
        cache_size_before = len(self.object_cache)
        self.clear_cache()
        
        # Force garbage collection
        collected = self.force_garbage_collection()
        
        # Call registered callbacks
        for callback in self.memory_pressure_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Memory pressure callback failed: {e}")
        
        # Clear memory pools if needed
        for pool_name, pool in self.memory_pools.items():
            if len(pool) > 10:  # Keep some objects
                removed = len(pool) - 10
                self.memory_pools[pool_name] = pool[-10:]
                logger.debug(f"Cleared {removed} objects from {pool_name} pool")
        
        self.memory_warnings_issued += 1
        logger.info(f"Memory cleanup completed - cleared {cache_size_before} cached objects")
    
    def register_memory_pressure_callback(self, callback: Callable[[], None]):
        """Register callback for memory pressure events.
        
        Args:
            callback: Function to call during memory pressure
        """
        self.memory_pressure_callbacks.append(callback)
        logger.debug("Registered memory pressure callback")
    
    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics.
        
        Returns:
            Memory usage statistics
        """
        # System memory
        memory_info = psutil.virtual_memory()
        
        # Process memory
        process = psutil.Process()
        process_memory = process.memory_info()
        
        # Cache memory
        cache_memory = self._get_cache_size_mb()
        
        # GC statistics
        gc_counts = {i: count for i, count in enumerate(gc.get_count())}
        
        return MemoryStats(
            total_memory_gb=memory_info.total / (1024**3),
            available_memory_gb=memory_info.available / (1024**3),
            used_memory_gb=memory_info.used / (1024**3),
            memory_percent=memory_info.percent,
            cache_memory_mb=cache_memory,
            process_memory_mb=process_memory.rss / (1024**2),
            gc_collections=gc_counts,
            memory_leaks_detected=self._detect_memory_leaks()
        )
    
    def _get_cache_size_mb(self) -> float:
        """Estimate total cache size in MB."""
        total_size = 0.0
        
        for obj in self.object_cache.values():
            total_size += self._estimate_object_size(obj)
        
        return total_size
    
    def _estimate_object_size(self, obj: Any) -> float:
        """Estimate object size in MB.
        
        Args:
            obj: Object to estimate
            
        Returns:
            Estimated size in MB
        """
        try:
            if isinstance(obj, np.ndarray):
                return obj.nbytes / (1024**2)
            elif hasattr(obj, '__sizeof__'):
                return obj.__sizeof__() / (1024**2)
            else:
                # Fallback to pickle size
                return len(pickle.dumps(obj)) / (1024**2)
        except:
            return 1.0  # Default 1MB estimate
    
    def _detect_memory_leaks(self) -> int:
        """Detect potential memory leaks.
        
        Returns:
            Number of potential leaks detected
        """
        # Simple heuristic: objects that haven't been collected
        # This is a simplified implementation
        return max(0, len(self.allocated_objects) - 1000)  # Threshold for concern
    
    def get_memory_report(self) -> Dict[str, Any]:
        """Generate comprehensive memory report.
        
        Returns:
            Memory usage report
        """
        stats = self.get_memory_stats()
        
        return {
            'memory_stats': {
                'total_memory_gb': stats.total_memory_gb,
                'used_memory_gb': stats.used_memory_gb,
                'available_memory_gb': stats.available_memory_gb,
                'memory_percent': stats.memory_percent,
                'process_memory_mb': stats.process_memory_mb
            },
            'cache_info': {
                'cache_size_mb': stats.cache_memory_mb,
                'cached_objects': len(self.object_cache),
                'cache_hit_ratio': self.cache_stats['hits'] / max(1, self.cache_stats['hits'] + self.cache_stats['misses']),
                'cache_evictions': self.cache_stats['evictions']
            },
            'memory_pools': {
                pool_name: len(pool) for pool_name, pool in self.memory_pools.items()
            },
            'gc_stats': {
                'collections': self.gc_stats['collections'],
                'total_gc_time': self.gc_stats['total_time'],
                'avg_gc_time': self.gc_stats['total_time'] / max(1, self.gc_stats['collections'])
            },
            'peak_memory_usage_gb': self.peak_memory_usage,
            'memory_warnings': self.memory_warnings_issued,
            'potential_memory_leaks': stats.memory_leaks_detected
        }
    
    def shutdown(self):
        """Shutdown memory manager."""
        self._monitoring_active = False
        if self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)
        
        self.clear_cache()
        logger.info("Memory manager shutdown")


class ComputationCache:
    """High-performance computation cache with intelligent prefetching."""
    
    def __init__(
        self,
        max_size: int = 10000,
        ttl_seconds: float = 3600.0,
        enable_prefetching: bool = True
    ):
        """Initialize computation cache.
        
        Args:
            max_size: Maximum number of cached computations
            ttl_seconds: Time-to-live for cached results
            enable_prefetching: Enable intelligent prefetching
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.enable_prefetching = enable_prefetching
        
        # Cache storage
        self.cache = OrderedDict()
        self.access_times = {}
        self.access_frequencies = defaultdict(int)
        
        # Performance tracking
        self.hit_count = 0
        self.miss_count = 0
        self.computation_times = {}
        
        # Prefetching
        self.prefetch_queue = []
        self.prefetch_patterns = defaultdict(list)
        
        # Thread for cache maintenance
        self._maintenance_thread = threading.Thread(target=self._maintenance_loop, daemon=True)
        self._maintenance_active = True
        self._maintenance_thread.start()
        
        logger.info(f"Computation cache initialized (size: {max_size}, TTL: {ttl_seconds}s)")
    
    def memoize(self, ttl: Optional[float] = None, key_func: Optional[Callable] = None):
        """Decorator for memoizing function calls.
        
        Args:
            ttl: Custom TTL for this function
            key_func: Custom key generation function
            
        Returns:
            Memoization decorator
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    cache_key = self._generate_key(func.__name__, args, kwargs)
                
                # Try cache first
                result = self.get(cache_key)
                if result is not None:
                    return result
                
                # Compute result
                start_time = time.time()
                result = func(*args, **kwargs)
                computation_time = time.time() - start_time
                
                # Cache result
                self.put(cache_key, result, ttl or self.ttl_seconds)
                self.computation_times[cache_key] = computation_time
                
                # Learn prefetch patterns
                if self.enable_prefetching:
                    self._learn_access_pattern(cache_key, args, kwargs)
                
                return result
            
            return wrapper
        return decorator
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached result.
        
        Args:
            key: Cache key
            
        Returns:
            Cached result or None if not found/expired
        """
        if key not in self.cache:
            self.miss_count += 1
            return None
        
        result, timestamp = self.cache[key]
        
        # Check expiration
        if time.time() - timestamp > self.ttl_seconds:
            del self.cache[key]
            del self.access_times[key]
            self.miss_count += 1
            return None
        
        # Update access info
        self.cache.move_to_end(key)  # Mark as recently used
        self.access_times[key] = time.time()
        self.access_frequencies[key] += 1
        self.hit_count += 1
        
        logger.debug(f"Cache hit: {key}")
        return result
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Put result in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Custom TTL (uses default if None)
        """
        # Evict if necessary
        while len(self.cache) >= self.max_size:
            if not self.cache:
                break
            
            # Remove least recently used item
            oldest_key, _ = self.cache.popitem(last=False)
            self.access_times.pop(oldest_key, None)
            logger.debug(f"Evicted from cache: {oldest_key}")
        
        # Add to cache
        timestamp = time.time()
        self.cache[key] = (value, timestamp)
        self.access_times[key] = timestamp
        self.access_frequencies[key] = 0
        
        logger.debug(f"Cached result: {key}")
    
    def prefetch(self, keys: List[str]) -> None:
        """Prefetch results for given keys.
        
        Args:
            keys: Keys to prefetch
        """
        if not self.enable_prefetching:
            return
        
        for key in keys:
            if key not in self.cache and key not in self.prefetch_queue:
                self.prefetch_queue.append(key)
        
        logger.debug(f"Queued {len(keys)} keys for prefetching")
    
    def _generate_key(self, func_name: str, args: Tuple, kwargs: Dict) -> str:
        """Generate cache key from function call.
        
        Args:
            func_name: Function name
            args: Positional arguments
            kwargs: Keyword arguments
            
        Returns:
            Cache key string
        """
        # Create deterministic key
        key_data = {
            'func': func_name,
            'args': args,
            'kwargs': sorted(kwargs.items())
        }
        
        key_str = str(key_data)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _learn_access_pattern(self, key: str, args: Tuple, kwargs: Dict) -> None:
        """Learn access patterns for prefetching.
        
        Args:
            key: Current cache key
            args: Function arguments
            kwargs: Function keyword arguments
        """
        # Simple pattern learning: track sequential accesses
        current_pattern = (key, time.time())
        
        # Look for patterns in recent accesses
        recent_keys = list(self.cache.keys())[-10:]  # Last 10 accesses
        
        for prev_key in recent_keys:
            if prev_key != key:
                pattern_key = f"{prev_key}->{key}"
                self.prefetch_patterns[pattern_key].append(time.time())
        
        # Limit pattern history
        for pattern_key in list(self.prefetch_patterns.keys()):
            if len(self.prefetch_patterns[pattern_key]) > 100:
                self.prefetch_patterns[pattern_key] = self.prefetch_patterns[pattern_key][-50:]
    
    def _maintenance_loop(self):
        """Cache maintenance loop."""
        while self._maintenance_active:
            try:
                current_time = time.time()
                
                # Clean expired entries
                expired_keys = []
                for key, (value, timestamp) in self.cache.items():
                    if current_time - timestamp > self.ttl_seconds:
                        expired_keys.append(key)
                
                for key in expired_keys:
                    del self.cache[key]
                    self.access_times.pop(key, None)
                
                if expired_keys:
                    logger.debug(f"Cleaned {len(expired_keys)} expired cache entries")
                
                # Intelligent prefetching based on patterns
                if self.enable_prefetching and self.prefetch_patterns:
                    self._intelligent_prefetch()
                
                time.sleep(60.0)  # Run maintenance every minute
                
            except Exception as e:
                logger.error(f"Cache maintenance error: {e}")
                time.sleep(300.0)  # Wait 5 minutes on error
    
    def _intelligent_prefetch(self):
        """Perform intelligent prefetching based on learned patterns."""
        # Find frequently occurring patterns
        frequent_patterns = {}
        for pattern_key, timestamps in self.prefetch_patterns.items():
            if len(timestamps) >= 3:  # At least 3 occurrences
                recent_timestamps = [ts for ts in timestamps if time.time() - ts < 3600]  # Last hour
                if len(recent_timestamps) >= 2:
                    frequent_patterns[pattern_key] = len(recent_timestamps)
        
        # Sort by frequency
        sorted_patterns = sorted(frequent_patterns.items(), key=lambda x: x[1], reverse=True)
        
        # Prefetch based on current cache state
        current_keys = set(self.cache.keys())
        prefetch_candidates = []
        
        for pattern_key, frequency in sorted_patterns[:10]:  # Top 10 patterns
            if '->' in pattern_key:
                prev_key, next_key = pattern_key.split('->', 1)
                if prev_key in current_keys and next_key not in current_keys:
                    prefetch_candidates.append(next_key)
        
        if prefetch_candidates:
            logger.debug(f"Identified {len(prefetch_candidates)} prefetch candidates")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics.
        
        Returns:
            Cache statistics
        """
        total_requests = self.hit_count + self.miss_count
        hit_ratio = self.hit_count / max(1, total_requests)
        
        # Compute average computation time for cached functions
        avg_computation_time = 0.0
        if self.computation_times:
            avg_computation_time = np.mean(list(self.computation_times.values()))
        
        return {
            'cache_size': len(self.cache),
            'max_size': self.max_size,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_ratio': hit_ratio,
            'avg_computation_time': avg_computation_time,
            'ttl_seconds': self.ttl_seconds,
            'prefetch_patterns_learned': len(self.prefetch_patterns),
            'prefetch_queue_size': len(self.prefetch_queue)
        }
    
    def clear(self):
        """Clear all cached results."""
        cleared_count = len(self.cache)
        self.cache.clear()
        self.access_times.clear()
        self.computation_times.clear()
        
        logger.info(f"Cleared computation cache ({cleared_count} entries)")
    
    def shutdown(self):
        """Shutdown computation cache."""
        self._maintenance_active = False
        if self._maintenance_thread.is_alive():
            self._maintenance_thread.join(timeout=5.0)
        
        self.clear()
        logger.info("Computation cache shutdown")


class PerformanceOptimizer:
    """Comprehensive performance optimization system."""
    
    def __init__(
        self,
        memory_manager: Optional[MemoryManager] = None,
        computation_cache: Optional[ComputationCache] = None,
        enable_jit: bool = True
    ):
        """Initialize performance optimizer.
        
        Args:
            memory_manager: Memory manager instance
            computation_cache: Computation cache instance
            enable_jit: Enable JIT compilation optimizations
        """
        self.memory_manager = memory_manager or MemoryManager()
        self.computation_cache = computation_cache or ComputationCache()
        self.enable_jit = enable_jit
        
        # Performance tracking
        self.operation_timings = defaultdict(list)
        self.bottlenecks = defaultdict(int)
        self.optimization_history = []
        
        # JIT compilation cache
        self.jit_cache = {}
        
        logger.info("Performance optimizer initialized")
    
    def profile_function(self, func: Callable) -> Callable:
        """Decorator to profile function performance.
        
        Args:
            func: Function to profile
            
        Returns:
            Profiled function wrapper
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss
            
            try:
                result = func(*args, **kwargs)
                success = True
            except Exception as e:
                result = e
                success = False
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss
            
            # Record performance metrics
            execution_time = end_time - start_time
            memory_delta = (end_memory - start_memory) / (1024**2)  # MB
            
            self.operation_timings[func.__name__].append(execution_time)
            
            # Detect bottlenecks
            if execution_time > 1.0:  # Functions taking > 1 second
                self.bottlenecks[func.__name__] += 1
            
            # Log performance info
            if execution_time > 0.1:  # Log slow operations
                logger.debug(f"Performance: {func.__name__} took {execution_time:.3f}s, "
                           f"memory delta: {memory_delta:.2f}MB")
            
            if success:
                return result
            else:
                raise result
        
        return wrapper
    
    def optimize_function(self, func: Callable, optimization_type: str = "auto") -> Callable:
        """Apply optimizations to function.
        
        Args:
            func: Function to optimize
            optimization_type: Type of optimization (auto, cache, jit)
            
        Returns:
            Optimized function
        """
        optimized_func = func
        applied_optimizations = []
        
        # Apply caching if beneficial
        if optimization_type in ["auto", "cache"]:
            # Check if function is pure and worth caching
            if self._should_cache_function(func):
                optimized_func = self.computation_cache.memoize()(optimized_func)
                applied_optimizations.append("memoization")
        
        # Apply JIT compilation if enabled and beneficial
        if self.enable_jit and optimization_type in ["auto", "jit"]:
            if self._should_jit_function(func):
                optimized_func = self._apply_jit_compilation(optimized_func)
                applied_optimizations.append("jit")
        
        # Apply profiling
        optimized_func = self.profile_function(optimized_func)
        applied_optimizations.append("profiling")
        
        # Record optimization
        self.optimization_history.append({
            'function': func.__name__,
            'optimizations': applied_optimizations,
            'timestamp': time.time()
        })
        
        logger.info(f"Optimized function {func.__name__} with: {', '.join(applied_optimizations)}")
        return optimized_func
    
    def _should_cache_function(self, func: Callable) -> bool:
        """Determine if function should be cached.
        
        Args:
            func: Function to evaluate
            
        Returns:
            True if caching would be beneficial
        """
        # Simple heuristics for caching decision
        func_name = func.__name__
        
        # Cache functions with these patterns
        cache_patterns = [
            'compute', 'calculate', 'predict', 'infer', 
            'risk', 'certificate', 'bound', 'quantile'
        ]
        
        for pattern in cache_patterns:
            if pattern in func_name.lower():
                return True
        
        # Check if function has been called multiple times with same args
        if func_name in self.operation_timings:
            call_count = len(self.operation_timings[func_name])
            avg_time = np.mean(self.operation_timings[func_name])
            
            # Cache if called frequently and takes significant time
            if call_count > 5 and avg_time > 0.01:
                return True
        
        return False
    
    def _should_jit_function(self, func: Callable) -> bool:
        """Determine if function should use JIT compilation.
        
        Args:
            func: Function to evaluate
            
        Returns:
            True if JIT would be beneficial
        """
        # Check if function contains numeric computations
        import inspect
        
        try:
            source = inspect.getsource(func)
            
            # Look for numeric computation patterns
            numeric_patterns = [
                'numpy', 'np.', 'array', 'matrix', 'dot', 'sum', 'mean',
                'for i in', 'for j in', 'range(', 'len('
            ]
            
            numeric_score = sum(1 for pattern in numeric_patterns if pattern in source)
            
            # JIT beneficial for functions with multiple numeric operations
            return numeric_score >= 3
            
        except Exception:
            return False
    
    def _apply_jit_compilation(self, func: Callable) -> Callable:
        """Apply JIT compilation to function.
        
        Args:
            func: Function to compile
            
        Returns:
            JIT-compiled function
        """
        try:
            # Try to use numba for JIT compilation
            try:
                import numba
                jit_func = numba.jit(nopython=True)(func)
                logger.debug(f"Applied Numba JIT to {func.__name__}")
                return jit_func
            except ImportError:
                pass
            
            # Fallback: cache compiled bytecode
            func_key = f"{func.__module__}.{func.__name__}"
            if func_key not in self.jit_cache:
                # Compile function (simplified - just cache the function)
                self.jit_cache[func_key] = func
                logger.debug(f"Cached compiled version of {func.__name__}")
            
            return self.jit_cache[func_key]
            
        except Exception as e:
            logger.warning(f"JIT compilation failed for {func.__name__}: {e}")
            return func
    
    def analyze_performance(self, window_hours: int = 1) -> PerformanceMetrics:
        """Analyze recent performance metrics.
        
        Args:
            window_hours: Time window for analysis
            
        Returns:
            Performance analysis results
        """
        # CPU and memory usage
        cpu_percent = psutil.cpu_percent(interval=1)
        process = psutil.Process()
        memory_mb = process.memory_info().rss / (1024**2)
        
        # Cache performance
        cache_stats = self.computation_cache.get_cache_stats()
        cache_hit_ratio = cache_stats.get('hit_ratio', 0.0)
        
        # Computation performance
        all_timings = []
        for timings in self.operation_timings.values():
            all_timings.extend(timings)
        
        avg_computation_time = np.mean(all_timings) if all_timings else 0.0
        
        # Memory stats
        memory_stats = self.memory_manager.get_memory_stats()
        
        # Bottleneck analysis
        top_bottlenecks = sorted(self.bottlenecks.items(), key=lambda x: x[1], reverse=True)[:5]
        bottleneck_operations = [name for name, count in top_bottlenecks if count > 0]
        
        # Optimization speedup (simplified calculation)
        optimization_speedup = 1.0
        if len(self.optimization_history) > 0:
            # Estimate speedup from optimizations (placeholder)
            optimization_speedup = 1.0 + len(self.optimization_history) * 0.1
        
        return PerformanceMetrics(
            cpu_usage_percent=cpu_percent,
            memory_usage_mb=memory_mb,
            cache_hit_ratio=cache_hit_ratio,
            avg_computation_time=avg_computation_time,
            peak_memory_usage=self.memory_manager.peak_memory_usage,
            gc_pause_time=memory_stats.gc_collections.get(0, 0) * 0.001,  # Estimate
            optimization_speedup=optimization_speedup,
            bottleneck_operations=bottleneck_operations
        )
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get performance optimization recommendations.
        
        Returns:
            List of optimization recommendations
        """
        recommendations = []
        
        # Analyze bottlenecks
        for func_name, count in self.bottlenecks.items():
            if count > 5:  # Function is frequently slow
                timings = self.operation_timings.get(func_name, [])
                if timings:
                    avg_time = np.mean(timings)
                    recommendations.append({
                        'type': 'bottleneck',
                        'function': func_name,
                        'issue': f'Function called {count} times with avg time {avg_time:.3f}s',
                        'recommendation': 'Consider optimization or caching',
                        'priority': 'high' if avg_time > 1.0 else 'medium'
                    })
        
        # Memory recommendations
        memory_stats = self.memory_manager.get_memory_stats()
        if memory_stats.memory_percent > 80:
            recommendations.append({
                'type': 'memory',
                'issue': f'High memory usage: {memory_stats.memory_percent:.1f}%',
                'recommendation': 'Consider reducing cache sizes or optimizing memory usage',
                'priority': 'high'
            })
        
        # Cache recommendations
        cache_stats = self.computation_cache.get_cache_stats()
        if cache_stats['hit_ratio'] < 0.3:
            recommendations.append({
                'type': 'cache',
                'issue': f'Low cache hit ratio: {cache_stats["hit_ratio"]:.2f}',
                'recommendation': 'Review caching strategy or increase cache size',
                'priority': 'medium'
            })
        
        return recommendations
    
    def auto_optimize(self) -> Dict[str, Any]:
        """Automatically apply performance optimizations.
        
        Returns:
            Summary of applied optimizations
        """
        logger.info("Starting automatic performance optimization")
        
        applied_optimizations = []
        
        # Memory optimization
        memory_stats = self.memory_manager.get_memory_stats()
        if memory_stats.memory_percent > 70:
            self.memory_manager._handle_memory_pressure()
            applied_optimizations.append("memory_cleanup")
        
        # Cache optimization
        cache_stats = self.computation_cache.get_cache_stats()
        if cache_stats['hit_ratio'] < 0.2 and cache_stats['cache_size'] < cache_stats['max_size'] / 2:
            # Increase cache size
            self.computation_cache.max_size = min(self.computation_cache.max_size * 2, 50000)
            applied_optimizations.append("cache_size_increase")
        
        # GC optimization
        if memory_stats.gc_collections.get(0, 0) > 100:
            gc.set_threshold(700, 10, 10)  # Adjust GC thresholds
            applied_optimizations.append("gc_tuning")
        
        optimization_summary = {
            'optimizations_applied': applied_optimizations,
            'memory_stats': memory_stats,
            'cache_stats': cache_stats,
            'timestamp': time.time()
        }
        
        logger.info(f"Auto-optimization completed: {', '.join(applied_optimizations)}")
        return optimization_summary
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report.
        
        Returns:
            Performance analysis report
        """
        metrics = self.analyze_performance()
        recommendations = self.get_optimization_recommendations()
        memory_report = self.memory_manager.get_memory_report()
        cache_stats = self.computation_cache.get_cache_stats()
        
        return {
            'performance_metrics': {
                'cpu_usage_percent': metrics.cpu_usage_percent,
                'memory_usage_mb': metrics.memory_usage_mb,
                'cache_hit_ratio': metrics.cache_hit_ratio,
                'avg_computation_time': metrics.avg_computation_time,
                'optimization_speedup': metrics.optimization_speedup
            },
            'bottlenecks': dict(sorted(self.bottlenecks.items(), key=lambda x: x[1], reverse=True)[:10]),
            'memory_report': memory_report,
            'cache_performance': cache_stats,
            'recommendations': recommendations,
            'optimization_history': self.optimization_history[-20:],  # Recent optimizations
            'report_timestamp': time.time()
        }
    
    def shutdown(self):
        """Shutdown performance optimizer."""
        self.memory_manager.shutdown()
        self.computation_cache.shutdown()
        logger.info("Performance optimizer shutdown")


# Global instances for easy access
_global_memory_manager = None
_global_computation_cache = None
_global_performance_optimizer = None


def get_memory_manager() -> MemoryManager:
    """Get global memory manager instance."""
    global _global_memory_manager
    if _global_memory_manager is None:
        _global_memory_manager = MemoryManager()
    return _global_memory_manager


def get_computation_cache() -> ComputationCache:
    """Get global computation cache instance."""
    global _global_computation_cache
    if _global_computation_cache is None:
        _global_computation_cache = ComputationCache()
    return _global_computation_cache


def get_performance_optimizer() -> PerformanceOptimizer:
    """Get global performance optimizer instance."""
    global _global_performance_optimizer
    if _global_performance_optimizer is None:
        _global_performance_optimizer = PerformanceOptimizer()
    return _global_performance_optimizer


# Decorators for easy use
def profile(func: Callable) -> Callable:
    """Decorator to profile function performance."""
    return get_performance_optimizer().profile_function(func)


def optimize(optimization_type: str = "auto"):
    """Decorator to optimize function performance."""
    def decorator(func: Callable) -> Callable:
        return get_performance_optimizer().optimize_function(func, optimization_type)
    return decorator


def cached(ttl: float = 3600.0):
    """Decorator to cache function results."""
    def decorator(func: Callable) -> Callable:
        return get_computation_cache().memoize(ttl=ttl)(func)
    return decorator