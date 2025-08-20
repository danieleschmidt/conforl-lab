"""
High-performance optimization engine for ConfoRL.
Generation 3: Performance optimization, caching, and scalability.
"""

import time
import threading
import multiprocessing
from typing import Any, Dict, List, Optional, Callable, Tuple, Union
from collections import defaultdict, deque
from dataclasses import dataclass
from functools import wraps, lru_cache
import hashlib
import pickle

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    class np:
        @staticmethod
        def array(data): return data
        @staticmethod
        def mean(data): return sum(data) / len(data) if data else 0
        @staticmethod
        def std(data): 
            if not data: return 0
            mean_val = sum(data) / len(data)
            return (sum((x - mean_val)**2 for x in data) / len(data)) ** 0.5


@dataclass
class PerformanceMetrics:
    """Performance metrics for operations."""
    operation_name: str
    execution_time: float
    memory_usage: int
    cpu_usage: float
    cache_hit_rate: float
    throughput: float
    timestamp: float


class AdaptiveCache:
    """High-performance adaptive cache with usage pattern learning."""
    
    def __init__(
        self, 
        max_size: int = 10000, 
        ttl_seconds: int = 3600,
        adaptive_sizing: bool = True
    ):
        """Initialize adaptive cache.
        
        Args:
            max_size: Maximum cache size
            ttl_seconds: Time-to-live for cache entries
            adaptive_sizing: Enable adaptive cache sizing
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.adaptive_sizing = adaptive_sizing
        
        # Cache storage
        self._cache = {}
        self._access_times = {}
        self._access_counts = defaultdict(int)
        self._creation_times = {}
        
        # Performance tracking
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Adaptive sizing
        self._size_history = deque(maxlen=100)
        self._performance_history = deque(maxlen=100)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            current_time = time.time()
            
            if key not in self._cache:
                self.misses += 1
                return None
            
            # Check TTL
            if current_time - self._creation_times[key] > self.ttl_seconds:
                self._remove(key)
                self.misses += 1
                return None
            
            # Update access tracking
            self._access_times[key] = current_time
            self._access_counts[key] += 1
            self.hits += 1
            
            return self._cache[key]
    
    def put(self, key: str, value: Any) -> None:
        """Put value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        with self._lock:
            current_time = time.time()
            
            # Remove existing entry if present
            if key in self._cache:
                self._remove(key)
            
            # Check size limit and evict if necessary
            if len(self._cache) >= self.max_size:
                self._evict_lru()
            
            # Add new entry
            self._cache[key] = value
            self._creation_times[key] = current_time
            self._access_times[key] = current_time
            self._access_counts[key] = 1
            
            # Adaptive sizing
            if self.adaptive_sizing:
                self._update_adaptive_sizing()
    
    def _remove(self, key: str) -> None:
        """Remove entry from cache."""
        if key in self._cache:
            del self._cache[key]
            del self._creation_times[key]
            del self._access_times[key]
            del self._access_counts[key]
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._cache:
            return
        
        # Find LRU entry
        lru_key = min(
            self._access_times.keys(),
            key=lambda k: (self._access_counts[k], self._access_times[k])
        )
        
        self._remove(lru_key)
        self.evictions += 1
    
    def _update_adaptive_sizing(self) -> None:
        """Update cache size based on performance."""
        hit_rate = self.get_hit_rate()
        current_size = len(self._cache)
        
        self._size_history.append(current_size)
        self._performance_history.append(hit_rate)
        
        if len(self._performance_history) >= 10:
            # Adjust size based on hit rate trend
            recent_performance = list(self._performance_history)[-5:]
            avg_performance = sum(recent_performance) / len(recent_performance)
            
            if avg_performance < 0.7 and self.max_size < 50000:
                # Low hit rate, increase cache size
                self.max_size = min(self.max_size * 1.2, 50000)
            elif avg_performance > 0.95 and self.max_size > 1000:
                # Very high hit rate, can reduce size
                self.max_size = max(self.max_size * 0.9, 1000)
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate."""
        total_requests = self.hits + self.misses
        return self.hits / total_requests if total_requests > 0 else 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'hits': self.hits,
                'misses': self.misses,
                'evictions': self.evictions,
                'hit_rate': self.get_hit_rate(),
                'top_keys': sorted(
                    self._access_counts.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:10]
            }
    
    def clear(self) -> None:
        """Clear cache."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
            self._access_counts.clear()
            self._creation_times.clear()


class PerformanceProfiler:
    """Performance profiler for ConfoRL operations."""
    
    def __init__(self):
        """Initialize performance profiler."""
        self.metrics = defaultdict(list)
        self.active_operations = {}
        self._lock = threading.Lock()
    
    def start_operation(self, operation_name: str) -> str:
        """Start profiling an operation.
        
        Args:
            operation_name: Name of operation
            
        Returns:
            Operation ID for stopping profiling
        """
        operation_id = f"{operation_name}_{time.time()}_{threading.current_thread().ident}"
        
        with self._lock:
            self.active_operations[operation_id] = {
                'name': operation_name,
                'start_time': time.time(),
                'start_memory': self._get_memory_usage(),
                'start_cpu': self._get_cpu_usage()
            }
        
        return operation_id
    
    def stop_operation(self, operation_id: str) -> PerformanceMetrics:
        """Stop profiling an operation.
        
        Args:
            operation_id: Operation ID from start_operation
            
        Returns:
            Performance metrics for the operation
        """
        with self._lock:
            if operation_id not in self.active_operations:
                raise ValueError(f"Operation {operation_id} not found")
            
            op_data = self.active_operations.pop(operation_id)
            
            end_time = time.time()
            execution_time = end_time - op_data['start_time']
            memory_usage = self._get_memory_usage() - op_data['start_memory']
            cpu_usage = self._get_cpu_usage() - op_data['start_cpu']
            
            metrics = PerformanceMetrics(
                operation_name=op_data['name'],
                execution_time=execution_time,
                memory_usage=max(0, memory_usage),
                cpu_usage=max(0, cpu_usage),
                cache_hit_rate=0.0,  # Will be updated if cache is used
                throughput=1.0 / execution_time if execution_time > 0 else 0.0,
                timestamp=end_time
            )
            
            self.metrics[op_data['name']].append(metrics)
            
            # Keep only recent metrics
            if len(self.metrics[op_data['name']]) > 1000:
                self.metrics[op_data['name']] = self.metrics[op_data['name']][-500:]
            
            return metrics
    
    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss
        except ImportError:
            return 0
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        try:
            import psutil
            return psutil.cpu_percent()
        except ImportError:
            return 0.0
    
    def get_operation_stats(self, operation_name: str) -> Optional[Dict[str, Any]]:
        """Get statistics for an operation.
        
        Args:
            operation_name: Name of operation
            
        Returns:
            Operation statistics or None if not found
        """
        if operation_name not in self.metrics:
            return None
        
        metrics_list = self.metrics[operation_name]
        execution_times = [m.execution_time for m in metrics_list]
        memory_usages = [m.memory_usage for m in metrics_list]
        throughputs = [m.throughput for m in metrics_list]
        
        return {
            'operation_name': operation_name,
            'total_calls': len(metrics_list),
            'avg_execution_time': np.mean(execution_times),
            'min_execution_time': min(execution_times),
            'max_execution_time': max(execution_times),
            'p95_execution_time': np.percentile(execution_times, 95),
            'avg_memory_usage': np.mean(memory_usages),
            'avg_throughput': np.mean(throughputs),
            'total_execution_time': sum(execution_times)
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all operations."""
        return {
            name: self.get_operation_stats(name)
            for name in self.metrics.keys()
        }


class ConcurrentProcessor:
    """High-performance concurrent processing for ConfoRL operations."""
    
    def __init__(self, max_workers: Optional[int] = None):
        """Initialize concurrent processor.
        
        Args:
            max_workers: Maximum number of worker threads/processes
        """
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self.thread_pool = None
        self.process_pool = None
        self._lock = threading.Lock()
        
        # Performance tracking
        self.task_counts = defaultdict(int)
        self.task_times = defaultdict(list)
    
    def map_concurrent(
        self, 
        func: Callable, 
        items: List[Any], 
        use_processes: bool = False,
        chunk_size: Optional[int] = None
    ) -> List[Any]:
        """Map function over items concurrently.
        
        Args:
            func: Function to apply
            items: Items to process
            use_processes: Use processes instead of threads
            chunk_size: Chunk size for processing
            
        Returns:
            List of results
        """
        if not items:
            return []
        
        start_time = time.time()
        
        try:
            if use_processes:
                import multiprocessing as mp
                with mp.Pool(self.max_workers) as pool:
                    if chunk_size:
                        results = pool.map(func, items, chunk_size)
                    else:
                        results = pool.map(func, items)
            else:
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    results = list(executor.map(func, items))
            
            # Track performance
            execution_time = time.time() - start_time
            with self._lock:
                self.task_counts['map_concurrent'] += len(items)
                self.task_times['map_concurrent'].append(execution_time)
            
            return results
            
        except Exception as e:
            # Fallback to sequential processing
            results = [func(item) for item in items]
            return results
    
    def submit_task(
        self, 
        func: Callable, 
        *args, 
        use_processes: bool = False,
        **kwargs
    ) -> Any:
        """Submit a single task for concurrent execution.
        
        Args:
            func: Function to execute
            *args: Function arguments
            use_processes: Use processes instead of threads
            **kwargs: Function keyword arguments
            
        Returns:
            Future object or immediate result
        """
        start_time = time.time()
        
        try:
            if use_processes:
                import multiprocessing as mp
                with mp.Pool(1) as pool:
                    result = pool.apply(func, args, kwargs)
            else:
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(func, *args, **kwargs)
                    result = future.result()
            
            # Track performance
            execution_time = time.time() - start_time
            with self._lock:
                self.task_counts['submit_task'] += 1
                self.task_times['submit_task'].append(execution_time)
            
            return result
            
        except Exception as e:
            # Fallback to direct execution
            return func(*args, **kwargs)
    
    def batch_process(
        self,
        func: Callable,
        items: List[Any],
        batch_size: int = 100,
        use_processes: bool = False
    ) -> List[Any]:
        """Process items in batches for optimal performance.
        
        Args:
            func: Function to apply to each batch
            items: Items to process
            batch_size: Size of each batch
            use_processes: Use processes instead of threads
            
        Returns:
            List of results
        """
        if not items:
            return []
        
        # Create batches
        batches = [
            items[i:i + batch_size]
            for i in range(0, len(items), batch_size)
        ]
        
        # Process batches concurrently
        batch_results = self.map_concurrent(func, batches, use_processes)
        
        # Flatten results
        results = []
        for batch_result in batch_results:
            if isinstance(batch_result, list):
                results.extend(batch_result)
            else:
                results.append(batch_result)
        
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        with self._lock:
            stats = {}
            for task_type, times in self.task_times.items():
                if times:
                    stats[task_type] = {
                        'total_tasks': self.task_counts[task_type],
                        'avg_time': np.mean(times),
                        'min_time': min(times),
                        'max_time': max(times),
                        'total_time': sum(times)
                    }
            return stats


# Global instances
_adaptive_cache = AdaptiveCache()
_performance_profiler = PerformanceProfiler()
_concurrent_processor = ConcurrentProcessor()


def cached(ttl: int = 3600, key_func: Optional[Callable] = None):
    """Decorator for caching function results.
    
    Args:
        ttl: Time-to-live in seconds
        key_func: Function to generate cache key
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                key_data = (func.__name__, args, tuple(sorted(kwargs.items())))
                cache_key = hashlib.md5(str(key_data).encode()).hexdigest()
            
            # Try cache first
            cached_result = _adaptive_cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Cache result
            _adaptive_cache.put(cache_key, result)
            
            return result
        
        return wrapper
    return decorator


def profile_performance(operation_name: str):
    """Decorator for performance profiling.
    
    Args:
        operation_name: Name of operation to profile
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            op_id = _performance_profiler.start_operation(operation_name)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                _performance_profiler.stop_operation(op_id)
        
        return wrapper
    return decorator


def get_cache_stats() -> Dict[str, Any]:
    """Get global cache statistics."""
    return _adaptive_cache.get_stats()


def get_performance_stats() -> Dict[str, Any]:
    """Get global performance statistics."""
    return _performance_profiler.get_all_stats()


def clear_cache():
    """Clear global cache."""
    _adaptive_cache.clear()


def optimize_function(func: Callable, cache_ttl: int = 3600) -> Callable:
    """Optimize function with caching and profiling.
    
    Args:
        func: Function to optimize
        cache_ttl: Cache time-to-live
        
    Returns:
        Optimized function
    """
    @cached(ttl=cache_ttl)
    @profile_performance(func.__name__)
    @wraps(func)
    def optimized_func(*args, **kwargs):
        return func(*args, **kwargs)
    
    return optimized_func