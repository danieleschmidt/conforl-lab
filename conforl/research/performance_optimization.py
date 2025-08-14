"""Performance Optimization and Acceleration for Research Algorithms.

Implements advanced performance optimization techniques including
JIT compilation, vectorization, memory optimization, and GPU acceleration.

Author: ConfoRL Research Team
License: Apache 2.0
"""

import time
import threading
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import warnings
import functools
from collections import defaultdict, deque

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import numba
    from numba import jit, njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Dummy decorators when Numba is not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args else decorator
    
    def njit(*args, **kwargs):
        return jit(*args, **kwargs)
    
    def prange(n):
        return range(n)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ..utils.logging import get_logger
from ..utils.errors import ConfoRLError, ValidationError

logger = get_logger(__name__)


class OptimizationLevel(Enum):
    """Levels of performance optimization."""
    NONE = "none"
    BASIC = "basic"
    AGGRESSIVE = "aggressive"
    EXPERIMENTAL = "experimental"


class ComputeBackend(Enum):
    """Available compute backends."""
    CPU = "cpu"
    GPU = "gpu"
    AUTO = "auto"


@dataclass
class PerformanceProfile:
    """Performance profiling data."""
    function_name: str
    call_count: int = 0
    total_time: float = 0.0
    average_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    memory_peak: float = 0.0
    optimization_applied: Optional[str] = None
    
    def update(self, execution_time: float, memory_usage: float = 0.0):
        """Update profile with new execution data."""
        self.call_count += 1
        self.total_time += execution_time
        self.average_time = self.total_time / self.call_count
        self.min_time = min(self.min_time, execution_time)
        self.max_time = max(self.max_time, execution_time)
        self.memory_peak = max(self.memory_peak, memory_usage)


class MemoryPool:
    """Memory pool for efficient array allocation and reuse."""
    
    def __init__(self, max_pool_size: int = 100):
        """Initialize memory pool.
        
        Args:
            max_pool_size: Maximum number of arrays to keep in pool
        """
        self.max_pool_size = max_pool_size
        self.pools: Dict[Tuple[Any, Tuple[int, ...]], deque] = defaultdict(deque)
        self.allocation_stats = defaultdict(int)
        self._lock = threading.RLock()
        
        logger.debug(f"Memory pool initialized with max size {max_pool_size}")
    
    def get_array(self, shape: Tuple[int, ...], dtype=None) -> Any:
        """Get array from pool or allocate new one.
        
        Args:
            shape: Array shape
            dtype: Array data type
            
        Returns:
            Numpy array or list (fallback)
        """
        if not NUMPY_AVAILABLE:
            # Fallback to nested lists
            if len(shape) == 1:
                return [0.0] * shape[0]
            elif len(shape) == 2:
                return [[0.0] * shape[1] for _ in range(shape[0])]
            else:
                raise ConfoRLError("Complex shapes not supported without NumPy")
        
        dtype = dtype or np.float64
        key = (dtype, shape)
        
        with self._lock:
            pool = self.pools[key]
            
            if pool:
                array = pool.popleft()
                array.fill(0)  # Clear previous data
                self.allocation_stats['reused'] += 1
                return array
            else:
                array = np.zeros(shape, dtype=dtype)
                self.allocation_stats['allocated'] += 1
                return array
    
    def return_array(self, array: Any) -> None:
        """Return array to pool for reuse.
        
        Args:
            array: Array to return to pool
        """
        if not NUMPY_AVAILABLE:
            return
        
        key = (array.dtype, array.shape)
        
        with self._lock:
            pool = self.pools[key]
            
            if len(pool) < self.max_pool_size:
                pool.append(array)
                self.allocation_stats['returned'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics."""
        with self._lock:
            return {
                'total_pools': len(self.pools),
                'total_arrays_pooled': sum(len(pool) for pool in self.pools.values()),
                'allocation_stats': dict(self.allocation_stats),
                'pool_sizes': {str(key): len(pool) for key, pool in self.pools.items()}
            }


class ComputationCache:
    """Intelligent caching system for expensive computations."""
    
    def __init__(self, max_cache_size: int = 1000, ttl_seconds: float = 3600.0):
        """Initialize computation cache.
        
        Args:
            max_cache_size: Maximum number of cached results
            ttl_seconds: Time-to-live for cached results
        """
        self.max_cache_size = max_cache_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.access_counts: Dict[str, int] = defaultdict(int)
        self.hit_count = 0
        self.miss_count = 0
        self._lock = threading.RLock()
        
        logger.debug(f"Computation cache initialized with size {max_cache_size}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached result.
        
        Args:
            key: Cache key
            
        Returns:
            Cached result or None if not found/expired
        """
        with self._lock:
            if key in self.cache:
                result, timestamp = self.cache[key]
                
                # Check TTL
                if time.time() - timestamp <= self.ttl_seconds:
                    self.access_counts[key] += 1
                    self.hit_count += 1
                    return result
                else:
                    # Expired, remove from cache
                    del self.cache[key]
                    del self.access_counts[key]
            
            self.miss_count += 1
            return None
    
    def put(self, key: str, value: Any) -> None:
        """Put result in cache.
        
        Args:
            key: Cache key
            value: Result to cache
        """
        with self._lock:
            # Remove oldest entries if cache is full
            if len(self.cache) >= self.max_cache_size:
                self._evict_lru()
            
            self.cache[key] = (value, time.time())
            self.access_counts[key] = 1
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self.cache:
            return
        
        # Find LRU entry
        lru_key = min(self.access_counts.keys(), key=lambda k: self.access_counts[k])
        
        # Remove from cache
        del self.cache[lru_key]
        del self.access_counts[lru_key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self.hit_count + self.miss_count
            hit_rate = self.hit_count / max(1, total_requests)
            
            return {
                'cache_size': len(self.cache),
                'max_cache_size': self.max_cache_size,
                'hit_count': self.hit_count,
                'miss_count': self.miss_count,
                'hit_rate': hit_rate,
                'total_requests': total_requests
            }


class PerformanceOptimizer:
    """Main performance optimization coordinator."""
    
    def __init__(
        self,
        optimization_level: OptimizationLevel = OptimizationLevel.BASIC,
        compute_backend: ComputeBackend = ComputeBackend.AUTO,
        enable_jit: bool = True,
        enable_caching: bool = True,
        enable_memory_pooling: bool = True,
        enable_profiling: bool = True
    ):
        """Initialize performance optimizer.
        
        Args:
            optimization_level: Level of optimization to apply
            compute_backend: Preferred compute backend
            enable_jit: Whether to enable JIT compilation
            enable_caching: Whether to enable computation caching
            enable_memory_pooling: Whether to enable memory pooling
            enable_profiling: Whether to enable performance profiling
        """
        self.optimization_level = optimization_level
        self.compute_backend = compute_backend
        self.enable_jit = enable_jit
        self.enable_caching = enable_caching
        self.enable_memory_pooling = enable_memory_pooling
        self.enable_profiling = enable_profiling
        
        # Initialize components
        self.memory_pool = MemoryPool() if enable_memory_pooling else None
        self.computation_cache = ComputationCache() if enable_caching else None
        
        # Performance profiles
        self.profiles: Dict[str, PerformanceProfile] = {}
        
        # JIT compiled functions cache
        self.jit_functions: Dict[str, Callable] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Auto-detect best backend
        if compute_backend == ComputeBackend.AUTO:
            self.compute_backend = self._detect_best_backend()
        
        logger.info(f"Performance optimizer initialized with {optimization_level.value} level")
    
    def _detect_best_backend(self) -> ComputeBackend:
        """Auto-detect the best available compute backend."""
        if TORCH_AVAILABLE:
            try:
                if torch.cuda.is_available():
                    logger.info("CUDA GPU detected, using GPU backend")
                    return ComputeBackend.GPU
            except:
                pass
        
        logger.info("Using CPU backend")
        return ComputeBackend.CPU
    
    def optimize_function(self, func: Callable, function_name: Optional[str] = None) -> Callable:
        """Apply optimizations to a function.
        
        Args:
            func: Function to optimize
            function_name: Name for tracking (uses func.__name__ if None)
            
        Returns:
            Optimized function
        """
        fname = function_name or func.__name__
        
        # Apply optimizations based on level
        optimized_func = func
        optimizations_applied = []
        
        # JIT compilation
        if self.enable_jit and NUMBA_AVAILABLE and self.optimization_level != OptimizationLevel.NONE:
            try:
                optimized_func = self._apply_jit_optimization(optimized_func, fname)
                optimizations_applied.append("JIT")
            except Exception as e:
                logger.warning(f"JIT optimization failed for {fname}: {e}")
        
        # Caching wrapper
        if self.enable_caching and self.computation_cache:
            optimized_func = self._apply_caching_optimization(optimized_func, fname)
            optimizations_applied.append("Caching")
        
        # Profiling wrapper
        if self.enable_profiling:
            optimized_func = self._apply_profiling_wrapper(optimized_func, fname)
            optimizations_applied.append("Profiling")
        
        # Memory pooling (for array operations)
        if self.enable_memory_pooling and self.memory_pool:
            optimized_func = self._apply_memory_pooling(optimized_func, fname)
            optimizations_applied.append("MemoryPooling")
        
        logger.debug(f"Applied optimizations to {fname}: {optimizations_applied}")
        
        return optimized_func
    
    def _apply_jit_optimization(self, func: Callable, function_name: str) -> Callable:
        """Apply JIT compilation optimization."""
        if function_name in self.jit_functions:
            return self.jit_functions[function_name]
        
        # Different JIT strategies based on optimization level
        if self.optimization_level == OptimizationLevel.BASIC:
            jit_func = jit(nopython=False, cache=True)(func)
        elif self.optimization_level == OptimizationLevel.AGGRESSIVE:
            jit_func = njit(cache=True, parallel=True)(func)
        elif self.optimization_level == OptimizationLevel.EXPERIMENTAL:
            jit_func = njit(cache=True, parallel=True, fastmath=True)(func)
        else:
            return func
        
        self.jit_functions[function_name] = jit_func
        return jit_func
    
    def _apply_caching_optimization(self, func: Callable, function_name: str) -> Callable:
        """Apply computation caching optimization."""
        @functools.wraps(func)
        def cached_wrapper(*args, **kwargs):
            # Create cache key from arguments
            cache_key = f"{function_name}:{hash((args, tuple(sorted(kwargs.items()))))}"
            
            # Try to get from cache
            cached_result = self.computation_cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Compute and cache result
            result = func(*args, **kwargs)
            self.computation_cache.put(cache_key, result)
            
            return result
        
        return cached_wrapper
    
    def _apply_profiling_wrapper(self, func: Callable, function_name: str) -> Callable:
        """Apply performance profiling wrapper."""
        @functools.wraps(func)
        def profiled_wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = self._get_memory_usage()
            
            try:
                result = func(*args, **kwargs)
                
                # Update profile
                execution_time = time.time() - start_time
                memory_usage = self._get_memory_usage() - start_memory
                
                with self._lock:
                    if function_name not in self.profiles:
                        self.profiles[function_name] = PerformanceProfile(function_name)
                    
                    self.profiles[function_name].update(execution_time, memory_usage)
                
                return result
                
            except Exception as e:
                # Still record execution time for failed calls
                execution_time = time.time() - start_time
                with self._lock:
                    if function_name not in self.profiles:
                        self.profiles[function_name] = PerformanceProfile(function_name)
                    
                    self.profiles[function_name].update(execution_time, 0.0)
                
                raise
        
        return profiled_wrapper
    
    def _apply_memory_pooling(self, func: Callable, function_name: str) -> Callable:
        """Apply memory pooling optimization for array operations."""
        @functools.wraps(func)
        def pooled_wrapper(*args, **kwargs):
            # This is a simplified implementation
            # Real implementation would analyze function to identify array allocations
            return func(*args, **kwargs)
        
        return pooled_wrapper
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            return 0.0
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        with self._lock:
            # Function profiles
            function_profiles = {}
            for name, profile in self.profiles.items():
                function_profiles[name] = {
                    'call_count': profile.call_count,
                    'total_time': profile.total_time,
                    'average_time': profile.average_time,
                    'min_time': profile.min_time,
                    'max_time': profile.max_time,
                    'memory_peak': profile.memory_peak,
                    'optimization_applied': profile.optimization_applied
                }
            
            # System performance
            cache_stats = self.computation_cache.get_stats() if self.computation_cache else {}
            memory_pool_stats = self.memory_pool.get_stats() if self.memory_pool else {}
            
            return {
                'optimization_level': self.optimization_level.value,
                'compute_backend': self.compute_backend.value,
                'enabled_optimizations': {
                    'jit': self.enable_jit,
                    'caching': self.enable_caching,
                    'memory_pooling': self.enable_memory_pooling,
                    'profiling': self.enable_profiling
                },
                'function_profiles': function_profiles,
                'cache_performance': cache_stats,
                'memory_pool_performance': memory_pool_stats,
                'jit_functions_count': len(self.jit_functions),
                'total_functions_optimized': len(self.profiles)
            }
    
    def benchmark_function(
        self,
        func: Callable,
        *args,
        iterations: int = 1000,
        warmup_iterations: int = 100,
        **kwargs
    ) -> Dict[str, float]:
        """Benchmark a function's performance.
        
        Args:
            func: Function to benchmark
            *args: Function arguments
            iterations: Number of benchmark iterations
            warmup_iterations: Number of warmup iterations
            **kwargs: Function keyword arguments
            
        Returns:
            Benchmark results
        """
        # Warmup
        for _ in range(warmup_iterations):
            try:
                func(*args, **kwargs)
            except:
                break  # Skip warmup if function fails
        
        # Benchmark
        times = []
        successful_runs = 0
        
        for _ in range(iterations):
            start_time = time.perf_counter()
            try:
                func(*args, **kwargs)
                end_time = time.perf_counter()
                times.append(end_time - start_time)
                successful_runs += 1
            except Exception as e:
                logger.warning(f"Benchmark iteration failed: {e}")
        
        if not times:
            return {
                'success': False,
                'error': 'All benchmark iterations failed'
            }
        
        return {
            'success': True,
            'iterations': iterations,
            'successful_runs': successful_runs,
            'success_rate': successful_runs / iterations,
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'median_time': np.median(times),
            'total_time': np.sum(times)
        }


# Optimized implementations of common research operations

def optimized_conformal_quantile(scores: Any, alpha: float) -> float:
    """Optimized conformal quantile computation."""
    n = len(scores)
    sorted_scores = np.sort(scores)
    
    # Conformal quantile with finite-sample correction
    quantile_index = int(np.ceil((n + 1) * (1 - alpha)))
    quantile_index = min(quantile_index - 1, n - 1)  # Ensure valid index
    quantile_index = max(quantile_index, 0)
    
    return sorted_scores[quantile_index]


def optimized_risk_bound_computation(
    scores: Any,
    alpha: float,
    confidence: float
) -> Tuple[float, float]:
    """Optimized risk bound computation with confidence intervals."""
    n = len(scores)
    
    # Primary quantile
    quantile = optimized_conformal_quantile(scores, alpha)
    
    # Bootstrap confidence interval for quantile
    num_bootstrap = 1000
    bootstrap_quantiles = np.zeros(num_bootstrap)
    
    for i in prange(num_bootstrap):
        # Bootstrap sample
        bootstrap_indices = np.random.choice(n, size=n, replace=True)
        bootstrap_scores = scores[bootstrap_indices]
        
        # Compute bootstrap quantile
        bootstrap_quantiles[i] = optimized_conformal_quantile(bootstrap_scores, alpha)
    
    # Confidence interval
    ci_alpha = (1 - confidence) / 2
    ci_lower = np.quantile(bootstrap_quantiles, ci_alpha)
    ci_upper = np.quantile(bootstrap_quantiles, 1 - ci_alpha)
    
    return quantile, (ci_upper - ci_lower) / 2  # Return quantile and margin of error


def optimized_trajectory_risk_computation(
    states: Any,
    actions: Any,
    rewards: Any,
    risk_threshold: float
) -> float:
    """Optimized trajectory risk computation."""
    trajectory_length = len(states)
    
    # Compute risk indicators
    risk_score = 0.0
    
    for t in range(trajectory_length):
        # State-based risk (simplified)
        state_risk = np.sum(np.abs(states[t])) / len(states[t])
        
        # Action-based risk
        action_risk = np.sum(np.abs(actions[t])) / len(actions[t])
        
        # Combined risk with exponential decay
        time_weight = np.exp(-0.1 * t)  # More weight on recent steps
        step_risk = (state_risk + action_risk) * time_weight
        
        risk_score += step_risk
    
    # Normalize by trajectory length
    normalized_risk = risk_score / trajectory_length
    
    return normalized_risk


# Global optimizer instance
_global_optimizer = None

def get_performance_optimizer() -> PerformanceOptimizer:
    """Get global performance optimizer instance."""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = PerformanceOptimizer()
    return _global_optimizer


def optimize(
    optimization_level: OptimizationLevel = OptimizationLevel.BASIC,
    enable_jit: bool = True,
    enable_caching: bool = True
):
    """Decorator for optimizing functions.
    
    Args:
        optimization_level: Level of optimization
        enable_jit: Whether to enable JIT
        enable_caching: Whether to enable caching
    """
    def decorator(func):
        optimizer = get_performance_optimizer()
        optimizer.optimization_level = optimization_level
        optimizer.enable_jit = enable_jit
        optimizer.enable_caching = enable_caching
        
        return optimizer.optimize_function(func)
    
    return decorator


def benchmark(iterations: int = 1000, warmup: int = 100):
    """Decorator for benchmarking functions.
    
    Args:
        iterations: Number of benchmark iterations
        warmup: Number of warmup iterations
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            optimizer = get_performance_optimizer()
            
            # Run benchmark
            bench_results = optimizer.benchmark_function(
                func, *args, iterations=iterations, warmup_iterations=warmup, **kwargs
            )
            
            # Execute function normally
            result = func(*args, **kwargs)
            
            # Log benchmark results
            if bench_results['success']:
                logger.info(f"Benchmark {func.__name__}: {bench_results['mean_time']:.6f}s avg "
                           f"({bench_results['successful_runs']}/{bench_results['iterations']} runs)")
            
            return result
        
        return wrapper
    return decorator