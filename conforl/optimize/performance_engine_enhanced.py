"""Enhanced Performance Engine for ConfoRL Scaling."""

import time
import threading
import multiprocessing
import queue
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Callable, Tuple
import logging
from collections import deque, defaultdict
import numpy as np

from .cache import AdaptiveCache
from .concurrent import BatchProcessor
from ..utils.logging import get_logger

logger = get_logger(__name__)

class PerformanceEngineEnhanced:
    """Enhanced performance engine for scalable RL operations."""
    
    def __init__(
        self,
        max_workers: int = None,
        enable_multiprocessing: bool = True,
        cache_size: int = 10000,
        batch_size: int = 32,
        optimization_level: str = "balanced"  # conservative, balanced, aggressive
    ):
        """Initialize enhanced performance engine.
        
        Args:
            max_workers: Maximum number of worker threads/processes
            enable_multiprocessing: Whether to use multiprocessing for CPU-bound tasks
            cache_size: Size of performance cache
            batch_size: Default batch size for operations
            optimization_level: Level of optimization to apply
        """
        self.max_workers = max_workers or min(32, multiprocessing.cpu_count() * 2)
        self.enable_multiprocessing = enable_multiprocessing
        self.batch_size = batch_size
        self.optimization_level = optimization_level
        
        # Performance caches
        self.prediction_cache = AdaptiveCache(max_size=cache_size, ttl=300)
        self.computation_cache = AdaptiveCache(max_size=cache_size//2, ttl=600)
        self.batch_cache = AdaptiveCache(max_size=cache_size//4, ttl=120)
        
        # Executors for different types of work
        self.thread_executor = ThreadPoolExecutor(max_workers=self.max_workers)
        if self.enable_multiprocessing:
            try:
                self.process_executor = ProcessPoolExecutor(max_workers=min(8, multiprocessing.cpu_count()))
            except Exception as e:
                logger.warning(f"Could not create process executor: {e}")
                self.process_executor = None
        else:
            self.process_executor = None
        
        # Batch processors for different operations
        self.batch_processors = {
            'prediction': BatchProcessor(batch_size=batch_size),
            'training': BatchProcessor(batch_size=batch_size * 2),
            'evaluation': BatchProcessor(batch_size=batch_size // 2)
        }
        
        # Performance metrics
        self.metrics = {
            'cache_hits': 0,
            'cache_misses': 0,
            'batch_operations': 0,
            'parallel_operations': 0,
            'total_operations': 0,
            'total_time_saved': 0.0,
            'average_response_time': 0.0
        }
        
        # Adaptive optimization
        self.performance_history = deque(maxlen=1000)
        self.optimization_state = {
            'current_batch_size': batch_size,
            'current_cache_ttl': 300,
            'last_optimization': time.time()
        }
        
        self._lock = threading.RLock()
        
    def predict_batch_optimized(
        self,
        states: List[Any],
        predictor_func: Callable,
        use_cache: bool = True,
        parallel: bool = True
    ) -> List[Any]:
        """Optimized batch prediction with caching and parallelization."""
        start_time = time.time()
        
        try:
            with self._lock:
                self.metrics['total_operations'] += 1
                
            # Check cache first if enabled
            cached_results = {}
            uncached_states = []
            uncached_indices = []
            
            if use_cache:
                for i, state in enumerate(states):
                    cache_key = f"predict_{hash(str(state))}"
                    cached = self.prediction_cache.get(cache_key)
                    if cached is not None:
                        cached_results[i] = cached
                        with self._lock:
                            self.metrics['cache_hits'] += 1
                    else:
                        uncached_states.append(state)
                        uncached_indices.append(i)
                        with self._lock:
                            self.metrics['cache_misses'] += 1
            else:
                uncached_states = states
                uncached_indices = list(range(len(states)))
            
            # Process uncached states
            if uncached_states:
                if parallel and len(uncached_states) > 4 and self.thread_executor:
                    # Parallel processing
                    results = self._predict_parallel(uncached_states, predictor_func)
                    with self._lock:
                        self.metrics['parallel_operations'] += 1
                else:
                    # Sequential processing
                    results = [predictor_func(state) for state in uncached_states]
                
                # Cache new results
                if use_cache:
                    for state, result in zip(uncached_states, results):
                        cache_key = f"predict_{hash(str(state))}"
                        self.prediction_cache.put(cache_key, result)
                
                # Combine cached and new results
                final_results = [None] * len(states)
                for i, result in zip(uncached_indices, results):
                    final_results[i] = result
                for i, result in cached_results.items():
                    final_results[i] = result
            else:
                # All results were cached
                final_results = [cached_results[i] for i in range(len(states))]
            
            # Record performance
            end_time = time.time()
            operation_time = end_time - start_time
            self._record_performance('batch_predict', len(states), operation_time)
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error in optimized batch prediction: {e}")
            # Fallback to simple sequential processing
            return [predictor_func(state) for state in states]
    
    def _predict_parallel(self, states: List[Any], predictor_func: Callable) -> List[Any]:
        """Parallel prediction processing."""
        try:
            # Determine optimal chunk size
            chunk_size = max(1, len(states) // self.max_workers)
            chunks = [states[i:i + chunk_size] for i in range(0, len(states), chunk_size)]
            
            # Submit chunks to thread pool
            future_to_chunk = {}
            for chunk in chunks:
                future = self.thread_executor.submit(self._process_chunk, chunk, predictor_func)
                future_to_chunk[future] = chunk
            
            # Collect results
            results = []
            for future in as_completed(future_to_chunk):
                try:
                    chunk_results = future.result(timeout=30)  # 30 second timeout
                    results.extend(chunk_results)
                except Exception as e:
                    logger.error(f"Error in parallel chunk processing: {e}")
                    # Fallback: process chunk sequentially
                    chunk = future_to_chunk[future]
                    chunk_results = [predictor_func(state) for state in chunk]
                    results.extend(chunk_results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in parallel processing: {e}")
            # Fallback to sequential
            return [predictor_func(state) for state in states]
    
    def _process_chunk(self, chunk: List[Any], predictor_func: Callable) -> List[Any]:
        """Process a chunk of states."""
        return [predictor_func(state) for state in chunk]
    
    def training_step_optimized(
        self,
        training_func: Callable,
        batch_data: Any,
        use_cache: bool = True
    ) -> Any:
        """Optimized training step with caching and resource management."""
        start_time = time.time()
        
        try:
            # Check computation cache
            if use_cache:
                cache_key = f"train_{hash(str(batch_data))}"
                cached_result = self.computation_cache.get(cache_key)
                if cached_result is not None:
                    with self._lock:
                        self.metrics['cache_hits'] += 1
                    return cached_result
                else:
                    with self._lock:
                        self.metrics['cache_misses'] += 1
            
            # Perform training step
            if self.optimization_level == "aggressive" and self.process_executor:
                # Use process executor for CPU-intensive training
                future = self.process_executor.submit(training_func, batch_data)
                result = future.result(timeout=60)  # 1 minute timeout
                with self._lock:
                    self.metrics['parallel_operations'] += 1
            else:
                result = training_func(batch_data)
            
            # Cache result
            if use_cache:
                cache_key = f"train_{hash(str(batch_data))}"
                self.computation_cache.put(cache_key, result)
            
            # Record performance
            end_time = time.time()
            operation_time = end_time - start_time
            self._record_performance('training_step', 1, operation_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in optimized training step: {e}")
            # Fallback to direct call
            return training_func(batch_data)
    
    def adaptive_batch_processing(
        self,
        items: List[Any],
        process_func: Callable,
        operation_type: str = "prediction"
    ) -> List[Any]:
        """Adaptive batch processing that optimizes batch size based on performance."""
        if operation_type not in self.batch_processors:
            self.batch_processors[operation_type] = BatchProcessor(batch_size=self.batch_size)
        
        processor = self.batch_processors[operation_type]
        
        # Process in adaptive batches
        results = []
        start_time = time.time()
        
        for batch in processor.create_batches(items):
            batch_start = time.time()
            batch_results = self.predict_batch_optimized(
                batch, process_func, use_cache=True, parallel=True
            )
            batch_time = time.time() - batch_start
            
            results.extend(batch_results)
            
            # Adapt batch size based on performance
            if operation_type in self.batch_processors:
                self._adapt_batch_size(operation_type, len(batch), batch_time)
        
        total_time = time.time() - start_time
        self._record_performance(f"adaptive_batch_{operation_type}", len(items), total_time)
        
        with self._lock:
            self.metrics['batch_operations'] += 1
        
        return results
    
    def _adapt_batch_size(self, operation_type: str, batch_size: int, processing_time: float):
        """Adapt batch size based on processing performance."""
        processor = self.batch_processors[operation_type]
        
        # Calculate throughput
        throughput = batch_size / processing_time if processing_time > 0 else 0
        
        # Adaptive logic
        if throughput > 0:
            if processing_time < 0.1:  # Very fast, can increase batch size
                new_batch_size = min(batch_size * 1.2, self.batch_size * 4)
            elif processing_time > 1.0:  # Too slow, decrease batch size
                new_batch_size = max(batch_size * 0.8, 1)
            else:
                new_batch_size = batch_size
            
            processor.batch_size = int(new_batch_size)
    
    def _record_performance(self, operation: str, items_count: int, time_taken: float):
        """Record performance metrics for optimization."""
        with self._lock:
            self.performance_history.append({
                'operation': operation,
                'items_count': items_count,
                'time_taken': time_taken,
                'throughput': items_count / time_taken if time_taken > 0 else 0,
                'timestamp': time.time()
            })
            
            # Update average response time
            total_time = sum(h['time_taken'] for h in self.performance_history)
            self.metrics['average_response_time'] = total_time / len(self.performance_history)
    
    def optimize_parameters(self) -> Dict[str, Any]:
        """Automatically optimize parameters based on performance history."""
        if len(self.performance_history) < 10:
            return {}
        
        try:
            # Analyze recent performance
            recent_history = list(self.performance_history)[-100:]
            
            # Calculate average throughput by operation type
            operation_throughput = defaultdict(list)
            for record in recent_history:
                operation_throughput[record['operation']].append(record['throughput'])
            
            optimizations = {}
            
            # Optimize cache TTL based on hit rate
            hit_rate = self.metrics['cache_hits'] / max(1, self.metrics['cache_hits'] + self.metrics['cache_misses'])
            if hit_rate > 0.8:
                # High hit rate, can increase TTL
                new_ttl = min(self.optimization_state['current_cache_ttl'] * 1.5, 3600)
                optimizations['cache_ttl'] = new_ttl
            elif hit_rate < 0.3:
                # Low hit rate, decrease TTL
                new_ttl = max(self.optimization_state['current_cache_ttl'] * 0.7, 60)
                optimizations['cache_ttl'] = new_ttl
            
            # Optimize batch sizes based on throughput
            for operation, throughputs in operation_throughput.items():
                if throughputs:
                    avg_throughput = np.mean(throughputs)
                    if avg_throughput > 50:  # High throughput
                        optimizations[f'{operation}_batch_size'] = 'increase'
                    elif avg_throughput < 10:  # Low throughput
                        optimizations[f'{operation}_batch_size'] = 'decrease'
            
            self.optimization_state['last_optimization'] = time.time()
            return optimizations
            
        except Exception as e:
            logger.error(f"Error in parameter optimization: {e}")
            return {}
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        with self._lock:
            hit_rate = (
                self.metrics['cache_hits'] / 
                max(1, self.metrics['cache_hits'] + self.metrics['cache_misses'])
            )
            
            return {
                **self.metrics,
                'cache_hit_rate': hit_rate,
                'optimization_level': self.optimization_level,
                'max_workers': self.max_workers,
                'current_batch_sizes': {
                    name: processor.batch_size 
                    for name, processor in self.batch_processors.items()
                },
                'performance_history_length': len(self.performance_history),
                'last_optimization': self.optimization_state['last_optimization']
            }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on performance engine."""
        try:
            # Test thread executor
            thread_healthy = True
            try:
                future = self.thread_executor.submit(lambda: "test")
                result = future.result(timeout=1)
                thread_healthy = result == "test"
            except Exception:
                thread_healthy = False
            
            # Test process executor if available
            process_healthy = True
            if self.process_executor:
                try:
                    future = self.process_executor.submit(lambda: "test")
                    result = future.result(timeout=2)
                    process_healthy = result == "test"
                except Exception:
                    process_healthy = False
            
            # Check cache health
            cache_healthy = all([
                self.prediction_cache.get_stats()['hit_rate'] >= 0,
                self.computation_cache.get_stats()['hit_rate'] >= 0
            ])
            
            overall_healthy = thread_healthy and process_healthy and cache_healthy
            
            return {
                'healthy': overall_healthy,
                'thread_executor_healthy': thread_healthy,
                'process_executor_healthy': process_healthy,
                'cache_healthy': cache_healthy,
                'total_operations': self.metrics['total_operations']
            }
            
        except Exception as e:
            logger.error(f"Error in performance engine health check: {e}")
            return {'healthy': False, 'error': str(e)}
    
    def cleanup(self):
        """Clean up resources."""
        try:
            if hasattr(self, 'thread_executor'):
                self.thread_executor.shutdown(wait=True)
            if hasattr(self, 'process_executor') and self.process_executor:
                self.process_executor.shutdown(wait=True)
        except Exception as e:
            logger.error(f"Error cleaning up performance engine: {e}")

# Global performance engine instance
performance_engine = PerformanceEngineEnhanced()

# Convenience functions
def predict_batch_fast(states: List[Any], predictor_func: Callable) -> List[Any]:
    """Fast batch prediction using performance engine."""
    return performance_engine.predict_batch_optimized(states, predictor_func)

def train_step_fast(training_func: Callable, batch_data: Any) -> Any:
    """Fast training step using performance engine."""
    return performance_engine.training_step_optimized(training_func, batch_data)

def get_performance_metrics() -> Dict[str, Any]:
    """Get current performance metrics."""
    return performance_engine.get_performance_metrics()