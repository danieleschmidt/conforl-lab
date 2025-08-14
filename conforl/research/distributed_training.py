"""Distributed Training and Scaling for Research Algorithms.

Implements distributed training, parallel processing, and auto-scaling
for large-scale research experiments in conformal RL.

Author: ConfoRL Research Team
License: Apache 2.0
"""

import time
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import queue
import warnings
from collections import defaultdict

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from ..core.types import TrajectoryData
from ..utils.logging import get_logger
from ..utils.errors import ConfoRLError, ValidationError
from .error_recovery import ErrorRecoveryManager, with_retry, with_circuit_breaker

logger = get_logger(__name__)


class DistributionStrategy(Enum):
    """Distribution strategies for parallel training."""
    DATA_PARALLEL = "data_parallel"
    MODEL_PARALLEL = "model_parallel"
    PIPELINE_PARALLEL = "pipeline_parallel"
    HYBRID = "hybrid"


class ScalingPolicy(Enum):
    """Auto-scaling policies."""
    FIXED = "fixed"
    CPU_BASED = "cpu_based"
    MEMORY_BASED = "memory_based"
    QUEUE_BASED = "queue_based"
    ADAPTIVE = "adaptive"


@dataclass
class WorkerStatus:
    """Status of a distributed worker."""
    worker_id: str
    is_active: bool = True
    current_task: Optional[str] = None
    tasks_completed: int = 0
    tasks_failed: int = 0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    last_heartbeat: float = field(default_factory=time.time)
    
    @property
    def success_rate(self) -> float:
        total = self.tasks_completed + self.tasks_failed
        return self.tasks_completed / max(1, total)


@dataclass
class DistributedTask:
    """Task for distributed execution."""
    task_id: str
    task_type: str
    function: Callable
    args: Tuple
    kwargs: Dict[str, Any]
    priority: int = 0
    timeout: float = 300.0
    retry_count: int = 0
    max_retries: int = 3
    assigned_worker: Optional[str] = None
    created_time: float = field(default_factory=time.time)


class ResourceMonitor:
    """Monitor system resources for auto-scaling decisions."""
    
    def __init__(self, monitoring_interval: float = 5.0):
        """Initialize resource monitor.
        
        Args:
            monitoring_interval: Interval between resource checks in seconds
        """
        self.monitoring_interval = monitoring_interval
        self.cpu_history: List[float] = []
        self.memory_history: List[float] = []
        self.queue_size_history: List[int] = []
        self.monitoring_active = False
        self._monitor_thread = None
        
        logger.info(f"Resource monitor initialized with {monitoring_interval}s interval")
    
    def start_monitoring(self) -> None:
        """Start resource monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        
        logger.info("Resource monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop resource monitoring."""
        self.monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=10.0)
        
        logger.info("Resource monitoring stopped")
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Get system metrics
                cpu_percent = self._get_cpu_usage()
                memory_percent = self._get_memory_usage()
                
                # Store history (keep last 100 measurements)
                self.cpu_history.append(cpu_percent)
                self.memory_history.append(memory_percent)
                
                if len(self.cpu_history) > 100:
                    self.cpu_history.pop(0)
                if len(self.memory_history) > 100:
                    self.memory_history.pop(0)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.warning(f"Resource monitoring error: {e}")
                time.sleep(self.monitoring_interval)
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        try:
            import psutil
            return psutil.cpu_percent(interval=1)
        except ImportError:
            # Fallback: estimate based on load average
            try:
                import os
                load_avg = os.getloadavg()[0]
                cpu_count = os.cpu_count() or 1
                return min(100.0, (load_avg / cpu_count) * 100)
            except:
                return 50.0  # Conservative estimate
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage percentage."""
        try:
            import psutil
            return psutil.virtual_memory().percent
        except ImportError:
            # Fallback: rough estimate
            return 30.0  # Conservative estimate
    
    def get_scaling_recommendation(self, current_workers: int, queue_size: int) -> int:
        """Get recommendation for number of workers based on current metrics.
        
        Args:
            current_workers: Current number of workers
            queue_size: Current task queue size
            
        Returns:
            Recommended number of workers
        """
        if not self.cpu_history or not self.memory_history:
            return current_workers
        
        # Recent averages
        if NUMPY_AVAILABLE:
            recent_cpu = np.mean(self.cpu_history[-10:]) if len(self.cpu_history) >= 10 else np.mean(self.cpu_history)
            recent_memory = np.mean(self.memory_history[-10:]) if len(self.memory_history) >= 10 else np.mean(self.memory_history)
        else:
            # Fallback to basic average
            recent_cpu_data = self.cpu_history[-10:] if len(self.cpu_history) >= 10 else self.cpu_history
            recent_memory_data = self.memory_history[-10:] if len(self.memory_history) >= 10 else self.memory_history
            recent_cpu = sum(recent_cpu_data) / max(1, len(recent_cpu_data))
            recent_memory = sum(recent_memory_data) / max(1, len(recent_memory_data))
        
        # Scaling logic
        recommended_workers = current_workers
        
        # Scale up conditions
        if (recent_cpu > 80.0 or queue_size > current_workers * 10):
            recommended_workers = min(current_workers + 2, mp.cpu_count())
        elif (recent_cpu > 60.0 and queue_size > current_workers * 5):
            recommended_workers = min(current_workers + 1, mp.cpu_count())
        
        # Scale down conditions
        elif (recent_cpu < 20.0 and queue_size < current_workers):
            recommended_workers = max(current_workers - 1, 1)
        elif (recent_cpu < 40.0 and queue_size == 0):
            recommended_workers = max(current_workers - 1, 2)
        
        # Memory pressure check
        if recent_memory > 90.0:
            recommended_workers = max(recommended_workers - 1, 1)
        
        return recommended_workers


class DistributedTrainingManager:
    """Manager for distributed training and parallel execution."""
    
    def __init__(
        self,
        strategy: DistributionStrategy = DistributionStrategy.DATA_PARALLEL,
        initial_workers: int = None,
        scaling_policy: ScalingPolicy = ScalingPolicy.ADAPTIVE,
        max_workers: int = None,
        enable_auto_scaling: bool = True
    ):
        """Initialize distributed training manager.
        
        Args:
            strategy: Distribution strategy
            initial_workers: Initial number of workers (defaults to CPU count)
            scaling_policy: Auto-scaling policy
            max_workers: Maximum number of workers
            enable_auto_scaling: Whether to enable auto-scaling
        """
        self.strategy = strategy
        self.initial_workers = initial_workers or min(mp.cpu_count(), 4)
        self.scaling_policy = scaling_policy
        self.max_workers = max_workers or mp.cpu_count()
        self.enable_auto_scaling = enable_auto_scaling
        
        # Worker management
        self.workers: Dict[str, WorkerStatus] = {}
        self.task_queue: queue.PriorityQueue = queue.PriorityQueue()
        self.result_queue: queue.Queue = queue.Queue()
        self.completed_tasks: Dict[str, Any] = {}
        self.failed_tasks: Dict[str, Exception] = {}
        
        # Executors
        self.thread_executor: Optional[ThreadPoolExecutor] = None
        self.process_executor: Optional[ProcessPoolExecutor] = None
        
        # Resource monitoring
        self.resource_monitor = ResourceMonitor()
        self.error_recovery = ErrorRecoveryManager()
        
        # Synchronization
        self._lock = threading.RLock()
        self._shutdown_event = threading.Event()
        
        # Performance metrics
        self.total_tasks_submitted = 0
        self.total_tasks_completed = 0
        self.total_tasks_failed = 0
        self.worker_utilization_history: List[float] = []
        
        logger.info(f"Distributed training manager initialized with {self.initial_workers} workers")
    
    def start(self) -> None:
        """Start the distributed training system."""
        with self._lock:
            if self.thread_executor is not None:
                logger.warning("Distributed training already started")
                return
            
            # Start executors
            self.thread_executor = ThreadPoolExecutor(
                max_workers=self.initial_workers,
                thread_name_prefix="ConfoRL-Worker"
            )
            
            if self.strategy in [DistributionStrategy.DATA_PARALLEL, DistributionStrategy.HYBRID]:
                self.process_executor = ProcessPoolExecutor(
                    max_workers=min(self.initial_workers, 4)  # Limit process workers
                )
            
            # Initialize workers
            for i in range(self.initial_workers):
                worker_id = f"worker-{i}"
                self.workers[worker_id] = WorkerStatus(worker_id=worker_id)
            
            # Start resource monitoring
            if self.enable_auto_scaling:
                self.resource_monitor.start_monitoring()
            
            # Start task processing
            self._start_task_processing()
            
        logger.info(f"Distributed training started with {len(self.workers)} workers")
    
    def shutdown(self, wait: bool = True, timeout: float = 30.0) -> None:
        """Shutdown the distributed training system.
        
        Args:
            wait: Whether to wait for current tasks to complete
            timeout: Maximum time to wait for shutdown
        """
        logger.info("Shutting down distributed training system...")
        
        self._shutdown_event.set()
        
        # Stop resource monitoring
        self.resource_monitor.stop_monitoring()
        
        # Shutdown executors
        if self.thread_executor:
            self.thread_executor.shutdown(wait=wait)
            self.thread_executor = None
        
        if self.process_executor:
            self.process_executor.shutdown(wait=wait)
            self.process_executor = None
        
        # Clear workers
        with self._lock:
            self.workers.clear()
        
        logger.info("Distributed training system shutdown complete")
    
    def submit_task(
        self,
        task_id: str,
        function: Callable,
        *args,
        priority: int = 0,
        timeout: float = 300.0,
        use_processes: bool = False,
        **kwargs
    ) -> None:
        """Submit a task for distributed execution.
        
        Args:
            task_id: Unique task identifier
            function: Function to execute
            *args: Function arguments
            priority: Task priority (higher = more important)
            timeout: Task timeout in seconds
            use_processes: Whether to use process pool instead of thread pool
            **kwargs: Function keyword arguments
        """
        task = DistributedTask(
            task_id=task_id,
            task_type="process" if use_processes else "thread",
            function=function,
            args=args,
            kwargs=kwargs,
            priority=priority,
            timeout=timeout
        )
        
        # Add to queue (priority queue uses negative priority for max-heap behavior)
        self.task_queue.put((-priority, time.time(), task))
        
        with self._lock:
            self.total_tasks_submitted += 1
        
        logger.debug(f"Task {task_id} submitted with priority {priority}")
    
    def get_result(self, task_id: str, timeout: float = None) -> Any:
        """Get result for a specific task.
        
        Args:
            task_id: Task identifier
            timeout: Maximum time to wait for result
            
        Returns:
            Task result
            
        Raises:
            ConfoRLError: If task failed or timed out
        """
        start_time = time.time()
        
        while True:
            # Check completed tasks
            if task_id in self.completed_tasks:
                return self.completed_tasks.pop(task_id)
            
            # Check failed tasks
            if task_id in self.failed_tasks:
                error = self.failed_tasks.pop(task_id)
                raise ConfoRLError(f"Task {task_id} failed: {error}")
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                raise ConfoRLError(f"Timeout waiting for task {task_id}")
            
            time.sleep(0.1)  # Small delay to avoid busy waiting
    
    def wait_for_all_tasks(self, timeout: float = None) -> Dict[str, Any]:
        """Wait for all submitted tasks to complete.
        
        Args:
            timeout: Maximum time to wait
            
        Returns:
            Dictionary of all completed task results
        """
        start_time = time.time()
        
        while True:
            with self._lock:
                total_finished = len(self.completed_tasks) + len(self.failed_tasks)
                if total_finished >= self.total_tasks_submitted:
                    break
            
            if timeout and (time.time() - start_time) > timeout:
                logger.warning(f"Timeout waiting for all tasks (completed: {total_finished}/{self.total_tasks_submitted})")
                break
            
            time.sleep(0.5)
        
        # Return all completed results
        return dict(self.completed_tasks)
    
    def _start_task_processing(self) -> None:
        """Start task processing thread."""
        def process_tasks():
            while not self._shutdown_event.is_set():
                try:
                    # Get task from queue
                    try:
                        priority, submit_time, task = self.task_queue.get(timeout=1.0)
                    except queue.Empty:
                        continue
                    
                    # Execute task
                    self._execute_task(task)
                    
                    # Auto-scaling check
                    if self.enable_auto_scaling and self.scaling_policy == ScalingPolicy.ADAPTIVE:
                        self._check_auto_scaling()
                    
                except Exception as e:
                    logger.error(f"Task processing error: {e}")
        
        processing_thread = threading.Thread(target=process_tasks, daemon=True)
        processing_thread.start()
    
    @with_retry(max_retries=3, retry_delay=1.0)
    @with_circuit_breaker(threshold=5, timeout=60.0)
    def _execute_task(self, task: DistributedTask) -> None:
        """Execute a distributed task."""
        try:
            # Choose executor based on task type
            if task.task_type == "process" and self.process_executor:
                future = self.process_executor.submit(
                    self._wrapped_task_execution,
                    task.function,
                    task.args,
                    task.kwargs
                )
            else:
                future = self.thread_executor.submit(
                    self._wrapped_task_execution,
                    task.function,
                    task.args,
                    task.kwargs
                )
            
            # Wait for completion with timeout
            try:
                result = future.result(timeout=task.timeout)
                self.completed_tasks[task.task_id] = result
                
                with self._lock:
                    self.total_tasks_completed += 1
                
                logger.debug(f"Task {task.task_id} completed successfully")
                
            except Exception as e:
                self.failed_tasks[task.task_id] = e
                
                with self._lock:
                    self.total_tasks_failed += 1
                
                logger.error(f"Task {task.task_id} failed: {e}")
                
                # Retry logic
                if task.retry_count < task.max_retries:
                    task.retry_count += 1
                    logger.info(f"Retrying task {task.task_id} (attempt {task.retry_count + 1})")
                    
                    # Re-submit with lower priority
                    self.task_queue.put((-(task.priority - 1), time.time(), task))
        
        except Exception as e:
            logger.error(f"Task execution setup failed for {task.task_id}: {e}")
            self.failed_tasks[task.task_id] = e
            
            with self._lock:
                self.total_tasks_failed += 1
    
    def _wrapped_task_execution(self, function: Callable, args: Tuple, kwargs: Dict[str, Any]) -> Any:
        """Wrapper for task execution with error handling."""
        try:
            return function(*args, **kwargs)
        except Exception as e:
            logger.error(f"Task function execution failed: {e}")
            raise
    
    def _check_auto_scaling(self) -> None:
        """Check if auto-scaling is needed."""
        if not self.enable_auto_scaling:
            return
        
        current_workers = len(self.workers)
        queue_size = self.task_queue.qsize()
        
        recommended_workers = self.resource_monitor.get_scaling_recommendation(
            current_workers, queue_size
        )
        
        if recommended_workers != current_workers:
            self._scale_workers(recommended_workers)
    
    def _scale_workers(self, target_workers: int) -> None:
        """Scale the number of workers.
        
        Args:
            target_workers: Target number of workers
        """
        target_workers = max(1, min(target_workers, self.max_workers))
        current_workers = len(self.workers)
        
        if target_workers == current_workers:
            return
        
        with self._lock:
            if target_workers > current_workers:
                # Scale up
                for i in range(current_workers, target_workers):
                    worker_id = f"worker-{i}"
                    self.workers[worker_id] = WorkerStatus(worker_id=worker_id)
                
                # Increase thread pool size
                if self.thread_executor:
                    self.thread_executor._max_workers = target_workers
                
                logger.info(f"Scaled up from {current_workers} to {target_workers} workers")
                
            else:
                # Scale down
                workers_to_remove = list(self.workers.keys())[target_workers:]
                for worker_id in workers_to_remove:
                    del self.workers[worker_id]
                
                logger.info(f"Scaled down from {current_workers} to {target_workers} workers")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        with self._lock:
            total_tasks = self.total_tasks_submitted
            completed_tasks = self.total_tasks_completed
            failed_tasks = self.total_tasks_failed
            
            # Calculate success rate
            success_rate = completed_tasks / max(1, completed_tasks + failed_tasks)
            
            # Worker utilization
            active_workers = sum(1 for worker in self.workers.values() if worker.is_active)
            worker_utilization = active_workers / max(1, len(self.workers))
            
            # Queue metrics
            queue_size = self.task_queue.qsize()
            
            return {
                'total_tasks_submitted': total_tasks,
                'total_tasks_completed': completed_tasks,
                'total_tasks_failed': failed_tasks,
                'success_rate': success_rate,
                'current_workers': len(self.workers),
                'active_workers': active_workers,
                'worker_utilization': worker_utilization,
                'queue_size': queue_size,
                'distribution_strategy': self.strategy.value,
                'scaling_policy': self.scaling_policy.value,
                'auto_scaling_enabled': self.enable_auto_scaling,
                'resource_metrics': {
                    'cpu_history_length': len(self.resource_monitor.cpu_history),
                    'memory_history_length': len(self.resource_monitor.memory_history),
                    'recent_cpu_avg': np.mean(self.resource_monitor.cpu_history[-10:]) if self.resource_monitor.cpu_history else 0.0,
                    'recent_memory_avg': np.mean(self.resource_monitor.memory_history[-10:]) if self.resource_monitor.memory_history else 0.0
                }
            }


class ParallelBenchmarkRunner:
    """Parallel execution of research benchmarks."""
    
    def __init__(self, distributed_manager: DistributedTrainingManager):
        """Initialize parallel benchmark runner.
        
        Args:
            distributed_manager: Distributed training manager instance
        """
        self.distributed_manager = distributed_manager
        self.benchmark_results: Dict[str, Any] = {}
        
        logger.info("Parallel benchmark runner initialized")
    
    def run_parallel_benchmarks(
        self,
        benchmark_configs: List[Dict[str, Any]],
        max_parallel: int = None
    ) -> Dict[str, Any]:
        """Run multiple benchmarks in parallel.
        
        Args:
            benchmark_configs: List of benchmark configurations
            max_parallel: Maximum number of parallel benchmarks
            
        Returns:
            Dictionary of benchmark results
        """
        max_parallel = max_parallel or len(benchmark_configs)
        
        # Submit benchmark tasks
        for i, config in enumerate(benchmark_configs):
            task_id = f"benchmark_{i}_{config.get('name', 'unnamed')}"
            
            self.distributed_manager.submit_task(
                task_id=task_id,
                function=self._run_single_benchmark,
                config=config,
                priority=config.get('priority', 0),
                use_processes=config.get('use_processes', False)
            )
        
        # Wait for all benchmarks to complete
        logger.info(f"Running {len(benchmark_configs)} benchmarks in parallel...")
        results = self.distributed_manager.wait_for_all_tasks(timeout=3600)  # 1 hour timeout
        
        # Process results
        self.benchmark_results.update(results)
        
        return results
    
    def _run_single_benchmark(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single benchmark configuration.
        
        Args:
            config: Benchmark configuration
            
        Returns:
            Benchmark results
        """
        start_time = time.time()
        
        try:
            # Extract configuration
            benchmark_name = config.get('name', 'unnamed')
            algorithm = config.get('algorithm')
            environment = config.get('environment')
            num_episodes = config.get('num_episodes', 100)
            num_runs = config.get('num_runs', 5)
            
            logger.info(f"Starting benchmark: {benchmark_name}")
            
            # Placeholder for actual benchmark execution
            # In a real implementation, this would call the specific benchmark
            
            # Simulate benchmark execution
            run_results = []
            for run_id in range(num_runs):
                # Simulate run
                episode_returns = [np.random.normal(100, 20) for _ in range(num_episodes)]
                safety_violations = [np.random.poisson(0.5) for _ in range(num_episodes)]
                
                run_result = {
                    'run_id': run_id,
                    'mean_return': np.mean(episode_returns),
                    'std_return': np.std(episode_returns),
                    'mean_violations': np.mean(safety_violations),
                    'episodes_completed': num_episodes
                }
                
                run_results.append(run_result)
            
            # Aggregate results
            all_returns = [r['mean_return'] for r in run_results]
            all_violations = [r['mean_violations'] for r in run_results]
            
            result = {
                'benchmark_name': benchmark_name,
                'algorithm': algorithm,
                'environment': environment,
                'num_runs': num_runs,
                'num_episodes_per_run': num_episodes,
                'mean_return': np.mean(all_returns),
                'std_return': np.std(all_returns),
                'mean_violations': np.mean(all_violations),
                'std_violations': np.std(all_violations),
                'runtime_seconds': time.time() - start_time,
                'success': True,
                'run_results': run_results
            }
            
            logger.info(f"Benchmark {benchmark_name} completed in {result['runtime_seconds']:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Benchmark {config.get('name', 'unnamed')} failed: {e}")
            
            return {
                'benchmark_name': config.get('name', 'unnamed'),
                'success': False,
                'error': str(e),
                'runtime_seconds': time.time() - start_time
            }


# Global distributed manager instance
_global_distributed_manager = None

def get_distributed_manager() -> DistributedTrainingManager:
    """Get global distributed training manager instance."""
    global _global_distributed_manager
    if _global_distributed_manager is None:
        _global_distributed_manager = DistributedTrainingManager()
    return _global_distributed_manager


def distributed_task(priority: int = 0, timeout: float = 300.0, use_processes: bool = False):
    """Decorator for marking functions as distributed tasks.
    
    Args:
        priority: Task priority
        timeout: Task timeout
        use_processes: Whether to use process pool
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            manager = get_distributed_manager()
            task_id = f"{func.__name__}_{int(time.time() * 1000)}"
            
            manager.submit_task(
                task_id=task_id,
                function=func,
                *args,
                priority=priority,
                timeout=timeout,
                use_processes=use_processes,
                **kwargs
            )
            
            return manager.get_result(task_id)
        
        return wrapper
    return decorator