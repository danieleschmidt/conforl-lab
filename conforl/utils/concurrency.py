"""Thread safety and concurrency utilities for ConfoRL."""

import threading
import queue
import time
import concurrent.futures
from typing import Any, Callable, Dict, List, Optional, TypeVar, Generic
from functools import wraps
from contextlib import contextmanager

from .logging import get_logger
from .errors import ConfoRLError

logger = get_logger(__name__)

T = TypeVar('T')


class ThreadSafeCounter:
    """Thread-safe counter implementation."""
    
    def __init__(self, initial_value: int = 0):
        """Initialize counter.
        
        Args:
            initial_value: Initial counter value
        """
        self._value = initial_value
        self._lock = threading.Lock()
    
    def increment(self, delta: int = 1) -> int:
        """Increment counter by delta.
        
        Args:
            delta: Amount to increment
            
        Returns:
            New counter value
        """
        with self._lock:
            self._value += delta
            return self._value
    
    def decrement(self, delta: int = 1) -> int:
        """Decrement counter by delta.
        
        Args:
            delta: Amount to decrement
            
        Returns:
            New counter value
        """
        with self._lock:
            self._value -= delta
            return self._value
    
    def get(self) -> int:
        """Get current counter value.
        
        Returns:
            Current counter value
        """
        with self._lock:
            return self._value
    
    def set(self, value: int) -> int:
        """Set counter to specific value.
        
        Args:
            value: New counter value
            
        Returns:
            New counter value
        """
        with self._lock:
            self._value = value
            return self._value


class ThreadSafeDict(Generic[T]):
    """Thread-safe dictionary implementation."""
    
    def __init__(self):
        """Initialize thread-safe dictionary."""
        self._data: Dict[str, T] = {}
        self._lock = threading.RLock()  # Reentrant lock
    
    def get(self, key: str, default: Optional[T] = None) -> Optional[T]:
        """Get value by key.
        
        Args:
            key: Dictionary key
            default: Default value if key not found
            
        Returns:
            Value or default
        """
        with self._lock:
            return self._data.get(key, default)
    
    def set(self, key: str, value: T) -> None:
        """Set value by key.
        
        Args:
            key: Dictionary key
            value: Value to set
        """
        with self._lock:
            self._data[key] = value
    
    def delete(self, key: str) -> bool:
        """Delete key from dictionary.
        
        Args:
            key: Key to delete
            
        Returns:
            True if key was deleted, False if not found
        """
        with self._lock:
            if key in self._data:
                del self._data[key]
                return True
            return False
    
    def keys(self) -> List[str]:
        """Get all keys.
        
        Returns:
            List of keys
        """
        with self._lock:
            return list(self._data.keys())
    
    def values(self) -> List[T]:
        """Get all values.
        
        Returns:
            List of values
        """
        with self._lock:
            return list(self._data.values())
    
    def items(self) -> List[tuple]:
        """Get all key-value pairs.
        
        Returns:
            List of (key, value) tuples
        """
        with self._lock:
            return list(self._data.items())
    
    def update(self, other: Dict[str, T]) -> None:
        """Update dictionary with another dictionary.
        
        Args:
            other: Dictionary to update from
        """
        with self._lock:
            self._data.update(other)
    
    def clear(self) -> None:
        """Clear all items from dictionary."""
        with self._lock:
            self._data.clear()
    
    def size(self) -> int:
        """Get number of items in dictionary.
        
        Returns:
            Number of items
        """
        with self._lock:
            return len(self._data)


class ThreadPool:
    """Enhanced thread pool for ConfoRL operations."""
    
    def __init__(self, max_workers: int = 4, thread_name_prefix: str = "ConfoRL"):
        """Initialize thread pool.
        
        Args:
            max_workers: Maximum number of worker threads
            thread_name_prefix: Prefix for thread names
        """
        self.max_workers = max_workers
        self.thread_name_prefix = thread_name_prefix
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=thread_name_prefix
        )
        self.active_tasks = ThreadSafeCounter()
        self.completed_tasks = ThreadSafeCounter()
        self.failed_tasks = ThreadSafeCounter()
    
    def submit(self, fn: Callable, *args, **kwargs) -> concurrent.futures.Future:
        """Submit a function for execution.
        
        Args:
            fn: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Future object
        """
        self.active_tasks.increment()
        
        def wrapped_fn(*args, **kwargs):
            try:
                result = fn(*args, **kwargs)
                self.completed_tasks.increment()
                return result
            except Exception as e:
                self.failed_tasks.increment()
                logger.error(f"Task failed in thread pool: {e}")
                raise
            finally:
                self.active_tasks.decrement()
        
        return self.executor.submit(wrapped_fn, *args, **kwargs)
    
    def map(self, fn: Callable, iterable) -> List[Any]:
        """Apply function to items in iterable using thread pool.
        
        Args:
            fn: Function to apply
            iterable: Items to process
            
        Returns:
            List of results
        """
        futures = [self.submit(fn, item) for item in iterable]
        return [future.result() for future in futures]
    
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown thread pool.
        
        Args:
            wait: Whether to wait for completion
        """
        self.executor.shutdown(wait=wait)
    
    def get_stats(self) -> Dict[str, int]:
        """Get thread pool statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            "max_workers": self.max_workers,
            "active_tasks": self.active_tasks.get(),
            "completed_tasks": self.completed_tasks.get(),
            "failed_tasks": self.failed_tasks.get()
        }


class RateLimiter:
    """Rate limiter for controlling operation frequency."""
    
    def __init__(self, max_calls: int, time_window: float):
        """Initialize rate limiter.
        
        Args:
            max_calls: Maximum number of calls allowed
            time_window: Time window in seconds
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
        self.lock = threading.Lock()
    
    def acquire(self, timeout: Optional[float] = None) -> bool:
        """Acquire permission to make a call.
        
        Args:
            timeout: Maximum time to wait for permission
            
        Returns:
            True if permission granted, False if timeout
        """
        start_time = time.time()
        
        while True:
            with self.lock:
                now = time.time()
                
                # Remove old calls outside the time window
                self.calls = [call_time for call_time in self.calls 
                             if now - call_time < self.time_window]
                
                # Check if we can make a new call
                if len(self.calls) < self.max_calls:
                    self.calls.append(now)
                    return True
            
            # Check timeout
            if timeout is not None and time.time() - start_time > timeout:
                return False
            
            # Wait a bit before trying again
            time.sleep(0.01)
    
    @contextmanager
    def limit(self, timeout: Optional[float] = None):
        """Context manager for rate limiting.
        
        Args:
            timeout: Maximum time to wait for permission
            
        Raises:
            ConfoRLError: If rate limit exceeded and timeout
        """
        if not self.acquire(timeout):
            raise ConfoRLError("Rate limit exceeded", "RATE_LIMIT_EXCEEDED")
        
        try:
            yield
        finally:
            pass  # Nothing to clean up


def thread_safe(func: Callable) -> Callable:
    """Decorator to make a function thread-safe with a lock.
    
    Args:
        func: Function to make thread-safe
        
    Returns:
        Thread-safe wrapped function
    """
    lock = threading.Lock()
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        with lock:
            return func(*args, **kwargs)
    
    return wrapper


def singleton_per_thread(cls):
    """Decorator to make a class singleton per thread.
    
    Args:
        cls: Class to make singleton per thread
        
    Returns:
        Modified class with singleton behavior per thread
    """
    instances = threading.local()
    
    def get_instance(*args, **kwargs):
        if not hasattr(instances, 'instance'):
            instances.instance = cls(*args, **kwargs)
        return instances.instance
    
    return get_instance


class WorkerPool:
    """Worker pool for processing tasks asynchronously."""
    
    def __init__(self, num_workers: int = 2, queue_size: int = 100):
        """Initialize worker pool.
        
        Args:
            num_workers: Number of worker threads
            queue_size: Maximum queue size
        """
        self.num_workers = num_workers
        self.task_queue = queue.Queue(maxsize=queue_size)
        self.workers = []
        self.shutdown_flag = threading.Event()
        self.stats = ThreadSafeDict()
        
        # Initialize statistics
        self.stats.set("tasks_submitted", 0)
        self.stats.set("tasks_completed", 0)
        self.stats.set("tasks_failed", 0)
        
        # Start worker threads
        for i in range(num_workers):
            worker = threading.Thread(target=self._worker_loop, name=f"Worker-{i}")
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
        
        logger.info(f"Started worker pool with {num_workers} workers")
    
    def submit_task(self, func: Callable, *args, **kwargs) -> bool:
        """Submit a task for processing.
        
        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            True if task was submitted, False if queue full
        """
        try:
            task = (func, args, kwargs)
            self.task_queue.put(task, block=False)
            self.stats.set("tasks_submitted", self.stats.get("tasks_submitted", 0) + 1)
            return True
        except queue.Full:
            logger.warning("Worker pool queue is full, rejecting task")
            return False
    
    def _worker_loop(self):
        """Main worker loop."""
        worker_name = threading.current_thread().name
        logger.debug(f"Started worker: {worker_name}")
        
        while not self.shutdown_flag.is_set():
            try:
                # Get task with timeout
                task = self.task_queue.get(timeout=1.0)
                
                if task is None:  # Shutdown signal
                    break
                
                func, args, kwargs = task
                
                try:
                    # Execute task
                    start_time = time.time()
                    func(*args, **kwargs)
                    duration = time.time() - start_time
                    
                    self.stats.set("tasks_completed", self.stats.get("tasks_completed", 0) + 1)
                    logger.debug(f"Task completed in {duration:.3f}s by {worker_name}")
                    
                except Exception as e:
                    self.stats.set("tasks_failed", self.stats.get("tasks_failed", 0) + 1)
                    logger.error(f"Task failed in {worker_name}: {e}")
                
                finally:
                    self.task_queue.task_done()
                    
            except queue.Empty:
                continue  # Timeout, check shutdown flag
            except Exception as e:
                logger.error(f"Worker {worker_name} error: {e}")
        
        logger.debug(f"Worker {worker_name} shutting down")
    
    def shutdown(self, timeout: float = 5.0):
        """Shutdown worker pool.
        
        Args:
            timeout: Maximum time to wait for workers to finish
        """
        logger.info("Shutting down worker pool")
        
        # Signal shutdown
        self.shutdown_flag.set()
        
        # Add shutdown signals to queue
        for _ in self.workers:
            try:
                self.task_queue.put(None, block=False)
            except queue.Full:
                pass
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=timeout)
            if worker.is_alive():
                logger.warning(f"Worker {worker.name} did not shutdown cleanly")
    
    def get_stats(self) -> Dict[str, int]:
        """Get worker pool statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            "num_workers": self.num_workers,
            "queue_size": self.task_queue.qsize(),
            "tasks_submitted": self.stats.get("tasks_submitted", 0),
            "tasks_completed": self.stats.get("tasks_completed", 0),
            "tasks_failed": self.stats.get("tasks_failed", 0),
            "active_workers": sum(1 for w in self.workers if w.is_alive())
        }


# Global worker pool instance
_global_worker_pool: Optional[WorkerPool] = None


def get_worker_pool() -> WorkerPool:
    """Get global worker pool instance."""
    global _global_worker_pool
    if _global_worker_pool is None:
        _global_worker_pool = WorkerPool()
    return _global_worker_pool


def submit_background_task(func: Callable, *args, **kwargs) -> bool:
    """Submit a task to run in background.
    
    Args:
        func: Function to execute
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        True if task was submitted successfully
    """
    return get_worker_pool().submit_task(func, *args, **kwargs)


def shutdown_worker_pool():
    """Shutdown global worker pool."""
    global _global_worker_pool
    if _global_worker_pool:
        _global_worker_pool.shutdown()
        _global_worker_pool = None