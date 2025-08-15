"""Concurrent processing utilities for scalable training and inference."""

import asyncio
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Callable, Optional, Union
import queue
import time
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Minimal numpy-like interface
    class np:
        @staticmethod
        def random():
            import random
            class Random:
                @staticmethod
                def random():
                    return random.random()
            return Random()

from ..utils.logging import get_logger
from ..core.types import TrajectoryData, RiskCertificate
from ..risk.controllers import AdaptiveRiskController

logger = get_logger(__name__)


class ConcurrentProcessor:
    """High-performance concurrent task processor."""
    
    def __init__(self, max_workers: int = 4):
        """Initialize concurrent processor.
        
        Args:
            max_workers: Maximum number of worker threads/processes
        """
        self.max_workers = max_workers
        self.thread_executor = ThreadPoolExecutor(max_workers=max_workers)
        self.process_executor = ProcessPoolExecutor(max_workers=max_workers)
    
    def execute_concurrent(self, func: Callable, tasks: List[Any], use_processes: bool = False) -> List[Any]:
        """Execute tasks concurrently.
        
        Args:
            func: Function to execute
            tasks: List of task inputs
            use_processes: Whether to use processes instead of threads
            
        Returns:
            List of results in order
        """
        executor = self.process_executor if use_processes else self.thread_executor
        
        futures = [executor.submit(func, task) for task in tasks]
        results = [future.result() for future in futures]
        
        return results
    
    def __del__(self):
        """Cleanup executors."""
        try:
            self.thread_executor.shutdown(wait=False)
            self.process_executor.shutdown(wait=False)
        except:
            pass


class ParallelTraining:
    """Parallel training system for multiple agents or environments."""
    
    def __init__(
        self,
        num_workers: int = None,
        use_processes: bool = True,
        shared_buffer: bool = True
    ):
        """Initialize parallel training system.
        
        Args:
            num_workers: Number of parallel workers (default: CPU count)
            use_processes: Whether to use processes vs threads
            shared_buffer: Whether to use shared replay buffer
        """
        self.num_workers = num_workers or mp.cpu_count()
        self.use_processes = use_processes
        self.shared_buffer = shared_buffer
        
        # Worker management
        self.workers = []
        self.worker_queues = []
        self.result_queue = queue.Queue()
        
        # Shared state
        self.shared_memory = {}
        self.coordination_lock = threading.Lock()
        
        # Performance tracking
        self.training_stats = {
            'episodes_completed': 0,
            'total_steps': 0,
            'worker_utilization': {},
            'throughput': 0.0
        }
        
        logger.info(f"Initialized parallel training with {self.num_workers} workers")
    
    def start_workers(self, worker_func: Callable, worker_args: List[Any]):
        """Start parallel training workers.
        
        Args:
            worker_func: Function to run in each worker
            worker_args: List of arguments for each worker
        """
        self.workers.clear()
        self.worker_queues.clear()
        
        if self.use_processes:
            # Process-based parallelism
            for i in range(self.num_workers):
                worker_queue = mp.Queue()
                worker_process = mp.Process(
                    target=self._worker_loop,
                    args=(i, worker_func, worker_args[i], worker_queue, self.result_queue)
                )
                worker_process.start()
                
                self.workers.append(worker_process)
                self.worker_queues.append(worker_queue)
        else:
            # Thread-based parallelism
            for i in range(self.num_workers):
                worker_queue = queue.Queue()
                worker_thread = threading.Thread(
                    target=self._worker_loop,
                    args=(i, worker_func, worker_args[i], worker_queue, self.result_queue)
                )
                worker_thread.start()
                
                self.workers.append(worker_thread)
                self.worker_queues.append(worker_queue)
        
        logger.info(f"Started {len(self.workers)} parallel workers")
    
    def _worker_loop(
        self,
        worker_id: int,
        worker_func: Callable,
        worker_args: Any,
        command_queue: queue.Queue,
        result_queue: queue.Queue
    ):
        """Main loop for worker processes/threads."""
        logger.info(f"Worker {worker_id} started")
        
        try:
            # Initialize worker
            worker_state = worker_func(worker_id, worker_args, 'init')
            
            while True:
                try:
                    # Check for commands
                    command = command_queue.get(timeout=0.1)
                    
                    if command == 'stop':
                        break
                    elif command[0] == 'train':
                        # Training step
                        result = worker_func(worker_id, worker_args, 'train', worker_state, command[1])
                        result_queue.put(('train_result', worker_id, result))
                    elif command[0] == 'evaluate':
                        # Evaluation step
                        result = worker_func(worker_id, worker_args, 'evaluate', worker_state, command[1])
                        result_queue.put(('eval_result', worker_id, result))
                
                except queue.Empty:
                    # No command, continue with default training
                    result = worker_func(worker_id, worker_args, 'step', worker_state)
                    if result:
                        result_queue.put(('step_result', worker_id, result))
                
                except Exception as e:
                    logger.error(f"Worker {worker_id} error: {e}")
                    result_queue.put(('error', worker_id, str(e)))
        
        except Exception as e:
            logger.error(f"Worker {worker_id} fatal error: {e}")
        
        finally:
            logger.info(f"Worker {worker_id} stopped")
    
    def send_command(self, worker_id: int, command: Any):
        """Send command to specific worker.
        
        Args:
            worker_id: ID of target worker
            command: Command to send
        """
        if 0 <= worker_id < len(self.worker_queues):
            self.worker_queues[worker_id].put(command)
        else:
            logger.warning(f"Invalid worker ID: {worker_id}")
    
    def broadcast_command(self, command: Any):
        """Broadcast command to all workers.
        
        Args:
            command: Command to broadcast
        """
        for queue in self.worker_queues:
            queue.put(command)
    
    def collect_results(self, timeout: float = 1.0) -> List[Any]:
        """Collect results from workers.
        
        Args:
            timeout: Timeout for collecting results
            
        Returns:
            List of results from workers
        """
        results = []
        end_time = time.time() + timeout
        
        while time.time() < end_time:
            try:
                result = self.result_queue.get(timeout=0.1)
                results.append(result)
                
                # Update statistics
                if result[0] == 'step_result':
                    self.training_stats['total_steps'] += 1
                elif result[0] == 'train_result':
                    self.training_stats['episodes_completed'] += 1
            
            except queue.Empty:
                continue
        
        return results
    
    def stop_workers(self):
        """Stop all workers."""
        logger.info("Stopping parallel workers")
        
        # Send stop command to all workers
        self.broadcast_command('stop')
        
        # Wait for workers to finish
        for worker in self.workers:
            if hasattr(worker, 'join'):
                worker.join(timeout=5.0)
            if hasattr(worker, 'terminate'):
                if worker.is_alive():
                    worker.terminate()
        
        self.workers.clear()
        self.worker_queues.clear()
        
        logger.info("All workers stopped")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics.
        
        Returns:
            Dictionary with training statistics
        """
        # Calculate throughput
        if self.training_stats['total_steps'] > 0:
            # Simplified throughput calculation
            self.training_stats['throughput'] = self.training_stats['total_steps'] / max(1, time.time() - getattr(self, 'start_time', time.time()))
        
        return self.training_stats.copy()


class AsyncRiskController:
    """Asynchronous risk controller for real-time monitoring."""
    
    def __init__(
        self,
        base_controller: AdaptiveRiskController,
        update_interval: float = 1.0,
        max_queue_size: int = 1000
    ):
        """Initialize async risk controller.
        
        Args:
            base_controller: Base risk controller to wrap
            update_interval: Update interval in seconds
            max_queue_size: Maximum size of update queue
        """
        self.base_controller = base_controller
        self.update_interval = update_interval
        self.max_queue_size = max_queue_size
        
        # Async components
        self.update_queue = asyncio.Queue(maxsize=max_queue_size)
        self.running = False
        self.background_task = None
        
        # Thread-safe state
        self._current_certificate = None
        self._certificate_lock = threading.Lock()
        
        # Performance tracking
        self.updates_processed = 0
        self.queue_overflows = 0
    
    async def start(self):
        """Start async risk monitoring."""
        if self.running:
            return
        
        self.running = True
        self.background_task = asyncio.create_task(self._update_loop())
        logger.info("Async risk controller started")
    
    async def stop(self):
        """Stop async risk monitoring."""
        if not self.running:
            return
        
        self.running = False
        if self.background_task:
            self.background_task.cancel()
            try:
                await self.background_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Async risk controller stopped")
    
    async def _update_loop(self):
        """Main update loop for async processing."""
        while self.running:
            try:
                # Process updates from queue
                updates_processed = 0
                start_time = time.time()
                
                while not self.update_queue.empty() and updates_processed < 10:
                    try:
                        trajectory, risk_measure = await asyncio.wait_for(
                            self.update_queue.get(), timeout=0.1
                        )
                        
                        # Update base controller
                        self.base_controller.update(trajectory, risk_measure)
                        
                        # Update cached certificate
                        with self._certificate_lock:
                            self._current_certificate = self.base_controller.get_certificate()
                        
                        updates_processed += 1
                        self.updates_processed += 1
                        
                    except asyncio.TimeoutError:
                        break
                
                # Log performance
                if updates_processed > 0:
                    processing_time = time.time() - start_time
                    logger.debug(f"Processed {updates_processed} risk updates in {processing_time:.3f}s")
                
                # Wait for next update interval
                await asyncio.sleep(self.update_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in async risk controller: {e}")
                await asyncio.sleep(1.0)  # Backoff on error
    
    async def update_async(self, trajectory: TrajectoryData, risk_measure):
        """Submit async update to risk controller.
        
        Args:
            trajectory: Trajectory data
            risk_measure: Risk measure to use
        """
        try:
            self.update_queue.put_nowait((trajectory, risk_measure))
        except asyncio.QueueFull:
            # Queue overflow - drop oldest update
            try:
                await asyncio.wait_for(self.update_queue.get(), timeout=0.01)
                self.update_queue.put_nowait((trajectory, risk_measure))
                self.queue_overflows += 1
                logger.warning("Risk update queue overflow - dropped old update")
            except (asyncio.TimeoutError, asyncio.QueueFull):
                logger.error("Failed to submit risk update - queue full")
    
    def get_certificate(self) -> Optional[RiskCertificate]:
        """Get current risk certificate (thread-safe).
        
        Returns:
            Current risk certificate or None
        """
        with self._certificate_lock:
            return self._current_certificate
    
    def get_stats(self) -> Dict[str, Any]:
        """Get async controller statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            'running': self.running,
            'updates_processed': self.updates_processed,
            'queue_overflows': self.queue_overflows,
            'queue_size': self.update_queue.qsize() if hasattr(self.update_queue, 'qsize') else 0,
            'max_queue_size': self.max_queue_size
        }


class BatchProcessor:
    """Batch processor for efficient parallel computation."""
    
    def __init__(
        self,
        batch_size: int = 32,
        max_workers: int = None,
        timeout: float = 30.0
    ):
        """Initialize batch processor.
        
        Args:
            batch_size: Size of processing batches
            max_workers: Maximum number of worker threads
            timeout: Timeout for batch processing
        """
        self.batch_size = batch_size
        self.max_workers = max_workers or min(32, mp.cpu_count() + 4)
        self.timeout = timeout
        
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Performance tracking
        self.batches_processed = 0
        self.total_items_processed = 0
        self.total_processing_time = 0.0
    
    def process_batch(
        self,
        items: List[Any],
        process_func: Callable,
        *args,
        **kwargs
    ) -> List[Any]:
        """Process items in parallel batches.
        
        Args:
            items: Items to process
            process_func: Function to apply to each item
            *args: Additional arguments for process_func
            **kwargs: Additional keyword arguments for process_func
            
        Returns:
            List of processed results
        """
        if not items:
            return []
        
        start_time = time.time()
        results = []
        
        # Split items into batches
        batches = [
            items[i:i + self.batch_size]
            for i in range(0, len(items), self.batch_size)
        ]
        
        # Submit batches to executor
        future_to_batch = {
            self.executor.submit(self._process_batch_items, batch, process_func, *args, **kwargs): batch
            for batch in batches
        }
        
        # Collect results
        for future in as_completed(future_to_batch, timeout=self.timeout):
            batch = future_to_batch[future]
            try:
                batch_results = future.result()
                results.extend(batch_results)
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                # Return None for failed items
                results.extend([None] * len(batch))
        
        # Update statistics
        processing_time = time.time() - start_time
        self.batches_processed += len(batches)
        self.total_items_processed += len(items)
        self.total_processing_time += processing_time
        
        logger.debug(f"Processed {len(items)} items in {len(batches)} batches ({processing_time:.3f}s)")
        
        return results
    
    def _process_batch_items(
        self,
        batch: List[Any],
        process_func: Callable,
        *args,
        **kwargs
    ) -> List[Any]:
        """Process items in a single batch."""
        return [process_func(item, *args, **kwargs) for item in batch]
    
    def shutdown(self):
        """Shutdown the batch processor."""
        self.executor.shutdown(wait=True)
        logger.info("Batch processor shutdown")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batch processing statistics.
        
        Returns:
            Dictionary with processing statistics
        """
        avg_batch_time = (
            self.total_processing_time / self.batches_processed
            if self.batches_processed > 0 else 0.0
        )
        
        avg_item_time = (
            self.total_processing_time / self.total_items_processed
            if self.total_items_processed > 0 else 0.0
        )
        
        return {
            'batches_processed': self.batches_processed,
            'total_items_processed': self.total_items_processed,
            'total_processing_time': self.total_processing_time,
            'avg_batch_time': avg_batch_time,
            'avg_item_time': avg_item_time,
            'throughput_items_per_second': (
                self.total_items_processed / self.total_processing_time
                if self.total_processing_time > 0 else 0.0
            )
        }