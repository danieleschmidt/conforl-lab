"""Performance profiling and monitoring utilities."""

import time
import threading
import gc
import json
import os
from typing import Dict, Any, List, Optional, Callable, Union
from collections import defaultdict, deque
import functools
from pathlib import Path

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import tracemalloc
    TRACEMALLOC_AVAILABLE = True
except ImportError:
    TRACEMALLOC_AVAILABLE = False

try:
    import cProfile
    import pstats
    CPROFILE_AVAILABLE = True
except ImportError:
    CPROFILE_AVAILABLE = False

from ..utils.logging import get_logger

logger = get_logger(__name__)


class PerformanceProfiler:
    """Performance profiler with context management."""
    
    def __init__(self):
        """Initialize performance profiler."""
        self.results = {}
        self.current_operations = {}
    
    def profile(self, operation_name: str):
        """Context manager for profiling operations."""
        return ProfileContext(self, operation_name)
    
    def start_operation(self, name: str) -> None:
        """Start profiling an operation."""
        import time
        self.current_operations[name] = time.time()
    
    def end_operation(self, name: str) -> None:
        """End profiling an operation."""
        import time
        if name in self.current_operations:
            total_time = time.time() - self.current_operations[name]
            self.results[name] = {
                'total_time': total_time,
                'timestamp': time.time()
            }
            del self.current_operations[name]
    
    def get_results(self) -> Dict[str, Any]:
        """Get profiling results."""
        return self.results.copy()
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            return psutil.Process().memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0


class ProfileContext:
    """Context manager for profiling."""
    
    def __init__(self, profiler: PerformanceProfiler, operation_name: str):
        self.profiler = profiler
        self.operation_name = operation_name
    
    def __enter__(self):
        self.profiler.start_operation(self.operation_name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.profiler.end_operation(self.operation_name)


class OriginalPerformanceProfiler:
    """Comprehensive performance profiler for ConfoRL operations."""
    
    def __init__(
        self,
        enable_memory_tracking: bool = True,
        enable_cpu_tracking: bool = True,
        history_size: int = 1000
    ):
        """Initialize performance profiler.
        
        Args:
            enable_memory_tracking: Whether to track memory usage
            enable_cpu_tracking: Whether to track CPU usage
            history_size: Size of performance history to keep
        """
        self.enable_memory_tracking = enable_memory_tracking
        self.enable_cpu_tracking = enable_cpu_tracking
        self.history_size = history_size
        
        # Performance data storage
        self.function_stats = defaultdict(lambda: {
            'call_count': 0,
            'total_time': 0.0,
            'avg_time': 0.0,
            'min_time': float('inf'),
            'max_time': 0.0,
            'recent_times': deque(maxlen=100)
        })
        
        self.system_stats = {
            'cpu_percent': deque(maxlen=history_size),
            'memory_percent': deque(maxlen=history_size),
            'memory_bytes': deque(maxlen=history_size),
            'timestamps': deque(maxlen=history_size)
        }
        
        # Monitoring thread
        self.monitoring_thread = None
        self.monitoring_active = False
        self.monitoring_interval = 1.0  # seconds
        
        # Memory profiling
        if self.enable_memory_tracking:
            tracemalloc.start()
        
        logger.info("Performance profiler initialized")
    
    def profile_function(self, func_name: Optional[str] = None):
        """Decorator to profile function performance.
        
        Args:
            func_name: Optional custom name for the function
            
        Returns:
            Decorated function
        """
        def decorator(func: Callable):
            name = func_name or f"{func.__module__}.{func.__name__}"
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                start_memory = self._get_memory_usage() if self.enable_memory_tracking else 0
                
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    end_time = time.time()
                    end_memory = self._get_memory_usage() if self.enable_memory_tracking else 0
                    
                    execution_time = end_time - start_time
                    memory_delta = end_memory - start_memory
                    
                    self._record_function_stats(name, execution_time, memory_delta)
            
            return wrapper
        return decorator
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        except:
            return 0.0
    
    def _record_function_stats(
        self,
        func_name: str,
        execution_time: float,
        memory_delta: float = 0.0
    ):
        """Record performance statistics for a function.
        
        Args:
            func_name: Name of the function
            execution_time: Time taken to execute
            memory_delta: Change in memory usage
        """
        stats = self.function_stats[func_name]
        
        stats['call_count'] += 1
        stats['total_time'] += execution_time
        stats['avg_time'] = stats['total_time'] / stats['call_count']
        stats['min_time'] = min(stats['min_time'], execution_time)
        stats['max_time'] = max(stats['max_time'], execution_time)
        stats['recent_times'].append(execution_time)
        
        if memory_delta != 0.0:
            if 'memory_delta' not in stats:
                stats['memory_delta'] = []
            stats['memory_delta'].append(memory_delta)
    
    def start_monitoring(self, interval: float = 1.0):
        """Start system resource monitoring.
        
        Args:
            interval: Monitoring interval in seconds
        """
        if self.monitoring_active:
            return
        
        self.monitoring_interval = interval
        self.monitoring_active = True
        
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        
        logger.info(f"Started system monitoring (interval: {interval}s)")
    
    def stop_monitoring(self):
        """Stop system resource monitoring."""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        logger.info("Stopped system monitoring")
    
    def _monitoring_loop(self):
        """Main loop for system resource monitoring."""
        while self.monitoring_active:
            try:
                current_time = time.time()
                
                if self.enable_cpu_tracking:
                    cpu_percent = psutil.cpu_percent(interval=None)
                    self.system_stats['cpu_percent'].append(cpu_percent)
                
                if self.enable_memory_tracking:
                    process = psutil.Process()
                    memory_info = process.memory_info()
                    memory_percent = process.memory_percent()
                    
                    self.system_stats['memory_percent'].append(memory_percent)
                    self.system_stats['memory_bytes'].append(memory_info.rss)
                
                self.system_stats['timestamps'].append(current_time)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(self.monitoring_interval)
    
    def get_function_stats(self, func_name: Optional[str] = None) -> Dict[str, Any]:
        """Get function performance statistics.
        
        Args:
            func_name: Specific function name (None for all)
            
        Returns:
            Dictionary with function statistics
        """
        if func_name:
            if func_name in self.function_stats:
                return dict(self.function_stats[func_name])
            else:
                return {}
        
        return {name: dict(stats) for name, stats in self.function_stats.items()}
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system resource statistics.
        
        Returns:
            Dictionary with system statistics
        """
        stats = {}
        
        if self.system_stats['cpu_percent']:
            cpu_data = list(self.system_stats['cpu_percent'])
            stats['cpu'] = {
                'current': cpu_data[-1] if cpu_data else 0,
                'average': sum(cpu_data) / len(cpu_data),
                'max': max(cpu_data),
                'min': min(cpu_data)
            }
        
        if self.system_stats['memory_percent']:
            mem_percent_data = list(self.system_stats['memory_percent'])
            mem_bytes_data = list(self.system_stats['memory_bytes'])
            
            stats['memory'] = {
                'current_percent': mem_percent_data[-1] if mem_percent_data else 0,
                'current_bytes': mem_bytes_data[-1] if mem_bytes_data else 0,
                'current_mb': mem_bytes_data[-1] / 1024 / 1024 if mem_bytes_data else 0,
                'average_percent': sum(mem_percent_data) / len(mem_percent_data),
                'max_bytes': max(mem_bytes_data) if mem_bytes_data else 0,
                'min_bytes': min(mem_bytes_data) if mem_bytes_data else 0
            }
        
        return stats
    
    def get_top_functions(self, limit: int = 10, sort_by: str = 'total_time') -> List[Dict[str, Any]]:
        """Get top functions by performance metric.
        
        Args:
            limit: Number of functions to return
            sort_by: Metric to sort by ('total_time', 'avg_time', 'call_count')
            
        Returns:
            List of function statistics sorted by metric
        """
        functions = []
        
        for name, stats in self.function_stats.items():
            func_data = {
                'name': name,
                'call_count': stats['call_count'],
                'total_time': stats['total_time'],
                'avg_time': stats['avg_time'],
                'min_time': stats['min_time'],
                'max_time': stats['max_time']
            }
            functions.append(func_data)
        
        # Sort by specified metric
        if sort_by in ['total_time', 'avg_time', 'call_count', 'max_time']:
            functions.sort(key=lambda x: x[sort_by], reverse=True)
        
        return functions[:limit]
    
    def generate_report(self) -> str:
        """Generate comprehensive performance report.
        
        Returns:
            Formatted performance report string
        """
        report = ["=== ConfoRL Performance Report ===\n"]
        
        # System statistics
        system_stats = self.get_system_stats()
        if system_stats:
            report.append("System Resources:")
            if 'cpu' in system_stats:
                cpu = system_stats['cpu']
                report.append(f"  CPU: {cpu['current']:.1f}% (avg: {cpu['average']:.1f}%, max: {cpu['max']:.1f}%)")
            
            if 'memory' in system_stats:
                mem = system_stats['memory']
                report.append(f"  Memory: {mem['current_mb']:.1f} MB ({mem['current_percent']:.1f}%)")
            
            report.append("")
        
        # Top functions by total time
        top_functions = self.get_top_functions(limit=10, sort_by='total_time')
        if top_functions:
            report.append("Top Functions by Total Time:")
            report.append(f"{'Function':<40} {'Calls':<8} {'Total (s)':<10} {'Avg (ms)':<10} {'Max (ms)':<10}")
            report.append("-" * 88)
            
            for func in top_functions:
                report.append(f"{func['name']:<40} {func['call_count']:<8} "
                            f"{func['total_time']:<10.3f} {func['avg_time']*1000:<10.2f} "
                            f"{func['max_time']*1000:<10.2f}")
            
            report.append("")
        
        # Performance summary
        total_functions = len(self.function_stats)
        total_calls = sum(stats['call_count'] for stats in self.function_stats.values())
        total_time = sum(stats['total_time'] for stats in self.function_stats.values())
        
        report.append("Performance Summary:")
        report.append(f"  Functions profiled: {total_functions}")
        report.append(f"  Total function calls: {total_calls}")
        report.append(f"  Total execution time: {total_time:.3f} seconds")
        
        if total_calls > 0:
            avg_call_time = total_time / total_calls
            report.append(f"  Average call time: {avg_call_time*1000:.2f} ms")
        
        return "\n".join(report)
    
    def reset_stats(self):
        """Reset all performance statistics."""
        self.function_stats.clear()
        for key in self.system_stats:
            self.system_stats[key].clear()
        
        logger.info("Performance statistics reset")


class MemoryProfiler:
    """Specialized memory profiler for tracking memory leaks and usage patterns."""
    
    def __init__(self):
        """Initialize memory profiler."""
        self.snapshots = []
        self.tracking_active = False
        
        if not tracemalloc.is_tracing():
            tracemalloc.start()
        
        logger.info("Memory profiler initialized")
    
    def take_snapshot(self, name: str = None) -> int:
        """Take a memory snapshot.
        
        Args:
            name: Optional name for the snapshot
            
        Returns:
            Snapshot index
        """
        if not tracemalloc.is_tracing():
            logger.warning("tracemalloc not active - starting now")
            tracemalloc.start()
        
        snapshot = tracemalloc.take_snapshot()
        snapshot_data = {
            'snapshot': snapshot,
            'name': name or f"snapshot_{len(self.snapshots)}",
            'timestamp': time.time(),
            'memory_usage': self._get_current_memory()
        }
        
        self.snapshots.append(snapshot_data)
        logger.info(f"Memory snapshot taken: {snapshot_data['name']}")
        
        return len(self.snapshots) - 1
    
    def _get_current_memory(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'rss_mb': memory_info.rss / 1024 / 1024,
                'vms_mb': memory_info.vms / 1024 / 1024,
                'percent': process.memory_percent()
            }
        except:
            return {'rss_mb': 0, 'vms_mb': 0, 'percent': 0}
    
    def compare_snapshots(
        self,
        snapshot1_idx: int,
        snapshot2_idx: int,
        limit: int = 10
    ) -> Dict[str, Any]:
        """Compare two memory snapshots.
        
        Args:
            snapshot1_idx: Index of first snapshot
            snapshot2_idx: Index of second snapshot
            limit: Number of top differences to show
            
        Returns:
            Dictionary with comparison results
        """
        if not (0 <= snapshot1_idx < len(self.snapshots) and 
                0 <= snapshot2_idx < len(self.snapshots)):
            raise ValueError("Invalid snapshot indices")
        
        snapshot1 = self.snapshots[snapshot1_idx]['snapshot']
        snapshot2 = self.snapshots[snapshot2_idx]['snapshot']
        
        # Compare snapshots
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        
        # Extract top differences
        differences = []
        for stat in top_stats[:limit]:
            differences.append({
                'file': stat.traceback.format(),
                'size_diff_mb': stat.size_diff / 1024 / 1024,
                'count_diff': stat.count_diff,
                'size_mb': stat.size / 1024 / 1024,
                'count': stat.count
            })
        
        # Memory usage comparison
        mem1 = self.snapshots[snapshot1_idx]['memory_usage']
        mem2 = self.snapshots[snapshot2_idx]['memory_usage']
        
        memory_diff = {
            'rss_mb_diff': mem2['rss_mb'] - mem1['rss_mb'],
            'vms_mb_diff': mem2['vms_mb'] - mem1['vms_mb'],
            'percent_diff': mem2['percent'] - mem1['percent']
        }
        
        return {
            'snapshot1': self.snapshots[snapshot1_idx]['name'],
            'snapshot2': self.snapshots[snapshot2_idx]['name'],
            'memory_diff': memory_diff,
            'top_differences': differences,
            'total_size_diff_mb': sum(d['size_diff_mb'] for d in differences)
        }
    
    def detect_memory_leaks(self, threshold_mb: float = 10.0) -> List[Dict[str, Any]]:
        """Detect potential memory leaks by analyzing snapshots.
        
        Args:
            threshold_mb: Threshold for flagging memory growth (MB)
            
        Returns:
            List of potential memory leaks
        """
        if len(self.snapshots) < 2:
            return []
        
        leaks = []
        
        # Compare consecutive snapshots
        for i in range(1, len(self.snapshots)):
            comparison = self.compare_snapshots(i-1, i)
            
            if comparison['memory_diff']['rss_mb_diff'] > threshold_mb:
                leaks.append({
                    'between_snapshots': f"{comparison['snapshot1']} -> {comparison['snapshot2']}",
                    'memory_growth_mb': comparison['memory_diff']['rss_mb_diff'],
                    'top_contributors': comparison['top_differences'][:3]
                })
        
        return leaks
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get summary of memory usage across all snapshots.
        
        Returns:
            Dictionary with memory usage summary
        """
        if not self.snapshots:
            return {}
        
        memory_usage = [s['memory_usage']['rss_mb'] for s in self.snapshots]
        
        return {
            'snapshots_count': len(self.snapshots),
            'memory_usage': {
                'min_mb': min(memory_usage),
                'max_mb': max(memory_usage),
                'current_mb': memory_usage[-1],
                'growth_mb': memory_usage[-1] - memory_usage[0] if len(memory_usage) > 1 else 0
            },
            'potential_leaks': len(self.detect_memory_leaks())
        }