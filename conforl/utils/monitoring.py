"""
Comprehensive monitoring and metrics collection system.
Generation 2: Real-time monitoring, alerting, and health checks.
"""

import time
import json
import threading
from typing import Any, Dict, List, Optional, Callable, Union
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from enum import Enum

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    class np:
        @staticmethod
        def mean(data): return sum(data) / len(data) if data else 0
        @staticmethod 
        def std(data): 
            if not data: return 0
            mean_val = sum(data) / len(data)
            return (sum((x - mean_val)**2 for x in data) / len(data)) ** 0.5
        @staticmethod
        def percentile(data, q): 
            if not data: return 0
            sorted_data = sorted(data)
            idx = int(q / 100 * len(sorted_data))
            return sorted_data[min(idx, len(sorted_data) - 1)]


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricPoint:
    """Single metric measurement point."""
    timestamp: float
    value: Union[int, float]
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Alert:
    """System alert with details."""
    level: AlertLevel
    message: str
    timestamp: float
    metric_name: str
    current_value: Union[int, float]
    threshold: Union[int, float]
    metadata: Optional[Dict[str, Any]] = None


class HealthChecker:
    """System health monitoring and checks."""
    
    def __init__(self):
        """Initialize health checker."""
        self.checks = {}
        self.last_check_time = {}
        self.health_history = deque(maxlen=100)
        
    def register_check(self, name: str, check_func: Callable[[], bool], interval: float = 60.0):
        """Register a health check function.
        
        Args:
            name: Check name
            check_func: Function that returns True if healthy
            interval: Check interval in seconds
        """
        self.checks[name] = {
            'function': check_func,
            'interval': interval,
            'last_result': None,
            'last_error': None
        }
        
    def run_check(self, name: str) -> bool:
        """Run a specific health check.
        
        Args:
            name: Check name
            
        Returns:
            True if healthy, False otherwise
        """
        if name not in self.checks:
            return False
            
        check = self.checks[name]
        current_time = time.time()
        
        # Skip if checked recently
        if (name in self.last_check_time and 
            current_time - self.last_check_time[name] < check['interval']):
            return check['last_result'] or False
        
        try:
            result = check['function']()
            check['last_result'] = result
            check['last_error'] = None
            self.last_check_time[name] = current_time
            
            self.health_history.append({
                'timestamp': current_time,
                'check': name,
                'result': result,
                'error': None
            })
            
            return result
            
        except Exception as e:
            check['last_result'] = False
            check['last_error'] = str(e)
            self.last_check_time[name] = current_time
            
            self.health_history.append({
                'timestamp': current_time,
                'check': name,
                'result': False,
                'error': str(e)
            })
            
            return False
    
    def run_all_checks(self) -> Dict[str, bool]:
        """Run all health checks.
        
        Returns:
            Dictionary of check results
        """
        results = {}
        for name in self.checks:
            results[name] = self.run_check(name)
        return results
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status.
        
        Returns:
            Health status summary
        """
        results = self.run_all_checks()
        
        return {
            'overall_healthy': all(results.values()),
            'individual_checks': results,
            'total_checks': len(self.checks),
            'passing_checks': sum(results.values()),
            'last_check_time': max(self.last_check_time.values()) if self.last_check_time else None,
            'recent_failures': [
                h for h in list(self.health_history)[-10:] 
                if not h['result']
            ]
        }


class MetricsCollector:
    """Advanced metrics collection and analysis system."""
    
    def __init__(self, retention_hours: int = 24):
        """Initialize metrics collector.
        
        Args:
            retention_hours: How long to keep metrics in memory
        """
        self.retention_seconds = retention_hours * 3600
        self.metrics = defaultdict(lambda: deque())
        self.counters = defaultdict(int)
        self.gauges = defaultdict(float)
        self.histograms = defaultdict(list)
        
        # Alert configuration
        self.alert_thresholds = {}
        self.alert_callbacks = []
        self.alerts_history = deque(maxlen=1000)
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Health checker
        self.health_checker = HealthChecker()
        self._register_default_health_checks()
    
    def _register_default_health_checks(self):
        """Register default health checks."""
        self.health_checker.register_check(
            "metrics_collection",
            lambda: len(self.metrics) > 0,
            interval=30.0
        )
        
        self.health_checker.register_check(
            "memory_usage",
            lambda: self._check_memory_usage(),
            interval=60.0
        )
    
    def _check_memory_usage(self) -> bool:
        """Check if memory usage is reasonable."""
        total_points = sum(len(points) for points in self.metrics.values())
        return total_points < 100000  # Arbitrary limit
    
    def record_metric(
        self, 
        name: str, 
        value: Union[int, float], 
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            metadata: Additional metadata
        """
        with self._lock:
            timestamp = time.time()
            point = MetricPoint(timestamp, value, metadata)
            
            self.metrics[name].append(point)
            self.gauges[name] = value  # Update gauge
            
            # Clean old data
            self._clean_old_data(name)
            
            # Check for alerts
            self._check_alerts(name, value)
    
    def increment_counter(self, name: str, amount: int = 1):
        """Increment a counter metric.
        
        Args:
            name: Counter name
            amount: Amount to increment
        """
        with self._lock:
            self.counters[name] += amount
            self.record_metric(f"{name}_total", self.counters[name])
    
    def record_histogram(self, name: str, value: Union[int, float]):
        """Record a value in histogram.
        
        Args:
            name: Histogram name
            value: Value to record
        """
        with self._lock:
            self.histograms[name].append(value)
            
            # Keep only recent values
            if len(self.histograms[name]) > 10000:
                self.histograms[name] = self.histograms[name][-5000:]
            
            self.record_metric(name, value)
    
    def set_alert_threshold(
        self, 
        metric_name: str, 
        threshold: Union[int, float],
        level: AlertLevel = AlertLevel.WARNING,
        condition: str = "greater"
    ):
        """Set alert threshold for a metric.
        
        Args:
            metric_name: Name of metric to monitor
            threshold: Threshold value
            level: Alert level
            condition: 'greater', 'less', 'equal'
        """
        self.alert_thresholds[metric_name] = {
            'threshold': threshold,
            'level': level,
            'condition': condition
        }
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add callback for alert notifications.
        
        Args:
            callback: Function to call when alert is triggered
        """
        self.alert_callbacks.append(callback)
    
    def _check_alerts(self, metric_name: str, value: Union[int, float]):
        """Check if metric value triggers an alert."""
        if metric_name not in self.alert_thresholds:
            return
        
        config = self.alert_thresholds[metric_name]
        threshold = config['threshold']
        condition = config['condition']
        
        triggered = False
        if condition == "greater" and value > threshold:
            triggered = True
        elif condition == "less" and value < threshold:
            triggered = True
        elif condition == "equal" and value == threshold:
            triggered = True
        
        if triggered:
            alert = Alert(
                level=config['level'],
                message=f"Metric {metric_name} triggered alert: {value} {condition} {threshold}",
                timestamp=time.time(),
                metric_name=metric_name,
                current_value=value,
                threshold=threshold
            )
            
            self.alerts_history.append(alert)
            
            # Notify callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    print(f"Alert callback failed: {e}")
    
    def _clean_old_data(self, metric_name: str):
        """Clean old metric data based on retention policy."""
        current_time = time.time()
        cutoff_time = current_time - self.retention_seconds
        
        points = self.metrics[metric_name]
        while points and points[0].timestamp < cutoff_time:
            points.popleft()
    
    def get_metric_stats(self, metric_name: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a metric.
        
        Args:
            metric_name: Name of metric
            
        Returns:
            Statistics dictionary or None if metric doesn't exist
        """
        if metric_name not in self.metrics:
            return None
        
        with self._lock:
            points = list(self.metrics[metric_name])
        
        if not points:
            return None
        
        values = [p.value for p in points]
        
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': np.mean(values),
            'std': np.std(values),
            'p50': np.percentile(values, 50),
            'p95': np.percentile(values, 95),
            'p99': np.percentile(values, 99),
            'current': values[-1] if values else None,
            'first_timestamp': points[0].timestamp,
            'last_timestamp': points[-1].timestamp
        }
    
    def get_all_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics.
        
        Returns:
            Summary of all collected metrics
        """
        with self._lock:
            metric_names = list(self.metrics.keys())
            counter_names = list(self.counters.keys())
            histogram_names = list(self.histograms.keys())
        
        return {
            'total_metrics': len(metric_names),
            'total_counters': len(counter_names),
            'total_histograms': len(histogram_names),
            'metric_names': metric_names,
            'counter_values': dict(self.counters),
            'gauge_values': dict(self.gauges),
            'recent_alerts': [asdict(alert) for alert in list(self.alerts_history)[-10:]],
            'health_status': self.health_checker.get_health_status()
        }
    
    def export_prometheus_format(self) -> str:
        """Export metrics in Prometheus format.
        
        Returns:
            Prometheus-formatted metrics string
        """
        lines = []
        
        # Export gauges
        for name, value in self.gauges.items():
            safe_name = name.replace('-', '_').replace('.', '_')
            lines.append(f"# TYPE conforl_{safe_name} gauge")
            lines.append(f"conforl_{safe_name} {value}")
        
        # Export counters
        for name, value in self.counters.items():
            safe_name = name.replace('-', '_').replace('.', '_')
            lines.append(f"# TYPE conforl_{safe_name}_total counter")
            lines.append(f"conforl_{safe_name}_total {value}")
        
        # Export histograms (simplified)
        for name, values in self.histograms.items():
            if values:
                safe_name = name.replace('-', '_').replace('.', '_')
                lines.append(f"# TYPE conforl_{safe_name} histogram")
                lines.append(f"conforl_{safe_name}_sum {sum(values)}")
                lines.append(f"conforl_{safe_name}_count {len(values)}")
        
        return '\n'.join(lines)
    
    def export_json_format(self) -> str:
        """Export metrics in JSON format.
        
        Returns:
            JSON-formatted metrics string
        """
        data = {
            'timestamp': time.time(),
            'summary': self.get_all_metrics_summary(),
            'detailed_stats': {
                name: self.get_metric_stats(name)
                for name in self.metrics.keys()
            }
        }
        return json.dumps(data, indent=2)


# Global metrics collector instance
_metrics_collector = MetricsCollector()

def record_metric(name: str, value: Union[int, float], **metadata):
    """Global function to record a metric."""
    _metrics_collector.record_metric(name, value, metadata or None)

def increment_counter(name: str, amount: int = 1):
    """Global function to increment a counter."""
    _metrics_collector.increment_counter(name, amount)

def record_timing(name: str, duration: float):
    """Global function to record timing metrics."""
    _metrics_collector.record_histogram(f"{name}_duration_seconds", duration)
    _metrics_collector.record_metric(f"{name}_last_duration", duration)

def get_metrics_summary() -> Dict[str, Any]:
    """Global function to get metrics summary."""
    return _metrics_collector.get_all_metrics_summary()

def set_alert_threshold(metric_name: str, threshold: Union[int, float], level: AlertLevel = AlertLevel.WARNING):
    """Global function to set alert threshold."""
    _metrics_collector.set_alert_threshold(metric_name, threshold, level)

def get_health_status() -> Dict[str, Any]:
    """Global function to get health status."""
    return _metrics_collector.health_checker.get_health_status()


class MonitoringContext:
    """Context manager for automatic metrics recording."""
    
    def __init__(self, operation_name: str, record_errors: bool = True):
        """Initialize monitoring context.
        
        Args:
            operation_name: Name of operation being monitored
            record_errors: Whether to record errors as metrics
        """
        self.operation_name = operation_name
        self.record_errors = record_errors
        self.start_time = None
    
    def __enter__(self):
        """Enter monitoring context."""
        self.start_time = time.time()
        increment_counter(f"{self.operation_name}_started")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit monitoring context."""
        duration = time.time() - self.start_time
        record_timing(self.operation_name, duration)
        
        if exc_type is None:
            increment_counter(f"{self.operation_name}_success")
        else:
            increment_counter(f"{self.operation_name}_error")
            if self.record_errors:
                record_metric(f"{self.operation_name}_last_error", 1, {
                    'error_type': exc_type.__name__,
                    'error_message': str(exc_val)
                })


def monitor_operation(operation_name: str):
    """Decorator for automatic operation monitoring.
    
    Args:
        operation_name: Name of operation to monitor
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            with MonitoringContext(operation_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator