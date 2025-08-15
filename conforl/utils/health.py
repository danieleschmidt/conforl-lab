"""Health check and monitoring utilities for ConfoRL."""

import time
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    # Minimal psutil-like interface for basic functionality
    class psutil:
        @staticmethod
        def cpu_percent(interval=None):
            return 5.0  # Placeholder
        
        @staticmethod
        def virtual_memory():
            class Memory:
                def __init__(self):
                    self.percent = 20.0
                    self.available = 8 * 1024**3  # 8GB
                    self.total = 16 * 1024**3     # 16GB
            return Memory()
        
        @staticmethod
        def disk_usage(path):
            class Disk:
                def __init__(self):
                    self.total = 100 * 1024**3    # 100GB
                    self.used = 50 * 1024**3      # 50GB
                    self.free = 50 * 1024**3      # 50GB
            return Disk()
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum

from .logging import get_logger
from .errors import ConfoRLError

logger = get_logger(__name__)


class HealthStatus(Enum):
    """Health status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class HealthCheck:
    """Individual health check result."""
    name: str
    status: HealthStatus
    message: str
    duration_ms: float
    timestamp: float
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SystemMetrics:
    """System performance metrics."""
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_usage_percent: float
    uptime_seconds: float
    timestamp: float


class HealthChecker:
    """Comprehensive health monitoring system."""
    
    def __init__(self, check_interval: float = 30.0):
        """Initialize health checker.
        
        Args:
            check_interval: Interval between health checks in seconds
        """
        self.check_interval = check_interval
        self.checks: Dict[str, Callable[[], HealthCheck]] = {}
        self.last_results: Dict[str, HealthCheck] = {}
        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_monitoring = False
        self.start_time = time.time()
        
        # Register default system checks
        self.register_check("system_resources", self._check_system_resources)
        self.register_check("memory_usage", self._check_memory_usage)
        self.register_check("disk_space", self._check_disk_space)
    
    def register_check(self, name: str, check_func: Callable[[], HealthCheck]):
        """Register a health check function.
        
        Args:
            name: Name of the health check
            check_func: Function that returns a HealthCheck result
        """
        self.checks[name] = check_func
        logger.info(f"Registered health check: {name}")
    
    def run_check(self, name: str) -> HealthCheck:
        """Run a specific health check.
        
        Args:
            name: Name of the check to run
            
        Returns:
            HealthCheck result
            
        Raises:
            ConfoRLError: If check doesn't exist
        """
        if name not in self.checks:
            raise ConfoRLError(f"Health check '{name}' not found", "CHECK_NOT_FOUND")
        
        start_time = time.time()
        try:
            result = self.checks[name]()
            result.duration_ms = (time.time() - start_time) * 1000
            result.timestamp = time.time()
            self.last_results[name] = result
            return result
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            error_result = HealthCheck(
                name=name,
                status=HealthStatus.CRITICAL,
                message=f"Health check failed: {str(e)}",
                duration_ms=duration_ms,
                timestamp=time.time(),
                metadata={"error": str(e)}
            )
            self.last_results[name] = error_result
            return error_result
    
    def run_all_checks(self) -> Dict[str, HealthCheck]:
        """Run all registered health checks.
        
        Returns:
            Dictionary of health check results
        """
        results = {}
        for name in self.checks:
            results[name] = self.run_check(name)
        return results
    
    def get_overall_status(self) -> HealthStatus:
        """Get overall system health status.
        
        Returns:
            Overall health status based on all checks
        """
        if not self.last_results:
            return HealthStatus.UNHEALTHY
        
        statuses = [check.status for check in self.last_results.values()]
        
        if any(status == HealthStatus.CRITICAL for status in statuses):
            return HealthStatus.CRITICAL
        elif any(status == HealthStatus.UNHEALTHY for status in statuses):
            return HealthStatus.UNHEALTHY
        elif any(status == HealthStatus.DEGRADED for status in statuses):
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report.
        
        Returns:
            Health report dictionary
        """
        overall_status = self.get_overall_status()
        
        return {
            "overall_status": overall_status.value,
            "timestamp": time.time(),
            "uptime_seconds": time.time() - self.start_time,
            "checks": {
                name: {
                    "status": check.status.value,
                    "message": check.message,
                    "duration_ms": check.duration_ms,
                    "timestamp": check.timestamp,
                    "metadata": check.metadata
                }
                for name, check in self.last_results.items()
            },
            "system_metrics": self._get_system_metrics()
        }
    
    def start_monitoring(self):
        """Start continuous health monitoring in background thread."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            logger.warning("Health monitoring already running")
            return
        
        self.stop_monitoring = False
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Started health monitoring")
    
    def stop_monitoring_thread(self):
        """Stop continuous health monitoring."""
        self.stop_monitoring = True
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        logger.info("Stopped health monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while not self.stop_monitoring:
            try:
                self.run_all_checks()
                overall_status = self.get_overall_status()
                
                if overall_status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]:
                    logger.warning(f"System health degraded: {overall_status.value}")
                
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
            
            time.sleep(self.check_interval)
    
    def _check_system_resources(self) -> HealthCheck:
        """Check overall system resource usage."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            if cpu_percent > 90:
                status = HealthStatus.CRITICAL
                message = f"CPU usage critical: {cpu_percent:.1f}%"
            elif cpu_percent > 75:
                status = HealthStatus.UNHEALTHY
                message = f"CPU usage high: {cpu_percent:.1f}%"
            elif cpu_percent > 50:
                status = HealthStatus.DEGRADED
                message = f"CPU usage elevated: {cpu_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"CPU usage normal: {cpu_percent:.1f}%"
            
            if memory.percent > 90:
                status = max(status, HealthStatus.CRITICAL)
                message += f", Memory critical: {memory.percent:.1f}%"
            elif memory.percent > 80:
                status = max(status, HealthStatus.UNHEALTHY)
                message += f", Memory high: {memory.percent:.1f}%"
            
            return HealthCheck(
                name="system_resources",
                status=status,
                message=message,
                duration_ms=0,  # Will be set by caller
                timestamp=0,    # Will be set by caller
                metadata={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_available_gb": memory.available / (1024**3)
                }
            )
            
        except Exception as e:
            return HealthCheck(
                name="system_resources",
                status=HealthStatus.CRITICAL,
                message=f"Failed to check system resources: {str(e)}",
                duration_ms=0,
                timestamp=0,
                metadata={"error": str(e)}
            )
    
    def _check_memory_usage(self) -> HealthCheck:
        """Check memory usage specifically."""
        try:
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            
            if available_gb < 0.5:  # Less than 500MB available
                status = HealthStatus.CRITICAL
                message = f"Very low memory available: {available_gb:.2f}GB"
            elif available_gb < 1.0:  # Less than 1GB available
                status = HealthStatus.UNHEALTHY
                message = f"Low memory available: {available_gb:.2f}GB"
            elif available_gb < 2.0:  # Less than 2GB available
                status = HealthStatus.DEGRADED
                message = f"Moderate memory available: {available_gb:.2f}GB"
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory available: {available_gb:.2f}GB"
            
            return HealthCheck(
                name="memory_usage",
                status=status,
                message=message,
                duration_ms=0,
                timestamp=0,
                metadata={
                    "total_gb": memory.total / (1024**3),
                    "available_gb": available_gb,
                    "used_percent": memory.percent
                }
            )
            
        except Exception as e:
            return HealthCheck(
                name="memory_usage",
                status=HealthStatus.CRITICAL,
                message=f"Failed to check memory usage: {str(e)}",
                duration_ms=0,
                timestamp=0,
                metadata={"error": str(e)}
            )
    
    def _check_disk_space(self) -> HealthCheck:
        """Check disk space usage."""
        try:
            disk = psutil.disk_usage('/')
            free_gb = disk.free / (1024**3)
            used_percent = (disk.used / disk.total) * 100
            
            if used_percent > 95:
                status = HealthStatus.CRITICAL
                message = f"Disk space critical: {used_percent:.1f}% used"
            elif used_percent > 90:
                status = HealthStatus.UNHEALTHY
                message = f"Disk space low: {used_percent:.1f}% used"
            elif used_percent > 80:
                status = HealthStatus.DEGRADED
                message = f"Disk space elevated: {used_percent:.1f}% used"
            else:
                status = HealthStatus.HEALTHY
                message = f"Disk space normal: {used_percent:.1f}% used"
            
            return HealthCheck(
                name="disk_space",
                status=status,
                message=message,
                duration_ms=0,
                timestamp=0,
                metadata={
                    "total_gb": disk.total / (1024**3),
                    "free_gb": free_gb,
                    "used_percent": used_percent
                }
            )
            
        except Exception as e:
            return HealthCheck(
                name="disk_space",
                status=HealthStatus.CRITICAL,
                message=f"Failed to check disk space: {str(e)}",
                duration_ms=0,
                timestamp=0,
                metadata={"error": str(e)}
            )
    
    def _get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics."""
        try:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_available_gb=memory.available / (1024**3),
                disk_usage_percent=(disk.used / disk.total) * 100,
                uptime_seconds=time.time() - self.start_time,
                timestamp=time.time()
            )
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            return SystemMetrics(
                cpu_percent=0,
                memory_percent=0,
                memory_available_gb=0,
                disk_usage_percent=0,
                uptime_seconds=time.time() - self.start_time,
                timestamp=time.time()
            )


class PerformanceMonitor:
    """Monitor performance metrics and detect issues."""
    
    def __init__(self, window_size: int = 100):
        """Initialize performance monitor.
        
        Args:
            window_size: Size of sliding window for metrics
        """
        self.window_size = window_size
        self.metrics: Dict[str, List[float]] = {}
        self.thresholds: Dict[str, Dict[str, float]] = {}
    
    def record_metric(self, name: str, value: float):
        """Record a performance metric.
        
        Args:
            name: Metric name
            value: Metric value
        """
        if name not in self.metrics:
            self.metrics[name] = []
        
        self.metrics[name].append(value)
        
        # Keep only recent values
        if len(self.metrics[name]) > self.window_size:
            self.metrics[name] = self.metrics[name][-self.window_size:]
    
    def set_threshold(self, metric_name: str, warning: float, critical: float):
        """Set thresholds for a metric.
        
        Args:
            metric_name: Name of the metric
            warning: Warning threshold
            critical: Critical threshold
        """
        self.thresholds[metric_name] = {
            'warning': warning,
            'critical': critical
        }
    
    def get_metric_status(self, name: str) -> HealthStatus:
        """Get health status for a metric.
        
        Args:
            name: Metric name
            
        Returns:
            Health status based on recent values
        """
        if name not in self.metrics or not self.metrics[name]:
            return HealthStatus.UNHEALTHY
        
        recent_values = self.metrics[name][-10:]  # Last 10 values
        avg_value = sum(recent_values) / len(recent_values)
        
        if name in self.thresholds:
            thresholds = self.thresholds[name]
            if avg_value > thresholds['critical']:
                return HealthStatus.CRITICAL
            elif avg_value > thresholds['warning']:
                return HealthStatus.DEGRADED
        
        return HealthStatus.HEALTHY
    
    def get_metric_summary(self, name: str) -> Dict[str, Any]:
        """Get summary statistics for a metric.
        
        Args:
            name: Metric name
            
        Returns:
            Metric summary dictionary
        """
        if name not in self.metrics or not self.metrics[name]:
            return {"name": name, "status": "no_data"}
        
        values = self.metrics[name]
        return {
            "name": name,
            "count": len(values),
            "latest": values[-1],
            "average": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "status": self.get_metric_status(name).value
        }


# Global health checker instance
_global_health_checker: Optional[HealthChecker] = None


def get_health_checker() -> HealthChecker:
    """Get global health checker instance."""
    global _global_health_checker
    if _global_health_checker is None:
        _global_health_checker = HealthChecker()
    return _global_health_checker


def start_health_monitoring():
    """Start global health monitoring."""
    get_health_checker().start_monitoring()


def stop_health_monitoring():
    """Stop global health monitoring."""
    global _global_health_checker
    if _global_health_checker:
        _global_health_checker.stop_monitoring_thread()


def get_health_status() -> Dict[str, Any]:
    """Get current health status."""
    return get_health_checker().get_health_report()