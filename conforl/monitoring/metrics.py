"""Comprehensive metrics collection and tracking."""

import time
import threading
from typing import Dict, List, Any, Optional, Callable, Union
from collections import defaultdict, deque
import statistics
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MetricValue:
    """Container for a metric value with metadata."""
    value: Union[int, float]
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricSummary:
    """Summary statistics for a metric."""
    name: str
    count: int
    mean: float
    median: float
    std_dev: float
    min_value: float
    max_value: float
    percentile_95: float
    percentile_99: float
    last_value: float
    last_timestamp: float


class MetricsCollector:
    """Thread-safe metrics collector with automatic aggregation."""
    
    def __init__(
        self,
        buffer_size: int = 10000,
        aggregation_interval: float = 60.0,
        auto_export: bool = True
    ):
        """Initialize metrics collector.
        
        Args:
            buffer_size: Maximum number of metric values to store per metric
            aggregation_interval: Interval for automatic aggregation (seconds)
            auto_export: Whether to automatically export metrics
        """
        self.buffer_size = buffer_size
        self.aggregation_interval = aggregation_interval
        self.auto_export = auto_export
        
        # Metric storage
        self._metrics = defaultdict(lambda: deque(maxlen=buffer_size))
        self._aggregated_metrics = {}
        self._metric_metadata = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Aggregation thread
        self._aggregation_thread = None
        self._stop_aggregation = threading.Event()
        
        # Export callbacks
        self._export_callbacks = []
        
        if auto_export:
            self._start_aggregation_thread()
        
        logger.info(f"MetricsCollector initialized with buffer_size={buffer_size}")
    
    def record(
        self,
        metric_name: str,
        value: Union[int, float],
        tags: Optional[Dict[str, str]] = None,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record a metric value.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            tags: Optional tags for the metric
            timestamp: Timestamp (current time if None)
            metadata: Additional metadata
        """
        if timestamp is None:
            timestamp = time.time()
        
        metric_value = MetricValue(
            value=float(value),
            timestamp=timestamp,
            tags=tags or {},
            metadata=metadata or {}
        )
        
        with self._lock:
            self._metrics[metric_name].append(metric_value)
            
            # Store metric metadata
            if metric_name not in self._metric_metadata:
                self._metric_metadata[metric_name] = {
                    'first_recorded': timestamp,
                    'data_type': type(value).__name__,
                    'tags_seen': set(),
                    'total_records': 0
                }
            
            # Update metadata
            meta = self._metric_metadata[metric_name]
            meta['total_records'] += 1
            meta['last_recorded'] = timestamp
            meta['tags_seen'].update(tags.keys() if tags else [])
    
    def increment(
        self,
        metric_name: str,
        value: Union[int, float] = 1,
        tags: Optional[Dict[str, str]] = None
    ):
        """Increment a counter metric.
        
        Args:
            metric_name: Name of the counter
            value: Amount to increment
            tags: Optional tags
        """
        self.record(f"{metric_name}.count", value, tags)
    
    def gauge(
        self,
        metric_name: str,
        value: Union[int, float],
        tags: Optional[Dict[str, str]] = None
    ):
        """Record a gauge metric (current value).
        
        Args:
            metric_name: Name of the gauge
            value: Current value
            tags: Optional tags
        """
        self.record(f"{metric_name}.gauge", value, tags)
    
    def histogram(
        self,
        metric_name: str,
        value: Union[int, float],
        tags: Optional[Dict[str, str]] = None
    ):
        """Record a histogram metric (for timing, sizes, etc.).
        
        Args:
            metric_name: Name of the histogram
            value: Value to record
            tags: Optional tags
        """
        self.record(f"{metric_name}.histogram", value, tags)
    
    def timer(self, metric_name: str, tags: Optional[Dict[str, str]] = None):
        """Context manager for timing operations.
        
        Args:
            metric_name: Name of the timer metric
            tags: Optional tags
            
        Returns:
            Timer context manager
        """
        return TimerContext(self, metric_name, tags)
    
    def get_metric_summary(self, metric_name: str) -> Optional[MetricSummary]:
        """Get summary statistics for a metric.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            Metric summary or None if metric doesn't exist
        """
        with self._lock:
            if metric_name not in self._metrics:
                return None
            
            values = [mv.value for mv in self._metrics[metric_name]]
            if not values:
                return None
            
            # Calculate statistics
            count = len(values)
            mean = statistics.mean(values)
            median = statistics.median(values)
            
            if count > 1:
                std_dev = statistics.stdev(values)
            else:
                std_dev = 0.0
            
            min_value = min(values)
            max_value = max(values)
            
            # Percentiles
            sorted_values = sorted(values)
            percentile_95 = sorted_values[int(0.95 * count)] if count > 0 else 0
            percentile_99 = sorted_values[int(0.99 * count)] if count > 0 else 0
            
            # Last value info
            last_metric = self._metrics[metric_name][-1]
            
            return MetricSummary(
                name=metric_name,
                count=count,
                mean=mean,
                median=median,
                std_dev=std_dev,
                min_value=min_value,
                max_value=max_value,
                percentile_95=percentile_95,
                percentile_99=percentile_99,
                last_value=last_metric.value,
                last_timestamp=last_metric.timestamp
            )
    
    def get_all_metrics(self) -> Dict[str, List[MetricValue]]:
        """Get all stored metrics.
        
        Returns:
            Dictionary of metric name to list of values
        """
        with self._lock:
            return {name: list(values) for name, values in self._metrics.items()}
    
    def get_metric_names(self) -> List[str]:
        """Get list of all metric names.
        
        Returns:
            List of metric names
        """
        with self._lock:
            return list(self._metrics.keys())
    
    def clear_metrics(self, metric_names: Optional[List[str]] = None):
        """Clear stored metrics.
        
        Args:
            metric_names: Specific metrics to clear (all if None)
        """
        with self._lock:
            if metric_names is None:
                self._metrics.clear()
                self._metric_metadata.clear()
                logger.info("Cleared all metrics")
            else:
                for name in metric_names:
                    if name in self._metrics:
                        del self._metrics[name]
                    if name in self._metric_metadata:
                        del self._metric_metadata[name]
                logger.info(f"Cleared metrics: {metric_names}")
    
    def add_export_callback(self, callback: Callable[[Dict[str, MetricSummary]], None]):
        """Add callback for metric export.
        
        Args:
            callback: Function to call with aggregated metrics
        """
        self._export_callbacks.append(callback)
        logger.debug("Added metrics export callback")
    
    def _start_aggregation_thread(self):
        """Start background aggregation thread."""
        if self._aggregation_thread is not None:
            return
        
        self._aggregation_thread = threading.Thread(
            target=self._aggregation_loop,
            daemon=True
        )
        self._aggregation_thread.start()
        logger.debug("Started metrics aggregation thread")
    
    def _aggregation_loop(self):
        """Background loop for metric aggregation and export."""
        while not self._stop_aggregation.wait(self.aggregation_interval):
            try:
                self._aggregate_and_export()
            except Exception as e:
                logger.error(f"Error in metrics aggregation: {e}")
    
    def _aggregate_and_export(self):
        """Aggregate metrics and call export callbacks."""
        summaries = {}
        
        with self._lock:
            for metric_name in list(self._metrics.keys()):
                summary = self.get_metric_summary(metric_name)
                if summary:
                    summaries[metric_name] = summary
        
        if summaries:
            # Store aggregated metrics
            self._aggregated_metrics.update(summaries)
            
            # Call export callbacks
            for callback in self._export_callbacks:
                try:
                    callback(summaries)
                except Exception as e:
                    logger.error(f"Error in metrics export callback: {e}")
    
    def stop(self):
        """Stop the metrics collector."""
        if self._aggregation_thread:
            self._stop_aggregation.set()
            self._aggregation_thread.join(timeout=5.0)
        
        logger.info("MetricsCollector stopped")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


class TimerContext:
    """Context manager for timing code execution."""
    
    def __init__(
        self,
        metrics_collector: MetricsCollector,
        metric_name: str,
        tags: Optional[Dict[str, str]] = None
    ):
        """Initialize timer context.
        
        Args:
            metrics_collector: Metrics collector instance
            metric_name: Name of the timer metric
            tags: Optional tags
        """
        self.metrics_collector = metrics_collector
        self.metric_name = metric_name
        self.tags = tags
        self.start_time = None
    
    def __enter__(self):
        """Start timing."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and record metric."""
        if self.start_time is not None:
            duration = time.time() - self.start_time
            self.metrics_collector.histogram(
                f"{self.metric_name}.duration",
                duration,
                self.tags
            )


class PerformanceTracker:
    """High-level performance tracking for ConfoRL operations."""
    
    def __init__(self, metrics_collector: Optional[MetricsCollector] = None):
        """Initialize performance tracker.
        
        Args:
            metrics_collector: Metrics collector to use (creates new if None)
        """
        self.metrics = metrics_collector or MetricsCollector()
        self.session_start = time.time()
        
        # Performance counters
        self.training_episodes = 0
        self.training_steps = 0
        self.inference_requests = 0
        self.risk_evaluations = 0
        
        logger.info("PerformanceTracker initialized")
    
    def track_training_episode(
        self,
        episode_reward: float,
        episode_length: int,
        episode_risk: float,
        algorithm: str = "unknown"
    ):
        """Track training episode performance.
        
        Args:
            episode_reward: Total episode reward
            episode_length: Number of steps in episode
            episode_risk: Episode risk value
            algorithm: Algorithm name
        """
        tags = {"algorithm": algorithm}
        
        self.metrics.record("training.episode.reward", episode_reward, tags)
        self.metrics.record("training.episode.length", episode_length, tags)
        self.metrics.record("training.episode.risk", episode_risk, tags)
        self.metrics.increment("training.episodes.total", 1, tags)
        
        self.training_episodes += 1
        self.training_steps += episode_length
    
    def track_training_step(
        self,
        step_reward: float,
        step_risk: float,
        action_value: Optional[float] = None,
        algorithm: str = "unknown"
    ):
        """Track individual training step.
        
        Args:
            step_reward: Step reward
            step_risk: Step risk
            action_value: Action value (Q-value, etc.)
            algorithm: Algorithm name
        """
        tags = {"algorithm": algorithm}
        
        self.metrics.record("training.step.reward", step_reward, tags)
        self.metrics.record("training.step.risk", step_risk, tags)
        
        if action_value is not None:
            self.metrics.record("training.step.action_value", action_value, tags)
        
        self.metrics.increment("training.steps.total", 1, tags)
    
    def track_inference(
        self,
        inference_time: float,
        risk_bound: float,
        confidence: float,
        algorithm: str = "unknown"
    ):
        """Track inference performance.
        
        Args:
            inference_time: Time taken for inference
            risk_bound: Risk bound from inference
            confidence: Confidence level
            algorithm: Algorithm name
        """
        tags = {"algorithm": algorithm}
        
        self.metrics.histogram("inference.duration", inference_time, tags)
        self.metrics.record("inference.risk_bound", risk_bound, tags)
        self.metrics.record("inference.confidence", confidence, tags)
        self.metrics.increment("inference.requests.total", 1, tags)
        
        self.inference_requests += 1
    
    def track_risk_evaluation(
        self,
        risk_value: float,
        risk_type: str,
        evaluation_time: float
    ):
        """Track risk evaluation performance.
        
        Args:
            risk_value: Computed risk value
            risk_type: Type of risk measure
            evaluation_time: Time taken for evaluation
        """
        tags = {"risk_type": risk_type}
        
        self.metrics.record("risk.value", risk_value, tags)
        self.metrics.histogram("risk.evaluation_time", evaluation_time, tags)
        self.metrics.increment("risk.evaluations.total", 1, tags)
        
        self.risk_evaluations += 1
    
    def track_memory_usage(self, memory_mb: float, memory_type: str = "system"):
        """Track memory usage.
        
        Args:
            memory_mb: Memory usage in MB
            memory_type: Type of memory (system, gpu, etc.)
        """
        tags = {"memory_type": memory_type}
        self.metrics.gauge("system.memory_usage", memory_mb, tags)
    
    def track_cpu_usage(self, cpu_percent: float):
        """Track CPU usage.
        
        Args:
            cpu_percent: CPU usage percentage
        """
        self.metrics.gauge("system.cpu_usage", cpu_percent)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary.
        
        Returns:
            Performance summary dictionary
        """
        session_duration = time.time() - self.session_start
        
        summary = {
            "session_duration_seconds": session_duration,
            "training_episodes": self.training_episodes,
            "training_steps": self.training_steps,
            "inference_requests": self.inference_requests,
            "risk_evaluations": self.risk_evaluations,
            "metrics": {}
        }
        
        # Add metric summaries
        for metric_name in self.metrics.get_metric_names():
            metric_summary = self.metrics.get_metric_summary(metric_name)
            if metric_summary:
                summary["metrics"][metric_name] = {
                    "count": metric_summary.count,
                    "mean": metric_summary.mean,
                    "std_dev": metric_summary.std_dev,
                    "p95": metric_summary.percentile_95,
                    "p99": metric_summary.percentile_99
                }
        
        return summary


class RiskMetrics:
    """Specialized metrics for risk monitoring and analysis."""
    
    def __init__(self, metrics_collector: Optional[MetricsCollector] = None):
        """Initialize risk metrics tracker.
        
        Args:
            metrics_collector: Metrics collector to use
        """
        self.metrics = metrics_collector or MetricsCollector()
        
        # Risk thresholds for alerting
        self.risk_thresholds = {
            "low": 0.05,
            "medium": 0.15,
            "high": 0.25,
            "critical": 0.5
        }
        
        logger.info("RiskMetrics initialized")
    
    def track_risk_violation(
        self,
        risk_value: float,
        risk_threshold: float,
        severity: str,
        context: Optional[Dict[str, str]] = None
    ):
        """Track risk violation event.
        
        Args:
            risk_value: Observed risk value
            risk_threshold: Risk threshold that was violated
            severity: Severity level
            context: Additional context information
        """
        tags = {"severity": severity}
        if context:
            tags.update(context)
        
        self.metrics.record("risk.violation.value", risk_value, tags)
        self.metrics.record("risk.violation.threshold", risk_threshold, tags)
        self.metrics.increment("risk.violations.count", 1, tags)
        
        # Track violation magnitude
        violation_magnitude = risk_value - risk_threshold
        self.metrics.record("risk.violation.magnitude", violation_magnitude, tags)
    
    def track_safety_intervention(
        self,
        intervention_type: str,
        intervention_duration: float,
        success: bool
    ):
        """Track safety intervention event.
        
        Args:
            intervention_type: Type of intervention (fallback, emergency_stop, etc.)
            intervention_duration: Duration of intervention
            success: Whether intervention was successful
        """
        tags = {
            "intervention_type": intervention_type,
            "success": str(success)
        }
        
        self.metrics.increment("safety.interventions.count", 1, tags)
        self.metrics.histogram("safety.intervention.duration", intervention_duration, tags)
    
    def track_risk_trend(
        self,
        current_risk: float,
        previous_risk: float,
        window_size: int
    ):
        """Track risk trend over time.
        
        Args:
            current_risk: Current risk value
            previous_risk: Previous risk value
            window_size: Size of trend analysis window
        """
        risk_change = current_risk - previous_risk
        risk_change_percent = (risk_change / previous_risk * 100) if previous_risk > 0 else 0
        
        tags = {"window_size": str(window_size)}
        
        self.metrics.record("risk.trend.change", risk_change, tags)
        self.metrics.record("risk.trend.change_percent", risk_change_percent, tags)
    
    def track_certificate_validity(
        self,
        certificate_age_seconds: float,
        coverage_guarantee: float,
        sample_size: int,
        method: str
    ):
        """Track risk certificate validity metrics.
        
        Args:
            certificate_age_seconds: Age of certificate in seconds
            coverage_guarantee: Coverage guarantee level
            sample_size: Sample size used for certificate
            method: Conformal method used
        """
        tags = {"method": method}
        
        self.metrics.record("risk.certificate.age", certificate_age_seconds, tags)
        self.metrics.record("risk.certificate.coverage", coverage_guarantee, tags)
        self.metrics.record("risk.certificate.sample_size", sample_size, tags)
    
    def get_risk_alert_level(self, current_risk: float) -> str:
        """Determine alert level based on current risk.
        
        Args:
            current_risk: Current risk value
            
        Returns:
            Alert level (low, medium, high, critical)
        """
        if current_risk >= self.risk_thresholds["critical"]:
            return "critical"
        elif current_risk >= self.risk_thresholds["high"]:
            return "high"
        elif current_risk >= self.risk_thresholds["medium"]:
            return "medium"
        else:
            return "low"
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk metrics summary.
        
        Returns:
            Risk summary dictionary
        """
        summary = {
            "thresholds": self.risk_thresholds,
            "metrics": {}
        }
        
        # Get risk-related metric summaries
        risk_metrics = [name for name in self.metrics.get_metric_names() if "risk" in name]
        
        for metric_name in risk_metrics:
            metric_summary = self.metrics.get_metric_summary(metric_name)
            if metric_summary:
                summary["metrics"][metric_name] = {
                    "current": metric_summary.last_value,
                    "mean": metric_summary.mean,
                    "max": metric_summary.max_value,
                    "count": metric_summary.count
                }
        
        return summary