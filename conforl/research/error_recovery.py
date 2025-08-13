"""Error Recovery and Fault Tolerance for Research Algorithms.

Implements comprehensive error recovery mechanisms for research algorithms
including circuit breakers, retry logic, and graceful degradation.

Author: ConfoRL Research Team
License: Apache 2.0
"""

import time
import threading
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import warnings

from ..utils.logging import get_logger
from ..utils.errors import ConfoRLError, ValidationError

logger = get_logger(__name__)


class RecoveryStrategy(Enum):
    """Error recovery strategies."""
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAKER = "circuit_breaker"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    FAIL_FAST = "fail_fast"


@dataclass
class ErrorContext:
    """Context information for errors."""
    error_type: str
    error_message: str
    timestamp: float
    function_name: str
    retry_count: int = 0
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class CircuitBreakerState:
    """Circuit breaker state."""
    is_open: bool = False
    failure_count: int = 0
    last_failure_time: float = 0.0
    success_count: int = 0
    total_calls: int = 0


class ErrorRecoveryManager:
    """Centralized error recovery and fault tolerance manager."""
    
    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_timeout: float = 60.0,
        enable_metrics: bool = True
    ):
        """Initialize error recovery manager.
        
        Args:
            max_retries: Maximum number of retry attempts
            retry_delay: Base delay between retries (exponential backoff)
            circuit_breaker_threshold: Failures before opening circuit
            circuit_breaker_timeout: Timeout before attempting to close circuit
            enable_metrics: Whether to collect error metrics
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.circuit_breaker_timeout = circuit_breaker_timeout
        self.enable_metrics = enable_metrics
        
        # Circuit breakers per function
        self._circuit_breakers: Dict[str, CircuitBreakerState] = defaultdict(CircuitBreakerState)
        
        # Error tracking
        self._error_history: deque = deque(maxlen=1000)
        self._error_counts: Dict[str, int] = defaultdict(int)
        self._recovery_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Fallback functions registry
        self._fallback_functions: Dict[str, Callable] = {}
        
        logger.info(f"Error recovery manager initialized with max_retries={max_retries}")
    
    def register_fallback(self, function_name: str, fallback_func: Callable) -> None:
        """Register fallback function for a specific function.
        
        Args:
            function_name: Name of function to provide fallback for
            fallback_func: Fallback function to call on failure
        """
        with self._lock:
            self._fallback_functions[function_name] = fallback_func
            logger.debug(f"Registered fallback for {function_name}")
    
    def with_recovery(
        self,
        strategy: RecoveryStrategy = RecoveryStrategy.RETRY,
        function_name: Optional[str] = None
    ):
        """Decorator for adding error recovery to functions.
        
        Args:
            strategy: Recovery strategy to use
            function_name: Name for tracking (uses actual function name if None)
        """
        def decorator(func: Callable):
            def wrapper(*args, **kwargs):
                fname = function_name or func.__name__
                return self._execute_with_recovery(func, fname, strategy, *args, **kwargs)
            return wrapper
        return decorator
    
    def _execute_with_recovery(
        self,
        func: Callable,
        function_name: str,
        strategy: RecoveryStrategy,
        *args,
        **kwargs
    ) -> Any:
        """Execute function with specified recovery strategy."""
        if strategy == RecoveryStrategy.CIRCUIT_BREAKER:
            return self._execute_with_circuit_breaker(func, function_name, *args, **kwargs)
        elif strategy == RecoveryStrategy.RETRY:
            return self._execute_with_retry(func, function_name, *args, **kwargs)
        elif strategy == RecoveryStrategy.FALLBACK:
            return self._execute_with_fallback(func, function_name, *args, **kwargs)
        elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
            return self._execute_with_degradation(func, function_name, *args, **kwargs)
        elif strategy == RecoveryStrategy.FAIL_FAST:
            return func(*args, **kwargs)
        else:
            raise ValidationError(f"Unknown recovery strategy: {strategy}")
    
    def _execute_with_circuit_breaker(
        self,
        func: Callable,
        function_name: str,
        *args,
        **kwargs
    ) -> Any:
        """Execute with circuit breaker pattern."""
        with self._lock:
            cb_state = self._circuit_breakers[function_name]
            cb_state.total_calls += 1
            
            # Check if circuit is open
            if cb_state.is_open:
                time_since_failure = time.time() - cb_state.last_failure_time
                if time_since_failure < self.circuit_breaker_timeout:
                    self._record_error(
                        function_name,
                        "CircuitBreakerOpen",
                        f"Circuit breaker open for {function_name}"
                    )
                    raise ConfoRLError(f"Circuit breaker open for {function_name}")
                else:
                    # Try to close circuit (half-open state)
                    logger.info(f"Attempting to close circuit breaker for {function_name}")
        
        try:
            result = func(*args, **kwargs)
            
            # Success - reset circuit breaker
            with self._lock:
                cb_state.success_count += 1
                if cb_state.is_open:
                    logger.info(f"Circuit breaker closed for {function_name}")
                cb_state.is_open = False
                cb_state.failure_count = 0
            
            return result
            
        except Exception as e:
            with self._lock:
                cb_state.failure_count += 1
                cb_state.last_failure_time = time.time()
                
                if cb_state.failure_count >= self.circuit_breaker_threshold:
                    cb_state.is_open = True
                    logger.warning(f"Circuit breaker opened for {function_name} after {cb_state.failure_count} failures")
            
            self._record_error(function_name, type(e).__name__, str(e))
            raise
    
    def _execute_with_retry(
        self,
        func: Callable,
        function_name: str,
        *args,
        **kwargs
    ) -> Any:
        """Execute with retry logic."""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                result = func(*args, **kwargs)
                
                if attempt > 0:
                    logger.info(f"Function {function_name} succeeded on attempt {attempt + 1}")
                    with self._lock:
                        self._recovery_stats[function_name]['retry_successes'] += 1
                
                return result
                
            except Exception as e:
                last_exception = e
                self._record_error(function_name, type(e).__name__, str(e), attempt)
                
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Function {function_name} failed on attempt {attempt + 1}, retrying in {delay}s: {e}")
                    time.sleep(delay)
                else:
                    logger.error(f"Function {function_name} failed after {self.max_retries + 1} attempts")
                    with self._lock:
                        self._recovery_stats[function_name]['retry_failures'] += 1
        
        # All retries exhausted
        raise ConfoRLError(f"Function {function_name} failed after {self.max_retries + 1} attempts: {last_exception}")
    
    def _execute_with_fallback(
        self,
        func: Callable,
        function_name: str,
        *args,
        **kwargs
    ) -> Any:
        """Execute with fallback function."""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self._record_error(function_name, type(e).__name__, str(e))
            
            # Try fallback function
            fallback_func = self._fallback_functions.get(function_name)
            if fallback_func:
                try:
                    logger.warning(f"Using fallback for {function_name}: {e}")
                    result = fallback_func(*args, **kwargs)
                    
                    with self._lock:
                        self._recovery_stats[function_name]['fallback_successes'] += 1
                    
                    return result
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed for {function_name}: {fallback_error}")
                    with self._lock:
                        self._recovery_stats[function_name]['fallback_failures'] += 1
                    raise ConfoRLError(f"Both primary and fallback failed for {function_name}: {e}, {fallback_error}")
            else:
                logger.error(f"No fallback registered for {function_name}")
                raise ConfoRLError(f"Function {function_name} failed and no fallback available: {e}")
    
    def _execute_with_degradation(
        self,
        func: Callable,
        function_name: str,
        *args,
        **kwargs
    ) -> Any:
        """Execute with graceful degradation."""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self._record_error(function_name, type(e).__name__, str(e))
            
            # Return degraded result based on function name
            logger.warning(f"Graceful degradation for {function_name}: {e}")
            
            with self._lock:
                self._recovery_stats[function_name]['degraded_responses'] += 1
            
            # Return safe default based on common function patterns
            if 'predict' in function_name.lower():
                return 0.0  # Conservative prediction
            elif 'certificate' in function_name.lower():
                return {'risk_bound': 1.0, 'confidence': 0.0}  # Worst-case certificate
            elif 'generate' in function_name.lower():
                return None  # No generation
            else:
                return None  # Generic safe default
    
    def _record_error(
        self,
        function_name: str,
        error_type: str,
        error_message: str,
        retry_count: int = 0
    ) -> None:
        """Record error for tracking and analysis."""
        if not self.enable_metrics:
            return
        
        error_context = ErrorContext(
            error_type=error_type,
            error_message=error_message,
            timestamp=time.time(),
            function_name=function_name,
            retry_count=retry_count
        )
        
        with self._lock:
            self._error_history.append(error_context)
            self._error_counts[f"{function_name}:{error_type}"] += 1
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        with self._lock:
            # Recent error rates
            recent_errors = [e for e in self._error_history if time.time() - e.timestamp < 3600]  # Last hour
            
            # Circuit breaker states
            cb_stats = {}
            for func_name, cb_state in self._circuit_breakers.items():
                cb_stats[func_name] = {
                    'is_open': cb_state.is_open,
                    'failure_count': cb_state.failure_count,
                    'success_count': cb_state.success_count,
                    'total_calls': cb_state.total_calls,
                    'success_rate': cb_state.success_count / max(1, cb_state.total_calls)
                }
            
            # Error type distribution
            error_types = defaultdict(int)
            for error in recent_errors:
                error_types[error.error_type] += 1
            
            # Recovery statistics
            recovery_summary = {}
            for func_name, stats in self._recovery_stats.items():
                recovery_summary[func_name] = dict(stats)
            
            return {
                'total_errors': len(self._error_history),
                'recent_errors_1h': len(recent_errors),
                'error_types': dict(error_types),
                'circuit_breakers': cb_stats,
                'recovery_stats': recovery_summary,
                'most_common_errors': sorted(
                    self._error_counts.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10],
                'fallback_functions_registered': len(self._fallback_functions)
            }
    
    def reset_circuit_breaker(self, function_name: str) -> None:
        """Manually reset circuit breaker for a function."""
        with self._lock:
            if function_name in self._circuit_breakers:
                cb_state = self._circuit_breakers[function_name]
                cb_state.is_open = False
                cb_state.failure_count = 0
                cb_state.success_count = 0
                logger.info(f"Circuit breaker reset for {function_name}")
            else:
                logger.warning(f"No circuit breaker found for {function_name}")
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on error recovery system."""
        with self._lock:
            recent_errors = [e for e in self._error_history if time.time() - e.timestamp < 300]  # Last 5 minutes
            
            # Check for high error rates
            error_rate_warnings = []
            if len(recent_errors) > 50:
                error_rate_warnings.append("High error rate detected in last 5 minutes")
            
            # Check circuit breaker states
            open_circuits = [name for name, state in self._circuit_breakers.items() if state.is_open]
            
            # Overall health status
            health_status = "healthy"
            if open_circuits:
                health_status = "degraded"
            if len(recent_errors) > 100:
                health_status = "unhealthy"
            
            return {
                'status': health_status,
                'recent_error_count': len(recent_errors),
                'open_circuit_breakers': open_circuits,
                'warnings': error_rate_warnings,
                'total_functions_monitored': len(self._circuit_breakers),
                'fallback_coverage': len(self._fallback_functions) / max(1, len(self._circuit_breakers))
            }


# Global error recovery manager instance
_global_recovery_manager = None

def get_recovery_manager() -> ErrorRecoveryManager:
    """Get global error recovery manager instance."""
    global _global_recovery_manager
    if _global_recovery_manager is None:
        _global_recovery_manager = ErrorRecoveryManager()
    return _global_recovery_manager


# Convenience decorators
def with_retry(max_retries: int = 3, retry_delay: float = 1.0):
    """Decorator for adding retry logic to functions."""
    def decorator(func):
        manager = get_recovery_manager()
        manager.max_retries = max_retries
        manager.retry_delay = retry_delay
        return manager.with_recovery(RecoveryStrategy.RETRY)(func)
    return decorator


def with_circuit_breaker(threshold: int = 5, timeout: float = 60.0):
    """Decorator for adding circuit breaker to functions."""
    def decorator(func):
        manager = get_recovery_manager()
        manager.circuit_breaker_threshold = threshold
        manager.circuit_breaker_timeout = timeout
        return manager.with_recovery(RecoveryStrategy.CIRCUIT_BREAKER)(func)
    return decorator


def with_fallback(fallback_func: Callable):
    """Decorator for adding fallback function."""
    def decorator(func):
        manager = get_recovery_manager()
        manager.register_fallback(func.__name__, fallback_func)
        return manager.with_recovery(RecoveryStrategy.FALLBACK)(func)
    return decorator


# Health monitoring for research algorithms
class ResearchAlgorithmMonitor:
    """Monitor health and performance of research algorithms."""
    
    def __init__(self, algorithm_name: str):
        self.algorithm_name = algorithm_name
        self.recovery_manager = get_recovery_manager()
        self.start_time = time.time()
        self.iteration_count = 0
        self.performance_metrics = defaultdict(list)
        
        logger.info(f"Research algorithm monitor initialized for {algorithm_name}")
    
    def record_iteration(self, metrics: Dict[str, float]) -> None:
        """Record performance metrics for an iteration."""
        self.iteration_count += 1
        
        for metric_name, value in metrics.items():
            self.performance_metrics[metric_name].append(value)
            
            # Keep only recent metrics
            if len(self.performance_metrics[metric_name]) > 1000:
                self.performance_metrics[metric_name] = self.performance_metrics[metric_name][-1000:]
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive health summary for the algorithm."""
        runtime = time.time() - self.start_time
        
        # Performance trends
        performance_summary = {}
        for metric_name, values in self.performance_metrics.items():
            if values:
                recent_values = values[-100:]  # Last 100 iterations
                performance_summary[metric_name] = {
                    'current': values[-1],
                    'recent_mean': sum(recent_values) / len(recent_values),
                    'recent_trend': 'improving' if len(values) > 10 and values[-1] > values[-10] else 'stable'
                }
        
        # Error recovery status
        recovery_stats = self.recovery_manager.get_error_statistics()
        health_check = self.recovery_manager.health_check()
        
        return {
            'algorithm_name': self.algorithm_name,
            'runtime_seconds': runtime,
            'iterations_completed': self.iteration_count,
            'iterations_per_second': self.iteration_count / max(1, runtime),
            'performance_metrics': performance_summary,
            'error_recovery_status': health_check,
            'total_errors': recovery_stats['total_errors'],
            'recent_errors': recovery_stats['recent_errors_1h']
        }