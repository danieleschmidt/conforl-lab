"""Enhanced Circuit Breaker Pattern for Robust Error Handling."""

import time
import threading
from enum import Enum
from typing import Callable, Any, Optional, Dict, List
from collections import deque
import traceback

from .logging import get_logger
from .errors import ConfoRLError, CircuitBreakerError

logger = get_logger(__name__)

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"         # Failing, rejecting calls
    HALF_OPEN = "half_open"  # Testing if service recovered

class CircuitBreakerEnhanced:
    """Enhanced circuit breaker for robust error handling."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception,
        name: str = "circuit_breaker"
    ):
        """Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures to trigger open state
            recovery_timeout: Seconds to wait before trying half-open
            expected_exception: Exception type to count as failure
            name: Identifier for this circuit breaker
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.name = name
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        self.success_count = 0
        
        # Enhanced monitoring
        self.call_history = deque(maxlen=1000)  # Track recent calls
        self.error_types = {}  # Count different error types
        self.performance_metrics = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'circuit_open_calls': 0,
            'average_response_time': 0.0
        }
        
        self._lock = threading.RLock()
        
    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap functions with circuit breaker."""
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        wrapper.__name__ = f"circuit_breaker_{func.__name__}"
        return wrapper
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Call function through circuit breaker."""
        with self._lock:
            self.performance_metrics['total_calls'] += 1
            
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    logger.info(f"Circuit breaker '{self.name}' moving to HALF_OPEN")
                else:
                    self.performance_metrics['circuit_open_calls'] += 1
                    error_msg = f"Circuit breaker '{self.name}' is OPEN"
                    self._record_call(False, error_msg, 0.0)
                    raise CircuitBreakerError(error_msg)
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                response_time = time.time() - start_time
                self._on_success(response_time)
                return result
                
            except self.expected_exception as e:
                response_time = time.time() - start_time
                self._on_failure(str(e), type(e).__name__, response_time)
                raise
            except Exception as e:
                # Unexpected exception - still count as failure but log separately
                response_time = time.time() - start_time
                logger.error(f"Unexpected exception in circuit breaker '{self.name}': {e}")
                self._on_failure(str(e), type(e).__name__, response_time)
                raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        return (
            self.last_failure_time is not None and
            time.time() - self.last_failure_time >= self.recovery_timeout
        )
    
    def _on_success(self, response_time: float) -> None:
        """Handle successful call."""
        self.performance_metrics['successful_calls'] += 1
        self._update_average_response_time(response_time)
        self._record_call(True, "success", response_time)
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            # Require at least 3 successes to close circuit
            if self.success_count >= 3:
                self._reset()
        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            self.failure_count = max(0, self.failure_count - 1)
    
    def _on_failure(self, error_msg: str, error_type: str, response_time: float) -> None:
        """Handle failed call."""
        self.performance_metrics['failed_calls'] += 1
        self._update_average_response_time(response_time)
        self._record_call(False, error_msg, response_time)
        
        # Track error types
        self.error_types[error_type] = self.error_types.get(error_type, 0) + 1
        
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit breaker '{self.name}' failed in HALF_OPEN, returning to OPEN")
        elif self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit breaker '{self.name}' opened after {self.failure_count} failures")
    
    def _reset(self) -> None:
        """Reset circuit breaker to closed state."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        logger.info(f"Circuit breaker '{self.name}' reset to CLOSED")
    
    def _record_call(self, success: bool, message: str, response_time: float) -> None:
        """Record call in history."""
        self.call_history.append({
            'timestamp': time.time(),
            'success': success,
            'message': message,
            'response_time': response_time,
            'state': self.state.value
        })
    
    def _update_average_response_time(self, response_time: float) -> None:
        """Update running average of response time."""
        total_calls = self.performance_metrics['total_calls']
        current_avg = self.performance_metrics['average_response_time']
        
        # Rolling average
        self.performance_metrics['average_response_time'] = (
            (current_avg * (total_calls - 1) + response_time) / total_calls
        )
    
    def get_state(self) -> CircuitState:
        """Get current circuit breaker state."""
        return self.state
    
    def get_failure_count(self) -> int:
        """Get current failure count."""
        return self.failure_count
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics."""
        with self._lock:
            recent_calls = list(self.call_history)[-100:]  # Last 100 calls
            recent_success_rate = (
                sum(1 for call in recent_calls if call['success']) / len(recent_calls)
                if recent_calls else 0.0
            )
            
            return {
                'name': self.name,
                'state': self.state.value,
                'failure_count': self.failure_count,
                'failure_threshold': self.failure_threshold,
                'last_failure_time': self.last_failure_time,
                'recovery_timeout': self.recovery_timeout,
                'performance_metrics': self.performance_metrics.copy(),
                'recent_success_rate': recent_success_rate,
                'error_types': self.error_types.copy(),
                'total_recorded_calls': len(self.call_history)
            }
    
    def force_open(self) -> None:
        """Force circuit breaker to open state (for testing/manual control)."""
        with self._lock:
            self.state = CircuitState.OPEN
            self.last_failure_time = time.time()
            logger.warning(f"Circuit breaker '{self.name}' manually forced OPEN")
    
    def force_close(self) -> None:
        """Force circuit breaker to close state (for testing/manual control)."""
        with self._lock:
            self._reset()
            logger.info(f"Circuit breaker '{self.name}' manually forced CLOSED")

class CircuitBreakerManager:
    """Manager for multiple circuit breakers."""
    
    def __init__(self):
        self.circuit_breakers = {}
        self._lock = threading.RLock()
    
    def create_circuit_breaker(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception
    ) -> CircuitBreakerEnhanced:
        """Create and register a new circuit breaker."""
        with self._lock:
            if name in self.circuit_breakers:
                return self.circuit_breakers[name]
            
            cb = CircuitBreakerEnhanced(
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                expected_exception=expected_exception,
                name=name
            )
            self.circuit_breakers[name] = cb
            return cb
    
    def get_circuit_breaker(self, name: str) -> Optional[CircuitBreakerEnhanced]:
        """Get circuit breaker by name."""
        return self.circuit_breakers.get(name)
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all circuit breakers."""
        with self._lock:
            return {
                name: cb.get_metrics() 
                for name, cb in self.circuit_breakers.items()
            }
    
    def health_check(self) -> Dict[str, Any]:
        """Get overall health status."""
        with self._lock:
            total_breakers = len(self.circuit_breakers)
            open_breakers = sum(
                1 for cb in self.circuit_breakers.values() 
                if cb.get_state() == CircuitState.OPEN
            )
            
            return {
                'total_circuit_breakers': total_breakers,
                'open_circuit_breakers': open_breakers,
                'healthy': open_breakers == 0,
                'health_score': (total_breakers - open_breakers) / max(1, total_breakers)
            }

# Global circuit breaker manager
circuit_manager = CircuitBreakerManager()

# Convenience functions
def circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    expected_exception: type = Exception
) -> CircuitBreakerEnhanced:
    """Get or create a circuit breaker."""
    return circuit_manager.create_circuit_breaker(
        name, failure_threshold, recovery_timeout, expected_exception
    )

def get_circuit_health() -> Dict[str, Any]:
    """Get overall circuit breaker health."""
    return circuit_manager.health_check()