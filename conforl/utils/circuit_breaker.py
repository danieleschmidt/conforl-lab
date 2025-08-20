"""
Circuit breaker pattern for fault tolerance and system resilience.
Generation 2: Robust error handling and automatic recovery.
"""

import time
import threading
from typing import Any, Callable, Optional, Dict
from enum import Enum
from dataclasses import dataclass

from .errors import CircuitBreakerError, ServiceUnavailableError


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Blocking requests due to failures
    HALF_OPEN = "half_open" # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5         # Number of failures before opening
    timeout_seconds: int = 60          # How long to wait before trying again
    expected_exception: type = Exception # Exception type that counts as failure
    recovery_timeout: int = 30         # Time to stay in half-open state
    success_threshold: int = 3         # Successful calls needed to close circuit


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance."""
    
    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        """Initialize circuit breaker.
        
        Args:
            config: Circuit breaker configuration
        """
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.last_success_time = None
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Statistics
        self.total_calls = 0
        self.total_failures = 0
        self.total_successes = 0
        self.state_changes = []
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerError: If circuit is open
            Exception: Original function exceptions
        """
        with self._lock:
            self.total_calls += 1
            
            # Check circuit state
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to_half_open()
                else:
                    raise CircuitBreakerError(
                        f"Circuit breaker is OPEN. Last failure: {self.last_failure_time}"
                    )
            
            elif self.state == CircuitState.HALF_OPEN:
                if time.time() - self.last_failure_time > self.config.recovery_timeout:
                    # Reset to closed if we've been in half-open too long
                    self._transition_to_closed()
        
        # Execute the function
        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
            
        except Exception as e:
            if isinstance(e, self.config.expected_exception):
                self._record_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit."""
        if self.last_failure_time is None:
            return True
        
        return (time.time() - self.last_failure_time) >= self.config.timeout_seconds
    
    def _record_success(self):
        """Record a successful call."""
        with self._lock:
            self.success_count += 1
            self.total_successes += 1
            self.last_success_time = time.time()
            
            if self.state == CircuitState.HALF_OPEN:
                if self.success_count >= self.config.success_threshold:
                    self._transition_to_closed()
    
    def _record_failure(self):
        """Record a failed call."""
        with self._lock:
            self.failure_count += 1
            self.total_failures += 1
            self.last_failure_time = time.time()
            
            if self.state == CircuitState.CLOSED:
                if self.failure_count >= self.config.failure_threshold:
                    self._transition_to_open()
            
            elif self.state == CircuitState.HALF_OPEN:
                # Any failure in half-open goes back to open
                self._transition_to_open()
    
    def _transition_to_open(self):
        """Transition circuit to OPEN state."""
        self.state = CircuitState.OPEN
        self.success_count = 0
        self._record_state_change("OPEN")
    
    def _transition_to_half_open(self):
        """Transition circuit to HALF_OPEN state."""
        self.state = CircuitState.HALF_OPEN
        self.failure_count = 0
        self.success_count = 0
        self._record_state_change("HALF_OPEN")
    
    def _transition_to_closed(self):
        """Transition circuit to CLOSED state."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self._record_state_change("CLOSED")
    
    def _record_state_change(self, new_state: str):
        """Record state change for monitoring."""
        self.state_changes.append({
            'timestamp': time.time(),
            'new_state': new_state,
            'failure_count': self.failure_count,
            'success_count': self.success_count
        })
        
        # Keep only recent state changes
        if len(self.state_changes) > 100:
            self.state_changes = self.state_changes[-50:]
    
    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status.
        
        Returns:
            Status dictionary
        """
        with self._lock:
            return {
                'state': self.state.value,
                'failure_count': self.failure_count,
                'success_count': self.success_count,
                'total_calls': self.total_calls,
                'total_failures': self.total_failures,
                'total_successes': self.total_successes,
                'failure_rate': self.total_failures / self.total_calls if self.total_calls > 0 else 0,
                'last_failure_time': self.last_failure_time,
                'last_success_time': self.last_success_time,
                'recent_state_changes': self.state_changes[-5:],
                'config': {
                    'failure_threshold': self.config.failure_threshold,
                    'timeout_seconds': self.config.timeout_seconds,
                    'recovery_timeout': self.config.recovery_timeout,
                    'success_threshold': self.config.success_threshold
                }
            }
    
    def reset(self):
        """Manually reset circuit breaker to CLOSED state."""
        with self._lock:
            self._transition_to_closed()
    
    def force_open(self):
        """Manually force circuit breaker to OPEN state."""
        with self._lock:
            self._transition_to_open()


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""
    
    def __init__(self):
        """Initialize circuit breaker registry."""
        self._breakers = {}
        self._lock = threading.Lock()
    
    def get_breaker(
        self, 
        name: str, 
        config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """Get or create a circuit breaker.
        
        Args:
            name: Breaker name
            config: Configuration (only used for new breakers)
            
        Returns:
            Circuit breaker instance
        """
        with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(config)
            return self._breakers[name]
    
    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all circuit breakers.
        
        Returns:
            Dictionary of breaker statuses
        """
        with self._lock:
            return {
                name: breaker.get_status()
                for name, breaker in self._breakers.items()
            }
    
    def reset_all(self):
        """Reset all circuit breakers."""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()


# Global circuit breaker registry
_breaker_registry = CircuitBreakerRegistry()

def circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    timeout_seconds: int = 60,
    expected_exception: type = Exception
):
    """Decorator for automatic circuit breaker protection.
    
    Args:
        name: Circuit breaker name
        failure_threshold: Failures before opening circuit
        timeout_seconds: Time to wait before retry
        expected_exception: Exception type that triggers circuit
    """
    def decorator(func):
        config = CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            timeout_seconds=timeout_seconds,
            expected_exception=expected_exception
        )
        breaker = _breaker_registry.get_breaker(name, config)
        
        def wrapper(*args, **kwargs):
            return breaker.call(func, *args, **kwargs)
        
        wrapper.circuit_breaker = breaker
        return wrapper
    
    return decorator


def get_circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
    """Get a circuit breaker instance.
    
    Args:
        name: Breaker name
        config: Configuration for new breakers
        
    Returns:
        Circuit breaker instance
    """
    return _breaker_registry.get_breaker(name, config)


def get_all_circuit_breakers_status() -> Dict[str, Dict[str, Any]]:
    """Get status of all circuit breakers.
    
    Returns:
        Dictionary of all breaker statuses
    """
    return _breaker_registry.get_all_status()


class RetryPolicy:
    """Retry policy for resilient operations."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_backoff: bool = True,
        jitter: bool = True
    ):
        """Initialize retry policy.
        
        Args:
            max_attempts: Maximum retry attempts
            base_delay: Base delay between retries
            max_delay: Maximum delay between retries
            exponential_backoff: Use exponential backoff
            jitter: Add random jitter to delay
        """
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_backoff = exponential_backoff
        self.jitter = jitter
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry policy.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: Last exception if all retries failed
        """
        last_exception = None
        
        for attempt in range(self.max_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt == self.max_attempts - 1:
                    # Last attempt, re-raise
                    raise
                
                # Calculate delay
                delay = self.base_delay
                if self.exponential_backoff:
                    delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                
                if self.jitter:
                    import random
                    delay *= (0.5 + random.random() * 0.5)  # 50-100% of calculated delay
                
                time.sleep(delay)
        
        # Should not reach here, but just in case
        raise last_exception


def with_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    exponential_backoff: bool = True
):
    """Decorator for automatic retry with exponential backoff.
    
    Args:
        max_attempts: Maximum retry attempts
        base_delay: Base delay between retries
        exponential_backoff: Use exponential backoff
    """
    def decorator(func):
        policy = RetryPolicy(max_attempts, base_delay, exponential_backoff=exponential_backoff)
        
        def wrapper(*args, **kwargs):
            return policy.execute(func, *args, **kwargs)
        
        return wrapper
    
    return decorator