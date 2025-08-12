"""Custom exception classes for ConfoRL."""


class ConfoRLError(Exception):
    """Base exception class for ConfoRL."""
    
    def __init__(self, message: str, error_code: str = None):
        """Initialize ConfoRL error.
        
        Args:
            message: Error message
            error_code: Optional error code for programmatic handling
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "GENERAL_ERROR"
    
    def __str__(self) -> str:
        return f"[{self.error_code}] {self.message}"


class ConfigurationError(ConfoRLError):
    """Exception raised for configuration-related errors."""
    
    def __init__(self, message: str, config_key: str = None):
        """Initialize configuration error.
        
        Args:
            message: Error message
            config_key: Configuration key that caused the error
        """
        super().__init__(message, "CONFIG_ERROR")
        self.config_key = config_key


class ValidationError(ConfoRLError):
    """Exception raised for data validation errors."""
    
    def __init__(self, message: str, validation_type: str = None):
        """Initialize validation error.
        
        Args:
            message: Error message
            validation_type: Type of validation that failed
        """
        super().__init__(message, "VALIDATION_ERROR")
        self.validation_type = validation_type


class RiskControlError(ConfoRLError):
    """Exception raised for risk control failures."""
    
    def __init__(self, message: str, risk_level: float = None):
        """Initialize risk control error.
        
        Args:
            message: Error message
            risk_level: Current risk level when error occurred
        """
        super().__init__(message, "RISK_CONTROL_ERROR")
        self.risk_level = risk_level


class ConformalPredictionError(ConfoRLError):
    """Exception raised for conformal prediction errors."""
    
    def __init__(self, message: str, coverage: float = None):
        """Initialize conformal prediction error.
        
        Args:
            message: Error message
            coverage: Target coverage when error occurred
        """
        super().__init__(message, "CONFORMAL_ERROR")
        self.coverage = coverage


class DeploymentError(ConfoRLError):
    """Exception raised for deployment-related errors."""
    
    def __init__(self, message: str, deployment_stage: str = None):
        """Initialize deployment error.
        
        Args:
            message: Error message
            deployment_stage: Stage of deployment when error occurred
        """
        super().__init__(message, "DEPLOYMENT_ERROR")
        self.deployment_stage = deployment_stage


class EnvironmentError(ConfoRLError):
    """Exception raised for environment-related errors."""
    
    def __init__(self, message: str, env_name: str = None):
        """Initialize environment error.
        
        Args:
            message: Error message
            env_name: Name of environment that caused error
        """
        super().__init__(message, "ENVIRONMENT_ERROR")
        self.env_name = env_name


class AlgorithmError(ConfoRLError):
    """Exception raised for RL algorithm errors."""
    
    def __init__(self, message: str, algorithm_name: str = None):
        """Initialize algorithm error.
        
        Args:
            message: Error message
            algorithm_name: Name of algorithm that failed
        """
        super().__init__(message, "ALGORITHM_ERROR")
        self.algorithm_name = algorithm_name


class DataError(ConfoRLError):
    """Exception raised for data-related errors."""
    
    def __init__(self, message: str, data_source: str = None):
        """Initialize data error.
        
        Args:
            message: Error message
            data_source: Source of data that caused error
        """
        super().__init__(message, "DATA_ERROR")
        self.data_source = data_source


class SecurityError(ConfoRLError):
    """Exception raised for security-related errors."""
    
    def __init__(self, message: str, security_context: str = None):
        """Initialize security error.
        
        Args:
            message: Error message
            security_context: Security context where error occurred
        """
        super().__init__(message, "SECURITY_ERROR")
        self.security_context = security_context


class TrainingError(ConfoRLError):
    """Exception raised for training-related errors."""
    
    def __init__(self, message: str, training_step: int = None):
        """Initialize training error.
        
        Args:
            message: Error message
            training_step: Training step when error occurred
        """
        super().__init__(message, "TRAINING_ERROR")
        self.training_step = training_step


class CircuitBreakerError(ConfoRLError):
    """Exception raised when circuit breaker is open."""
    
    def __init__(self, message: str, failure_count: int = None):
        """Initialize circuit breaker error.
        
        Args:
            message: Error message
            failure_count: Number of failures that triggered circuit breaker
        """
        super().__init__(message, "CIRCUIT_BREAKER_ERROR")
        self.failure_count = failure_count


class ErrorRecovery:
    """Utility class for error recovery and resilience patterns."""
    
    def __init__(self, max_retries: int = 3, backoff_factor: float = 1.5):
        """Initialize error recovery.
        
        Args:
            max_retries: Maximum number of retry attempts
            backoff_factor: Exponential backoff multiplier
        """
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.retry_count = 0
        
    def retry_with_backoff(self, func, *args, **kwargs):
        """Retry function with exponential backoff.
        
        Args:
            func: Function to retry
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function
            
        Returns:
            Function result if successful
            
        Raises:
            ConfoRLError: If all retries exhausted
        """
        import time
        
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == self.max_retries:
                    raise ConfoRLError(
                        f"Function failed after {self.max_retries} retries: {str(e)}",
                        error_code="RETRY_EXHAUSTED"
                    ) from e
                
                # Exponential backoff
                wait_time = (self.backoff_factor ** attempt)
                time.sleep(wait_time)
                self.retry_count += 1
    
    def reset(self):
        """Reset retry counter."""
        self.retry_count = 0


class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        """Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time to wait before attempting recovery
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
    def call(self, func, *args, **kwargs):
        """Call function through circuit breaker.
        
        Args:
            func: Function to call
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result if successful
            
        Raises:
            CircuitBreakerError: If circuit is open
        """
        import time
        
        if self.state == "OPEN":
            if time.time() - self.last_failure_time < self.recovery_timeout:
                raise CircuitBreakerError(
                    f"Circuit breaker is OPEN. {self.failure_count} failures recorded.",
                    failure_count=self.failure_count
                )
            else:
                self.state = "HALF_OPEN"
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def _on_failure(self):
        """Handle failed call."""
        import time
        
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
    
    def reset(self):
        """Reset circuit breaker to closed state."""
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"