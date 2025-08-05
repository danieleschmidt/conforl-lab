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