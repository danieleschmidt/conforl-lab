"""Utility functions and error handling for ConfoRL."""

# Always available
from .errors import ConfoRLError, ConfigurationError, ValidationError, DeploymentError
from .logging import setup_logging, get_logger

# Conditional imports for modules with dependencies
try:
    from .validation import validate_config, validate_environment, validate_dataset
except ImportError:
    pass

try:
    from .security import sanitize_input, validate_file_path, check_permissions
except ImportError:
    pass

__all__ = [
    "validate_config",
    "validate_environment", 
    "validate_dataset",
    "ConfoRLError",
    "ConfigurationError",
    "ValidationError",
    "DeploymentError",
    "setup_logging",
    "get_logger",
    "sanitize_input",
    "validate_file_path",
    "check_permissions",
]