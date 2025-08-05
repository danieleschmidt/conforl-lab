"""Security utilities for input sanitization and validation."""

import re
import os
import stat
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import hashlib

from .errors import SecurityError
from .logging import get_logger

logger = get_logger(__name__)


def sanitize_input(
    input_value: Any,
    input_type: str = "string",
    max_length: Optional[int] = None,
    allowed_chars: Optional[str] = None
) -> Any:
    """Sanitize user input to prevent injection attacks.
    
    Args:
        input_value: Input value to sanitize
        input_type: Expected input type ('string', 'number', 'path')
        max_length: Maximum allowed length for strings
        allowed_chars: Regex pattern of allowed characters
        
    Returns:
        Sanitized input value
        
    Raises:
        SecurityError: If input fails security checks
    """
    if input_value is None:
        return None
    
    if input_type == "string":
        if not isinstance(input_value, str):
            raise SecurityError(f"Expected string, got {type(input_value)}")
        
        # Check length
        if max_length and len(input_value) > max_length:
            raise SecurityError(f"Input length {len(input_value)} exceeds maximum {max_length}")
        
        # Check for dangerous patterns
        dangerous_patterns = [
            r'<script.*?>.*?</script>',  # Script tags
            r'javascript:',              # JavaScript URLs
            r'on\w+\s*=',               # Event handlers
            r'(\.\./)+',                # Path traversal
            r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]',  # Control characters
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, input_value, re.IGNORECASE):
                raise SecurityError(f"Dangerous pattern detected: {pattern}")
        
        # Check allowed characters
        if allowed_chars and not re.match(f"^[{allowed_chars}]*$", input_value):
            raise SecurityError(f"Input contains disallowed characters")
        
        return input_value.strip()
    
    elif input_type == "number":
        try:
            if isinstance(input_value, str):
                # Only allow numeric characters, decimal point, and minus sign
                if not re.match(r'^-?\d*\.?\d*$', input_value.strip()):
                    raise SecurityError("Invalid numeric format")
                return float(input_value) if '.' in input_value else int(input_value)
            return input_value
        except (ValueError, TypeError):
            raise SecurityError(f"Cannot convert to number: {input_value}")
    
    elif input_type == "path":
        return sanitize_file_path(str(input_value))
    
    else:
        raise SecurityError(f"Unknown input type: {input_type}")


def sanitize_file_path(file_path: Union[str, Path]) -> Path:
    """Sanitize file path to prevent directory traversal attacks.
    
    Args:
        file_path: File path to sanitize
        
    Returns:
        Sanitized Path object
        
    Raises:
        SecurityError: If path is potentially dangerous
    """
    if not file_path:
        raise SecurityError("Empty file path")
    
    path_obj = Path(file_path)
    
    # Check for path traversal attempts
    path_str = str(path_obj)
    if '..' in path_str:
        raise SecurityError(f"Path traversal detected: {path_str}")
    
    # Check for absolute paths with dangerous locations
    if path_obj.is_absolute():
        dangerous_paths = ['/etc', '/usr', '/bin', '/sbin', '/root', '/home']
        path_str_lower = path_str.lower()
        for dangerous in dangerous_paths:
            if path_str_lower.startswith(dangerous.lower()):
                logger.warning(f"Access to potentially dangerous path: {path_str}")
    
    # Resolve to absolute path and check again
    try:
        resolved_path = path_obj.resolve()
        
        # Ensure resolved path doesn't escape intended directory
        # This is environment-specific and may need adjustment
        cwd = Path.cwd()
        try:
            resolved_path.relative_to(cwd)
        except ValueError:
            logger.warning(f"Path resolves outside current directory: {resolved_path}")
    
    except (OSError, RuntimeError) as e:
        raise SecurityError(f"Cannot resolve path {path_obj}: {str(e)}")
    
    return resolved_path


def validate_file_path(
    file_path: Union[str, Path],
    must_exist: bool = False,
    check_readable: bool = False,
    check_writable: bool = False
) -> Path:
    """Validate file path with security checks.
    
    Args:
        file_path: File path to validate
        must_exist: Whether file must exist
        check_readable: Whether to check read permissions
        check_writable: Whether to check write permissions
        
    Returns:
        Validated Path object
        
    Raises:
        SecurityError: If validation fails
    """
    sanitized_path = sanitize_file_path(file_path)
    
    if must_exist and not sanitized_path.exists():
        raise SecurityError(f"File does not exist: {sanitized_path}")
    
    if sanitized_path.exists():
        if check_readable and not os.access(sanitized_path, os.R_OK):
            raise SecurityError(f"File not readable: {sanitized_path}")
        
        if check_writable and not os.access(sanitized_path, os.W_OK):
            raise SecurityError(f"File not writable: {sanitized_path}")
    
    return sanitized_path


def check_permissions(file_path: Union[str, Path]) -> Dict[str, bool]:
    """Check file permissions safely.
    
    Args:
        file_path: File path to check
        
    Returns:
        Dictionary with permission information
    """
    path_obj = Path(file_path)
    
    permissions = {
        'exists': False,
        'readable': False,
        'writable': False,
        'executable': False,
        'is_file': False,
        'is_directory': False,
    }
    
    try:
        if path_obj.exists():
            permissions['exists'] = True
            permissions['readable'] = os.access(path_obj, os.R_OK)
            permissions['writable'] = os.access(path_obj, os.W_OK)
            permissions['executable'] = os.access(path_obj, os.X_OK)
            permissions['is_file'] = path_obj.is_file()
            permissions['is_directory'] = path_obj.is_dir()
    except (OSError, PermissionError):
        pass  # Keep default False values
    
    return permissions


def hash_sensitive_data(data: str, salt: Optional[str] = None) -> str:
    """Hash sensitive data securely.
    
    Args:
        data: Data to hash
        salt: Optional salt (generated if not provided)
        
    Returns:
        Hexadecimal hash string
    """
    if salt is None:
        salt = os.urandom(32).hex()
    
    # Use SHA-256 with salt
    hash_obj = hashlib.sha256()
    hash_obj.update(salt.encode('utf-8'))
    hash_obj.update(data.encode('utf-8'))
    
    return salt + hash_obj.hexdigest()


def verify_hash(data: str, hashed_data: str) -> bool:
    """Verify data against hash.
    
    Args:
        data: Original data
        hashed_data: Hash to verify against
        
    Returns:
        True if data matches hash
    """
    if len(hashed_data) < 64:  # 32 bytes salt = 64 hex chars
        return False
    
    salt = hashed_data[:64]  # First 64 chars are salt
    expected_hash = hashed_data[64:]  # Rest is hash
    
    # Compute hash with extracted salt
    hash_obj = hashlib.sha256()
    hash_obj.update(salt.encode('utf-8'))
    hash_obj.update(data.encode('utf-8'))
    
    return hash_obj.hexdigest() == expected_hash


def sanitize_config_dict(config: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize configuration dictionary.
    
    Args:
        config: Configuration dictionary to sanitize
        
    Returns:
        Sanitized configuration dictionary
    """
    sanitized = {}
    
    for key, value in config.items():
        # Sanitize key
        try:
            clean_key = sanitize_input(key, "string", max_length=100, allowed_chars=r'\w\-\.')
        except SecurityError:
            logger.warning(f"Skipping config key with security issues: {key}")
            continue
        
        # Sanitize value based on type
        try:
            if isinstance(value, str):
                clean_value = sanitize_input(value, "string", max_length=1000)
            elif isinstance(value, (int, float)):
                clean_value = sanitize_input(value, "number")
            elif isinstance(value, dict):
                clean_value = sanitize_config_dict(value)  # Recursive
            elif isinstance(value, list):
                clean_value = [
                    sanitize_input(item, "string", max_length=1000) 
                    if isinstance(item, str) else item
                    for item in value
                ]
            else:
                clean_value = value  # Keep as-is for other types
            
            sanitized[clean_key] = clean_value
            
        except SecurityError as e:
            logger.warning(f"Skipping config value with security issues: {key}={value} - {e}")
    
    return sanitized


def log_security_event(event_type: str, details: Dict[str, Any]):
    """Log security-related events.
    
    Args:
        event_type: Type of security event
        details: Event details dictionary
    """
    # Sanitize event details before logging
    safe_details = {}
    for key, value in details.items():
        if isinstance(value, str):
            # Truncate long strings and remove sensitive patterns
            safe_value = value[:200]  # Limit length
            safe_value = re.sub(r'password=\w+', 'password=***', safe_value, flags=re.IGNORECASE)
            safe_value = re.sub(r'token=[\w\-]+', 'token=***', safe_value, flags=re.IGNORECASE)
            safe_details[key] = safe_value
        else:
            safe_details[key] = str(value)[:100]  # Convert to string and limit
    
    logger.warning(f"Security Event [{event_type}]: {safe_details}")


class SecurityContext:
    """Context manager for security-sensitive operations."""
    
    def __init__(self, operation: str, user: str = "system"):
        """Initialize security context.
        
        Args:
            operation: Description of operation
            user: User performing operation
        """
        self.operation = operation
        self.user = user
        self.start_time = None
    
    def __enter__(self):
        """Enter security context."""
        import time
        self.start_time = time.time()
        logger.info(f"Starting secure operation: {self.operation} (user: {self.user})")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit security context."""
        import time
        duration = time.time() - self.start_time
        
        if exc_type is not None:
            logger.error(f"Secure operation failed: {self.operation} - {exc_val}")
            log_security_event("OPERATION_FAILED", {
                "operation": self.operation,
                "user": self.user,
                "error": str(exc_val),
                "duration": duration
            })
        else:
            logger.info(f"Secure operation completed: {self.operation} ({duration:.2f}s)")
        
        return False  # Don't suppress exceptions