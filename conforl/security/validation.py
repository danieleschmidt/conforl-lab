"""Security validation and input sanitization.

Comprehensive input validation and sanitization to prevent
injection attacks and ensure data integrity in safe RL systems.
"""

import re
import json
import pickle
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
import hashlib
import logging
from dataclasses import dataclass

from ..utils.logging import get_logger
from ..utils.errors import ValidationError, SecurityError

logger = get_logger(__name__)


@dataclass
class ValidationRule:
    """Security validation rule."""
    
    field_name: str
    rule_type: str  # 'type', 'range', 'regex', 'whitelist', 'custom'
    rule_value: Any
    required: bool = True
    description: Optional[str] = None


class SecurityValidator:
    """Comprehensive security validator for ConfoRL inputs."""
    
    def __init__(self):
        """Initialize security validator."""
        self.validation_rules = {}
        self.validation_history = []
        self.failed_validations = []
        
        # Default security rules
        self._setup_default_rules()
        
        logger.info("Security validator initialized")
    
    def _setup_default_rules(self):
        """Setup default security validation rules."""
        # Risk parameters
        self.add_rule(ValidationRule(
            'target_risk', 'range', (0.0, 1.0),
            description='Risk must be between 0 and 1'
        ))
        
        self.add_rule(ValidationRule(
            'confidence', 'range', (0.0, 1.0),
            description='Confidence must be between 0 and 1'
        ))
        
        # File paths - prevent directory traversal
        self.add_rule(ValidationRule(
            'file_path', 'custom', self._validate_file_path,
            description='File path must be safe (no directory traversal)'
        ))
        
        # Model parameters
        self.add_rule(ValidationRule(
            'learning_rate', 'range', (1e-6, 1.0),
            description='Learning rate must be reasonable'
        ))
        
        # Array dimensions
        self.add_rule(ValidationRule(
            'array_size', 'range', (1, 1000000),
            description='Array size must be reasonable'
        ))
        
        # String inputs - prevent injection
        self.add_rule(ValidationRule(
            'algorithm_name', 'regex', r'^[a-zA-Z0-9_-]+$',
            description='Algorithm name must be alphanumeric with _ or -'
        ))
    
    def add_rule(self, rule: ValidationRule) -> None:
        """Add validation rule."""
        self.validation_rules[rule.field_name] = rule
        logger.debug(f"Added validation rule for {rule.field_name}")
    
    def validate_input(self, field_name: str, value: Any) -> Tuple[bool, Optional[str]]:
        """Validate single input field.
        
        Args:
            field_name: Name of field to validate
            value: Value to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if field_name not in self.validation_rules:
            # No rule defined - allow but log
            logger.debug(f"No validation rule for field {field_name}")
            return True, None
        
        rule = self.validation_rules[field_name]
        
        try:
            # Check if required
            if value is None:
                if rule.required:
                    return False, f"Field {field_name} is required"
                else:
                    return True, None
            
            # Apply validation rule
            if rule.rule_type == 'type':
                is_valid = isinstance(value, rule.rule_value)
                error_msg = f"Field {field_name} must be of type {rule.rule_value.__name__}"
            
            elif rule.rule_type == 'range':
                min_val, max_val = rule.rule_value
                is_valid = min_val <= value <= max_val
                error_msg = f"Field {field_name} must be between {min_val} and {max_val}"
            
            elif rule.rule_type == 'regex':
                is_valid = bool(re.match(rule.rule_value, str(value)))
                error_msg = f"Field {field_name} format is invalid"
            
            elif rule.rule_type == 'whitelist':
                is_valid = value in rule.rule_value
                error_msg = f"Field {field_name} must be one of {rule.rule_value}"
            
            elif rule.rule_type == 'custom':
                is_valid, custom_error = rule.rule_value(value)
                error_msg = custom_error or f"Field {field_name} failed custom validation"
            
            else:
                raise ValidationError(f"Unknown rule type: {rule.rule_type}")
            
            # Log validation result
            validation_record = {
                'field_name': field_name,
                'value_hash': hashlib.md5(str(value).encode()).hexdigest()[:8],
                'is_valid': is_valid,
                'rule_type': rule.rule_type,
                'timestamp': logger.handlers[0].formatter.formatTime(logging.LogRecord(
                    '', 0, '', 0, '', (), None
                )) if logger.handlers else 'unknown'
            }
            
            self.validation_history.append(validation_record)
            
            if not is_valid:
                self.failed_validations.append(validation_record)
                logger.warning(f"Validation failed for {field_name}: {error_msg}")
            
            return is_valid, error_msg if not is_valid else None
            
        except Exception as e:
            error_msg = f"Validation error for {field_name}: {e}"
            logger.error(error_msg)
            return False, error_msg
    
    def validate_dict(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate dictionary of inputs.
        
        Args:
            data: Dictionary to validate
            
        Returns:
            Tuple of (all_valid, error_messages)
        """
        all_valid = True
        error_messages = []
        
        for field_name, value in data.items():
            is_valid, error_msg = self.validate_input(field_name, value)
            
            if not is_valid:
                all_valid = False
                error_messages.append(error_msg)
        
        # Check for required fields that are missing
        for field_name, rule in self.validation_rules.items():
            if rule.required and field_name not in data:
                all_valid = False
                error_messages.append(f"Required field {field_name} is missing")
        
        return all_valid, error_messages
    
    def _validate_file_path(self, file_path: Union[str, Path]) -> Tuple[bool, Optional[str]]:
        """Validate file path for security (prevent directory traversal)."""
        try:
            path_str = str(file_path)
            
            # Check for directory traversal attempts
            dangerous_patterns = ['../', '..\\', '/../', '\\..\\']
            for pattern in dangerous_patterns:
                if pattern in path_str:
                    return False, "Directory traversal detected in file path"
            
            # Check for absolute paths that might escape sandbox
            if path_str.startswith('/etc/') or path_str.startswith('/proc/'):
                return False, "Access to system directories not allowed"
            
            # Check file extension whitelist for models
            if file_path.suffix:
                safe_extensions = {'.pkl', '.json', '.yaml', '.yml', '.txt', '.csv', '.npy', '.npz'}
                if file_path.suffix.lower() not in safe_extensions:
                    return False, f"File extension {file_path.suffix} not allowed"
            
            return True, None
            
        except Exception as e:
            return False, f"File path validation error: {e}"
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        total_validations = len(self.validation_history)
        failed_validations = len(self.failed_validations)
        
        # Failure rate by field
        field_failures = {}
        for failure in self.failed_validations:
            field = failure['field_name']
            field_failures[field] = field_failures.get(field, 0) + 1
        
        return {
            'total_validations': total_validations,
            'failed_validations': failed_validations,
            'success_rate': (total_validations - failed_validations) / max(1, total_validations),
            'field_failure_counts': field_failures,
            'rules_configured': len(self.validation_rules)
        }


class InputSanitizer:
    """Sanitizes inputs to prevent injection attacks and ensure safety."""
    
    def __init__(self):
        """Initialize input sanitizer."""
        self.sanitization_history = []
        
        # SQL injection patterns
        self.sql_patterns = [
            r"('\s*(;|union|select|insert|update|delete|drop|create|alter)\s*)",
            r"(\s*(;|union|select|insert|update|delete|drop|create|alter)\s*)",
            r"('.*--)",
            r"(\s*--\s*)"
        ]
        
        # Command injection patterns
        self.command_patterns = [
            r"(;|\||&|`|\$\(|\$\{)",
            r"(\.\./)",
            r"(\\x[0-9a-fA-F]{2})",
        ]
        
        # Script injection patterns
        self.script_patterns = [
            r"(<script.*?>.*?</script>)",
            r"(javascript:)",
            r"(on\w+\s*=)",
        ]
        
        logger.info("Input sanitizer initialized")
    
    def sanitize_string(self, input_str: str, strict: bool = False) -> str:
        """Sanitize string input.
        
        Args:
            input_str: Input string to sanitize
            strict: If True, remove suspicious patterns. If False, escape them.
            
        Returns:
            Sanitized string
        """
        if not isinstance(input_str, str):
            raise ValueError("Input must be a string")
        
        original_str = input_str
        sanitized = input_str
        
        # Remove/escape SQL injection patterns
        for pattern in self.sql_patterns:
            if re.search(pattern, sanitized, re.IGNORECASE):
                if strict:
                    sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
                else:
                    sanitized = sanitized.replace(';', '&#59;')
                    sanitized = sanitized.replace('--', '&#45;&#45;')
        
        # Remove/escape command injection patterns
        for pattern in self.command_patterns:
            if re.search(pattern, sanitized):
                if strict:
                    sanitized = re.sub(pattern, '', sanitized)
                else:
                    sanitized = sanitized.replace('|', '&#124;')
                    sanitized = sanitized.replace('&', '&amp;')
                    sanitized = sanitized.replace('`', '&#96;')
        
        # Remove/escape script injection patterns
        for pattern in self.script_patterns:
            if re.search(pattern, sanitized, re.IGNORECASE):
                if strict:
                    sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
                else:
                    sanitized = sanitized.replace('<', '&lt;')
                    sanitized = sanitized.replace('>', '&gt;')
        
        # Log if sanitization occurred
        if sanitized != original_str:
            self.sanitization_history.append({
                'original_hash': hashlib.md5(original_str.encode()).hexdigest()[:8],
                'sanitized_hash': hashlib.md5(sanitized.encode()).hexdigest()[:8],
                'strict': strict,
                'timestamp': 'now'  # Simplified
            })
            
            logger.info(f"String sanitized (strict={strict})")
        
        return sanitized
    
    def sanitize_dict(self, data: Dict[str, Any], strict: bool = False) -> Dict[str, Any]:
        """Sanitize dictionary inputs recursively.
        
        Args:
            data: Dictionary to sanitize
            strict: Strict sanitization mode
            
        Returns:
            Sanitized dictionary
        """
        sanitized_data = {}
        
        for key, value in data.items():
            # Sanitize key
            sanitized_key = self.sanitize_string(str(key), strict=True)  # Always strict for keys
            
            # Sanitize value based on type
            if isinstance(value, str):
                sanitized_value = self.sanitize_string(value, strict=strict)
            elif isinstance(value, dict):
                sanitized_value = self.sanitize_dict(value, strict=strict)
            elif isinstance(value, list):
                sanitized_value = self.sanitize_list(value, strict=strict)
            else:
                sanitized_value = value  # Numbers, booleans, etc. pass through
            
            sanitized_data[sanitized_key] = sanitized_value
        
        return sanitized_data
    
    def sanitize_list(self, data: List[Any], strict: bool = False) -> List[Any]:
        """Sanitize list inputs.
        
        Args:
            data: List to sanitize
            strict: Strict sanitization mode
            
        Returns:
            Sanitized list
        """
        sanitized_list = []
        
        for item in data:
            if isinstance(item, str):
                sanitized_item = self.sanitize_string(item, strict=strict)
            elif isinstance(item, dict):
                sanitized_item = self.sanitize_dict(item, strict=strict)
            elif isinstance(item, list):
                sanitized_item = self.sanitize_list(item, strict=strict)
            else:
                sanitized_item = item
            
            sanitized_list.append(sanitized_item)
        
        return sanitized_list
    
    def detect_injection_attempt(self, input_str: str) -> Dict[str, Any]:
        """Detect potential injection attempts.
        
        Args:
            input_str: String to analyze
            
        Returns:
            Detection results
        """
        if not isinstance(input_str, str):
            return {'detected': False, 'patterns': [], 'risk_level': 'none'}
        
        detected_patterns = []
        
        # Check SQL patterns
        for i, pattern in enumerate(self.sql_patterns):
            if re.search(pattern, input_str, re.IGNORECASE):
                detected_patterns.append(f"sql_{i}")
        
        # Check command patterns
        for i, pattern in enumerate(self.command_patterns):
            if re.search(pattern, input_str):
                detected_patterns.append(f"command_{i}")
        
        # Check script patterns
        for i, pattern in enumerate(self.script_patterns):
            if re.search(pattern, input_str, re.IGNORECASE):
                detected_patterns.append(f"script_{i}")
        
        # Determine risk level
        if len(detected_patterns) >= 3:
            risk_level = 'critical'
        elif len(detected_patterns) >= 2:
            risk_level = 'high'
        elif len(detected_patterns) == 1:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        detection_result = {
            'detected': len(detected_patterns) > 0,
            'patterns': detected_patterns,
            'pattern_count': len(detected_patterns),
            'risk_level': risk_level,
            'input_length': len(input_str)
        }
        
        if detection_result['detected']:
            logger.warning(f"Injection attempt detected: {risk_level} risk, "
                          f"{len(detected_patterns)} patterns")
        
        return detection_result
    
    def sanitize_numpy_array(self, array: np.ndarray) -> np.ndarray:
        """Sanitize numpy array (check for NaN, Inf, reasonable values)."""
        if not isinstance(array, np.ndarray):
            raise ValueError("Input must be numpy array")
        
        # Check for problematic values
        has_nan = np.isnan(array).any()
        has_inf = np.isinf(array).any()
        
        sanitized = array.copy()
        
        # Replace NaN with zeros
        if has_nan:
            sanitized = np.nan_to_num(sanitized, nan=0.0)
            logger.warning("NaN values replaced with zeros in array")
        
        # Replace Inf with large finite values
        if has_inf:
            sanitized = np.nan_to_num(sanitized, posinf=1e10, neginf=-1e10)
            logger.warning("Infinite values replaced with finite values")
        
        # Check for extremely large values that might cause overflow
        max_safe_value = 1e15
        if np.abs(sanitized).max() > max_safe_value:
            # Clip to safe range
            sanitized = np.clip(sanitized, -max_safe_value, max_safe_value)
            logger.warning(f"Array values clipped to safe range [±{max_safe_value}]")
        
        return sanitized
    
    def get_sanitization_stats(self) -> Dict[str, Any]:
        """Get sanitization statistics."""
        return {
            'total_sanitizations': len(self.sanitization_history),
            'recent_sanitizations': len([s for s in self.sanitization_history[-100:]]),
            'sql_patterns_count': len(self.sql_patterns),
            'command_patterns_count': len(self.command_patterns),
            'script_patterns_count': len(self.script_patterns)
        }


# Global instances for easy access
security_validator = SecurityValidator()
input_sanitizer = InputSanitizer()


def validate_and_sanitize(
    data: Dict[str, Any], 
    sanitize: bool = True,
    strict_sanitization: bool = False
) -> Tuple[Dict[str, Any], bool, List[str]]:
    """Convenience function to validate and sanitize data.
    
    Args:
        data: Data to process
        sanitize: Whether to sanitize inputs
        strict_sanitization: Use strict sanitization mode
        
    Returns:
        Tuple of (processed_data, is_valid, error_messages)
    """
    # Validate first
    is_valid, error_messages = security_validator.validate_dict(data)
    
    # Sanitize if requested
    if sanitize:
        try:
            processed_data = input_sanitizer.sanitize_dict(data, strict=strict_sanitization)
        except Exception as e:
            error_messages.append(f"Sanitization failed: {e}")
            processed_data = data
            is_valid = False
    else:
        processed_data = data
    
    return processed_data, is_valid, error_messages