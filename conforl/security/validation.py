"""Security validation and input sanitization.

Comprehensive input validation and sanitization to prevent
injection attacks and ensure data integrity in safe RL systems.
"""

import re
import json
import pickle
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Minimal numpy-like interface for basic functionality
    class np:
        @staticmethod
        def isnan(data):
            return False
        @staticmethod
        def isinf(data):
            return False
        @staticmethod
        def any(data):
            return any(data) if hasattr(data, '__iter__') else False
        @staticmethod
        def abs(data):
            if hasattr(data, '__iter__'):
                return [abs(x) for x in data]
            return abs(data)
        @staticmethod
        def max(data):
            if hasattr(data, '__iter__'):
                return max(data) if data else 0
            return data
        @staticmethod
        def clip(data, min_val, max_val):
            if hasattr(data, '__iter__'):
                return [max(min_val, min(x, max_val)) for x in data]
            return max(min_val, min(data, max_val))
        @staticmethod
        def nan_to_num(data, nan=0.0, posinf=1e10, neginf=-1e10):
            return data  # Simplified
        @staticmethod
        def copy(data):
            return data.copy() if hasattr(data, 'copy') else data
        ndarray = list  # Use list as fallback
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
import hashlib
import time
import uuid
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict
import warnings

from ..utils.logging import get_logger
from ..utils.errors import ValidationError, SecurityError

logger = get_logger(__name__)


class ThreatLevel(Enum):
    """Security threat levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AttackVector(Enum):
    """Types of security attack vectors."""
    CODE_INJECTION = "code_injection"
    PATH_TRAVERSAL = "path_traversal"
    BUFFER_OVERFLOW = "buffer_overflow"
    DESERIALIZATION = "deserialization"
    MODEL_POISONING = "model_poisoning"
    ADVERSARIAL_INPUT = "adversarial_input"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_EXFILTRATION = "data_exfiltration"


@dataclass
class SecurityAlert:
    """Security alert for threat detection."""
    
    threat_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    threat_level: ThreatLevel = ThreatLevel.LOW
    attack_vector: AttackVector = AttackVector.ADVERSARIAL_INPUT
    description: str = ""
    source_ip: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    mitigation_applied: bool = False
    false_positive_likelihood: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "threat_id": self.threat_id,
            "threat_level": self.threat_level.value,
            "attack_vector": self.attack_vector.value,
            "description": self.description,
            "source_ip": self.source_ip,
            "timestamp": self.timestamp,
            "mitigation_applied": self.mitigation_applied,
            "false_positive_likelihood": self.false_positive_likelihood
        }


class AdvancedSecurityValidator:
    """Advanced security validator with threat detection and mitigation."""
    
    def __init__(self):
        """Initialize security validator."""
        self.threat_signatures = self._load_threat_signatures()
        self.anomaly_detector = AnomalyDetector()
        self.rate_limiter = RateLimiter()
        self.security_alerts = []
        self.blocked_ips = set()
        self.security_metrics = defaultdict(int)
        
        logger.info("Initialized AdvancedSecurityValidator with threat detection")
    
    def validate_and_sanitize(self, data: Any, context: str = "unknown") -> Tuple[Any, List[SecurityAlert]]:
        """Comprehensive validation and sanitization with threat detection."""
        alerts = []
        
        try:
            # Rate limiting check
            if not self.rate_limiter.is_allowed(context):
                alert = SecurityAlert(
                    threat_level=ThreatLevel.HIGH,
                    attack_vector=AttackVector.PRIVILEGE_ESCALATION,
                    description=f"Rate limit exceeded for context: {context}"
                )
                alerts.append(alert)
                raise SecurityError("Rate limit exceeded")
            
            # Anomaly detection
            anomaly_score = self.anomaly_detector.detect_anomaly(data, context)
            if anomaly_score > 0.8:
                alert = SecurityAlert(
                    threat_level=ThreatLevel.MEDIUM,
                    attack_vector=AttackVector.ADVERSARIAL_INPUT,
                    description=f"Anomalous input detected (score: {anomaly_score:.3f})"
                )
                alerts.append(alert)
            
            # Signature-based threat detection
            threat_alerts = self._detect_threats(data, context)
            alerts.extend(threat_alerts)
            
            # Sanitize data based on type
            sanitized_data = self._sanitize_by_type(data)
            
            # Additional validation checks
            validation_alerts = self._run_validation_checks(sanitized_data, context)
            alerts.extend(validation_alerts)
            
            # Log security events
            if alerts:
                self._log_security_events(alerts, context)
            
            return sanitized_data, alerts
            
        except Exception as e:
            # Security error - log and create critical alert
            critical_alert = SecurityAlert(
                threat_level=ThreatLevel.CRITICAL,
                attack_vector=AttackVector.CODE_INJECTION,
                description=f"Security validation failed: {str(e)}"
            )
            alerts.append(critical_alert)
            self._log_security_events([critical_alert], context)
            raise SecurityError(f"Security validation failed: {e}")
    
    def _detect_threats(self, data: Any, context: str) -> List[SecurityAlert]:
        """Detect security threats using signature-based analysis."""
        alerts = []
        
        if isinstance(data, str):
            # Check for code injection patterns
            for pattern, vector in self.threat_signatures["injection"].items():
                if re.search(pattern, data, re.IGNORECASE):
                    alert = SecurityAlert(
                        threat_level=ThreatLevel.HIGH,
                        attack_vector=vector,
                        description=f"Potential {vector.value} detected: {pattern}"
                    )
                    alerts.append(alert)
            
            # Check for path traversal
            if self._is_path_traversal(data):
                alert = SecurityAlert(
                    threat_level=ThreatLevel.HIGH,
                    attack_vector=AttackVector.PATH_TRAVERSAL,
                    description="Path traversal attempt detected"
                )
                alerts.append(alert)
        
        elif isinstance(data, (bytes, bytearray)):
            # Check for malicious payloads in binary data
            if self._contains_suspicious_patterns(data):
                alert = SecurityAlert(
                    threat_level=ThreatLevel.MEDIUM,
                    attack_vector=AttackVector.BUFFER_OVERFLOW,
                    description="Suspicious binary patterns detected"
                )
                alerts.append(alert)
        
        return alerts
    
    def _sanitize_by_type(self, data: Any) -> Any:
        """Sanitize data based on its type."""
        if isinstance(data, str):
            return self._sanitize_string(data)
        elif isinstance(data, dict):
            return self._sanitize_dict(data)
        elif isinstance(data, list):
            return self._sanitize_list(data)
        elif NUMPY_AVAILABLE and isinstance(data, np.ndarray):
            return self._sanitize_numpy_array(data)
        elif isinstance(data, (int, float)):
            return self._sanitize_numeric(data)
        else:
            # For unknown types, apply conservative sanitization
            return self._sanitize_object(data)
    
    def _sanitize_string(self, text: str) -> str:
        """Sanitize string input to prevent injection attacks."""
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # Remove control characters except common whitespace
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\t\n\r')
        
        # Limit length to prevent DoS
        max_length = 10000
        if len(text) > max_length:
            logger.warning(f"String truncated from {len(text)} to {max_length} characters")
            text = text[:max_length]
        
        # Escape HTML/XML entities
        text = (text.replace('&', '&amp;')
                   .replace('<', '&lt;')
                   .replace('>', '&gt;')
                   .replace('"', '&quot;')
                   .replace("'", '&#x27;'))
        
        return text
    
    def _sanitize_dict(self, data: Dict[Any, Any]) -> Dict[Any, Any]:
        """Sanitize dictionary recursively."""
        sanitized = {}
        max_keys = 1000  # Prevent DoS through large dictionaries
        
        for i, (key, value) in enumerate(data.items()):
            if i >= max_keys:
                logger.warning(f"Dictionary truncated at {max_keys} keys")
                break
            
            # Sanitize key and value
            sanitized_key = self._sanitize_by_type(key)
            sanitized_value = self._sanitize_by_type(value)
            sanitized[sanitized_key] = sanitized_value
        
        return sanitized
    
    def _sanitize_list(self, data: List[Any]) -> List[Any]:
        """Sanitize list recursively."""
        max_items = 10000  # Prevent DoS through large lists
        
        if len(data) > max_items:
            logger.warning(f"List truncated from {len(data)} to {max_items} items")
            data = data[:max_items]
        
        return [self._sanitize_by_type(item) for item in data]
    
    def _sanitize_numpy_array(self, data: np.ndarray) -> np.ndarray:
        """Sanitize NumPy array."""
        # Check for suspicious values
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            logger.warning("Replacing NaN/Inf values in numpy array")
            data = np.nan_to_num(data, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # Clip extreme values
        if data.dtype in [np.float32, np.float64]:
            data = np.clip(data, -1e10, 1e10)
        
        # Limit array size to prevent memory exhaustion
        max_elements = 1000000
        if data.size > max_elements:
            logger.warning(f"Array truncated from {data.size} to {max_elements} elements")
            data = data.flat[:max_elements].reshape(-1)
        
        return np.copy(data)  # Return copy to prevent reference manipulation
    
    def _sanitize_numeric(self, data: Union[int, float]) -> Union[int, float]:
        """Sanitize numeric values."""
        if isinstance(data, float):
            # Handle special float values
            if np.isnan(data):
                return 0.0
            elif np.isinf(data):
                return 1e10 if data > 0 else -1e10
            else:
                # Clip extreme values
                return max(-1e10, min(1e10, data))
        else:
            # Clip large integers
            return max(-2**31, min(2**31 - 1, data))
    
    def _sanitize_object(self, data: Any) -> Any:
        """Conservative sanitization for unknown object types."""
        # For safety, convert to string representation and sanitize
        try:
            str_repr = str(data)
            return self._sanitize_string(str_repr)
        except Exception:
            return "SANITIZED_OBJECT"
    
    def _is_path_traversal(self, path: str) -> bool:
        """Check if string contains path traversal patterns."""
        traversal_patterns = [
            r'\.\./',           # ../
            r'\.\.\\'           # ..\
            r'%2e%2e%2f',       # URL encoded ../
            r'%2e%2e%5c',       # URL encoded ..\
            r'\.\.%2f',         # Mixed encoding
            r'\.\.%5c'
        ]
        
        return any(re.search(pattern, path, re.IGNORECASE) for pattern in traversal_patterns)
    
    def _contains_suspicious_patterns(self, data: bytes) -> bool:
        """Check binary data for suspicious patterns."""
        # Look for common shellcode patterns
        suspicious_bytes = [
            b'\x90\x90\x90\x90',  # NOP sled
            b'\xcc\xcc\xcc\xcc',  # Debug breaks
            b'\x00\x00\x00\x00'   # Null padding (potential overflow)
        ]
        
        return any(pattern in data for pattern in suspicious_bytes)
    
    def _run_validation_checks(self, data: Any, context: str) -> List[SecurityAlert]:
        """Run additional validation checks."""
        alerts = []
        
        # Check for suspicious file extensions in strings
        if isinstance(data, str) and self._has_executable_extension(data):
            alert = SecurityAlert(
                threat_level=ThreatLevel.MEDIUM,
                attack_vector=AttackVector.CODE_INJECTION,
                description="Potential executable file reference detected"
            )
            alerts.append(alert)
        
        # Check for suspicious network addresses
        if isinstance(data, str) and self._contains_private_network_info(data):
            alert = SecurityAlert(
                threat_level=ThreatLevel.LOW,
                attack_vector=AttackVector.DATA_EXFILTRATION,
                description="Private network information detected"
            )
            alerts.append(alert)
        
        return alerts
    
    def _has_executable_extension(self, text: str) -> bool:
        """Check if text contains executable file extensions."""
        executable_extensions = ['.exe', '.bat', '.cmd', '.sh', '.ps1', '.py', '.js']
        return any(ext in text.lower() for ext in executable_extensions)
    
    def _contains_private_network_info(self, text: str) -> bool:
        """Check if text contains private network information."""
        # Look for IP addresses, especially private ranges
        ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
        ips = re.findall(ip_pattern, text)
        
        for ip in ips:
            parts = ip.split('.')
            try:
                first_octet = int(parts[0])
                second_octet = int(parts[1])
                
                # Check for private IP ranges
                if (first_octet == 10 or 
                    (first_octet == 172 and 16 <= second_octet <= 31) or
                    (first_octet == 192 and second_octet == 168)):
                    return True
            except ValueError:
                continue
        
        return False
    
    def _load_threat_signatures(self) -> Dict[str, Dict[str, AttackVector]]:
        """Load threat detection signatures."""
        return {
            "injection": {
                r"(union\s+select|insert\s+into|delete\s+from|drop\s+table)": AttackVector.CODE_INJECTION,
                r"<script[^>]*>.*?</script>": AttackVector.CODE_INJECTION,
                r"javascript\s*:": AttackVector.CODE_INJECTION,
                r"eval\s*\(": AttackVector.CODE_INJECTION,
                r"exec\s*\(": AttackVector.CODE_INJECTION,
                r"import\s+os": AttackVector.CODE_INJECTION,
                r"__import__": AttackVector.CODE_INJECTION,
                r"subprocess\.|os\.system": AttackVector.CODE_INJECTION
            }
        }
    
    def _log_security_events(self, alerts: List[SecurityAlert], context: str):
        """Log security events for audit trail."""
        for alert in alerts:
            self.security_alerts.append(alert)
            self.security_metrics[f"{alert.threat_level.value}_threats"] += 1
            
            log_level = "warning" if alert.threat_level in [ThreatLevel.LOW, ThreatLevel.MEDIUM] else "error"
            getattr(logger, log_level)(
                f"Security Alert [{alert.threat_level.value}]: {alert.description} "
                f"(Context: {context}, Vector: {alert.attack_vector.value})"
            )
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """Get security metrics for monitoring."""
        return {
            "total_alerts": len(self.security_alerts),
            "alert_breakdown": dict(self.security_metrics),
            "blocked_ips": len(self.blocked_ips),
            "recent_alerts": len([a for a in self.security_alerts 
                                if time.time() - a.timestamp < 3600])  # Last hour
        }


class AnomalyDetector:
    """Anomaly detection for unusual input patterns."""
    
    def __init__(self):
        """Initialize anomaly detector."""
        self.baseline_stats = {}
        self.anomaly_threshold = 2.0  # Standard deviations
        
    def detect_anomaly(self, data: Any, context: str) -> float:
        """Detect anomalies in input data."""
        try:
            # Simple statistical anomaly detection
            features = self._extract_features(data)
            
            if context not in self.baseline_stats:
                # Initialize baseline
                self.baseline_stats[context] = {"features": [], "mean": None, "std": None}
            
            baseline = self.baseline_stats[context]
            baseline["features"].append(features)
            
            # Keep only recent observations
            if len(baseline["features"]) > 1000:
                baseline["features"] = baseline["features"][-1000:]
            
            # Compute anomaly score if we have enough data
            if len(baseline["features"]) > 10:
                feature_values = [f.get("size", 0) for f in baseline["features"]]
                mean_val = np.mean(feature_values)
                std_val = np.std(feature_values)
                
                if std_val > 0:
                    current_size = features.get("size", 0)
                    z_score = abs(current_size - mean_val) / std_val
                    return min(z_score / self.anomaly_threshold, 1.0)
            
            return 0.0
            
        except Exception:
            return 0.0  # Conservative fallback
    
    def _extract_features(self, data: Any) -> Dict[str, float]:
        """Extract features for anomaly detection."""
        features = {"size": 0, "complexity": 0, "entropy": 0}
        
        try:
            if isinstance(data, str):
                features["size"] = len(data)
                features["complexity"] = len(set(data)) / max(len(data), 1)
                features["entropy"] = self._compute_entropy(data)
            elif isinstance(data, (list, tuple)):
                features["size"] = len(data)
                features["complexity"] = len(set(str(item) for item in data)) / max(len(data), 1)
            elif isinstance(data, dict):
                features["size"] = len(data)
                features["complexity"] = len(data.keys()) + len(data.values())
            elif NUMPY_AVAILABLE and isinstance(data, np.ndarray):
                features["size"] = data.size
                if data.size > 0:
                    features["complexity"] = np.unique(data).size / data.size
            
        except Exception:
            pass  # Use default values
        
        return features
    
    def _compute_entropy(self, text: str) -> float:
        """Compute Shannon entropy of text."""
        if not text:
            return 0.0
        
        char_counts = defaultdict(int)
        for char in text:
            char_counts[char] += 1
        
        length = len(text)
        entropy = 0.0
        
        for count in char_counts.values():
            prob = count / length
            if prob > 0:
                entropy -= prob * np.log(prob)
        
        return entropy


class RateLimiter:
    """Rate limiter for preventing abuse."""
    
    def __init__(self, max_requests: int = 100, time_window: int = 60):
        """Initialize rate limiter."""
        self.max_requests = max_requests
        self.time_window = time_window
        self.request_history = defaultdict(list)
    
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed under rate limit."""
        current_time = time.time()
        
        # Clean old requests
        self.request_history[identifier] = [
            timestamp for timestamp in self.request_history[identifier]
            if current_time - timestamp < self.time_window
        ]
        
        # Check if under limit
        if len(self.request_history[identifier]) < self.max_requests:
            self.request_history[identifier].append(current_time)
            return True
        
        return False
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
    
    def sanitize_numpy_array(self, array) -> any:
        """Sanitize numpy array (check for NaN, Inf, reasonable values)."""
        if not NUMPY_AVAILABLE:
            return array  # Skip validation if numpy not available
        
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