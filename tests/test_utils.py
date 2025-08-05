"""Tests for utility functions."""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import os

from conforl.utils.validation import (
    validate_config,
    validate_environment,
    validate_dataset,
    validate_trajectory_data,
    validate_risk_parameters,
    validate_file_paths
)
from conforl.utils.errors import (
    ConfigurationError,
    ValidationError,
    EnvironmentError,
    DataError,
    SecurityError
)
from conforl.utils.security import (
    sanitize_input,
    sanitize_file_path,
    validate_file_path,
    check_permissions,
    sanitize_config_dict
)
from conforl.utils.logging import setup_logging, get_logger, JSONFormatter


class TestValidation:
    """Test cases for validation utilities."""
    
    def test_validate_config_valid(self, sample_config):
        """Test validation of valid configuration."""
        validated = validate_config(sample_config)
        
        assert validated == sample_config
        assert validated['learning_rate'] == 0.001
        assert validated['target_risk'] == 0.05
    
    def test_validate_config_invalid_type(self):
        """Test validation with non-dict config."""
        with pytest.raises(ConfigurationError):
            validate_config("not a dict")
    
    def test_validate_config_invalid_learning_rate(self):
        """Test validation with invalid learning rate."""
        config = {'learning_rate': 0.0}
        with pytest.raises(ConfigurationError, match="learning_rate must be in"):
            validate_config(config)
        
        config = {'learning_rate': 2.0}
        with pytest.raises(ConfigurationError):
            validate_config(config)
    
    def test_validate_config_invalid_target_risk(self):
        """Test validation with invalid target risk."""
        config = {'target_risk': 0.0}
        with pytest.raises(ConfigurationError):
            validate_config(config)
        
        config = {'target_risk': 1.5}
        with pytest.raises(ConfigurationError):
            validate_config(config)
    
    def test_validate_environment(self, simple_env):
        """Test environment validation."""
        # Should not raise an error
        validate_environment(simple_env)
    
    def test_validate_environment_missing_attributes(self):
        """Test validation with invalid environment."""
        invalid_env = object()  # Missing required attributes
        
        with pytest.raises(EnvironmentError):
            validate_environment(invalid_env)
    
    def test_validate_dataset_valid(self, sample_dataset):
        """Test validation of valid dataset."""
        validated = validate_dataset(sample_dataset)
        assert validated == sample_dataset
    
    def test_validate_dataset_invalid_type(self):
        """Test validation with non-dict dataset."""
        with pytest.raises(DataError):
            validate_dataset("not a dict")
    
    def test_validate_dataset_missing_keys(self):
        """Test validation with missing required keys."""
        incomplete_dataset = {
            'observations': np.random.random((10, 4)),
            'actions': np.random.random((10, 1))
            # Missing 'rewards'
        }
        
        with pytest.raises(DataError, match="missing required key"):
            validate_dataset(incomplete_dataset)
    
    def test_validate_dataset_length_mismatch(self):
        """Test validation with mismatched data lengths."""
        dataset = {
            'observations': np.random.random((10, 4)),
            'actions': np.random.random((5, 1)),  # Wrong length
            'rewards': np.random.random((10,))
        }
        
        with pytest.raises(DataError, match="length mismatch"):
            validate_dataset(dataset)
    
    def test_validate_dataset_nan_values(self):
        """Test validation with NaN values."""
        dataset = {
            'observations': np.array([[1.0, 2.0], [np.nan, 4.0]]),
            'actions': np.array([1.0, 2.0]),
            'rewards': np.array([1.0, 2.0])
        }
        
        with pytest.raises(DataError, match="contain NaN values"):
            validate_dataset(dataset)
    
    def test_validate_trajectory_data(self):
        """Test trajectory data validation."""
        states = np.random.random((5, 2))
        actions = np.random.randint(0, 2, 5)
        rewards = np.random.random(5)
        dones = np.array([False] * 5)
        
        # Should not raise an error
        validate_trajectory_data(states, actions, rewards, dones)
    
    def test_validate_trajectory_data_invalid_types(self):
        """Test validation with invalid data types."""
        with pytest.raises(ValidationError):
            validate_trajectory_data("not array", np.array([1]), np.array([1]), np.array([True]))
    
    def test_validate_trajectory_data_length_mismatch(self):
        """Test validation with mismatched trajectory lengths."""
        states = np.random.random((5, 2))
        actions = np.random.randint(0, 2, 3)  # Wrong length
        rewards = np.random.random(5)
        dones = np.array([False] * 5)
        
        with pytest.raises(ValidationError, match="length mismatch"):
            validate_trajectory_data(states, actions, rewards, dones)
    
    def test_validate_risk_parameters(self):
        """Test risk parameter validation."""
        # Valid parameters
        validate_risk_parameters(0.05, 0.95, 0.9)
        
        # Invalid target_risk
        with pytest.raises(ValidationError):
            validate_risk_parameters(0.0, 0.95)
        
        with pytest.raises(ValidationError):
            validate_risk_parameters(1.5, 0.95)
        
        # Invalid confidence
        with pytest.raises(ValidationError):
            validate_risk_parameters(0.05, 0.0)
    
    def test_validate_file_paths(self):
        """Test file path validation."""
        paths = ["./test.txt", "/tmp/test.log"]
        validated = validate_file_paths(paths)
        
        assert len(validated) == 2
        assert all(isinstance(p, Path) for p in validated)
    
    def test_validate_file_paths_traversal(self):
        """Test path traversal detection."""
        paths = ["../../../etc/passwd"]
        
        with pytest.raises(ValidationError, match="Path traversal detected"):
            validate_file_paths(paths)


class TestSecurity:
    """Test cases for security utilities."""
    
    def test_sanitize_input_string(self):
        """Test string input sanitization."""
        # Valid string
        result = sanitize_input("hello world", "string")
        assert result == "hello world"
        
        # String with whitespace
        result = sanitize_input("  hello  ", "string")
        assert result == "hello"
    
    def test_sanitize_input_max_length(self):
        """Test string length validation."""
        long_string = "a" * 1000
        
        with pytest.raises(SecurityError, match="exceeds maximum"):
            sanitize_input(long_string, "string", max_length=100)
    
    def test_sanitize_input_dangerous_patterns(self):
        """Test detection of dangerous patterns."""
        dangerous_inputs = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "onclick=alert('xss')",
            "../../../etc/passwd"
        ]
        
        for dangerous_input in dangerous_inputs:
            with pytest.raises(SecurityError, match="Dangerous pattern detected"):
                sanitize_input(dangerous_input, "string")
    
    def test_sanitize_input_number(self):
        """Test number input sanitization."""
        # Valid numbers
        assert sanitize_input("123", "number") == 123
        assert sanitize_input("123.45", "number") == 123.45
        assert sanitize_input("-123.45", "number") == -123.45
        
        # Invalid numbers
        with pytest.raises(SecurityError):
            sanitize_input("abc", "number")
        
        with pytest.raises(SecurityError):
            sanitize_input("123abc", "number")
    
    def test_sanitize_file_path(self):
        """Test file path sanitization."""
        # Valid path
        result = sanitize_file_path("./test.txt")
        assert isinstance(result, Path)
        
        # Path traversal
        with pytest.raises(SecurityError, match="Path traversal detected"):
            sanitize_file_path("../../../etc/passwd")
    
    def test_validate_file_path_permissions(self):
        """Test file path validation with permissions."""
        with tempfile.NamedTemporaryFile() as tmp_file:
            # Should not raise error for existing readable file
            validate_file_path(tmp_file.name, must_exist=True, check_readable=True)
    
    def test_check_permissions(self):
        """Test permission checking."""
        with tempfile.NamedTemporaryFile() as tmp_file:
            perms = check_permissions(tmp_file.name)
            
            assert perms['exists'] == True
            assert perms['is_file'] == True
            assert perms['is_directory'] == False
            assert perms['readable'] == True
    
    def test_sanitize_config_dict(self):
        """Test configuration dictionary sanitization."""
        config = {
            'valid_key': 'valid_value',
            'numeric_key': 42,
            'list_key': ['item1', 'item2'],
            'nested_dict': {'inner_key': 'inner_value'}
        }
        
        sanitized = sanitize_config_dict(config)
        
        assert 'valid_key' in sanitized
        assert sanitized['valid_key'] == 'valid_value'
        assert sanitized['numeric_key'] == 42
        assert 'list_key' in sanitized
        assert 'nested_dict' in sanitized


class TestLogging:
    """Test cases for logging utilities."""
    
    def test_setup_logging(self):
        """Test logging setup."""
        logger = setup_logging(level="INFO", include_console=False)
        assert logger.name == "ConfoRL"
        assert logger.level <= 20  # INFO level or lower
    
    def test_get_logger(self):
        """Test logger retrieval."""
        logger = get_logger("test_module")
        assert logger.name == "ConfoRL.test_module"
    
    def test_json_formatter(self):
        """Test JSON log formatting."""
        import logging
        
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test message",
            args=(),
            exc_info=None
        )
        
        formatted = formatter.format(record)
        
        # Should be valid JSON
        import json
        parsed = json.loads(formatted)
        
        assert parsed['level'] == 'INFO'
        assert parsed['message'] == 'test message'
        assert 'timestamp' in parsed
    
    def test_logging_with_context(self):
        """Test contextual logging."""
        from conforl.utils.logging import ContextualLogger
        
        base_logger = get_logger("test")
        context = {"experiment_id": "exp_123", "run_id": "run_456"}
        
        contextual_logger = ContextualLogger(base_logger, context)
        
        # Should not raise an error
        contextual_logger.info("Test message", extra_key="extra_value")
    
    def test_performance_logger(self):
        """Test performance logging."""
        from conforl.utils.logging import PerformanceLogger
        import time
        
        base_logger = get_logger("test")
        perf_logger = PerformanceLogger(base_logger)
        
        # Test timer functionality
        perf_logger.start_timer("test_operation")
        time.sleep(0.01)  # Small delay
        perf_logger.end_timer("test_operation")
        
        # Test metrics logging
        metrics = {"accuracy": 0.95, "loss": 0.05}
        perf_logger.log_metrics(metrics, prefix="eval_")
    
    def test_security_logger(self):
        """Test security logging."""
        from conforl.utils.logging import SecurityLogger
        
        base_logger = get_logger("test")
        sec_logger = SecurityLogger(base_logger)
        
        # Test access logging
        sec_logger.log_access_attempt("sensitive_resource", "user123", allowed=True)
        sec_logger.log_access_attempt("sensitive_resource", "user456", allowed=False, reason="insufficient permissions")
        
        # Test security event logging
        sec_logger.log_security_event("LOGIN_ATTEMPT", "Failed login from IP 1.2.3.4")
        
        # Test data access logging
        sec_logger.log_data_access("user_data", "read", "system")


class TestErrors:
    """Test cases for custom error classes."""
    
    def test_conforl_error(self):
        """Test base ConfoRLError."""
        from conforl.utils.errors import ConfoRLError
        
        error = ConfoRLError("Test message", "TEST_CODE")
        assert str(error) == "[TEST_CODE] Test message"
        assert error.error_code == "TEST_CODE"
        assert error.message == "Test message"
    
    def test_configuration_error(self):
        """Test ConfigurationError."""
        error = ConfigurationError("Invalid config", "test_key")
        assert error.error_code == "CONFIG_ERROR"
        assert error.config_key == "test_key"
    
    def test_validation_error(self):
        """Test ValidationError."""
        error = ValidationError("Validation failed", "input_validation")
        assert error.error_code == "VALIDATION_ERROR"
        assert error.validation_type == "input_validation"
    
    def test_risk_control_error(self):
        """Test RiskControlError."""
        from conforl.utils.errors import RiskControlError
        
        error = RiskControlError("Risk too high", 0.15)
        assert error.error_code == "RISK_CONTROL_ERROR"
        assert error.risk_level == 0.15
    
    def test_deployment_error(self):
        """Test DeploymentError."""
        from conforl.utils.errors import DeploymentError
        
        error = DeploymentError("Deployment failed", "initialization")
        assert error.error_code == "DEPLOYMENT_ERROR"
        assert error.deployment_stage == "initialization"