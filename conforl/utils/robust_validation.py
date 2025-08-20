"""
Comprehensive input validation and sanitization for robust operations.
Generation 2: Enhanced validation with security and error handling.
"""

import re
import json
import hashlib
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path

from .errors import (
    ValidationError, SecurityError, ConfigurationError,
    InvalidTrajectoryError, InvalidRiskParameterError
)


class RobustValidator:
    """Enhanced validator with security and comprehensive error handling."""
    
    def __init__(self, strict_mode: bool = True):
        """Initialize robust validator.
        
        Args:
            strict_mode: If True, apply strict validation rules
        """
        self.strict_mode = strict_mode
        self.validation_history = []
        
        # Security patterns for input sanitization
        self.dangerous_patterns = [
            r'<script.*?>.*?</script>',  # XSS
            r'javascript:',              # JavaScript injection
            r'eval\s*\(',               # Code injection
            r'exec\s*\(',               # Code execution
            r'import\s+os',             # OS import
            r'__.*__',                  # Python internals
            r'\.\./\.\.',               # Directory traversal
        ]
        
    def validate_and_sanitize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize configuration with comprehensive checks.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Sanitized and validated configuration
            
        Raises:
            ConfigurationError: If configuration is invalid
            SecurityError: If configuration contains security threats
        """
        try:
            # Deep copy to avoid modifying original
            sanitized_config = self._deep_copy_sanitize(config)
            
            # Validate required fields
            self._validate_config_structure(sanitized_config)
            
            # Security validation
            self._validate_config_security(sanitized_config)
            
            # Value range validation
            self._validate_config_ranges(sanitized_config)
            
            # Log successful validation
            self.validation_history.append({
                'type': 'config_validation',
                'status': 'success',
                'config_hash': self._compute_hash(sanitized_config)
            })
            
            return sanitized_config
            
        except Exception as e:
            self.validation_history.append({
                'type': 'config_validation',
                'status': 'failure',
                'error': str(e)
            })
            raise ConfigurationError(f"Configuration validation failed: {e}")
    
    def validate_trajectory_robust(self, trajectory_data: Any) -> bool:
        """Robustly validate trajectory data with comprehensive checks.
        
        Args:
            trajectory_data: Trajectory data to validate
            
        Returns:
            True if valid
            
        Raises:
            InvalidTrajectoryError: If trajectory is invalid
        """
        try:
            # Check basic structure
            if not hasattr(trajectory_data, 'states'):
                raise InvalidTrajectoryError("Trajectory missing required 'states' attribute")
            
            if not hasattr(trajectory_data, 'actions'):
                raise InvalidTrajectoryError("Trajectory missing required 'actions' attribute")
            
            # Check data consistency
            states_len = len(trajectory_data.states)
            actions_len = len(trajectory_data.actions)
            
            if states_len == 0:
                raise InvalidTrajectoryError("Trajectory cannot have empty states")
            
            if abs(states_len - actions_len) > 1:  # Allow off-by-one for terminal states
                raise InvalidTrajectoryError(
                    f"State-action length mismatch: {states_len} states, {actions_len} actions"
                )
            
            # Validate individual components
            self._validate_states_robust(trajectory_data.states)
            self._validate_actions_robust(trajectory_data.actions)
            
            if hasattr(trajectory_data, 'rewards'):
                self._validate_rewards_robust(trajectory_data.rewards)
            
            # Security check for embedded data
            if hasattr(trajectory_data, 'infos'):
                self._validate_infos_security(trajectory_data.infos)
            
            return True
            
        except Exception as e:
            raise InvalidTrajectoryError(f"Trajectory validation failed: {e}")
    
    def validate_risk_parameters_comprehensive(
        self, 
        target_risk: float, 
        confidence: float,
        **kwargs
    ) -> Tuple[float, float, Dict[str, Any]]:
        """Comprehensive validation of risk parameters.
        
        Args:
            target_risk: Target risk level
            confidence: Confidence level
            **kwargs: Additional parameters
            
        Returns:
            Validated (target_risk, confidence, additional_params)
            
        Raises:
            InvalidRiskParameterError: If parameters are invalid
        """
        try:
            # Validate target_risk
            if not isinstance(target_risk, (int, float)):
                raise InvalidRiskParameterError(f"target_risk must be numeric, got {type(target_risk)}")
            
            if not (0.0 <= target_risk <= 1.0):
                raise InvalidRiskParameterError(f"target_risk must be in [0,1], got {target_risk}")
            
            # Validate confidence
            if not isinstance(confidence, (int, float)):
                raise InvalidRiskParameterError(f"confidence must be numeric, got {type(confidence)}")
            
            if not (0.0 <= confidence <= 1.0):
                raise InvalidRiskParameterError(f"confidence must be in [0,1], got {confidence}")
            
            # Validate consistency
            if confidence <= target_risk:
                if self.strict_mode:
                    raise InvalidRiskParameterError(
                        f"confidence ({confidence}) should be > target_risk ({target_risk})"
                    )
            
            # Validate additional parameters
            validated_kwargs = {}
            for key, value in kwargs.items():
                if key in ['window_size', 'sample_size']:
                    if not isinstance(value, int) or value <= 0:
                        raise InvalidRiskParameterError(f"{key} must be positive integer, got {value}")
                    validated_kwargs[key] = value
                elif key in ['learning_rate', 'alpha']:
                    if not isinstance(value, (int, float)) or not (0.0 < value < 1.0):
                        raise InvalidRiskParameterError(f"{key} must be in (0,1), got {value}")
                    validated_kwargs[key] = value
                else:
                    # Sanitize unknown parameters
                    validated_kwargs[key] = self._sanitize_value(value)
            
            return float(target_risk), float(confidence), validated_kwargs
            
        except Exception as e:
            raise InvalidRiskParameterError(f"Risk parameter validation failed: {e}")
    
    def _deep_copy_sanitize(self, obj: Any) -> Any:
        """Deep copy with sanitization."""
        if isinstance(obj, dict):
            return {
                self._sanitize_key(k): self._deep_copy_sanitize(v) 
                for k, v in obj.items()
            }
        elif isinstance(obj, list):
            return [self._deep_copy_sanitize(item) for item in obj]
        elif isinstance(obj, str):
            return self._sanitize_string(obj)
        else:
            return obj
    
    def _sanitize_string(self, text: str) -> str:
        """Sanitize string input for security."""
        if not isinstance(text, str):
            return str(text)
        
        # Check for dangerous patterns
        for pattern in self.dangerous_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                raise SecurityError(f"Dangerous pattern detected: {pattern}")
        
        # Basic sanitization
        text = re.sub(r'[<>"\']', '', text)  # Remove potentially dangerous chars
        text = text.strip()
        
        # Length limit
        if len(text) > 10000:
            text = text[:10000] + "..."
        
        return text
    
    def _sanitize_key(self, key: str) -> str:
        """Sanitize dictionary keys."""
        if not isinstance(key, str):
            key = str(key)
        
        # Only allow alphanumeric and safe characters
        key = re.sub(r'[^a-zA-Z0-9_.-]', '_', key)
        
        return key
    
    def _sanitize_value(self, value: Any) -> Any:
        """Sanitize arbitrary values."""
        if isinstance(value, str):
            return self._sanitize_string(value)
        elif isinstance(value, (list, tuple)):
            return [self._sanitize_value(v) for v in value]
        elif isinstance(value, dict):
            return self._deep_copy_sanitize(value)
        else:
            return value
    
    def _validate_config_structure(self, config: Dict[str, Any]) -> None:
        """Validate configuration structure."""
        # Check for required fields based on config type
        if 'target_risk' in config:
            if not isinstance(config['target_risk'], (int, float)):
                raise ConfigurationError("target_risk must be numeric")
        
        if 'confidence' in config:
            if not isinstance(config['confidence'], (int, float)):
                raise ConfigurationError("confidence must be numeric")
    
    def _validate_config_security(self, config: Dict[str, Any]) -> None:
        """Validate configuration for security threats."""
        config_str = json.dumps(config, default=str)
        
        for pattern in self.dangerous_patterns:
            if re.search(pattern, config_str, re.IGNORECASE):
                raise SecurityError(f"Dangerous pattern in config: {pattern}")
    
    def _validate_config_ranges(self, config: Dict[str, Any]) -> None:
        """Validate configuration value ranges."""
        range_validations = {
            'target_risk': (0.0, 1.0),
            'confidence': (0.0, 1.0),
            'learning_rate': (0.0, 1.0),
            'window_size': (1, 100000),
        }
        
        for key, (min_val, max_val) in range_validations.items():
            if key in config:
                value = config[key]
                if not (min_val <= value <= max_val):
                    raise ConfigurationError(
                        f"{key} must be in [{min_val}, {max_val}], got {value}"
                    )
    
    def _validate_states_robust(self, states: List[Any]) -> None:
        """Robustly validate states."""
        for i, state in enumerate(states):
            if state is None:
                raise InvalidTrajectoryError(f"State {i} is None")
            
            # Check for reasonable state dimensions
            if hasattr(state, '__len__') and len(state) > 10000:
                raise InvalidTrajectoryError(f"State {i} too large: {len(state)} dimensions")
    
    def _validate_actions_robust(self, actions: List[Any]) -> None:
        """Robustly validate actions."""
        for i, action in enumerate(actions):
            if action is None:
                raise InvalidTrajectoryError(f"Action {i} is None")
            
            # Check for reasonable action values
            if isinstance(action, (int, float)):
                if abs(action) > 1e6:
                    raise InvalidTrajectoryError(f"Action {i} value too large: {action}")
    
    def _validate_rewards_robust(self, rewards: List[Any]) -> None:
        """Robustly validate rewards."""
        for i, reward in enumerate(rewards):
            if reward is None:
                raise InvalidTrajectoryError(f"Reward {i} is None")
            
            if isinstance(reward, (int, float)):
                if abs(reward) > 1e6:
                    raise InvalidTrajectoryError(f"Reward {i} value too large: {reward}")
    
    def _validate_infos_security(self, infos: List[Dict[str, Any]]) -> None:
        """Validate info dictionaries for security."""
        for i, info in enumerate(infos):
            if not isinstance(info, dict):
                continue
            
            info_str = json.dumps(info, default=str)
            for pattern in self.dangerous_patterns:
                if re.search(pattern, info_str, re.IGNORECASE):
                    raise SecurityError(f"Dangerous pattern in info {i}: {pattern}")
    
    def _compute_hash(self, obj: Any) -> str:
        """Compute hash of object for validation history."""
        obj_str = json.dumps(obj, sort_keys=True, default=str)
        return hashlib.sha256(obj_str.encode()).hexdigest()[:16]
    
    def get_validation_report(self) -> Dict[str, Any]:
        """Get validation history report."""
        total_validations = len(self.validation_history)
        successful = sum(1 for v in self.validation_history if v['status'] == 'success')
        
        return {
            'total_validations': total_validations,
            'successful_validations': successful,
            'success_rate': successful / total_validations if total_validations > 0 else 0,
            'recent_validations': self.validation_history[-10:],  # Last 10
        }


# Global robust validator instance
_robust_validator = RobustValidator()

def validate_config_robust(config: Dict[str, Any]) -> Dict[str, Any]:
    """Global function for robust config validation."""
    return _robust_validator.validate_and_sanitize_config(config)

def validate_trajectory_robust(trajectory: Any) -> bool:
    """Global function for robust trajectory validation."""
    return _robust_validator.validate_trajectory_robust(trajectory)

def validate_risk_params_robust(target_risk: float, confidence: float, **kwargs) -> Tuple[float, float, Dict[str, Any]]:
    """Global function for robust risk parameter validation."""
    return _robust_validator.validate_risk_parameters_comprehensive(target_risk, confidence, **kwargs)