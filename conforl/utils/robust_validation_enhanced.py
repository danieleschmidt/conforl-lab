"""Enhanced robust validation with comprehensive error handling."""

import numpy as np
import os
import sys
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
import json
import hashlib
import time

from .errors import ValidationError, SecurityError, ConfigurationError
from .logging import get_logger
from .security import SecurityContext, sanitize_input

logger = get_logger(__name__)

class RobustValidator:
    """Comprehensive validation system for all ConfoRL components."""
    
    def __init__(self, strict_mode: bool = True, log_validation: bool = True):
        """Initialize robust validator.
        
        Args:
            strict_mode: If True, raises exceptions on validation failures
            log_validation: If True, logs all validation attempts
        """
        self.strict_mode = strict_mode
        self.log_validation = log_validation
        self.validation_cache = {}
        self.failed_validations = []
        
    def validate_environment_config(self, env_config: Dict[str, Any]) -> bool:
        """Validate environment configuration with detailed checks."""
        if self.log_validation:
            logger.info(f"Validating environment config: {list(env_config.keys())}")
            
        try:
            # Required fields
            required_fields = ['env_name']
            for field in required_fields:
                if field not in env_config:
                    raise ValidationError(f"Missing required field: {field}")
            
            # Validate environment name
            env_name = env_config['env_name']
            if not isinstance(env_name, str) or not env_name.strip():
                raise ValidationError(f"Invalid environment name: {env_name}")
            
            # Security validation
            if not SecurityContext.is_safe_string(env_name):
                raise SecurityError(f"Environment name contains unsafe characters: {env_name}")
            
            # Optional field validation
            if 'max_episode_steps' in env_config:
                steps = env_config['max_episode_steps']
                if not isinstance(steps, int) or steps <= 0:
                    raise ValidationError(f"Invalid max_episode_steps: {steps}")
            
            return True
            
        except (ValidationError, SecurityError) as e:
            self._handle_validation_error(f"Environment config validation failed: {e}")
            return False
        except Exception as e:
            self._handle_validation_error(f"Unexpected error in environment validation: {e}")
            return False
    
    def validate_algorithm_parameters(self, params: Dict[str, Any]) -> bool:
        """Validate algorithm parameters with type checking."""
        if self.log_validation:
            logger.info("Validating algorithm parameters")
            
        try:
            # Learning rate validation
            if 'learning_rate' in params:
                lr = params['learning_rate']
                if not isinstance(lr, (int, float)) or lr <= 0 or lr > 1:
                    raise ValidationError(f"Invalid learning rate: {lr}")
            
            # Batch size validation  
            if 'batch_size' in params:
                batch_size = params['batch_size']
                if not isinstance(batch_size, int) or batch_size <= 0:
                    raise ValidationError(f"Invalid batch size: {batch_size}")
            
            # Buffer size validation
            if 'buffer_size' in params:
                buffer_size = params['buffer_size']
                if not isinstance(buffer_size, int) or buffer_size < 1000:
                    raise ValidationError(f"Invalid buffer size: {buffer_size}")
            
            # Risk parameters
            if 'target_risk' in params:
                risk = params['target_risk']
                if not isinstance(risk, (int, float)) or risk < 0 or risk > 1:
                    raise ValidationError(f"Invalid target risk: {risk}")
            
            return True
            
        except ValidationError as e:
            self._handle_validation_error(f"Algorithm parameter validation failed: {e}")
            return False
        except Exception as e:
            self._handle_validation_error(f"Unexpected error in parameter validation: {e}")
            return False
    
    def validate_trajectory_data_robust(self, trajectory_data: Any) -> bool:
        """Robust validation of trajectory data with comprehensive checks."""
        if self.log_validation:
            logger.info("Validating trajectory data")
            
        try:
            # Check if trajectory data exists
            if trajectory_data is None:
                raise ValidationError("Trajectory data is None")
            
            # Check required attributes
            required_attrs = ['states', 'actions', 'rewards']
            for attr in required_attrs:
                if not hasattr(trajectory_data, attr):
                    raise ValidationError(f"Missing trajectory attribute: {attr}")
            
            states = getattr(trajectory_data, 'states', [])
            actions = getattr(trajectory_data, 'actions', [])
            rewards = getattr(trajectory_data, 'rewards', [])
            
            # Check data consistency
            if len(states) == 0:
                raise ValidationError("Empty trajectory states")
            
            if len(actions) != len(states):
                raise ValidationError(f"State-action length mismatch: {len(states)} vs {len(actions)}")
            
            if len(rewards) != len(states):
                raise ValidationError(f"State-reward length mismatch: {len(states)} vs {len(rewards)}")
            
            # Validate data types and ranges
            for i, (state, action, reward) in enumerate(zip(states, actions, rewards)):
                # State validation
                if state is None:
                    raise ValidationError(f"None state at index {i}")
                
                # Action validation
                if action is None:
                    raise ValidationError(f"None action at index {i}")
                
                # Reward validation
                if not isinstance(reward, (int, float)) or np.isnan(reward) or np.isinf(reward):
                    raise ValidationError(f"Invalid reward at index {i}: {reward}")
            
            return True
            
        except ValidationError as e:
            self._handle_validation_error(f"Trajectory validation failed: {e}")
            return False
        except Exception as e:
            self._handle_validation_error(f"Unexpected error in trajectory validation: {e}")
            return False
    
    def validate_file_path_secure(self, file_path: Union[str, Path]) -> bool:
        """Secure file path validation preventing directory traversal."""
        if self.log_validation:
            logger.info(f"Validating file path: {file_path}")
            
        try:
            if not file_path:
                raise ValidationError("Empty file path")
            
            path = Path(file_path).resolve()
            
            # Check for directory traversal attempts
            if '..' in str(path):
                raise SecurityError(f"Directory traversal attempt detected: {file_path}")
            
            # Check for suspicious characters
            suspicious_chars = ['<', '>', '|', '&', ';']
            if any(char in str(path) for char in suspicious_chars):
                raise SecurityError(f"Suspicious characters in path: {file_path}")
            
            # Validate path length
            if len(str(path)) > 1000:
                raise ValidationError(f"Path too long: {len(str(path))} characters")
            
            return True
            
        except (ValidationError, SecurityError) as e:
            self._handle_validation_error(f"File path validation failed: {e}")
            return False
        except Exception as e:
            self._handle_validation_error(f"Unexpected error in path validation: {e}")
            return False
    
    def validate_model_checkpoint(self, checkpoint_path: Union[str, Path]) -> bool:
        """Validate model checkpoint integrity and security."""
        if self.log_validation:
            logger.info(f"Validating model checkpoint: {checkpoint_path}")
            
        try:
            if not self.validate_file_path_secure(checkpoint_path):
                return False
            
            path = Path(checkpoint_path)
            
            # Check file exists
            if not path.exists():
                raise ValidationError(f"Checkpoint file does not exist: {checkpoint_path}")
            
            # Check file size (reasonable bounds)
            file_size = path.stat().st_size
            if file_size > 1_000_000_000:  # 1GB limit
                raise ValidationError(f"Checkpoint file too large: {file_size} bytes")
            
            if file_size < 100:  # Minimum reasonable size
                raise ValidationError(f"Checkpoint file too small: {file_size} bytes")
            
            # TODO: Add checkpoint content validation (model structure, etc.)
            return True
            
        except ValidationError as e:
            self._handle_validation_error(f"Checkpoint validation failed: {e}")
            return False
        except Exception as e:
            self._handle_validation_error(f"Unexpected error in checkpoint validation: {e}")
            return False
    
    def validate_memory_usage(self, threshold_mb: float = 1000.0) -> bool:
        """Validate current memory usage is within reasonable bounds."""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / (1024 * 1024)
            
            if self.log_validation:
                logger.info(f"Current memory usage: {memory_mb:.2f} MB")
            
            if memory_mb > threshold_mb:
                raise ValidationError(f"Memory usage too high: {memory_mb:.2f} MB > {threshold_mb} MB")
            
            return True
            
        except ImportError:
            if self.log_validation:
                logger.warning("psutil not available, skipping memory validation")
            return True
        except ValidationError as e:
            self._handle_validation_error(f"Memory validation failed: {e}")
            return False
        except Exception as e:
            self._handle_validation_error(f"Unexpected error in memory validation: {e}")
            return False
    
    def _handle_validation_error(self, error_msg: str):
        """Handle validation errors based on strict mode."""
        self.failed_validations.append({
            'timestamp': time.time(),
            'error': error_msg
        })
        
        if self.log_validation:
            logger.error(error_msg)
        
        if self.strict_mode:
            raise ValidationError(error_msg)
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of validation attempts and failures."""
        return {
            'total_failures': len(self.failed_validations),
            'recent_failures': self.failed_validations[-10:],  # Last 10 failures
            'strict_mode': self.strict_mode
        }

# Global validator instance
robust_validator = RobustValidator()

# Convenience functions
def validate_env_config(config: Dict[str, Any]) -> bool:
    """Validate environment configuration."""
    return robust_validator.validate_environment_config(config)

def validate_algorithm_params(params: Dict[str, Any]) -> bool:
    """Validate algorithm parameters."""
    return robust_validator.validate_algorithm_parameters(params)

def validate_trajectory_robust(trajectory: Any) -> bool:
    """Validate trajectory data robustly."""
    return robust_validator.validate_trajectory_data_robust(trajectory)

def validate_file_path(path: Union[str, Path]) -> bool:
    """Validate file path securely."""
    return robust_validator.validate_file_path_secure(path)

def validate_checkpoint(path: Union[str, Path]) -> bool:
    """Validate model checkpoint."""
    return robust_validator.validate_model_checkpoint(path)

def check_memory_usage(threshold_mb: float = 1000.0) -> bool:
    """Check memory usage."""
    return robust_validator.validate_memory_usage(threshold_mb)