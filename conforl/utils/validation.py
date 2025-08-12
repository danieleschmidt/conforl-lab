"""Input validation and data sanitization utilities."""

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    class np:
        @staticmethod
        def ndarray():
            return list  # Use list as fallback
        @staticmethod
        def isnan(data):
            return False  # Simplified
        @staticmethod
        def isinf(data):
            return False  # Simplified
        @staticmethod
        def any(data):
            return any(data) if hasattr(data, '__iter__') else False
        @staticmethod
        def all(data):
            return all(data) if hasattr(data, '__iter__') else True
        @staticmethod
        def isin(data, values):
            if hasattr(data, '__iter__'):
                return [x in values for x in data]
            return data in values
from typing import Dict, Any, Optional, Union, List
from pathlib import Path

try:
    import gymnasium as gym
    GYMNASIUM_AVAILABLE = True
except ImportError:
    GYMNASIUM_AVAILABLE = False
    # Create minimal gym interface for validation
    class gym:
        class spaces:
            class Discrete:
                pass

from .errors import ValidationError, ConfigurationError, EnvironmentError, DataError


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate configuration parameters.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Validated and sanitized configuration
        
    Raises:
        ConfigurationError: If configuration is invalid
    """
    if not isinstance(config, dict):
        raise ConfigurationError("Configuration must be a dictionary")
    
    validated_config = config.copy()
    
    # Validate common parameters
    if 'learning_rate' in config:
        lr = config['learning_rate']
        if not isinstance(lr, (int, float)) or lr <= 0 or lr > 1:
            raise ConfigurationError(
                f"learning_rate must be in (0, 1], got {lr}",
                config_key='learning_rate'
            )
    
    if 'buffer_size' in config:
        buffer_size = config['buffer_size']
        if not isinstance(buffer_size, int) or buffer_size <= 0:
            raise ConfigurationError(
                f"buffer_size must be positive integer, got {buffer_size}",
                config_key='buffer_size'
            )
    
    if 'batch_size' in config:
        batch_size = config['batch_size']
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ConfigurationError(
                f"batch_size must be positive integer, got {batch_size}",
                config_key='batch_size'
            )
    
    # Validate risk parameters
    if 'target_risk' in config:
        target_risk = config['target_risk']
        if not isinstance(target_risk, (int, float)) or not 0 < target_risk < 1:
            raise ConfigurationError(
                f"target_risk must be in (0, 1), got {target_risk}",
                config_key='target_risk'
            )
    
    if 'confidence' in config:
        confidence = config['confidence']
        if not isinstance(confidence, (int, float)) or not 0 < confidence < 1:
            raise ConfigurationError(
                f"confidence must be in (0, 1), got {confidence}",
                config_key='confidence'
            )
    
    # Validate device
    if 'device' in config:
        device = config['device']
        if not isinstance(device, str) or device not in ['cpu', 'cuda', 'auto']:
            raise ConfigurationError(
                f"device must be 'cpu', 'cuda', or 'auto', got {device}",
                config_key='device'
            )
    
    return validated_config


def validate_environment(env) -> None:
    """Validate Gymnasium environment compatibility.
    
    Args:
        env: Gymnasium environment to validate
        
    Raises:
        EnvironmentError: If environment is incompatible
    """
    if not GYMNASIUM_AVAILABLE:
        # Skip detailed validation if gymnasium not available
        if not hasattr(env, 'reset') or not hasattr(env, 'step'):
            raise EnvironmentError("Environment missing basic methods")
        return
    if not hasattr(env, 'observation_space'):
        raise EnvironmentError("Environment missing observation_space")
    
    if not hasattr(env, 'action_space'):
        raise EnvironmentError("Environment missing action_space")
    
    if not hasattr(env, 'reset'):
        raise EnvironmentError("Environment missing reset method")
    
    if not hasattr(env, 'step'):
        raise EnvironmentError("Environment missing step method")
    
    # Validate observation space
    obs_space = env.observation_space
    if hasattr(obs_space, 'shape'):
        if len(obs_space.shape) == 0:
            raise EnvironmentError("Observation space has invalid shape")
    
    # Validate action space
    action_space = env.action_space
    if hasattr(action_space, 'shape'):
        if len(action_space.shape) == 0 and not isinstance(action_space, gym.spaces.Discrete):
            raise EnvironmentError("Action space has invalid shape")
    
    # Test reset functionality
    try:
        state, info = env.reset()
        if state is None:
            raise EnvironmentError("Environment reset returns None state")
    except Exception as e:
        raise EnvironmentError(f"Environment reset failed: {str(e)}")


def validate_dataset(dataset: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Validate offline RL dataset.
    
    Args:
        dataset: Dataset dictionary with keys like 'observations', 'actions', etc.
        
    Returns:
        Validated dataset
        
    Raises:
        DataError: If dataset is invalid
    """
    if not isinstance(dataset, dict):
        raise DataError("Dataset must be a dictionary")
    
    required_keys = ['observations', 'actions', 'rewards']
    for key in required_keys:
        if key not in dataset:
            raise DataError(f"Dataset missing required key: {key}")
    
    # Validate data consistency
    obs_data = dataset['observations']
    action_data = dataset['actions']
    reward_data = dataset['rewards']
    
    if NUMPY_AVAILABLE:
        if not isinstance(obs_data, np.ndarray):
            raise DataError("observations must be numpy array")
        
        if not isinstance(action_data, np.ndarray):
            raise DataError("actions must be numpy array")
        
        if not isinstance(reward_data, np.ndarray):
            raise DataError("rewards must be numpy array")
    else:
        # Use list-based validation when numpy not available
        if not isinstance(obs_data, list):
            raise DataError("observations must be list")
        
        if not isinstance(action_data, list):
            raise DataError("actions must be list")
        
        if not isinstance(reward_data, list):
            raise DataError("rewards must be list")
    
    # Check data lengths match
    n_obs = len(obs_data)
    n_actions = len(action_data)
    n_rewards = len(reward_data)
    
    if not (n_obs == n_actions == n_rewards):
        raise DataError(
            f"Dataset length mismatch: obs={n_obs}, actions={n_actions}, rewards={n_rewards}"
        )
    
    if n_obs == 0:
        raise DataError("Dataset is empty")
    
    # Validate data ranges
    if NUMPY_AVAILABLE:
        if np.any(np.isnan(obs_data)):
            raise DataError("observations contain NaN values")
        
        if np.any(np.isnan(action_data)):
            raise DataError("actions contain NaN values")
        
        if np.any(np.isnan(reward_data)):
            raise DataError("rewards contain NaN values")
        
        if np.any(np.isinf(obs_data)):
            raise DataError("observations contain infinite values")
        
        if np.any(np.isinf(action_data)):
            raise DataError("actions contain infinite values")
        
        if np.any(np.isinf(reward_data)):
            raise DataError("rewards contain infinite values")
    # When numpy not available, skip NaN/inf checks for simplicity
    
    # Optional validation for next_observations and terminals
    if 'next_observations' in dataset:
        next_obs = dataset['next_observations']
        if len(next_obs) != n_obs:
            raise DataError("next_observations length mismatch")
        if np.any(np.isnan(next_obs)) or np.any(np.isinf(next_obs)):
            raise DataError("next_observations contain invalid values")
    
    if 'terminals' in dataset:
        terminals = dataset['terminals']
        if len(terminals) != n_obs:
            raise DataError("terminals length mismatch")
        if not np.all(np.isin(terminals, [0, 1, True, False])):
            raise DataError("terminals must be boolean or 0/1 values")
    
    return dataset


def validate_trajectory_data(
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    dones: np.ndarray
) -> None:
    """Validate trajectory data arrays.
    
    Args:
        states: State sequence
        actions: Action sequence
        rewards: Reward sequence
        dones: Done flags sequence
        
    Raises:
        ValidationError: If trajectory data is invalid
    """
    # Check types
    if NUMPY_AVAILABLE:
        if not isinstance(states, np.ndarray):
            raise ValidationError("states must be numpy array", "trajectory")
        
        if not isinstance(actions, np.ndarray):
            raise ValidationError("actions must be numpy array", "trajectory")
        
        if not isinstance(rewards, np.ndarray):
            raise ValidationError("rewards must be numpy array", "trajectory")
        
        if not isinstance(dones, np.ndarray):
            raise ValidationError("dones must be numpy array", "trajectory")
    else:
        # Use list-based validation when numpy not available
        if not isinstance(states, list):
            raise ValidationError("states must be list", "trajectory")
        
        if not isinstance(actions, list):
            raise ValidationError("actions must be list", "trajectory")
        
        if not isinstance(rewards, list):
            raise ValidationError("rewards must be list", "trajectory")
        
        if not isinstance(dones, list):
            raise ValidationError("dones must be list", "trajectory")
    
    # Check lengths
    n_states = len(states)
    n_actions = len(actions)
    n_rewards = len(rewards)
    n_dones = len(dones)
    
    if not (n_states == n_actions == n_rewards == n_dones):
        raise ValidationError(
            f"Trajectory length mismatch: states={n_states}, actions={n_actions}, "
            f"rewards={n_rewards}, dones={n_dones}",
            "trajectory"
        )
    
    if n_states == 0:
        raise ValidationError("Empty trajectory", "trajectory")
    
    # Check for invalid values
    if np.any(np.isnan(states)) or np.any(np.isinf(states)):
        raise ValidationError("states contain invalid values", "trajectory")
    
    if np.any(np.isnan(actions)) or np.any(np.isinf(actions)):
        raise ValidationError("actions contain invalid values", "trajectory")
    
    if np.any(np.isnan(rewards)) or np.any(np.isinf(rewards)):
        raise ValidationError("rewards contain invalid values", "trajectory")


def validate_risk_parameters(
    target_risk: float,
    confidence: float,
    coverage: Optional[float] = None
) -> None:
    """Validate risk control parameters.
    
    Args:
        target_risk: Target risk level
        confidence: Confidence level
        coverage: Optional coverage level
        
    Raises:
        ValidationError: If parameters are invalid
    """
    if not isinstance(target_risk, (int, float)):
        raise ValidationError("target_risk must be numeric", "risk_params")
    
    if not 0 < target_risk < 1:
        raise ValidationError(
            f"target_risk must be in (0, 1), got {target_risk}",
            "risk_params"
        )
    
    if not isinstance(confidence, (int, float)):
        raise ValidationError("confidence must be numeric", "risk_params")
    
    if not 0 < confidence < 1:
        raise ValidationError(
            f"confidence must be in (0, 1), got {confidence}",
            "risk_params"
        )
    
    if coverage is not None:
        if not isinstance(coverage, (int, float)):
            raise ValidationError("coverage must be numeric", "risk_params")
        
        if not 0 < coverage < 1:
            raise ValidationError(
                f"coverage must be in (0, 1), got {coverage}",
                "risk_params"
            )


def validate_file_paths(paths: List[Union[str, Path]]) -> List[Path]:
    """Validate and sanitize file paths.
    
    Args:
        paths: List of file paths to validate
        
    Returns:
        List of validated Path objects
        
    Raises:
        ValidationError: If any path is invalid
    """
    validated_paths = []
    
    for i, path in enumerate(paths):
        try:
            path_obj = Path(path)
            
            # Check for path traversal attempts
            if '..' in str(path_obj):
                raise ValidationError(
                    f"Path traversal detected in path {i}: {path}",
                    "file_path"
                )
            
            # Convert to absolute path for safety
            abs_path = path_obj.resolve()
            validated_paths.append(abs_path)
            
        except Exception as e:
            raise ValidationError(
                f"Invalid path at index {i}: {path} - {str(e)}",
                "file_path"
            )
    
    return validated_paths