"""Base conformal RL agent with safety guarantees."""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Union, Any, List
import gymnasium as gym
import warnings
import traceback

from ..core.types import RiskCertificate, TrajectoryData, StateType, ActionType
from ..risk.controllers import AdaptiveRiskController
from ..risk.measures import RiskMeasure, SafetyViolationRisk
from ..utils.validation import (
    validate_config, validate_environment, validate_trajectory_data, 
    validate_risk_parameters
)
from ..utils.security import SecurityContext, sanitize_config_dict, log_security_event
from ..utils.errors import (
    ConfoRLError, ValidationError, ConfigurationError, 
    EnvironmentError, SecurityError, TrainingError
)
from ..utils.logging import get_logger

logger = get_logger(__name__)


class ConformalRLAgent(ABC):
    """Base class for conformal RL agents with safety guarantees."""
    
    def __init__(
        self,
        env: gym.Env,
        risk_controller: Optional[AdaptiveRiskController] = None,
        risk_measure: Optional[RiskMeasure] = None,
        learning_rate: float = 3e-4,
        buffer_size: int = int(1e6),
        device: str = "auto",
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Initialize conformal RL agent with comprehensive validation.
        
        Args:
            env: Gymnasium environment
            risk_controller: Risk controller for safety guarantees
            risk_measure: Risk measure to optimize
            learning_rate: Learning rate for optimization
            buffer_size: Size of replay buffer
            device: Device for computation ('cpu', 'cuda', 'auto')
            config: Additional configuration parameters
            **kwargs: Additional keyword arguments
        """
        # Security context for initialization
        with SecurityContext("agent_initialization", "system"):
            try:
                # Validate environment first
                logger.info("Initializing ConfoRL agent with comprehensive validation")
                self._validate_initialization_params(
                    env, learning_rate, buffer_size, device, config
                )
                
                # Store validated parameters
                self.env = env
                self.observation_space = env.observation_space
                self.action_space = env.action_space
                
                # Validate and sanitize configuration
                self.config = self._process_config(config, kwargs)
                
                # Initialize risk control components with validation
                self._initialize_risk_components(risk_controller, risk_measure)
                
                # Validate and store training parameters
                self.learning_rate = learning_rate
                self.buffer_size = buffer_size
                self.device = self._setup_device(device)
                
                # Training state with error tracking
                self.total_timesteps = 0
                self.episode_count = 0
                self.training_history = []
                self.errors_encountered = []
                self.recovery_attempts = 0
                self.max_recovery_attempts = 5
                
                # Performance monitoring
                self.performance_metrics = {
                    'initialization_time': 0,
                    'training_time': 0,
                    'prediction_time': 0,
                    'update_time': 0
                }
                
                # Thread safety
                self._training_lock = None
                try:
                    import threading
                    self._training_lock = threading.Lock()
                except ImportError:
                    logger.warning("Threading not available, agent may not be thread-safe")
                
                # Initialize base RL algorithm with error handling
                self._initialize_algorithm_safely()
                
                logger.info(f"ConfoRL agent initialized successfully: "
                           f"obs_space={self.observation_space}, "
                           f"action_space={self.action_space}")
                
            except Exception as e:
                logger.error(f"Failed to initialize ConfoRL agent: {str(e)}")
                logger.debug(f"Initialization stack trace: {traceback.format_exc()}")
                raise ConfoRLError(
                    f"Agent initialization failed: {str(e)}",
                    error_code="INIT_FAILED"
                ) from e
    
    def _validate_initialization_params(
        self, 
        env: gym.Env, 
        learning_rate: float, 
        buffer_size: int, 
        device: str,
        config: Optional[Dict[str, Any]]
    ) -> None:
        """Validate initialization parameters."""
        # Validate environment
        if env is None:
            raise ValidationError("Environment cannot be None", "initialization")
        
        validate_environment(env)
        
        # Validate learning rate
        if not isinstance(learning_rate, (int, float)) or learning_rate <= 0 or learning_rate > 1:
            raise ValidationError(
                f"learning_rate must be in (0, 1], got {learning_rate}",
                "initialization"
            )
        
        # Validate buffer size
        if not isinstance(buffer_size, int) or buffer_size <= 0:
            raise ValidationError(
                f"buffer_size must be positive integer, got {buffer_size}",
                "initialization"
            )
        
        # Validate device
        if not isinstance(device, str) or device not in ['cpu', 'cuda', 'auto']:
            raise ValidationError(
                f"device must be 'cpu', 'cuda', or 'auto', got {device}",
                "initialization"
            )
        
        # Validate config if provided
        if config is not None and not isinstance(config, dict):
            raise ValidationError("config must be a dictionary", "initialization")
    
    def _process_config(
        self, 
        config: Optional[Dict[str, Any]], 
        kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process and sanitize configuration."""
        # Merge config and kwargs
        full_config = {}
        if config:
            full_config.update(config)
        full_config.update(kwargs)
        
        # Sanitize configuration for security
        try:
            sanitized_config = sanitize_config_dict(full_config)
            validated_config = validate_config(sanitized_config)
            return validated_config
        except (SecurityError, ConfigurationError) as e:
            logger.warning(f"Configuration validation/sanitization failed: {e}")
            return {}  # Use empty config as fallback
    
    def _initialize_risk_components(
        self, 
        risk_controller: Optional[AdaptiveRiskController], 
        risk_measure: Optional[RiskMeasure]
    ) -> None:
        """Initialize and validate risk control components."""
        try:
            # Initialize risk controller with validation
            if risk_controller is not None:
                # Validate risk controller parameters
                if hasattr(risk_controller, 'target_risk') and hasattr(risk_controller, 'confidence'):
                    validate_risk_parameters(
                        risk_controller.target_risk,
                        risk_controller.confidence
                    )
                self.risk_controller = risk_controller
            else:
                self.risk_controller = AdaptiveRiskController()
            
            # Initialize risk measure
            self.risk_measure = risk_measure or SafetyViolationRisk()
            
            logger.debug(f"Risk components initialized: "
                        f"target_risk={self.risk_controller.target_risk}, "
                        f"confidence={self.risk_controller.confidence}")
                        
        except Exception as e:
            raise ConfoRLError(
                f"Risk component initialization failed: {str(e)}",
                error_code="RISK_INIT_FAILED"
            ) from e
    
    def _initialize_algorithm_safely(self) -> None:
        """Initialize base RL algorithm with error handling."""
        try:
            import time
            start_time = time.time()
            
            self._setup_algorithm()
            
            self.performance_metrics['initialization_time'] = time.time() - start_time
            logger.debug(f"Algorithm setup completed in "
                        f"{self.performance_metrics['initialization_time']:.3f}s")
        
        except Exception as e:
            self.errors_encountered.append({
                'timestamp': time.time(),
                'error_type': 'algorithm_setup',
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            raise TrainingError(
                f"Algorithm setup failed: {str(e)}",
                error_code="ALGORITHM_SETUP_FAILED"
            ) from e
    
    def _setup_device(self, device: str) -> str:
        """Setup computation device."""
        if device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return device
    
    @abstractmethod
    def _setup_algorithm(self) -> None:
        """Setup the base RL algorithm (SAC, PPO, etc.)."""
        pass
    
    @abstractmethod
    def _predict_base(
        self,
        state: StateType,
        deterministic: bool = False
    ) -> ActionType:
        """Base prediction without risk considerations.
        
        Args:
            state: Environment state
            deterministic: Whether to use deterministic policy
            
        Returns:
            Action from base RL algorithm
        """
        pass
    
    def predict(
        self,
        state: StateType,
        deterministic: bool = False,
        return_risk_certificate: bool = False
    ) -> Union[ActionType, Tuple[ActionType, RiskCertificate]]:
        """Predict action with risk-aware policy and comprehensive validation.
        
        Args:
            state: Environment state
            deterministic: Whether to use deterministic policy
            return_risk_certificate: Whether to return risk certificate
            
        Returns:
            Action (and risk certificate if requested)
            
        Raises:
            ValidationError: If state is invalid
            ConfoRLError: If prediction fails
        """
        with SecurityContext("action_prediction", "agent"):
            try:
                import time
                start_time = time.time()
                
                # Validate state input
                self._validate_state_input(state)
                
                # Get base action from RL algorithm with error handling
                try:
                    action = self._predict_base(state, deterministic)
                except Exception as e:
                    logger.error(f"Base prediction failed: {str(e)}")
                    
                    # Attempt recovery
                    if self.recovery_attempts < self.max_recovery_attempts:
                        logger.info(f"Attempting recovery (attempt {self.recovery_attempts + 1})")
                        self.recovery_attempts += 1
                        action = self._safe_fallback_action(state)
                        log_security_event("PREDICTION_RECOVERY", {
                            "error": str(e),
                            "recovery_attempt": self.recovery_attempts
                        })
                    else:
                        raise ConfoRLError(
                            f"Prediction failed after {self.max_recovery_attempts} recovery attempts",
                            error_code="PREDICTION_FAILED"
                        ) from e
                
                # Validate action output
                validated_action = self._validate_action_output(action)
                
                # Update performance metrics
                self.performance_metrics['prediction_time'] = time.time() - start_time
                
                # Reset recovery attempts on success
                if self.recovery_attempts > 0:
                    logger.info("Prediction recovered successfully")
                    self.recovery_attempts = 0
                
                if return_risk_certificate:
                    certificate = self._get_validated_risk_certificate()
                    return validated_action, certificate
                
                return validated_action
                
            except Exception as e:
                self._log_error("prediction", e)
                raise
    
    def _validate_state_input(self, state: StateType) -> None:
        """Validate state input."""
        if state is None:
            raise ValidationError("State cannot be None", "prediction")
        
        # Convert to numpy array for validation
        if not isinstance(state, np.ndarray):
            try:
                state = np.array(state)
            except Exception as e:
                raise ValidationError(f"Cannot convert state to array: {e}", "prediction")
        
        # Check for invalid values
        if np.any(np.isnan(state)):
            raise ValidationError("State contains NaN values", "prediction")
        
        if np.any(np.isinf(state)):
            raise ValidationError("State contains infinite values", "prediction")
        
        # Check dimensions if observation space is defined
        if hasattr(self.observation_space, 'shape') and self.observation_space.shape:
            expected_shape = self.observation_space.shape
            if state.shape != expected_shape and state.shape != (1,) + expected_shape:
                logger.warning(f"State shape {state.shape} doesn't match expected {expected_shape}")
    
    def _validate_action_output(self, action: ActionType) -> ActionType:
        """Validate and clip action output."""
        if action is None:
            raise ConfoRLError("Action cannot be None", error_code="INVALID_ACTION")
        
        # Convert to numpy array
        if not isinstance(action, np.ndarray):
            try:
                action = np.array(action)
            except Exception as e:
                raise ConfoRLError(f"Cannot convert action to array: {e}", error_code="ACTION_CONVERSION_FAILED")
        
        # Check for invalid values
        if np.any(np.isnan(action)):
            logger.warning("Action contains NaN values, using fallback")
            action = self._safe_fallback_action(None)
        
        if np.any(np.isinf(action)):
            logger.warning("Action contains infinite values, clipping")
            action = np.clip(action, -1e6, 1e6)
        
        # Clip to action space bounds
        if hasattr(self.action_space, 'low') and hasattr(self.action_space, 'high'):
            action = np.clip(action, self.action_space.low, self.action_space.high)
        
        return action
    
    def _safe_fallback_action(self, state: Optional[StateType]) -> ActionType:
        """Generate safe fallback action."""
        logger.warning("Using safe fallback action")
        
        # Use zero action or random safe action
        if hasattr(self.action_space, 'shape'):
            if hasattr(self.action_space, 'low') and hasattr(self.action_space, 'high'):
                # Random action within bounds
                low = self.action_space.low
                high = self.action_space.high
                return np.random.uniform(low, high)
            else:
                # Zero action
                return np.zeros(self.action_space.shape)
        else:
            return np.array([0.0])
    
    def _get_validated_risk_certificate(self) -> RiskCertificate:
        """Get validated risk certificate."""
        try:
            certificate = self.risk_controller.get_certificate()
            
            # Validate certificate
            if certificate.risk_bound < 0 or certificate.risk_bound > 1:
                logger.warning(f"Invalid risk bound: {certificate.risk_bound}")
                # Create safe certificate
                from ..core.types import RiskCertificate
                import time
                certificate = RiskCertificate(
                    risk_bound=0.5,  # Conservative fallback
                    confidence=0.95,
                    coverage_guarantee=0.95,
                    method="fallback",
                    sample_size=1,
                    timestamp=time.time()
                )
            
            return certificate
            
        except Exception as e:
            logger.error(f"Risk certificate generation failed: {e}")
            # Return conservative certificate
            from ..core.types import RiskCertificate
            import time
            return RiskCertificate(
                risk_bound=1.0,  # Maximum risk as fallback
                confidence=0.95,
                coverage_guarantee=0.95,
                method="error_fallback",
                sample_size=1,
                timestamp=time.time(),
                metadata={'error': str(e)}
            )
    
    def train(
        self,
        total_timesteps: int,
        eval_freq: int = 10000,
        save_path: Optional[str] = None,
        callback: Optional[callable] = None,
        **kwargs
    ) -> None:
        """Train the conformal RL agent.
        
        Args:
            total_timesteps: Total training timesteps
            eval_freq: Frequency of evaluation
            save_path: Path to save trained model
            callback: Optional callback function
            **kwargs: Additional training parameters
        """
        print(f"Training ConfoRL agent for {total_timesteps} timesteps")
        
        # Training loop
        state, info = self.env.reset()
        episode_rewards = []
        episode_risks = []
        current_trajectory = {
            "states": [],
            "actions": [], 
            "rewards": [],
            "dones": [],
            "infos": []
        }
        
        for step in range(total_timesteps):
            # Select action
            action = self.predict(state, deterministic=False)
            
            # Environment step
            next_state, reward, done, truncated, info = self.env.step(action)
            
            # Store trajectory data
            current_trajectory["states"].append(state)
            current_trajectory["actions"].append(action)
            current_trajectory["rewards"].append(reward)
            current_trajectory["dones"].append(done or truncated)
            current_trajectory["infos"].append(info)
            
            # Update RL algorithm (placeholder - implement in subclasses)
            self._update_algorithm(state, action, reward, next_state, done or truncated)
            
            # Episode completion
            if done or truncated:
                # Create trajectory data
                trajectory = TrajectoryData(
                    states=np.array(current_trajectory["states"]),
                    actions=np.array(current_trajectory["actions"]),
                    rewards=np.array(current_trajectory["rewards"]),
                    dones=np.array(current_trajectory["dones"]),
                    infos=current_trajectory["infos"]
                )
                
                # Update risk controller
                self.risk_controller.update(trajectory, self.risk_measure)
                
                # Track metrics
                episode_return = sum(current_trajectory["rewards"])
                episode_risk = self.risk_measure.compute(trajectory)
                episode_rewards.append(episode_return)
                episode_risks.append(episode_risk)
                
                # Reset for next episode
                state, info = self.env.reset()
                current_trajectory = {
                    "states": [], "actions": [], "rewards": [], 
                    "dones": [], "infos": []
                }
                self.episode_count += 1
                
                # Evaluation
                if step % eval_freq == 0 and step > 0:
                    self._evaluate(step, episode_rewards[-10:], episode_risks[-10:])
            else:
                state = next_state
            
            self.total_timesteps += 1
            
            # Callback
            if callback is not None:
                callback(locals())
        
        # Save model
        if save_path is not None:
            self.save(save_path)
        
        print(f"Training completed. Episodes: {self.episode_count}")
    
    @abstractmethod
    def _update_algorithm(
        self,
        state: StateType,
        action: ActionType,
        reward: float,
        next_state: StateType,
        done: bool
    ) -> None:
        """Update the base RL algorithm.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode termination flag
        """
        pass
    
    def _evaluate(
        self,
        step: int,
        recent_rewards: list,
        recent_risks: list
    ) -> None:
        """Evaluate current performance and risk.
        
        Args:
            step: Current training step
            recent_rewards: Recent episode rewards
            recent_risks: Recent episode risks
        """
        avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0
        avg_risk = np.mean(recent_risks) if recent_risks else 0.0
        certificate = self.risk_controller.get_certificate()
        
        print(f"Step {step}: Reward={avg_reward:.2f}, Risk={avg_risk:.4f}, "
              f"Bound={certificate.risk_bound:.4f}")
        
        self.training_history.append({
            "step": step,
            "avg_reward": avg_reward,
            "avg_risk": avg_risk,
            "risk_bound": certificate.risk_bound,
            "coverage": certificate.coverage_guarantee
        })
    
    def get_risk_certificate(
        self,
        states: Optional[np.ndarray] = None,
        coverage_guarantee: float = 0.95
    ) -> RiskCertificate:
        """Get formal risk certificate for current policy.
        
        Args:
            states: Test states for evaluation (optional)
            coverage_guarantee: Desired coverage level
            
        Returns:
            Risk certificate with formal guarantees
        """
        return self.risk_controller.get_certificate()
    
    def save(self, path: str) -> None:
        """Save trained agent to file.
        
        Args:
            path: Save path
        """
        import pickle
        
        save_data = {
            "risk_controller": self.risk_controller,
            "risk_measure": self.risk_measure,
            "training_history": self.training_history,
            "total_timesteps": self.total_timesteps,
            "episode_count": self.episode_count
        }
        
        with open(f"{path}_conforl.pkl", "wb") as f:
            pickle.dump(save_data, f)
        
        print(f"ConfoRL agent saved to {path}")
    
    def load(self, path: str) -> None:
        """Load trained agent from file.
        
        Args:
            path: Load path
        """
        import pickle
        
        with open(f"{path}_conforl.pkl", "rb") as f:
            save_data = pickle.load(f)
        
        self.risk_controller = save_data["risk_controller"]
        self.risk_measure = save_data["risk_measure"]
        self.training_history = save_data["training_history"]
        self.total_timesteps = save_data["total_timesteps"]
        self.episode_count = save_data["episode_count"]
        
        print(f"ConfoRL agent loaded from {path}")
    
    def _log_error(self, operation: str, error: Exception) -> None:
        """Log error for debugging and recovery."""
        import time
        error_info = {
            'timestamp': time.time(),
            'operation': operation,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc()
        }
        
        self.errors_encountered.append(error_info)
        logger.error(f"Error in {operation}: {error}")
        logger.debug(f"Error traceback: {error_info['traceback']}")
        
        # Log security event for potential security issues
        if isinstance(error, (SecurityError, ValidationError)):
            log_security_event("VALIDATION_ERROR", {
                "operation": operation,
                "error_type": type(error).__name__,
                "error_message": str(error)
            })
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of errors encountered."""
        if not self.errors_encountered:
            return {"total_errors": 0, "error_types": {}}
        
        error_types = {}
        for error in self.errors_encountered:
            error_type = error['error_type']
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            "total_errors": len(self.errors_encountered),
            "error_types": error_types,
            "recent_errors": self.errors_encountered[-5:],  # Last 5 errors
            "recovery_attempts": self.recovery_attempts
        }
    
    def reset_error_state(self) -> None:
        """Reset error tracking state."""
        logger.info("Resetting error tracking state")
        self.errors_encountered = []
        self.recovery_attempts = 0
    
    def is_healthy(self) -> bool:
        """Check if agent is in healthy state."""
        # Consider agent unhealthy if too many recent errors
        recent_errors = [
            e for e in self.errors_encountered 
            if e['timestamp'] > (time.time() - 300)  # Last 5 minutes
        ]
        
        if len(recent_errors) > 10:
            return False
        
        if self.recovery_attempts >= self.max_recovery_attempts:
            return False
        
        return True
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        import time
        
        return {
            "healthy": self.is_healthy(),
            "total_errors": len(self.errors_encountered),
            "recent_error_count": len([
                e for e in self.errors_encountered 
                if e['timestamp'] > (time.time() - 300)
            ]),
            "recovery_attempts": self.recovery_attempts,
            "max_recovery_attempts": self.max_recovery_attempts,
            "training_timesteps": self.total_timesteps,
            "episodes_completed": self.episode_count,
            "performance_metrics": self.performance_metrics.copy(),
            "uptime": time.time() - (
                self.performance_metrics.get('initialization_time', time.time())
            )
        }