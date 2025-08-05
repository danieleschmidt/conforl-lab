"""Base conformal RL agent with safety guarantees."""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Union, Any
import gymnasium as gym

from ..core.types import RiskCertificate, TrajectoryData, StateType, ActionType
from ..risk.controllers import AdaptiveRiskController
from ..risk.measures import RiskMeasure, SafetyViolationRisk


class ConformalRLAgent(ABC):
    """Base class for conformal RL agents with safety guarantees."""
    
    def __init__(
        self,
        env: gym.Env,
        risk_controller: Optional[AdaptiveRiskController] = None,
        risk_measure: Optional[RiskMeasure] = None,
        learning_rate: float = 3e-4,
        buffer_size: int = int(1e6),
        device: str = "auto"
    ):
        """Initialize conformal RL agent.
        
        Args:
            env: Gymnasium environment
            risk_controller: Risk controller for safety guarantees
            risk_measure: Risk measure to optimize
            learning_rate: Learning rate for optimization
            buffer_size: Size of replay buffer
            device: Device for computation ('cpu', 'cuda', 'auto')
        """
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        
        # Risk control components
        self.risk_controller = risk_controller or AdaptiveRiskController()
        self.risk_measure = risk_measure or SafetyViolationRisk()
        
        # Training parameters
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.device = self._setup_device(device)
        
        # Training state
        self.total_timesteps = 0
        self.episode_count = 0
        self.training_history = []
        
        # Initialize base RL algorithm
        self._setup_algorithm()
    
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
        """Predict action with risk-aware policy.
        
        Args:
            state: Environment state
            deterministic: Whether to use deterministic policy
            return_risk_certificate: Whether to return risk certificate
            
        Returns:
            Action (and risk certificate if requested)
        """
        # Get base action from RL algorithm
        action = self._predict_base(state, deterministic)
        
        if return_risk_certificate:
            certificate = self.risk_controller.get_certificate()
            return action, certificate
        
        return action
    
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