"""ConformaPPO: Proximal Policy Optimization with conformal risk control."""

import numpy as np
from typing import Optional
import gymnasium as gym

from .base import ConformalRLAgent
from ..risk.controllers import AdaptiveRiskController
from ..risk.measures import RiskMeasure
from ..core.types import StateType, ActionType


class ConformaPPO(ConformalRLAgent):
    """PPO algorithm with conformal risk guarantees."""
    
    def __init__(
        self,
        env: gym.Env,
        risk_controller: Optional[AdaptiveRiskController] = None,
        risk_measure: Optional[RiskMeasure] = None,
        learning_rate: float = 3e-4,
        buffer_size: int = int(2048),
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        device: str = "auto"
    ):
        """Initialize ConformaPPO agent.
        
        Args:
            env: Gymnasium environment
            risk_controller: Risk controller for safety guarantees
            risk_measure: Risk measure to optimize
            learning_rate: Learning rate for optimization
            buffer_size: Size of rollout buffer
            batch_size: Batch size for training
            n_epochs: Number of epochs per update
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_range: PPO clipping range
            device: Device for computation
        """
        # PPO-specific parameters
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        
        # Initialize base agent
        super().__init__(
            env=env,
            risk_controller=risk_controller,
            risk_measure=risk_measure,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            device=device
        )
    
    def _setup_algorithm(self) -> None:
        """Setup PPO algorithm components."""
        obs_dim = self.observation_space.shape[0] if hasattr(self.observation_space, 'shape') else 1
        act_dim = self.action_space.shape[0] if hasattr(self.action_space, 'shape') else 1
        
        # Initialize simple policy for demonstration
        self.policy_mean = np.zeros(act_dim)
        self.policy_log_std = np.log(np.ones(act_dim))
        self.value_baseline = 0.0
        
        # PPO rollout buffer
        self.rollout_buffer = []
        self.current_rollout = []
        
        print(f"ConformaPPO initialized for env with obs_dim={obs_dim}, act_dim={act_dim}")
    
    def _predict_base(
        self,
        state: StateType,
        deterministic: bool = False
    ) -> ActionType:
        """Base PPO prediction without risk considerations.
        
        Args:
            state: Environment state
            deterministic: Whether to use deterministic policy
            
        Returns:
            Action from PPO policy
        """
        if deterministic:
            action = self.policy_mean
        else:
            std = np.exp(self.policy_log_std)
            action = np.random.normal(self.policy_mean, std)
        
        # Clip to action space bounds
        if hasattr(self.action_space, 'low') and hasattr(self.action_space, 'high'):
            action = np.clip(action, self.action_space.low, self.action_space.high)
        
        return action
    
    def _update_algorithm(
        self,
        state: StateType,
        action: ActionType,
        reward: float,
        next_state: StateType,
        done: bool
    ) -> None:
        """Update PPO algorithm.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode termination flag
        """
        # Store transition in current rollout
        transition = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'log_prob': self._compute_log_prob(action)
        }
        self.current_rollout.append(transition)
        
        # Update when rollout is complete or buffer is full
        if done or len(self.current_rollout) >= self.buffer_size:
            self._process_rollout()
            self.current_rollout = []
    
    def _compute_log_prob(self, action: ActionType) -> float:
        """Compute log probability of action under current policy."""
        std = np.exp(self.policy_log_std)
        diff = action - self.policy_mean
        log_prob = -0.5 * np.sum((diff / std) ** 2 + 2 * self.policy_log_std + np.log(2 * np.pi))
        return float(log_prob)
    
    def _process_rollout(self) -> None:
        """Process completed rollout and update policy."""
        if len(self.current_rollout) == 0:
            return
        
        # Compute advantages using GAE
        advantages = self._compute_advantages()
        
        # Add to rollout buffer
        for i, transition in enumerate(self.current_rollout):
            transition['advantage'] = advantages[i]
            transition['return'] = advantages[i] + self.value_baseline
        
        self.rollout_buffer.extend(self.current_rollout)
        
        # Keep buffer manageable
        if len(self.rollout_buffer) > self.buffer_size * 2:
            self.rollout_buffer = self.rollout_buffer[-self.buffer_size:]
        
        # Update policy if enough data
        if len(self.rollout_buffer) >= self.buffer_size:
            self._update_policy()
    
    def _compute_advantages(self) -> np.ndarray:
        """Compute GAE advantages for current rollout."""
        rewards = [t['reward'] for t in self.current_rollout]
        
        # Simple advantage computation (not full GAE)
        returns = []
        running_return = 0
        for reward in reversed(rewards):
            running_return = reward + self.gamma * running_return
            returns.insert(0, running_return)
        
        advantages = np.array(returns) - self.value_baseline
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        return advantages
    
    def _update_policy(self) -> None:
        """Update PPO policy using rollout buffer."""
        # Sample from rollout buffer
        n_samples = min(len(self.rollout_buffer), self.buffer_size)
        sample_indices = np.random.choice(len(self.rollout_buffer), n_samples, replace=False)
        
        # Compute policy gradients (simplified)
        policy_gradient = np.zeros_like(self.policy_mean)
        advantages = []
        
        for idx in sample_indices:
            transition = self.rollout_buffer[idx]
            action = transition['action']
            advantage = transition['advantage']
            advantages.append(advantage)
            
            # Simplified policy gradient
            if advantage > 0:
                policy_gradient += 0.001 * (action - self.policy_mean)
        
        # Update policy parameters
        self.policy_mean += self.learning_rate * policy_gradient
        
        # Update value baseline
        if advantages:
            self.value_baseline = 0.9 * self.value_baseline + 0.1 * np.mean(advantages)
        
        # Risk-aware policy adjustment
        certificate = self.risk_controller.get_certificate()
        if certificate.risk_bound > self.risk_controller.target_risk * 1.2:
            # Reduce policy variance when risk is high
            self.policy_log_std = np.maximum(self.policy_log_std - 0.01, np.log(0.1))
        else:
            # Allow more exploration when risk is acceptable
            self.policy_log_std = np.minimum(self.policy_log_std + 0.005, np.log(1.0))
    
    def get_ppo_info(self) -> dict:
        """Get PPO-specific information.
        
        Returns:
            Dict with PPO algorithm details
        """
        return {
            "algorithm": "ConformaPPO",
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_range": self.clip_range,
            "policy_mean": self.policy_mean.tolist(),
            "policy_std": np.exp(self.policy_log_std).tolist(),
            "value_baseline": self.value_baseline,
            "buffer_size": len(self.rollout_buffer),
            "current_rollout_size": len(self.current_rollout)
        }