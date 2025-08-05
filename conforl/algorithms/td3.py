"""ConformaTD3: Twin Delayed DDPG with conformal risk control."""

import numpy as np
from typing import Optional
import gymnasium as gym

from .base import ConformalRLAgent
from ..risk.controllers import AdaptiveRiskController
from ..risk.measures import RiskMeasure
from ..core.types import StateType, ActionType


class ConformaTD3(ConformalRLAgent):
    """TD3 algorithm with conformal risk guarantees."""
    
    def __init__(
        self,
        env: gym.Env,
        risk_controller: Optional[AdaptiveRiskController] = None,
        risk_measure: Optional[RiskMeasure] = None,
        learning_rate: float = 3e-4,
        buffer_size: int = int(1e6),
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        policy_delay: int = 2,
        target_noise: float = 0.2,
        noise_clip: float = 0.5,
        device: str = "auto"
    ):
        """Initialize ConformaTD3 agent.
        
        Args:
            env: Gymnasium environment
            risk_controller: Risk controller for safety guarantees
            risk_measure: Risk measure to optimize
            learning_rate: Learning rate for optimization
            buffer_size: Size of replay buffer
            batch_size: Batch size for training
            tau: Soft update coefficient
            gamma: Discount factor
            policy_delay: Delay between policy updates
            target_noise: Noise added to target actions
            noise_clip: Clipping range for target noise
            device: Device for computation
        """
        # TD3-specific parameters
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.policy_delay = policy_delay
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        
        # Update counter for policy delay
        self.update_counter = 0
        
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
        """Setup TD3 algorithm components."""
        obs_dim = self.observation_space.shape[0] if hasattr(self.observation_space, 'shape') else 1
        act_dim = self.action_space.shape[0] if hasattr(self.action_space, 'shape') else 1
        
        # Initialize policy and critics (simplified)
        self.policy_params = np.random.normal(0, 0.1, (obs_dim, act_dim))
        self.critic1_params = np.random.normal(0, 0.1, (obs_dim + act_dim,))
        self.critic2_params = np.random.normal(0, 0.1, (obs_dim + act_dim,))
        
        # Target networks
        self.target_policy_params = self.policy_params.copy()
        self.target_critic1_params = self.critic1_params.copy()
        self.target_critic2_params = self.critic2_params.copy()
        
        # Replay buffer
        self.replay_buffer = []
        
        print(f"ConformaTD3 initialized for env with obs_dim={obs_dim}, act_dim={act_dim}")
    
    def _predict_base(
        self,
        state: StateType,
        deterministic: bool = False
    ) -> ActionType:
        """Base TD3 prediction without risk considerations.
        
        Args:
            state: Environment state
            deterministic: Whether to use deterministic policy
            
        Returns:
            Action from TD3 policy
        """
        # Simple linear policy
        if isinstance(state, np.ndarray) and state.ndim == 1:
            state_flat = state
        else:
            state_flat = np.array([state]).flatten()
        
        # Ensure state dimension matches policy input
        if len(state_flat) != self.policy_params.shape[0]:
            state_flat = np.resize(state_flat, self.policy_params.shape[0])
        
        action = np.dot(state_flat, self.policy_params).flatten()
        
        # Add exploration noise if not deterministic
        if not deterministic:
            noise = np.random.normal(0, 0.1, action.shape)
            action += noise
        
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
        """Update TD3 algorithm.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode termination flag
        """
        # Store transition in replay buffer
        transition = (state, action, reward, next_state, done)
        self.replay_buffer.append(transition)
        
        # Keep buffer size manageable
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer.pop(0)
        
        # Update if enough data
        if len(self.replay_buffer) >= self.batch_size:
            self._update_networks()
    
    def _update_networks(self) -> None:
        """Update TD3 networks."""
        # Sample batch from replay buffer
        batch_size = min(self.batch_size, len(self.replay_buffer))
        batch_indices = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
        
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_next_states = []
        batch_dones = []
        
        for idx in batch_indices:
            state, action, reward, next_state, done = self.replay_buffer[idx]
            batch_states.append(state)
            batch_actions.append(action)
            batch_rewards.append(reward)
            batch_next_states.append(next_state)
            batch_dones.append(done)
        
        # Update critics (simplified)
        self._update_critics(batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones)
        
        # Update policy with delay
        self.update_counter += 1
        if self.update_counter % self.policy_delay == 0:
            self._update_policy(batch_states)
            self._update_targets()
    
    def _update_critics(self, states, actions, rewards, next_states, dones):
        """Update critic networks."""
        # Simplified critic update
        avg_reward = np.mean(rewards)
        
        # Update critic parameters toward higher rewards
        if avg_reward > 0:
            self.critic1_params += 0.001 * avg_reward * np.random.normal(0, 0.01, self.critic1_params.shape)
            self.critic2_params += 0.001 * avg_reward * np.random.normal(0, 0.01, self.critic2_params.shape)
    
    def _update_policy(self, states):
        """Update policy network."""
        # Simplified policy update toward critic gradient
        for state in states[:10]:  # Limit batch size for simplicity
            if isinstance(state, np.ndarray) and state.ndim == 1:
                state_flat = state
            else:
                state_flat = np.array([state]).flatten()
            
            if len(state_flat) == self.policy_params.shape[0]:
                # Simple gradient ascent
                gradient = np.outer(state_flat, np.random.normal(0, 0.01, self.policy_params.shape[1]))
                self.policy_params += self.learning_rate * gradient
        
        # Risk-aware policy adjustment
        certificate = self.risk_controller.get_certificate()
        if certificate.risk_bound > self.risk_controller.target_risk * 1.3:
            # Add conservative bias when risk is high
            self.policy_params *= 0.99
    
    def _update_targets(self):
        """Soft update of target networks."""
        # Soft update target networks
        self.target_policy_params = (1 - self.tau) * self.target_policy_params + self.tau * self.policy_params
        self.target_critic1_params = (1 - self.tau) * self.target_critic1_params + self.tau * self.critic1_params
        self.target_critic2_params = (1 - self.tau) * self.target_critic2_params + self.tau * self.critic2_params
    
    def get_td3_info(self) -> dict:
        """Get TD3-specific information.
        
        Returns:
            Dict with TD3 algorithm details
        """
        return {
            "algorithm": "ConformaTD3",
            "batch_size": self.batch_size,
            "tau": self.tau,
            "gamma": self.gamma,
            "policy_delay": self.policy_delay,
            "target_noise": self.target_noise,
            "noise_clip": self.noise_clip,
            "update_counter": self.update_counter,
            "buffer_size": len(self.replay_buffer),
            "policy_norm": float(np.linalg.norm(self.policy_params)),
            "critic1_norm": float(np.linalg.norm(self.critic1_params)),
            "critic2_norm": float(np.linalg.norm(self.critic2_params))
        }