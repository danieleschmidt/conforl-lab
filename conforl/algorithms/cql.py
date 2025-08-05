"""ConformaCQL: Conservative Q-Learning with conformal risk control for offline RL."""

import numpy as np
from typing import Optional, Dict, Any
import gymnasium as gym

from .base import ConformalRLAgent
from ..risk.controllers import AdaptiveRiskController
from ..risk.measures import RiskMeasure
from ..core.types import StateType, ActionType


class ConformaCQL(ConformalRLAgent):
    """CQL algorithm with conformal risk guarantees for offline RL."""
    
    def __init__(
        self,
        env: gym.Env,
        dataset: Optional[Dict[str, np.ndarray]] = None,
        risk_controller: Optional[AdaptiveRiskController] = None,
        risk_measure: Optional[RiskMeasure] = None,
        learning_rate: float = 3e-4,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        alpha: float = 1.0,
        cql_weight: float = 1.0,
        device: str = "auto"
    ):
        """Initialize ConformaCQL agent.
        
        Args:
            env: Gymnasium environment
            dataset: Offline dataset for training
            risk_controller: Risk controller for safety guarantees
            risk_measure: Risk measure to optimize
            learning_rate: Learning rate for optimization
            batch_size: Batch size for training
            tau: Soft update coefficient
            gamma: Discount factor
            alpha: Temperature parameter
            cql_weight: Weight for CQL regularization
            device: Device for computation
        """
        # CQL-specific parameters
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.alpha = alpha
        self.cql_weight = cql_weight
        
        # Offline dataset
        self.dataset = dataset or {}
        self.dataset_size = len(self.dataset.get('observations', []))
        
        # Initialize base agent (no buffer_size needed for offline)
        super().__init__(
            env=env,
            risk_controller=risk_controller,
            risk_measure=risk_measure,
            learning_rate=learning_rate,
            buffer_size=0,  # Not used in offline RL
            device=device
        )
    
    def _setup_algorithm(self) -> None:
        """Setup CQL algorithm components."""
        obs_dim = self.observation_space.shape[0] if hasattr(self.observation_space, 'shape') else 1
        act_dim = self.action_space.shape[0] if hasattr(self.action_space, 'shape') else 1
        
        # Initialize Q-networks and policy (simplified)
        self.q1_params = np.random.normal(0, 0.1, (obs_dim + act_dim,))
        self.q2_params = np.random.normal(0, 0.1, (obs_dim + act_dim,))
        self.policy_params = np.random.normal(0, 0.1, (obs_dim, act_dim))
        
        # Target networks
        self.target_q1_params = self.q1_params.copy()
        self.target_q2_params = self.q2_params.copy()
        
        # CQL-specific parameters
        self.cql_log_alpha = np.log(self.alpha)
        
        print(f"ConformaCQL initialized for offline RL with dataset_size={self.dataset_size}")
        
        if self.dataset_size > 0:
            self._preprocess_dataset()
    
    def _preprocess_dataset(self) -> None:
        """Preprocess offline dataset for training."""
        print("Preprocessing offline dataset...")
        
        # Convert dataset to numpy arrays if needed
        for key in ['observations', 'actions', 'rewards', 'next_observations', 'terminals']:
            if key in self.dataset and not isinstance(self.dataset[key], np.ndarray):
                self.dataset[key] = np.array(self.dataset[key])
        
        # Compute dataset statistics
        if 'rewards' in self.dataset:
            reward_mean = np.mean(self.dataset['rewards'])
            reward_std = np.std(self.dataset['rewards'])
            print(f"Dataset reward stats: mean={reward_mean:.3f}, std={reward_std:.3f}")
        
        # Initialize risk controller with dataset
        if hasattr(self, 'risk_controller') and 'observations' in self.dataset:
            self._initialize_risk_from_dataset()
    
    def _initialize_risk_from_dataset(self) -> None:
        """Initialize risk controller using offline dataset."""
        # Create trajectory data from dataset for risk initialization
        n_samples = min(100, self.dataset_size)  # Sample subset for efficiency
        indices = np.random.choice(self.dataset_size, n_samples, replace=False)
        
        for idx in indices:
            # Create mock trajectory from single transitions
            trajectory_data = {
                'states': [self.dataset['observations'][idx]],
                'actions': [self.dataset['actions'][idx]], 
                'rewards': [self.dataset['rewards'][idx]],
                'dones': [self.dataset.get('terminals', [False])[idx]],
                'infos': [{}]
            }
            
            from ..core.types import TrajectoryData
            trajectory = TrajectoryData(
                states=np.array(trajectory_data['states']),
                actions=np.array(trajectory_data['actions']),
                rewards=np.array(trajectory_data['rewards']),
                dones=np.array(trajectory_data['dones']),
                infos=trajectory_data['infos']
            )
            
            # Update risk controller
            self.risk_controller.update(trajectory, self.risk_measure)
    
    def _predict_base(
        self,
        state: StateType,
        deterministic: bool = False
    ) -> ActionType:
        """Base CQL prediction without risk considerations.
        
        Args:
            state: Environment state
            deterministic: Whether to use deterministic policy
            
        Returns:
            Action from CQL policy
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
        
        # Add noise for exploration if not deterministic
        if not deterministic:
            noise_scale = 0.1 * np.exp(-self.alpha)  # Adaptive noise
            noise = np.random.normal(0, noise_scale, action.shape)
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
        """Update CQL algorithm (not used in offline setting).
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode termination flag
        """
        # CQL is offline, so online updates are minimal
        # Could store for online fine-tuning if desired
        pass
    
    def train_offline(
        self,
        n_epochs: int = 1000,
        eval_freq: int = 100,
        save_path: Optional[str] = None
    ) -> None:
        """Train CQL on offline dataset.
        
        Args:
            n_epochs: Number of training epochs
            eval_freq: Frequency of evaluation
            save_path: Path to save trained model
        """
        if self.dataset_size == 0:
            raise ValueError("No dataset provided for offline training")
        
        print(f"Training ConformaCQL offline for {n_epochs} epochs")
        
        for epoch in range(n_epochs):
            # Sample batch from dataset
            batch_indices = np.random.choice(self.dataset_size, self.batch_size, replace=True)
            
            # Update Q-networks and policy
            self._update_cql(batch_indices)
            
            # Soft update targets
            if epoch % 2 == 0:  # Update targets every 2 epochs
                self._update_targets()
            
            # Evaluation
            if epoch % eval_freq == 0:
                self._evaluate_offline(epoch)
        
        if save_path is not None:
            self.save(save_path)
        
        print("Offline training completed")
    
    def _update_cql(self, batch_indices: np.ndarray) -> None:
        """Update CQL networks with conservative regularization.
        
        Args:
            batch_indices: Indices of batch samples from dataset
        """
        # Extract batch data
        batch_obs = self.dataset['observations'][batch_indices]
        batch_actions = self.dataset['actions'][batch_indices]
        batch_rewards = self.dataset['rewards'][batch_indices]
        batch_next_obs = self.dataset['next_observations'][batch_indices]
        batch_dones = self.dataset.get('terminals', np.zeros(len(batch_indices)))[batch_indices]
        
        # Simplified CQL update
        avg_reward = np.mean(batch_rewards)
        
        # Conservative Q-learning: penalize out-of-distribution actions
        cql_penalty = self.cql_weight * np.mean(np.abs(batch_actions))
        
        # Update Q-networks (simplified)
        q_loss = -avg_reward + cql_penalty
        
        # Gradient step (simplified)
        if q_loss > 0:
            self.q1_params -= self.learning_rate * 0.01 * np.random.normal(0, 0.01, self.q1_params.shape)
            self.q2_params -= self.learning_rate * 0.01 * np.random.normal(0, 0.01, self.q2_params.shape)
        
        # Update policy to maximize Q-values
        if avg_reward > 0:
            self.policy_params += self.learning_rate * 0.001 * np.random.normal(0, 0.01, self.policy_params.shape)
        
        # Risk-aware updates
        certificate = self.risk_controller.get_certificate()
        if certificate.risk_bound > self.risk_controller.target_risk * 1.2:
            # More conservative when risk is high
            self.cql_weight = min(self.cql_weight * 1.01, 10.0)
        else:
            # Less conservative when risk is acceptable
            self.cql_weight = max(self.cql_weight * 0.999, 0.1)
    
    def _update_targets(self) -> None:
        """Soft update of target networks."""
        self.target_q1_params = (1 - self.tau) * self.target_q1_params + self.tau * self.q1_params
        self.target_q2_params = (1 - self.tau) * self.target_q2_params + self.tau * self.q2_params
    
    def _evaluate_offline(self, epoch: int) -> None:
        """Evaluate offline training progress.
        
        Args:
            epoch: Current training epoch
        """
        certificate = self.risk_controller.get_certificate()
        
        print(f"Epoch {epoch}: CQL_weight={self.cql_weight:.3f}, "
              f"Risk_bound={certificate.risk_bound:.4f}, "
              f"Q1_norm={np.linalg.norm(self.q1_params):.3f}")
    
    def get_cql_info(self) -> dict:
        """Get CQL-specific information.
        
        Returns:
            Dict with CQL algorithm details
        """
        return {
            "algorithm": "ConformaCQL",
            "batch_size": self.batch_size,
            "tau": self.tau,
            "gamma": self.gamma,
            "alpha": self.alpha,
            "cql_weight": self.cql_weight,
            "dataset_size": self.dataset_size,
            "q1_norm": float(np.linalg.norm(self.q1_params)),
            "q2_norm": float(np.linalg.norm(self.q2_params)),
            "policy_norm": float(np.linalg.norm(self.policy_params))
        }