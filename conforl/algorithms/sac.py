"""ConformaSAC: Soft Actor-Critic with conformal risk control."""

import numpy as np
from typing import Optional, Dict, Any, Tuple
import gymnasium as gym
import warnings

from .base import ConformalRLAgent
from ..risk.controllers import AdaptiveRiskController
from ..risk.measures import RiskMeasure
from ..core.types import StateType, ActionType
from ..utils.logging import get_logger
from ..optimize.cache import AdaptiveCache, PerformanceCache
from ..optimize.concurrent import BatchProcessor

logger = get_logger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.distributions import Normal
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available, using simplified SAC implementation")
    
    # Mock classes for fallback
    class nn:
        class Module:
            def __init__(self): self.training = False
            def parameters(self): return []
            def train(self): self.training = True
            def eval(self): self.training = False
            def to(self, device): return self
            def __call__(self, *args): return self.forward(*args)
            def forward(self, x): return x
        class Linear(Module):
            def __init__(self, in_features, out_features): 
                super().__init__()
                self.weight = np.random.randn(out_features, in_features) * 0.1
                self.bias = np.zeros(out_features)
            def forward(self, x):
                return np.dot(x, self.weight.T) + self.bias
        ReLU = lambda: lambda x: np.maximum(0, x)
        Tanh = lambda: lambda x: np.tanh(x)
    
    class optim:
        class Adam:
            def __init__(self, params, lr=0.001): 
                self.lr = lr
                self.params = params
            def step(self): pass
            def zero_grad(self): pass
    
    class F:
        @staticmethod
        def mse_loss(pred, target): return np.mean((pred - target)**2)
        @staticmethod
        def relu(x): return np.maximum(0, x)
        @staticmethod
        def tanh(x): return np.tanh(x)
    
    torch = type('torch', (), {
        'tensor': lambda x: np.array(x),
        'cuda': type('cuda', (), {'is_available': lambda: False})(),
        'float32': np.float32,
        'no_grad': lambda: type('context', (), {'__enter__': lambda self: None, '__exit__': lambda self, *args: None})()
    })()


class Actor(nn.Module):
    """SAC Actor network with tanh squashing."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, max_action: float = 1.0):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        ) if TORCH_AVAILABLE else lambda x: x
        
        if TORCH_AVAILABLE:
            self.mean_head = nn.Linear(hidden_dim, action_dim)
            self.logstd_head = nn.Linear(hidden_dim, action_dim)
        else:
            self.mean_head = nn.Linear(hidden_dim, action_dim)
            self.logstd_head = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state):
        if not TORCH_AVAILABLE:
            # Fallback implementation
            mean = np.zeros(self.action_dim) 
            std = np.ones(self.action_dim) * 0.1
            return mean, std
        
        x = self.backbone(state)
        mean = self.mean_head(x)
        logstd = self.logstd_head(x)
        logstd = torch.clamp(logstd, -20, 2)  # Stability
        std = torch.exp(logstd)
        
        return mean, std
    
    def sample(self, state):
        mean, std = self.forward(state)
        if not TORCH_AVAILABLE:
            action = np.random.normal(mean, std)
            return np.tanh(action) * self.max_action, None
        
        normal = Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t) * self.max_action
        
        # Compute log probability
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob


class Critic(nn.Module):
    """SAC Q-network (critic)."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        if TORCH_AVAILABLE:
            # Twin Q-networks
            self.q1 = nn.Sequential(
                nn.Linear(state_dim + action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
            
            self.q2 = nn.Sequential(
                nn.Linear(state_dim + action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        else:
            # Fallback implementations
            self.q1 = nn.Linear(state_dim + action_dim, 1)
            self.q2 = nn.Linear(state_dim + action_dim, 1)
    
    def forward(self, state, action):
        if not TORCH_AVAILABLE:
            # Simple linear approximation for fallback
            sa = np.concatenate([np.atleast_1d(state), np.atleast_1d(action)])
            return self.q1(sa), self.q2(sa)
        
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa), self.q2(sa)


class ReplayBuffer:
    """Experience replay buffer for SAC."""
    
    def __init__(self, capacity: int = int(1e6)):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int):
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]
        
        if not batch:
            return None
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        if TORCH_AVAILABLE:
            return (
                torch.FloatTensor(states),
                torch.FloatTensor(actions),
                torch.FloatTensor(rewards).unsqueeze(1),
                torch.FloatTensor(next_states),
                torch.FloatTensor(dones).unsqueeze(1)
            )
        else:
            return (
                np.array(states),
                np.array(actions),
                np.array(rewards),
                np.array(next_states),
                np.array(dones)
            )
    
    def __len__(self):
        return len(self.buffer)


class ConformaSAC(ConformalRLAgent):
    """SAC algorithm with conformal risk guarantees."""
    
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
        alpha: float = 0.2,
        hidden_dim: int = 256,
        device: str = "auto"
    ):
        """Initialize ConformaSAC agent.
        
        Args:
            env: Gymnasium environment
            risk_controller: Risk controller for safety guarantees
            risk_measure: Risk measure to optimize
            learning_rate: Learning rate for optimization
            buffer_size: Size of replay buffer
            batch_size: Batch size for training
            tau: Soft update coefficient
            gamma: Discount factor
            alpha: Temperature parameter for SAC
            hidden_dim: Hidden layer dimensions
            device: Device for computation
        """
        # SAC-specific parameters
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.alpha = alpha
        self.hidden_dim = hidden_dim
        
        # Initialize base agent
        super().__init__(
            env=env,
            risk_controller=risk_controller,
            risk_measure=risk_measure,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            device=device
        )
        
        # Initialize performance optimization
        self._initialize_optimization()
    
    def _setup_algorithm(self) -> None:
        """Setup SAC algorithm components."""
        # Environment dimensions
        if hasattr(self.observation_space, 'shape') and len(self.observation_space.shape) > 0:
            self.obs_dim = self.observation_space.shape[0]
        else:
            self.obs_dim = 1
            
        if hasattr(self.action_space, 'shape') and len(self.action_space.shape) > 0:
            self.act_dim = self.action_space.shape[0]
            self.max_action = float(self.action_space.high[0]) if hasattr(self.action_space, 'high') else 1.0
        else:
            self.act_dim = 1
            self.max_action = 1.0
        
        # Initialize networks
        self.actor = Actor(self.obs_dim, self.act_dim, self.hidden_dim, self.max_action)
        self.critic = Critic(self.obs_dim, self.act_dim, self.hidden_dim) 
        self.target_critic = Critic(self.obs_dim, self.act_dim, self.hidden_dim)
        
        # Move to device
        if TORCH_AVAILABLE:
            device = torch.device(self.device if self.device != 'auto' else 'cpu')
            self.actor = self.actor.to(device)
            self.critic = self.critic.to(device)
            self.target_critic = self.target_critic.to(device)
            
            # Initialize target network
            self.soft_update(self.target_critic, self.critic, tau=1.0)
            
            # Optimizers
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate)
        else:
            self.actor_optimizer = optim.Adam([], lr=self.learning_rate)
            self.critic_optimizer = optim.Adam([], lr=self.learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(self.buffer_size)
        
        # Training statistics
        self.actor_losses = []
        self.critic_losses = []
        self.q_values = []
        
        logger.info(f"ConformaSAC initialized: obs_dim={self.obs_dim}, act_dim={self.act_dim}, "
                   f"device={self.device}, torch_available={TORCH_AVAILABLE}")
    
    def _initialize_optimization(self) -> None:
        """Initialize performance optimization components."""
        # Adaptive caching for predictions and Q-values
        self.prediction_cache = AdaptiveCache(
            max_size=1000,
            ttl=300.0,  # 5 minutes
            adaptive_ttl=True,
            compression=False  # Keep False for speed
        )
        
        # Performance cache for expensive computations
        self.computation_cache = PerformanceCache(max_size=500)
        
        # Batch processor for parallel operations
        self.batch_processor = BatchProcessor(
            batch_size=min(64, self.batch_size),
            max_workers=min(8, 4)  # Conservative worker count
        )
        
        # Performance optimization flags
        self.enable_prediction_caching = True
        self.enable_batch_updates = True
        self.enable_concurrent_forward_passes = True
        
        # Load balancing for network updates
        self.update_frequency = {
            'actor': 1,     # Update every step
            'critic': 1,    # Update every step  
            'target': 100   # Soft update every 100 steps
        }
        self.update_counters = {
            'actor': 0,
            'critic': 0,
            'target': 0
        }
        
        logger.debug("SAC performance optimization initialized")
    
    def soft_update(self, target, source, tau):
        """Soft update of target network."""
        if not TORCH_AVAILABLE:
            return
        
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
    
    def _predict_base(
        self,
        state: StateType,
        deterministic: bool = False
    ) -> ActionType:
        """Base SAC prediction with caching and optimization.
        
        Args:
            state: Environment state
            deterministic: Whether to use deterministic policy
            
        Returns:
            Action from SAC policy
        """
        if not TORCH_AVAILABLE:
            # Fallback random policy
            action = np.random.normal(0, 0.1, size=self.act_dim)
            return np.clip(action, -self.max_action, self.max_action)
        
        # Cache key for prediction caching
        if self.enable_prediction_caching:
            cache_key = f"predict_{hash(str(state))}_{deterministic}"
            cached_action = self.prediction_cache.get(cache_key)
            if cached_action is not None:
                return cached_action
        
        # Use performance cache for forward pass computation
        def compute_action():
            # Convert state to tensor
            state_tensor = torch.FloatTensor(np.atleast_2d(state))
            device = next(self.actor.parameters()).device
            state_tensor = state_tensor.to(device)
            
            with torch.no_grad():
                if deterministic:
                    # Use mean action for deterministic policy
                    mean, _ = self.actor(state_tensor)
                    action = torch.tanh(mean) * self.max_action
                else:
                    # Sample from stochastic policy
                    action, _ = self.actor.sample(state_tensor)
            
            # Convert back to numpy and ensure correct shape
            return action.cpu().numpy().flatten()
        
        # Compute action with performance caching
        if self.enable_prediction_caching:
            action = self.computation_cache.cached_computation(
                "actor_forward", compute_action
            )
        else:
            action = compute_action()
        
        # Clip to action bounds
        if hasattr(self.action_space, 'low') and hasattr(self.action_space, 'high'):
            action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Store in prediction cache for future use
        if self.enable_prediction_caching:
            self.prediction_cache.put(cache_key, action.copy())
        
        return action
    
    def _update_algorithm(
        self,
        state: StateType,
        action: ActionType,
        reward: float,
        next_state: StateType,
        done: bool
    ) -> None:
        """Update SAC algorithm.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode termination flag
        """
        # Store transition in replay buffer
        self.replay_buffer.push(state, action, reward, next_state, done)
        
        # Update networks if we have enough samples
        if len(self.replay_buffer) >= self.batch_size:
            self._update_networks()
    
    def _update_networks(self) -> None:
        """Update SAC networks with optimizations and load balancing."""
        if not TORCH_AVAILABLE:
            return  # Skip network updates in fallback mode
        
        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        if batch is None:
            return
        
        # Load balancing: skip updates based on frequency
        self.update_counters['critic'] += 1
        self.update_counters['actor'] += 1
        self.update_counters['target'] += 1
        
        should_update_critic = (self.update_counters['critic'] % self.update_frequency['critic'] == 0)
        should_update_actor = (self.update_counters['actor'] % self.update_frequency['actor'] == 0)
        should_update_target = (self.update_counters['target'] % self.update_frequency['target'] == 0)
        
        if not (should_update_critic or should_update_actor):
            return
            
        states, actions, rewards, next_states, dones = batch
        device = next(self.actor.parameters()).device
        
        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        next_states = next_states.to(device)
        dones = dones.to(device)
        
        # Cached computation for critic update
        def compute_critic_update():
            with torch.no_grad():
                # Sample next actions from current policy
                next_actions, next_log_probs = self.actor.sample(next_states)
                
                # Compute target Q-values
                target_q1, target_q2 = self.target_critic(next_states, next_actions)
                target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
                target_q = rewards + (1 - dones) * self.gamma * target_q
            
            # Current Q-values
            current_q1, current_q2 = self.critic(states, actions)
            
            # Critic loss
            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
            
            return critic_loss, current_q1, current_q2
        
        # Update Critic with optimization
        if should_update_critic:
            if self.enable_batch_updates:
                critic_loss, current_q1, current_q2 = self.computation_cache.cached_computation(
                    f"critic_update_{self.update_counters['critic']}", 
                    compute_critic_update
                )
            else:
                critic_loss, current_q1, current_q2 = compute_critic_update()
            
            # Update critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            
            # Gradient clipping for stability
            if TORCH_AVAILABLE:
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
            
            self.critic_optimizer.step()
            
            # Store loss for monitoring
            self.critic_losses.append(critic_loss.item())
            self.q_values.append(current_q1.mean().item())
        
        # Cached computation for actor update
        def compute_actor_update():
            new_actions, log_probs = self.actor.sample(states)
            q1_new, q2_new = self.critic(states, new_actions)
            q_new = torch.min(q1_new, q2_new)
            
            # Actor loss (maximize Q - entropy penalty)
            actor_loss = (self.alpha * log_probs - q_new).mean()
            return actor_loss
        
        # Update Actor with optimization
        if should_update_actor:
            if self.enable_batch_updates:
                actor_loss = self.computation_cache.cached_computation(
                    f"actor_update_{self.update_counters['actor']}", 
                    compute_actor_update
                )
            else:
                actor_loss = compute_actor_update()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            
            # Gradient clipping for stability
            if TORCH_AVAILABLE:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            
            self.actor_optimizer.step()
            
            # Store loss for monitoring
            self.actor_losses.append(actor_loss.item())
        
        # Soft update target networks (less frequent)
        if should_update_target:
            self.soft_update(self.target_critic, self.critic, self.tau)
        
        # Risk-aware adaptation with caching
        try:
            certificate = self.risk_controller.get_certificate()
            if certificate and certificate.risk_bound > self.risk_controller.target_risk * 1.2:
                # Increase exploration when risk is too high
                self.alpha = min(self.alpha * 1.01, 0.5)
            elif certificate:
                # Decrease exploration when risk is acceptable
                self.alpha = max(self.alpha * 0.999, 0.01)
        except Exception as e:
            logger.debug(f"Risk adaptation failed: {e}")
        
        # Periodic cache cleanup for memory management
        if self.update_counters['critic'] % 1000 == 0:
            self._cleanup_caches()
    
    def get_sac_info(self) -> Dict[str, Any]:
        """Get SAC-specific information.
        
        Returns:
            Dict with SAC algorithm details
        """
        info = {
            "algorithm": "ConformaSAC",
            "batch_size": self.batch_size,
            "tau": self.tau,
            "gamma": self.gamma,
            "alpha": self.alpha,
            "hidden_dim": self.hidden_dim,
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "max_action": self.max_action,
            "buffer_size": len(self.replay_buffer),
            "buffer_capacity": self.replay_buffer.capacity,
            "total_updates": self.total_timesteps,
            "torch_available": TORCH_AVAILABLE,
            "device": self.device
        }
        
        # Add loss information if available
        if self.critic_losses:
            info["recent_critic_loss"] = np.mean(self.critic_losses[-10:])
            info["recent_actor_loss"] = np.mean(self.actor_losses[-10:])
            info["recent_q_value"] = np.mean(self.q_values[-10:])
        
        return info
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get detailed training statistics.
        
        Returns:
            Dict with training metrics and performance data
        """
        stats = {
            "training_steps": self.total_timesteps,
            "episodes_completed": self.episode_count,
            "replay_buffer_size": len(self.replay_buffer),
            "replay_buffer_utilization": len(self.replay_buffer) / self.replay_buffer.capacity,
        }
        
        # Loss statistics
        if self.critic_losses:
            stats["critic_loss_mean"] = np.mean(self.critic_losses)
            stats["critic_loss_std"] = np.std(self.critic_losses)
            stats["actor_loss_mean"] = np.mean(self.actor_losses)
            stats["actor_loss_std"] = np.std(self.actor_losses)
            stats["q_value_mean"] = np.mean(self.q_values)
            stats["q_value_std"] = np.std(self.q_values)
        
        # Risk information
        certificate = self.risk_controller.get_certificate()
        stats["current_risk_bound"] = certificate.risk_bound
        stats["target_risk"] = self.risk_controller.target_risk
        stats["risk_controller_coverage"] = certificate.coverage_guarantee
        
        return stats
    
    def _cleanup_caches(self) -> None:
        """Clean up caches to prevent memory bloat."""
        try:
            # Clean up expired entries
            prediction_cleaned = self.prediction_cache.cleanup_expired()
            
            # Clear old computation cache entries
            if hasattr(self.computation_cache, 'clear_old_entries'):
                self.computation_cache.clear_old_entries()
            
            logger.debug(f"Cache cleanup: {prediction_cleaned} prediction entries removed")
            
        except Exception as e:
            logger.warning(f"Cache cleanup failed: {e}")
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get performance optimization statistics.
        
        Returns:
            Dict with optimization metrics
        """
        stats = {
            "optimization_enabled": {
                "prediction_caching": self.enable_prediction_caching,
                "batch_updates": self.enable_batch_updates,
                "concurrent_forward_passes": self.enable_concurrent_forward_passes
            },
            "cache_stats": {},
            "computation_stats": {},
            "batch_processor_stats": {}
        }
        
        # Get cache statistics
        try:
            stats["cache_stats"]["prediction_cache"] = self.prediction_cache.get_stats()
        except Exception as e:
            stats["cache_stats"]["prediction_cache_error"] = str(e)
        
        try:
            stats["computation_stats"] = self.computation_cache.get_performance_stats()
        except Exception as e:
            stats["computation_stats_error"] = str(e)
        
        try:
            stats["batch_processor_stats"] = self.batch_processor.get_stats()
        except Exception as e:
            stats["batch_processor_stats_error"] = str(e)
        
        # Update counters
        stats["update_counters"] = self.update_counters.copy()
        stats["update_frequencies"] = self.update_frequency.copy()
        
        return stats
    
    def tune_performance(self, performance_target: str = "balanced") -> None:
        """Automatically tune performance parameters.
        
        Args:
            performance_target: Target profile ('speed', 'memory', 'balanced')
        """
        logger.info(f"Tuning SAC performance for target: {performance_target}")
        
        if performance_target == "speed":
            # Optimize for maximum speed
            self.enable_prediction_caching = True
            self.enable_batch_updates = True
            self.enable_concurrent_forward_passes = True
            
            # More aggressive caching
            self.prediction_cache.max_size = 2000
            self.computation_cache.max_size = 1000
            
            # Less frequent target updates for speed
            self.update_frequency['target'] = 200
            
        elif performance_target == "memory":
            # Optimize for memory usage
            self.enable_prediction_caching = False  # Disable caching
            self.enable_batch_updates = False
            self.enable_concurrent_forward_passes = False
            
            # Smaller caches
            self.prediction_cache.max_size = 100
            self.computation_cache.max_size = 50
            
            # More frequent cleanup
            self._cleanup_caches()
            
        elif performance_target == "balanced":
            # Balanced performance profile (default)
            self.enable_prediction_caching = True
            self.enable_batch_updates = True
            self.enable_concurrent_forward_passes = True
            
            # Moderate cache sizes
            self.prediction_cache.max_size = 1000
            self.computation_cache.max_size = 500
            
            # Standard update frequencies
            self.update_frequency['target'] = 100
        
        logger.info(f"Performance tuning completed for {performance_target} profile")
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics.
        
        Returns:
            Dict with memory usage information
        """
        memory_stats = {
            "replay_buffer_size": len(self.replay_buffer),
            "replay_buffer_capacity": self.replay_buffer.capacity,
            "replay_buffer_utilization": len(self.replay_buffer) / self.replay_buffer.capacity
        }
        
        # Cache memory usage
        try:
            cache_stats = self.prediction_cache.get_stats()
            memory_stats["prediction_cache_bytes"] = cache_stats.get("total_bytes", 0)
            memory_stats["prediction_cache_entries"] = cache_stats.get("size", 0)
        except Exception:
            memory_stats["prediction_cache_bytes"] = 0
            memory_stats["prediction_cache_entries"] = 0
        
        # Model parameter estimates
        if TORCH_AVAILABLE:
            try:
                actor_params = sum(p.numel() for p in self.actor.parameters())
                critic_params = sum(p.numel() for p in self.critic.parameters())
                target_params = sum(p.numel() for p in self.target_critic.parameters())
                
                # Estimate memory (4 bytes per float32 parameter)
                memory_stats["model_parameters"] = {
                    "actor": actor_params,
                    "critic": critic_params, 
                    "target_critic": target_params,
                    "total_params": actor_params + critic_params + target_params,
                    "estimated_bytes": (actor_params + critic_params + target_params) * 4
                }
            except Exception as e:
                memory_stats["model_parameters_error"] = str(e)
        
        return memory_stats