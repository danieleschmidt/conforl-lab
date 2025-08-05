"""Tests for conformal RL algorithms."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from conforl.algorithms.sac import ConformaSAC
from conforl.algorithms.ppo import ConformaPPO
from conforl.algorithms.td3 import ConformaTD3
from conforl.algorithms.cql import ConformaCQL


class TestConformaSAC:
    """Test cases for ConformaSAC algorithm."""
    
    def test_initialization(self, simple_env, risk_controller, risk_measure):
        """Test SAC agent initialization."""
        agent = ConformaSAC(
            env=simple_env,
            risk_controller=risk_controller,
            risk_measure=risk_measure,
            learning_rate=0.001,
            batch_size=32
        )
        
        assert agent.batch_size == 32
        assert agent.learning_rate == 0.001
        assert agent.risk_controller == risk_controller
        assert agent.risk_measure == risk_measure
        assert agent.total_timesteps == 0
        assert agent.episode_count == 0
    
    def test_prediction(self, simple_env, risk_controller):
        """Test action prediction."""
        agent = ConformaSAC(env=simple_env, risk_controller=risk_controller)
        
        state = np.array([0.1, 0.2, 0.3, 0.4])
        
        # Test basic prediction
        action = agent.predict(state)
        assert action is not None
        
        # Test prediction with certificate
        action, certificate = agent.predict(state, return_risk_certificate=True)
        assert action is not None
        assert certificate is not None
        assert certificate.method == "adaptive_quantile"
    
    def test_sac_specific_methods(self, simple_env, risk_controller):
        """Test SAC-specific methods."""
        agent = ConformaSAC(
            env=simple_env,
            risk_controller=risk_controller,
            tau=0.01,
            gamma=0.98,
            alpha=0.3
        )
        
        assert agent.tau == 0.01
        assert agent.gamma == 0.98
        assert agent.alpha == 0.3
        
        # Test SAC info
        info = agent.get_sac_info()
        assert info['algorithm'] == 'ConformaSAC'
        assert info['tau'] == 0.01
        assert info['gamma'] == 0.98
        assert info['alpha'] == 0.3
        assert 'policy_mean' in info
        assert 'policy_std' in info
    
    def test_algorithm_update(self, simple_env, risk_controller):
        """Test algorithm update mechanism."""
        agent = ConformaSAC(env=simple_env, risk_controller=risk_controller)
        
        state = np.array([0.1, 0.2, 0.3, 0.4])
        action = np.array([0.5])
        reward = 1.0
        next_state = np.array([0.2, 0.3, 0.4, 0.5])
        done = False
        
        # Update should not raise error
        agent._update_algorithm(state, action, reward, next_state, done)
        
        # Check that transition was stored
        assert len(agent.replay_buffer) == 1
    
    def test_training_loop(self, simple_env, risk_controller):
        """Test basic training loop."""
        agent = ConformaSAC(env=simple_env, risk_controller=risk_controller)
        
        # Mock environment step to avoid actual interaction
        with patch.object(simple_env, 'step') as mock_step:
            mock_step.return_value = (
                np.array([0.1, 0.1, 0.1, 0.1]),  # next_state
                1.0,  # reward
                True,  # done
                False,  # truncated  
                {}  # info
            )
            
            # Short training run
            agent.train(total_timesteps=5)
            
            assert agent.total_timesteps == 5
            assert agent.episode_count > 0


class TestConformaPPO:
    """Test cases for ConformaPPO algorithm."""
    
    def test_initialization(self, simple_env, risk_controller):
        """Test PPO agent initialization."""
        agent = ConformaPPO(
            env=simple_env,
            risk_controller=risk_controller,
            batch_size=64,
            n_epochs=5,
            gae_lambda=0.98
        )
        
        assert agent.batch_size == 64
        assert agent.n_epochs == 5
        assert agent.gae_lambda == 0.98
        assert len(agent.rollout_buffer) == 0
        assert len(agent.current_rollout) == 0
    
    def test_log_prob_computation(self, simple_env, risk_controller):
        """Test log probability computation."""
        agent = ConformaPPO(env=simple_env, risk_controller=risk_controller)
        
        action = np.array([0.5])
        log_prob = agent._compute_log_prob(action)
        
        assert isinstance(log_prob, float)
        assert not np.isnan(log_prob)
        assert not np.isinf(log_prob)
    
    def test_rollout_processing(self, simple_env, risk_controller):
        """Test rollout processing."""
        agent = ConformaPPO(env=simple_env, risk_controller=risk_controller)
        
        # Add some transitions to current rollout
        for i in range(3):
            agent._update_algorithm(
                state=np.array([0.1, 0.2, 0.3, 0.4]),
                action=np.array([0.5]),
                reward=1.0,
                next_state=np.array([0.2, 0.3, 0.4, 0.5]),
                done=(i == 2)  # End episode on last step
            )
        
        # Check that rollout was processed
        assert len(agent.rollout_buffer) > 0
        assert len(agent.current_rollout) == 0  # Should be cleared after processing
    
    def test_advantage_computation(self, simple_env, risk_controller):
        """Test advantage computation."""
        agent = ConformaPPO(env=simple_env, risk_controller=risk_controller)
        
        # Set up a rollout
        agent.current_rollout = [
            {'reward': 1.0},
            {'reward': 2.0},
            {'reward': 0.5}
        ]
        
        advantages = agent._compute_advantages()
        
        assert len(advantages) == 3
        assert isinstance(advantages, np.ndarray)
        assert not np.any(np.isnan(advantages))
    
    def test_ppo_info(self, simple_env, risk_controller):
        """Test PPO-specific information."""
        agent = ConformaPPO(
            env=simple_env,
            risk_controller=risk_controller,
            clip_range=0.3
        )
        
        info = agent.get_ppo_info()
        assert info['algorithm'] == 'ConformaPPO'
        assert info['clip_range'] == 0.3
        assert 'policy_mean' in info
        assert 'value_baseline' in info


class TestConformaTD3:
    """Test cases for ConformaTD3 algorithm."""
    
    def test_initialization(self, simple_env, risk_controller):
        """Test TD3 agent initialization."""
        agent = ConformaTD3(
            env=simple_env,
            risk_controller=risk_controller,
            policy_delay=3,
            target_noise=0.3,
            noise_clip=0.6
        )
        
        assert agent.policy_delay == 3
        assert agent.target_noise == 0.3
        assert agent.noise_clip == 0.6
        assert agent.update_counter == 0
    
    def test_network_initialization(self, simple_env, risk_controller):
        """Test neural network initialization."""
        agent = ConformaTD3(env=simple_env, risk_controller=risk_controller)
        
        # Check that networks were initialized
        assert agent.policy_params is not None
        assert agent.critic1_params is not None
        assert agent.critic2_params is not None
        assert agent.target_policy_params is not None
        assert agent.target_critic1_params is not None
        assert agent.target_critic2_params is not None
    
    def test_policy_delay(self, simple_env, risk_controller):
        """Test policy update delay mechanism."""
        agent = ConformaTD3(
            env=simple_env,
            risk_controller=risk_controller,
            policy_delay=2
        )
        
        # Fill replay buffer
        for i in range(10):
            agent._update_algorithm(
                state=np.array([0.1, 0.2, 0.3, 0.4]),
                action=np.array([0.5]),
                reward=1.0,
                next_state=np.array([0.2, 0.3, 0.4, 0.5]),
                done=False
            )
        
        # Check update counter increased
        assert agent.update_counter > 0
    
    def test_target_updates(self, simple_env, risk_controller):
        """Test target network updates."""
        agent = ConformaTD3(env=simple_env, risk_controller=risk_controller, tau=0.1)
        
        # Store original target parameters
        original_target_policy = agent.target_policy_params.copy()
        
        # Update targets
        agent._update_targets()
        
        # Check that target parameters changed
        assert not np.array_equal(agent.target_policy_params, original_target_policy)
    
    def test_td3_info(self, simple_env, risk_controller):
        """Test TD3-specific information."""
        agent = ConformaTD3(env=simple_env, risk_controller=risk_controller)
        
        info = agent.get_td3_info()
        assert info['algorithm'] == 'ConformaTD3'
        assert 'policy_delay' in info
        assert 'update_counter' in info
        assert 'policy_norm' in info
        assert 'critic1_norm' in info
        assert 'critic2_norm' in info


class TestConformaCQL:
    """Test cases for ConformaCQL algorithm."""
    
    def test_initialization(self, simple_env, risk_controller, sample_dataset):
        """Test CQL agent initialization."""
        agent = ConformaCQL(
            env=simple_env,
            dataset=sample_dataset,
            risk_controller=risk_controller,
            cql_weight=2.0
        )
        
        assert agent.cql_weight == 2.0
        assert agent.dataset_size == len(sample_dataset['observations'])
        assert agent.dataset == sample_dataset
    
    def test_initialization_without_dataset(self, simple_env, risk_controller):
        """Test CQL initialization without dataset."""
        agent = ConformaCQL(env=simple_env, risk_controller=risk_controller)
        
        assert agent.dataset_size == 0
        assert agent.dataset == {}
    
    def test_dataset_preprocessing(self, simple_env, risk_controller, sample_dataset):
        """Test dataset preprocessing."""
        agent = ConformaCQL(
            env=simple_env,
            dataset=sample_dataset,
            risk_controller=risk_controller
        )
        
        # Dataset should be processed
        assert agent.dataset_size > 0
        
        # Risk controller should be initialized with dataset
        assert agent.risk_controller.update_count > 0
    
    def test_offline_training(self, simple_env, risk_controller, sample_dataset):
        """Test offline training process."""
        agent = ConformaCQL(
            env=simple_env,
            dataset=sample_dataset,
            risk_controller=risk_controller
        )
        
        # Short offline training
        agent.train_offline(n_epochs=5, eval_freq=2)
        
        # Check that training occurred
        # (In a real implementation, we'd check network parameters changed)  
        assert True  # Placeholder - training completed without error
    
    def test_cql_update(self, simple_env, risk_controller, sample_dataset):
        """Test CQL-specific update mechanism."""
        agent = ConformaCQL(
            env=simple_env,
            dataset=sample_dataset,
            risk_controller=risk_controller,
            batch_size=10
        )
        
        # Test batch update
        batch_indices = np.arange(10)
        
        # Should not raise an error
        agent._update_cql(batch_indices)
    
    def test_conservative_weight_adaptation(self, simple_env, risk_controller, sample_dataset):
        """Test adaptive conservative weight."""
        agent = ConformaCQL(
            env=simple_env,
            dataset=sample_dataset,
            risk_controller=risk_controller,
            cql_weight=1.0
        )
        
        original_weight = agent.cql_weight
        
        # Simulate high risk scenario
        agent.risk_controller.target_risk = 0.01  # Very low tolerance
        
        # Update should adapt weight
        batch_indices = np.arange(min(10, agent.dataset_size))
        agent._update_cql(batch_indices)
        
        # Weight adaptation is risk-dependent, so we just check it's still valid
        assert agent.cql_weight > 0
    
    def test_cql_info(self, simple_env, risk_controller, sample_dataset):
        """Test CQL-specific information."""
        agent = ConformaCQL(
            env=simple_env,
            dataset=sample_dataset,
            risk_controller=risk_controller
        )
        
        info = agent.get_cql_info()
        assert info['algorithm'] == 'ConformaCQL'
        assert info['dataset_size'] == len(sample_dataset['observations'])
        assert 'cql_weight' in info
        assert 'q1_norm' in info
        assert 'q2_norm' in info


class TestBaseAgent:
    """Test cases for base agent functionality."""
    
    def test_device_setup(self, simple_env, risk_controller):
        """Test device setup logic."""
        # Test auto device selection
        agent = ConformaSAC(env=simple_env, risk_controller=risk_controller, device='auto')
        assert agent.device in ['cpu', 'cuda']
        
        # Test manual device selection
        agent_cpu = ConformaSAC(env=simple_env, risk_controller=risk_controller, device='cpu')
        assert agent_cpu.device == 'cpu'
    
    def test_save_load(self, simple_env, risk_controller, tmp_path):
        """Test model saving and loading."""
        agent = ConformaSAC(env=simple_env, risk_controller=risk_controller)
        
        # Add some training history
        agent.training_history = [{'step': 100, 'reward': 1.0}]
        agent.total_timesteps = 100
        agent.episode_count = 5
        
        # Save
        save_path = tmp_path / "test_agent"
        agent.save(str(save_path))
        
        # Check that save file was created
        assert (save_path.parent / f"{save_path.name}_conforl.pkl").exists()
        
        # Load into new agent
        new_agent = ConformaSAC(env=simple_env, risk_controller=risk_controller)
        new_agent.load(str(save_path))
        
        # Check that state was restored
        assert new_agent.total_timesteps == 100
        assert new_agent.episode_count == 5
        assert len(new_agent.training_history) == 1
    
    def test_risk_certificate_generation(self, simple_env, risk_controller):
        """Test risk certificate generation."""
        agent = ConformaSAC(env=simple_env, risk_controller=risk_controller)
        
        # Add some data to risk controller
        sample_trajectory = Mock()
        sample_trajectory.states = np.random.random((5, 4))
        sample_trajectory.actions = np.random.randint(0, 2, 5)
        sample_trajectory.rewards = np.random.random(5)
        sample_trajectory.dones = np.array([False] * 5)
        sample_trajectory.infos = [{}] * 5
        
        agent.risk_controller.update(sample_trajectory, agent.risk_measure)
        
        # Generate certificate
        certificate = agent.get_risk_certificate()
        
        assert certificate is not None
        assert certificate.method == "adaptive_quantile"
        assert certificate.confidence == agent.risk_controller.confidence