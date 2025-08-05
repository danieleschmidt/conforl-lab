"""Pytest configuration and shared fixtures."""

import pytest
import numpy as np
import gymnasium as gym
from unittest.mock import Mock

from conforl.core.types import TrajectoryData, RiskCertificate
from conforl.risk.controllers import AdaptiveRiskController
from conforl.risk.measures import SafetyViolationRisk


@pytest.fixture
def simple_env():
    """Create a simple test environment."""
    try:
        return gym.make('CartPole-v1')
    except:
        # Mock environment if CartPole not available
        env = Mock()
        env.observation_space = Mock()
        env.observation_space.shape = (4,)
        env.action_space = Mock()
        env.action_space.shape = (1,)
        env.action_space.sample.return_value = 0
        env.reset.return_value = (np.array([0.1, 0.1, 0.1, 0.1]), {})
        env.step.return_value = (np.array([0.1, 0.1, 0.1, 0.1]), 1.0, False, False, {})
        return env


@pytest.fixture
def sample_trajectory():
    """Create sample trajectory data for testing."""
    states = np.random.random((10, 4))
    actions = np.random.randint(0, 2, 10)
    rewards = np.random.random(10)
    dones = np.array([False] * 9 + [True])
    infos = [{'constraint_violation': 0.0} for _ in range(10)]
    
    return TrajectoryData(
        states=states,
        actions=actions,
        rewards=rewards,
        dones=dones,
        infos=infos
    )


@pytest.fixture
def risk_controller():
    """Create a risk controller for testing."""
    return AdaptiveRiskController(
        target_risk=0.05,
        confidence=0.95,
        window_size=100
    )


@pytest.fixture
def risk_measure():
    """Create a risk measure for testing."""
    return SafetyViolationRisk()


@pytest.fixture
def sample_config():
    """Create sample configuration for testing."""
    return {
        'learning_rate': 0.001,
        'buffer_size': 10000,
        'batch_size': 32,
        'target_risk': 0.05,
        'confidence': 0.95,
        'device': 'cpu'
    }


@pytest.fixture
def sample_dataset():
    """Create sample offline dataset for testing."""
    n_samples = 100
    obs_dim = 4
    act_dim = 1
    
    return {
        'observations': np.random.random((n_samples, obs_dim)),
        'actions': np.random.random((n_samples, act_dim)),
        'rewards': np.random.random(n_samples),
        'next_observations': np.random.random((n_samples, obs_dim)),
        'terminals': np.random.choice([True, False], n_samples)
    }


@pytest.fixture
def mock_certificate():
    """Create mock risk certificate for testing."""
    return RiskCertificate(
        risk_bound=0.05,
        confidence=0.95,
        coverage_guarantee=0.94,
        method="test",
        sample_size=100,
        timestamp=1234567890.0
    )