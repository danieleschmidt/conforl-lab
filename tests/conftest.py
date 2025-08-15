"""Pytest configuration for ConfoRL tests."""

import pytest
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    class np:
        @staticmethod
        def array(data):
            return data if isinstance(data, list) else [data]
        @staticmethod
        def zeros(shape):
            return [0.0] * (shape[0] if hasattr(shape, '__iter__') else shape)

try:
    import gymnasium as gym
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False

@pytest.fixture
def mock_env():
    """Mock gymnasium environment for testing."""
    if GYM_AVAILABLE:
        return gym.make('CartPole-v1')
    
    # Mock environment
    class MockEnv:
        def __init__(self):
            self.observation_space = MockSpace(shape=(4,))
            self.action_space = MockSpace(shape=(2,))
        
        def reset(self):
            return [0.0, 0.0, 0.0, 0.0], {}
        
        def step(self, action):
            return [0.0, 0.0, 0.0, 0.0], 1.0, False, False, {}
    
    class MockSpace:
        def __init__(self, shape):
            self.shape = shape
    
    return MockEnv()

@pytest.fixture
def sample_trajectory():
    """Sample trajectory data for testing."""
    return {
        'states': [[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5]],
        'actions': [0, 1],
        'rewards': [1.0, 1.0],
        'dones': [False, True]
    }

@pytest.fixture
def risk_config():
    """Risk configuration for testing."""
    return {
        'target_risk': 0.05,
        'confidence': 0.95,
        'window_size': 100
    }