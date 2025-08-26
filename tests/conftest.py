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
    from conforl.core.types import TrajectoryData
    
    # Create 10-step trajectory as expected by tests
    if NUMPY_AVAILABLE:
        states = np.random.random((10, 4))
        actions = np.random.randint(0, 2, 10)
        rewards = np.random.random(10)
        dones = np.array([False] * 9 + [True])  # Episode ends at step 10
    else:
        states = [[0.1, 0.2, 0.3, 0.4] for _ in range(10)]
        actions = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        rewards = [1.0] * 10
        dones = [False] * 9 + [True]
    
    infos = [{'step': i} for i in range(10)]
    
    return TrajectoryData(
        states=states,
        actions=actions,
        rewards=rewards,
        dones=dones,
        infos=infos
    )

@pytest.fixture
def risk_config():
    """Risk configuration for testing."""
    return {
        'target_risk': 0.05,
        'confidence': 0.95,
        'window_size': 100
    }

@pytest.fixture
def mock_certificate():
    """Mock risk certificate for testing."""
    from conforl.core.types import RiskCertificate
    import time
    
    return RiskCertificate(
        risk_bound=0.05,
        confidence=0.95,
        coverage_guarantee=0.94,
        method="test_method",
        sample_size=100,
        timestamp=time.time(),
        metadata={"test": "data"}
    )

@pytest.fixture
def risk_measure():
    """Mock risk measure for testing."""
    from conforl.risk.measures import SafetyViolationRisk
    return SafetyViolationRisk()