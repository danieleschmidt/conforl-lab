"""Core type definitions for ConfoRL."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, Tuple

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Create minimal numpy-like interface for basic functionality
    class np:
        @staticmethod
        def array(data):
            return data
        @staticmethod
        def where(condition):
            if hasattr(condition, '__iter__'):
                return [i for i, x in enumerate(condition) if x]
            return []


@dataclass
class RiskCertificate:
    """Certificate providing formal risk guarantees.
    
    Attributes:
        risk_bound: Upper bound on risk with given confidence
        confidence: Confidence level (e.g., 0.95 for 95%)
        coverage_guarantee: Finite-sample coverage probability
        method: Conformal method used ('split', 'adaptive', etc.)
        sample_size: Calibration set size used
        timestamp: When certificate was generated
    """
    risk_bound: float
    confidence: float
    coverage_guarantee: float
    method: str
    sample_size: int
    timestamp: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass  
class ConformalSet:
    """Conformal prediction set for actions or values.
    
    Attributes:
        prediction_set: Set of predictions with coverage guarantee
        quantiles: Confidence quantiles used
        coverage: Target coverage level
        nonconformity_scores: Scores used to construct set
    """
    prediction_set: Union[List, Any]  # np.ndarray when available
    quantiles: Tuple[float, float]
    coverage: float
    nonconformity_scores: Union[List, Any]  # np.ndarray when available


@dataclass
class TrajectoryData:
    """Container for RL trajectory data.
    
    Attributes:
        states: Sequence of environment states
        actions: Sequence of actions taken
        rewards: Sequence of rewards received
        dones: Sequence of episode termination flags
        infos: Additional environment information
        risks: Risk values computed for trajectory
    """
    states: Union[List, Any]  # np.ndarray when available
    actions: Union[List, Any]  # np.ndarray when available  
    rewards: Union[List, Any]  # np.ndarray when available
    dones: Union[List, Any]  # np.ndarray when available
    infos: List[Dict[str, Any]]
    risks: Optional[Union[List, Any]] = None  # np.ndarray when available
    
    def __len__(self) -> int:
        return len(self.states)
    
    @property
    def episode_length(self) -> int:
        """Length of trajectory until first done=True."""
        try:
            if NUMPY_AVAILABLE and hasattr(self.dones, 'ndim'):
                return int(np.where(self.dones)[0][0]) + 1
            else:
                # Fallback for list-based implementation
                for i, done in enumerate(self.dones):
                    if done:
                        return i + 1
                return len(self)
        except (IndexError, TypeError):
            return len(self)


# Type aliases for common use cases  
StateType = Union[List, Any, Dict[str, Any]]  # np.ndarray when available
ActionType = Union[List, Any, int, float]  # np.ndarray when available
RewardType = float
InfoType = Dict[str, Any]