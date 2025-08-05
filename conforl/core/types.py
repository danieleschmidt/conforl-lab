"""Core type definitions for ConfoRL."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np


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
    prediction_set: Union[np.ndarray, List]
    quantiles: Tuple[float, float]
    coverage: float
    nonconformity_scores: np.ndarray


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
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    infos: List[Dict[str, Any]]
    risks: Optional[np.ndarray] = None
    
    def __len__(self) -> int:
        return len(self.states)
    
    @property
    def episode_length(self) -> int:
        """Length of trajectory until first done=True."""
        try:
            return int(np.where(self.dones)[0][0]) + 1
        except IndexError:
            return len(self)


# Type aliases for common use cases
StateType = Union[np.ndarray, Dict[str, np.ndarray]]
ActionType = Union[np.ndarray, int, float]
RewardType = float
InfoType = Dict[str, Any]