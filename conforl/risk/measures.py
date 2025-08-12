"""Risk measure implementations for RL safety."""

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    class np:
        @staticmethod
        def array(data):
            return data
        @staticmethod
        def mean(data):
            return sum(data) / len(data) if data else 0
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
from ..core.types import TrajectoryData


class RiskMeasure(ABC):
    """Base class for risk measures in RL."""
    
    def __init__(self, name: str):
        """Initialize risk measure.
        
        Args:
            name: Human-readable name for this risk measure
        """
        self.name = name
    
    @abstractmethod
    def compute(self, trajectory: TrajectoryData) -> float:
        """Compute risk value for a trajectory.
        
        Args:
            trajectory: RL trajectory data
            
        Returns:
            Risk value (higher = more risky)
        """
        pass
    
    def compute_batch(self, trajectories: list[TrajectoryData]) -> Union[list, Any]:
        """Compute risk for batch of trajectories.
        
        Args:
            trajectories: List of RL trajectories
            
        Returns:
            Array of risk values
        """
        return np.array([self.compute(traj) for traj in trajectories])


class SafetyViolationRisk(RiskMeasure):
    """Risk measure for safety constraint violations."""
    
    def __init__(
        self,
        constraint_key: str = "constraint_violation",
        violation_threshold: float = 0.0
    ):
        """Initialize safety violation risk measure.
        
        Args:
            constraint_key: Key in trajectory info containing constraint values
            violation_threshold: Threshold above which constraint is violated
        """
        super().__init__("safety_violation")
        self.constraint_key = constraint_key
        self.violation_threshold = violation_threshold
    
    def compute(self, trajectory: TrajectoryData) -> float:
        """Compute fraction of steps with safety violations.
        
        Args:
            trajectory: RL trajectory data
            
        Returns:
            Fraction of steps with constraint violations [0, 1]
        """
        violations = []
        
        for info in trajectory.infos:
            if self.constraint_key in info:
                constraint_value = info[self.constraint_key]
                violations.append(constraint_value > self.violation_threshold)
            else:
                violations.append(False)
        
        if not violations:
            return 0.0
            
        return float(np.mean(violations))


class PerformanceRisk(RiskMeasure):
    """Risk measure for performance degradation."""
    
    def __init__(
        self,
        target_return: float,
        risk_type: str = "shortfall"
    ):
        """Initialize performance risk measure.
        
        Args:
            target_return: Minimum acceptable return
            risk_type: Type of performance risk ('shortfall', 'variance')
        """
        super().__init__(f"performance_{risk_type}")
        self.target_return = target_return
        self.risk_type = risk_type
        
        if risk_type not in ["shortfall", "variance"]:
            raise ValueError(f"Unknown risk_type: {risk_type}")
    
    def compute(self, trajectory: TrajectoryData) -> float:
        """Compute performance risk for trajectory.
        
        Args:
            trajectory: RL trajectory data
            
        Returns:
            Performance risk value
        """
        total_return = np.sum(trajectory.rewards)
        
        if self.risk_type == "shortfall":
            # Shortfall risk: max(0, target - actual)
            shortfall = max(0.0, self.target_return - total_return)
            return shortfall / abs(self.target_return) if self.target_return != 0 else shortfall
            
        elif self.risk_type == "variance":
            # Return variance as risk proxy
            if len(trajectory.rewards) <= 1:
                return 0.0
            return float(np.var(trajectory.rewards))
        
        return 0.0


class CatastrophicFailureRisk(RiskMeasure):
    """Risk measure for catastrophic failures (rare, severe events)."""
    
    def __init__(
        self,
        failure_threshold: float = -100.0,
        severity_weight: float = 1.0
    ):
        """Initialize catastrophic failure risk measure.
        
        Args:
            failure_threshold: Reward threshold below which event is catastrophic
            severity_weight: Weight for failure severity
        """
        super().__init__("catastrophic_failure")
        self.failure_threshold = failure_threshold
        self.severity_weight = severity_weight
    
    def compute(self, trajectory: TrajectoryData) -> float:
        """Compute catastrophic failure risk.
        
        Args:
            trajectory: RL trajectory data
            
        Returns:
            Catastrophic failure risk (0 = no failures, 1 = severe failure)
        """
        min_reward = np.min(trajectory.rewards)
        
        if min_reward >= self.failure_threshold:
            return 0.0
        
        # Severity increases with how far below threshold
        severity = abs(min_reward - self.failure_threshold)
        normalized_severity = min(1.0, severity * self.severity_weight / abs(self.failure_threshold))
        
        return float(normalized_severity)


class CustomRiskMeasure(RiskMeasure):
    """Flexible risk measure with user-defined computation."""
    
    def __init__(
        self,
        name: str,
        compute_fn: callable,
        **kwargs
    ):
        """Initialize custom risk measure.
        
        Args:
            name: Name for this risk measure
            compute_fn: Function that takes TrajectoryData and returns float
            **kwargs: Additional parameters passed to compute_fn
        """
        super().__init__(name)
        self.compute_fn = compute_fn
        self.kwargs = kwargs
    
    def compute(self, trajectory: TrajectoryData) -> float:
        """Compute risk using custom function.
        
        Args:
            trajectory: RL trajectory data
            
        Returns:
            Risk value from custom computation
        """
        return float(self.compute_fn(trajectory, **self.kwargs))