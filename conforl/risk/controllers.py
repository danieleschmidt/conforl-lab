"""Risk controllers for adaptive conformal RL."""

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
        @staticmethod
        def percentile(data, q):
            if not data:
                return 0
            sorted_data = sorted(data)
            idx = int(q / 100 * len(sorted_data))
            return sorted_data[min(idx, len(sorted_data) - 1)]
from typing import Dict, List, Optional, Tuple, Union
import time
from collections import deque

from ..core.types import RiskCertificate, TrajectoryData
from ..core.conformal import SplitConformalPredictor
from .measures import RiskMeasure


class AdaptiveRiskController:
    """Adaptive risk controller with online quantile updates."""
    
    def __init__(
        self,
        target_risk: float = 0.05,
        confidence: float = 0.95,
        window_size: int = 1000,
        learning_rate: float = 0.01,
        initial_quantile: float = 0.9
    ):
        """Initialize adaptive risk controller.
        
        Args:
            target_risk: Target risk level (e.g., 0.05 for 5% risk)
            confidence: Confidence level for guarantees
            window_size: Size of sliding window for adaptation
            learning_rate: Rate of quantile adaptation
            initial_quantile: Initial quantile estimate
        """
        self.target_risk = target_risk
        self.confidence = confidence
        self.window_size = window_size
        self.learning_rate = learning_rate
        
        # State tracking
        self.current_quantile = initial_quantile
        self.risk_history = deque(maxlen=window_size)
        self.score_history = deque(maxlen=window_size)
        self.update_count = 0
        
    def update(
        self,
        trajectory: TrajectoryData,
        risk_measure: RiskMeasure
    ) -> None:
        """Update controller with new trajectory data.
        
        Args:
            trajectory: New RL trajectory
            risk_measure: Risk measure to evaluate trajectory
        """
        # Compute risk for this trajectory
        risk_value = risk_measure.compute(trajectory)
        self.risk_history.append(risk_value)
        
        # Compute nonconformity score (risk relative to quantile)
        score = abs(risk_value - self.current_quantile)
        self.score_history.append(score)
        
        # Adaptive quantile update
        if len(self.risk_history) >= 10:  # Minimum data requirement
            empirical_risk = np.mean(list(self.risk_history)[-10:])
            risk_error = empirical_risk - self.target_risk
            
            # Update quantile to track target risk
            self.current_quantile += self.learning_rate * risk_error
            self.current_quantile = np.clip(self.current_quantile, 0.0, 1.0)
        
        self.update_count += 1
    
    def get_risk_bound(self) -> float:
        """Get current risk bound estimate.
        
        Returns:
            Current risk bound
        """
        if len(self.risk_history) == 0:
            return self.target_risk
        
        # Use recent empirical risk with confidence adjustment
        recent_risks = list(self.risk_history)[-min(100, len(self.risk_history)):]
        empirical_risk = np.mean(recent_risks)
        
        # Add confidence margin based on sample size
        n = len(recent_risks)
        confidence_margin = np.sqrt(np.log(2/0.05) / (2 * n)) if n > 0 else 0.1
        
        return min(1.0, empirical_risk + confidence_margin)
    
    def get_certificate(self) -> RiskCertificate:
        """Generate risk certificate with current guarantees.
        
        Returns:
            Risk certificate with adaptive bounds
        """
        risk_bound = self.get_risk_bound()
        sample_size = len(self.risk_history)
        
        # Finite-sample coverage adjustment
        coverage_guarantee = self.confidence
        if sample_size > 0:
            coverage_guarantee = max(
                0.5,
                self.confidence - 2 * np.sqrt(np.log(2/0.05) / (2 * sample_size))
            )
        
        return RiskCertificate(
            risk_bound=risk_bound,
            confidence=self.confidence,
            coverage_guarantee=coverage_guarantee,
            method="adaptive_quantile",
            sample_size=sample_size,
            timestamp=time.time(),
            metadata={
                "current_quantile": self.current_quantile,
                "target_risk": self.target_risk,
                "window_size": self.window_size,
                "update_count": self.update_count
            }
        )


class MultiRiskController:
    """Controller for multiple simultaneous risk constraints."""
    
    def __init__(
        self,
        risk_constraints: List[Tuple[str, float]],
        confidence: float = 0.95,
        combination_method: str = "bonferroni"
    ):
        """Initialize multi-risk controller.
        
        Args:
            risk_constraints: List of (risk_name, target_level) pairs
            confidence: Overall confidence level
            combination_method: Method for combining multiple constraints
        """
        self.risk_constraints = dict(risk_constraints)
        self.confidence = confidence
        self.combination_method = combination_method
        
        # Individual controllers for each risk
        self.controllers = {}
        for risk_name, target_level in risk_constraints:
            # Adjust confidence for multiple testing
            if combination_method == "bonferroni":
                adjusted_conf = 1 - (1 - confidence) / len(risk_constraints)
            else:
                adjusted_conf = confidence
                
            self.controllers[risk_name] = AdaptiveRiskController(
                target_risk=target_level,
                confidence=adjusted_conf
            )
    
    def update(
        self,
        trajectory: TrajectoryData,
        risk_measures: Dict[str, RiskMeasure]
    ) -> None:
        """Update all risk controllers.
        
        Args:
            trajectory: RL trajectory data
            risk_measures: Dict mapping risk names to RiskMeasure objects
        """
        for risk_name, controller in self.controllers.items():
            if risk_name in risk_measures:
                controller.update(trajectory, risk_measures[risk_name])
    
    def check_constraints(self) -> Tuple[bool, Dict[str, float]]:
        """Check if all risk constraints are satisfied.
        
        Returns:
            (all_satisfied, risk_bounds_dict)
        """
        risk_bounds = {}
        all_satisfied = True
        
        for risk_name, controller in self.controllers.items():
            bound = controller.get_risk_bound()
            risk_bounds[risk_name] = bound
            
            target = self.risk_constraints[risk_name]
            if bound > target * 1.1:  # 10% tolerance
                all_satisfied = False
        
        return all_satisfied, risk_bounds
    
    def get_certificates(self) -> Dict[str, RiskCertificate]:
        """Get risk certificates for all constraints.
        
        Returns:
            Dict mapping risk names to certificates
        """
        return {
            name: controller.get_certificate()
            for name, controller in self.controllers.items()
        }


class OnlineRiskAdaptation:
    """Online adaptation of risk bounds during deployment."""
    
    def __init__(
        self,
        initial_quantile: float = 0.9,
        learning_rate: float = 0.01,
        target_coverage: float = 0.95,
        adaptation_window: int = 100
    ):
        """Initialize online risk adaptation.
        
        Args:
            initial_quantile: Starting quantile estimate
            learning_rate: Rate of online updates
            target_coverage: Target coverage probability
            adaptation_window: Window for computing empirical coverage
        """
        self.initial_quantile = initial_quantile
        self.learning_rate = learning_rate
        self.target_coverage = target_coverage
        self.adaptation_window = adaptation_window
        
        # Online state
        self.current_quantile = initial_quantile
        self.coverage_history = deque(maxlen=adaptation_window)
        self.prediction_history = deque(maxlen=adaptation_window)
        self.step_count = 0
    
    def update_online(
        self,
        prediction: float,
        actual_risk: float
    ) -> float:
        """Update quantile based on new observation.
        
        Args:
            prediction: Predicted risk value
            actual_risk: Observed risk value
            
        Returns:
            Updated quantile estimate
        """
        # Check if prediction covered actual risk
        covered = actual_risk <= prediction
        self.coverage_history.append(covered)
        self.prediction_history.append(prediction)
        
        # Update quantile based on coverage error
        if len(self.coverage_history) >= 10:
            empirical_coverage = np.mean(list(self.coverage_history))
            coverage_error = empirical_coverage - self.target_coverage
            
            # Adjust quantile: increase if under-covering, decrease if over-covering
            self.current_quantile -= self.learning_rate * coverage_error
            self.current_quantile = np.clip(self.current_quantile, 0.1, 0.99)
        
        self.step_count += 1
        return self.current_quantile
    
    def get_current_bound(self, base_prediction: float) -> float:
        """Get current risk bound for a base prediction.
        
        Args:
            base_prediction: Base risk prediction
            
        Returns:
            Adjusted risk bound with current quantile
        """
        # Apply current quantile as adjustment factor
        adjusted_bound = base_prediction * (1 + self.current_quantile)
        return min(1.0, adjusted_bound)
    
    def get_adaptation_stats(self) -> Dict[str, float]:
        """Get statistics about online adaptation.
        
        Returns:
            Dict with adaptation statistics
        """
        stats = {
            "current_quantile": self.current_quantile,
            "step_count": self.step_count,
        }
        
        if len(self.coverage_history) > 0:
            stats.update({
                "empirical_coverage": np.mean(list(self.coverage_history)),
                "target_coverage": self.target_coverage,
                "coverage_gap": abs(np.mean(list(self.coverage_history)) - self.target_coverage)
            })
        
        return stats