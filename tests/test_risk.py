"""Tests for risk measures and controllers."""

import pytest
import numpy as np
import time

from conforl.risk.measures import (
    SafetyViolationRisk, 
    PerformanceRisk, 
    CatastrophicFailureRisk,
    CustomRiskMeasure
)
from conforl.risk.controllers import (
    AdaptiveRiskController,
    MultiRiskController, 
    OnlineRiskAdaptation
)
from conforl.core.types import TrajectoryData


class TestSafetyViolationRisk:
    """Test cases for SafetyViolationRisk."""
    
    def test_initialization(self):
        """Test risk measure initialization."""
        risk_measure = SafetyViolationRisk()
        assert risk_measure.name == "safety_violation"
        assert risk_measure.constraint_key == "constraint_violation"
        assert risk_measure.violation_threshold == 0.0
    
    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        risk_measure = SafetyViolationRisk(
            constraint_key="custom_constraint",
            violation_threshold=0.5
        )
        assert risk_measure.constraint_key == "custom_constraint"
        assert risk_measure.violation_threshold == 0.5
    
    def test_compute_no_violations(self):
        """Test computation with no violations."""
        risk_measure = SafetyViolationRisk()
        
        trajectory = TrajectoryData(
            states=np.random.random((5, 2)),
            actions=np.random.randint(0, 2, 5),
            rewards=np.random.random(5),
            dones=np.array([False] * 5),
            infos=[{'constraint_violation': 0.0} for _ in range(5)]
        )
        
        risk = risk_measure.compute(trajectory)
        assert risk == 0.0
    
    def test_compute_with_violations(self):
        """Test computation with violations."""
        risk_measure = SafetyViolationRisk()
        
        trajectory = TrajectoryData(
            states=np.random.random((4, 2)),
            actions=np.random.randint(0, 2, 4),
            rewards=np.random.random(4),
            dones=np.array([False] * 4),
            infos=[
                {'constraint_violation': 0.0},
                {'constraint_violation': 1.0},  # Violation
                {'constraint_violation': 0.0},
                {'constraint_violation': 1.0}   # Violation
            ]
        )
        
        risk = risk_measure.compute(trajectory)
        assert risk == 0.5  # 2 out of 4 steps
    
    def test_compute_missing_constraint(self):
        """Test computation with missing constraint info."""
        risk_measure = SafetyViolationRisk()
        
        trajectory = TrajectoryData(
            states=np.random.random((3, 2)),
            actions=np.random.randint(0, 2, 3),
            rewards=np.random.random(3),
            dones=np.array([False] * 3),
            infos=[{}, {}, {}]  # No constraint info
        )
        
        risk = risk_measure.compute(trajectory)
        assert risk == 0.0
    
    def test_compute_batch(self):
        """Test batch computation."""
        risk_measure = SafetyViolationRisk()
        
        trajectories = []
        for i in range(3):
            trajectory = TrajectoryData(
                states=np.random.random((2, 2)),
                actions=np.random.randint(0, 2, 2),
                rewards=np.random.random(2),
                dones=np.array([False, False]),
                infos=[{'constraint_violation': float(i % 2)} for _ in range(2)]
            )
            trajectories.append(trajectory)
        
        risks = risk_measure.compute_batch(trajectories)
        assert len(risks) == 3
        assert isinstance(risks, np.ndarray)


class TestPerformanceRisk:
    """Test cases for PerformanceRisk."""
    
    def test_shortfall_risk(self):
        """Test shortfall risk computation."""
        risk_measure = PerformanceRisk(target_return=10.0, risk_type="shortfall")
        
        trajectory = TrajectoryData(
            states=np.random.random((3, 2)),
            actions=np.random.randint(0, 2, 3),
            rewards=np.array([2.0, 3.0, 1.0]),  # Total: 6.0
            dones=np.array([False, False, True]),
            infos=[{}] * 3
        )
        
        risk = risk_measure.compute(trajectory)
        expected_risk = (10.0 - 6.0) / 10.0  # Normalized shortfall
        assert abs(risk - expected_risk) < 1e-6
    
    def test_shortfall_no_risk(self):
        """Test shortfall when target is met."""
        risk_measure = PerformanceRisk(target_return=5.0, risk_type="shortfall")
        
        trajectory = TrajectoryData(
            states=np.random.random((2, 2)),
            actions=np.random.randint(0, 2, 2),
            rewards=np.array([3.0, 4.0]),  # Total: 7.0 > 5.0
            dones=np.array([False, True]),
            infos=[{}] * 2
        )
        
        risk = risk_measure.compute(trajectory)
        assert risk == 0.0
    
    def test_variance_risk(self):
        """Test variance risk computation."""
        risk_measure = PerformanceRisk(target_return=5.0, risk_type="variance")
        
        trajectory = TrajectoryData(
            states=np.random.random((4, 2)),
            actions=np.random.randint(0, 2, 4),
            rewards=np.array([1.0, 2.0, 3.0, 4.0]),
            dones=np.array([False] * 4),
            infos=[{}] * 4
        )
        
        risk = risk_measure.compute(trajectory)
        expected_variance = np.var([1.0, 2.0, 3.0, 4.0])
        assert abs(risk - expected_variance) < 1e-6
    
    def test_invalid_risk_type(self):
        """Test initialization with invalid risk type."""
        with pytest.raises(ValueError):
            PerformanceRisk(target_return=5.0, risk_type="invalid")


class TestCatastrophicFailureRisk:
    """Test cases for CatastrophicFailureRisk."""
    
    def test_no_failure(self):
        """Test computation with no catastrophic failures."""
        risk_measure = CatastrophicFailureRisk(failure_threshold=-10.0)
        
        trajectory = TrajectoryData(
            states=np.random.random((3, 2)),
            actions=np.random.randint(0, 2, 3),
            rewards=np.array([1.0, 2.0, 0.5]),  # All above threshold
            dones=np.array([False] * 3),
            infos=[{}] * 3
        )
        
        risk = risk_measure.compute(trajectory)
        assert risk == 0.0
    
    def test_with_failure(self):
        """Test computation with catastrophic failure."""
        risk_measure = CatastrophicFailureRisk(
            failure_threshold=-5.0,
            severity_weight=1.0
        )
        
        trajectory = TrajectoryData(
            states=np.random.random((3, 2)),
            actions=np.random.randint(0, 2, 3),
            rewards=np.array([1.0, -10.0, 2.0]),  # -10.0 is catastrophic
            dones=np.array([False] * 3),
            infos=[{}] * 3
        )
        
        risk = risk_measure.compute(trajectory)
        assert risk > 0.0
        assert risk <= 1.0


class TestCustomRiskMeasure:
    """Test cases for CustomRiskMeasure."""
    
    def test_custom_risk_function(self):
        """Test custom risk measure with user-defined function."""
        def custom_risk_func(trajectory, threshold=0.5):
            return float(np.mean(trajectory.rewards) < threshold)
        
        risk_measure = CustomRiskMeasure(
            name="custom_test",
            compute_fn=custom_risk_func,
            threshold=1.0
        )
        
        trajectory = TrajectoryData(
            states=np.random.random((3, 2)),
            actions=np.random.randint(0, 2, 3),
            rewards=np.array([0.2, 0.3, 0.4]),  # Mean: 0.3 < 1.0
            dones=np.array([False] * 3),
            infos=[{}] * 3
        )
        
        risk = risk_measure.compute(trajectory)
        assert risk == 1.0  # Mean < threshold


class TestAdaptiveRiskController:
    """Test cases for AdaptiveRiskController."""
    
    def test_initialization(self):
        """Test controller initialization."""
        controller = AdaptiveRiskController(
            target_risk=0.05,
            confidence=0.95,
            window_size=100
        )
        
        assert controller.target_risk == 0.05
        assert controller.confidence == 0.95
        assert controller.window_size == 100
        assert controller.update_count == 0
        assert len(controller.risk_history) == 0
    
    def test_update(self, sample_trajectory, risk_measure):
        """Test controller update."""
        controller = AdaptiveRiskController()
        
        initial_quantile = controller.current_quantile
        controller.update(sample_trajectory, risk_measure)
        
        assert controller.update_count == 1
        assert len(controller.risk_history) == 1
        assert len(controller.score_history) == 1
    
    def test_multiple_updates(self, risk_measure):
        """Test multiple updates."""
        controller = AdaptiveRiskController(window_size=50)
        
        for i in range(20):
            trajectory = TrajectoryData(
                states=np.random.random((5, 2)),
                actions=np.random.randint(0, 2, 5),
                rewards=np.random.random(5),
                dones=np.array([False] * 5),
                infos=[{}] * 5
            )
            controller.update(trajectory, risk_measure)
        
        assert controller.update_count == 20
        assert len(controller.risk_history) == 20
    
    def test_risk_bound_calculation(self, sample_trajectory, risk_measure):
        """Test risk bound calculation."""
        controller = AdaptiveRiskController(target_risk=0.1)
        
        # Add some updates
        for _ in range(10):
            controller.update(sample_trajectory, risk_measure)
        
        risk_bound = controller.get_risk_bound()
        assert isinstance(risk_bound, float)
        assert 0 <= risk_bound <= 1
    
    def test_certificate_generation(self, sample_trajectory, risk_measure):
        """Test risk certificate generation."""
        controller = AdaptiveRiskController()
        
        controller.update(sample_trajectory, risk_measure)
        certificate = controller.get_certificate()
        
        assert certificate.method == "adaptive_quantile"
        assert certificate.confidence == controller.confidence
        assert certificate.sample_size == 1
        assert certificate.timestamp is not None
        assert isinstance(certificate.metadata, dict)


class TestMultiRiskController:
    """Test cases for MultiRiskController."""
    
    def test_initialization(self):
        """Test multi-risk controller initialization."""
        risk_constraints = [
            ("safety", 0.05),
            ("performance", 0.1)
        ]
        
        controller = MultiRiskController(
            risk_constraints=risk_constraints,
            confidence=0.95
        )
        
        assert len(controller.risk_constraints) == 2
        assert controller.risk_constraints["safety"] == 0.05
        assert controller.risk_constraints["performance"] == 0.1
        assert len(controller.controllers) == 2
    
    def test_update(self, sample_trajectory):
        """Test multi-risk controller update."""
        controller = MultiRiskController([("test", 0.05)])
        
        risk_measures = {
            "test": SafetyViolationRisk()
        }
        
        controller.update(sample_trajectory, risk_measures)
        
        # Check that individual controller was updated
        assert controller.controllers["test"].update_count == 1
    
    def test_constraint_checking(self, sample_trajectory):
        """Test constraint satisfaction checking."""
        controller = MultiRiskController([
            ("risk1", 0.05),
            ("risk2", 0.1)
        ])
        
        risk_measures = {
            "risk1": SafetyViolationRisk(),
            "risk2": SafetyViolationRisk()
        }
        
        # Update with some data
        controller.update(sample_trajectory, risk_measures)
        
        satisfied, bounds = controller.check_constraints()
        assert isinstance(satisfied, bool)
        assert isinstance(bounds, dict)
        assert "risk1" in bounds
        assert "risk2" in bounds
    
    def test_certificate_generation(self, sample_trajectory):
        """Test certificate generation for all constraints."""
        controller = MultiRiskController([("test", 0.05)])
        
        risk_measures = {"test": SafetyViolationRisk()}
        controller.update(sample_trajectory, risk_measures)
        
        certificates = controller.get_certificates()
        assert isinstance(certificates, dict)
        assert "test" in certificates
        assert certificates["test"].method == "adaptive_quantile"


class TestOnlineRiskAdaptation:
    """Test cases for OnlineRiskAdaptation."""
    
    def test_initialization(self):
        """Test online adaptation initialization."""
        adapter = OnlineRiskAdaptation(
            initial_quantile=0.9,
            learning_rate=0.01,
            target_coverage=0.95
        )
        
        assert adapter.current_quantile == 0.9
        assert adapter.learning_rate == 0.01
        assert adapter.target_coverage == 0.95
        assert adapter.step_count == 0
    
    def test_online_update(self):
        """Test online quantile update."""
        adapter = OnlineRiskAdaptation(initial_quantile=0.5, learning_rate=0.1)
        
        # Update with good coverage (actual < prediction)
        new_quantile = adapter.update_online(
            prediction=1.0,
            actual_risk=0.5
        )
        
        assert adapter.step_count == 1
        assert len(adapter.coverage_history) == 1
        assert adapter.coverage_history[0] == True  # Covered
    
    def test_quantile_adaptation(self):
        """Test quantile adaptation based on coverage."""
        adapter = OnlineRiskAdaptation(
            initial_quantile=0.5,
            learning_rate=0.1,
            target_coverage=0.9
        )
        
        # Add several updates with poor coverage
        for _ in range(15):
            adapter.update_online(prediction=0.5, actual_risk=1.0)  # Under-covering
        
        # Quantile should increase due to under-coverage
        assert adapter.current_quantile > 0.5
    
    def test_bound_calculation(self):
        """Test risk bound calculation."""
        adapter = OnlineRiskAdaptation(initial_quantile=0.8)
        
        base_prediction = 0.1
        bound = adapter.get_current_bound(base_prediction)
        
        expected_bound = min(1.0, base_prediction * (1 + 0.8))
        assert abs(bound - expected_bound) < 1e-6
    
    def test_adaptation_stats(self):
        """Test adaptation statistics."""
        adapter = OnlineRiskAdaptation()
        
        # Add some updates
        for i in range(5):
            adapter.update_online(prediction=0.5, actual_risk=0.3)
        
        stats = adapter.get_adaptation_stats()
        assert stats['step_count'] == 5
        assert stats['current_quantile'] == adapter.current_quantile
        assert 'empirical_coverage' in stats
        assert 'target_coverage' in stats