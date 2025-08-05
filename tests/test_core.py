"""Tests for core conformal prediction functionality."""

import pytest
import numpy as np
import time

from conforl.core.conformal import SplitConformalPredictor
from conforl.core.types import RiskCertificate, ConformalSet, TrajectoryData


class TestSplitConformalPredictor:
    """Test cases for SplitConformalPredictor."""
    
    def test_initialization(self):
        """Test predictor initialization."""
        predictor = SplitConformalPredictor(coverage=0.95)
        assert predictor.coverage == 0.95
        assert predictor.alpha == 0.05
        assert predictor.method == "split_conformal"
        assert predictor.quantile is None
    
    def test_invalid_coverage(self):
        """Test initialization with invalid coverage."""
        with pytest.raises(ValueError):
            SplitConformalPredictor(coverage=0.0)
        
        with pytest.raises(ValueError):
            SplitConformalPredictor(coverage=1.0)
        
        with pytest.raises(ValueError):
            SplitConformalPredictor(coverage=1.5)
    
    def test_calibration(self):
        """Test calibration process."""
        predictor = SplitConformalPredictor(coverage=0.9)
        
        # Generate calibration data
        calibration_data = np.random.random((100, 2))
        calibration_scores = np.random.random(100)
        
        predictor.calibrate(calibration_data, calibration_scores)
        
        assert predictor.quantile is not None
        assert predictor.calibration_size == 100
        assert 0 <= predictor.quantile <= 1
    
    def test_prediction_without_calibration(self):
        """Test prediction without calibration raises error."""
        predictor = SplitConformalPredictor()
        test_data = np.array([1.0, 2.0, 3.0])
        
        with pytest.raises(ValueError, match="Must call calibrate"):
            predictor.predict(test_data)
    
    def test_prediction_after_calibration(self):
        """Test prediction after calibration."""
        predictor = SplitConformalPredictor(coverage=0.9)
        
        # Calibrate
        calibration_scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        predictor.calibrate(np.zeros((5, 2)), calibration_scores)
        
        # Predict
        test_predictions = np.array([1.0, 2.0, 3.0])
        conformal_set = predictor.predict(test_predictions)
        
        assert isinstance(conformal_set, ConformalSet)
        assert conformal_set.prediction_set.shape == (3, 2)  # Lower and upper bounds
        assert conformal_set.coverage == 0.9
    
    def test_risk_certificate_generation(self):
        """Test risk certificate generation."""
        predictor = SplitConformalPredictor(coverage=0.95)
        
        # Calibrate
        calibration_scores = np.random.random(50)
        predictor.calibrate(np.zeros((50, 2)), calibration_scores)
        
        # Generate certificate
        test_data = np.random.random((10, 2))
        
        def dummy_risk_function(x):
            return np.random.random(len(x))
        
        certificate = predictor.get_risk_certificate(test_data, dummy_risk_function)
        
        assert isinstance(certificate, RiskCertificate)
        assert certificate.method == "split_conformal"
        assert certificate.confidence == 0.95
        assert certificate.sample_size == 50
        assert certificate.timestamp is not None
        assert isinstance(certificate.metadata, dict)


class TestTrajectoryData:
    """Test cases for TrajectoryData."""
    
    def test_trajectory_creation(self, sample_trajectory):
        """Test trajectory data creation."""
        assert len(sample_trajectory) == 10
        assert sample_trajectory.episode_length <= 10
        assert sample_trajectory.states.shape == (10, 4)
        assert sample_trajectory.actions.shape == (10,)
        assert sample_trajectory.rewards.shape == (10,)
        assert sample_trajectory.dones.shape == (10,)
        assert len(sample_trajectory.infos) == 10
    
    def test_episode_length_calculation(self):
        """Test episode length calculation."""
        # Episode ending at step 5
        dones = np.array([False, False, False, False, False, True, False, False])
        
        trajectory = TrajectoryData(
            states=np.random.random((8, 2)),
            actions=np.random.randint(0, 2, 8),
            rewards=np.random.random(8),
            dones=dones,
            infos=[{}] * 8
        )
        
        assert trajectory.episode_length == 6  # Index 5 + 1
    
    def test_episode_length_no_done(self):
        """Test episode length when no done flag is set."""
        dones = np.array([False, False, False, False])
        
        trajectory = TrajectoryData(
            states=np.random.random((4, 2)),
            actions=np.random.randint(0, 2, 4),
            rewards=np.random.random(4),
            dones=dones,
            infos=[{}] * 4
        )
        
        assert trajectory.episode_length == 4  # Full length


class TestRiskCertificate:
    """Test cases for RiskCertificate."""
    
    def test_certificate_creation(self, mock_certificate):
        """Test risk certificate creation."""
        assert mock_certificate.risk_bound == 0.05
        assert mock_certificate.confidence == 0.95
        assert mock_certificate.coverage_guarantee == 0.94
        assert mock_certificate.method == "test"
        assert mock_certificate.sample_size == 100
        assert mock_certificate.timestamp == 1234567890.0
    
    def test_certificate_with_metadata(self):
        """Test certificate with metadata."""
        metadata = {"test_key": "test_value", "numeric_key": 42}
        
        certificate = RiskCertificate(
            risk_bound=0.1,
            confidence=0.9,
            coverage_guarantee=0.85,
            method="test_method",
            sample_size=50,
            metadata=metadata
        )
        
        assert certificate.metadata == metadata
        assert certificate.metadata["test_key"] == "test_value"
        assert certificate.metadata["numeric_key"] == 42


class TestConformalSet:
    """Test cases for ConformalSet."""
    
    def test_conformal_set_creation(self):
        """Test conformal set creation."""
        prediction_set = np.array([[1.0, 3.0], [2.0, 4.0]])  # Intervals
        quantiles = (0.05, 0.95)
        coverage = 0.9
        scores = np.array([0.1, 0.2])
        
        conformal_set = ConformalSet(
            prediction_set=prediction_set,
            quantiles=quantiles,
            coverage=coverage,
            nonconformity_scores=scores
        )
        
        assert np.array_equal(conformal_set.prediction_set, prediction_set)
        assert conformal_set.quantiles == quantiles
        assert conformal_set.coverage == coverage
        assert np.array_equal(conformal_set.nonconformity_scores, scores)
    
    def test_conformal_set_list_predictions(self):
        """Test conformal set with list predictions."""
        prediction_set = [1, 2, 3, 4]  # Discrete predictions
        
        conformal_set = ConformalSet(
            prediction_set=prediction_set,
            quantiles=(0.1, 0.9),
            coverage=0.8,
            nonconformity_scores=np.array([0.1, 0.2, 0.3, 0.4])
        )
        
        assert conformal_set.prediction_set == prediction_set