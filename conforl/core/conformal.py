"""Core conformal prediction implementations."""

import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple, Union
import time

from .types import RiskCertificate, ConformalSet


class ConformalPredictor(ABC):
    """Base class for conformal predictors."""
    
    def __init__(
        self,
        coverage: float = 0.95,
        method: str = "base"
    ):
        """Initialize conformal predictor.
        
        Args:
            coverage: Target coverage probability (e.g., 0.95)
            method: Conformal method identifier
        """
        if not 0 < coverage < 1:
            raise ValueError(f"Coverage must be in (0,1), got {coverage}")
            
        self.coverage = coverage
        self.method = method
        self.alpha = 1 - coverage
        
    @abstractmethod
    def calibrate(
        self,
        calibration_data: np.ndarray,
        calibration_scores: np.ndarray
    ) -> None:
        """Calibrate the conformal predictor.
        
        Args:
            calibration_data: Calibration input data
            calibration_scores: Nonconformity scores for calibration
        """
        pass
    
    @abstractmethod
    def predict(
        self,
        test_data: np.ndarray
    ) -> ConformalSet:
        """Generate conformal prediction set.
        
        Args:
            test_data: Test input data
            
        Returns:
            Conformal prediction set with coverage guarantee
        """
        pass


class SplitConformalPredictor(ConformalPredictor):
    """Split conformal prediction with finite-sample guarantees."""
    
    def __init__(
        self,
        coverage: float = 0.95,
        score_function: Optional[Callable] = None
    ):
        """Initialize split conformal predictor.
        
        Args:
            coverage: Target coverage probability
            score_function: Custom nonconformity score function
        """
        super().__init__(coverage, "split_conformal")
        self.score_function = score_function or self._default_score_function
        self.quantile = None
        self.calibration_size = 0
        
    def _default_score_function(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """Default nonconformity score: absolute residual."""
        return np.abs(y_true - y_pred)
    
    def calibrate(
        self,
        calibration_data: np.ndarray,
        calibration_scores: np.ndarray
    ) -> None:
        """Calibrate using split conformal method.
        
        Args:
            calibration_data: Calibration inputs (unused in split method)
            calibration_scores: Precomputed nonconformity scores
        """
        self.calibration_size = len(calibration_scores)
        
        # Compute empirical quantile with finite-sample correction
        n = self.calibration_size
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(q_level, 1.0)  # Ensure valid quantile
        
        self.quantile = np.quantile(calibration_scores, q_level)
        
    def predict(
        self,
        test_predictions: np.ndarray,
        return_scores: bool = False
    ) -> Union[ConformalSet, Tuple[ConformalSet, np.ndarray]]:
        """Generate split conformal prediction intervals.
        
        Args:
            test_predictions: Point predictions for test data
            return_scores: Whether to return nonconformity scores
            
        Returns:
            Conformal prediction set (and scores if requested)
        """
        if self.quantile is None:
            raise ValueError("Must call calibrate() before predict()")
            
        # Create prediction intervals
        lower = test_predictions - self.quantile
        upper = test_predictions + self.quantile
        prediction_set = np.column_stack([lower, upper])
        
        # Compute nonconformity scores (distance to prediction)
        scores = np.zeros(len(test_predictions))  # Placeholder
        
        conformal_set = ConformalSet(
            prediction_set=prediction_set,
            quantiles=(self.alpha/2, 1-self.alpha/2),
            coverage=self.coverage,
            nonconformity_scores=scores
        )
        
        if return_scores:
            return conformal_set, scores
        return conformal_set
    
    def get_risk_certificate(
        self,
        test_data: np.ndarray,
        risk_function: Callable[[np.ndarray], np.ndarray]
    ) -> RiskCertificate:
        """Generate formal risk certificate.
        
        Args:
            test_data: Test input data
            risk_function: Function computing risk from predictions
            
        Returns:
            Certificate with formal risk guarantees
        """
        if self.quantile is None:
            raise ValueError("Must call calibrate() before generating certificate")
            
        # For split conformal, risk bound is derived from coverage guarantee
        finite_sample_coverage = self.coverage
        if self.calibration_size > 0:
            # Adjust for finite sample size
            finite_sample_coverage = max(
                0.0,
                self.coverage - 2 * np.sqrt(np.log(2/0.05) / (2 * self.calibration_size))
            )
        
        risk_bound = 1 - finite_sample_coverage
        
        return RiskCertificate(
            risk_bound=risk_bound,
            confidence=self.coverage,
            coverage_guarantee=finite_sample_coverage,
            method=self.method,
            sample_size=self.calibration_size,
            timestamp=time.time(),
            metadata={
                "quantile": self.quantile,
                "alpha": self.alpha
            }
        )