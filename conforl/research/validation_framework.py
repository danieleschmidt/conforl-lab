"""Comprehensive Validation Framework for Research Algorithms.

Provides rigorous validation, verification, and statistical testing
for research contributions in conformal RL.

Author: ConfoRL Research Team
License: Apache 2.0
"""

import time
import math
import warnings
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import threading
from collections import defaultdict

try:
    import numpy as np
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    stats = None

from ..core.types import RiskCertificate, TrajectoryData
from ..utils.logging import get_logger
from ..utils.errors import ConfoRLError, ValidationError

logger = get_logger(__name__)


class ValidationLevel(Enum):
    """Levels of validation rigor."""
    BASIC = "basic"
    COMPREHENSIVE = "comprehensive"
    RESEARCH_GRADE = "research_grade"
    PUBLICATION_READY = "publication_ready"


class StatisticalTest(Enum):
    """Statistical tests for validation."""
    T_TEST = "t_test"
    WILCOXON = "wilcoxon"
    MANN_WHITNEY = "mann_whitney"
    KOLMOGOROV_SMIRNOV = "ks_test"
    PERMUTATION_TEST = "permutation"
    BOOTSTRAP = "bootstrap"


@dataclass
class ValidationResult:
    """Result of a validation test."""
    test_name: str
    passed: bool
    score: float
    threshold: float
    confidence_interval: Optional[Tuple[float, float]] = None
    p_value: Optional[float] = None
    effect_size: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class TheoreticalBoundValidation:
    """Validation of theoretical bounds."""
    theoretical_bound: float
    empirical_violation_rate: float
    confidence_level: float
    sample_size: int
    bound_tightness: float
    valid: bool
    margin_of_error: float


class AlgorithmValidator:
    """Comprehensive validator for research algorithms."""
    
    def __init__(
        self,
        validation_level: ValidationLevel = ValidationLevel.COMPREHENSIVE,
        significance_level: float = 0.05,
        min_sample_size: int = 100,
        enable_statistical_tests: bool = True
    ):
        """Initialize algorithm validator.
        
        Args:
            validation_level: Level of validation rigor
            significance_level: Statistical significance threshold
            min_sample_size: Minimum samples required for tests
            enable_statistical_tests: Whether to run statistical tests
        """
        self.validation_level = validation_level
        self.significance_level = significance_level
        self.min_sample_size = min_sample_size
        self.enable_statistical_tests = enable_statistical_tests
        
        # Validation results storage
        self.validation_results: List[ValidationResult] = []
        self.theoretical_validations: List[TheoreticalBoundValidation] = []
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Statistical test implementations
        self.statistical_tests = {
            StatisticalTest.T_TEST: self._t_test,
            StatisticalTest.WILCOXON: self._wilcoxon_test,
            StatisticalTest.MANN_WHITNEY: self._mann_whitney_test,
            StatisticalTest.KOLMOGOROV_SMIRNOV: self._ks_test,
            StatisticalTest.PERMUTATION_TEST: self._permutation_test,
            StatisticalTest.BOOTSTRAP: self._bootstrap_test
        }
        
        logger.info(f"Algorithm validator initialized with {validation_level.value} level")
    
    def validate_algorithm_performance(
        self,
        algorithm_results: Dict[str, List[float]],
        baseline_results: Dict[str, List[float]],
        performance_thresholds: Optional[Dict[str, float]] = None
    ) -> Dict[str, ValidationResult]:
        """Validate algorithm performance against baselines.
        
        Args:
            algorithm_results: Results from algorithm being tested
            baseline_results: Baseline comparison results
            performance_thresholds: Minimum performance thresholds
            
        Returns:
            Dictionary of validation results by metric
        """
        results = {}
        
        # Validate input data
        self._validate_input_data(algorithm_results, baseline_results)
        
        for metric_name in algorithm_results.keys():
            if metric_name not in baseline_results:
                logger.warning(f"Baseline missing for metric {metric_name}")
                continue
            
            alg_values = algorithm_results[metric_name]
            baseline_values = baseline_results[metric_name]
            
            # Basic statistical comparison
            results[metric_name] = self._compare_performance(
                alg_values,
                baseline_values,
                metric_name,
                performance_thresholds.get(metric_name) if performance_thresholds else None
            )
        
        with self._lock:
            self.validation_results.extend(results.values())
        
        return results
    
    def validate_theoretical_bounds(
        self,
        theoretical_bounds: List[float],
        empirical_violations: List[float],
        confidence_level: float = 0.95
    ) -> TheoreticalBoundValidation:
        """Validate theoretical bounds against empirical data.
        
        Args:
            theoretical_bounds: Theoretical risk bounds
            empirical_violations: Observed violation rates
            confidence_level: Confidence level for bounds
            
        Returns:
            Theoretical bound validation result
        """
        if len(theoretical_bounds) != len(empirical_violations):
            raise ValidationError("Theoretical bounds and empirical violations must have same length")
        
        if len(theoretical_bounds) < self.min_sample_size:
            logger.warning(f"Sample size {len(theoretical_bounds)} below minimum {self.min_sample_size}")
        
        # Check if empirical violations are within theoretical bounds
        violations_within_bounds = sum(
            emp <= theo for emp, theo in zip(empirical_violations, theoretical_bounds)
        )
        
        empirical_coverage = violations_within_bounds / len(theoretical_bounds)
        expected_coverage = confidence_level
        
        # Statistical test for coverage
        if SCIPY_AVAILABLE and len(theoretical_bounds) >= self.min_sample_size:
            # Binomial test for coverage probability
            p_value = stats.binom_test(
                violations_within_bounds,
                len(theoretical_bounds),
                expected_coverage
            )
            
            # Margin of error for coverage
            margin_of_error = 1.96 * math.sqrt(
                empirical_coverage * (1 - empirical_coverage) / len(theoretical_bounds)
            )
        else:
            p_value = None
            margin_of_error = 0.1  # Conservative estimate
        
        # Bound tightness (average relative gap)
        bound_tightness = np.mean([
            (theo - emp) / max(theo, 1e-6) for theo, emp in zip(theoretical_bounds, empirical_violations)
        ]) if theoretical_bounds else 0.0
        
        # Overall validation
        valid = (
            empirical_coverage >= expected_coverage - margin_of_error and
            (p_value is None or p_value >= self.significance_level)
        )
        
        validation = TheoreticalBoundValidation(
            theoretical_bound=np.mean(theoretical_bounds),
            empirical_violation_rate=np.mean(empirical_violations),
            confidence_level=confidence_level,
            sample_size=len(theoretical_bounds),
            bound_tightness=bound_tightness,
            valid=valid,
            margin_of_error=margin_of_error
        )
        
        with self._lock:
            self.theoretical_validations.append(validation)
        
        logger.info(f"Theoretical bounds validation: valid={valid}, coverage={empirical_coverage:.3f}")
        
        return validation
    
    def validate_conformal_prediction(
        self,
        predictions: List[float],
        true_values: List[float],
        prediction_sets: List[Tuple[float, float]],
        confidence_level: float = 0.95
    ) -> ValidationResult:
        """Validate conformal prediction coverage and efficiency.
        
        Args:
            predictions: Point predictions
            true_values: True target values
            prediction_sets: Conformal prediction intervals
            confidence_level: Target confidence level
            
        Returns:
            Validation result for conformal prediction
        """
        if len(predictions) != len(true_values) or len(predictions) != len(prediction_sets):
            raise ValidationError("All input lists must have same length")
        
        # Coverage validation
        coverage_count = sum(
            lower <= true_val <= upper
            for true_val, (lower, upper) in zip(true_values, prediction_sets)
        )
        empirical_coverage = coverage_count / len(predictions)
        
        # Efficiency (average interval width)
        avg_width = np.mean([upper - lower for lower, upper in prediction_sets])
        
        # Statistical test for coverage
        if SCIPY_AVAILABLE and len(predictions) >= self.min_sample_size:
            p_value = stats.binom_test(coverage_count, len(predictions), confidence_level)
        else:
            p_value = None
        
        # Validation criteria
        coverage_valid = empirical_coverage >= confidence_level - 0.05  # 5% tolerance
        
        result = ValidationResult(
            test_name="conformal_prediction_validation",
            passed=coverage_valid,
            score=empirical_coverage,
            threshold=confidence_level,
            p_value=p_value,
            metadata={
                'empirical_coverage': empirical_coverage,
                'average_width': avg_width,
                'sample_size': len(predictions),
                'efficiency_score': 1.0 / max(avg_width, 1e-6)  # Inverse of width
            }
        )
        
        with self._lock:
            self.validation_results.append(result)
        
        return result
    
    def validate_risk_controller(
        self,
        risk_scores: List[float],
        risk_threshold: float,
        target_violation_rate: float
    ) -> ValidationResult:
        """Validate risk controller performance.
        
        Args:
            risk_scores: Observed risk scores
            risk_threshold: Risk threshold
            target_violation_rate: Target violation rate
            
        Returns:
            Risk controller validation result
        """
        violations = [score > risk_threshold for score in risk_scores]
        violation_rate = sum(violations) / len(violations)
        
        # Test if violation rate is significantly below target
        if SCIPY_AVAILABLE and len(risk_scores) >= self.min_sample_size:
            p_value = stats.binom_test(
                sum(violations),
                len(violations),
                target_violation_rate,
                alternative='less'
            )
        else:
            p_value = None
        
        # Conservative validation
        passed = violation_rate <= target_violation_rate + 0.02  # 2% tolerance
        
        result = ValidationResult(
            test_name="risk_controller_validation",
            passed=passed,
            score=violation_rate,
            threshold=target_violation_rate,
            p_value=p_value,
            metadata={
                'violation_count': sum(violations),
                'total_samples': len(violations),
                'risk_threshold': risk_threshold
            }
        )
        
        with self._lock:
            self.validation_results.append(result)
        
        return result
    
    def _validate_input_data(
        self,
        algorithm_results: Dict[str, List[float]],
        baseline_results: Dict[str, List[float]]
    ) -> None:
        """Validate input data quality."""
        for metric_name, values in algorithm_results.items():
            if not values:
                raise ValidationError(f"Empty results for metric {metric_name}")
            
            if len(values) < 2:
                logger.warning(f"Insufficient data for metric {metric_name}: {len(values)} samples")
            
            # Check for invalid values
            if any(not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v) for v in values):
                raise ValidationError(f"Invalid values detected in {metric_name}")
        
        for metric_name, values in baseline_results.items():
            if not values:
                logger.warning(f"Empty baseline for metric {metric_name}")
                continue
            
            if any(not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v) for v in values):
                raise ValidationError(f"Invalid baseline values detected in {metric_name}")
    
    def _compare_performance(
        self,
        algorithm_values: List[float],
        baseline_values: List[float],
        metric_name: str,
        threshold: Optional[float] = None
    ) -> ValidationResult:
        """Compare algorithm performance against baseline."""
        alg_mean = np.mean(algorithm_values)
        baseline_mean = np.mean(baseline_values)
        
        # Effect size (Cohen's d)
        pooled_std = math.sqrt(
            ((len(algorithm_values) - 1) * np.var(algorithm_values, ddof=1) +
             (len(baseline_values) - 1) * np.var(baseline_values, ddof=1)) /
            (len(algorithm_values) + len(baseline_values) - 2)
        )
        
        effect_size = (alg_mean - baseline_mean) / max(pooled_std, 1e-6)
        
        # Statistical test
        p_value = None
        if (SCIPY_AVAILABLE and self.enable_statistical_tests and
            len(algorithm_values) >= 5 and len(baseline_values) >= 5):
            
            # Choose appropriate test
            if len(algorithm_values) >= 30 and len(baseline_values) >= 30:
                # Use t-test for large samples
                statistic, p_value = stats.ttest_ind(algorithm_values, baseline_values)
            else:
                # Use Mann-Whitney U test for small samples
                statistic, p_value = stats.mannwhitneyu(
                    algorithm_values, baseline_values, alternative='two-sided'
                )
        
        # Confidence interval for mean difference
        if SCIPY_AVAILABLE and len(algorithm_values) >= 5 and len(baseline_values) >= 5:
            # Bootstrap confidence interval for mean difference
            diff_mean = alg_mean - baseline_mean
            # Simplified CI - would use proper bootstrap in production
            pooled_se = pooled_std * math.sqrt(1/len(algorithm_values) + 1/len(baseline_values))
            ci_margin = 1.96 * pooled_se
            confidence_interval = (diff_mean - ci_margin, diff_mean + ci_margin)
        else:
            confidence_interval = None
        
        # Determine if improvement is significant
        if threshold is not None:
            passed = alg_mean >= threshold
            score = alg_mean
            test_threshold = threshold
        else:
            # Check if significantly better than baseline
            passed = (
                alg_mean > baseline_mean and
                (p_value is None or p_value < self.significance_level) and
                abs(effect_size) > 0.2  # Small effect size threshold
            )
            score = alg_mean - baseline_mean
            test_threshold = 0.0
        
        return ValidationResult(
            test_name=f"performance_comparison_{metric_name}",
            passed=passed,
            score=score,
            threshold=test_threshold,
            confidence_interval=confidence_interval,
            p_value=p_value,
            effect_size=effect_size,
            metadata={
                'algorithm_mean': alg_mean,
                'baseline_mean': baseline_mean,
                'algorithm_std': np.std(algorithm_values),
                'baseline_std': np.std(baseline_values),
                'sample_sizes': (len(algorithm_values), len(baseline_values))
            }
        )
    
    def _t_test(self, sample1: List[float], sample2: List[float]) -> Tuple[float, float]:
        """Perform t-test."""
        if not SCIPY_AVAILABLE:
            logger.warning("SciPy not available, skipping t-test")
            return 0.0, 1.0
        
        statistic, p_value = stats.ttest_ind(sample1, sample2)
        return statistic, p_value
    
    def _wilcoxon_test(self, sample1: List[float], sample2: List[float]) -> Tuple[float, float]:
        """Perform Wilcoxon rank-sum test."""
        if not SCIPY_AVAILABLE:
            logger.warning("SciPy not available, skipping Wilcoxon test")
            return 0.0, 1.0
        
        if len(sample1) != len(sample2):
            logger.warning("Wilcoxon test requires equal sample sizes, using Mann-Whitney instead")
            return self._mann_whitney_test(sample1, sample2)
        
        statistic, p_value = stats.wilcoxon(sample1, sample2)
        return statistic, p_value
    
    def _mann_whitney_test(self, sample1: List[float], sample2: List[float]) -> Tuple[float, float]:
        """Perform Mann-Whitney U test."""
        if not SCIPY_AVAILABLE:
            logger.warning("SciPy not available, skipping Mann-Whitney test")
            return 0.0, 1.0
        
        statistic, p_value = stats.mannwhitneyu(sample1, sample2, alternative='two-sided')
        return statistic, p_value
    
    def _ks_test(self, sample1: List[float], sample2: List[float]) -> Tuple[float, float]:
        """Perform Kolmogorov-Smirnov test."""
        if not SCIPY_AVAILABLE:
            logger.warning("SciPy not available, skipping KS test")
            return 0.0, 1.0
        
        statistic, p_value = stats.ks_2samp(sample1, sample2)
        return statistic, p_value
    
    def _permutation_test(self, sample1: List[float], sample2: List[float]) -> Tuple[float, float]:
        """Perform permutation test."""
        # Simple permutation test implementation
        combined = sample1 + sample2
        n1, n2 = len(sample1), len(sample2)
        
        observed_diff = np.mean(sample1) - np.mean(sample2)
        
        # Perform permutations
        num_permutations = min(1000, math.factorial(min(n1 + n2, 10)))
        larger_diffs = 0
        
        for _ in range(num_permutations):
            # Shuffle and split
            np.random.shuffle(combined)
            perm_sample1 = combined[:n1]
            perm_sample2 = combined[n1:]
            
            perm_diff = np.mean(perm_sample1) - np.mean(perm_sample2)
            if abs(perm_diff) >= abs(observed_diff):
                larger_diffs += 1
        
        p_value = larger_diffs / num_permutations
        return observed_diff, p_value
    
    def _bootstrap_test(self, sample1: List[float], sample2: List[float]) -> Tuple[float, float]:
        """Perform bootstrap test."""
        n_bootstrap = 1000
        
        # Bootstrap mean differences
        bootstrap_diffs = []
        for _ in range(n_bootstrap):
            boot_sample1 = [sample1[i] for i in np.random.choice(len(sample1), len(sample1), replace=True)]
            boot_sample2 = [sample2[i] for i in np.random.choice(len(sample2), len(sample2), replace=True)]
            
            boot_diff = np.mean(boot_sample1) - np.mean(boot_sample2)
            bootstrap_diffs.append(boot_diff)
        
        # Calculate p-value (two-tailed test against zero difference)
        observed_diff = np.mean(sample1) - np.mean(sample2)
        extreme_count = sum(1 for diff in bootstrap_diffs if abs(diff) >= abs(observed_diff))
        p_value = extreme_count / n_bootstrap
        
        return observed_diff, p_value
    
    def run_comprehensive_validation(
        self,
        algorithm_data: Dict[str, Any],
        baseline_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run comprehensive validation suite.
        
        Args:
            algorithm_data: Algorithm performance data
            baseline_data: Baseline comparison data
            
        Returns:
            Comprehensive validation report
        """
        validation_report = {
            'timestamp': time.time(),
            'validation_level': self.validation_level.value,
            'tests_performed': [],
            'overall_passed': True,
            'critical_failures': [],
            'warnings': []
        }
        
        try:
            # Performance validation
            if 'performance_metrics' in algorithm_data and 'performance_metrics' in baseline_data:
                perf_results = self.validate_algorithm_performance(
                    algorithm_data['performance_metrics'],
                    baseline_data['performance_metrics']
                )
                validation_report['performance_validation'] = {
                    name: {
                        'passed': result.passed,
                        'score': result.score,
                        'p_value': result.p_value,
                        'effect_size': result.effect_size
                    } for name, result in perf_results.items()
                }
                
                validation_report['tests_performed'].append('performance_comparison')
                
                # Check for critical failures
                failed_tests = [name for name, result in perf_results.items() if not result.passed]
                if failed_tests:
                    validation_report['overall_passed'] = False
                    validation_report['critical_failures'].extend(failed_tests)
            
            # Theoretical bounds validation
            if ('theoretical_bounds' in algorithm_data and 
                'empirical_violations' in algorithm_data):
                
                bound_validation = self.validate_theoretical_bounds(
                    algorithm_data['theoretical_bounds'],
                    algorithm_data['empirical_violations']
                )
                
                validation_report['theoretical_validation'] = {
                    'valid': bound_validation.valid,
                    'empirical_coverage': bound_validation.empirical_violation_rate,
                    'bound_tightness': bound_validation.bound_tightness,
                    'sample_size': bound_validation.sample_size
                }
                
                validation_report['tests_performed'].append('theoretical_bounds')
                
                if not bound_validation.valid:
                    validation_report['overall_passed'] = False
                    validation_report['critical_failures'].append('theoretical_bounds_invalid')
            
            # Conformal prediction validation
            if all(key in algorithm_data for key in ['predictions', 'true_values', 'prediction_sets']):
                conformal_result = self.validate_conformal_prediction(
                    algorithm_data['predictions'],
                    algorithm_data['true_values'],
                    algorithm_data['prediction_sets']
                )
                
                validation_report['conformal_validation'] = {
                    'passed': conformal_result.passed,
                    'coverage': conformal_result.score,
                    'p_value': conformal_result.p_value,
                    'efficiency': conformal_result.metadata.get('efficiency_score', 0.0)
                }
                
                validation_report['tests_performed'].append('conformal_prediction')
                
                if not conformal_result.passed:
                    validation_report['warnings'].append('conformal_coverage_below_target')
            
        except Exception as e:
            logger.error(f"Validation failed with error: {e}")
            validation_report['overall_passed'] = False
            validation_report['critical_failures'].append(f"validation_error: {str(e)}")
        
        # Summary statistics
        validation_report['summary'] = {
            'total_tests': len(validation_report['tests_performed']),
            'tests_passed': validation_report['overall_passed'],
            'critical_failure_count': len(validation_report['critical_failures']),
            'warning_count': len(validation_report['warnings'])
        }
        
        logger.info(f"Comprehensive validation completed: passed={validation_report['overall_passed']}")
        
        return validation_report
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validation results."""
        with self._lock:
            passed_tests = sum(1 for result in self.validation_results if result.passed)
            total_tests = len(self.validation_results)
            
            # Recent test results
            recent_results = [r for r in self.validation_results if time.time() - r.timestamp < 3600]
            
            return {
                'total_tests_run': total_tests,
                'tests_passed': passed_tests,
                'pass_rate': passed_tests / max(1, total_tests),
                'recent_tests_1h': len(recent_results),
                'theoretical_validations': len(self.theoretical_validations),
                'validation_level': self.validation_level.value,
                'significance_level': self.significance_level,
                'min_sample_size': self.min_sample_size
            }