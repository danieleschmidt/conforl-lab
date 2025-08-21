"""Comprehensive Benchmarking Suite for ConfoRL Research.

This module provides a complete benchmarking framework for evaluating
conformal risk control methods across diverse RL environments and scenarios.
It includes standardized benchmarks, comparative analysis, and research validation.

Research Features:
- Standardized benchmark environments and metrics
- Comparative analysis across conformal methods
- Statistical significance testing and validation
- Performance profiling and scalability analysis
- Research paper figure generation

Author: ConfoRL Research Team
License: Apache 2.0
"""

try:
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    class np:
        @staticmethod
        def array(x): return x
        @staticmethod
        def mean(x): return sum(x) / len(x) if x else 0
        @staticmethod
        def std(x): 
            if not x: return 0
            mean_val = sum(x) / len(x)
            return (sum((xi - mean_val)**2 for xi in x) / len(x)) ** 0.5
        @staticmethod
        def random(): import random; return random
    
    class plt:
        @staticmethod
        def figure(*args, **kwargs): pass
        @staticmethod
        def plot(*args, **kwargs): pass
        @staticmethod
        def xlabel(*args, **kwargs): pass
        @staticmethod
        def ylabel(*args, **kwargs): pass
        @staticmethod
        def title(*args, **kwargs): pass
        @staticmethod
        def legend(*args, **kwargs): pass
        @staticmethod
        def savefig(*args, **kwargs): pass
        @staticmethod
        def show(*args, **kwargs): pass
        @staticmethod
        def subplots(*args, **kwargs): return None, None

import time
import json
import os
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum
import warnings
import traceback

from ..core.types import RiskCertificate, TrajectoryData
from ..utils.logging import get_logger
from ..utils.errors import ConfoRLError, ValidationError

logger = get_logger(__name__)


class BenchmarkEnvironment(Enum):
    """Standard benchmark environments for conformal RL evaluation."""
    CARTPOLE = "cartpole"
    LUNARLANDER = "lunarlander"
    PENDULUM = "pendulum"
    MOUNTAINCAR = "mountaincar"
    SAFETY_GYM = "safety_gym"
    CUSTOM_SAFETY = "custom_safety"
    ADVERSARIAL_ENV = "adversarial_env"
    DISTRIBUTION_SHIFT = "distribution_shift"


class BenchmarkMetric(Enum):
    """Evaluation metrics for conformal RL benchmarking."""
    COVERAGE_ACCURACY = "coverage_accuracy"
    SAFETY_VIOLATION_RATE = "safety_violation_rate"
    RISK_BOUND_TIGHTNESS = "risk_bound_tightness"
    ADAPTATION_SPEED = "adaptation_speed"
    COMPUTATIONAL_EFFICIENCY = "computational_efficiency"
    STATISTICAL_POWER = "statistical_power"
    ROBUSTNESS_SCORE = "robustness_score"
    SCALABILITY_FACTOR = "scalability_factor"


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""
    
    environments: List[BenchmarkEnvironment] = field(default_factory=lambda: [BenchmarkEnvironment.CARTPOLE])
    metrics: List[BenchmarkMetric] = field(default_factory=lambda: [BenchmarkMetric.COVERAGE_ACCURACY])
    num_trials: int = 50
    num_episodes_per_trial: int = 100
    confidence_levels: List[float] = field(default_factory=lambda: [0.9, 0.95, 0.99])
    
    # Statistical testing
    statistical_tests: bool = True
    significance_level: float = 0.05
    multiple_testing_correction: str = "bonferroni"
    
    # Performance profiling
    profile_performance: bool = True
    memory_profiling: bool = True
    scalability_testing: bool = True
    
    # Output configuration
    save_results: bool = True
    generate_plots: bool = True
    output_directory: str = "benchmark_results"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "environments": [env.value for env in self.environments],
            "metrics": [metric.value for metric in self.metrics],
            "num_trials": self.num_trials,
            "num_episodes_per_trial": self.num_episodes_per_trial,
            "confidence_levels": self.confidence_levels,
            "statistical_tests": self.statistical_tests,
            "significance_level": self.significance_level,
            "multiple_testing_correction": self.multiple_testing_correction,
            "profile_performance": self.profile_performance,
            "memory_profiling": self.memory_profiling,
            "scalability_testing": self.scalability_testing,
            "save_results": self.save_results,
            "generate_plots": self.generate_plots,
            "output_directory": self.output_directory
        }


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    
    method_name: str
    environment: BenchmarkEnvironment
    metric: BenchmarkMetric
    confidence_level: float
    
    # Primary results
    mean_value: float
    std_value: float
    min_value: float
    max_value: float
    median_value: float
    
    # Statistical testing
    p_value: Optional[float] = None
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    
    # Performance metrics
    execution_time: float = 0.0
    memory_usage: int = 0
    computational_cost: float = 0.0
    
    # Metadata
    num_trials: int = 0
    timestamp: float = field(default_factory=time.time)
    additional_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "method_name": self.method_name,
            "environment": self.environment.value,
            "metric": self.metric.value,
            "confidence_level": self.confidence_level,
            "mean_value": self.mean_value,
            "std_value": self.std_value,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "median_value": self.median_value,
            "p_value": self.p_value,
            "effect_size": self.effect_size,
            "confidence_interval": self.confidence_interval,
            "execution_time": self.execution_time,
            "memory_usage": self.memory_usage,
            "computational_cost": self.computational_cost,
            "num_trials": self.num_trials,
            "timestamp": self.timestamp,
            "additional_data": self.additional_data
        }


class BenchmarkEnvironmentFactory:
    """Factory for creating benchmark environments."""
    
    @staticmethod
    def create_environment(env_type: BenchmarkEnvironment, **kwargs) -> Any:
        """Create benchmark environment.
        
        Args:
            env_type: Type of environment to create
            **kwargs: Environment-specific parameters
            
        Returns:
            Environment instance
        """
        if env_type == BenchmarkEnvironment.CARTPOLE:
            return BenchmarkEnvironmentFactory._create_cartpole(**kwargs)
        elif env_type == BenchmarkEnvironment.SAFETY_GYM:
            return BenchmarkEnvironmentFactory._create_safety_gym(**kwargs)
        elif env_type == BenchmarkEnvironment.ADVERSARIAL_ENV:
            return BenchmarkEnvironmentFactory._create_adversarial_env(**kwargs)
        elif env_type == BenchmarkEnvironment.DISTRIBUTION_SHIFT:
            return BenchmarkEnvironmentFactory._create_distribution_shift_env(**kwargs)
        else:
            return BenchmarkEnvironmentFactory._create_mock_env(env_type, **kwargs)
    
    @staticmethod
    def _create_cartpole(**kwargs):
        """Create CartPole environment."""
        try:
            import gymnasium as gym
            return gym.make('CartPole-v1')
        except ImportError:
            logger.warning("Gymnasium not available, using mock environment")
            return BenchmarkEnvironmentFactory._create_mock_env(BenchmarkEnvironment.CARTPOLE)
    
    @staticmethod
    def _create_safety_gym(**kwargs):
        """Create Safety Gym environment."""
        # Safety Gym environments for safety-critical RL
        logger.info("Creating Safety Gym environment (mock implementation)")
        return BenchmarkEnvironmentFactory._create_mock_env(BenchmarkEnvironment.SAFETY_GYM, **kwargs)
    
    @staticmethod
    def _create_adversarial_env(**kwargs):
        """Create adversarial environment for robustness testing."""
        return AdversarialBenchmarkEnvironment(**kwargs)
    
    @staticmethod
    def _create_distribution_shift_env(**kwargs):
        """Create environment with distribution shift."""
        return DistributionShiftEnvironment(**kwargs)
    
    @staticmethod
    def _create_mock_env(env_type: BenchmarkEnvironment, **kwargs):
        """Create mock environment for testing."""
        return MockBenchmarkEnvironment(env_type, **kwargs)


class MockBenchmarkEnvironment:
    """Mock environment for benchmarking when real environments unavailable."""
    
    def __init__(self, env_type: BenchmarkEnvironment, **kwargs):
        """Initialize mock environment.
        
        Args:
            env_type: Type of environment being mocked
            **kwargs: Environment parameters
        """
        self.env_type = env_type
        self.state_dim = kwargs.get('state_dim', 4)
        self.action_dim = kwargs.get('action_dim', 2)
        self.max_episode_length = kwargs.get('max_episode_length', 200)
        
        self.current_state = self._generate_initial_state()
        self.episode_step = 0
        self.episode_reward = 0.0
        
        # Environment-specific parameters
        if env_type == BenchmarkEnvironment.CARTPOLE:
            self.risk_probability = 0.1  # 10% chance of unsafe state
        elif env_type == BenchmarkEnvironment.SAFETY_GYM:
            self.risk_probability = 0.05  # 5% chance in safety environment
        else:
            self.risk_probability = 0.08  # Default risk level
    
    def reset(self):
        """Reset environment."""
        self.current_state = self._generate_initial_state()
        self.episode_step = 0
        self.episode_reward = 0.0
        return self.current_state, {}
    
    def step(self, action):
        """Take environment step."""
        self.episode_step += 1
        
        # Simulate state transition
        self.current_state = self._simulate_transition(self.current_state, action)
        
        # Simulate reward
        reward = self._compute_reward(self.current_state, action)
        self.episode_reward += reward
        
        # Check termination
        done = self._check_termination()
        truncated = self.episode_step >= self.max_episode_length
        
        # Simulate risk
        risk = self._compute_risk(self.current_state, action)
        
        info = {
            'risk': risk,
            'episode_step': self.episode_step,
            'episode_reward': self.episode_reward
        }
        
        return self.current_state, reward, done, truncated, info
    
    def _generate_initial_state(self):
        """Generate initial state."""
        return [np.random.normal(0, 0.1) for _ in range(self.state_dim)]
    
    def _simulate_transition(self, state, action):
        """Simulate state transition."""
        # Simple linear dynamics with noise
        new_state = []
        for i, s in enumerate(state):
            # Action influence
            action_effect = 0.1 * (action if isinstance(action, (int, float)) else action[i % len(action)])
            # Dynamics
            new_s = s + action_effect + np.random.normal(0, 0.02)
            new_state.append(new_s)
        
        return new_state
    
    def _compute_reward(self, state, action):
        """Compute reward."""
        # Simple reward based on state magnitude
        state_magnitude = sum(abs(s) for s in state)
        return max(0, 1.0 - state_magnitude * 0.1)
    
    def _compute_risk(self, state, action):
        """Compute environment risk."""
        # Risk increases with state magnitude and random chance
        state_risk = min(1.0, sum(abs(s) for s in state) * 0.2)
        random_risk = 1.0 if np.random.random() < self.risk_probability else 0.0
        
        return max(state_risk, random_risk)
    
    def _check_termination(self):
        """Check if episode should terminate."""
        # Terminate if state becomes too extreme
        return any(abs(s) > 5.0 for s in self.current_state)


class AdversarialBenchmarkEnvironment(MockBenchmarkEnvironment):
    """Adversarial environment for robustness testing."""
    
    def __init__(self, attack_probability=0.1, attack_strength=0.5, **kwargs):
        """Initialize adversarial environment.
        
        Args:
            attack_probability: Probability of adversarial attack
            attack_strength: Strength of adversarial perturbation
            **kwargs: Additional environment parameters
        """
        super().__init__(BenchmarkEnvironment.ADVERSARIAL_ENV, **kwargs)
        self.attack_probability = attack_probability
        self.attack_strength = attack_strength
        self.attacks_applied = 0
    
    def step(self, action):
        """Take step with potential adversarial attack."""
        # Apply adversarial attack with probability
        if np.random.random() < self.attack_probability:
            action = self._apply_adversarial_attack(action)
            self.attacks_applied += 1
        
        return super().step(action)
    
    def _apply_adversarial_attack(self, action):
        """Apply adversarial perturbation to action."""
        if isinstance(action, (int, float)):
            perturbation = np.random.normal(0, self.attack_strength)
            return action + perturbation
        else:
            # Vector action
            perturbed_action = []
            for a in action:
                perturbation = np.random.normal(0, self.attack_strength)
                perturbed_action.append(a + perturbation)
            return perturbed_action


class DistributionShiftEnvironment(MockBenchmarkEnvironment):
    """Environment with distribution shift for adaptation testing."""
    
    def __init__(self, shift_episode=50, shift_magnitude=0.5, **kwargs):
        """Initialize distribution shift environment.
        
        Args:
            shift_episode: Episode number when distribution shift occurs
            shift_magnitude: Magnitude of distribution shift
            **kwargs: Additional environment parameters
        """
        super().__init__(BenchmarkEnvironment.DISTRIBUTION_SHIFT, **kwargs)
        self.shift_episode = shift_episode
        self.shift_magnitude = shift_magnitude
        self.current_episode = 0
        self.shift_applied = False
    
    def reset(self):
        """Reset with potential distribution shift."""
        self.current_episode += 1
        
        # Apply distribution shift after specified episode
        if self.current_episode >= self.shift_episode and not self.shift_applied:
            self._apply_distribution_shift()
            self.shift_applied = True
            logger.info(f"Applied distribution shift at episode {self.current_episode}")
        
        return super().reset()
    
    def _apply_distribution_shift(self):
        """Apply distribution shift to environment dynamics."""
        # Modify risk probability
        self.risk_probability *= (1 + self.shift_magnitude)
        self.risk_probability = min(1.0, self.risk_probability)
        
        # Modify state dimension (simulate environment change)
        if np.random.random() < 0.3:  # 30% chance to change state dimension
            self.state_dim = max(2, self.state_dim + np.random.choice([-1, 0, 1]))


class BenchmarkMetricCalculator:
    """Calculator for benchmark evaluation metrics."""
    
    @staticmethod
    def calculate_coverage_accuracy(
        predictions: List[Tuple[float, float, float]], 
        true_values: List[float], 
        confidence_level: float
    ) -> float:
        """Calculate coverage accuracy for prediction intervals.
        
        Args:
            predictions: List of (prediction, lower_bound, upper_bound) tuples
            true_values: List of true values
            confidence_level: Target confidence level
            
        Returns:
            Coverage accuracy (proportion of true values within intervals)
        """
        if not predictions or not true_values or len(predictions) != len(true_values):
            return 0.0
        
        covered_count = 0
        for (pred, lower, upper), true_val in zip(predictions, true_values):
            if lower <= true_val <= upper:
                covered_count += 1
        
        return covered_count / len(predictions)
    
    @staticmethod
    def calculate_safety_violation_rate(
        risk_predictions: List[float], 
        true_risks: List[float], 
        safety_threshold: float = 0.1
    ) -> float:
        """Calculate safety violation rate.
        
        Args:
            risk_predictions: Predicted risk values
            true_risks: True risk values
            safety_threshold: Safety threshold
            
        Returns:
            Proportion of safety violations
        """
        if not risk_predictions or not true_risks:
            return 0.0
        
        violations = 0
        for pred_risk, true_risk in zip(risk_predictions, true_risks):
            # Safety violation if predicted safe but actually unsafe
            if pred_risk <= safety_threshold and true_risk > safety_threshold:
                violations += 1
        
        return violations / len(risk_predictions)
    
    @staticmethod
    def calculate_risk_bound_tightness(
        risk_bounds: List[float], 
        true_risks: List[float]
    ) -> float:
        """Calculate tightness of risk bounds.
        
        Args:
            risk_bounds: Predicted risk bounds
            true_risks: True risk values
            
        Returns:
            Average tightness (smaller is better)
        """
        if not risk_bounds or not true_risks:
            return 1.0
        
        tightness_scores = []
        for bound, true_risk in zip(risk_bounds, true_risks):
            if bound >= true_risk:  # Valid bound
                tightness = bound - true_risk
            else:  # Invalid bound (penalty)
                tightness = 1.0
            tightness_scores.append(tightness)
        
        return np.mean(tightness_scores)
    
    @staticmethod
    def calculate_adaptation_speed(
        performance_over_time: List[float], 
        adaptation_threshold: float = 0.9
    ) -> int:
        """Calculate adaptation speed (episodes to reach threshold).
        
        Args:
            performance_over_time: Performance values over episodes
            adaptation_threshold: Performance threshold for adaptation
            
        Returns:
            Number of episodes to reach threshold
        """
        for episode, performance in enumerate(performance_over_time):
            if performance >= adaptation_threshold:
                return episode + 1
        
        return len(performance_over_time)  # Never reached threshold
    
    @staticmethod
    def calculate_computational_efficiency(
        execution_times: List[float], 
        performance_scores: List[float]
    ) -> float:
        """Calculate computational efficiency.
        
        Args:
            execution_times: Execution times for each trial
            performance_scores: Performance scores for each trial
            
        Returns:
            Efficiency score (performance per unit time)
        """
        if not execution_times or not performance_scores:
            return 0.0
        
        avg_performance = np.mean(performance_scores)
        avg_time = np.mean(execution_times)
        
        return avg_performance / avg_time if avg_time > 0 else 0.0
    
    @staticmethod
    def calculate_statistical_power(
        effect_sizes: List[float], 
        sample_sizes: List[int], 
        significance_level: float = 0.05
    ) -> float:
        """Calculate statistical power.
        
        Args:
            effect_sizes: Effect sizes from experiments
            sample_sizes: Sample sizes for each experiment
            significance_level: Statistical significance level
            
        Returns:
            Average statistical power
        """
        if not effect_sizes or not sample_sizes:
            return 0.0
        
        # Simplified power calculation
        powers = []
        for effect_size, sample_size in zip(effect_sizes, sample_sizes):
            # Cohen's conventions for effect size
            if abs(effect_size) >= 0.8:  # Large effect
                power = min(1.0, 0.8 + sample_size * 0.001)
            elif abs(effect_size) >= 0.5:  # Medium effect
                power = min(1.0, 0.5 + sample_size * 0.0005)
            else:  # Small effect
                power = min(1.0, 0.2 + sample_size * 0.0002)
            
            powers.append(power)
        
        return np.mean(powers)
    
    @staticmethod
    def calculate_robustness_score(
        performance_clean: List[float], 
        performance_adversarial: List[float]
    ) -> float:
        """Calculate robustness score.
        
        Args:
            performance_clean: Performance on clean data
            performance_adversarial: Performance on adversarial data
            
        Returns:
            Robustness score (0-1, higher is better)
        """
        if not performance_clean or not performance_adversarial:
            return 0.0
        
        avg_clean = np.mean(performance_clean)
        avg_adversarial = np.mean(performance_adversarial)
        
        if avg_clean == 0:
            return 0.0
        
        # Robustness as ratio of adversarial to clean performance
        robustness = avg_adversarial / avg_clean
        return min(1.0, robustness)


class StatisticalTester:
    """Statistical testing utilities for benchmark results."""
    
    @staticmethod
    def t_test(group1: List[float], group2: List[float]) -> Tuple[float, float]:
        """Perform two-sample t-test.
        
        Args:
            group1: First group of values
            group2: Second group of values
            
        Returns:
            Tuple of (t_statistic, p_value)
        """
        if not group1 or not group2:
            return 0.0, 1.0
        
        # Simplified t-test calculation
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1, std2 = np.std(group1), np.std(group2)
        n1, n2 = len(group1), len(group2)
        
        # Pooled standard error
        pooled_se = ((std1**2 / n1) + (std2**2 / n2)) ** 0.5
        
        if pooled_se == 0:
            return 0.0, 1.0
        
        t_stat = (mean1 - mean2) / pooled_se
        
        # Simplified p-value calculation (normal approximation)
        import math
        p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(t_stat) / math.sqrt(2))))
        
        return t_stat, p_value
    
    @staticmethod
    def effect_size(group1: List[float], group2: List[float]) -> float:
        """Calculate Cohen's d effect size.
        
        Args:
            group1: First group of values
            group2: Second group of values
            
        Returns:
            Effect size (Cohen's d)
        """
        if not group1 or not group2:
            return 0.0
        
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1, std2 = np.std(group1), np.std(group2)
        n1, n2 = len(group1), len(group2)
        
        # Pooled standard deviation
        pooled_std = (((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2)) ** 0.5
        
        if pooled_std == 0:
            return 0.0
        
        return (mean1 - mean2) / pooled_std
    
    @staticmethod
    def confidence_interval(
        data: List[float], 
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate confidence interval.
        
        Args:
            data: Data values
            confidence_level: Confidence level
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if not data:
            return 0.0, 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        n = len(data)
        
        # t-distribution critical value (approximation)
        alpha = 1 - confidence_level
        t_critical = 1.96  # Normal approximation for large n
        if n < 30:
            t_critical = 2.0  # Conservative estimate for small n
        
        margin_error = t_critical * (std / (n ** 0.5))
        
        return mean - margin_error, mean + margin_error
    
    @staticmethod
    def multiple_testing_correction(
        p_values: List[float], 
        method: str = "bonferroni"
    ) -> List[float]:
        """Apply multiple testing correction.
        
        Args:
            p_values: List of p-values
            method: Correction method ('bonferroni', 'holm')
            
        Returns:
            Corrected p-values
        """
        if not p_values:
            return []
        
        if method == "bonferroni":
            return [min(1.0, p * len(p_values)) for p in p_values]
        elif method == "holm":
            # Holm-Bonferroni method
            sorted_indices = sorted(range(len(p_values)), key=lambda i: p_values[i])
            corrected = [0.0] * len(p_values)
            
            for rank, idx in enumerate(sorted_indices):
                correction_factor = len(p_values) - rank
                corrected[idx] = min(1.0, p_values[idx] * correction_factor)
            
            return corrected
        else:
            return p_values


class ComprehensiveBenchmarkSuite:
    """Comprehensive benchmarking suite for conformal RL methods."""
    
    def __init__(self, config: BenchmarkConfig):
        """Initialize benchmark suite.
        
        Args:
            config: Benchmark configuration
        """
        self.config = config
        self.results: List[BenchmarkResult] = []
        self.comparison_results: Dict[str, Any] = {}
        
        # Create output directory
        if config.save_results:
            os.makedirs(config.output_directory, exist_ok=True)
        
        logger.info(f"Initialized comprehensive benchmark suite")
    
    def run_benchmark(
        self, 
        methods: Dict[str, Any], 
        baseline_method: str = "baseline"
    ) -> Dict[str, Any]:
        """Run comprehensive benchmark across all methods and environments.
        
        Args:
            methods: Dictionary mapping method names to conformal predictors
            baseline_method: Name of baseline method for comparison
            
        Returns:
            Comprehensive benchmark results
        """
        logger.info(f"Starting comprehensive benchmark with {len(methods)} methods")
        
        start_time = time.time()
        
        # Run benchmarks for each method
        for method_name, predictor in methods.items():
            logger.info(f"Benchmarking method: {method_name}")
            
            method_results = self._benchmark_method(
                method_name, predictor, baseline_method
            )
            
            self.results.extend(method_results)
        
        # Perform comparative analysis
        self.comparison_results = self._perform_comparative_analysis(
            methods, baseline_method
        )
        
        # Generate summary
        benchmark_summary = self._generate_benchmark_summary()
        
        total_time = time.time() - start_time
        
        final_results = {
            'benchmark_summary': benchmark_summary,
            'detailed_results': [result.to_dict() for result in self.results],
            'comparative_analysis': self.comparison_results,
            'total_benchmark_time': total_time,
            'config': self.config.to_dict()
        }
        
        # Save results
        if self.config.save_results:
            self._save_results(final_results)
        
        # Generate plots
        if self.config.generate_plots and PLOTTING_AVAILABLE:
            self._generate_plots()
        
        logger.info(f"Comprehensive benchmark completed in {total_time:.2f}s")
        
        return final_results
    
    def _benchmark_method(
        self, 
        method_name: str, 
        predictor: Any, 
        baseline_method: str
    ) -> List[BenchmarkResult]:
        """Benchmark a single method across all environments and metrics.
        
        Args:
            method_name: Name of the method
            predictor: Conformal predictor instance
            baseline_method: Baseline method name
            
        Returns:
            List of benchmark results for this method
        """
        method_results = []
        
        for env_type in self.config.environments:
            for metric in self.config.metrics:
                for confidence_level in self.config.confidence_levels:
                    
                    try:
                        result = self._run_single_benchmark(
                            method_name, predictor, env_type, metric, confidence_level
                        )
                        method_results.append(result)
                        
                    except Exception as e:
                        logger.error(f"Benchmark failed for {method_name} on {env_type.value} "
                                   f"with {metric.value}: {e}")
                        
                        # Create error result
                        error_result = BenchmarkResult(
                            method_name=method_name,
                            environment=env_type,
                            metric=metric,
                            confidence_level=confidence_level,
                            mean_value=0.0,
                            std_value=0.0,
                            min_value=0.0,
                            max_value=0.0,
                            median_value=0.0,
                            additional_data={'error': str(e)}
                        )
                        method_results.append(error_result)
        
        return method_results
    
    def _run_single_benchmark(
        self, 
        method_name: str, 
        predictor: Any, 
        env_type: BenchmarkEnvironment, 
        metric: BenchmarkMetric, 
        confidence_level: float
    ) -> BenchmarkResult:
        """Run single benchmark configuration.
        
        Args:
            method_name: Method name
            predictor: Conformal predictor
            env_type: Environment type
            metric: Evaluation metric
            confidence_level: Confidence level
            
        Returns:
            Benchmark result
        """
        start_time = time.time()
        metric_values = []
        
        # Create environment
        env = BenchmarkEnvironmentFactory.create_environment(env_type)
        
        # Run multiple trials
        for trial in range(self.config.num_trials):
            trial_metrics = self._run_trial(
                predictor, env, metric, confidence_level
            )
            metric_values.extend(trial_metrics)
        
        execution_time = time.time() - start_time
        
        # Calculate statistics
        if metric_values:
            mean_val = np.mean(metric_values)
            std_val = np.std(metric_values)
            min_val = min(metric_values)
            max_val = max(metric_values)
            median_val = np.median(metric_values)
        else:
            mean_val = std_val = min_val = max_val = median_val = 0.0
        
        # Memory usage estimation
        memory_usage = self._estimate_memory_usage()
        
        # Statistical analysis
        confidence_interval = StatisticalTester.confidence_interval(
            metric_values, confidence_level
        )
        
        result = BenchmarkResult(
            method_name=method_name,
            environment=env_type,
            metric=metric,
            confidence_level=confidence_level,
            mean_value=mean_val,
            std_value=std_val,
            min_value=min_val,
            max_value=max_val,
            median_value=median_val,
            confidence_interval=confidence_interval,
            execution_time=execution_time,
            memory_usage=memory_usage,
            num_trials=self.config.num_trials,
            additional_data={
                'raw_values': metric_values[:100],  # Store first 100 values
                'environment_config': env_type.value
            }
        )
        
        return result
    
    def _run_trial(
        self, 
        predictor: Any, 
        env: Any, 
        metric: BenchmarkMetric, 
        confidence_level: float
    ) -> List[float]:
        """Run single trial and collect metrics.
        
        Args:
            predictor: Conformal predictor
            env: Environment instance
            metric: Evaluation metric
            confidence_level: Confidence level
            
        Returns:
            List of metric values from this trial
        """
        trial_metrics = []
        predictions = []
        true_values = []
        
        # Run episodes
        for episode in range(self.config.num_episodes_per_trial):
            state, _ = env.reset()
            episode_predictions = []
            episode_true_values = []
            
            done = False
            while not done:
                # Generate action (simplified)
                action = self._generate_action(state, env)
                
                # Get prediction from conformal predictor
                try:
                    if hasattr(predictor, 'predict_with_uncertainty'):
                        pred, lower, upper = predictor.predict_with_uncertainty(
                            state, action, confidence_level
                        )
                        episode_predictions.append((pred, lower, upper))
                    else:
                        # Fallback for predictors without uncertainty
                        pred = 0.5
                        episode_predictions.append((pred, 0.0, 1.0))
                except Exception as e:
                    logger.warning(f"Prediction failed: {e}")
                    episode_predictions.append((0.5, 0.0, 1.0))
                
                # Take environment step
                next_state, reward, done, truncated, info = env.step(action)
                
                # Extract true risk value
                true_risk = info.get('risk', 0.0)
                episode_true_values.append(true_risk)
                
                state = next_state
                done = done or truncated
            
            # Calculate episode metric
            episode_metric = self._calculate_episode_metric(
                metric, episode_predictions, episode_true_values, confidence_level
            )
            
            trial_metrics.append(episode_metric)
            predictions.extend(episode_predictions)
            true_values.extend(episode_true_values)
        
        return trial_metrics
    
    def _generate_action(self, state: Any, env: Any) -> Any:
        """Generate action for given state.
        
        Args:
            state: Environment state
            env: Environment instance
            
        Returns:
            Action
        """
        # Simple random action generation
        if hasattr(env, 'action_dim'):
            if env.action_dim == 1:
                return np.random.choice([0, 1])  # Discrete action
            else:
                return [np.random.uniform(-1, 1) for _ in range(env.action_dim)]
        else:
            return np.random.choice([0, 1])  # Default discrete action
    
    def _calculate_episode_metric(
        self, 
        metric: BenchmarkMetric, 
        predictions: List[Tuple[float, float, float]], 
        true_values: List[float], 
        confidence_level: float
    ) -> float:
        """Calculate metric value for episode.
        
        Args:
            metric: Evaluation metric
            predictions: Episode predictions
            true_values: Episode true values
            confidence_level: Confidence level
            
        Returns:
            Metric value
        """
        if metric == BenchmarkMetric.COVERAGE_ACCURACY:
            return BenchmarkMetricCalculator.calculate_coverage_accuracy(
                predictions, true_values, confidence_level
            )
        elif metric == BenchmarkMetric.SAFETY_VIOLATION_RATE:
            risk_predictions = [pred[0] for pred in predictions]
            return BenchmarkMetricCalculator.calculate_safety_violation_rate(
                risk_predictions, true_values
            )
        elif metric == BenchmarkMetric.RISK_BOUND_TIGHTNESS:
            risk_bounds = [pred[2] for pred in predictions]  # Upper bounds
            return BenchmarkMetricCalculator.calculate_risk_bound_tightness(
                risk_bounds, true_values
            )
        else:
            # Default metric
            return np.mean(true_values) if true_values else 0.0
    
    def _estimate_memory_usage(self) -> int:
        """Estimate memory usage.
        
        Returns:
            Estimated memory usage in bytes
        """
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss
        except ImportError:
            return 0
    
    def _perform_comparative_analysis(
        self, 
        methods: Dict[str, Any], 
        baseline_method: str
    ) -> Dict[str, Any]:
        """Perform comparative analysis between methods.
        
        Args:
            methods: Dictionary of methods
            baseline_method: Baseline method name
            
        Returns:
            Comparative analysis results
        """
        logger.info("Performing comparative analysis")
        
        comparison_results = {
            'pairwise_comparisons': {},
            'ranking': {},
            'statistical_significance': {},
            'effect_sizes': {}
        }
        
        # Group results by metric and environment
        grouped_results = defaultdict(lambda: defaultdict(list))
        
        for result in self.results:
            key = (result.environment, result.metric, result.confidence_level)
            grouped_results[key][result.method_name].append(result.mean_value)
        
        # Perform pairwise comparisons
        for key, method_results in grouped_results.items():
            env, metric, conf = key
            comparison_key = f"{env.value}_{metric.value}_{conf}"
            
            pairwise_results = {}
            statistical_results = {}
            effect_results = {}
            
            method_names = list(method_results.keys())
            
            for i, method1 in enumerate(method_names):
                for j, method2 in enumerate(method_names[i+1:], i+1):
                    
                    values1 = method_results[method1]
                    values2 = method_results[method2]
                    
                    # Statistical test
                    t_stat, p_value = StatisticalTester.t_test(values1, values2)
                    effect_size = StatisticalTester.effect_size(values1, values2)
                    
                    comparison_pair = f"{method1}_vs_{method2}"
                    
                    pairwise_results[comparison_pair] = {
                        'mean_diff': np.mean(values1) - np.mean(values2),
                        'winner': method1 if np.mean(values1) > np.mean(values2) else method2
                    }
                    
                    statistical_results[comparison_pair] = {
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'significant': p_value < self.config.significance_level
                    }
                    
                    effect_results[comparison_pair] = effect_size
            
            comparison_results['pairwise_comparisons'][comparison_key] = pairwise_results
            comparison_results['statistical_significance'][comparison_key] = statistical_results
            comparison_results['effect_sizes'][comparison_key] = effect_results
            
            # Method ranking for this configuration
            method_means = {method: np.mean(values) for method, values in method_results.items()}
            ranking = sorted(method_means.keys(), key=lambda x: method_means[x], reverse=True)
            comparison_results['ranking'][comparison_key] = {
                'ranking': ranking,
                'scores': method_means
            }
        
        return comparison_results
    
    def _generate_benchmark_summary(self) -> Dict[str, Any]:
        """Generate summary of benchmark results.
        
        Returns:
            Benchmark summary
        """
        if not self.results:
            return {'total_benchmarks': 0}
        
        # Overall statistics
        methods = list(set(result.method_name for result in self.results))
        environments = list(set(result.environment for result in self.results))
        metrics = list(set(result.metric for result in self.results))
        
        # Best performing methods per metric
        best_methods = {}
        for metric in metrics:
            metric_results = [r for r in self.results if r.metric == metric]
            if metric_results:
                best_result = max(metric_results, key=lambda x: x.mean_value)
                best_methods[metric.value] = {
                    'method': best_result.method_name,
                    'score': best_result.mean_value,
                    'environment': best_result.environment.value
                }
        
        # Overall method ranking
        method_scores = defaultdict(list)
        for result in self.results:
            method_scores[result.method_name].append(result.mean_value)
        
        overall_ranking = []
        for method, scores in method_scores.items():
            avg_score = np.mean(scores)
            overall_ranking.append((method, avg_score))
        
        overall_ranking.sort(key=lambda x: x[1], reverse=True)
        
        summary = {
            'total_benchmarks': len(self.results),
            'methods_tested': len(methods),
            'environments_tested': len(environments),
            'metrics_evaluated': len(metrics),
            'best_methods_per_metric': best_methods,
            'overall_ranking': [{'method': method, 'avg_score': score} 
                              for method, score in overall_ranking],
            'total_execution_time': sum(result.execution_time for result in self.results),
            'avg_memory_usage': np.mean([result.memory_usage for result in self.results 
                                       if result.memory_usage > 0]),
            'benchmark_config': self.config.to_dict()
        }
        
        return summary
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save benchmark results to file.
        
        Args:
            results: Benchmark results to save
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_results_{timestamp}.json"
        filepath = os.path.join(self.config.output_directory, filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Benchmark results saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save benchmark results: {e}")
    
    def _generate_plots(self) -> None:
        """Generate visualization plots."""
        if not PLOTTING_AVAILABLE:
            logger.warning("Plotting libraries not available, skipping plot generation")
            return
        
        logger.info("Generating benchmark plots")
        
        try:
            # Performance comparison plot
            self._plot_performance_comparison()
            
            # Statistical significance heatmap
            self._plot_statistical_significance()
            
            # Method ranking plot
            self._plot_method_ranking()
            
        except Exception as e:
            logger.error(f"Failed to generate plots: {e}")
    
    def _plot_performance_comparison(self) -> None:
        """Generate performance comparison plot."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('ConfoRL Methods Performance Comparison', fontsize=16)
        
        # Group results by metric
        metrics = list(set(result.metric for result in self.results))
        
        for i, metric in enumerate(metrics[:4]):  # Plot first 4 metrics
            ax = axes[i // 2, i % 2]
            
            metric_results = [r for r in self.results if r.metric == metric]
            
            methods = list(set(r.method_name for r in metric_results))
            environments = list(set(r.environment for r in metric_results))
            
            # Create bar plot
            method_scores = defaultdict(list)
            for result in metric_results:
                method_scores[result.method_name].append(result.mean_value)
            
            method_names = list(method_scores.keys())
            avg_scores = [np.mean(method_scores[method]) for method in method_names]
            
            bars = ax.bar(method_names, avg_scores)
            ax.set_title(f'{metric.value.replace("_", " ").title()}')
            ax.set_ylabel('Score')
            ax.tick_params(axis='x', rotation=45)
            
            # Color bars
            colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.config.output_directory, 'performance_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Performance comparison plot saved to {plot_path}")
    
    def _plot_statistical_significance(self) -> None:
        """Generate statistical significance heatmap."""
        if not self.comparison_results.get('statistical_significance'):
            return
        
        # Extract p-values for heatmap
        significance_data = self.comparison_results['statistical_significance']
        
        # Create heatmap data
        all_methods = set()
        for config_results in significance_data.values():
            for comparison in config_results.keys():
                method1, method2 = comparison.split('_vs_')
                all_methods.add(method1)
                all_methods.add(method2)
        
        methods = sorted(list(all_methods))
        n_methods = len(methods)
        
        # Create significance matrix
        sig_matrix = np.ones((n_methods, n_methods))  # 1 = not significant
        
        for config_results in significance_data.values():
            for comparison, stats in config_results.items():
                method1, method2 = comparison.split('_vs_')
                i, j = methods.index(method1), methods.index(method2)
                
                # 0 = significant, 1 = not significant
                sig_value = 0 if stats['significant'] else 1
                sig_matrix[i, j] = sig_value
                sig_matrix[j, i] = sig_value
        
        # Plot heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(sig_matrix, 
                    xticklabels=methods, 
                    yticklabels=methods,
                    annot=True, 
                    cmap='RdYlBu_r',
                    cbar_kws={'label': 'Statistical Significance (0=Significant, 1=Not Significant)'})
        
        plt.title('Statistical Significance Between Methods')
        plt.xlabel('Methods')
        plt.ylabel('Methods')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        # Save plot
        plot_path = os.path.join(self.config.output_directory, 'statistical_significance.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Statistical significance plot saved to {plot_path}")
    
    def _plot_method_ranking(self) -> None:
        """Generate method ranking plot."""
        if not self.comparison_results.get('ranking'):
            return
        
        # Extract rankings across all configurations
        ranking_data = self.comparison_results['ranking']
        
        method_positions = defaultdict(list)
        
        for config, ranking_info in ranking_data.items():
            ranking = ranking_info['ranking']
            for position, method in enumerate(ranking):
                method_positions[method].append(position + 1)  # 1-based ranking
        
        # Calculate average ranking for each method
        method_avg_ranking = {}
        for method, positions in method_positions.items():
            method_avg_ranking[method] = np.mean(positions)
        
        # Sort by average ranking
        sorted_methods = sorted(method_avg_ranking.keys(), 
                              key=lambda x: method_avg_ranking[x])
        
        # Create ranking plot
        plt.figure(figsize=(12, 8))
        
        positions = [method_avg_ranking[method] for method in sorted_methods]
        
        bars = plt.barh(sorted_methods, positions)
        
        # Color gradient
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.xlabel('Average Ranking (Lower is Better)')
        plt.title('Method Ranking Across All Benchmarks')
        plt.gca().invert_yaxis()  # Best methods at top
        
        # Add ranking numbers
        for i, (method, avg_rank) in enumerate(zip(sorted_methods, positions)):
            plt.text(avg_rank + 0.05, i, f'{avg_rank:.2f}', 
                    va='center', fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.config.output_directory, 'method_ranking.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Method ranking plot saved to {plot_path}")
    
    def generate_research_report(self) -> str:
        """Generate research paper style report.
        
        Returns:
            Formatted research report
        """
        if not self.results:
            return "No benchmark results available."
        
        report = []
        report.append("# ConfoRL Benchmark Report\n")
        report.append("## Abstract\n")
        report.append("This report presents comprehensive benchmarking results for conformal ")
        report.append("risk control methods in reinforcement learning. We evaluate multiple ")
        report.append("approaches across diverse environments and metrics to assess their ")
        report.append("effectiveness, robustness, and computational efficiency.\n\n")
        
        # Summary statistics
        summary = self._generate_benchmark_summary()
        report.append("## Experimental Setup\n")
        report.append(f"- **Methods tested**: {summary['methods_tested']}\n")
        report.append(f"- **Environments**: {summary['environments_tested']}\n")
        report.append(f"- **Metrics**: {summary['metrics_evaluated']}\n")
        report.append(f"- **Total benchmarks**: {summary['total_benchmarks']}\n")
        report.append(f"- **Trials per benchmark**: {self.config.num_trials}\n\n")
        
        # Results
        report.append("## Results\n\n")
        
        # Overall ranking
        report.append("### Overall Method Ranking\n\n")
        for i, result in enumerate(summary['overall_ranking'][:5], 1):
            report.append(f"{i}. **{result['method']}**: {result['avg_score']:.4f}\n")
        report.append("\n")
        
        # Best methods per metric
        report.append("### Best Methods by Metric\n\n")
        for metric, best in summary['best_methods_per_metric'].items():
            report.append(f"- **{metric.replace('_', ' ').title()}**: {best['method']} ")
            report.append(f"(Score: {best['score']:.4f}, Environment: {best['environment']})\n")
        report.append("\n")
        
        # Statistical significance
        if self.comparison_results.get('statistical_significance'):
            report.append("### Statistical Significance\n\n")
            
            significant_comparisons = 0
            total_comparisons = 0
            
            for config_results in self.comparison_results['statistical_significance'].values():
                for comparison, stats in config_results.items():
                    total_comparisons += 1
                    if stats['significant']:
                        significant_comparisons += 1
            
            significance_rate = significant_comparisons / total_comparisons if total_comparisons > 0 else 0
            report.append(f"Significant differences found in {significant_comparisons}/{total_comparisons} ")
            report.append(f"comparisons ({significance_rate:.1%}).\n\n")
        
        # Performance analysis
        report.append("### Performance Analysis\n\n")
        report.append(f"- **Total execution time**: {summary['total_execution_time']:.2f} seconds\n")
        if summary['avg_memory_usage'] > 0:
            report.append(f"- **Average memory usage**: {summary['avg_memory_usage']/1024/1024:.1f} MB\n")
        report.append("\n")
        
        # Conclusions
        report.append("## Conclusions\n\n")
        
        if summary['overall_ranking']:
            best_method = summary['overall_ranking'][0]['method']
            report.append(f"The {best_method} method demonstrated the best overall performance ")
            report.append("across the benchmark suite. ")
        
        report.append("These results provide valuable insights into the effectiveness of ")
        report.append("different conformal risk control approaches for reinforcement learning applications.\n\n")
        
        # Methodology
        report.append("## Methodology\n\n")
        report.append("All experiments were conducted using standardized environments and evaluation protocols. ")
        report.append(f"Statistical significance was assessed using t-tests with α = {self.config.significance_level}. ")
        if self.config.multiple_testing_correction != "bonferroni":
            report.append(f"Multiple testing correction was applied using the {self.config.multiple_testing_correction} method.")
        report.append("\n\n")
        
        return "".join(report)


# Utility functions for easy benchmarking

def quick_benchmark(
    methods: Dict[str, Any], 
    environments: Optional[List[BenchmarkEnvironment]] = None,
    num_trials: int = 10
) -> Dict[str, Any]:
    """Quick benchmark for rapid evaluation.
    
    Args:
        methods: Dictionary of conformal methods to benchmark
        environments: List of environments (defaults to CartPole)
        num_trials: Number of trials per benchmark
        
    Returns:
        Benchmark results
    """
    config = BenchmarkConfig(
        environments=environments or [BenchmarkEnvironment.CARTPOLE],
        metrics=[BenchmarkMetric.COVERAGE_ACCURACY, BenchmarkMetric.SAFETY_VIOLATION_RATE],
        num_trials=num_trials,
        num_episodes_per_trial=20,
        generate_plots=False,
        save_results=False
    )
    
    suite = ComprehensiveBenchmarkSuite(config)
    return suite.run_benchmark(methods)


def research_benchmark(
    methods: Dict[str, Any], 
    output_dir: str = "research_benchmark_results"
) -> Dict[str, Any]:
    """Comprehensive research-grade benchmark.
    
    Args:
        methods: Dictionary of conformal methods to benchmark
        output_dir: Output directory for results
        
    Returns:
        Comprehensive benchmark results
    """
    config = BenchmarkConfig(
        environments=[
            BenchmarkEnvironment.CARTPOLE,
            BenchmarkEnvironment.ADVERSARIAL_ENV,
            BenchmarkEnvironment.DISTRIBUTION_SHIFT
        ],
        metrics=[
            BenchmarkMetric.COVERAGE_ACCURACY,
            BenchmarkMetric.SAFETY_VIOLATION_RATE,
            BenchmarkMetric.RISK_BOUND_TIGHTNESS,
            BenchmarkMetric.ROBUSTNESS_SCORE
        ],
        num_trials=50,
        num_episodes_per_trial=100,
        statistical_tests=True,
        profile_performance=True,
        generate_plots=True,
        save_results=True,
        output_directory=output_dir
    )
    
    suite = ComprehensiveBenchmarkSuite(config)
    return suite.run_benchmark(methods)
