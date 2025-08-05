"""Self-improving and adaptive optimization components."""

import time
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Tuple
from collections import deque, defaultdict
from dataclasses import dataclass
import threading
from abc import ABC, abstractmethod

from ..utils.logging import get_logger
from .metrics import MetricsCollector, PerformanceTracker

logger = get_logger(__name__)


@dataclass
class HyperparameterSuggestion:
    """Suggestion for hyperparameter optimization."""
    parameters: Dict[str, Any]
    expected_improvement: float
    confidence: float
    metadata: Dict[str, Any]


class AdaptiveTuner:
    """Adaptive tuning system that learns and adjusts parameters automatically."""
    
    def __init__(
        self,
        adaptation_interval: float = 300.0,  # 5 minutes
        learning_rate: float = 0.01,
        memory_size: int = 1000
    ):
        """Initialize adaptive tuner.
        
        Args:
            adaptation_interval: How often to adapt parameters (seconds)
            learning_rate: Rate of parameter adaptation
            memory_size: Size of performance memory
        """
        self.adaptation_interval = adaptation_interval
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        
        # Parameter tracking
        self.parameters = {}
        self.parameter_bounds = {}
        self.parameter_history = defaultdict(lambda: deque(maxlen=memory_size))
        self.performance_history = deque(maxlen=memory_size)
        
        # Adaptation state
        self.last_adaptation = 0
        self.adaptation_count = 0
        self.improvement_count = 0
        
        # Threading
        self._lock = threading.RLock()
        
        logger.info("AdaptiveTuner initialized")
    
    def register_parameter(
        self,
        name: str,
        initial_value: float,
        bounds: Tuple[float, float],
        adaptation_weight: float = 1.0
    ):
        """Register a parameter for adaptive tuning.
        
        Args:
            name: Parameter name
            initial_value: Initial parameter value
            bounds: (min_value, max_value) bounds
            adaptation_weight: Weight for adaptation (higher = more aggressive)
        """
        with self._lock:
            self.parameters[name] = {
                'value': initial_value,
                'bounds': bounds,
                'weight': adaptation_weight,
                'gradient': 0.0,
                'momentum': 0.0
            }
            self.parameter_bounds[name] = bounds
            
            logger.info(f"Registered parameter: {name} = {initial_value} (bounds: {bounds})")
    
    def get_parameter(self, name: str) -> Optional[float]:
        """Get current parameter value.
        
        Args:
            name: Parameter name
            
        Returns:
            Current parameter value or None if not found
        """
        with self._lock:
            if name in self.parameters:
                return self.parameters[name]['value']
            return None
    
    def update_performance(self, performance_score: float, context: Optional[Dict[str, Any]] = None):
        """Update performance feedback for adaptation.
        
        Args:
            performance_score: Performance metric (higher is better)
            context: Additional context information
        """
        with self._lock:
            timestamp = time.time()
            
            # Store performance with current parameter values
            param_snapshot = {name: info['value'] for name, info in self.parameters.items()}
            
            performance_record = {
                'score': performance_score,
                'parameters': param_snapshot,
                'timestamp': timestamp,
                'context': context or {}
            }
            
            self.performance_history.append(performance_record)
            
            # Update parameter histories
            for name, value in param_snapshot.items():
                self.parameter_history[name].append((value, performance_score, timestamp))
            
            # Check if adaptation is due
            if timestamp - self.last_adaptation >= self.adaptation_interval:
                self._adapt_parameters()
    
    def _adapt_parameters(self):
        """Adapt parameters based on performance history."""
        if len(self.performance_history) < 10:
            return  # Need minimum history
        
        logger.debug("Starting parameter adaptation")
        
        # Get recent performance trend
        recent_performance = list(self.performance_history)[-10:]
        performance_trend = self._calculate_performance_trend(recent_performance)
        
        for param_name, param_info in self.parameters.items():
            if param_name not in self.parameter_history:
                continue
            
            # Calculate parameter gradient from recent history
            gradient = self._estimate_parameter_gradient(param_name, recent_performance)
            
            if gradient is not None:
                # Update momentum (exponential moving average of gradients)
                param_info['momentum'] = 0.9 * param_info['momentum'] + 0.1 * gradient
                
                # Adapt parameter value
                adaptation_step = self.learning_rate * param_info['weight'] * param_info['momentum']
                
                # Apply bounds
                new_value = param_info['value'] + adaptation_step
                min_val, max_val = param_info['bounds']
                new_value = np.clip(new_value, min_val, max_val)
                
                # Update parameter
                old_value = param_info['value']
                param_info['value'] = new_value
                
                if abs(new_value - old_value) > 1e-6:
                    logger.info(f"Adapted {param_name}: {old_value:.6f} -> {new_value:.6f} (gradient: {gradient:.6f})")
        
        self.last_adaptation = time.time()
        self.adaptation_count += 1
        
        # Check if overall performance improved
        if performance_trend > 0:
            self.improvement_count += 1
    
    def _calculate_performance_trend(self, recent_performance: List[Dict[str, Any]]) -> float:
        """Calculate performance trend from recent history.
        
        Args:
            recent_performance: Recent performance records
            
        Returns:
            Performance trend (positive = improving)
        """
        if len(recent_performance) < 5:
            return 0.0
        
        scores = [record['score'] for record in recent_performance]
        
        # Simple linear trend calculation
        x = np.arange(len(scores))
        y = np.array(scores)
        
        if len(scores) > 1:
            slope, _ = np.polyfit(x, y, 1)
            return slope
        
        return 0.0
    
    def _estimate_parameter_gradient(
        self,
        param_name: str,
        recent_performance: List[Dict[str, Any]]
    ) -> Optional[float]:
        """Estimate gradient of performance with respect to parameter.
        
        Args:
            param_name: Name of parameter
            recent_performance: Recent performance records
            
        Returns:
            Estimated gradient or None
        """
        # Extract parameter values and corresponding performance scores
        param_values = []
        scores = []
        
        for record in recent_performance:
            if param_name in record['parameters']:
                param_values.append(record['parameters'][param_name])
                scores.append(record['score'])
        
        if len(param_values) < 3:
            return None
        
        param_values = np.array(param_values)
        scores = np.array(scores)
        
        # Simple finite difference approximation
        if len(param_values) > 1:
            # Use correlation as gradient estimate
            correlation = np.corrcoef(param_values, scores)[0, 1]
            if not np.isnan(correlation):
                return correlation
        
        return None
    
    def get_adaptation_stats(self) -> Dict[str, Any]:
        """Get adaptation statistics.
        
        Returns:
            Adaptation statistics
        """
        with self._lock:
            return {
                'adaptation_count': self.adaptation_count,
                'improvement_count': self.improvement_count,
                'improvement_rate': (
                    self.improvement_count / max(1, self.adaptation_count)
                ),
                'parameters': {
                    name: {
                        'current_value': info['value'],
                        'bounds': info['bounds'],
                        'momentum': info['momentum']
                    }
                    for name, info in self.parameters.items()
                },
                'performance_history_size': len(self.performance_history)
            }


class HyperparameterOptimizer:
    """Bayesian hyperparameter optimization for ConfoRL agents."""
    
    def __init__(
        self,
        optimization_budget: int = 100,
        exploration_weight: float = 0.1,
        random_seed: Optional[int] = None
    ):
        """Initialize hyperparameter optimizer.
        
        Args:
            optimization_budget: Maximum number of evaluations
            exploration_weight: Weight for exploration vs exploitation
            random_seed: Random seed for reproducibility
        """
        self.optimization_budget = optimization_budget
        self.exploration_weight = exploration_weight
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Parameter space definition
        self.parameter_space = {}
        self.evaluation_history = []
        self.best_parameters = None
        self.best_score = -np.inf
        
        # Gaussian Process surrogate model (simplified)
        self.surrogate_model = None
        
        logger.info("HyperparameterOptimizer initialized")
    
    def add_parameter(
        self,
        name: str,
        parameter_type: str,
        bounds: Tuple[float, float],
        log_scale: bool = False
    ):
        """Add parameter to optimization space.
        
        Args:
            name: Parameter name
            parameter_type: Type ('continuous', 'integer', 'categorical')
            bounds: Parameter bounds (min, max)
            log_scale: Whether to use log scale for optimization
        """
        self.parameter_space[name] = {
            'type': parameter_type,
            'bounds': bounds,
            'log_scale': log_scale
        }
        
        logger.info(f"Added parameter to optimization space: {name} ({parameter_type})")
    
    def suggest_parameters(self) -> HyperparameterSuggestion:
        """Suggest next set of parameters to evaluate.
        
        Returns:
            Parameter suggestion with expected improvement
        """
        if len(self.evaluation_history) < 3:
            # Random exploration for initial points
            parameters = self._sample_random_parameters()
            expected_improvement = 1.0  # High for exploration
            confidence = 0.5
        else:
            # Use surrogate model for informed suggestions
            parameters, expected_improvement, confidence = self._suggest_with_surrogate()
        
        return HyperparameterSuggestion(
            parameters=parameters,
            expected_improvement=expected_improvement,
            confidence=confidence,
            metadata={'method': 'bayesian_optimization'}
        )
    
    def _sample_random_parameters(self) -> Dict[str, Any]:
        """Sample random parameters from the parameter space.
        
        Returns:
            Random parameter configuration
        """
        parameters = {}
        
        for name, space_info in self.parameter_space.items():
            bounds = space_info['bounds']
            param_type = space_info['type']
            log_scale = space_info['log_scale']
            
            if param_type == 'continuous':
                if log_scale:
                    log_low, log_high = np.log(bounds[0]), np.log(bounds[1])
                    value = np.exp(np.random.uniform(log_low, log_high))
                else:
                    value = np.random.uniform(bounds[0], bounds[1])
                parameters[name] = float(value)
                
            elif param_type == 'integer':
                value = np.random.randint(int(bounds[0]), int(bounds[1]) + 1)
                parameters[name] = int(value)
        
        return parameters
    
    def _suggest_with_surrogate(self) -> Tuple[Dict[str, Any], float, float]:
        """Suggest parameters using surrogate model.
        
        Returns:
            (parameters, expected_improvement, confidence)
        """
        # Simplified surrogate model (in practice, would use proper GP)
        # Generate multiple candidates and pick best according to acquisition function
        
        best_ei = -np.inf
        best_params = None
        best_confidence = 0.0
        
        # Sample multiple candidates
        for _ in range(20):
            candidate_params = self._sample_random_parameters()
            
            # Estimate expected improvement (simplified)
            ei, confidence = self._estimate_expected_improvement(candidate_params)
            
            if ei > best_ei:
                best_ei = ei
                best_params = candidate_params
                best_confidence = confidence
        
        return best_params, best_ei, best_confidence
    
    def _estimate_expected_improvement(
        self,
        parameters: Dict[str, Any]
    ) -> Tuple[float, float]:
        """Estimate expected improvement for parameter configuration.
        
        Args:
            parameters: Parameter configuration
            
        Returns:
            (expected_improvement, confidence)
        """
        # Simplified EI calculation based on distance to previous evaluations
        if not self.evaluation_history:
            return 1.0, 0.5
        
        # Find similar parameter configurations
        similarities = []
        for eval_record in self.evaluation_history:
            similarity = self._parameter_similarity(parameters, eval_record['parameters'])
            similarities.append((similarity, eval_record['score']))
        
        # Weighted average of scores by similarity
        total_weight = sum(sim for sim, _ in similarities)
        if total_weight > 0:
            predicted_score = sum(
                sim * score for sim, score in similarities
            ) / total_weight
        else:
            predicted_score = 0.0
        
        # Expected improvement
        improvement = max(0, predicted_score - self.best_score)
        confidence = min(1.0, total_weight / len(similarities)) if similarities else 0.5
        
        # Add exploration bonus
        exploration_bonus = self.exploration_weight * (1.0 - confidence)
        
        return improvement + exploration_bonus, confidence
    
    def _parameter_similarity(
        self,
        params1: Dict[str, Any],
        params2: Dict[str, Any]
    ) -> float:
        """Calculate similarity between parameter configurations.
        
        Args:
            params1: First parameter configuration
            params2: Second parameter configuration
            
        Returns:
            Similarity score (0-1)
        """
        if not params1 or not params2:
            return 0.0
        
        common_params = set(params1.keys()) & set(params2.keys())
        if not common_params:
            return 0.0
        
        similarities = []
        for param_name in common_params:
            val1, val2 = params1[param_name], params2[param_name]
            
            # Normalize by parameter range
            if param_name in self.parameter_space:
                bounds = self.parameter_space[param_name]['bounds']
                param_range = bounds[1] - bounds[0]
                if param_range > 0:
                    normalized_diff = abs(val1 - val2) / param_range
                    similarity = max(0, 1 - normalized_diff)
                    similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    def report_result(self, parameters: Dict[str, Any], score: float):
        """Report evaluation result for parameter configuration.
        
        Args:
            parameters: Parameter configuration that was evaluated
            score: Performance score achieved
        """
        evaluation_record = {
            'parameters': parameters.copy(),
            'score': score,
            'timestamp': time.time(),
            'evaluation_id': len(self.evaluation_history)
        }
        
        self.evaluation_history.append(evaluation_record)
        
        # Update best configuration
        if score > self.best_score:
            self.best_score = score
            self.best_parameters = parameters.copy()
            logger.info(f"New best hyperparameters found: score={score:.4f}")
    
    def get_best_parameters(self) -> Optional[Dict[str, Any]]:
        """Get best parameter configuration found so far.
        
        Returns:
            Best parameter configuration or None
        """
        return self.best_parameters.copy() if self.best_parameters else None
    
    def get_optimization_progress(self) -> Dict[str, Any]:
        """Get optimization progress statistics.
        
        Returns:
            Optimization progress information
        """
        scores = [record['score'] for record in self.evaluation_history]
        
        return {
            'evaluations_completed': len(self.evaluation_history),
            'evaluations_remaining': max(0, self.optimization_budget - len(self.evaluation_history)),
            'best_score': self.best_score,
            'current_score_mean': np.mean(scores) if scores else 0.0,
            'current_score_std': np.std(scores) if len(scores) > 1 else 0.0,
            'improvement_over_time': scores.copy()
        }


class SelfImprovingAgent:
    """Self-improving agent that continuously optimizes its own performance."""
    
    def __init__(
        self,
        base_agent,
        performance_tracker: Optional[PerformanceTracker] = None,
        improvement_threshold: float = 0.05,
        evaluation_window: int = 100
    ):
        """Initialize self-improving agent wrapper.
        
        Args:
            base_agent: Base RL agent to wrap
            performance_tracker: Performance tracker instance
            improvement_threshold: Minimum improvement to trigger adaptation
            evaluation_window: Window size for performance evaluation
        """
        self.base_agent = base_agent
        self.performance_tracker = performance_tracker or PerformanceTracker()
        self.improvement_threshold = improvement_threshold
        self.evaluation_window = evaluation_window
        
        # Self-improvement components
        self.adaptive_tuner = AdaptiveTuner()
        self.hyperopt = HyperparameterOptimizer()
        
        # Performance tracking
        self.baseline_performance = None
        self.recent_performance = deque(maxlen=evaluation_window)
        self.improvement_count = 0
        
        # Register key parameters for adaptation
        self._register_adaptable_parameters()
        
        logger.info("SelfImprovingAgent initialized")
    
    def _register_adaptable_parameters(self):
        """Register key parameters for adaptive tuning."""
        # Learning rate adaptation
        if hasattr(self.base_agent, 'learning_rate'):
            self.adaptive_tuner.register_parameter(
                'learning_rate',
                self.base_agent.learning_rate,
                (1e-5, 1e-2),
                adaptation_weight=0.5
            )
        
        # Risk tolerance adaptation
        if hasattr(self.base_agent, 'risk_controller'):
            if hasattr(self.base_agent.risk_controller, 'target_risk'):
                self.adaptive_tuner.register_parameter(
                    'target_risk',
                    self.base_agent.risk_controller.target_risk,
                    (0.001, 0.1),
                    adaptation_weight=0.3
                )
        
        # Exploration parameters
        if hasattr(self.base_agent, 'policy_std'):
            current_std = np.mean(self.base_agent.policy_std) if hasattr(self.base_agent.policy_std, '__len__') else self.base_agent.policy_std
            self.adaptive_tuner.register_parameter(
                'exploration_std',
                current_std,
                (0.01, 2.0),
                adaptation_weight=0.4
            )
    
    def train(self, *args, **kwargs):
        """Train with self-improvement."""
        # Track training performance
        initial_episodes = getattr(self.base_agent, 'episode_count', 0)
        
        # Train base agent
        result = self.base_agent.train(*args, **kwargs)
        
        # Evaluate improvement
        final_episodes = getattr(self.base_agent, 'episode_count', 0)
        episodes_trained = final_episodes - initial_episodes
        
        if episodes_trained > 0:
            self._evaluate_and_improve()
        
        return result
    
    def predict(self, *args, **kwargs):
        """Predict with performance tracking."""
        start_time = time.time()
        
        # Apply adaptive parameters
        self._apply_adaptive_parameters()
        
        # Get prediction from base agent
        result = self.base_agent.predict(*args, **kwargs)
        
        # Track inference performance
        inference_time = time.time() - start_time
        
        if hasattr(result, '__len__') and len(result) == 2:
            # Result includes risk certificate
            action, certificate = result
            self.performance_tracker.track_inference(
                inference_time=inference_time,
                risk_bound=certificate.risk_bound,
                confidence=certificate.confidence,
                algorithm=self._get_algorithm_name()
            )
        else:
            # Just action
            self.performance_tracker.track_inference(
                inference_time=inference_time,
                risk_bound=0.0,
                confidence=1.0,
                algorithm=self._get_algorithm_name()
            )
        
        return result
    
    def _apply_adaptive_parameters(self):
        """Apply adaptively tuned parameters to base agent."""
        # Update learning rate
        new_lr = self.adaptive_tuner.get_parameter('learning_rate')
        if new_lr is not None and hasattr(self.base_agent, 'learning_rate'):
            self.base_agent.learning_rate = new_lr
        
        # Update risk tolerance
        new_risk = self.adaptive_tuner.get_parameter('target_risk')
        if (new_risk is not None and 
            hasattr(self.base_agent, 'risk_controller') and
            hasattr(self.base_agent.risk_controller, 'target_risk')):
            self.base_agent.risk_controller.target_risk = new_risk
        
        # Update exploration
        new_std = self.adaptive_tuner.get_parameter('exploration_std')
        if new_std is not None and hasattr(self.base_agent, 'policy_std'):
            if hasattr(self.base_agent.policy_std, 'fill'):
                # Array-like
                self.base_agent.policy_std.fill(new_std)
            else:
                # Scalar
                self.base_agent.policy_std = new_std
    
    def _evaluate_and_improve(self):
        """Evaluate recent performance and trigger improvements."""
        # Get recent performance metrics
        perf_summary = self.performance_tracker.get_performance_summary()
        
        # Calculate performance score (composite metric)
        performance_score = self._calculate_performance_score(perf_summary)
        
        # Update adaptive tuner
        self.adaptive_tuner.update_performance(performance_score)
        
        # Track performance history
        self.recent_performance.append(performance_score)
        
        # Check for improvement opportunity
        if len(self.recent_performance) >= 20:
            if self._should_trigger_improvement():
                self._trigger_hyperparameter_optimization()
    
    def _calculate_performance_score(self, perf_summary: Dict[str, Any]) -> float:
        """Calculate composite performance score.
        
        Args:
            perf_summary: Performance summary from tracker
            
        Returns:
            Composite performance score
        """
        score = 0.0
        
        # Reward component
        if 'training.episode.reward' in perf_summary.get('metrics', {}):
            reward_stats = perf_summary['metrics']['training.episode.reward']
            score += reward_stats.get('mean', 0) * 0.4
        
        # Risk component (lower is better)
        if 'training.episode.risk' in perf_summary.get('metrics', {}):
            risk_stats = perf_summary['metrics']['training.episode.risk']
            risk_penalty = risk_stats.get('mean', 0) * 10  # Scale risk
            score -= risk_penalty * 0.3
        
        # Efficiency component (faster inference is better)
        if 'inference.duration' in perf_summary.get('metrics', {}):
            inference_stats = perf_summary['metrics']['inference.duration']
            avg_inference_time = inference_stats.get('mean', 1.0)
            efficiency_bonus = max(0, 1.0 - avg_inference_time) * 0.3
            score += efficiency_bonus
        
        return score
    
    def _should_trigger_improvement(self) -> bool:
        """Check if improvement should be triggered.
        
        Returns:
            True if improvement should be triggered
        """
        if len(self.recent_performance) < 10:
            return False
        
        # Check if performance has plateaued
        recent_scores = list(self.recent_performance)[-10:]
        older_scores = list(self.recent_performance)[-20:-10]
        
        recent_mean = np.mean(recent_scores)
        older_mean = np.mean(older_scores)
        
        improvement = recent_mean - older_mean
        
        return improvement < self.improvement_threshold
    
    def _trigger_hyperparameter_optimization(self):
        """Trigger hyperparameter optimization."""
        logger.info("Triggering hyperparameter optimization due to performance plateau")
        
        # Add current parameters as baseline
        current_params = {
            'learning_rate': getattr(self.base_agent, 'learning_rate', 0.001),
            'target_risk': getattr(
                getattr(self.base_agent, 'risk_controller', None),
                'target_risk',
                0.05
            )
        }
        
        current_score = np.mean(list(self.recent_performance)[-5:])
        self.hyperopt.report_result(current_params, current_score)
        
        # Get suggestion for next configuration
        suggestion = self.hyperopt.suggest_parameters()
        
        logger.info(f"Hyperparameter suggestion: {suggestion.parameters}")
        
        # Apply suggested parameters
        for param_name, param_value in suggestion.parameters.items():
            if param_name == 'learning_rate' and hasattr(self.base_agent, 'learning_rate'):
                self.base_agent.learning_rate = param_value
            elif param_name == 'target_risk':
                if (hasattr(self.base_agent, 'risk_controller') and
                    hasattr(self.base_agent.risk_controller, 'target_risk')):
                    self.base_agent.risk_controller.target_risk = param_value
    
    def _get_algorithm_name(self) -> str:
        """Get algorithm name from base agent.
        
        Returns:
            Algorithm name
        """
        return getattr(self.base_agent, '__class__', type(self.base_agent)).__name__
    
    def get_improvement_stats(self) -> Dict[str, Any]:
        """Get self-improvement statistics.
        
        Returns:
            Improvement statistics
        """
        return {
            'improvement_count': self.improvement_count,
            'adaptive_tuner': self.adaptive_tuner.get_adaptation_stats(),
            'hyperopt_progress': self.hyperopt.get_optimization_progress(),
            'recent_performance': list(self.recent_performance),
            'performance_trend': (
                np.mean(list(self.recent_performance)[-5:]) - 
                np.mean(list(self.recent_performance)[:5])
                if len(self.recent_performance) >= 10 else 0.0
            )
        }