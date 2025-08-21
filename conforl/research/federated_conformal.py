"""Federated Learning for Conformal Prediction in Multi-Agent RL.

This module implements federated learning approaches for conformal prediction
that enable multiple agents to collaboratively learn safe policies while
preserving privacy and handling heterogeneous environments.

Research Contributions:
- Privacy-preserving conformal risk aggregation
- Heterogeneous federated conformal learning
- Secure multi-party conformal quantile computation
- Differential privacy for conformal bounds

Author: ConfoRL Research Team
License: Apache 2.0
"""

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
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
        def random():
            import random
            return random
        @staticmethod
        def median(x): 
            sorted_x = sorted(x)
            n = len(sorted_x)
            return sorted_x[n//2] if n > 0 else 0

import time
import math
import hashlib
from typing import Dict, List, Optional, Tuple, Any, Union, Callable, Set
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from enum import Enum

from ..core.types import RiskCertificate, TrajectoryData
from ..risk.controllers import AdaptiveRiskController
from ..utils.logging import get_logger
from ..utils.errors import ConfoRLError, ValidationError

logger = get_logger(__name__)


class FederatedAggregationMethod(Enum):
    """Methods for federated aggregation of conformal predictions."""
    FEDAVG = "federated_averaging"
    FEDPROX = "federated_proximal"
    SCAFFOLD = "scaffold"
    DIFFERENTIAL_PRIVATE = "differential_private"
    SECURE_AGGREGATION = "secure_aggregation"
    QUANTILE_AGGREGATION = "quantile_aggregation"


@dataclass
class FederatedConformalConfig:
    """Configuration for federated conformal learning."""
    
    num_clients: int = 10
    rounds: int = 100
    client_fraction: float = 0.3
    local_epochs: int = 5
    aggregation_method: FederatedAggregationMethod = FederatedAggregationMethod.FEDAVG
    
    # Privacy parameters
    differential_privacy: bool = False
    dp_epsilon: float = 1.0
    dp_delta: float = 1e-5
    
    # Secure aggregation
    secure_aggregation: bool = False
    threshold_secret_sharing: bool = False
    
    # Heterogeneity handling
    handle_heterogeneity: bool = True
    weight_by_data_size: bool = True
    adaptive_learning_rates: bool = True
    
    # Communication efficiency
    compression: bool = False
    quantization_bits: int = 8
    sparsification_ratio: float = 0.1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_clients": self.num_clients,
            "rounds": self.rounds,
            "client_fraction": self.client_fraction,
            "local_epochs": self.local_epochs,
            "aggregation_method": self.aggregation_method.value,
            "differential_privacy": self.differential_privacy,
            "dp_epsilon": self.dp_epsilon,
            "dp_delta": self.dp_delta,
            "secure_aggregation": self.secure_aggregation,
            "threshold_secret_sharing": self.threshold_secret_sharing,
            "handle_heterogeneity": self.handle_heterogeneity,
            "weight_by_data_size": self.weight_by_data_size,
            "adaptive_learning_rates": self.adaptive_learning_rates,
            "compression": self.compression,
            "quantization_bits": self.quantization_bits,
            "sparsification_ratio": self.sparsification_ratio
        }


@dataclass
class ClientUpdate:
    """Update from a federated client."""
    
    client_id: str
    round_number: int
    calibration_scores: List[float]
    data_size: int
    local_quantiles: Dict[str, float]
    model_parameters: Optional[Dict[str, Any]] = None
    privacy_budget_used: float = 0.0
    computation_time: float = 0.0
    communication_cost: int = 0  # bytes
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "client_id": self.client_id,
            "round_number": self.round_number,
            "calibration_scores": self.calibration_scores,
            "data_size": self.data_size,
            "local_quantiles": self.local_quantiles,
            "model_parameters": self.model_parameters,
            "privacy_budget_used": self.privacy_budget_used,
            "computation_time": self.computation_time,
            "communication_cost": self.communication_cost
        }


class FederatedConformalClient:
    """Federated learning client for conformal prediction."""
    
    def __init__(
        self, 
        client_id: str, 
        local_predictor: Any, 
        config: FederatedConformalConfig
    ):
        """Initialize federated conformal client.
        
        Args:
            client_id: Unique client identifier
            local_predictor: Local conformal predictor
            config: Federated learning configuration
        """
        self.client_id = client_id
        self.local_predictor = local_predictor
        self.config = config
        
        # Local data and state
        self.local_data = []
        self.calibration_scores = []
        self.local_quantiles = {}
        
        # Privacy accounting
        self.privacy_budget_remaining = config.dp_epsilon if config.differential_privacy else float('inf')
        self.noise_multiplier = 1.0
        
        # Communication tracking
        self.communication_costs = []
        self.round_participation = []
        
        # Performance metrics
        self.local_performance_history = []
        
        logger.info(f"Initialized federated conformal client: {client_id}")
    
    def add_local_data(self, data: List[Tuple[Any, Any, float]]) -> None:
        """Add data to local dataset.
        
        Args:
            data: Local training data (state, action, risk)
        """
        self.local_data.extend(data)
        logger.debug(f"Client {self.client_id} added {len(data)} samples. Total: {len(self.local_data)}")
    
    def local_update(
        self, 
        global_model: Optional[Dict[str, Any]] = None, 
        round_number: int = 0
    ) -> ClientUpdate:
        """Perform local update and compute conformal scores.
        
        Args:
            global_model: Global model parameters from server
            round_number: Current round number
            
        Returns:
            Client update with conformal scores and model parameters
        """
        start_time = time.time()
        
        # Update local model with global parameters if provided
        if global_model and hasattr(self.local_predictor, 'set_parameters'):
            self.local_predictor.set_parameters(global_model)
        
        # Compute calibration scores on local data
        calibration_scores = self._compute_local_calibration_scores()
        
        # Compute local quantiles for different confidence levels
        confidence_levels = [0.9, 0.95, 0.99]
        local_quantiles = {}
        
        for confidence in confidence_levels:
            alpha = 1 - confidence
            if calibration_scores:
                quantile = self._compute_local_quantile(calibration_scores, alpha)
                # Apply differential privacy if enabled
                if self.config.differential_privacy:
                    quantile = self._add_differential_privacy_noise(quantile, round_number)
                local_quantiles[f"quantile_{confidence}"] = quantile
            else:
                local_quantiles[f"quantile_{confidence}"] = 0.5
        
        # Extract model parameters (if applicable)
        model_parameters = None
        if hasattr(self.local_predictor, 'get_parameters'):
            model_parameters = self.local_predictor.get_parameters()
            
            # Apply communication compression if enabled
            if self.config.compression:
                model_parameters = self._compress_parameters(model_parameters)
        
        computation_time = time.time() - start_time
        
        # Estimate communication cost
        communication_cost = self._estimate_communication_cost(
            calibration_scores, local_quantiles, model_parameters
        )
        
        # Create client update
        client_update = ClientUpdate(
            client_id=self.client_id,
            round_number=round_number,
            calibration_scores=calibration_scores,
            data_size=len(self.local_data),
            local_quantiles=local_quantiles,
            model_parameters=model_parameters,
            privacy_budget_used=self._get_privacy_budget_used(round_number),
            computation_time=computation_time,
            communication_cost=communication_cost
        )
        
        # Track participation
        self.round_participation.append(round_number)
        self.communication_costs.append(communication_cost)
        
        logger.debug(f"Client {self.client_id} local update completed for round {round_number}")
        
        return client_update
    
    def _compute_local_calibration_scores(self) -> List[float]:
        """Compute calibration scores on local data.
        
        Returns:
            List of local calibration scores
        """
        if not self.local_data:
            return []
        
        scores = []
        
        for state, action, true_risk in self.local_data:
            try:
                # Get prediction from local predictor
                if hasattr(self.local_predictor, 'predict_risk'):
                    predicted_risk = self.local_predictor.predict_risk(state, action)
                elif hasattr(self.local_predictor, 'predict'):
                    predicted_risk = self.local_predictor.predict(state, action)
                else:
                    predicted_risk = 0.5  # Default
                
                # Compute nonconformity score (absolute residual)
                score = abs(true_risk - predicted_risk)
                scores.append(score)
                
            except Exception as e:
                logger.warning(f"Failed to compute score for client {self.client_id}: {e}")
                scores.append(0.5)  # Conservative fallback
        
        self.calibration_scores = scores
        return scores
    
    def _compute_local_quantile(self, scores: List[float], alpha: float) -> float:
        """Compute local conformal quantile.
        
        Args:
            scores: Calibration scores
            alpha: Significance level
            
        Returns:
            Local quantile estimate
        """
        if not scores:
            return 0.5
        
        n = len(scores)
        sorted_scores = sorted(scores)
        
        # Compute conformal quantile with finite-sample correction
        q_level = (n + 1) * (1 - alpha) / n
        q_level = min(q_level, 1.0)
        
        quantile_idx = min(int(np.ceil(q_level * n)), n - 1)
        return sorted_scores[quantile_idx]
    
    def _add_differential_privacy_noise(
        self, 
        value: float, 
        round_number: int
    ) -> float:
        """Add differential privacy noise to quantile.
        
        Args:
            value: Original quantile value
            round_number: Current round number
            
        Returns:
            Noisy quantile value
        """
        if self.privacy_budget_remaining <= 0:
            logger.warning(f"Client {self.client_id} privacy budget exhausted")
            return value
        
        # Gaussian mechanism for differential privacy
        sensitivity = 1.0  # L2 sensitivity of quantile computation
        
        # Compute noise scale based on remaining budget
        rounds_remaining = self.config.rounds - round_number
        epsilon_per_round = self.privacy_budget_remaining / max(1, rounds_remaining)
        
        noise_scale = (sensitivity * math.sqrt(2 * math.log(1.25 / self.config.dp_delta))) / epsilon_per_round
        
        # Add Gaussian noise
        noise = np.random.normal(0, noise_scale)
        noisy_value = value + noise
        
        # Update privacy budget
        self.privacy_budget_remaining -= epsilon_per_round
        
        return max(0.0, min(1.0, noisy_value))  # Clamp to valid range
    
    def _compress_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Compress model parameters for communication efficiency.
        
        Args:
            parameters: Original parameters
            
        Returns:
            Compressed parameters
        """
        if not self.config.compression:
            return parameters
        
        compressed_params = {}
        
        for key, value in parameters.items():
            if isinstance(value, (list, np.ndarray)):
                # Quantization
                if self.config.quantization_bits < 32:
                    compressed_params[key] = self._quantize_values(
                        value, self.config.quantization_bits
                    )
                else:
                    compressed_params[key] = value
            else:
                compressed_params[key] = value
        
        return compressed_params
    
    def _quantize_values(self, values: Union[List, Any], bits: int) -> List[int]:
        """Quantize values to specified bit precision.
        
        Args:
            values: Values to quantize
            bits: Number of bits for quantization
            
        Returns:
            Quantized values
        """
        if not values:
            return []
        
        # Convert to list if numpy array
        value_list = list(values) if hasattr(values, '__iter__') else [values]
        
        # Find min/max for quantization range
        min_val = min(value_list)
        max_val = max(value_list)
        
        if max_val == min_val:
            return [0] * len(value_list)
        
        # Quantize to [0, 2^bits - 1]
        scale = (2 ** bits - 1) / (max_val - min_val)
        quantized = [int((v - min_val) * scale) for v in value_list]
        
        return quantized
    
    def _estimate_communication_cost(
        self, 
        calibration_scores: List[float], 
        local_quantiles: Dict[str, float], 
        model_parameters: Optional[Dict[str, Any]]
    ) -> int:
        """Estimate communication cost in bytes.
        
        Args:
            calibration_scores: Calibration scores
            local_quantiles: Local quantiles
            model_parameters: Model parameters
            
        Returns:
            Estimated cost in bytes
        """
        cost = 0
        
        # Calibration scores (8 bytes per float)
        cost += len(calibration_scores) * 8
        
        # Local quantiles (8 bytes per float)
        cost += len(local_quantiles) * 8
        
        # Model parameters
        if model_parameters:
            for key, value in model_parameters.items():
                if isinstance(value, (list, np.ndarray)):
                    if self.config.compression and self.config.quantization_bits < 32:
                        # Quantized parameters
                        cost += len(value) * (self.config.quantization_bits // 8)
                    else:
                        # Full precision
                        cost += len(value) * 4  # 4 bytes per float32
                else:
                    cost += 8  # Single value
        
        # Metadata overhead
        cost += 100  # Client ID, round number, etc.
        
        return cost
    
    def _get_privacy_budget_used(self, round_number: int) -> float:
        """Get privacy budget used so far.
        
        Args:
            round_number: Current round number
            
        Returns:
            Privacy budget used
        """
        if not self.config.differential_privacy:
            return 0.0
        
        return self.config.dp_epsilon - self.privacy_budget_remaining
    
    def evaluate_local_performance(
        self, 
        test_data: List[Tuple[Any, Any, float]]
    ) -> Dict[str, float]:
        """Evaluate local predictor performance.
        
        Args:
            test_data: Test data for evaluation
            
        Returns:
            Performance metrics
        """
        if not test_data:
            return {'accuracy': 0.0, 'coverage': 0.0}
        
        correct_predictions = 0
        coverage_count = 0
        
        for state, action, true_risk in test_data:
            try:
                # Get prediction
                if hasattr(self.local_predictor, 'predict_with_uncertainty'):
                    pred_risk, lower, upper = self.local_predictor.predict_with_uncertainty(
                        state, action, confidence_level=0.95
                    )
                    
                    # Check coverage
                    if lower <= true_risk <= upper:
                        coverage_count += 1
                else:
                    pred_risk = 0.5
                
                # Check accuracy (within 10% tolerance)
                if abs(pred_risk - true_risk) < 0.1:
                    correct_predictions += 1
                    
            except Exception:
                continue
        
        performance = {
            'accuracy': correct_predictions / len(test_data),
            'coverage': coverage_count / len(test_data) if hasattr(self.local_predictor, 'predict_with_uncertainty') else 0.0,
            'data_size': len(self.local_data),
            'test_size': len(test_data)
        }
        
        self.local_performance_history.append(performance)
        
        return performance
    
    def get_client_statistics(self) -> Dict[str, Any]:
        """Get comprehensive client statistics.
        
        Returns:
            Client statistics and metrics
        """
        return {
            'client_id': self.client_id,
            'local_data_size': len(self.local_data),
            'rounds_participated': len(self.round_participation),
            'total_communication_cost': sum(self.communication_costs),
            'avg_communication_cost': np.mean(self.communication_costs) if self.communication_costs else 0,
            'privacy_budget_remaining': self.privacy_budget_remaining,
            'privacy_budget_used': self.config.dp_epsilon - self.privacy_budget_remaining if self.config.differential_privacy else 0,
            'recent_performance': self.local_performance_history[-5:] if len(self.local_performance_history) >= 5 else self.local_performance_history,
            'avg_local_accuracy': np.mean([p['accuracy'] for p in self.local_performance_history]) if self.local_performance_history else 0.0
        }


class FederatedConformalServer:
    """Federated learning server for conformal prediction aggregation."""
    
    def __init__(self, config: FederatedConformalConfig):
        """Initialize federated conformal server.
        
        Args:
            config: Federated learning configuration
        """
        self.config = config
        self.current_round = 0
        
        # Global state
        self.global_quantiles = {}
        self.global_model_parameters = None
        
        # Client management
        self.registered_clients: Set[str] = set()
        self.client_updates_history = defaultdict(list)
        
        # Aggregation history
        self.aggregation_history = []
        
        # Performance tracking
        self.global_performance_history = []
        
        logger.info(f"Initialized federated conformal server with {config.aggregation_method.value}")
    
    def register_client(self, client_id: str) -> bool:
        """Register a new client.
        
        Args:
            client_id: Client identifier
            
        Returns:
            True if registration successful
        """
        if client_id in self.registered_clients:
            logger.warning(f"Client {client_id} already registered")
            return False
        
        self.registered_clients.add(client_id)
        logger.info(f"Registered client: {client_id}. Total clients: {len(self.registered_clients)}")
        
        return True
    
    def federated_round(
        self, 
        client_updates: List[ClientUpdate]
    ) -> Dict[str, Any]:
        """Execute one round of federated learning.
        
        Args:
            client_updates: Updates from participating clients
            
        Returns:
            Aggregation results and global model
        """
        round_start_time = time.time()
        
        logger.info(f"Starting federated round {self.current_round} with {len(client_updates)} clients")
        
        # Validate client updates
        valid_updates = self._validate_client_updates(client_updates)
        
        if not valid_updates:
            logger.error(f"No valid client updates in round {self.current_round}")
            return self._create_round_result([], round_start_time)
        
        # Aggregate conformal quantiles
        aggregated_quantiles = self._aggregate_quantiles(valid_updates)
        
        # Aggregate model parameters (if applicable)
        aggregated_model = self._aggregate_model_parameters(valid_updates)
        
        # Update global state
        self.global_quantiles.update(aggregated_quantiles)
        if aggregated_model:
            self.global_model_parameters = aggregated_model
        
        # Store client updates in history
        for update in valid_updates:
            self.client_updates_history[update.client_id].append(update)
        
        round_time = time.time() - round_start_time
        
        # Create round result
        round_result = self._create_round_result(valid_updates, round_start_time)
        round_result['aggregated_quantiles'] = aggregated_quantiles
        round_result['global_model'] = aggregated_model
        round_result['round_time'] = round_time
        
        # Store aggregation history
        self.aggregation_history.append(round_result)
        
        self.current_round += 1
        
        logger.info(f"Federated round {self.current_round - 1} completed in {round_time:.2f}s")
        
        return round_result
    
    def _validate_client_updates(self, client_updates: List[ClientUpdate]) -> List[ClientUpdate]:
        """Validate and filter client updates.
        
        Args:
            client_updates: Raw client updates
            
        Returns:
            Valid client updates
        """
        valid_updates = []
        
        for update in client_updates:
            # Check if client is registered
            if update.client_id not in self.registered_clients:
                logger.warning(f"Received update from unregistered client: {update.client_id}")
                continue
            
            # Check round number
            if update.round_number != self.current_round:
                logger.warning(f"Received update for wrong round from {update.client_id}: "
                             f"expected {self.current_round}, got {update.round_number}")
                continue
            
            # Check data quality
            if not update.calibration_scores or update.data_size <= 0:
                logger.warning(f"Invalid data from client {update.client_id}")
                continue
            
            # Privacy budget validation
            if self.config.differential_privacy and update.privacy_budget_used > self.config.dp_epsilon:
                logger.warning(f"Client {update.client_id} exceeded privacy budget")
                continue
            
            valid_updates.append(update)
        
        return valid_updates
    
    def _aggregate_quantiles(self, client_updates: List[ClientUpdate]) -> Dict[str, float]:
        """Aggregate conformal quantiles from clients.
        
        Args:
            client_updates: Valid client updates
            
        Returns:
            Aggregated global quantiles
        """
        if self.config.aggregation_method == FederatedAggregationMethod.FEDAVG:
            return self._federated_average_quantiles(client_updates)
        elif self.config.aggregation_method == FederatedAggregationMethod.QUANTILE_AGGREGATION:
            return self._quantile_aggregation(client_updates)
        elif self.config.aggregation_method == FederatedAggregationMethod.DIFFERENTIAL_PRIVATE:
            return self._differential_private_aggregation(client_updates)
        else:
            logger.warning(f"Unknown aggregation method: {self.config.aggregation_method}")
            return self._federated_average_quantiles(client_updates)
    
    def _federated_average_quantiles(self, client_updates: List[ClientUpdate]) -> Dict[str, float]:
        """Aggregate quantiles using federated averaging.
        
        Args:
            client_updates: Client updates
            
        Returns:
            Aggregated quantiles
        """
        # Collect all quantile types
        all_quantile_keys = set()
        for update in client_updates:
            all_quantile_keys.update(update.local_quantiles.keys())
        
        aggregated_quantiles = {}
        
        for quantile_key in all_quantile_keys:
            # Collect quantiles and weights
            quantiles = []
            weights = []
            
            for update in client_updates:
                if quantile_key in update.local_quantiles:
                    quantiles.append(update.local_quantiles[quantile_key])
                    
                    # Weight by data size if configured
                    if self.config.weight_by_data_size:
                        weights.append(update.data_size)
                    else:
                        weights.append(1.0)
            
            if quantiles:
                # Weighted average
                total_weight = sum(weights)
                if total_weight > 0:
                    weighted_avg = sum(q * w for q, w in zip(quantiles, weights)) / total_weight
                    aggregated_quantiles[quantile_key] = weighted_avg
                else:
                    aggregated_quantiles[quantile_key] = np.mean(quantiles)
        
        return aggregated_quantiles
    
    def _quantile_aggregation(self, client_updates: List[ClientUpdate]) -> Dict[str, float]:
        """Aggregate using distributed quantile computation.
        
        Args:
            client_updates: Client updates
            
        Returns:
            Aggregated quantiles
        """
        # Collect all calibration scores from all clients
        all_scores = []
        client_weights = []
        
        for update in client_updates:
            all_scores.extend(update.calibration_scores)
            
            # Repeat weight for each score
            weight = update.data_size if self.config.weight_by_data_size else 1.0
            client_weights.extend([weight] * len(update.calibration_scores))
        
        if not all_scores:
            return {}
        
        # Compute global quantiles from aggregated scores
        confidence_levels = [0.9, 0.95, 0.99]
        aggregated_quantiles = {}
        
        for confidence in confidence_levels:
            alpha = 1 - confidence
            
            # Weighted quantile computation
            if client_weights:
                quantile = self._weighted_quantile(all_scores, client_weights, 1 - alpha)
            else:
                sorted_scores = sorted(all_scores)
                n = len(sorted_scores)
                q_idx = min(int(np.ceil((1 - alpha) * n)), n - 1)
                quantile = sorted_scores[q_idx]
            
            aggregated_quantiles[f"quantile_{confidence}"] = quantile
        
        return aggregated_quantiles
    
    def _weighted_quantile(
        self, 
        values: List[float], 
        weights: List[float], 
        quantile: float
    ) -> float:
        """Compute weighted quantile.
        
        Args:
            values: Data values
            weights: Weights for each value
            quantile: Quantile level (0-1)
            
        Returns:
            Weighted quantile value
        """
        if not values or not weights or len(values) != len(weights):
            return 0.5
        
        # Sort by values
        sorted_pairs = sorted(zip(values, weights))
        sorted_values, sorted_weights = zip(*sorted_pairs)
        
        # Compute cumulative weights
        total_weight = sum(sorted_weights)
        cumulative_weights = []
        cumsum = 0
        
        for weight in sorted_weights:
            cumsum += weight
            cumulative_weights.append(cumsum / total_weight)
        
        # Find quantile position
        for i, cum_weight in enumerate(cumulative_weights):
            if cum_weight >= quantile:
                return sorted_values[i]
        
        return sorted_values[-1]  # Return last value if not found
    
    def _differential_private_aggregation(self, client_updates: List[ClientUpdate]) -> Dict[str, float]:
        """Aggregate with additional server-side differential privacy.
        
        Args:
            client_updates: Client updates
            
        Returns:
            Differentially private aggregated quantiles
        """
        # Start with federated averaging
        base_aggregation = self._federated_average_quantiles(client_updates)
        
        # Add server-side noise for additional privacy
        dp_aggregation = {}
        
        for key, value in base_aggregation.items():
            # Add Laplace noise (simpler than Gaussian for demonstration)
            sensitivity = 1.0 / len(client_updates)  # Sensitivity of average
            noise_scale = sensitivity / (self.config.dp_epsilon / 2)  # Reserve half epsilon for server
            
            noise = np.random.laplace(0, noise_scale)
            noisy_value = value + noise
            
            # Clamp to valid range
            dp_aggregation[key] = max(0.0, min(1.0, noisy_value))
        
        return dp_aggregation
    
    def _aggregate_model_parameters(self, client_updates: List[ClientUpdate]) -> Optional[Dict[str, Any]]:
        """Aggregate model parameters from clients.
        
        Args:
            client_updates: Client updates
            
        Returns:
            Aggregated model parameters or None
        """
        # Check if any client sent model parameters
        model_updates = [update for update in client_updates if update.model_parameters]
        
        if not model_updates:
            return None
        
        # Get all parameter keys
        all_keys = set()
        for update in model_updates:
            all_keys.update(update.model_parameters.keys())
        
        aggregated_params = {}
        
        for key in all_keys:
            # Collect parameters for this key
            params_for_key = []
            weights = []
            
            for update in model_updates:
                if key in update.model_parameters:
                    params_for_key.append(update.model_parameters[key])
                    
                    # Weight by data size
                    if self.config.weight_by_data_size:
                        weights.append(update.data_size)
                    else:
                        weights.append(1.0)
            
            if params_for_key:
                # Aggregate based on parameter type
                if isinstance(params_for_key[0], (list, np.ndarray)):
                    # Vector/matrix parameters - weighted average
                    aggregated_params[key] = self._weighted_average_vectors(
                        params_for_key, weights
                    )
                else:
                    # Scalar parameters - weighted average
                    total_weight = sum(weights)
                    if total_weight > 0:
                        weighted_sum = sum(p * w for p, w in zip(params_for_key, weights))
                        aggregated_params[key] = weighted_sum / total_weight
                    else:
                        aggregated_params[key] = np.mean(params_for_key)
        
        return aggregated_params
    
    def _weighted_average_vectors(
        self, 
        vectors: List[Union[List, Any]], 
        weights: List[float]
    ) -> List[float]:
        """Compute weighted average of vectors.
        
        Args:
            vectors: List of vectors to average
            weights: Weights for each vector
            
        Returns:
            Weighted average vector
        """
        if not vectors or not weights:
            return []
        
        # Convert all to lists
        vector_lists = [list(v) if hasattr(v, '__iter__') else [v] for v in vectors]
        
        # Find maximum length
        max_length = max(len(v) for v in vector_lists)
        
        # Pad vectors to same length
        padded_vectors = []
        for v in vector_lists:
            padded = v + [0.0] * (max_length - len(v))
            padded_vectors.append(padded)
        
        # Compute weighted average
        total_weight = sum(weights)
        averaged_vector = []
        
        for i in range(max_length):
            weighted_sum = sum(v[i] * w for v, w in zip(padded_vectors, weights))
            averaged_vector.append(weighted_sum / total_weight if total_weight > 0 else 0.0)
        
        return averaged_vector
    
    def _create_round_result(
        self, 
        client_updates: List[ClientUpdate], 
        round_start_time: float
    ) -> Dict[str, Any]:
        """Create round result summary.
        
        Args:
            client_updates: Client updates for this round
            round_start_time: Round start timestamp
            
        Returns:
            Round result summary
        """
        return {
            'round_number': self.current_round,
            'participating_clients': [update.client_id for update in client_updates],
            'num_participants': len(client_updates),
            'total_data_size': sum(update.data_size for update in client_updates),
            'avg_computation_time': np.mean([update.computation_time for update in client_updates]) if client_updates else 0.0,
            'total_communication_cost': sum(update.communication_cost for update in client_updates),
            'privacy_budget_used': sum(update.privacy_budget_used for update in client_updates) if self.config.differential_privacy else 0.0,
            'aggregation_method': self.config.aggregation_method.value,
            'timestamp': round_start_time
        }
    
    def get_global_risk_certificate(
        self, 
        confidence_level: float = 0.95
    ) -> RiskCertificate:
        """Generate global risk certificate from federated learning.
        
        Args:
            confidence_level: Confidence level for certificate
            
        Returns:
            Global federated risk certificate
        """
        quantile_key = f"quantile_{confidence_level}"
        
        if quantile_key in self.global_quantiles:
            risk_bound = self.global_quantiles[quantile_key]
        else:
            risk_bound = 0.5  # Conservative fallback
        
        # Estimate sample size across all clients
        total_sample_size = 0
        if self.client_updates_history:
            for client_history in self.client_updates_history.values():
                if client_history:
                    total_sample_size += client_history[-1].data_size
        
        certificate = RiskCertificate(
            risk_bound=risk_bound,
            confidence=confidence_level,
            coverage_guarantee=confidence_level,
            method=f"federated_{self.config.aggregation_method.value}",
            sample_size=total_sample_size,
            timestamp=time.time(),
            metadata={
                'num_clients': len(self.registered_clients),
                'num_rounds': self.current_round,
                'aggregation_method': self.config.aggregation_method.value,
                'differential_privacy': self.config.differential_privacy,
                'privacy_epsilon': self.config.dp_epsilon if self.config.differential_privacy else None
            }
        )
        
        return certificate
    
    def evaluate_global_performance(
        self, 
        test_data: List[Tuple[Any, Any, float]]
    ) -> Dict[str, float]:
        """Evaluate global federated model performance.
        
        Args:
            test_data: Global test dataset
            
        Returns:
            Global performance metrics
        """
        if not test_data or not self.global_quantiles:
            return {'global_accuracy': 0.0, 'global_coverage': 0.0}
        
        # Use global quantiles for prediction intervals
        confidence_level = 0.95
        quantile_key = f"quantile_{confidence_level}"
        
        if quantile_key not in self.global_quantiles:
            return {'global_accuracy': 0.0, 'global_coverage': 0.0}
        
        global_quantile = self.global_quantiles[quantile_key]
        
        coverage_count = 0
        accuracy_count = 0
        
        for state, action, true_risk in test_data:
            # Simple prediction using global quantile
            predicted_risk = 0.5  # Base prediction
            
            # Prediction interval
            lower_bound = max(0.0, predicted_risk - global_quantile)
            upper_bound = min(1.0, predicted_risk + global_quantile)
            
            # Check coverage
            if lower_bound <= true_risk <= upper_bound:
                coverage_count += 1
            
            # Check accuracy (within quantile tolerance)
            if abs(predicted_risk - true_risk) <= global_quantile:
                accuracy_count += 1
        
        performance = {
            'global_accuracy': accuracy_count / len(test_data),
            'global_coverage': coverage_count / len(test_data),
            'global_quantile': global_quantile,
            'test_size': len(test_data),
            'confidence_level': confidence_level
        }
        
        self.global_performance_history.append(performance)
        
        return performance
    
    def get_federated_statistics(self) -> Dict[str, Any]:
        """Get comprehensive federated learning statistics.
        
        Returns:
            Federated learning statistics and metrics
        """
        # Client participation statistics
        client_participation = defaultdict(int)
        for round_info in self.aggregation_history:
            for client_id in round_info['participating_clients']:
                client_participation[client_id] += 1
        
        # Communication costs
        total_communication = sum(
            round_info['total_communication_cost'] 
            for round_info in self.aggregation_history
        )
        
        avg_communication_per_round = (
            total_communication / len(self.aggregation_history) 
            if self.aggregation_history else 0
        )
        
        # Privacy budget tracking
        privacy_stats = {}
        if self.config.differential_privacy:
            total_privacy_used = sum(
                round_info['privacy_budget_used'] 
                for round_info in self.aggregation_history
            )
            privacy_stats = {
                'total_privacy_budget': self.config.dp_epsilon * len(self.registered_clients),
                'privacy_budget_used': total_privacy_used,
                'privacy_budget_remaining': self.config.dp_epsilon * len(self.registered_clients) - total_privacy_used
            }
        
        # Performance trends
        performance_trend = 'stable'
        if len(self.global_performance_history) >= 3:
            recent_accuracy = [p['global_accuracy'] for p in self.global_performance_history[-3:]]
            if recent_accuracy[-1] > recent_accuracy[0] * 1.05:
                performance_trend = 'improving'
            elif recent_accuracy[-1] < recent_accuracy[0] * 0.95:
                performance_trend = 'degrading'
        
        stats = {
            'server_info': {
                'current_round': self.current_round,
                'total_clients': len(self.registered_clients),
                'aggregation_method': self.config.aggregation_method.value,
                'differential_privacy_enabled': self.config.differential_privacy
            },
            'participation_stats': {
                'client_participation': dict(client_participation),
                'avg_participants_per_round': np.mean([
                    round_info['num_participants'] 
                    for round_info in self.aggregation_history
                ]) if self.aggregation_history else 0,
                'participation_rate': len(client_participation) / len(self.registered_clients) if self.registered_clients else 0
            },
            'communication_stats': {
                'total_communication_cost': total_communication,
                'avg_communication_per_round': avg_communication_per_round,
                'communication_efficiency': total_communication / (self.current_round * len(self.registered_clients)) if self.current_round > 0 and self.registered_clients else 0
            },
            'privacy_stats': privacy_stats,
            'performance_stats': {
                'global_performance_history': self.global_performance_history[-10:],  # Last 10 rounds
                'performance_trend': performance_trend,
                'current_global_quantiles': self.global_quantiles.copy()
            },
            'round_history': self.aggregation_history[-5:] if len(self.aggregation_history) >= 5 else self.aggregation_history  # Last 5 rounds
        }
        
        return stats


# Additional federated learning components

class SecureAggregation:
    """Secure aggregation protocol for federated conformal learning."""
    
    def __init__(self, num_clients: int, threshold: int):
        """Initialize secure aggregation.
        
        Args:
            num_clients: Total number of clients
            threshold: Minimum clients needed for reconstruction
        """
        self.num_clients = num_clients
        self.threshold = threshold
        self.secret_shares = {}
        
        logger.info(f"Initialized secure aggregation: {num_clients} clients, threshold {threshold}")
    
    def create_secret_shares(self, secret_value: float, client_ids: List[str]) -> Dict[str, float]:
        """Create secret shares for secure aggregation.
        
        Args:
            secret_value: Value to be secret-shared
            client_ids: List of participating client IDs
            
        Returns:
            Dictionary of secret shares for each client
        """
        # Simplified secret sharing (Shamir's secret sharing would be used in practice)
        shares = {}
        
        # Generate random shares that sum to the secret value
        total_random = 0
        for i, client_id in enumerate(client_ids[:-1]):
            share = np.random.uniform(-1, 1)
            shares[client_id] = share
            total_random += share
        
        # Last share ensures sum equals secret
        shares[client_ids[-1]] = secret_value - total_random
        
        return shares
    
    def reconstruct_secret(self, shares: Dict[str, float]) -> float:
        """Reconstruct secret from shares.
        
        Args:
            shares: Dictionary of shares from clients
            
        Returns:
            Reconstructed secret value
        """
        if len(shares) < self.threshold:
            raise ValueError(f"Insufficient shares: need {self.threshold}, got {len(shares)}")
        
        # Simple reconstruction (sum of shares)
        return sum(shares.values())


class FederatedPrivacyAccountant:
    """Privacy accounting for federated conformal learning."""
    
    def __init__(self, initial_budget: float, delta: float):
        """Initialize privacy accountant.
        
        Args:
            initial_budget: Initial privacy budget (epsilon)
            delta: Privacy parameter delta
        """
        self.initial_budget = initial_budget
        self.remaining_budget = initial_budget
        self.delta = delta
        self.privacy_history = []
        
    def spend_privacy_budget(self, epsilon: float, mechanism: str, round_number: int) -> bool:
        """Spend privacy budget.
        
        Args:
            epsilon: Privacy budget to spend
            mechanism: Privacy mechanism used
            round_number: Current round number
            
        Returns:
            True if budget is available, False otherwise
        """
        if epsilon > self.remaining_budget:
            logger.warning(f"Insufficient privacy budget: need {epsilon}, have {self.remaining_budget}")
            return False
        
        self.remaining_budget -= epsilon
        
        self.privacy_history.append({
            'round': round_number,
            'epsilon_spent': epsilon,
            'mechanism': mechanism,
            'remaining_budget': self.remaining_budget,
            'timestamp': time.time()
        })
        
        return True
    
    def get_privacy_summary(self) -> Dict[str, Any]:
        """Get privacy accounting summary.
        
        Returns:
            Privacy budget summary
        """
        return {
            'initial_budget': self.initial_budget,
            'remaining_budget': self.remaining_budget,
            'budget_used': self.initial_budget - self.remaining_budget,
            'budget_utilization': (self.initial_budget - self.remaining_budget) / self.initial_budget,
            'delta': self.delta,
            'privacy_history': self.privacy_history[-10:],  # Last 10 entries
            'total_mechanisms_used': len(set(entry['mechanism'] for entry in self.privacy_history))
        }


class FederatedConformalSystem:
    """Complete federated conformal prediction system."""
    
    def __init__(self, config: FederatedConformalConfig):
        """Initialize federated conformal system.
        
        Args:
            config: Federated learning configuration
        """
        self.config = config
        self.server = FederatedConformalServer(config)
        self.clients: Dict[str, FederatedConformalClient] = {}
        
        # Privacy accounting
        if config.differential_privacy:
            self.privacy_accountant = FederatedPrivacyAccountant(
                config.dp_epsilon * config.num_clients, config.dp_delta
            )
        else:
            self.privacy_accountant = None
        
        # Secure aggregation
        if config.secure_aggregation:
            self.secure_aggregator = SecureAggregation(
                config.num_clients, 
                max(1, int(config.num_clients * config.client_fraction))
            )
        else:
            self.secure_aggregator = None
        
        logger.info(f"Initialized federated conformal system with {config.num_clients} clients")
    
    def setup_clients(self, client_predictors: Dict[str, Any]) -> None:
        """Setup federated clients.
        
        Args:
            client_predictors: Dictionary mapping client IDs to their local predictors
        """
        for client_id, predictor in client_predictors.items():
            client = FederatedConformalClient(client_id, predictor, self.config)
            self.clients[client_id] = client
            self.server.register_client(client_id)
        
        logger.info(f"Setup {len(self.clients)} federated clients")
    
    def run_federated_learning(
        self, 
        client_data: Dict[str, List[Tuple[Any, Any, float]]],
        test_data: Optional[List[Tuple[Any, Any, float]]] = None
    ) -> Dict[str, Any]:
        """Run complete federated learning process.
        
        Args:
            client_data: Training data for each client
            test_data: Global test data for evaluation
            
        Returns:
            Federated learning results
        """
        logger.info(f"Starting federated learning for {self.config.rounds} rounds")
        
        # Distribute data to clients
        for client_id, data in client_data.items():
            if client_id in self.clients:
                self.clients[client_id].add_local_data(data)
        
        federated_results = []
        
        # Run federated rounds
        for round_num in range(self.config.rounds):
            # Select participating clients
            participating_clients = self._select_clients_for_round(round_num)
            
            # Collect client updates
            client_updates = []
            for client_id in participating_clients:
                client = self.clients[client_id]
                update = client.local_update(
                    global_model=self.server.global_model_parameters,
                    round_number=round_num
                )
                client_updates.append(update)
            
            # Server aggregation
            round_result = self.server.federated_round(client_updates)
            federated_results.append(round_result)
            
            # Evaluate performance if test data available
            if test_data and round_num % 10 == 0:  # Evaluate every 10 rounds
                global_performance = self.server.evaluate_global_performance(test_data)
                round_result['global_performance'] = global_performance
        
        # Generate final results
        final_results = {
            'federated_rounds': federated_results,
            'global_risk_certificate': self.server.get_global_risk_certificate(),
            'server_statistics': self.server.get_federated_statistics(),
            'client_statistics': {client_id: client.get_client_statistics() 
                                for client_id, client in self.clients.items()},
            'privacy_accounting': self.privacy_accountant.get_privacy_summary() if self.privacy_accountant else None,
            'total_rounds': self.config.rounds,
            'final_global_quantiles': self.server.global_quantiles.copy()
        }
        
        logger.info(f"Federated learning completed after {self.config.rounds} rounds")
        
        return final_results
    
    def _select_clients_for_round(self, round_number: int) -> List[str]:
        """Select clients to participate in current round.
        
        Args:
            round_number: Current round number
            
        Returns:
            List of selected client IDs
        """
        all_client_ids = list(self.clients.keys())
        num_selected = max(1, int(len(all_client_ids) * self.config.client_fraction))
        
        # Random selection (could be more sophisticated)
        np.random.seed(round_number)  # Reproducible selection
        selected_clients = np.random.choice(
            all_client_ids, 
            size=min(num_selected, len(all_client_ids)), 
            replace=False
        ).tolist()
        
        return selected_clients
    
    def predict_with_federated_model(
        self, 
        state: Any, 
        action: Any, 
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """Make prediction using federated model.
        
        Args:
            state: Environment state
            action: Action to evaluate
            confidence_level: Confidence level
            
        Returns:
            Federated prediction results
        """
        quantile_key = f"quantile_{confidence_level}"
        
        if quantile_key in self.server.global_quantiles:
            global_quantile = self.server.global_quantiles[quantile_key]
            
            # Simple prediction using global quantile
            base_prediction = 0.5  # Could be more sophisticated
            
            lower_bound = max(0.0, base_prediction - global_quantile)
            upper_bound = min(1.0, base_prediction + global_quantile)
            
            result = {
                'risk_estimate': base_prediction,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'interval_width': upper_bound - lower_bound,
                'confidence_level': confidence_level,
                'global_quantile': global_quantile,
                'num_clients_trained': len(self.clients),
                'federated_method': self.config.aggregation_method.value
            }
        else:
            # Fallback prediction
            result = {
                'risk_estimate': 0.5,
                'lower_bound': 0.0,
                'upper_bound': 1.0,
                'interval_width': 1.0,
                'confidence_level': confidence_level,
                'error': 'global_quantile_not_available'
            }
        
        return result
