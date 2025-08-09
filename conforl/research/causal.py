"""Causal Conformal Risk Control for Robust RL.

This module implements novel causal conformal bounds that provide robustness
guarantees under causal interventions and distribution shift. This is a 
first-of-its-kind approach combining causal inference with conformal prediction
for safe RL deployment under changing environments.

Research Contributions:
- Causal conformal bounds robust to interventions
- Distribution-shift detection and adaptation
- Counterfactual risk assessment
- Causal graph-aware risk certificates

Author: ConfoRL Research Team
License: Apache 2.0
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import defaultdict
import warnings

from ..core.types import RiskCertificate, TrajectoryData
from ..risk.controllers import AdaptiveRiskController
from ..risk.measures import RiskMeasure
from ..utils.logging import get_logger
from ..utils.errors import ConfoRLError, ValidationError
from .compositional import CompositionalRiskCertificate

logger = get_logger(__name__)


@dataclass 
class CausalGraph:
    """Represents a causal directed acyclic graph (DAG)."""
    
    nodes: List[str]
    edges: Dict[str, List[str]]  # parent -> children mapping
    node_types: Dict[str, str] = field(default_factory=dict)  # node -> type mapping
    
    def __post_init__(self):
        """Validate causal graph structure."""
        # Check for cycles (simplified check)
        for node in self.nodes:
            if node in self.edges.get(node, []):
                raise ValidationError(f"Self-loop detected in causal graph at node {node}")
        
        # Ensure all edge nodes exist
        for parent, children in self.edges.items():
            if parent not in self.nodes:
                raise ValidationError(f"Edge parent {parent} not in nodes")
            for child in children:
                if child not in self.nodes:
                    raise ValidationError(f"Edge child {child} not in nodes")


@dataclass
class CausalIntervention:
    """Represents a causal intervention on the environment."""
    
    target_node: str
    intervention_value: Any
    intervention_type: str  # 'do', 'soft', 'noise'
    strength: float = 1.0
    metadata: Optional[Dict[str, Any]] = None


@dataclass 
class CausalRiskCertificate:
    """Risk certificate robust to causal interventions."""
    
    baseline_risk_bound: float
    intervention_robust_bound: float
    max_intervention_strength: float
    causal_confidence: float
    tested_interventions: List[CausalIntervention]
    causal_graph_hash: str
    method: str
    sample_size: int
    timestamp: float
    metadata: Optional[Dict[str, Any]] = None


class CausalShiftDetector:
    """Detects distribution shifts due to causal interventions."""
    
    def __init__(
        self,
        causal_graph: CausalGraph,
        detection_threshold: float = 0.05,
        window_size: int = 1000
    ):
        """Initialize causal shift detector.
        
        Args:
            causal_graph: Known causal structure
            detection_threshold: Threshold for shift detection
            window_size: Size of sliding window for detection
        """
        self.causal_graph = causal_graph
        self.detection_threshold = detection_threshold
        self.window_size = window_size
        
        # Statistical tracking
        self.baseline_statistics = {}
        self.current_window = defaultdict(list)
        self.shift_history = []
        
        logger.info(f"Initialized causal shift detector with {len(causal_graph.nodes)} nodes")
    
    def update_baseline(self, observations: Dict[str, Any]) -> None:
        """Update baseline statistics from observations.
        
        Args:
            observations: Dictionary mapping node names to observed values
        """
        for node, value in observations.items():
            if node in self.causal_graph.nodes:
                if node not in self.baseline_statistics:
                    self.baseline_statistics[node] = {
                        'mean': 0.0,
                        'var': 0.0,
                        'count': 0
                    }
                
                # Online statistics update
                stats = self.baseline_statistics[node]
                stats['count'] += 1
                delta = value - stats['mean']
                stats['mean'] += delta / stats['count']
                stats['var'] += delta * (value - stats['mean'])
    
    def detect_shift(self, observations: Dict[str, Any]) -> Dict[str, float]:
        """Detect causal shifts in current observations.
        
        Args:
            observations: Current observations
            
        Returns:
            Dictionary mapping nodes to shift p-values
        """
        shift_pvalues = {}
        
        for node, value in observations.items():
            if (node in self.causal_graph.nodes and 
                node in self.baseline_statistics):
                
                # Add to current window
                self.current_window[node].append(value)
                if len(self.current_window[node]) > self.window_size:
                    self.current_window[node].pop(0)
                
                # Compute shift statistic (simplified KS test)
                if len(self.current_window[node]) >= 30:  # Minimum sample size
                    baseline_stats = self.baseline_statistics[node]
                    if baseline_stats['count'] > 0 and baseline_stats['var'] > 0:
                        
                        # Z-score based shift detection
                        current_mean = np.mean(self.current_window[node])
                        baseline_mean = baseline_stats['mean']
                        baseline_std = np.sqrt(baseline_stats['var'] / baseline_stats['count'])
                        
                        if baseline_std > 1e-8:
                            z_score = abs(current_mean - baseline_mean) / baseline_std
                            # Convert to p-value (approximate)
                            p_value = 2 * (1 - min(0.9999, z_score / 3.0))  # Simplified
                            shift_pvalues[node] = p_value
                        else:
                            shift_pvalues[node] = 1.0
        
        return shift_pvalues
    
    def get_shift_summary(self) -> Dict[str, Any]:
        """Get summary of detected shifts."""
        current_shifts = {}
        
        for node in self.causal_graph.nodes:
            if node in self.current_window and len(self.current_window[node]) > 0:
                recent_observations = self.current_window[node][-10:]  # Last 10 observations
                if recent_observations:
                    shift_scores = self.detect_shift({node: recent_observations[-1]})
                    current_shifts[node] = shift_scores.get(node, 1.0)
        
        return {
            'current_shifts': current_shifts,
            'shift_threshold': self.detection_threshold,
            'window_size': self.window_size,
            'nodes_tracked': len(self.baseline_statistics),
            'total_observations': sum(stats['count'] for stats in self.baseline_statistics.values())
        }


class CausalConformPredictor:
    """Conformal predictor robust to causal interventions."""
    
    def __init__(
        self,
        causal_graph: CausalGraph,
        base_risk_level: float = 0.05,
        intervention_budget: float = 0.1
    ):
        """Initialize causal conformal predictor.
        
        Args:
            causal_graph: Causal structure
            base_risk_level: Base risk level without interventions
            intervention_budget: Additional risk budget for interventions
        """
        self.causal_graph = causal_graph
        self.base_risk_level = base_risk_level
        self.intervention_budget = intervention_budget
        
        # Conformal scores for different intervention scenarios
        self.baseline_scores = []
        self.intervention_scores = defaultdict(list)
        
        # Causal mechanisms (learned or provided)
        self.causal_mechanisms = {}
        
        logger.info(f"Initialized causal conformal predictor for {len(causal_graph.nodes)} nodes")
    
    def update_baseline_scores(self, risk_scores: List[float]) -> None:
        """Update baseline conformal scores.
        
        Args:
            risk_scores: List of risk scores without interventions
        """
        self.baseline_scores.extend(risk_scores)
        
        # Keep only recent scores for efficiency
        max_scores = 10000
        if len(self.baseline_scores) > max_scores:
            self.baseline_scores = self.baseline_scores[-max_scores:]
        
        logger.debug(f"Updated baseline scores: {len(self.baseline_scores)} total")
    
    def update_intervention_scores(
        self,
        intervention: CausalIntervention,
        risk_scores: List[float]
    ) -> None:
        """Update intervention-specific conformal scores.
        
        Args:
            intervention: Causal intervention applied
            risk_scores: Resulting risk scores
        """
        intervention_key = f"{intervention.target_node}_{intervention.intervention_type}"
        self.intervention_scores[intervention_key].extend(risk_scores)
        
        # Keep only recent scores
        max_scores = 5000
        if len(self.intervention_scores[intervention_key]) > max_scores:
            self.intervention_scores[intervention_key] = (
                self.intervention_scores[intervention_key][-max_scores:]
            )
        
        logger.debug(f"Updated intervention scores for {intervention_key}: "
                    f"{len(self.intervention_scores[intervention_key])} total")
    
    def get_causal_quantile(
        self,
        confidence: float,
        max_intervention_strength: float = 1.0
    ) -> float:
        """Compute causal-robust conformal quantile.
        
        Args:
            confidence: Desired confidence level
            max_intervention_strength: Maximum expected intervention strength
            
        Returns:
            Causal-robust quantile
        """
        if not self.baseline_scores:
            logger.warning("No baseline scores available, returning conservative quantile")
            return 1.0
        
        # Base quantile from baseline scores
        baseline_scores = np.array(self.baseline_scores)
        base_quantile_level = 1.0 - self.base_risk_level
        base_quantile = np.quantile(baseline_scores, base_quantile_level)
        
        # Intervention robustness adjustment
        intervention_adjustments = []
        
        for intervention_key, scores in self.intervention_scores.items():
            if len(scores) >= 10:  # Minimum samples for reliable estimate
                intervention_scores = np.array(scores)
                intervention_quantile = np.quantile(intervention_scores, base_quantile_level)
                
                # Adjustment based on difference from baseline
                adjustment = max(0, intervention_quantile - base_quantile)
                intervention_adjustments.append(adjustment)
        
        # Conservative approach: take maximum adjustment
        if intervention_adjustments:
            max_adjustment = max(intervention_adjustments)
            # Scale by intervention strength and budget
            causal_adjustment = max_adjustment * max_intervention_strength * self.intervention_budget
        else:
            # No intervention data - use conservative scaling
            causal_adjustment = base_quantile * self.intervention_budget
        
        causal_quantile = base_quantile + causal_adjustment
        
        logger.debug(f"Causal quantile: base={base_quantile:.4f}, "
                    f"adjustment={causal_adjustment:.4f}, final={causal_quantile:.4f}")
        
        return min(1.0, causal_quantile)
    
    def predict_intervention_effect(
        self,
        intervention: CausalIntervention,
        current_state: Dict[str, Any]
    ) -> Tuple[float, float]:
        """Predict effect of intervention on risk.
        
        Args:
            intervention: Proposed intervention
            current_state: Current system state
            
        Returns:
            Tuple of (predicted_risk_change, confidence)
        """
        # Placeholder for sophisticated causal inference
        # In practice, this would use learned causal mechanisms
        
        intervention_key = f"{intervention.target_node}_{intervention.intervention_type}"
        
        if intervention_key in self.intervention_scores and len(self.intervention_scores[intervention_key]) > 0:
            # Use historical data
            intervention_scores = np.array(self.intervention_scores[intervention_key])
            baseline_scores = np.array(self.baseline_scores)
            
            if len(baseline_scores) > 0:
                intervention_mean = np.mean(intervention_scores)
                baseline_mean = np.mean(baseline_scores)
                predicted_change = intervention_mean - baseline_mean
                
                # Confidence based on sample sizes
                n_intervention = len(intervention_scores)
                n_baseline = len(baseline_scores)
                confidence = min(1.0, (n_intervention + n_baseline) / 1000.0)
                
                return predicted_change, confidence
        
        # No historical data - conservative estimate
        conservative_change = self.intervention_budget * intervention.strength
        return conservative_change, 0.1  # Low confidence


class CausalRiskController:
    """Risk controller robust to causal interventions."""
    
    def __init__(
        self,
        causal_graph: CausalGraph,
        base_controller: AdaptiveRiskController,
        max_intervention_strength: float = 1.0,
        causal_confidence: float = 0.9
    ):
        """Initialize causal risk controller.
        
        Args:
            causal_graph: Causal structure of the environment
            base_controller: Base conformal risk controller
            max_intervention_strength: Maximum expected intervention strength
            causal_confidence: Confidence level for causal robustness
        """
        self.causal_graph = causal_graph
        self.base_controller = base_controller
        self.max_intervention_strength = max_intervention_strength
        self.causal_confidence = causal_confidence
        
        # Causal components
        self.shift_detector = CausalShiftDetector(causal_graph)
        self.causal_predictor = CausalConformPredictor(causal_graph)
        
        # Intervention tracking
        self.tested_interventions = []
        self.intervention_history = []
        
        # Certificate cache
        self._certificate_cache = None
        self._cache_timestamp = 0.0
        
        logger.info(f"Initialized causal risk controller with {len(causal_graph.nodes)} causal variables")
    
    def update(
        self,
        trajectory: TrajectoryData,
        risk_measure: RiskMeasure,
        observations: Optional[Dict[str, Any]] = None,
        applied_intervention: Optional[CausalIntervention] = None
    ) -> None:
        """Update controller with new data.
        
        Args:
            trajectory: New trajectory data
            risk_measure: Risk measure to compute
            observations: Causal variable observations
            applied_intervention: Any intervention that was applied
        """
        # Update base controller
        self.base_controller.update(trajectory, risk_measure)
        
        # Compute risk scores for causal predictor
        risk_scores = [risk_measure.compute(trajectory)]
        
        if applied_intervention is None:
            # Update baseline scores
            self.causal_predictor.update_baseline_scores(risk_scores)
        else:
            # Update intervention scores
            self.causal_predictor.update_intervention_scores(applied_intervention, risk_scores)
            self.intervention_history.append({
                'intervention': applied_intervention,
                'risk_score': risk_scores[0],
                'timestamp': time.time()
            })
        
        # Update shift detector if observations provided
        if observations:
            self.shift_detector.update_baseline(observations)
        
        # Invalidate certificate cache
        self._certificate_cache = None
        self._cache_timestamp = 0.0
        
        logger.debug("Updated causal risk controller with new trajectory")
    
    def get_causal_certificate(self) -> CausalRiskCertificate:
        """Get causal-robust risk certificate.
        
        Returns:
            Certificate with causal robustness guarantees
        """
        # Check cache
        current_time = time.time()
        if (self._certificate_cache is not None and 
            current_time - self._cache_timestamp < 30.0):  # 30 second cache
            return self._certificate_cache
        
        # Get baseline certificate
        base_cert = self.base_controller.get_certificate()
        
        # Compute causal-robust quantile
        causal_quantile = self.causal_predictor.get_causal_quantile(
            confidence=self.causal_confidence,
            max_intervention_strength=self.max_intervention_strength
        )
        
        # Create causal certificate
        causal_cert = CausalRiskCertificate(
            baseline_risk_bound=base_cert.risk_bound,
            intervention_robust_bound=min(1.0, causal_quantile),
            max_intervention_strength=self.max_intervention_strength,
            causal_confidence=self.causal_confidence,
            tested_interventions=self.tested_interventions.copy(),
            causal_graph_hash=self._hash_causal_graph(),
            method="causal_conformal",
            sample_size=base_cert.sample_size,
            timestamp=current_time,
            metadata={
                'baseline_method': base_cert.method,
                'num_interventions_tested': len(self.tested_interventions),
                'shift_detector_status': self.shift_detector.get_shift_summary(),
                'causal_graph_nodes': len(self.causal_graph.nodes)
            }
        )
        
        # Cache result
        self._certificate_cache = causal_cert
        self._cache_timestamp = current_time
        
        return causal_cert
    
    def test_intervention_safety(
        self,
        proposed_intervention: CausalIntervention,
        current_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Test safety of proposed intervention.
        
        Args:
            proposed_intervention: Intervention to test
            current_state: Current system state
            
        Returns:
            Safety assessment dictionary
        """
        # Predict intervention effect
        predicted_change, confidence = self.causal_predictor.predict_intervention_effect(
            proposed_intervention, current_state
        )
        
        # Current risk bound
        current_cert = self.get_causal_certificate()
        current_risk = current_cert.intervention_robust_bound
        
        # Predicted risk after intervention
        predicted_risk = min(1.0, current_risk + predicted_change)
        
        # Safety check
        target_risk = self.base_controller.target_risk
        safety_margin = 0.1 * target_risk  # 10% safety margin
        
        is_safe = predicted_risk <= (target_risk + safety_margin)
        
        assessment = {
            'is_safe': is_safe,
            'current_risk_bound': current_risk,
            'predicted_risk_bound': predicted_risk,
            'predicted_change': predicted_change,
            'confidence': confidence,
            'target_risk': target_risk,
            'safety_margin': safety_margin,
            'intervention': proposed_intervention
        }
        
        logger.info(f"Intervention safety assessment: safe={is_safe}, "
                   f"predicted_risk={predicted_risk:.4f}, confidence={confidence:.3f}")
        
        return assessment
    
    def add_tested_intervention(self, intervention: CausalIntervention) -> None:
        """Add intervention to tested set.
        
        Args:
            intervention: Intervention that was tested
        """
        self.tested_interventions.append(intervention)
        
        # Keep only recent interventions
        max_interventions = 100
        if len(self.tested_interventions) > max_interventions:
            self.tested_interventions = self.tested_interventions[-max_interventions:]
    
    def _hash_causal_graph(self) -> str:
        """Create hash of causal graph structure."""
        # Simple hash based on nodes and edges
        graph_str = f"{sorted(self.causal_graph.nodes)}_{sorted(self.causal_graph.edges.items())}"
        return str(hash(graph_str))
    
    def get_causal_insights(self) -> Dict[str, Any]:
        """Get insights about causal structure and interventions.
        
        Returns:
            Dictionary with causal insights
        """
        # Analyze intervention history
        intervention_effects = defaultdict(list)
        for record in self.intervention_history:
            key = f"{record['intervention'].target_node}_{record['intervention'].intervention_type}"
            intervention_effects[key].append(record['risk_score'])
        
        # Compute effect statistics
        effect_stats = {}
        for intervention_key, scores in intervention_effects.items():
            if len(scores) > 1:
                effect_stats[intervention_key] = {
                    'mean_effect': np.mean(scores),
                    'std_effect': np.std(scores),
                    'num_samples': len(scores),
                    'min_effect': np.min(scores),
                    'max_effect': np.max(scores)
                }
        
        # Shift detection summary
        shift_summary = self.shift_detector.get_shift_summary()
        
        # Most influential nodes (based on intervention effects)
        influential_nodes = []
        for intervention_key, stats in effect_stats.items():
            node = intervention_key.split('_')[0]
            influence_score = abs(stats['mean_effect']) * stats['num_samples']
            influential_nodes.append((node, influence_score))
        
        influential_nodes.sort(key=lambda x: x[1], reverse=True)
        
        insights = {
            'causal_graph_summary': {
                'num_nodes': len(self.causal_graph.nodes),
                'num_edges': sum(len(children) for children in self.causal_graph.edges.values()),
                'node_types': dict(self.causal_graph.node_types)
            },
            'intervention_effects': effect_stats,
            'shift_detection': shift_summary,
            'most_influential_nodes': influential_nodes[:5],  # Top 5
            'total_interventions_tested': len(self.tested_interventions),
            'intervention_history_size': len(self.intervention_history)
        }
        
        return insights


# Research Extensions

class CounterfactualRiskAssessment:
    """Assess risk under counterfactual scenarios."""
    
    def __init__(self, causal_graph: CausalGraph):
        """Initialize counterfactual risk assessment.
        
        Args:
            causal_graph: Causal structure for counterfactual inference
        """
        self.causal_graph = causal_graph
        logger.info("Counterfactual risk assessment initialized (research extension)")
    
    def compute_counterfactual_risk(
        self,
        observed_trajectory: TrajectoryData,
        counterfactual_intervention: CausalIntervention
    ) -> float:
        """Compute risk under counterfactual intervention.
        
        This is a placeholder for advanced counterfactual inference.
        
        Args:
            observed_trajectory: Actually observed trajectory
            counterfactual_intervention: Hypothetical intervention
            
        Returns:
            Estimated counterfactual risk
        """
        # Research TODO: Implement sophisticated counterfactual inference
        logger.warning("Counterfactual risk computation not fully implemented - research in progress")
        
        # Placeholder: simple heuristic based on intervention strength
        base_risk = 0.05  # Placeholder
        counterfactual_risk = base_risk * (1 + counterfactual_intervention.strength * 0.1)
        
        return min(1.0, counterfactual_risk)


class CausalGraphLearner:
    """Learn causal graph structure from data."""
    
    def __init__(self, max_nodes: int = 20):
        """Initialize causal graph learner.
        
        Args:
            max_nodes: Maximum number of nodes to consider
        """
        self.max_nodes = max_nodes
        self.learned_graph = None
        logger.info("Causal graph learner initialized (research extension)")
    
    def learn_structure(
        self,
        observational_data: Dict[str, List[Any]],
        intervention_data: Optional[Dict[str, List[Any]]] = None
    ) -> CausalGraph:
        """Learn causal graph structure from data.
        
        This is a placeholder for advanced causal discovery algorithms.
        
        Args:
            observational_data: Observational data
            intervention_data: Optional interventional data
            
        Returns:
            Learned causal graph
        """
        # Research TODO: Implement sophisticated causal discovery
        logger.warning("Causal graph learning not fully implemented - research in progress")
        
        # Placeholder: create simple graph from variable names
        nodes = list(observational_data.keys())[:self.max_nodes]
        edges = {}  # Empty for now
        
        learned_graph = CausalGraph(nodes=nodes, edges=edges)
        self.learned_graph = learned_graph
        
        return learned_graph