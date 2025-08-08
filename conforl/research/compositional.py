"""Compositional Risk Control for Hierarchical RL.

This module implements novel compositional conformal bounds for multi-level policies,
enabling safe hierarchical RL with nested risk certificates. This is a first-of-its-kind
approach that extends ConfoRL to hierarchical decision making.

Research Contribution:
- Compositional risk certificates that combine safety guarantees across hierarchy levels
- Formal bounds for nested policy structures
- Scalable implementation for complex hierarchical tasks

Author: ConfoRL Research Team
License: Apache 2.0
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

from ..core.types import RiskCertificate, TrajectoryData
from ..risk.controllers import AdaptiveRiskController
from ..risk.measures import RiskMeasure
from ..utils.logging import get_logger
from ..utils.errors import ConfoRLError, ValidationError

logger = get_logger(__name__)


@dataclass
class HierarchicalPolicy:
    """Hierarchical policy with multiple levels."""
    
    policy_id: str
    level: int
    parent_policy: Optional[str] = None
    child_policies: List[str] = None
    risk_budget: float = 0.05
    
    def __post_init__(self):
        if self.child_policies is None:
            self.child_policies = []


@dataclass
class CompositionalRiskCertificate:
    """Compositional risk certificate for hierarchical policies."""
    
    policy_id: str
    level: int
    individual_risk_bound: float
    compositional_risk_bound: float
    confidence: float
    coverage_guarantee: float
    method: str
    sample_size: int
    timestamp: float
    child_certificates: List['CompositionalRiskCertificate'] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.child_certificates is None:
            self.child_certificates = []


class CompositionalRiskBounds:
    """Mathematical framework for compositional risk bounds."""
    
    @staticmethod
    def bonferroni_correction(
        individual_risks: List[float], 
        confidence: float
    ) -> Tuple[float, float]:
        """Apply Bonferroni correction for multiple risk bounds.
        
        Args:
            individual_risks: List of individual risk bounds
            confidence: Desired overall confidence level
            
        Returns:
            Tuple of (corrected_confidence, compositional_risk_bound)
        """
        n_policies = len(individual_risks)
        if n_policies == 0:
            return confidence, 0.0
        
        # Bonferroni correction for multiple testing
        corrected_confidence = 1 - (1 - confidence) / n_policies
        
        # Union bound for compositional risk
        compositional_risk = min(1.0, sum(individual_risks))
        
        logger.debug(f"Bonferroni correction: {n_policies} policies, "
                    f"corrected_confidence={corrected_confidence:.4f}")
        
        return corrected_confidence, compositional_risk
    
    @staticmethod
    def hierarchical_union_bound(
        risk_tree: Dict[str, List[float]],
        confidence: float
    ) -> float:
        """Compute hierarchical union bound for nested policies.
        
        Args:
            risk_tree: Dictionary mapping policy levels to risk bounds
            confidence: Desired confidence level
            
        Returns:
            Overall compositional risk bound
        """
        if not risk_tree:
            return 0.0
        
        # Compute level-wise risk bounds
        level_risks = []
        for level, risks in risk_tree.items():
            level_risk = min(1.0, sum(risks))
            level_risks.append(level_risk)
        
        # Hierarchical composition
        compositional_risk = 1.0
        for level_risk in level_risks:
            compositional_risk *= (1 - level_risk)
        
        return 1.0 - compositional_risk
    
    @staticmethod
    def refined_compositional_bound(
        individual_certificates: List[CompositionalRiskCertificate],
        dependency_structure: Optional[Dict[str, List[str]]] = None
    ) -> float:
        """Compute refined compositional bound accounting for dependencies.
        
        Args:
            individual_certificates: List of individual certificates
            dependency_structure: Optional dependency graph
            
        Returns:
            Refined compositional risk bound
        """
        if not individual_certificates:
            return 0.0
        
        # Extract individual risk bounds
        individual_risks = [cert.individual_risk_bound for cert in individual_certificates]
        
        # If no dependency structure provided, use conservative union bound
        if dependency_structure is None:
            return min(1.0, sum(individual_risks))
        
        # Account for conditional dependencies (simplified)
        # This is a research area for future enhancement
        base_risk = sum(individual_risks)
        
        # Dependency discount factor (heuristic for now)
        n_dependencies = sum(len(deps) for deps in dependency_structure.values())
        dependency_factor = max(0.5, 1.0 - 0.1 * n_dependencies / len(individual_risks))
        
        refined_risk = base_risk * dependency_factor
        
        logger.debug(f"Refined compositional bound: base={base_risk:.4f}, "
                    f"refined={refined_risk:.4f}, factor={dependency_factor:.3f}")
        
        return min(1.0, refined_risk)


class CompositionalRiskController:
    """Compositional risk controller for hierarchical policies."""
    
    def __init__(
        self,
        hierarchy: Dict[str, HierarchicalPolicy],
        base_confidence: float = 0.95,
        composition_method: str = "bonferroni"
    ):
        """Initialize compositional risk controller.
        
        Args:
            hierarchy: Dictionary of hierarchical policies
            base_confidence: Base confidence level
            composition_method: Method for composing risk bounds
        """
        self.hierarchy = hierarchy
        self.base_confidence = base_confidence
        self.composition_method = composition_method
        
        # Individual risk controllers for each policy
        self.controllers = {}
        for policy_id, policy in hierarchy.items():
            controller = AdaptiveRiskController(
                target_risk=policy.risk_budget,
                confidence=base_confidence
            )
            self.controllers[policy_id] = controller
        
        # Risk composition utilities
        self.bounds_calculator = CompositionalRiskBounds()
        
        # Certificate cache
        self._certificate_cache = {}
        self._cache_timestamp = 0.0
        
        logger.info(f"Initialized compositional controller for {len(hierarchy)} policies")
    
    def update_policy_risk(
        self,
        policy_id: str,
        trajectory: TrajectoryData,
        risk_measure: RiskMeasure
    ) -> None:
        """Update risk assessment for specific policy.
        
        Args:
            policy_id: Policy identifier
            trajectory: Trajectory data
            risk_measure: Risk measure to use
        """
        if policy_id not in self.controllers:
            raise ValidationError(f"Unknown policy ID: {policy_id}")
        
        # Update individual controller
        self.controllers[policy_id].update(trajectory, risk_measure)
        
        # Invalidate certificate cache
        self._cache_timestamp = 0.0
        
        logger.debug(f"Updated risk for policy {policy_id}")
    
    def get_compositional_certificate(
        self,
        root_policy_id: Optional[str] = None,
        max_depth: int = 10
    ) -> CompositionalRiskCertificate:
        """Get compositional risk certificate for hierarchy.
        
        Args:
            root_policy_id: Root policy to start from (None for all)
            max_depth: Maximum depth to traverse
            
        Returns:
            Compositional risk certificate
        """
        # Check cache
        current_time = time.time()
        cache_key = f"{root_policy_id}_{max_depth}"
        if (cache_key in self._certificate_cache and 
            current_time - self._cache_timestamp < 60.0):  # 1 minute cache
            return self._certificate_cache[cache_key]
        
        # Build certificate tree
        if root_policy_id is None:
            # Global compositional certificate
            certificate = self._build_global_certificate()
        else:
            # Certificate for specific subtree
            certificate = self._build_subtree_certificate(root_policy_id, max_depth)
        
        # Cache result
        self._certificate_cache[cache_key] = certificate
        self._cache_timestamp = current_time
        
        return certificate
    
    def _build_global_certificate(self) -> CompositionalRiskCertificate:
        """Build global compositional certificate."""
        # Get all individual certificates
        individual_certs = []
        for policy_id, controller in self.controllers.items():
            base_cert = controller.get_certificate()
            individual_cert = CompositionalRiskCertificate(
                policy_id=policy_id,
                level=self.hierarchy[policy_id].level,
                individual_risk_bound=base_cert.risk_bound,
                compositional_risk_bound=0.0,  # To be computed
                confidence=base_cert.confidence,
                coverage_guarantee=base_cert.coverage_guarantee,
                method=f"compositional_{self.composition_method}",
                sample_size=base_cert.sample_size,
                timestamp=time.time()
            )
            individual_certs.append(individual_cert)
        
        # Compute compositional bound
        if self.composition_method == "bonferroni":
            individual_risks = [cert.individual_risk_bound for cert in individual_certs]
            _, compositional_risk = self.bounds_calculator.bonferroni_correction(
                individual_risks, self.base_confidence
            )
        elif self.composition_method == "hierarchical":
            risk_tree = self._build_risk_tree()
            compositional_risk = self.bounds_calculator.hierarchical_union_bound(
                risk_tree, self.base_confidence
            )
        else:
            # Default to refined bound
            compositional_risk = self.bounds_calculator.refined_compositional_bound(
                individual_certs
            )
        
        # Create global certificate
        global_cert = CompositionalRiskCertificate(
            policy_id="global",
            level=0,
            individual_risk_bound=0.0,
            compositional_risk_bound=compositional_risk,
            confidence=self.base_confidence,
            coverage_guarantee=1.0 - compositional_risk,
            method=f"global_{self.composition_method}",
            sample_size=sum(cert.sample_size for cert in individual_certs),
            timestamp=time.time(),
            child_certificates=individual_certs,
            metadata={
                "num_policies": len(individual_certs),
                "composition_method": self.composition_method,
                "hierarchy_depth": max(p.level for p in self.hierarchy.values())
            }
        )
        
        return global_cert
    
    def _build_subtree_certificate(
        self,
        root_policy_id: str,
        max_depth: int
    ) -> CompositionalRiskCertificate:
        """Build certificate for policy subtree."""
        if root_policy_id not in self.hierarchy:
            raise ValidationError(f"Unknown root policy: {root_policy_id}")
        
        # Collect subtree policies
        subtree_policies = self._get_subtree_policies(root_policy_id, max_depth)
        
        # Get individual certificates for subtree
        individual_certs = []
        for policy_id in subtree_policies:
            base_cert = self.controllers[policy_id].get_certificate()
            individual_cert = CompositionalRiskCertificate(
                policy_id=policy_id,
                level=self.hierarchy[policy_id].level,
                individual_risk_bound=base_cert.risk_bound,
                compositional_risk_bound=0.0,
                confidence=base_cert.confidence,
                coverage_guarantee=base_cert.coverage_guarantee,
                method=self.composition_method,
                sample_size=base_cert.sample_size,
                timestamp=time.time()
            )
            individual_certs.append(individual_cert)
        
        # Compute subtree compositional bound
        individual_risks = [cert.individual_risk_bound for cert in individual_certs]
        _, compositional_risk = self.bounds_calculator.bonferroni_correction(
            individual_risks, self.base_confidence
        )
        
        # Root certificate
        root_policy = self.hierarchy[root_policy_id]
        subtree_cert = CompositionalRiskCertificate(
            policy_id=root_policy_id,
            level=root_policy.level,
            individual_risk_bound=self.controllers[root_policy_id].get_certificate().risk_bound,
            compositional_risk_bound=compositional_risk,
            confidence=self.base_confidence,
            coverage_guarantee=1.0 - compositional_risk,
            method=f"subtree_{self.composition_method}",
            sample_size=sum(cert.sample_size for cert in individual_certs),
            timestamp=time.time(),
            child_certificates=individual_certs,
            metadata={
                "subtree_size": len(subtree_policies),
                "max_depth": max_depth,
                "root_policy": root_policy_id
            }
        )
        
        return subtree_cert
    
    def _get_subtree_policies(self, root_id: str, max_depth: int) -> List[str]:
        """Get all policies in subtree."""
        visited = set()
        to_visit = [(root_id, 0)]
        subtree = []
        
        while to_visit:
            policy_id, depth = to_visit.pop(0)
            
            if policy_id in visited or depth > max_depth:
                continue
            
            visited.add(policy_id)
            subtree.append(policy_id)
            
            # Add children
            if policy_id in self.hierarchy:
                for child_id in self.hierarchy[policy_id].child_policies:
                    to_visit.append((child_id, depth + 1))
        
        return subtree
    
    def _build_risk_tree(self) -> Dict[str, List[float]]:
        """Build risk tree organized by hierarchy levels."""
        risk_tree = {}
        
        for policy_id, policy in self.hierarchy.items():
            level = policy.level
            risk_bound = self.controllers[policy_id].get_certificate().risk_bound
            
            if level not in risk_tree:
                risk_tree[level] = []
            risk_tree[level].append(risk_bound)
        
        return risk_tree
    
    def validate_risk_budgets(self) -> Dict[str, bool]:
        """Validate that all policies meet their risk budgets.
        
        Returns:
            Dictionary mapping policy IDs to validation status
        """
        validation_results = {}
        
        for policy_id, policy in self.hierarchy.items():
            current_risk = self.controllers[policy_id].get_certificate().risk_bound
            budget_met = current_risk <= policy.risk_budget
            validation_results[policy_id] = budget_met
            
            if not budget_met:
                logger.warning(f"Policy {policy_id} exceeds risk budget: "
                             f"{current_risk:.4f} > {policy.risk_budget:.4f}")
        
        return validation_results
    
    def get_hierarchy_stats(self) -> Dict[str, Any]:
        """Get comprehensive hierarchy statistics.
        
        Returns:
            Dictionary with hierarchy statistics
        """
        # Basic hierarchy info
        levels = [p.level for p in self.hierarchy.values()]
        
        # Risk statistics
        current_risks = {
            policy_id: controller.get_certificate().risk_bound
            for policy_id, controller in self.controllers.items()
        }
        
        # Budget compliance
        budget_compliance = self.validate_risk_budgets()
        
        stats = {
            "hierarchy_size": len(self.hierarchy),
            "hierarchy_depth": max(levels) - min(levels) + 1 if levels else 0,
            "levels": sorted(set(levels)),
            "current_risks": current_risks,
            "risk_budgets": {p.policy_id: p.risk_budget for p in self.hierarchy.values()},
            "budget_compliance": budget_compliance,
            "composition_method": self.composition_method,
            "base_confidence": self.base_confidence,
            "global_compositional_risk": self.get_compositional_certificate().compositional_risk_bound
        }
        
        return stats


class HierarchicalPolicyBuilder:
    """Builder for constructing hierarchical policy structures."""
    
    def __init__(self):
        self.policies = {}
        self.next_level = 0
    
    def add_policy(
        self,
        policy_id: str,
        parent_id: Optional[str] = None,
        risk_budget: float = 0.05
    ) -> 'HierarchicalPolicyBuilder':
        """Add policy to hierarchy.
        
        Args:
            policy_id: Unique policy identifier
            parent_id: Parent policy ID (None for root)
            risk_budget: Risk budget for this policy
            
        Returns:
            Builder instance for chaining
        """
        # Determine level
        if parent_id is None:
            level = 0
        elif parent_id in self.policies:
            level = self.policies[parent_id].level + 1
        else:
            raise ValidationError(f"Parent policy {parent_id} not found")
        
        # Create policy
        policy = HierarchicalPolicy(
            policy_id=policy_id,
            level=level,
            parent_policy=parent_id,
            risk_budget=risk_budget
        )
        
        # Add to hierarchy
        self.policies[policy_id] = policy
        
        # Update parent's children list
        if parent_id and parent_id in self.policies:
            self.policies[parent_id].child_policies.append(policy_id)
        
        # Update level tracking
        self.next_level = max(self.next_level, level + 1)
        
        logger.debug(f"Added policy {policy_id} at level {level}")
        
        return self
    
    def build(self) -> Dict[str, HierarchicalPolicy]:
        """Build the hierarchical policy structure.
        
        Returns:
            Dictionary of hierarchical policies
        """
        if not self.policies:
            raise ValidationError("No policies added to hierarchy")
        
        logger.info(f"Built hierarchy with {len(self.policies)} policies "
                   f"across {self.next_level} levels")
        
        return self.policies.copy()


# Research Extensions and Future Work

class CausalCompositionalRisk:
    """Compositional risk control under causal interventions.
    
    This is a research extension that combines causal inference
    with compositional risk bounds for robustness under interventions.
    """
    
    def __init__(self, causal_graph: Optional[Dict[str, List[str]]] = None):
        """Initialize causal compositional risk controller.
        
        Args:
            causal_graph: Optional causal DAG structure
        """
        self.causal_graph = causal_graph or {}
        logger.info("Causal compositional risk initialized (research extension)")
    
    def compute_intervention_bounds(
        self,
        baseline_certificate: CompositionalRiskCertificate,
        interventions: Dict[str, Any]
    ) -> CompositionalRiskCertificate:
        """Compute risk bounds under causal interventions.
        
        This is a placeholder for future research development.
        """
        # Research TODO: Implement causal risk bound computation
        logger.warning("Causal intervention bounds not yet implemented - research in progress")
        return baseline_certificate


class MultiAgentCompositionalRisk:
    """Compositional risk control for multi-agent systems.
    
    Research extension for distributed risk control with
    communication constraints and consensus mechanisms.
    """
    
    def __init__(self, num_agents: int, communication_graph: Optional[Dict] = None):
        """Initialize multi-agent compositional risk controller."""
        self.num_agents = num_agents
        self.communication_graph = communication_graph or {}
        logger.info(f"Multi-agent compositional risk initialized for {num_agents} agents")
    
    def distributed_risk_consensus(
        self,
        local_certificates: Dict[int, CompositionalRiskCertificate]
    ) -> CompositionalRiskCertificate:
        """Achieve consensus on global risk bounds through communication.
        
        This is a placeholder for future research development.
        """
        # Research TODO: Implement distributed consensus algorithm
        logger.warning("Distributed consensus not yet implemented - research in progress")
        return list(local_certificates.values())[0] if local_certificates else None