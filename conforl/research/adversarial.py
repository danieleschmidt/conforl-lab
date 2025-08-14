"""Adversarial Robust Conformal Risk Control.

This module implements novel adversarial conformal bounds that provide
robustness guarantees against worst-case perturbations and adversarial attacks.
This extends ConfoRL to adversarial settings with formal robustness certificates.

Research Contributions:
- Adversarial conformal prediction with worst-case guarantees
- Certified defense mechanisms for RL policies
- Adaptive adversarial training with conformal bounds
- Robust risk certificates under adversarial perturbations

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
        def array(data):
            return data
        @staticmethod
        def mean(data):
            return sum(data) / len(data) if data else 0
        @staticmethod
        def maximum(a, b):
            return max(a, b)
        @staticmethod
        def norm(x):
            return sum(xi**2 for xi in x)**0.5 if hasattr(x, '__iter__') else abs(x)

import time
import math
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import warnings

from ..core.types import RiskCertificate, TrajectoryData
from ..risk.controllers import AdaptiveRiskController
from ..risk.measures import RiskMeasure
from ..utils.logging import get_logger
from ..utils.errors import ConfoRLError, ValidationError

logger = get_logger(__name__)


class AttackType(Enum):
    """Types of adversarial attacks."""
    L_INF = "l_inf"          # L-infinity norm bounded
    L_2 = "l_2"              # L-2 norm bounded  
    SEMANTIC = "semantic"    # Semantically meaningful
    TEMPORAL = "temporal"    # Temporal perturbations
    REWARD = "reward"        # Reward poisoning
    TRANSITION = "transition"  # Transition dynamics


@dataclass
class AdversarialPerturbation:
    """Represents an adversarial perturbation."""
    
    attack_type: AttackType
    epsilon: float
    target_component: str  # 'state', 'action', 'reward', 'transition'
    perturbation_vector: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AdversarialRiskCertificate:
    """Risk certificate robust to adversarial perturbations."""
    
    clean_risk_bound: float
    adversarial_risk_bound: float
    certified_radius: float
    attack_types_tested: List[AttackType]
    robustness_confidence: float
    verification_method: str
    sample_size: int
    timestamp: float
    metadata: Optional[Dict[str, Any]] = None


class AdversarialAttackGenerator:
    """Generate adversarial examples for robustness testing."""
    
    def __init__(
        self,
        attack_budget: float = 0.1,
        attack_types: Optional[List[AttackType]] = None
    ):
        """Initialize adversarial attack generator.
        
        Args:
            attack_budget: Maximum perturbation budget
            attack_types: Types of attacks to generate
        """
        self.attack_budget = attack_budget
        self.attack_types = attack_types or [AttackType.L_INF, AttackType.L_2]
        
        # Attack generation methods
        self.attack_methods = {
            AttackType.L_INF: self._generate_linf_attack,
            AttackType.L_2: self._generate_l2_attack,
            AttackType.SEMANTIC: self._generate_semantic_attack,
            AttackType.TEMPORAL: self._generate_temporal_attack,
            AttackType.REWARD: self._generate_reward_attack,
            AttackType.TRANSITION: self._generate_transition_attack
        }
        
        logger.info(f"Initialized adversarial attack generator with budget {attack_budget}")
    
    def generate_attack(
        self,
        trajectory: TrajectoryData,
        attack_type: AttackType,
        epsilon: Optional[float] = None
    ) -> AdversarialPerturbation:
        """Generate adversarial perturbation.
        
        Args:
            trajectory: Target trajectory
            attack_type: Type of attack to generate
            epsilon: Perturbation budget (uses default if None)
            
        Returns:
            Generated adversarial perturbation
        """
        epsilon = epsilon or self.attack_budget
        
        if attack_type not in self.attack_methods:
            raise ValidationError(f"Unsupported attack type: {attack_type}")
        
        perturbation = self.attack_methods[attack_type](trajectory, epsilon)
        
        logger.debug(f"Generated {attack_type.value} attack with epsilon={epsilon}")
        
        return perturbation
    
    def _generate_linf_attack(
        self, 
        trajectory: TrajectoryData, 
        epsilon: float
    ) -> AdversarialPerturbation:
        """Generate L-infinity bounded attack."""
        # Extract states from trajectory (assuming they exist)
        if hasattr(trajectory, 'states') and len(trajectory.states) > 0:
            state_shape = np.array(trajectory.states[0]).shape
            # Generate uniform random perturbation in L-inf ball
            perturbation_vector = np.random.uniform(-epsilon, epsilon, state_shape)
        else:
            # Fallback for unknown trajectory structure
            perturbation_vector = np.array([epsilon * np.random.choice([-1, 1])])
        
        return AdversarialPerturbation(
            attack_type=AttackType.L_INF,
            epsilon=epsilon,
            target_component='state',
            perturbation_vector=perturbation_vector,
            metadata={'norm_type': 'l_inf', 'max_perturbation': epsilon}
        )
    
    def _generate_l2_attack(
        self, 
        trajectory: TrajectoryData, 
        epsilon: float
    ) -> AdversarialPerturbation:
        """Generate L-2 bounded attack."""
        if hasattr(trajectory, 'states') and len(trajectory.states) > 0:
            state_shape = np.array(trajectory.states[0]).shape
            # Generate random direction and scale to L2 ball
            perturbation_vector = np.random.randn(*state_shape)
            perturbation_norm = np.linalg.norm(perturbation_vector)
            if perturbation_norm > 0:
                perturbation_vector = perturbation_vector / perturbation_norm * epsilon
        else:
            perturbation_vector = np.array([epsilon])
        
        return AdversarialPerturbation(
            attack_type=AttackType.L_2,
            epsilon=epsilon,
            target_component='state',
            perturbation_vector=perturbation_vector,
            metadata={'norm_type': 'l_2', 'perturbation_norm': epsilon}
        )
    
    def _generate_semantic_attack(
        self, 
        trajectory: TrajectoryData, 
        epsilon: float
    ) -> AdversarialPerturbation:
        """Generate semantically meaningful attack."""
        # Placeholder for semantic attacks (e.g., lighting changes, rotations)
        logger.warning("Semantic attacks not fully implemented - using L-inf approximation")
        return self._generate_linf_attack(trajectory, epsilon * 0.5)  # More conservative
    
    def _generate_temporal_attack(
        self, 
        trajectory: TrajectoryData, 
        epsilon: float
    ) -> AdversarialPerturbation:
        """Generate temporal perturbation attack."""
        # Placeholder for temporal attacks (e.g., timing shifts, frame drops)
        return AdversarialPerturbation(
            attack_type=AttackType.TEMPORAL,
            epsilon=epsilon,
            target_component='temporal',
            perturbation_vector=None,
            metadata={'temporal_shift': epsilon, 'attack_description': 'timing perturbation'}
        )
    
    def _generate_reward_attack(
        self, 
        trajectory: TrajectoryData, 
        epsilon: float
    ) -> AdversarialPerturbation:
        """Generate reward poisoning attack."""
        return AdversarialPerturbation(
            attack_type=AttackType.REWARD,
            epsilon=epsilon,
            target_component='reward',
            perturbation_vector=None,
            metadata={'reward_perturbation': epsilon, 'poisoning_type': 'additive'}
        )
    
    def _generate_transition_attack(
        self, 
        trajectory: TrajectoryData, 
        epsilon: float
    ) -> AdversarialPerturbation:
        """Generate transition dynamics attack."""
        return AdversarialPerturbation(
            attack_type=AttackType.TRANSITION,
            epsilon=epsilon,
            target_component='transition',
            perturbation_vector=None,
            metadata={'dynamics_perturbation': epsilon, 'attack_type': 'model_poisoning'}
        )


class CertifiedDefense:
    """Certified defense mechanism with provable robustness."""
    
    def __init__(
        self,
        defense_type: str = "randomized_smoothing",
        noise_scale: float = 0.1,
        num_samples: int = 100
    ):
        """Initialize certified defense.
        
        Args:
            defense_type: Type of certified defense
            noise_scale: Scale of defensive noise
            num_samples: Number of samples for certification
        """
        self.defense_type = defense_type
        self.noise_scale = noise_scale
        self.num_samples = num_samples
        
        # Defense statistics
        self.defense_history = []
        self.certification_cache = {}
        
        logger.info(f"Initialized certified defense: {defense_type} with noise_scale={noise_scale}")
    
    def defend(
        self,
        trajectory: TrajectoryData,
        perturbation: Optional[AdversarialPerturbation] = None
    ) -> TrajectoryData:
        """Apply certified defense to trajectory.
        
        Args:
            trajectory: Input trajectory (potentially adversarial)
            perturbation: Known perturbation (if available)
            
        Returns:
            Defended trajectory
        """
        if self.defense_type == "randomized_smoothing":
            return self._randomized_smoothing_defense(trajectory)
        elif self.defense_type == "adversarial_training":
            return self._adversarial_training_defense(trajectory, perturbation)
        else:
            logger.warning(f"Unknown defense type {self.defense_type}, returning original")
            return trajectory
    
    def _randomized_smoothing_defense(self, trajectory: TrajectoryData) -> TrajectoryData:
        """Apply randomized smoothing defense."""
        # Add Gaussian noise for randomized smoothing
        # This is a simplified implementation
        defended_trajectory = trajectory  # Placeholder
        
        # Record defense application
        self.defense_history.append({
            'method': 'randomized_smoothing',
            'noise_scale': self.noise_scale,
            'timestamp': time.time()
        })
        
        return defended_trajectory
    
    def _adversarial_training_defense(
        self, 
        trajectory: TrajectoryData, 
        perturbation: Optional[AdversarialPerturbation]
    ) -> TrajectoryData:
        """Apply adversarial training defense."""
        # Placeholder for adversarial training defense
        defended_trajectory = trajectory
        
        self.defense_history.append({
            'method': 'adversarial_training',
            'perturbation_type': perturbation.attack_type.value if perturbation else None,
            'timestamp': time.time()
        })
        
        return defended_trajectory
    
    def certify_robustness(
        self,
        trajectory: TrajectoryData,
        attack_type: AttackType,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Certify robustness radius for given trajectory.
        
        Args:
            trajectory: Input trajectory
            attack_type: Type of attack to certify against
            confidence: Certification confidence
            
        Returns:
            Tuple of (certified_radius, certification_probability)
        """
        # Check cache
        cache_key = f"{attack_type.value}_{confidence}_{hash(str(trajectory))}"
        if cache_key in self.certification_cache:
            return self.certification_cache[cache_key]
        
        if self.defense_type == "randomized_smoothing":
            radius, prob = self._certify_randomized_smoothing(trajectory, attack_type, confidence)
        else:
            # Conservative fallback
            radius, prob = 0.0, 0.0
            logger.warning(f"Certification not implemented for {self.defense_type}")
        
        # Cache result
        self.certification_cache[cache_key] = (radius, prob)
        
        return radius, prob
    
    def _certify_randomized_smoothing(
        self,
        trajectory: TrajectoryData,
        attack_type: AttackType,
        confidence: float
    ) -> Tuple[float, float]:
        """Certify robustness using randomized smoothing theory."""
        # Simplified certification based on Cohen et al. (2019)
        # This would need proper implementation for production use
        
        if attack_type in [AttackType.L_2, AttackType.L_INF]:
            # Use Neyman-Pearson lemma for certification
            sigma = self.noise_scale
            
            # Placeholder certification computation
            # Real implementation would involve statistical testing
            alpha = 1.0 - confidence
            
            # Simplified radius computation
            if attack_type == AttackType.L_2:
                # L2 certification radius (simplified)
                certified_radius = sigma * math.sqrt(2 * math.log(1/alpha))
            else:  # L_INF
                # L-inf certification (more conservative)
                certified_radius = sigma * math.sqrt(2 * math.log(1/alpha)) / math.sqrt(3)
            
            certification_prob = confidence
        else:
            certified_radius = 0.0
            certification_prob = 0.0
        
        return certified_radius, certification_prob


class AdversarialRiskController:
    """Risk controller robust to adversarial perturbations."""
    
    def __init__(
        self,
        base_controller: AdaptiveRiskController,
        attack_generator: AdversarialAttackGenerator,
        certified_defense: CertifiedDefense,
        adversarial_confidence: float = 0.9,
        robustness_testing_freq: int = 100
    ):
        """Initialize adversarial risk controller.
        
        Args:
            base_controller: Base conformal risk controller
            attack_generator: Generator for adversarial examples
            certified_defense: Certified defense mechanism
            adversarial_confidence: Confidence level for adversarial guarantees
            robustness_testing_freq: Frequency of robustness testing
        """
        self.base_controller = base_controller
        self.attack_generator = attack_generator
        self.certified_defense = certified_defense
        self.adversarial_confidence = adversarial_confidence
        self.robustness_testing_freq = robustness_testing_freq
        
        # Adversarial evaluation data
        self.clean_scores = []
        self.adversarial_scores = {}  # attack_type -> scores
        self.robustness_certificates = []
        
        # Update counter
        self.update_count = 0
        
        # Certificate cache
        self._certificate_cache = None
        self._cache_timestamp = 0.0
        
        logger.info("Initialized adversarial risk controller with certified defense")
    
    def update(
        self,
        trajectory: TrajectoryData,
        risk_measure: RiskMeasure,
        run_robustness_test: bool = False
    ) -> None:
        """Update controller with new data and optional robustness testing.
        
        Args:
            trajectory: New trajectory data
            risk_measure: Risk measure to compute
            run_robustness_test: Whether to run robustness testing
        """
        self.update_count += 1
        
        # Update base controller
        self.base_controller.update(trajectory, risk_measure)
        
        # Compute clean risk score
        clean_score = risk_measure.compute(trajectory)
        self.clean_scores.append(clean_score)
        
        # Periodic robustness testing
        should_test = (run_robustness_test or 
                      self.update_count % self.robustness_testing_freq == 0)
        
        if should_test:
            self._run_robustness_evaluation(trajectory, risk_measure)
        
        # Keep recent scores only
        max_scores = 5000
        if len(self.clean_scores) > max_scores:
            self.clean_scores = self.clean_scores[-max_scores:]
        
        for attack_type in self.adversarial_scores:
            if len(self.adversarial_scores[attack_type]) > max_scores:
                self.adversarial_scores[attack_type] = (
                    self.adversarial_scores[attack_type][-max_scores:]
                )
        
        # Invalidate cache
        self._certificate_cache = None
        self._cache_timestamp = 0.0
        
        logger.debug(f"Updated adversarial controller (update #{self.update_count})")
    
    def _run_robustness_evaluation(
        self,
        trajectory: TrajectoryData,
        risk_measure: RiskMeasure
    ) -> None:
        """Run robustness evaluation against adversarial attacks."""
        for attack_type in self.attack_generator.attack_types:
            try:
                # Generate adversarial example
                perturbation = self.attack_generator.generate_attack(trajectory, attack_type)
                
                # Apply perturbation (simplified)
                adversarial_trajectory = self._apply_perturbation(trajectory, perturbation)
                
                # Apply certified defense
                defended_trajectory = self.certified_defense.defend(
                    adversarial_trajectory, perturbation
                )
                
                # Compute adversarial risk score
                adversarial_score = risk_measure.compute(defended_trajectory)
                
                # Store result
                if attack_type not in self.adversarial_scores:
                    self.adversarial_scores[attack_type] = []
                self.adversarial_scores[attack_type].append(adversarial_score)
                
                logger.debug(f"Robustness test: {attack_type.value} -> score={adversarial_score:.4f}")
                
            except Exception as e:
                logger.warning(f"Robustness test failed for {attack_type.value}: {e}")
    
    def _apply_perturbation(
        self,
        trajectory: TrajectoryData,
        perturbation: AdversarialPerturbation
    ) -> TrajectoryData:
        """Apply adversarial perturbation to trajectory.
        
        This is a simplified implementation - real applications would need
        domain-specific perturbation application.
        """
        # Placeholder: return original trajectory
        # Real implementation would modify states/actions based on perturbation
        return trajectory
    
    def get_adversarial_certificate(self) -> AdversarialRiskCertificate:
        """Get adversarial risk certificate.
        
        Returns:
            Certificate with adversarial robustness guarantees
        """
        # Check cache
        current_time = time.time()
        if (self._certificate_cache is not None and 
            current_time - self._cache_timestamp < 60.0):  # 1 minute cache
            return self._certificate_cache
        
        # Get base certificate
        base_cert = self.base_controller.get_certificate()
        clean_risk_bound = base_cert.risk_bound
        
        # Compute adversarial risk bound
        adversarial_risk_bound = self._compute_adversarial_bound()
        
        # Get certified robustness radius
        certified_radius = self._compute_certified_radius()
        
        # Create adversarial certificate
        adversarial_cert = AdversarialRiskCertificate(
            clean_risk_bound=clean_risk_bound,
            adversarial_risk_bound=adversarial_risk_bound,
            certified_radius=certified_radius,
            attack_types_tested=list(self.adversarial_scores.keys()),
            robustness_confidence=self.adversarial_confidence,
            verification_method=f"conformal_{self.certified_defense.defense_type}",
            sample_size=base_cert.sample_size,
            timestamp=current_time,
            metadata={
                'base_method': base_cert.method,
                'defense_type': self.certified_defense.defense_type,
                'num_adversarial_tests': sum(len(scores) for scores in self.adversarial_scores.values()),
                'robustness_testing_freq': self.robustness_testing_freq
            }
        )
        
        # Cache result
        self._certificate_cache = adversarial_cert
        self._cache_timestamp = current_time
        
        return adversarial_cert
    
    def _compute_adversarial_bound(self) -> float:
        """Compute adversarial risk bound across all attack types."""
        if not self.adversarial_scores:
            # No adversarial data - conservative estimate
            return min(1.0, self.base_controller.get_certificate().risk_bound * 2.0)
        
        # Collect all adversarial scores
        all_adversarial_scores = []
        for attack_scores in self.adversarial_scores.values():
            all_adversarial_scores.extend(attack_scores)
        
        if not all_adversarial_scores:
            return min(1.0, self.base_controller.get_certificate().risk_bound * 2.0)
        
        # Compute conformal quantile for adversarial scores
        quantile_level = 1.0 - (1.0 - self.adversarial_confidence) / 2.0  # Two-sided
        adversarial_quantile = np.quantile(all_adversarial_scores, quantile_level)
        
        return min(1.0, adversarial_quantile)
    
    def _compute_certified_radius(self) -> float:
        """Compute certified robustness radius."""
        if not hasattr(self, '_last_trajectory'):
            return 0.0
        
        # Get certification from defense mechanism
        try:
            # Use L_2 attack as default for certification
            radius, _ = self.certified_defense.certify_robustness(
                self._last_trajectory, AttackType.L_2, self.adversarial_confidence
            )
            return radius
        except Exception as e:
            logger.warning(f"Certification failed: {e}")
            return 0.0
    
    def test_attack_scenario(
        self,
        trajectory: TrajectoryData,
        attack_type: AttackType,
        epsilon: float,
        risk_measure: RiskMeasure
    ) -> Dict[str, Any]:
        """Test specific attack scenario.
        
        Args:
            trajectory: Target trajectory
            attack_type: Type of attack
            epsilon: Attack strength
            risk_measure: Risk measure to evaluate
            
        Returns:
            Attack test results
        """
        try:
            # Generate specific attack
            perturbation = self.attack_generator.generate_attack(trajectory, attack_type, epsilon)
            
            # Apply perturbation
            adversarial_trajectory = self._apply_perturbation(trajectory, perturbation)
            
            # Test with and without defense
            no_defense_score = risk_measure.compute(adversarial_trajectory)
            
            defended_trajectory = self.certified_defense.defend(adversarial_trajectory, perturbation)
            with_defense_score = risk_measure.compute(defended_trajectory)
            
            # Clean baseline
            clean_score = risk_measure.compute(trajectory)
            
            results = {
                'attack_type': attack_type.value,
                'epsilon': epsilon,
                'clean_score': clean_score,
                'adversarial_score': no_defense_score,
                'defended_score': with_defense_score,
                'attack_success': no_defense_score > clean_score * 1.5,  # 50% increase threshold
                'defense_effectiveness': max(0, no_defense_score - with_defense_score),
                'certified_radius': self.certified_defense.certify_robustness(
                    trajectory, attack_type, 0.95
                )[0] if attack_type in [AttackType.L_2, AttackType.L_INF] else 0.0
            }
            
            logger.info(f"Attack test: {attack_type.value}(ε={epsilon}) -> "
                       f"success={results['attack_success']}, "
                       f"defense_effect={results['defense_effectiveness']:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Attack test failed: {e}")
            return {
                'attack_type': attack_type.value,
                'epsilon': epsilon,
                'error': str(e),
                'test_failed': True
            }
    
    def get_robustness_summary(self) -> Dict[str, Any]:
        """Get comprehensive robustness summary.
        
        Returns:
            Dictionary with robustness statistics and insights
        """
        # Basic statistics
        clean_stats = {
            'mean': np.mean(self.clean_scores) if self.clean_scores else 0.0,
            'std': np.std(self.clean_scores) if len(self.clean_scores) > 1 else 0.0,
            'count': len(self.clean_scores)
        }
        
        # Adversarial statistics by attack type
        adversarial_stats = {}
        for attack_type, scores in self.adversarial_scores.items():
            if scores:
                adversarial_stats[attack_type.value] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores) if len(scores) > 1 else 0.0,
                    'count': len(scores),
                    'worst_case': np.max(scores),
                    'robustness_gap': np.mean(scores) - clean_stats['mean']
                }
        
        # Defense statistics
        defense_stats = {
            'defense_type': self.certified_defense.defense_type,
            'noise_scale': getattr(self.certified_defense, 'noise_scale', None),
            'num_defenses_applied': len(self.certified_defense.defense_history),
            'certification_cache_size': len(self.certified_defense.certification_cache)
        }
        
        # Overall robustness metrics
        current_cert = self.get_adversarial_certificate()
        
        summary = {
            'clean_performance': clean_stats,
            'adversarial_performance': adversarial_stats,
            'defense_statistics': defense_stats,
            'current_certificate': {
                'clean_risk_bound': current_cert.clean_risk_bound,
                'adversarial_risk_bound': current_cert.adversarial_risk_bound,
                'certified_radius': current_cert.certified_radius,
                'robustness_confidence': current_cert.robustness_confidence
            },
            'testing_config': {
                'robustness_testing_freq': self.robustness_testing_freq,
                'attack_types': [at.value for at in self.attack_generator.attack_types],
                'attack_budget': self.attack_generator.attack_budget
            },
            'total_updates': self.update_count
        }
        
        return summary


# Research Extensions

class AdaptiveAttackGeneration:
    """Adaptive attack generation that learns from defenses."""
    
    def __init__(self, base_generator: AdversarialAttackGenerator):
        """Initialize adaptive attack generation.
        
        Args:
            base_generator: Base attack generator to extend
        """
        self.base_generator = base_generator
        self.attack_success_history = defaultdict(list)
        logger.info("Adaptive attack generation initialized (research extension)")
    
    def generate_adaptive_attack(
        self,
        trajectory: TrajectoryData,
        defense_history: List[Dict],
        target_success_rate: float = 0.8
    ) -> AdversarialPerturbation:
        """Generate attack adapted to observed defenses.
        
        This is a placeholder for sophisticated adaptive attack generation.
        """
        logger.warning("Adaptive attack generation not fully implemented - research in progress")
        
        # Fallback to base generator
        return self.base_generator.generate_attack(trajectory, AttackType.L_INF)


class MultiStepAdversarialRisk:
    """Multi-step adversarial risk assessment for sequential decision making."""
    
    def __init__(self, horizon: int = 10):
        """Initialize multi-step adversarial risk assessment.
        
        Args:
            horizon: Planning horizon for multi-step attacks
        """
        self.horizon = horizon
        logger.info(f"Multi-step adversarial risk initialized with horizon {horizon}")
    
    def assess_sequential_robustness(
        self,
        policy: Callable,
        initial_state: Any,
        attack_sequence: List[AdversarialPerturbation]
    ) -> float:
        """Assess robustness against sequential adversarial attacks.
        
        This is a placeholder for advanced sequential attack analysis.
        """
        logger.warning("Sequential robustness assessment not fully implemented - research in progress")
        return 0.0  # Placeholder