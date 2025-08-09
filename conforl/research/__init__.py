"""ConfoRL Research Extensions.

This package contains cutting-edge research implementations that extend
the core ConfoRL framework with novel algorithmic contributions.

Research Areas Implemented:
- Compositional Risk Control for hierarchical RL
- Causal Conformal RL for interventional robustness  
- Multi-Agent Conformal Risk for distributed systems
- Adversarial Robustness with conformal guarantees

These implementations represent state-of-the-art research and push the
boundaries of safe reinforcement learning with formal guarantees.

Author: ConfoRL Research Team
License: Apache 2.0
"""

# Compositional Risk Control
from .compositional import (
    CompositionalRiskController,
    CompositionalRiskCertificate, 
    CompositionalRiskBounds,
    HierarchicalPolicy,
    HierarchicalPolicyBuilder,
    CausalCompositionalRisk,
    MultiAgentCompositionalRisk
)

# Causal Conformal Risk Control
from .causal import (
    CausalGraph,
    CausalIntervention,
    CausalRiskCertificate,
    CausalShiftDetector,
    CausalConformPredictor,
    CausalRiskController,
    CounterfactualRiskAssessment,
    CausalGraphLearner
)

# Adversarial Robust Conformal
from .adversarial import (
    AttackType,
    AdversarialPerturbation,
    AdversarialRiskCertificate,
    AdversarialAttackGenerator,
    CertifiedDefense,
    AdversarialRiskController,
    AdaptiveAttackGeneration,
    MultiStepAdversarialRisk
)

# Multi-Agent Risk Control
from .multi_agent import (
    CommunicationTopology,
    AgentInfo,
    CommunicationMessage,
    MultiAgentRiskCertificate,
    CommunicationNetwork,
    ConsensusAlgorithm,
    AverageConsensus,
    ByzantineRobustConsensus,
    MultiAgentRiskController,
    FederatedRiskLearning,
    ScalableConsensus
)

__all__ = [
    # Compositional
    "CompositionalRiskController",
    "CompositionalRiskCertificate",
    "CompositionalRiskBounds", 
    "HierarchicalPolicy",
    "HierarchicalPolicyBuilder",
    "CausalCompositionalRisk",
    "MultiAgentCompositionalRisk",
    
    # Causal
    "CausalGraph",
    "CausalIntervention",
    "CausalRiskCertificate",
    "CausalShiftDetector",
    "CausalConformPredictor",
    "CausalRiskController",
    "CounterfactualRiskAssessment",
    "CausalGraphLearner",
    
    # Adversarial
    "AttackType",
    "AdversarialPerturbation",
    "AdversarialRiskCertificate",
    "AdversarialAttackGenerator",
    "CertifiedDefense",
    "AdversarialRiskController",
    "AdaptiveAttackGeneration",
    "MultiStepAdversarialRisk",
    
    # Multi-Agent
    "CommunicationTopology",
    "AgentInfo",
    "CommunicationMessage",
    "MultiAgentRiskCertificate",
    "CommunicationNetwork",
    "ConsensusAlgorithm",
    "AverageConsensus",
    "ByzantineRobustConsensus",
    "MultiAgentRiskController",
    "FederatedRiskLearning",
    "ScalableConsensus"
]

__version__ = "0.1.0-research"