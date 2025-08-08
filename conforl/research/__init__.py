"""ConfoRL Research Extensions.

This package contains cutting-edge research implementations that extend
the core ConfoRL framework with novel algorithmic contributions.

Research Areas:
- Compositional Risk Control for hierarchical RL
- Causal Conformal RL for interventional robustness  
- Multi-Agent Conformal Risk for distributed systems
- Adversarial Robustness with conformal guarantees

These implementations represent active areas of research and may contain
experimental features that are subject to change.

Author: ConfoRL Research Team
License: Apache 2.0
"""

from .compositional import (
    CompositionalRiskController,
    CompositionalRiskCertificate, 
    CompositionalRiskBounds,
    HierarchicalPolicy,
    HierarchicalPolicyBuilder,
    CausalCompositionalRisk,
    MultiAgentCompositionalRisk
)

__all__ = [
    "CompositionalRiskController",
    "CompositionalRiskCertificate",
    "CompositionalRiskBounds", 
    "HierarchicalPolicy",
    "HierarchicalPolicyBuilder",
    "CausalCompositionalRisk",
    "MultiAgentCompositionalRisk"
]

__version__ = "0.1.0-research"