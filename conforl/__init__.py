"""ConfoRL: Adaptive Conformal Risk Control for Reinforcement Learning

This package provides provable finite-sample safety guarantees for RL through
conformal prediction theory.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragonlabs.ai"

from .algorithms import *
from .risk import *
from .deploy import *

__all__ = [
    # Core algorithms
    "ConformaSAC",
    "ConformaPPO", 
    "ConformaTD3",
    "ConformaCQL",
    
    # Risk controllers
    "AdaptiveRiskController",
    "MultiRiskController",
    "OnlineRiskAdaptation",
    
    # Risk measures
    "RiskMeasure",
    "SafetyViolationRisk",
    "PerformanceRisk",
    
    # Deployment
    "SafeDeploymentPipeline",
]