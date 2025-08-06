"""ConfoRL: Adaptive Conformal Risk Control for Reinforcement Learning

This package provides provable finite-sample safety guarantees for RL through
conformal prediction theory.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragonlabs.ai"

# Conditional imports to avoid dependency issues during development
try:
    from .algorithms import *
except ImportError:
    pass

try:
    from .risk import *
except ImportError:
    pass

try:
    from .deploy import *
except ImportError:
    pass

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