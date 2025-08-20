"""ConfoRL: Adaptive Conformal Risk Control for Reinforcement Learning

This package provides provable finite-sample safety guarantees for RL through
conformal prediction theory.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragonlabs.ai"

# Core imports that should always work
from .core.types import RiskCertificate, TrajectoryData

# Conditional imports to avoid dependency issues during development
_algorithm_imports = []
_risk_imports = []
_deploy_imports = []

try:
    from .risk.controllers import AdaptiveRiskController
    from .risk.measures import RiskMeasure, SafetyViolationRisk
    _risk_imports.extend(["AdaptiveRiskController", "RiskMeasure", "SafetyViolationRisk"])
except ImportError as e:
    print(f"Warning: Risk module imports failed: {e}")

try:
    from .algorithms.base import ConformalRLAgent
    _algorithm_imports.append("ConformalRLAgent")
except ImportError as e:
    print(f"Warning: Algorithm base import failed: {e}")

try:
    from .deploy.pipeline import SafeDeploymentPipeline
    _deploy_imports.append("SafeDeploymentPipeline")
except ImportError as e:
    print(f"Warning: Deploy module imports failed: {e}")

__all__ = [
    # Core types
    "RiskCertificate",
    "TrajectoryData",
] + _algorithm_imports + _risk_imports + _deploy_imports