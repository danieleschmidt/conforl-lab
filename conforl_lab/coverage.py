"""CoverageTracker: empirically verifies the conformal safety guarantee."""

import numpy as np


class CoverageTracker:
    """Records whether each environment step was safe and reports coverage.

    Tracks empirical safety coverage over an episode or evaluation run.
    The theoretical guarantee (Angelopoulos et al., 2021) states:

        P(next_state ∈ safe region) ≥ 1 − α

    with finite-sample validity for any exchangeable calibration set.
    This class verifies that the guarantee holds empirically.

    Parameters
    ----------
    alpha : float
        The miscoverage level used by the ConformalSafetyWrapper, in (0, 1).
        The expected lower bound on coverage is 1 − alpha.
    """

    def __init__(self, alpha: float = 0.1):
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha
        self._safe_steps: int = 0
        self._total_steps: int = 0

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, next_state: float, constraint) -> bool:
        """Record whether a transition landed in the safe region.

        Parameters
        ----------
        next_state : float
            The state reached after taking an action.
        constraint : SafetyConstraint
            Defines what "safe" means.

        Returns
        -------
        bool
            True if next_state is safe, False otherwise.
        """
        safe = constraint.is_safe(next_state)
        self._total_steps += 1
        if safe:
            self._safe_steps += 1
        return safe

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @property
    def empirical_coverage(self) -> float:
        """Fraction of steps where the next state was safe.

        Returns NaN if no steps have been recorded yet.
        """
        if self._total_steps == 0:
            return float("nan")
        return self._safe_steps / self._total_steps

    @property
    def theoretical_lower_bound(self) -> float:
        """Expected minimum coverage: 1 − alpha."""
        return 1.0 - self.alpha

    @property
    def guarantee_holds(self) -> bool:
        """True if empirical coverage meets or exceeds the theoretical bound."""
        if self._total_steps == 0:
            return False
        return self.empirical_coverage >= self.theoretical_lower_bound

    def reset(self) -> None:
        """Reset all recorded steps."""
        self._safe_steps = 0
        self._total_steps = 0

    def summary(self) -> str:
        """Return a human-readable summary of the coverage results."""
        cov = self.empirical_coverage
        bound = self.theoretical_lower_bound
        status = "✓ GUARANTEE HOLDS" if self.guarantee_holds else "✗ GUARANTEE VIOLATED"
        return (
            f"Coverage Report\n"
            f"  Steps recorded:       {self._total_steps}\n"
            f"  Safe steps:           {self._safe_steps}\n"
            f"  Empirical coverage:   {cov:.4f} ({cov*100:.1f}%)\n"
            f"  Theoretical bound:    {bound:.4f} ({bound*100:.1f}%)\n"
            f"  alpha:                {self.alpha}\n"
            f"  Status:               {status}"
        )

    def __repr__(self) -> str:
        return (
            f"CoverageTracker(alpha={self.alpha}, "
            f"steps={self._total_steps}, "
            f"coverage={self.empirical_coverage:.4f})"
        )
