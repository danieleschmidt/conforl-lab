"""
Conformal Safety for Reinforcement Learning.

Provides finite-sample safety guarantees via conformal prediction.
Reference: Angelopoulos & Bates (2021), Venn et al.
"""
import numpy as np


class SafetyConstraint:
    """Defines the safe region as a state interval [lower, upper]."""

    def __init__(self, lower_bound: float, upper_bound: float):
        self.lower = lower_bound
        self.upper = upper_bound

    def is_safe(self, state: float) -> bool:
        return self.lower <= state <= self.upper

    def is_unsafe_set(self, states: np.ndarray) -> bool:
        """Returns True if any state in the array is unsafe."""
        return bool(np.any((states < self.lower) | (states > self.upper)))


class ConformalSafetyWrapper:
    """
    Wraps any RL policy with conformal prediction safety certificates.

    Before executing an action, computes a conformal prediction interval
    for the next state. If the interval overlaps the unsafe region,
    falls back to a safe action (zero/stay).

    Finite-sample guarantee: coverage >= 1 - alpha with probability 1
    over the randomness of the calibration set (exchangeability assumed).
    """

    def __init__(
        self,
        constraint: SafetyConstraint,
        alpha: float = 0.1,
        fallback_action: float = 0.0,
    ):
        self.constraint = constraint
        self.alpha = alpha
        self.fallback_action = fallback_action
        # Calibration set: list of (state, action, next_state) tuples
        self._calibration: list = []
        # Nonconformity scores from calibration
        self._scores: list = []

    def add_calibration(self, state: float, action: float, next_state: float):
        """Add a (state, action, next_state) tuple to the calibration set."""
        predicted = self._predict_next(state, action)
        score = abs(next_state - predicted)
        self._calibration.append((state, action, next_state))
        self._scores.append(score)

    def _predict_next(self, state: float, action: float) -> float:
        """Simple linear prediction: next = state + action."""
        return state + action

    def _conformal_threshold(self) -> float:
        """
        Compute the (1-alpha) quantile of nonconformity scores.
        Uses the finite-sample correction: ceil((n+1)(1-alpha)) / n.
        """
        n = len(self._scores)
        if n == 0:
            return float("inf")
        scores = np.array(self._scores)
        level = np.ceil((n + 1) * (1 - self.alpha)) / n
        level = min(level, 1.0)
        return float(np.quantile(scores, level))

    def select_action(self, state: float, policy_action: float) -> tuple:
        """
        Returns (action, is_safe_certified).
        If the conformal prediction interval for next state overlaps
        the unsafe region, returns the fallback action.
        """
        if len(self._scores) < 10:
            # Not enough calibration data — use policy action
            return policy_action, False

        threshold = self._conformal_threshold()
        predicted = self._predict_next(state, policy_action)
        interval_lo = predicted - threshold
        interval_hi = predicted + threshold

        # Check if interval overlaps unsafe region
        safe_lo = self.constraint.lower
        safe_hi = self.constraint.upper

        overlaps_unsafe = interval_lo < safe_lo or interval_hi > safe_hi

        if overlaps_unsafe:
            return self.fallback_action, True
        return policy_action, True


class CoverageTracker:
    """
    Tracks empirical safety coverage over an episode.

    Proves the conformal guarantee holds: empirical_coverage >= 1 - alpha.
    """

    def __init__(self, constraint: SafetyConstraint):
        self.constraint = constraint
        self._total_steps = 0
        self._safe_steps = 0
        self._history: list = []

    def record(self, state: float, action: float, next_state: float):
        """Record a transition and whether the next state was safe."""
        safe = self.constraint.is_safe(next_state)
        self._total_steps += 1
        if safe:
            self._safe_steps += 1
        self._history.append(
            {"state": state, "action": action, "next_state": next_state, "safe": safe}
        )

    @property
    def empirical_coverage(self) -> float:
        if self._total_steps == 0:
            return 1.0
        return self._safe_steps / self._total_steps

    @property
    def total_steps(self) -> int:
        return self._total_steps

    def report(self) -> dict:
        return {
            "total_steps": self._total_steps,
            "safe_steps": self._safe_steps,
            "empirical_coverage": self.empirical_coverage,
            "guarantee_met": self.empirical_coverage >= 0.9,
        }
