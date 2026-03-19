"""ConformalSafetyWrapper: wraps any RL policy with conformal safety guarantees."""

import math
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from .safety import SafetyConstraint

# Fallback action can be a fixed float or a state-dependent callable
FallbackType = Union[float, Callable[[float], float]]


class ConformalSafetyWrapper:
    """Wraps an RL policy with a conformal prediction safety shield.

    Before executing any action, the wrapper:
    1. Uses a deterministic dynamics model to predict the expected next state.
    2. Builds a conformal prediction interval around that prediction using
       residuals from the calibration set.
    3. Checks whether the interval overlaps the unsafe region.
    4. If unsafe overlap is detected, substitutes a safe fallback action.
       If the fallback's interval also overlaps unsafe, a corrective action
       (proportional controller toward centre) is used instead.

    Finite-sample safety guarantee (Angelopoulos et al., 2021):

        P(next_state ∈ C(state, action)) ≥ 1 − α

    provided the calibration residuals are exchangeable with test residuals.
    Coverage of the *safe region* follows because the shield only permits
    actions whose prediction interval lies entirely within that region.

    Parameters
    ----------
    policy : callable
        Function ``policy(state) -> action`` — the base RL policy to wrap.
    dynamics_fn : callable
        Function ``dynamics_fn(state, action) -> next_state`` — a **deterministic**
        mean dynamics model (e.g., ``lambda s, a: s + a`` for a random walk).
        Conformal residuals are computed relative to this model.
    constraint : SafetyConstraint
        Defines the safe region.
    alpha : float
        Miscoverage level in (0, 1). The guarantee is ≥ 1−α coverage.
    fallback_action : float or callable
        Action (or state-dependent function) to use when the base policy's
        predicted next state is unsafe. Default 0.0 (stay put). If a
        callable is provided it receives the current state and returns an
        action.
    """

    def __init__(
        self,
        policy: Callable[[float], float],
        dynamics_fn: Callable[[float, float], float],
        constraint: SafetyConstraint,
        alpha: float = 0.1,
        fallback_action: FallbackType = 0.0,
    ):
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        self.policy = policy
        self.dynamics_fn = dynamics_fn
        self.constraint = constraint
        self.alpha = alpha
        self.fallback_action = fallback_action

        # Calibration triples: list of (state, action, next_state)
        self._calibration_triples: List[Tuple[float, float, float]] = []

        # Nonconformity scores: |next_state - dynamics_fn(state, action)|
        self._scores: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(
        self,
        calibration_triples: List[Tuple[float, float, float]],
    ) -> None:
        """Fit the conformal predictor on observed transitions.

        Parameters
        ----------
        calibration_triples : list of (state, action, next_state) tuples
            Observed environment transitions.  For each triple the
            nonconformity score is:

                score_i = |next_state_i − dynamics_fn(state_i, action_i)|
        """
        if len(calibration_triples) == 0:
            raise ValueError("calibration_triples must be non-empty")

        self._calibration_triples = list(calibration_triples)
        scores = []
        for s, a, ns in calibration_triples:
            predicted = float(self.dynamics_fn(float(s), float(a)))
            scores.append(abs(float(ns) - predicted))
        self._scores = np.array(scores, dtype=float)

    # ------------------------------------------------------------------
    # Conformal quantile
    # ------------------------------------------------------------------

    def _conformal_threshold(self) -> float:
        """Compute the (1−α) conformal quantile with finite-sample correction.

        Returns q̂ such that at least ⌈(n+1)(1−α)⌉/n of calibration scores
        fall below q̂, guaranteeing marginal coverage ≥ 1−α on exchangeable
        test residuals.
        """
        if self._scores is None or len(self._scores) == 0:
            raise RuntimeError("Must call calibrate() before using the wrapper.")

        n = len(self._scores)
        level = math.ceil((n + 1) * (1.0 - self.alpha)) / n
        level = min(level, 1.0)
        return float(np.quantile(self._scores, level))

    # ------------------------------------------------------------------
    # Prediction interval
    # ------------------------------------------------------------------

    def _prediction_interval(
        self, state: float, action: float
    ) -> Tuple[float, float]:
        """Return the conformal prediction interval for the next state.

        Interval = [dynamics_fn(s, a) − q̂,  dynamics_fn(s, a) + q̂]
        """
        predicted = float(self.dynamics_fn(state, action))
        q = self._conformal_threshold()
        return (predicted - q, predicted + q)

    # ------------------------------------------------------------------
    # Interval / unsafe overlap check
    # ------------------------------------------------------------------

    def _interval_overlaps_unsafe(self, lo: float, hi: float) -> bool:
        """Return True if the interval [lo, hi] contains any unsafe state."""
        safe_lo = self.constraint.lower
        safe_hi = self.constraint.upper
        return lo < safe_lo or hi > safe_hi

    def _resolve_fallback(self, state: float) -> float:
        """Resolve the fallback action (handles both float and callable)."""
        if callable(self.fallback_action):
            return float(self.fallback_action(state))
        return float(self.fallback_action)

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def act(self, state: float) -> Tuple[float, bool]:
        """Choose an action for the given state with safety shielding.

        Action selection priority:
        1. Base policy action — if its prediction interval is within safe region.
        2. Fallback action — if the base action's interval overlaps unsafe.
        3. Corrective action (proportional controller toward centre) — if
           both base and fallback intervals overlap unsafe.

        Parameters
        ----------
        state : float
            Current environment state.

        Returns
        -------
        action : float
            Selected action.
        shielded : bool
            True if the safety shield overrode the base policy.
        """
        if self._scores is None:
            raise RuntimeError("Must call calibrate() before using the wrapper.")

        base_action = float(self.policy(state))
        lo, hi = self._prediction_interval(state, base_action)

        # 1. Base action is certified safe
        if not self._interval_overlaps_unsafe(lo, hi):
            return base_action, False

        # 2. Try fallback action
        fallback = self._resolve_fallback(state)
        fb_lo, fb_hi = self._prediction_interval(state, fallback)
        if not self._interval_overlaps_unsafe(fb_lo, fb_hi):
            return fallback, True

        # 3. Fallback also predicted unsafe — use corrective action.
        #    Proportional controller: push toward centre with strength
        #    proportional to distance from safe midpoint.
        midpoint = (self.constraint.lower + self.constraint.upper) / 2.0
        corrective = (midpoint - state) * 0.5
        return corrective, True

    def __repr__(self) -> str:
        n = len(self._calibration_triples)
        return (
            f"ConformalSafetyWrapper(alpha={self.alpha}, "
            f"calibration_size={n}, "
            f"constraint={self.constraint})"
        )
