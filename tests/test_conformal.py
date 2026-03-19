"""Tests for conforl_lab: SafetyConstraint, ConformalSafetyWrapper, CoverageTracker."""

import math
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np

from conforl_lab import SafetyConstraint, ConformalSafetyWrapper, CoverageTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def simple_dynamics(state: float, action: float) -> float:
    return state + action


def make_triples(n: int = 100, seed: int = 42):
    rng = np.random.default_rng(seed)
    states = rng.uniform(-1.0, 1.0, n)
    actions = rng.uniform(-0.3, 0.3, n)
    next_states = states + actions + rng.normal(0, 0.1, n)
    return list(zip(states, actions, next_states))


def always_right(state: float) -> float:
    return 1.0


CONSTRAINT = SafetyConstraint(-2.0, 2.0)


# ===========================================================================
# SafetyConstraint tests
# ===========================================================================

class TestSafetyConstraint:
    def test_safe_inside(self):
        sc = SafetyConstraint(-2.0, 2.0)
        assert sc.is_safe(0.0)
        assert sc.is_safe(-2.0)
        assert sc.is_safe(2.0)

    def test_unsafe_outside(self):
        sc = SafetyConstraint(-2.0, 2.0)
        assert not sc.is_safe(2.1)
        assert not sc.is_safe(-3.0)

    def test_boundary_inclusive(self):
        sc = SafetyConstraint(-1.0, 1.0)
        assert sc.is_safe(-1.0)
        assert sc.is_safe(1.0)
        assert not sc.is_safe(-1.0001)
        assert not sc.is_safe(1.0001)

    def test_is_unsafe_array_all_safe(self):
        sc = SafetyConstraint(-2.0, 2.0)
        assert not sc.is_unsafe(np.array([0.0, 0.5, -1.0, 1.5]))

    def test_is_unsafe_array_one_bad(self):
        sc = SafetyConstraint(-2.0, 2.0)
        assert sc.is_unsafe(np.array([0.0, 3.0, -1.0]))

    def test_is_unsafe_array_all_bad(self):
        sc = SafetyConstraint(-1.0, 1.0)
        assert sc.is_unsafe(np.array([-5.0, 5.0]))

    def test_invalid_bounds(self):
        with pytest.raises(ValueError):
            SafetyConstraint(1.0, 0.0)
        with pytest.raises(ValueError):
            SafetyConstraint(1.0, 1.0)

    def test_repr(self):
        sc = SafetyConstraint(-2.0, 2.0)
        assert "SafetyConstraint" in repr(sc)


# ===========================================================================
# ConformalSafetyWrapper tests
# ===========================================================================

class TestConformalSafetyWrapper:
    def setup_method(self):
        self.wrapper = ConformalSafetyWrapper(
            policy=always_right,
            dynamics_fn=simple_dynamics,
            constraint=CONSTRAINT,
            alpha=0.1,
            fallback_action=0.0,
        )

    def test_calibrate(self):
        triples = make_triples(100)
        self.wrapper.calibrate(triples)
        assert self.wrapper._scores is not None
        assert len(self.wrapper._scores) == 100

    def test_calibrate_empty_raises(self):
        with pytest.raises(ValueError):
            self.wrapper.calibrate([])

    def test_act_before_calibrate_raises(self):
        with pytest.raises(RuntimeError):
            self.wrapper.act(0.0)

    def test_act_returns_float_bool(self):
        self.wrapper.calibrate(make_triples(100))
        action, shielded = self.wrapper.act(0.0)
        assert isinstance(action, float)
        assert isinstance(shielded, bool)

    def test_shield_activates_near_boundary(self):
        """State 1.9 + action 1.0 = 2.9 > 2.0 -> should shield."""
        rng = np.random.default_rng(0)
        # Tight calibration (small noise) -> small q_hat
        states = rng.uniform(-0.1, 0.1, 500)
        actions = rng.uniform(-0.05, 0.05, 500)
        next_states = states + actions + rng.normal(0, 0.02, 500)
        triples = list(zip(states, actions, next_states))
        self.wrapper.calibrate(triples)
        action, shielded = self.wrapper.act(1.9)
        assert shielded, "Shield should activate when predicted next state is unsafe"
        assert action != pytest.approx(1.0)  # not the base action

    def test_conformal_threshold_formula(self):
        """Quantile level uses finite-sample correction ceil((n+1)(1-alpha))/n."""
        triples = make_triples(99)
        self.wrapper.calibrate(triples)
        n, alpha = 99, 0.1
        level = min(math.ceil((n + 1) * (1 - alpha)) / n, 1.0)
        expected = float(np.quantile(self.wrapper._scores, level))
        assert self.wrapper._conformal_threshold() == pytest.approx(expected)

    def test_invalid_alpha(self):
        with pytest.raises(ValueError):
            ConformalSafetyWrapper(
                policy=always_right,
                dynamics_fn=simple_dynamics,
                constraint=CONSTRAINT,
                alpha=1.5,
            )

    def test_repr_contains_key_info(self):
        self.wrapper.calibrate(make_triples(50))
        r = repr(self.wrapper)
        assert "ConformalSafetyWrapper" in r
        assert "alpha=0.1" in r
        assert "calibration_size=50" in r

    def test_empirical_coverage_end_to_end(self):
        """Run 300 steps; coverage should be >= 0.9."""
        rng = np.random.default_rng(7)
        noise = 0.15  # smaller noise for reliable coverage

        def stochastic_env(s, a):
            return s + a + rng.normal(0, noise)

        cal_rng = np.random.default_rng(1)
        cal_states = cal_rng.uniform(-1.0, 1.0, 300)
        cal_actions = cal_rng.uniform(-0.3, 0.3, 300)
        cal_next = cal_states + cal_actions + cal_rng.normal(0, noise, 300)
        triples = list(zip(cal_states, cal_actions, cal_next))

        wrapper = ConformalSafetyWrapper(
            policy=always_right,
            dynamics_fn=simple_dynamics,
            constraint=CONSTRAINT,
            alpha=0.1,
            fallback_action=0.0,
        )
        wrapper.calibrate(triples)

        tracker = CoverageTracker(alpha=0.1)
        state = 0.0
        for _ in range(300):
            action, _ = wrapper.act(state)
            next_state = stochastic_env(state, action)
            tracker.record(next_state, CONSTRAINT)
            state = next_state
            if abs(state) > 2.5:
                state = float(rng.uniform(-0.5, 0.5))

        assert tracker.empirical_coverage >= 0.9, (
            f"Coverage {tracker.empirical_coverage:.4f} < 0.9"
        )


# ===========================================================================
# CoverageTracker tests
# ===========================================================================

class TestCoverageTracker:
    def test_initial_state(self):
        tracker = CoverageTracker(alpha=0.1)
        assert tracker._total_steps == 0
        assert math.isnan(tracker.empirical_coverage)

    def test_theoretical_bound(self):
        tracker = CoverageTracker(alpha=0.1)
        assert tracker.theoretical_lower_bound == pytest.approx(0.9)

    def test_record_safe(self):
        tracker = CoverageTracker(alpha=0.1)
        result = tracker.record(0.0, CONSTRAINT)
        assert result is True
        assert tracker._total_steps == 1
        assert tracker._safe_steps == 1

    def test_record_unsafe(self):
        tracker = CoverageTracker(alpha=0.1)
        result = tracker.record(5.0, CONSTRAINT)
        assert result is False
        assert tracker._total_steps == 1
        assert tracker._safe_steps == 0

    def test_full_coverage(self):
        tracker = CoverageTracker(alpha=0.1)
        for _ in range(10):
            tracker.record(0.0, CONSTRAINT)
        assert tracker.empirical_coverage == pytest.approx(1.0)

    def test_half_coverage(self):
        tracker = CoverageTracker(alpha=0.1)
        for _ in range(5):
            tracker.record(0.0, CONSTRAINT)
            tracker.record(5.0, CONSTRAINT)
        assert tracker.empirical_coverage == pytest.approx(0.5)

    def test_guarantee_holds(self):
        tracker = CoverageTracker(alpha=0.1)
        for _ in range(95):
            tracker.record(0.0, CONSTRAINT)
        for _ in range(5):
            tracker.record(5.0, CONSTRAINT)
        assert tracker.guarantee_holds is True

    def test_guarantee_fails(self):
        tracker = CoverageTracker(alpha=0.1)
        for _ in range(80):
            tracker.record(0.0, CONSTRAINT)
        for _ in range(20):
            tracker.record(5.0, CONSTRAINT)
        assert tracker.guarantee_holds is False

    def test_reset(self):
        tracker = CoverageTracker(alpha=0.1)
        for _ in range(10):
            tracker.record(0.0, CONSTRAINT)
        tracker.reset()
        assert tracker._total_steps == 0
        assert tracker._safe_steps == 0

    def test_summary_contains_key_info(self):
        tracker = CoverageTracker(alpha=0.1)
        for _ in range(9):
            tracker.record(0.0, CONSTRAINT)
        tracker.record(5.0, CONSTRAINT)
        s = tracker.summary()
        assert "Coverage Report" in s
        assert "90.0%" in s

    def test_invalid_alpha(self):
        with pytest.raises(ValueError):
            CoverageTracker(alpha=0.0)
        with pytest.raises(ValueError):
            CoverageTracker(alpha=1.1)

    def test_repr(self):
        tracker = CoverageTracker(alpha=0.1)
        tracker.record(0.0, CONSTRAINT)
        assert "CoverageTracker" in repr(tracker)
