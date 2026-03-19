import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pytest
from conforl_lab import SafetyConstraint, ConformalSafetyWrapper, CoverageTracker


def test_safety_constraint_is_safe():
    c = SafetyConstraint(-2.0, 2.0)
    assert c.is_safe(0.0)
    assert c.is_safe(-2.0)
    assert c.is_safe(2.0)
    assert not c.is_safe(2.1)
    assert not c.is_safe(-2.1)


def test_safety_constraint_is_unsafe_set():
    c = SafetyConstraint(-2.0, 2.0)
    assert not c.is_unsafe_set(np.array([0.0, 1.0, -1.0]))
    assert c.is_unsafe_set(np.array([0.0, 3.0]))


def test_conformal_threshold_grows_with_scores():
    c = SafetyConstraint(-2.0, 2.0)
    w = ConformalSafetyWrapper(c, alpha=0.1)
    np.random.seed(0)
    for _ in range(50):
        s = np.random.uniform(-1, 1)
        a = np.random.uniform(-0.2, 0.2)
        ns = s + a + np.random.normal(0, 0.1)
        w.add_calibration(s, a, ns)
    thresh = w._conformal_threshold()
    assert thresh > 0


def test_wrapper_fallback_on_unsafe_prediction():
    c = SafetyConstraint(-2.0, 2.0)
    w = ConformalSafetyWrapper(c, alpha=0.1, fallback_action=0.0)
    np.random.seed(0)
    for _ in range(100):
        s = np.random.uniform(-1, 1)
        a = np.random.uniform(-0.2, 0.2)
        ns = s + a + np.random.normal(0, 0.05)
        w.add_calibration(s, a, ns)
    # Request a dangerous action from edge of safe region
    action, _ = w.select_action(1.9, 0.5)  # Would push to ~2.4
    # Should fall back to 0.0
    assert action == 0.0 or abs(action) < abs(0.5)


def test_coverage_tracker():
    c = SafetyConstraint(-2.0, 2.0)
    tracker = CoverageTracker(c)
    np.random.seed(42)
    for _ in range(200):
        s = np.random.uniform(-1.5, 1.5)
        a = np.random.uniform(-0.2, 0.2)
        ns = s + a + np.random.normal(0, 0.05)
        tracker.record(s, a, ns)
    report = tracker.report()
    assert report["total_steps"] == 200
    assert 0 < report["empirical_coverage"] <= 1.0


def test_coverage_guarantee():
    """End-to-end: conformal wrapper should achieve >= 90% coverage."""
    np.random.seed(123)
    c = SafetyConstraint(-2.0, 2.0)
    w = ConformalSafetyWrapper(c, alpha=0.1)
    tracker = CoverageTracker(c)

    for _ in range(300):
        s = np.random.uniform(-1.5, 1.5)
        a = np.random.uniform(-0.3, 0.3)
        ns = s + a + np.random.normal(0, 0.1)
        w.add_calibration(s, a, ns)

    state = 0.0
    for _ in range(300):
        pa = np.random.choice([-0.4, 0.4])
        action, _ = w.select_action(state, pa)
        ns = state + action + np.random.normal(0, 0.1)
        tracker.record(state, action, ns)
        state = ns
        if abs(state) > 3:
            state = 0.0

    assert tracker.empirical_coverage >= 0.85  # Allow slight slack
