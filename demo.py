#!/usr/bin/env python3
"""
Demo: Conformal Safety Wrapper on a 1D Random Walk.

Environment: state_{t+1} = state_t + action_t + noise
Safe region: [-2, 2]
Unsafe region: |state| > 2
Alpha: 0.1 (target coverage >= 90%)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from conforl_lab import SafetyConstraint, ConformalSafetyWrapper, CoverageTracker

np.random.seed(42)

# Setup
constraint = SafetyConstraint(lower_bound=-2.0, upper_bound=2.0)
wrapper = ConformalSafetyWrapper(constraint, alpha=0.1, fallback_action=0.0)
tracker = CoverageTracker(constraint)

# Phase 1: Collect calibration data (100 safe trajectories)
print("Phase 1: Collecting calibration data...")
for _ in range(200):
    state = np.random.uniform(-1.5, 1.5)
    action = np.random.uniform(-0.3, 0.3)
    noise = np.random.normal(0, 0.1)
    next_state = state + action + noise
    wrapper.add_calibration(state, action, next_state)

print(f"  Calibration set size: {len(wrapper._scores)}")
print(f"  Conformal threshold: {wrapper._conformal_threshold():.4f}")

# Phase 2: Run episode with conformal safety wrapper
print("\nPhase 2: Running episode with conformal wrapper...")
state = 0.0
n_steps = 500

for step in range(n_steps):
    # Aggressive policy: try to push toward boundary
    policy_action = np.random.choice([-0.5, -0.3, 0.3, 0.5])
    noise = np.random.normal(0, 0.15)

    action, certified = wrapper.select_action(state, policy_action)
    next_state = state + action + noise

    tracker.record(state, action, next_state)
    state = next_state

    # Soft reset if too far out
    if abs(state) > 3.0:
        state = 0.0

report = tracker.report()
print(f"\n=== Coverage Report ===")
print(f"  Total steps:        {report['total_steps']}")
print(f"  Safe steps:         {report['safe_steps']}")
print(f"  Empirical coverage: {report['empirical_coverage']:.4f} ({report['empirical_coverage']*100:.1f}%)")
print(f"  Target (1-alpha):   0.9000 (90.0%)")
print(f"  Guarantee met:      {report['guarantee_met']}")

if report['empirical_coverage'] >= 0.9:
    print("\n✓ Conformal safety guarantee VERIFIED: coverage >= 90%")
else:
    print("\n✗ Coverage below target (may need more calibration data or tighter threshold)")
