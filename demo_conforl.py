#!/usr/bin/env python3
"""Demo: Conformal Safety Wrapper on a 1D Random-Walk Environment.

Environment:
    state_{t+1} = state_t + action + noise    (noise ~ N(0, 0.3))
    Unsafe region: |state| > 2  (i.e., safe = [-2, 2])

Dynamics model (deterministic mean):
    dynamics_fn(state, action) = state + action

This matches the *mean* of the true transition, so calibration residuals
equal the noise samples — which are i.i.d. Gaussian and thus exchangeable.

Base policy:
    Tries to move right (+1.0) — will eventually leave the safe region.

Conformal Safety Wrapper (alpha=0.1):
    Shields the agent when the prediction interval for the next state
    overlaps the unsafe region, replacing the action with fallback=0.0.

Expected result:
    Empirical coverage >= 0.90 (>= 90% of steps end in safe region).

Theorem (Angelopoulos et al., 2021; Venn et al.):
    For any exchangeable calibration set of size n, the conformal prediction
    set achieves marginal coverage P(Y ∈ C(X)) >= 1−α with finite-sample
    validity.
"""

import numpy as np
from conforl_lab import SafetyConstraint, ConformalSafetyWrapper, CoverageTracker


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def make_stochastic_env(noise_std: float = 0.3, seed: int = 42):
    """Factory for stochastic 1D random-walk environment step function."""
    rng = np.random.default_rng(seed)

    def step(state: float, action: float) -> float:
        return state + action + rng.normal(0.0, noise_std)

    return step


def deterministic_dynamics(state: float, action: float) -> float:
    """Mean transition model: next = state + action (no noise)."""
    return state + action


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

def aggressive_policy(state: float) -> float:
    """Always push right — intentionally unsafe near the boundary."""
    return 1.0


# ---------------------------------------------------------------------------
# Generate calibration data
# ---------------------------------------------------------------------------

def collect_calibration_data(noise_std: float = 0.3, n: int = 200, seed: int = 1):
    """Run near-center transitions to collect (state, action, next_state) triples.

    The calibration should cover the noise distribution of the real environment.
    Here we use mild random actions near the origin so the calibration residuals
    are representative of the test-time noise distribution.
    """
    rng = np.random.default_rng(seed)
    env_step = make_stochastic_env(noise_std=noise_std, seed=seed + 1000)
    triples = []
    state = 0.0
    for _ in range(n):
        action = float(rng.uniform(-0.5, 0.5))
        next_state = env_step(state, action)
        triples.append((state, action, next_state))
        # Stay near center for exchangeable calibration
        if abs(next_state) > 1.5:
            state = float(rng.uniform(-0.5, 0.5))
        else:
            state = next_state
    return triples


# ---------------------------------------------------------------------------
# Run experiment
# ---------------------------------------------------------------------------

def run_experiment(
    n_steps: int = 500,
    alpha: float = 0.1,
    noise_std: float = 0.3,
    n_calibration: int = 200,
    env_seed: int = 42,
    calib_seed: int = 1,
    eval_seed: int = 99,
):
    print("=" * 60)
    print("ConfoRL Demo: Conformal Safety Wrapper (1D Random Walk)")
    print("=" * 60)
    print(f"  alpha       = {alpha}   (target coverage >= {1-alpha:.0%})")
    print(f"  noise_std   = {noise_std}")
    print(f"  n_calib     = {n_calibration}")
    print(f"  n_eval      = {n_steps}")
    print()

    # Safety constraint and environment
    constraint = SafetyConstraint(lower_bound=-2.0, upper_bound=2.0)
    stochastic_env = make_stochastic_env(noise_std=noise_std, seed=env_seed)
    print(f"Safety constraint: {constraint}")

    # Calibration
    print(f"\nCollecting {n_calibration} calibration transitions...")
    cal_triples = collect_calibration_data(
        noise_std=noise_std, n=n_calibration, seed=calib_seed
    )

    # Conformal wrapper (uses deterministic mean dynamics model)
    wrapper = ConformalSafetyWrapper(
        policy=aggressive_policy,
        dynamics_fn=deterministic_dynamics,
        constraint=constraint,
        alpha=alpha,
        fallback_action=0.0,
    )
    wrapper.calibrate(cal_triples)

    q_hat = wrapper._conformal_threshold()
    print(f"Calibrated: {wrapper}")
    print(f"Conformal threshold q̂ = {q_hat:.4f}")
    print(f"Prediction interval half-width: ±{q_hat:.4f}")

    # Evaluation: run agent with conformal shield
    tracker = CoverageTracker(alpha=alpha)
    rng = np.random.default_rng(eval_seed)
    state = 0.0
    n_shielded = 0

    for _ in range(n_steps):
        action, shielded = wrapper.act(state)
        if shielded:
            n_shielded += 1
        next_state = stochastic_env(state, action)
        tracker.record(next_state, constraint)
        state = next_state
        # Episode reset if far out of bounds
        if abs(state) > 3.0:
            state = float(rng.uniform(-0.5, 0.5))

    # Report
    print("\n" + tracker.summary())
    print(f"\n  Actions shielded:     {n_shielded} / {n_steps} ({n_shielded/n_steps*100:.1f}%)")

    if tracker.guarantee_holds:
        print(
            f"\n[PASS] Empirical coverage {tracker.empirical_coverage:.4f} "
            f">= theoretical bound {tracker.theoretical_lower_bound:.4f}"
        )
    else:
        print(
            f"\n[FAIL] Empirical coverage {tracker.empirical_coverage:.4f} "
            f"< theoretical bound {tracker.theoretical_lower_bound:.4f}"
        )

    return tracker


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tracker = run_experiment(
        n_steps=500,
        alpha=0.1,
        noise_std=0.3,
        n_calibration=200,
    )
    assert tracker.guarantee_holds, (
        f"Coverage guarantee failed! "
        f"Got {tracker.empirical_coverage:.4f}, expected >= {tracker.theoretical_lower_bound:.4f}"
    )
    print("\nDemo complete.")
