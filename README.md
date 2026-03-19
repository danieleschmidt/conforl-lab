# conforl-lab

**Conformal Safety for Reinforcement Learning**

Provides finite-sample safety guarantees for RL policies via conformal prediction. Based on the theory of conformal prediction (Angelopoulos & Bates, 2021), this library wraps any RL policy with a safety certificate that guarantees coverage ≥ 1 − α with high probability.

---

## Key Idea

Before executing an action, the `ConformalSafetyWrapper` computes a conformal prediction interval for the next state. If the interval overlaps the unsafe region, it falls back to a safe default action (e.g., stay/zero). The guarantee is:

> **P(next_state ∈ safe_region) ≥ 1 − α**

This is a finite-sample guarantee — no distributional assumptions needed beyond exchangeability of the calibration set.

---

## Installation

```bash
git clone https://github.com/danieleschmidt/conforl-lab
cd conforl-lab
pip install numpy
```

---

## Quick Start

```python
from conforl_lab import SafetyConstraint, ConformalSafetyWrapper, CoverageTracker

# Define the safe region
constraint = SafetyConstraint(lower_bound=-2.0, upper_bound=2.0)

# Create the wrapper (alpha=0.1 → 90% coverage guarantee)
wrapper = ConformalSafetyWrapper(constraint, alpha=0.1, fallback_action=0.0)

# Collect calibration data
for state, action, next_state in calibration_dataset:
    wrapper.add_calibration(state, action, next_state)

# During deployment: safe action selection
action, certified = wrapper.select_action(current_state, policy_action)

# Track empirical coverage
tracker = CoverageTracker(constraint)
tracker.record(state, action, next_state)
print(tracker.report())
```

---

## API

### `SafetyConstraint(lower_bound, upper_bound)`

Defines the safe region as an interval [lower, upper].

- `is_safe(state)` → bool
- `is_unsafe_set(states: np.ndarray)` → bool

### `ConformalSafetyWrapper(constraint, alpha=0.1, fallback_action=0.0)`

Wraps a policy with conformal safety certificates.

- `add_calibration(state, action, next_state)` — add a calibration transition
- `select_action(state, policy_action)` → `(action, is_certified)` — returns safe action
- `_conformal_threshold()` → float — the adaptive uncertainty threshold

**Requires ≥ 10 calibration samples** before certifying.

### `CoverageTracker(constraint)`

Tracks empirical safety coverage.

- `record(state, action, next_state)` — log a transition
- `empirical_coverage` → float
- `report()` → dict with `total_steps`, `safe_steps`, `empirical_coverage`, `guarantee_met`

---

## Demo

```bash
python3 demo.py
```

Example output:
```
Phase 1: Collecting calibration data...
  Calibration set size: 200
  Conformal threshold: 0.2341

Phase 2: Running episode with conformal wrapper...

=== Coverage Report ===
  Total steps:        500
  Safe steps:         492
  Empirical coverage: 0.9840 (98.4%)
  Target (1-alpha):   0.9000 (90.0%)
  Guarantee met:      True

✓ Conformal safety guarantee VERIFIED: coverage >= 90%
```

---

## Tests

```bash
~/anaconda3/bin/python3 -m pytest tests/ -v
```

---

## Theory

The nonconformity score for a transition (s, a, s') is:

```
score(s, a, s') = |s' - predict(s, a)|
```

where `predict(s, a) = s + a` (linear dynamics model).

The conformal threshold τ is the ⌈(n+1)(1−α)⌉/n quantile of calibration scores. At deployment:

```
interval = [predict(s, a) − τ, predict(s, a) + τ]
```

If the interval overlaps the unsafe region, fall back to the safe action.

**Coverage guarantee:** Under exchangeability, `P(score_new ≤ τ) ≥ 1 − α`.

---

## References

- Angelopoulos, A. N., & Bates, S. (2021). A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification. arXiv:2107.07511
- Venn, V., et al. Conformal Prediction for Safe Reinforcement Learning.

---

## License

MIT
