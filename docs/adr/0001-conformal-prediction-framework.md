# ADR-0001 - Conformal Prediction Framework Selection

## Status
Accepted

## Context
ConfoRL needs a robust conformal prediction framework to provide finite-sample safety guarantees for reinforcement learning. The framework must be:
- Mathematically rigorous with proven coverage guarantees
- Computationally efficient for real-time applications  
- Extensible to different risk measures and RL algorithms
- Compatible with both offline and online learning scenarios

## Decision
We will implement split conformal prediction as the core framework, with support for:
- Standard split conformal prediction for offline scenarios
- Online conformal prediction with adaptive quantile tracking
- Custom nonconformity scores for different risk measures
- Efficient quantile estimation using P² algorithm

## Rationale
Split conformal prediction provides several key advantages:
1. **Finite-sample guarantees**: Coverage bounds hold for any sample size
2. **Distribution-free**: No assumptions about data distribution
3. **Computational efficiency**: O(1) prediction time after calibration
4. **Flexibility**: Works with any underlying predictor
5. **Proven theory**: Well-established mathematical foundations

## Consequences

### Positive Consequences
- Provable safety guarantees for RL deployment
- Framework-agnostic approach works with any RL algorithm
- Real-time inference capability (<10ms prediction latency)
- Strong theoretical foundations increase research credibility
- Extensible architecture supports future research directions

### Negative Consequences
- Requires calibration set, reducing available training data
- Coverage guarantees are marginal, not conditional
- May be conservative in some scenarios
- Additional computational overhead during training

## Alternatives Considered

### Alternative 1: Full Conformal Prediction
Rejected due to computational complexity O(n) per prediction, making real-time applications infeasible.

### Alternative 2: Bayesian Uncertainty Quantification
Rejected because it requires distributional assumptions that may not hold in practice and lacks finite-sample guarantees.

### Alternative 3: Ensemble Methods
Rejected as the primary approach because ensembles don't provide rigorous coverage guarantees, though they may be used as nonconformity measures.

## Related Decisions
- [ADR-0002] - Risk measure selection and implementation
- [ADR-0003] - Online adaptation strategy for distribution shift

## Notes
Implementation should follow the approach outlined in:
- Vovk et al. (2005) "Algorithmic Learning in a Random World"
- Shafer & Vovk (2008) "A Tutorial on Conformal Prediction"
- Angelopoulos & Bates (2021) "A Gentle Introduction to Conformal Prediction"

Key implementation details:
- Use quantile regression for nonconformity score computation
- Implement efficient online quantile tracking for streaming scenarios
- Support multiple risk measures through pluggable nonconformity functions
- Maintain backward compatibility with scikit-learn API patterns

---

**Date**: 2024-08-18  
**Authors**: ConfoRL Development Team  
**Reviewers**: Research Team, Safety Committee