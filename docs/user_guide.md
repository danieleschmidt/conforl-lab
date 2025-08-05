# ConfoRL User Guide

Welcome to ConfoRL! This guide will help you get started with adaptive conformal risk control for reinforcement learning.

## What is ConfoRL?

ConfoRL brings **provable finite-sample safety guarantees** to reinforcement learning through adaptive conformal prediction. Unlike traditional RL that provides no formal safety guarantees, ConfoRL ensures that P(failure) ≤ ε with high probability, enabling deployment in safety-critical applications.

### Key Benefits

- 🛡️ **Provable Safety**: Formal mathematical guarantees on risk bounds
- 🔄 **Adaptive**: Risk bounds improve automatically as more data is collected
- 🎯 **Algorithm Agnostic**: Works with any RL algorithm (SAC, PPO, TD3, etc.)
- 📊 **Distribution-Free**: No assumptions about environment dynamics
- 🚀 **Production-Ready**: Complete deployment pipeline with monitoring

## Installation

### Prerequisites

- Python 3.8+
- NumPy, JAX/PyTorch, Gymnasium

### Install from Source

```bash
git clone https://github.com/danieleschmidt/conforl-lab.git
cd conforl-lab
pip install -e .
```

### Install Dependencies

```bash
# Core dependencies
pip install -r requirements.txt

# Development dependencies (optional)
pip install -r requirements.txt[dev]

# Visualization dependencies (optional)
pip install -r requirements.txt[vis]
```

## Quick Start

### 1. Basic Training

```python
import conforl
import gymnasium as gym

# Create environment
env = gym.make('CartPole-v1')

# Initialize risk controller
risk_controller = conforl.AdaptiveRiskController(
    target_risk=0.05,    # 5% risk tolerance
    confidence=0.95      # 95% confidence
)

# Create agent with safety guarantees
agent = conforl.ConformaSAC(
    env=env,
    risk_controller=risk_controller
)

# Train with automatic risk adaptation
agent.train(total_timesteps=100000)
```

### 2. Safe Deployment

```python
# Deploy with safety monitoring
pipeline = conforl.SafeDeploymentPipeline(
    agent=agent,
    risk_monitor=True,
    fallback_policy=safe_policy
)

results = pipeline.deploy(env=env, num_episodes=100)
print(f"Average risk: {results['avg_risk']}")
```

### 3. Risk Certification

```python
# Get formal risk certificate
certificate = agent.get_risk_certificate()
print(f"Risk bound: {certificate.risk_bound} (confidence: {certificate.confidence})")
```

## Core Concepts

### Risk Measures

ConfoRL supports multiple types of risk measures:

#### Safety Violations
Monitor constraint violations in safety-critical environments:

```python
safety_risk = conforl.SafetyViolationRisk(
    constraint_key="constraint_violation",
    violation_threshold=0.0
)
```

#### Performance Risk
Ensure performance doesn't degrade below acceptable levels:

```python
performance_risk = conforl.PerformanceRisk(
    target_return=100.0,
    risk_type="shortfall"  # or "variance"
)
```

#### Custom Risk Measures
Define your own domain-specific risk measures:

```python
def collision_risk_fn(trajectory):
    distances = [info.get('min_distance', 1.0) for info in trajectory.infos]
    return sum(d < 0.1 for d in distances) / len(distances)

collision_risk = conforl.CustomRiskMeasure(
    name="collision_risk",
    compute_fn=collision_risk_fn
)
```

### Risk Controllers

Risk controllers manage how risk bounds are updated over time:

#### Adaptive Risk Controller
Automatically adapts to changing conditions:

```python
controller = conforl.AdaptiveRiskController(
    target_risk=0.05,
    confidence=0.95,
    window_size=1000,      # Adaptation window
    learning_rate=0.01     # Adaptation rate
)
```

#### Multi-Risk Controller
Handle multiple risk types simultaneously:

```python
multi_controller = conforl.MultiRiskController([
    ("safety", 0.01),      # 1% safety violation risk
    ("performance", 0.05), # 5% performance degradation risk
    ("collision", 0.001)   # 0.1% collision risk
])
```

#### Online Risk Adaptation
Real-time adaptation during deployment:

```python
online_controller = conforl.OnlineRiskAdaptation(
    initial_quantile=0.9,
    learning_rate=0.01,
    target_coverage=0.95
)
```

## Algorithms

ConfoRL provides conformal versions of popular RL algorithms:

### ConformaSAC
Soft Actor-Critic with conformal risk control:

```python
agent = conforl.ConformaSAC(
    env=env,
    risk_controller=risk_controller,
    learning_rate=3e-4,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    alpha=0.2
)
```

### ConformaPPO
Proximal Policy Optimization with safety guarantees:

```python
agent = conforl.ConformaPPO(
    env=env,
    risk_controller=risk_controller,
    learning_rate=3e-4,
    batch_size=64,
    n_epochs=10,
    clip_range=0.2
)
```

### ConformaTD3
Twin Delayed DDPG with risk bounds:

```python
agent = conforl.ConformaTD3(
    env=env,
    risk_controller=risk_controller,
    policy_delay=2,
    target_noise=0.2,
    noise_clip=0.5
)
```

### ConformaCQL (Offline RL)
Conservative Q-Learning for offline datasets:

```python
# Load offline dataset
dataset = load_d4rl_dataset('halfcheetah-expert-v2')

agent = conforl.ConformaCQL(
    env=env,
    dataset=dataset,
    risk_controller=risk_controller,
    cql_weight=1.0
)

# Train on offline data
agent.train_offline(n_epochs=1000)
```

## Advanced Features

### Self-Improving Agents

Create agents that automatically optimize their own hyperparameters:

```python
# Wrap any agent to make it self-improving
improving_agent = conforl.SelfImprovingAgent(
    base_agent=agent,
    improvement_threshold=0.05,
    evaluation_window=100
)

# Training automatically adapts parameters
improving_agent.train(total_timesteps=100000)

# Check improvement statistics
stats = improving_agent.get_improvement_stats()
print(f"Adaptations made: {stats['adaptive_tuner']['adaptation_count']}")
```

### Performance Optimization

#### Adaptive Caching
Automatically cache expensive computations:

```python
cache = conforl.AdaptiveCache(
    max_size=1000,
    adaptive_ttl=True,
    compression=True
)

# Cache expensive risk computations
def expensive_risk_computation(trajectory):
    # ... complex computation ...
    return risk_value

# Cached version
cached_result = cache.cached_computation(
    "risk_computation",
    expensive_risk_computation,
    trajectory
)
```

#### Parallel Training
Scale training across multiple workers:

```python
parallel_trainer = conforl.ParallelTraining(
    num_workers=4,
    use_processes=True
)

# Define worker function
def worker_func(worker_id, args, command, state=None, data=None):
    # Training logic for each worker
    return results

parallel_trainer.start_workers(worker_func, worker_args)
```

### Monitoring and Metrics

#### Comprehensive Metrics
Track everything that matters:

```python
metrics = conforl.MetricsCollector()

# Record different types of metrics
metrics.increment("episodes_completed")
metrics.gauge("current_risk", 0.03)
metrics.histogram("episode_length", 245)

# Time operations
with metrics.timer("training_step"):
    agent.train_step()

# Get summaries
summary = metrics.get_metric_summary("episode_length")
print(f"Average episode length: {summary.mean}")
```

#### Performance Tracking
Monitor agent performance over time:

```python
tracker = conforl.PerformanceTracker()

# Track training progress
tracker.track_training_episode(
    episode_reward=150.0,
    episode_length=200,
    episode_risk=0.02,
    algorithm="ConformaSAC"
)

# Track inference performance
tracker.track_inference(
    inference_time=0.01,
    risk_bound=0.03,
    confidence=0.95
)

# Get comprehensive summary
summary = tracker.get_performance_summary()
```

### Production Deployment

#### Safe Deployment Pipeline
Deploy with comprehensive safety monitoring:

```python
pipeline = conforl.SafeDeploymentPipeline(
    agent=trained_agent,
    fallback_policy=safe_fallback,
    risk_monitor=True,
    alert_threshold=0.8,
    max_risk_violations=5
)

# Deploy with detailed monitoring
results = pipeline.deploy(
    env=production_env,
    num_episodes=1000,
    eval_callback=custom_evaluator
)

# Monitor deployment status
status = pipeline.get_deployment_status()
if status['fallback_active']:
    print("Safety fallback is active!")
```

#### Emergency Controls
Built-in emergency stop capabilities:

```python
# Trigger emergency stop if needed
pipeline.emergency_stop("Risk threshold exceeded")

# Check deployment status
status = pipeline.get_deployment_status()
print(f"Deployment status: {status}")
```

## Safety-Critical Applications

### Autonomous Vehicles

```python
# Define collision risk measure
collision_risk = conforl.CustomRiskMeasure(
    name="collision_risk",
    compute_fn=lambda traj: compute_collision_probability(traj)
)

# Multi-objective risk control
risk_controller = conforl.MultiRiskController([
    ("collision", 0.001),      # 0.1% collision risk
    ("comfort", 0.05),         # 5% passenger discomfort
    ("efficiency", 0.10)       # 10% efficiency loss
])

# Train autonomous driving agent
agent = conforl.ConformaSAC(
    env=driving_env,
    risk_controller=risk_controller,
    risk_measure=collision_risk
)
```

### Medical Applications

```python
# Define treatment risk
treatment_risk = conforl.CustomRiskMeasure(
    name="adverse_outcome",
    compute_fn=lambda traj: assess_medical_risk(traj)
)

# Very conservative risk controller
risk_controller = conforl.AdaptiveRiskController(
    target_risk=0.001,  # 0.1% risk tolerance
    confidence=0.999    # 99.9% confidence
)

# Clinical decision support agent
agent = conforl.ConformaCQL(
    env=medical_env,
    dataset=clinical_dataset,
    risk_controller=risk_controller,
    risk_measure=treatment_risk
)
```

### Industrial Control

```python
# Define safety violation risk
safety_risk = conforl.SafetyViolationRisk(
    constraint_key="safety_constraint",
    violation_threshold=0.0
)

# Industrial control agent
agent = conforl.ConformaPPO(
    env=industrial_env,
    risk_controller=risk_controller,
    risk_measure=safety_risk
)

# Deploy with strict monitoring
pipeline = conforl.SafeDeploymentPipeline(
    agent=agent,
    fallback_policy=emergency_shutdown,
    alert_threshold=0.5,
    max_risk_violations=1
)
```

## Command Line Interface

ConfoRL provides a comprehensive CLI for common tasks:

### Training

```bash
# Train SAC agent
conforl train --algorithm sac --env CartPole-v1 --timesteps 100000 \
  --target-risk 0.05 --confidence 0.95 --save-path ./models/cartpole_sac

# Train with custom configuration
conforl train --algorithm ppo --env HalfCheetah-v4 --config config.yaml
```

### Evaluation

```bash
# Evaluate trained agent
conforl evaluate --model ./models/cartpole_sac --env CartPole-v1 \
  --episodes 100 --render

# Generate performance report
conforl evaluate --model ./models/cartpole_sac --env CartPole-v1 \
  --episodes 1000 --output-report evaluation_report.json
```

### Deployment

```bash
# Deploy to production
conforl deploy --model ./models/cartpole_sac --env CartPole-v1 \
  --monitor --episodes 1000 --log-dir ./deploy_logs

# Deploy with custom fallback
conforl deploy --model ./models/cartpole_sac --env CartPole-v1 \
  --fallback-policy ./policies/safe_policy.pkl
```

### Risk Certification

```bash
# Generate risk certificate
conforl certificate --model ./models/cartpole_sac --coverage 0.95 \
  --output certificate.json

# Validate certificate
conforl validate-certificate --certificate certificate.json \
  --test-data test_trajectories.pkl
```

## Configuration

### YAML Configuration

Create a configuration file for complex setups:

```yaml
# config.yaml
algorithm:
  name: "sac"
  learning_rate: 3e-4
  batch_size: 256
  buffer_size: 1000000

risk_control:
  target_risk: 0.05
  confidence: 0.95
  adaptation_window: 1000
  learning_rate: 0.01

environment:
  name: "HalfCheetah-v4"
  max_episode_steps: 1000

training:
  total_timesteps: 1000000
  eval_frequency: 10000
  save_frequency: 50000

deployment:
  monitoring: true
  alert_threshold: 0.8
  max_violations: 5
  fallback_policy: "safe_policy"

logging:
  level: "INFO"
  directory: "./logs"
  format: "json"
```

Use with CLI:

```bash
conforl train --config config.yaml
```

### Environment Variables

Configure ConfoRL using environment variables:

```bash
export CONFORL_LOG_LEVEL=DEBUG
export CONFORL_DATA_DIR=/data/conforl
export CONFORL_MODEL_DIR=/models/conforl
export CONFORL_CACHE_SIZE=1000
```

## Troubleshooting

### Common Issues

#### Training Instability
If training is unstable, try:
- Reducing learning rate
- Increasing risk tolerance slightly
- Using larger adaptation window
- Checking environment reward scaling

```python
# More conservative settings
risk_controller = conforl.AdaptiveRiskController(
    target_risk=0.1,        # Higher tolerance
    learning_rate=0.001,    # Slower adaptation
    window_size=2000        # Larger window
)
```

#### High Risk Violations
If you're seeing too many risk violations:
- Check risk measure implementation
- Verify environment provides constraint information
- Consider using fallback policy
- Increase confidence level

```python
# More conservative approach
risk_controller = conforl.AdaptiveRiskController(
    target_risk=0.01,       # Lower tolerance
    confidence=0.99         # Higher confidence
)
```

#### Performance Issues
For performance optimization:
- Enable caching for expensive computations
- Use parallel training for large-scale problems
- Profile code to identify bottlenecks
- Consider GPU acceleration if available

```python
# Enable performance optimizations
agent = conforl.ConformaSAC(
    env=env,
    risk_controller=risk_controller,
    device="cuda"  # Use GPU if available
)

# Add performance monitoring
tracker = conforl.PerformanceTracker()
agent.performance_tracker = tracker
```

### Debugging

#### Enable Debug Logging

```python
import logging
from conforl.utils.logging import setup_logging

setup_logging(level="DEBUG", include_console=True)
```

#### Check Metrics

```python
# Get detailed metrics
summary = agent.performance_tracker.get_performance_summary()
print(f"Training metrics: {summary}")

# Check risk controller state
risk_stats = agent.risk_controller.get_adaptation_stats()
print(f"Risk adaptation: {risk_stats}")
```

#### Validate Configuration

```python
from conforl.utils.validation import validate_config

try:
    validated_config = validate_config(your_config)
except ValidationError as e:
    print(f"Configuration error: {e}")
```

## Best Practices

### 1. Start Conservative
Begin with conservative risk settings and gradually adjust:

```python
# Start with higher risk tolerance
risk_controller = conforl.AdaptiveRiskController(
    target_risk=0.1,        # 10% initially
    confidence=0.95
)

# Gradually reduce as you gain confidence
# target_risk=0.05 -> 0.01 -> 0.005
```

### 2. Use Multiple Risk Measures
Combine different risk types for comprehensive safety:

```python
risk_measures = {
    "safety": conforl.SafetyViolationRisk(),
    "performance": conforl.PerformanceRisk(target_return=100),
    "stability": conforl.CustomRiskMeasure("stability", stability_fn)
}

multi_controller = conforl.MultiRiskController([
    ("safety", 0.01),
    ("performance", 0.05),
    ("stability", 0.02)
])
```

### 3. Monitor Everything
Use comprehensive monitoring in production:

```python
# Set up monitoring
metrics = conforl.MetricsCollector()
tracker = conforl.PerformanceTracker(metrics)
risk_metrics = conforl.RiskMetrics(metrics)

# Deploy with monitoring
pipeline = conforl.SafeDeploymentPipeline(
    agent=agent,
    risk_monitor=True,
    log_dir="./production_logs"
)
```

### 4. Test Thoroughly
Always test in safe environments first:

```python
# Test in simulation
test_results = pipeline.deploy(
    env=simulation_env,
    num_episodes=1000
)

# Verify safety guarantees hold
assert test_results['avg_risk'] <= target_risk * 1.1  # Allow 10% margin
```

### 5. Have Fallback Plans
Always implement safe fallback policies:

```python
def safe_fallback_policy(state, env):
    """Conservative policy that prioritizes safety."""
    # Implement domain-specific safe actions
    return safe_action

pipeline = conforl.SafeDeploymentPipeline(
    agent=agent,
    fallback_policy=safe_fallback_policy,
    max_risk_violations=3  # Low tolerance
)
```

## Next Steps

- Read the [API Reference](api_reference.md) for detailed documentation
- Check out [examples](../examples/) for specific use cases  
- Join our community discussions for support
- Contribute to the project on GitHub

For more advanced topics, see:
- [Developer Guide](developer_guide.md)
- [Deployment Guide](deployment_guide.md)
- [Research Extensions](research_extensions.md)