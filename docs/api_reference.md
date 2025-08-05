# ConfoRL API Reference

This document provides detailed API reference for all ConfoRL components.

## Core Components

### conforl.core.conformal

#### SplitConformalPredictor

```python
class SplitConformalPredictor(coverage=0.95, score_function=None)
```

Split conformal prediction with finite-sample guarantees.

**Parameters:**
- `coverage` (float): Target coverage probability (0, 1)
- `score_function` (callable, optional): Custom nonconformity score function

**Methods:**

##### calibrate(calibration_data, calibration_scores)
Calibrate the conformal predictor using split conformal method.

**Parameters:**
- `calibration_data` (np.ndarray): Calibration input data
- `calibration_scores` (np.ndarray): Precomputed nonconformity scores

##### predict(test_predictions, return_scores=False)
Generate split conformal prediction intervals.

**Parameters:**
- `test_predictions` (np.ndarray): Point predictions for test data
- `return_scores` (bool): Whether to return nonconformity scores

**Returns:**
- `ConformalSet`: Conformal prediction set with coverage guarantee

##### get_risk_certificate(test_data, risk_function)
Generate formal risk certificate.

**Parameters:**
- `test_data` (np.ndarray): Test input data
- `risk_function` (callable): Function computing risk from predictions

**Returns:**
- `RiskCertificate`: Certificate with formal risk guarantees

### conforl.core.types

#### RiskCertificate

```python
@dataclass
class RiskCertificate:
    risk_bound: float
    confidence: float
    coverage_guarantee: float
    method: str
    sample_size: int
    timestamp: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
```

Certificate providing formal risk guarantees.

#### TrajectoryData

```python
@dataclass
class TrajectoryData:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    infos: List[Dict[str, Any]]
    risks: Optional[np.ndarray] = None
```

Container for RL trajectory data.

**Properties:**
- `episode_length`: Length of trajectory until first done=True

## Algorithms

### conforl.algorithms.base

#### ConformalRLAgent

```python
class ConformalRLAgent(env, risk_controller=None, risk_measure=None, 
                      learning_rate=3e-4, buffer_size=1e6, device="auto")
```

Base class for conformal RL agents with safety guarantees.

**Parameters:**
- `env` (gym.Env): Gymnasium environment
- `risk_controller` (AdaptiveRiskController, optional): Risk controller
- `risk_measure` (RiskMeasure, optional): Risk measure to optimize
- `learning_rate` (float): Learning rate for optimization
- `buffer_size` (int): Size of replay buffer
- `device` (str): Device for computation ('cpu', 'cuda', 'auto')

**Methods:**

##### train(total_timesteps, eval_freq=10000, save_path=None, callback=None)
Train the conformal RL agent.

**Parameters:**
- `total_timesteps` (int): Total training timesteps
- `eval_freq` (int): Frequency of evaluation
- `save_path` (str, optional): Path to save trained model
- `callback` (callable, optional): Optional callback function

##### predict(state, deterministic=False, return_risk_certificate=False)
Predict action with risk-aware policy.

**Parameters:**
- `state`: Environment state
- `deterministic` (bool): Whether to use deterministic policy
- `return_risk_certificate` (bool): Whether to return risk certificate

**Returns:**
- Action or (Action, RiskCertificate) tuple

##### get_risk_certificate(states=None, coverage_guarantee=0.95)
Get formal risk certificate for current policy.

**Parameters:**
- `states` (np.ndarray, optional): Test states for evaluation
- `coverage_guarantee` (float): Desired coverage level

**Returns:**
- `RiskCertificate`: Risk certificate with formal guarantees

### conforl.algorithms.sac

#### ConformaSAC

```python
class ConformaSAC(env, risk_controller=None, risk_measure=None,
                  learning_rate=3e-4, buffer_size=1e6, batch_size=256,
                  tau=0.005, gamma=0.99, alpha=0.2, device="auto")
```

SAC algorithm with conformal risk guarantees.

**Additional Parameters:**
- `batch_size` (int): Batch size for training
- `tau` (float): Soft update coefficient
- `gamma` (float): Discount factor
- `alpha` (float): Temperature parameter for SAC

**Methods:**

##### get_sac_info()
Get SAC-specific information.

**Returns:**
- `dict`: Dictionary with SAC algorithm details

### conforl.algorithms.ppo

#### ConformaPPO

```python
class ConformaPPO(env, risk_controller=None, risk_measure=None,
                  learning_rate=3e-4, buffer_size=2048, batch_size=64,
                  n_epochs=10, gamma=0.99, gae_lambda=0.95, 
                  clip_range=0.2, device="auto")
```

PPO algorithm with conformal risk guarantees.

**Additional Parameters:**
- `n_epochs` (int): Number of epochs per update
- `gae_lambda` (float): GAE lambda parameter
- `clip_range` (float): PPO clipping range

### conforl.algorithms.cql

#### ConformaCQL

```python
class ConformaCQL(env, dataset=None, risk_controller=None,
                  risk_measure=None, learning_rate=3e-4, batch_size=256,
                  tau=0.005, gamma=0.99, alpha=1.0, cql_weight=1.0,
                  device="auto")
```

CQL algorithm with conformal risk guarantees for offline RL.

**Additional Parameters:**
- `dataset` (dict): Offline dataset for training
- `cql_weight` (float): Weight for CQL regularization

**Methods:**

##### train_offline(n_epochs=1000, eval_freq=100, save_path=None)
Train CQL on offline dataset.

**Parameters:**
- `n_epochs` (int): Number of training epochs
- `eval_freq` (int): Frequency of evaluation
- `save_path` (str, optional): Path to save trained model

## Risk Control

### conforl.risk.measures

#### RiskMeasure

```python
class RiskMeasure(name)
```

Base class for risk measures in RL.

**Methods:**

##### compute(trajectory)
Compute risk value for a trajectory.

**Parameters:**
- `trajectory` (TrajectoryData): RL trajectory data

**Returns:**
- `float`: Risk value (higher = more risky)

##### compute_batch(trajectories)
Compute risk for batch of trajectories.

**Parameters:**
- `trajectories` (List[TrajectoryData]): List of RL trajectories

**Returns:**
- `np.ndarray`: Array of risk values

#### SafetyViolationRisk

```python
class SafetyViolationRisk(constraint_key="constraint_violation", 
                         violation_threshold=0.0)
```

Risk measure for safety constraint violations.

#### PerformanceRisk

```python
class PerformanceRisk(target_return, risk_type="shortfall")
```

Risk measure for performance degradation.

**Parameters:**
- `target_return` (float): Minimum acceptable return
- `risk_type` (str): Type of performance risk ('shortfall', 'variance')

### conforl.risk.controllers

#### AdaptiveRiskController

```python
class AdaptiveRiskController(target_risk=0.05, confidence=0.95,
                           window_size=1000, learning_rate=0.01,
                           initial_quantile=0.9)
```

Adaptive risk controller with online quantile updates.

**Parameters:**
- `target_risk` (float): Target risk level
- `confidence` (float): Confidence level for guarantees
- `window_size` (int): Size of sliding window for adaptation
- `learning_rate` (float): Rate of quantile adaptation
- `initial_quantile` (float): Initial quantile estimate

**Methods:**

##### update(trajectory, risk_measure)
Update controller with new trajectory data.

**Parameters:**
- `trajectory` (TrajectoryData): New RL trajectory
- `risk_measure` (RiskMeasure): Risk measure to evaluate trajectory

##### get_risk_bound()
Get current risk bound estimate.

**Returns:**
- `float`: Current risk bound

##### get_certificate()
Generate risk certificate with current guarantees.

**Returns:**
- `RiskCertificate`: Risk certificate with adaptive bounds

## Deployment

### conforl.deploy.pipeline

#### SafeDeploymentPipeline

```python
class SafeDeploymentPipeline(agent, fallback_policy=None, risk_monitor=True,
                           alert_threshold=0.8, max_risk_violations=5,
                           log_dir="./deploy_logs")
```

Production deployment pipeline with safety guarantees.

**Parameters:**
- `agent` (ConformalRLAgent): Trained conformal RL agent
- `fallback_policy` (callable, optional): Safe fallback policy function
- `risk_monitor` (bool): Whether to enable risk monitoring
- `alert_threshold` (float): Risk threshold for alerts
- `max_risk_violations` (int): Max violations before fallback activation
- `log_dir` (str): Directory for deployment logs

**Methods:**

##### deploy(env, num_episodes=100, max_steps_per_episode=1000, eval_callback=None)
Deploy agent in production environment.

**Parameters:**
- `env`: Production environment
- `num_episodes` (int): Number of episodes to run
- `max_steps_per_episode` (int): Max steps per episode
- `eval_callback` (callable, optional): Optional evaluation callback

**Returns:**
- `dict`: Deployment statistics and results

##### emergency_stop(reason="Manual emergency stop")
Emergency stop of deployment.

**Parameters:**
- `reason` (str): Reason for emergency stop

## Monitoring

### conforl.monitoring.metrics

#### MetricsCollector

```python
class MetricsCollector(buffer_size=10000, aggregation_interval=60.0,
                      auto_export=True)
```

Thread-safe metrics collector with automatic aggregation.

**Methods:**

##### record(metric_name, value, tags=None, timestamp=None, metadata=None)
Record a metric value.

##### increment(metric_name, value=1, tags=None)
Increment a counter metric.

##### gauge(metric_name, value, tags=None)
Record a gauge metric (current value).

##### histogram(metric_name, value, tags=None)
Record a histogram metric (for timing, sizes, etc.).

##### timer(metric_name, tags=None)
Context manager for timing operations.

### conforl.monitoring.adaptive

#### SelfImprovingAgent

```python
class SelfImprovingAgent(base_agent, performance_tracker=None,
                        improvement_threshold=0.05, evaluation_window=100)
```

Self-improving agent that continuously optimizes its own performance.

**Parameters:**
- `base_agent`: Base RL agent to wrap
- `performance_tracker` (PerformanceTracker, optional): Performance tracker
- `improvement_threshold` (float): Minimum improvement to trigger adaptation
- `evaluation_window` (int): Window size for performance evaluation

## Utilities

### conforl.utils.validation

#### validate_config(config)
Validate configuration parameters.

**Parameters:**
- `config` (dict): Configuration dictionary

**Returns:**
- `dict`: Validated and sanitized configuration

**Raises:**
- `ConfigurationError`: If configuration is invalid

#### validate_environment(env)
Validate Gymnasium environment compatibility.

**Parameters:**
- `env` (gym.Env): Gymnasium environment to validate

**Raises:**
- `EnvironmentError`: If environment is incompatible

#### validate_dataset(dataset)
Validate offline RL dataset.

**Parameters:**
- `dataset` (dict): Dataset dictionary

**Returns:**
- `dict`: Validated dataset

**Raises:**
- `DataError`: If dataset is invalid

### conforl.utils.security

#### sanitize_input(input_value, input_type="string", max_length=None, allowed_chars=None)
Sanitize user input to prevent injection attacks.

**Parameters:**
- `input_value`: Input value to sanitize
- `input_type` (str): Expected input type ('string', 'number', 'path')
- `max_length` (int, optional): Maximum allowed length for strings
- `allowed_chars` (str, optional): Regex pattern of allowed characters

**Returns:**
- Sanitized input value

**Raises:**
- `SecurityError`: If input fails security checks

## Command Line Interface

### conforl.cli

The ConfoRL command-line interface provides several commands:

#### train
Train a conformal RL agent.

```bash
conforl train --algorithm sac --env CartPole-v1 --timesteps 100000
```

**Options:**
- `--algorithm, -a`: RL algorithm to use (sac, ppo, td3, cql)
- `--env, -e`: Environment name or path
- `--timesteps, -t`: Total training timesteps
- `--save-path, -s`: Path to save trained model
- `--target-risk`: Target risk level (0-1)
- `--confidence`: Confidence level (0-1)
- `--dataset`: Dataset path for offline RL (CQL)

#### evaluate
Evaluate a trained agent.

```bash
conforl evaluate --model ./models/agent --env CartPole-v1 --episodes 10
```

**Options:**
- `--model, -m`: Path to trained model
- `--env, -e`: Environment name
- `--episodes, -n`: Number of evaluation episodes
- `--render`: Render environment during evaluation

#### deploy
Deploy agent in production.

```bash
conforl deploy --model ./models/agent --env CartPole-v1 --monitor
```

**Options:**
- `--model, -m`: Path to trained model
- `--env, -e`: Environment name
- `--episodes, -n`: Number of deployment episodes
- `--monitor`: Enable risk monitoring
- `--fallback-policy`: Path to fallback policy

#### certificate
Generate risk certificate.

```bash
conforl certificate --model ./models/agent --coverage 0.95
```

**Options:**
- `--model, -m`: Path to trained model
- `--coverage`: Desired coverage level

## Examples

### Basic Usage

```python
import conforl
import gymnasium as gym

# Create environment
env = gym.make('CartPole-v1')

# Initialize conformal risk controller
risk_controller = conforl.AdaptiveRiskController(
    target_risk=0.05,
    confidence=0.95
)

# Create ConfoRL agent
agent = conforl.ConformaSAC(
    env=env,
    risk_controller=risk_controller
)

# Train with safety guarantees
agent.train(total_timesteps=100000)

# Deploy with certified risk bounds
state, info = env.reset()
action, risk_cert = agent.predict(state, return_risk_certificate=True)
print(f"Risk bound: {risk_cert.risk_bound}")
```

### Offline RL

```python
# Load offline dataset
dataset = {
    'observations': observations,
    'actions': actions,
    'rewards': rewards,
    'next_observations': next_observations,
    'terminals': terminals
}

# Train with offline conformal bounds
offline_agent = conforl.ConformaCQL(
    env=env,
    dataset=dataset,
    risk_controller=risk_controller
)

offline_agent.train_offline(n_epochs=1000)
```

### Production Deployment

```python
# Setup deployment pipeline
pipeline = conforl.SafeDeploymentPipeline(
    agent=trained_agent,
    risk_monitor=True,
    fallback_policy=safe_policy
)

# Deploy with monitoring
results = pipeline.deploy(
    env=production_env,
    num_episodes=1000
)

print(f"Safety interventions: {results['safety_interventions']}")
```