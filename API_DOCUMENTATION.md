# ConfoRL API Documentation

## Installation

```bash
pip install conforl
```

## Quick Start

```python
import conforl
from conforl.algorithms import ConformaSAC
from conforl.risk import AdaptiveRiskController

# Create environment (gymnasium compatible)
import gymnasium as gym
env = gym.make('CartPole-v1')

# Create risk controller
risk_controller = AdaptiveRiskController(
    target_risk=0.05,
    confidence=0.95
)

# Create conformal RL agent
agent = ConformaSAC(
    env=env,
    risk_controller=risk_controller
)

# Train with safety guarantees
agent.train(total_timesteps=50000)

# Deploy with risk certificates
state, _ = env.reset()
action, certificate = agent.predict(state, return_risk_certificate=True)

print(f"Action: {action}")
print(f"Risk bound: {certificate.risk_bound:.4f}")
print(f"Confidence: {certificate.confidence:.4f}")
```

## Core API Reference

### Conformal Prediction

#### SplitConformalPredictor

```python
class SplitConformalPredictor:
    """Split conformal predictor for distribution-free uncertainty quantification."""
    
    def __init__(self, coverage: float = 0.95):
        """Initialize conformal predictor.
        
        Args:
            coverage: Target coverage level (1-α)
        """
    
    def calibrate(self, calibration_scores: List[float]) -> None:
        """Calibrate conformal predictor with nonconformity scores.
        
        Args:
            calibration_scores: List of nonconformity scores from calibration data
        """
    
    def predict(self, test_predictions: Union[float, List[float]]) -> ConformalSet:
        """Generate conformal prediction set.
        
        Args:
            test_predictions: Point predictions for test inputs
            
        Returns:
            ConformalSet with prediction intervals and coverage guarantees
        """
    
    def get_prediction_interval(self, predictions: List[float]) -> Tuple[List[float], List[float]]:
        """Get prediction intervals for given predictions.
        
        Args:
            predictions: Point predictions
            
        Returns:
            Tuple of (lower_bounds, upper_bounds)
        """
```

#### Example Usage

```python
from conforl.core.conformal import SplitConformalPredictor

# Create predictor
predictor = SplitConformalPredictor(coverage=0.95)

# Calibrate with scores
calibration_scores = [0.1, 0.2, 0.3, 0.4, 0.5]
predictor.calibrate(calibration_scores)

# Make predictions
test_predictions = [0.6, 0.7, 0.8]
conformal_set = predictor.predict(test_predictions)

print(f"Coverage: {conformal_set.coverage}")
print(f"Prediction intervals: {conformal_set.intervals}")
```

### Risk Control

#### AdaptiveRiskController

```python
class AdaptiveRiskController:
    """Adaptive risk controller with online quantile updates."""
    
    def __init__(
        self,
        target_risk: float = 0.05,
        confidence: float = 0.95,
        window_size: int = 1000,
        learning_rate: float = 0.01
    ):
        """Initialize adaptive risk controller.
        
        Args:
            target_risk: Target risk level (α)
            confidence: Confidence level (1-α)
            window_size: Size of sliding window for adaptation
            learning_rate: Rate of quantile adaptation
        """
    
    def update(self, trajectory: TrajectoryData, risk_measure: RiskMeasure) -> None:
        """Update risk controller with new trajectory data.
        
        Args:
            trajectory: Trajectory data from environment interaction
            risk_measure: Risk measure for computing nonconformity scores
        """
    
    def get_certificate(self) -> RiskCertificate:
        """Generate risk certificate with current guarantees.
        
        Returns:
            RiskCertificate with risk bounds and coverage information
        """
    
    def get_risk_bound(self) -> float:
        """Get current risk bound estimate.
        
        Returns:
            Current risk bound (upper bound on violation probability)
        """
```

#### Example Usage

```python
from conforl.risk.controllers import AdaptiveRiskController
from conforl.risk.measures import SafetyViolationRisk
from conforl.core.types import TrajectoryData

# Create risk controller
controller = AdaptiveRiskController(
    target_risk=0.05,
    confidence=0.95,
    window_size=100
)

# Create risk measure
risk_measure = SafetyViolationRisk()

# Simulate trajectory data
trajectory = TrajectoryData(
    states=[[0.1, 0.2], [0.3, 0.4]],
    actions=[0, 1],
    rewards=[1.0, 1.0],
    dones=[False, True],
    infos=[{}, {}]
)

# Update controller
controller.update(trajectory, risk_measure)

# Get risk certificate
certificate = controller.get_certificate()
print(f"Risk bound: {certificate.risk_bound:.4f}")
print(f"Confidence: {certificate.confidence:.4f}")
```

### Algorithms

#### ConformaSAC

```python
class ConformaSAC:
    """Soft Actor-Critic with conformal risk control."""
    
    def __init__(
        self,
        env: gym.Env,
        risk_controller: AdaptiveRiskController,
        learning_rate: float = 3e-4,
        buffer_size: int = 1000000,
        batch_size: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005
    ):
        """Initialize ConformaSAC agent.
        
        Args:
            env: Gymnasium environment
            risk_controller: Risk controller for safety guarantees
            learning_rate: Learning rate for neural networks
            buffer_size: Size of replay buffer
            batch_size: Batch size for training
            gamma: Discount factor
            tau: Soft update coefficient
        """
    
    def train(
        self,
        total_timesteps: int,
        eval_freq: int = 10000,
        save_path: Optional[str] = None
    ) -> None:
        """Train the agent with safety guarantees.
        
        Args:
            total_timesteps: Total number of training timesteps
            eval_freq: Frequency of evaluation episodes
            save_path: Path to save trained model
        """
    
    def predict(
        self,
        state: np.ndarray,
        return_risk_certificate: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, RiskCertificate]]:
        """Predict action with optional risk certificate.
        
        Args:
            state: Current environment state
            return_risk_certificate: Whether to return risk certificate
            
        Returns:
            Action or tuple of (action, risk_certificate)
        """
    
    def save(self, path: str) -> None:
        """Save trained model.
        
        Args:
            path: Path to save model
        """
    
    def load(self, path: str) -> None:
        """Load trained model.
        
        Args:
            path: Path to load model from
        """
```

#### Example Usage

```python
from conforl.algorithms import ConformaSAC
from conforl.risk.controllers import AdaptiveRiskController
import gymnasium as gym

# Create environment
env = gym.make('CartPole-v1')

# Create risk controller
risk_controller = AdaptiveRiskController(target_risk=0.05, confidence=0.95)

# Create agent
agent = ConformaSAC(
    env=env,
    risk_controller=risk_controller,
    learning_rate=3e-4
)

# Train agent
agent.train(total_timesteps=50000, eval_freq=5000)

# Use trained agent
state, _ = env.reset()
action, certificate = agent.predict(state, return_risk_certificate=True)

print(f"Action: {action}")
print(f"Risk bound: {certificate.risk_bound:.4f}")

# Save model
agent.save("./models/conforma_sac")
```

### Data Types

#### RiskCertificate

```python
@dataclass
class RiskCertificate:
    """Certificate providing risk guarantees for an action."""
    
    risk_bound: float           # Upper bound on risk
    confidence: float           # Confidence level (1-α)
    coverage_guarantee: float   # Actual coverage achieved
    method: str                # Conformal method used
    sample_size: int           # Calibration sample size
    timestamp: float           # Generation timestamp
```

#### TrajectoryData

```python
@dataclass
class TrajectoryData:
    """Container for trajectory data from environment interaction."""
    
    states: List[np.ndarray]      # Sequence of states
    actions: List[np.ndarray]     # Sequence of actions
    rewards: List[float]          # Sequence of rewards
    dones: List[bool]            # Sequence of done flags
    infos: List[Dict[str, Any]]  # Sequence of info dictionaries
```

#### ConformalSet

```python
@dataclass
class ConformalSet:
    """Conformal prediction set with coverage guarantees."""
    
    intervals: List[Tuple[float, float]]  # Prediction intervals
    coverage: float                       # Target coverage level
    quantile: float                      # Conformal quantile used
    method: str                          # Conformal method
    size: int                           # Number of predictions
```

### Utilities

#### Logging

```python
from conforl.utils.logging import get_logger

# Get logger for current module
logger = get_logger(__name__)

# Log with different levels
logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")
```

#### Validation

```python
from conforl.utils.validation import validate_config, validate_risk_parameters

# Validate configuration
config = {
    'learning_rate': 3e-4,
    'batch_size': 256,
    'target_risk': 0.05
}
validated_config = validate_config(config)

# Validate risk parameters
validate_risk_parameters(target_risk=0.05, confidence=0.95)
```

#### Security

```python
from conforl.utils.security import SecurityContext, sanitize_config_dict

# Use security context
with SecurityContext("training", "user123"):
    # Perform security-sensitive operations
    sanitized_config = sanitize_config_dict(user_config)
```

## CLI Usage

### Training

```bash
# Train ConformaSAC on CartPole
conforl train --algorithm sac --env CartPole-v1 --timesteps 50000

# Train with custom risk parameters
conforl train --algorithm sac --env CartPole-v1 --timesteps 50000 \
    --target-risk 0.05 --confidence 0.95

# Train PPO with configuration file
conforl train --algorithm ppo --env LunarLander-v2 --config config.yaml
```

### Evaluation

```bash
# Evaluate trained model
conforl evaluate --model ./models/agent --env CartPole-v1 --episodes 10

# Evaluate with risk certificates
conforl evaluate --model ./models/agent --env CartPole-v1 \
    --episodes 10 --show-certificates
```

### Deployment

```bash
# Deploy agent with monitoring
conforl deploy --model ./models/agent --env CartPole-v1 --monitor

# Deploy with fallback policy
conforl deploy --model ./models/agent --env CartPole-v1 \
    --fallback-policy ./models/safe_policy
```

### Risk Certificates

```bash
# Generate risk certificate for model
conforl certificate --model ./models/agent --coverage 0.95

# Generate certificate with custom parameters
conforl certificate --model ./models/agent --coverage 0.99 \
    --sample-size 1000
```

## Configuration

### Environment Variables

```bash
# Logging configuration
export CONFORL_LOG_LEVEL=INFO
export CONFORL_LOG_DIR=./logs

# Performance configuration
export CONFORL_CACHE_SIZE=10000
export CONFORL_MAX_WORKERS=4

# Security configuration
export CONFORL_ENABLE_SECURITY=true
export CONFORL_AUDIT_LOG=true
```

### Configuration Files

#### YAML Configuration

```yaml
# config.yaml
algorithm:
  name: sac
  learning_rate: 3e-4
  buffer_size: 1000000
  batch_size: 256

risk_control:
  target_risk: 0.05
  confidence: 0.95
  window_size: 1000

environment:
  name: CartPole-v1
  max_episode_steps: 500

training:
  total_timesteps: 100000
  eval_freq: 10000
  save_freq: 25000

logging:
  level: INFO
  save_logs: true
  log_dir: ./logs
```

#### JSON Configuration

```json
{
  "algorithm": {
    "name": "sac",
    "learning_rate": 3e-4,
    "buffer_size": 1000000
  },
  "risk_control": {
    "target_risk": 0.05,
    "confidence": 0.95
  },
  "environment": {
    "name": "CartPole-v1"
  }
}
```

## Error Handling

### Custom Exceptions

```python
from conforl.utils.errors import ConfoRLError, ValidationError, RiskControlError

try:
    # ConfoRL operations
    agent.train(total_timesteps=50000)
except ValidationError as e:
    print(f"Validation error: {e}")
except RiskControlError as e:
    print(f"Risk control error: {e}")
except ConfoRLError as e:
    print(f"ConfoRL error: {e}")
```

### Error Codes

- `VALIDATION_ERROR`: Input validation failed
- `CONFORMAL_ERROR`: Conformal prediction failed
- `RISK_CONTROL_ERROR`: Risk control update failed
- `ALGORITHM_ERROR`: Algorithm training/prediction failed
- `DEPLOYMENT_ERROR`: Deployment operation failed

## Performance Tips

### Memory Optimization

```python
# Use smaller window sizes for memory-constrained environments
risk_controller = AdaptiveRiskController(window_size=100)

# Enable memory monitoring
from conforl.optimize.profiler import PerformanceProfiler
profiler = PerformanceProfiler()

with profiler.profile("training"):
    agent.train(total_timesteps=50000)

memory_usage = profiler.get_memory_usage()
print(f"Memory usage: {memory_usage:.2f} MB")
```

### Concurrency

```python
# Enable concurrent processing
from conforl.optimize.concurrent import ConcurrentProcessor

processor = ConcurrentProcessor(max_workers=4)

# Process multiple predictions concurrently
tasks = [state1, state2, state3, state4]
results = processor.execute_concurrent(agent.predict, tasks)
```

### Caching

```python
# Enable adaptive caching
from conforl.optimize.cache import AdaptiveCache

cache = AdaptiveCache(max_size=1000, ttl=3600)

# Cache expensive computations
cache.set("conformal_quantile", quantile_value)
cached_quantile = cache.get("conformal_quantile")
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure ConfoRL is properly installed
2. **Memory Issues**: Reduce window size or enable compression
3. **Performance Issues**: Enable caching and concurrent processing
4. **Convergence Issues**: Adjust learning rates and risk parameters

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable performance profiling
from conforl.optimize.profiler import PerformanceProfiler
profiler = PerformanceProfiler()

# Monitor resource usage
from conforl.utils.health import HealthMonitor
monitor = HealthMonitor()
health_status = monitor.check_health()
```

### Support

- **Documentation**: https://conforl.readthedocs.io
- **GitHub Issues**: https://github.com/terragon/conforl/issues
- **Discord**: https://discord.gg/conforl
- **Email**: support@terragon.ai

## Examples

See the `examples/` directory for complete examples:

- `basic_usage.py`: Getting started with ConfoRL
- `custom_environment.py`: Using custom environments
- `production_deployment.py`: Production deployment example
- `research_benchmark.py`: Research benchmarking example
