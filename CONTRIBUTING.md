# Contributing to ConfoRL

Thank you for your interest in contributing to ConfoRL! This document provides guidelines for contributing to the project.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Code Standards](#code-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Community](#community)

## Getting Started

ConfoRL is an open-source project that welcomes contributions from researchers, developers, and practitioners interested in safe reinforcement learning. We particularly welcome contributions in the following areas:

- 🔬 **New conformal techniques** for RL
- 🏗️ **Safety-critical environment** implementations
- 📊 **Theoretical analysis** and proofs
- 🚀 **Real-world deployment** case studies
- 📚 **Documentation** and tutorials
- 🐛 **Bug fixes** and performance improvements

## Development Setup

### Prerequisites

- Python 3.8+
- Git
- Docker (optional, for containerized development)

### Local Development

1. **Fork and clone the repository:**

```bash
git clone https://github.com/yourusername/conforl-lab.git
cd conforl-lab
```

2. **Create a virtual environment:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install development dependencies:**

```bash
pip install -e .[dev]
```

4. **Install pre-commit hooks:**

```bash
pre-commit install
```

5. **Verify installation:**

```bash
python -c "import conforl; print('ConfoRL installed successfully!')"
pytest tests/ -v
```

### Docker Development (Optional)

```bash
# Build development container
docker build -t conforl-dev -f Dockerfile .

# Run development container
docker run -it --rm -v $(pwd):/app conforl-dev bash
```

## Contributing Guidelines

### Types of Contributions

#### 🐛 Bug Reports
- Use the GitHub issue tracker
- Include minimal reproducible example
- Specify Python version and dependencies
- Describe expected vs actual behavior

#### 💡 Feature Requests
- Check existing issues for duplicates
- Describe the use case and motivation
- Provide implementation suggestions if possible
- Consider backward compatibility

#### 🔬 Research Contributions
- New conformal prediction methods
- Theoretical analysis and proofs
- Empirical studies and benchmarks
- Novel risk measures

#### 🏗️ Code Contributions
- Bug fixes and performance improvements
- New algorithms and techniques
- Infrastructure and tooling
- Documentation and examples

### Before You Start

1. **Check existing issues** to avoid duplicate work
2. **Create an issue** to discuss significant changes
3. **Fork the repository** and create a feature branch
4. **Follow coding standards** described below
5. **Write tests** for new functionality
6. **Update documentation** as needed

## Code Standards

### Python Style

We follow PEP 8 with some modifications:

```python
# Use type hints
def compute_risk(trajectory: TrajectoryData) -> float:
    """Compute risk value for trajectory."""
    pass

# Use descriptive variable names
risk_controller = AdaptiveRiskController()
conformal_predictor = SplitConformalPredictor()

# Use docstrings (Google style)
def train_agent(agent: ConformalRLAgent, timesteps: int) -> None:
    """Train conformal RL agent.
    
    Args:
        agent: ConfoRL agent to train
        timesteps: Number of training timesteps
        
    Raises:
        TrainingError: If training fails
    """
    pass
```

### Code Organization

```
conforl/
├── core/           # Core conformal prediction
├── algorithms/     # RL algorithms with conformal guarantees
├── risk/          # Risk measures and controllers
├── deploy/        # Production deployment
├── utils/         # Utility functions
├── optimize/      # Performance optimization
├── i18n/          # Internationalization
└── monitoring/    # Metrics and monitoring
```

### Error Handling

Use custom exception classes for better error messages:

```python
from conforl.utils.errors import ConfigurationError, ValidationError

def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(config, dict):
        raise ConfigurationError("Configuration must be a dictionary")
    
    if 'learning_rate' not in config:
        raise ConfigurationError("Missing learning_rate parameter", "learning_rate")
    
    return config
```

### Security

- Validate all user inputs
- Sanitize file paths
- Use secure random number generation
- Log security events appropriately

```python
from conforl.utils.security import sanitize_input, SecurityError

def load_model(path: str) -> Model:
    try:
        safe_path = sanitize_input(path, "path")
        return load_from_path(safe_path)
    except SecurityError as e:
        logger.warning(f"Security violation: {e}")
        raise
```

### Performance

- Use appropriate data structures
- Implement caching where beneficial
- Consider memory usage for large datasets
- Profile performance-critical code

```python
from conforl.optimize.cache import AdaptiveCache

# Use caching for expensive computations
cache = AdaptiveCache()

@cache.cached_computation
def expensive_risk_computation(trajectory):
    # Complex computation here
    return risk_value
```

## Testing

### Test Organization

```
tests/
├── test_core.py           # Core functionality tests
├── test_algorithms.py     # Algorithm tests
├── test_risk.py          # Risk measure tests
├── test_utils.py         # Utility function tests
├── integration/          # Integration tests
└── fixtures/             # Test fixtures and data
```

### Writing Tests

```python
import pytest
import numpy as np
from conforl.algorithms import ConformaSAC
from conforl.risk.controllers import AdaptiveRiskController

class TestConformaSAC:
    """Test cases for ConformaSAC algorithm."""
    
    def test_initialization(self, simple_env):
        """Test agent initialization."""
        risk_controller = AdaptiveRiskController()
        agent = ConformaSAC(env=simple_env, risk_controller=risk_controller)
        
        assert agent.risk_controller == risk_controller
        assert agent.total_timesteps == 0
    
    def test_prediction(self, simple_env):
        """Test action prediction."""
        agent = ConformaSAC(env=simple_env)
        state = np.array([0.1, 0.2, 0.3, 0.4])
        
        action = agent.predict(state)
        assert action is not None
        
        action, cert = agent.predict(state, return_risk_certificate=True)
        assert action is not None
        assert cert is not None
    
    @pytest.mark.slow
    def test_training(self, simple_env):
        """Test training process."""
        agent = ConformaSAC(env=simple_env)
        agent.train(total_timesteps=1000)
        
        assert agent.total_timesteps == 1000
        assert agent.episode_count > 0
```

### Test Requirements

- **Minimum 85% code coverage**
- **Fast unit tests** (< 1 second each)
- **Integration tests** for end-to-end workflows
- **Mock external dependencies** (environments, networks)
- **Test edge cases** and error conditions

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=conforl --cov-report=html

# Run specific test file
pytest tests/test_algorithms.py -v

# Run slow tests
pytest -m slow

# Run tests in parallel
pytest -n auto
```

## Documentation

### Docstring Format

Use Google-style docstrings:

```python
def adaptive_risk_control(
    trajectory: TrajectoryData,
    risk_measure: RiskMeasure,
    target_risk: float = 0.05
) -> RiskCertificate:
    """Perform adaptive risk control on trajectory.
    
    This function implements adaptive conformal risk control as described
    in [Reference]. It provides finite-sample guarantees on risk bounds.
    
    Args:
        trajectory: RL trajectory data containing states, actions, rewards
        risk_measure: Risk measure to evaluate trajectory
        target_risk: Target risk level between 0 and 1
        
    Returns:
        Risk certificate with formal guarantees
        
    Raises:
        ValidationError: If trajectory data is invalid
        RiskControlError: If risk control fails
        
    Example:
        >>> trajectory = TrajectoryData(states, actions, rewards, dones, infos)
        >>> risk_measure = SafetyViolationRisk()
        >>> certificate = adaptive_risk_control(trajectory, risk_measure)
        >>> print(f"Risk bound: {certificate.risk_bound}")
    """
    pass
```

### API Documentation

- Document all public functions and classes
- Include parameter types and descriptions
- Provide usage examples
- Document exceptions that may be raised

### User Documentation

- **User Guide**: High-level usage patterns
- **API Reference**: Detailed API documentation
- **Tutorials**: Step-by-step guides
- **Examples**: Working code examples

### Contributing to Documentation

```bash
# Build documentation locally
cd docs/
make html

# Serve documentation
python -m http.server 8000 -d _build/html/
```

## Submitting Changes

### Pull Request Process

1. **Create a feature branch:**

```bash
git checkout -b feature/new-conformal-method
```

2. **Make your changes:**
   - Follow code standards
   - Add tests
   - Update documentation
   - Run tests locally

3. **Commit your changes:**

```bash
git add .
git commit -m "feat: implement new conformal method for time series

- Add TimeSeriesConformalPredictor class
- Implement adaptive quantile estimation
- Add comprehensive tests and documentation
- Update API reference

Closes #123"
```

4. **Push to your fork:**

```bash
git push origin feature/new-conformal-method
```

5. **Create a pull request:**
   - Use descriptive title and description
   - Reference related issues
   - Include testing instructions
   - Add screenshots/examples if applicable

### Commit Message Format

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): brief description

Longer description if needed

- Bullet points for details
- Reference issues with #123
- Breaking changes marked with BREAKING CHANGE:
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `style`: Code style changes
- `ci`: CI/CD changes

### Review Process

1. **Automated checks** must pass:
   - Tests
   - Code coverage (≥85%)
   - Linting
   - Security scanning

2. **Manual review** by maintainers:
   - Code quality and style
   - Test completeness
   - Documentation updates
   - Architectural consistency

3. **Approval and merge**:
   - At least one maintainer approval
   - All conversations resolved
   - CI passes

## Community

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Discord/Slack**: Real-time chat (link in README)
- **Mailing List**: Announcements and updates

### Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

### Getting Help

- **Documentation**: Check docs/ directory
- **Examples**: See examples/ directory
- **Issues**: Search existing issues
- **Discussions**: Ask questions in GitHub Discussions

### Recognition

We value all contributions and will recognize contributors in:
- CONTRIBUTORS.md file
- Release notes
- Conference presentations (with permission)
- Academic publications (when appropriate)

## Research Contributions

### Theoretical Contributions

If you're contributing theoretical results:

1. **Provide formal statements** of theorems/lemmas
2. **Include proofs** or references to proofs
3. **Implement algorithms** based on theory
4. **Add empirical validation**
5. **Document assumptions** and limitations

Example:

```python
def hoeffding_bound_risk_certificate(
    samples: np.ndarray,
    confidence: float = 0.95
) -> RiskCertificate:
    """Generate risk certificate using Hoeffding's inequality.
    
    Theorem: For i.i.d. samples X_1, ..., X_n from [0,1], with probability
    ≥ 1-δ, the empirical mean satisfies:
    |μ_n - μ| ≤ sqrt(log(2/δ) / (2n))
    
    Proof: Follows directly from Hoeffding's inequality.
    
    Args:
        samples: I.i.d. samples from [0,1]
        confidence: Confidence level (1-δ)
        
    Returns:
        Risk certificate with Hoeffding bound
    """
    n = len(samples)
    delta = 1 - confidence
    empirical_mean = np.mean(samples)
    margin = np.sqrt(np.log(2/delta) / (2*n))
    
    return RiskCertificate(
        risk_bound=empirical_mean + margin,
        confidence=confidence,
        coverage_guarantee=confidence,
        method="hoeffding",
        sample_size=n
    )
```

### Empirical Studies

For empirical contributions:

1. **Use standardized benchmarks** when possible
2. **Report statistical significance** with confidence intervals
3. **Include computational requirements** and timing
4. **Provide reproducible code** and data
5. **Document hyperparameters** and experimental setup

### New Environments

When contributing safety-critical environments:

1. **Implement clear safety constraints**
2. **Provide risk ground truth** when possible
3. **Document environment dynamics**
4. **Include visualization** and rendering
5. **Add comprehensive tests**

## Development Roadmap

### Current Priorities

1. **Algorithm implementations**: More RL algorithms with conformal guarantees
2. **Theoretical extensions**: New conformal techniques for RL
3. **Real-world applications**: Case studies in safety-critical domains
4. **Performance optimization**: Scaling to larger problems
5. **Documentation**: Comprehensive guides and tutorials

### Long-term Goals

1. **Industry adoption**: Production deployments in safety-critical systems
2. **Academic integration**: Course materials and textbook chapters
3. **Standardization**: Common APIs for conformal RL
4. **Ecosystem growth**: Community-driven extensions and tools

## Questions?

If you have questions about contributing, please:

1. Check existing documentation
2. Search GitHub issues and discussions
3. Create a new issue with the "question" label
4. Reach out to maintainers

We appreciate your interest in making reinforcement learning safer and more reliable through conformal prediction!

---

*This contributing guide is a living document. Please suggest improvements by opening an issue or pull request.*