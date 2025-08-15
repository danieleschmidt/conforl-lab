# Contributing to ConfoRL

Thank you for your interest in contributing to ConfoRL! This guide will help you get started with contributing to our open-source project.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Contribution Guidelines](#contribution-guidelines)
5. [Code Standards](#code-standards)
6. [Testing](#testing)
7. [Documentation](#documentation)
8. [Submitting Changes](#submitting-changes)
9. [Review Process](#review-process)
10. [Community](#community)

## Code of Conduct

ConfoRL follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). Please read and follow these guidelines in all interactions.

## Getting Started

### Types of Contributions

We welcome several types of contributions:

- 🐛 **Bug Reports**: Report issues you encounter
- 🚀 **Feature Requests**: Suggest new features or improvements
- 📝 **Documentation**: Improve or add documentation
- 🧪 **Testing**: Add or improve test coverage
- 💻 **Code**: Implement new features or fix bugs
- 🔬 **Research**: Contribute algorithms or benchmarks
- 🎯 **Examples**: Add usage examples or tutorials

### Before Contributing

1. Check existing [issues](https://github.com/terragon/conforl/issues) and [pull requests](https://github.com/terragon/conforl/pulls)
2. Read through our [documentation](https://conforl.readthedocs.io)
3. Familiarize yourself with the codebase structure
4. Join our [Discord community](https://discord.gg/conforl) for discussions

## Development Setup

### Prerequisites

- Python 3.9+ (we test on 3.9, 3.10, 3.11, 3.12)
- Git
- Virtual environment tool (venv or conda)

### Local Development

1. **Fork and Clone**
   ```bash
   # Fork on GitHub, then clone your fork
   git clone https://github.com/YOUR_USERNAME/conforl.git
   cd conforl
   
   # Add upstream remote
   git remote add upstream https://github.com/terragon/conforl.git
   ```

2. **Set Up Environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install in development mode
   pip install -e .
   pip install -r requirements-dev.txt
   ```

3. **Verify Setup**
   ```bash
   # Run tests to ensure everything works
   pytest tests/ -v
   
   # Check code style
   black --check .
   mypy conforl/
   ```

### Development Tools

We use several tools to maintain code quality:

- **Black**: Code formatting
- **MyPy**: Type checking
- **Pytest**: Testing framework
- **Coverage**: Test coverage analysis
- **Pre-commit**: Git hooks for quality checks

Install pre-commit hooks:
```bash
pre-commit install
```

## Contribution Guidelines

### Issue Guidelines

**Bug Reports:**
- Use the bug report template
- Include minimal reproduction steps
- Provide environment details
- Add relevant logs or error messages

**Feature Requests:**
- Use the feature request template
- Clearly describe the use case
- Explain why it benefits the community
- Consider implementation complexity

### Pull Request Guidelines

1. **Branch from main**
   ```bash
   git checkout main
   git pull upstream main
   git checkout -b feature/your-feature-name
   ```

2. **Make focused changes**
   - One feature/fix per PR
   - Keep changes small and reviewable
   - Include tests for new functionality

3. **Follow commit conventions**
   ```bash
   git commit -m "feat(algorithms): add ConformaDQN implementation"
   git commit -m "fix(risk): handle edge case in quantile calculation"
   git commit -m "docs(api): update ConformaSAC documentation"
   ```

4. **Keep PR updated**
   ```bash
   git fetch upstream
   git rebase upstream/main
   git push origin feature/your-feature-name --force-with-lease
   ```

### Commit Message Format

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(algorithms): implement ConformaA2C algorithm

Add Advantage Actor-Critic algorithm with conformal risk control.
Includes adaptive risk controller integration and comprehensive tests.

Closes #123

fix(risk): handle division by zero in quantile calculation

Previously, empty calibration sets could cause division by zero errors.
Now returns default quantile value when no calibration data is available.

docs(deployment): update Kubernetes configuration examples

Update resource limits and add security context configuration
for production deployments.
```

## Code Standards

### Python Style

We follow [PEP 8](https://pep8.org/) with these specific requirements:

- **Line length**: 88 characters (Black default)
- **Imports**: Use isort with Black-compatible settings
- **Type hints**: Required for all public functions
- **Docstrings**: Google-style docstrings

### Code Organization

```
conforl/
├── algorithms/          # RL algorithm implementations
├── core/               # Core conformal prediction
├── risk/              # Risk control and measurement
├── utils/             # Utility functions
├── deploy/            # Deployment utilities
├── optimize/          # Performance optimizations
├── benchmarks/        # Benchmarking framework
└── examples/          # Usage examples
```

### Naming Conventions

- **Classes**: PascalCase (`ConformaSAC`, `RiskController`)
- **Functions/Variables**: snake_case (`get_risk_bound`, `target_risk`)
- **Constants**: UPPER_SNAKE_CASE (`DEFAULT_CONFIDENCE`, `MAX_BUFFER_SIZE`)
- **Private members**: Leading underscore (`_internal_method`)

### Type Hints

All public functions must include type hints:

```python
from typing import List, Optional, Tuple, Union
import numpy as np

def predict_with_confidence(
    state: np.ndarray,
    return_certificate: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, RiskCertificate]]:
    """Predict action with optional risk certificate.
    
    Args:
        state: Current environment state
        return_certificate: Whether to return risk certificate
        
    Returns:
        Action or tuple of (action, certificate)
    """
```

### Documentation

All public APIs require docstrings:

```python
def adaptive_quantile_update(
    current_quantile: float,
    nonconformity_score: float,
    learning_rate: float = 0.01
) -> float:
    """Update conformal quantile using adaptive learning.
    
    This function implements the adaptive quantile update rule from
    [Gibbs & Candes, 2021] for online conformal prediction.
    
    Args:
        current_quantile: Current quantile estimate
        nonconformity_score: New nonconformity score
        learning_rate: Learning rate for adaptation
        
    Returns:
        Updated quantile estimate
        
    Examples:
        >>> quantile = 0.5
        >>> score = 0.7
        >>> new_quantile = adaptive_quantile_update(quantile, score)
        >>> print(f"Updated quantile: {new_quantile:.3f}")
        Updated quantile: 0.502
        
    References:
        Gibbs, I., & Candes, E. (2021). Adaptive conformal inference under
        distribution shift. Advances in Neural Information Processing Systems.
    """
```

## Testing

### Test Structure

```
tests/
├── unit/              # Unit tests for individual components
├── integration/       # Integration tests for workflows
├── benchmarks/        # Performance and accuracy benchmarks
├── fixtures/          # Test data and utilities
└── conftest.py        # Pytest configuration
```

### Writing Tests

1. **Test file naming**: `test_<module_name>.py`
2. **Test function naming**: `test_<functionality>`
3. **Use fixtures**: For common setup
4. **Mock external dependencies**: Use unittest.mock

Example test:

```python
import pytest
import numpy as np
from unittest.mock import Mock

from conforl.risk.controllers import AdaptiveRiskController
from conforl.core.types import TrajectoryData

class TestAdaptiveRiskController:
    """Test suite for AdaptiveRiskController."""
    
    @pytest.fixture
    def risk_controller(self):
        """Create risk controller for testing."""
        return AdaptiveRiskController(
            target_risk=0.05,
            confidence=0.95
        )
    
    @pytest.fixture
    def sample_trajectory(self):
        """Create sample trajectory data."""
        return TrajectoryData(
            states=np.random.random((10, 4)),
            actions=np.random.randint(0, 2, 10),
            rewards=np.random.random(10),
            dones=np.array([False] * 9 + [True]),
            infos=[{} for _ in range(10)]
        )
    
    def test_initialization(self, risk_controller):
        """Test controller initialization."""
        assert risk_controller.target_risk == 0.05
        assert risk_controller.confidence == 0.95
        assert risk_controller.current_quantile > 0
    
    def test_update_with_trajectory(self, risk_controller, sample_trajectory):
        """Test updating controller with trajectory data."""
        initial_quantile = risk_controller.current_quantile
        
        # Mock risk measure
        risk_measure = Mock()
        risk_measure.compute_scores.return_value = [0.1, 0.2, 0.3]
        
        # Update controller
        risk_controller.update(sample_trajectory, risk_measure)
        
        # Verify quantile was updated
        assert risk_controller.current_quantile != initial_quantile
    
    def test_get_certificate(self, risk_controller):
        """Test risk certificate generation."""
        certificate = risk_controller.get_certificate()
        
        assert 0 <= certificate.risk_bound <= 1
        assert 0 <= certificate.confidence <= 1
        assert certificate.method == "adaptive_conformal"
    
    @pytest.mark.parametrize("target_risk", [0.01, 0.05, 0.1])
    def test_different_risk_levels(self, target_risk):
        """Test controller with different risk levels."""
        controller = AdaptiveRiskController(target_risk=target_risk)
        assert controller.target_risk == target_risk
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_risk_controllers.py

# Run with coverage
pytest --cov=conforl --cov-report=html

# Run performance benchmarks
pytest tests/benchmarks/ --benchmark-only
```

## Documentation

### Types of Documentation

1. **API Documentation**: Docstrings in code
2. **User Guides**: How-to guides and tutorials
3. **Examples**: Jupyter notebooks and scripts
4. **Research Documentation**: Academic papers and reports

### Building Documentation

```bash
# Install documentation dependencies
pip install -r requirements-docs.txt

# Build documentation
cd docs/
make html

# Serve locally
python -m http.server 8000 -d _build/html
```

### Writing Documentation

- Use clear, concise language
- Include code examples
- Add cross-references
- Test all code examples
- Update README for major changes

## Submitting Changes

### Before Submitting

1. **Run quality checks**
   ```bash
   # Format code
   black .
   isort .
   
   # Type checking
   mypy conforl/
   
   # Run tests
   pytest tests/ -v
   
   # Check coverage
   pytest --cov=conforl --cov-fail-under=85
   ```

2. **Update documentation**
   - Add docstrings for new functions
   - Update API documentation
   - Add examples if applicable

3. **Add tests**
   - Unit tests for new functionality
   - Integration tests for workflows
   - Update existing tests if needed

### Pull Request Process

1. **Create PR from feature branch**
2. **Fill out PR template completely**
3. **Link related issues**
4. **Add screenshots/demos if applicable**
5. **Request review from maintainers**

### PR Template

```markdown
## Description

Brief description of changes and motivation.

## Type of Change

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing

- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] New tests added for new functionality

## Checklist

- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] Changelog updated (if applicable)

## Screenshots/Demos

(If applicable)

## Additional Context

Any additional information or context.
```

## Review Process

### Review Criteria

Reviewers will check:

1. **Code Quality**
   - Follows style guidelines
   - Well-structured and readable
   - Appropriate abstractions

2. **Functionality**
   - Solves stated problem
   - Handles edge cases
   - Performance considerations

3. **Testing**
   - Adequate test coverage
   - Tests are meaningful
   - Tests pass consistently

4. **Documentation**
   - Clear docstrings
   - Updated user documentation
   - Examples provided

### Review Timeline

- **Initial review**: Within 3 business days
- **Follow-up reviews**: Within 2 business days
- **Final approval**: Within 1 business day

### Addressing Feedback

1. **Read feedback carefully**
2. **Ask questions if unclear**
3. **Make requested changes**
4. **Update tests if needed**
5. **Request re-review**

## Community

### Communication Channels

- **GitHub Discussions**: General questions and ideas
- **Discord**: Real-time chat and collaboration
- **Issues**: Bug reports and feature requests
- **Email**: security@terragon.ai for security issues

### Getting Help

1. **Check documentation first**
2. **Search existing issues**
3. **Ask in Discord #help channel**
4. **Create GitHub discussion**
5. **Open GitHub issue if bug**

### Recognition

We recognize contributors in several ways:

- **Contributor list**: In README and documentation
- **Release notes**: Major contributions highlighted
- **Discord roles**: Active contributors get special roles
- **Conference talks**: Opportunity to present work

## Advanced Contributing

### Research Contributions

For research contributions:

1. **Follow scientific standards**
2. **Include reproducibility guide**
3. **Add comprehensive benchmarks**
4. **Provide theoretical analysis**
5. **Submit to appropriate venues**

### Algorithm Implementations

For new algorithms:

1. **Start with issue discussion**
2. **Provide literature references**
3. **Include baseline comparisons**
4. **Add comprehensive tests**
5. **Document hyperparameters**

### Performance Optimizations

For performance improvements:

1. **Profile before optimizing**
2. **Include benchmarks**
3. **Maintain accuracy**
4. **Document trade-offs**
5. **Test on multiple platforms**

## Questions?

If you have questions not covered in this guide:

- **Discord**: #contributing channel
- **Email**: contribute@terragon.ai
- **GitHub**: Open a discussion

Thank you for contributing to ConfoRL! 🎉
