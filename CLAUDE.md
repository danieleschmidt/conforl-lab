# ConfoRL Development Guide for Claude

This document provides guidance for Claude when working on the ConfoRL project.

## Project Overview

ConfoRL is a research-grade Python library that brings **provable finite-sample safety guarantees** to reinforcement learning through adaptive conformal risk control. It's the first open-source implementation combining conformal prediction theory with both offline and online RL for deployment in safety-critical domains.

## Architecture

### Core Components

1. **conforl/core/**: Core conformal prediction utilities
   - `conformal.py`: Split conformal prediction implementation
   - `types.py`: Type definitions (RiskCertificate, TrajectoryData, etc.)

2. **conforl/algorithms/**: Conformal RL algorithms
   - `base.py`: Base ConformalRLAgent class
   - `sac.py`: ConformaSAC (Soft Actor-Critic with conformal guarantees)
   - `ppo.py`: ConformaPPO (PPO with conformal guarantees)
   - `td3.py`: ConformaTD3 (TD3 with conformal guarantees)
   - `cql.py`: ConformaCQL (Conservative Q-Learning for offline RL)

3. **conforl/risk/**: Risk measures and controllers
   - `measures.py`: Risk measure implementations
   - `controllers.py`: Adaptive risk controllers

4. **conforl/deploy/**: Production deployment utilities
   - `pipeline.py`: Safe deployment pipeline
   - `monitor.py`: Risk monitoring and logging

5. **conforl/utils/**: Utility functions
   - `validation.py`: Input validation and sanitization
   - `errors.py`: Custom exception classes
   - `logging.py`: Comprehensive logging setup
   - `security.py`: Security utilities

6. **conforl/optimize/**: Performance optimization
   - `cache.py`: Adaptive caching system
   - `concurrent.py`: Parallel processing utilities
   - `profiler.py`: Performance profiling
   - `scaling.py`: Auto-scaling and load balancing

7. **conforl/i18n/**: Internationalization
   - `translator.py`: Multi-language support
   - `compliance.py`: GDPR/CCPA compliance checking
   - `formats.py`: Localized formatting

8. **conforl/monitoring/**: Monitoring and self-improvement
   - `metrics.py`: Comprehensive metrics collection
   - `adaptive.py`: Self-improving agents and hyperparameter optimization

## Development Standards

### Code Quality
- Comprehensive error handling with custom exception classes
- Input validation and sanitization for security
- Thread-safe implementations where needed
- Type hints throughout the codebase
- Docstrings following Google style

### Testing
- Test coverage target: 85%+
- Tests located in `tests/` directory
- Use pytest for testing framework
- Include unit tests, integration tests, and end-to-end tests
- Mock external dependencies appropriately

### Security
- Input sanitization to prevent injection attacks
- Secure file path handling to prevent directory traversal
- Hash sensitive data using secure methods
- GDPR/CCPA compliance features built-in
- Security logging for audit trails

### Performance
- Adaptive caching with usage pattern learning
- Concurrent processing for scalability
- Performance profiling and monitoring
- Memory leak detection
- Auto-scaling capabilities

### Deployment
- Docker containerization with multi-stage builds
- Kubernetes manifests for production deployment
- Monitoring stack (Prometheus, Grafana, Jaeger)
- Health checks and readiness probes
- Rolling updates with zero downtime

## Key Design Patterns

### 1. Conformal Risk Control
```python
# All algorithms follow this pattern:
risk_controller = AdaptiveRiskController(target_risk=0.05, confidence=0.95)
agent = ConformaSAC(env=env, risk_controller=risk_controller)
action, certificate = agent.predict(state, return_risk_certificate=True)
```

### 2. Self-Improving Agents
```python
# Agents can continuously optimize themselves:
improving_agent = SelfImprovingAgent(base_agent, performance_tracker)
improving_agent.train()  # Automatically adapts parameters
```

### 3. Safe Deployment
```python
# Production deployment with safety guarantees:
pipeline = SafeDeploymentPipeline(agent, risk_monitor=True, fallback_policy=safe_policy)
results = pipeline.deploy(env, num_episodes=1000)
```

## Common Commands

### Development
```bash
# Install in development mode
pip install -e .

# Run tests
pytest tests/ -v --cov=conforl --cov-report=html

# Lint code
black conforl/ tests/
isort conforl/ tests/
mypy conforl/

# Security scan
bandit -r conforl/
```

### Training and Evaluation
```bash
# Train an agent
conforl train --algorithm sac --env CartPole-v1 --timesteps 100000

# Evaluate trained agent
conforl evaluate --model ./models/agent --env CartPole-v1 --episodes 10

# Deploy to production
conforl deploy --model ./models/agent --env CartPole-v1 --monitor

# Generate risk certificate
conforl certificate --model ./models/agent --coverage 0.95
```

### Deployment
```bash
# Build Docker image
docker build -t conforl:latest .

# Deploy with Docker Compose
docker-compose up -d

# Deploy to Kubernetes
./scripts/deploy.sh --environment production --version v0.1.0

# Monitor deployment
kubectl get pods -n conforl
kubectl logs -f deployment/conforl-app -n conforl
```

## Important Files

- `requirements.txt`: Python dependencies
- `setup.py`: Package configuration
- `pytest.ini`: Test configuration
- `Dockerfile`: Container build instructions
- `docker-compose.yml`: Multi-service deployment
- `kubernetes/`: Kubernetes manifests
- `.env.example`: Environment variables template

## When Adding New Features

1. **Add appropriate error handling** using custom exception classes
2. **Include comprehensive tests** with good coverage
3. **Add logging** with appropriate levels
4. **Consider security implications** and add validation
5. **Update documentation** and type hints
6. **Consider internationalization** for user-facing messages
7. **Add metrics** for monitoring if applicable
8. **Ensure thread safety** if the feature involves concurrency

## Troubleshooting

### Common Issues
1. **Import errors**: Check that all `__init__.py` files are present
2. **Test failures**: Ensure mock objects have all required attributes
3. **Type checking**: Use `Union`, `Optional`, and proper type annotations
4. **Memory leaks**: Use memory profiler to detect issues
5. **Performance**: Use performance profiler to identify bottlenecks

### Debugging
- Use `conforl.utils.logging.get_logger(__name__)` for consistent logging
- Enable debug logging: `CONFORL_LOG_LEVEL=DEBUG`
- Use performance profiler: `from conforl.optimize.profiler import PerformanceProfiler`
- Check metrics: Access metrics via `MetricsCollector`

## Contact and Support

For questions about the ConfoRL codebase, refer to:
- README.md for general project information
- Code comments and docstrings for implementation details
- Test files for usage examples
- This CLAUDE.md file for development guidance