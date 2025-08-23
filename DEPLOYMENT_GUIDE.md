# ConfoRL Production Deployment Guide

## Quick Start

### Local Development
```bash
# Start local environment
docker-compose -f docker-compose.production.yml up -d

# Check health
curl http://localhost:8000/health
```

### Production Deployment

#### Prerequisites
- Docker and Docker Compose
- Kubernetes cluster (for production)
- Container registry access

#### Deploy to Production
```bash
# Build and deploy
./scripts/deploy.sh production v1.0.0

# Monitor deployment
kubectl get pods -n conforl
kubectl logs -f deployment/conforl-app -n conforl
```

## Architecture

### Components
- **ConfoRL App**: Main application server
- **Nginx**: Reverse proxy and load balancer
- **Prometheus**: Metrics collection
- **Grafana**: Monitoring dashboards

### Scaling
- Horizontal Pod Autoscaler (HPA) configured
- Auto-scales from 3 to 50 replicas
- Based on CPU (70%) and memory (80%) utilization

### Monitoring
- Health checks on `/health` and `/ready` endpoints
- Prometheus metrics on `/metrics`
- Custom alerts for high error rate, latency, and service downtime

## Configuration

### Environment Variables
- `CONFORL_ENV`: Environment (development, staging, production)
- `CONFORL_LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `CONFORL_MAX_WORKERS`: Maximum worker processes

### Volumes
- `/app/models`: Persistent model storage
- `/app/logs`: Application logs

## Security

### Best Practices
- Non-root user in containers
- Resource limits enforced
- Security scanning in CI/CD
- TLS encryption in production

### Network Security
- Container network isolation
- Ingress controller with TLS
- Network policies for pod communication

## Troubleshooting

### Common Issues
1. **Pod not starting**: Check resource limits and node capacity
2. **High latency**: Review HPA scaling and resource allocation
3. **Memory leaks**: Monitor memory usage and restart policies

### Debug Commands
```bash
# Check pod status
kubectl describe pod -n conforl

# View logs
kubectl logs -f deployment/conforl-app -n conforl

# Port forward for debugging
kubectl port-forward service/conforl-service 8000:80 -n conforl
```

## Monitoring and Alerting

### Dashboards
- Application Performance: Grafana dashboard at http://localhost:3000
- Infrastructure: Prometheus metrics at http://localhost:9090

### Key Metrics
- Request rate and latency
- Error rate and types
- Resource utilization
- Safety violation rates

### Alerts
- High error rate (>5% for 2 minutes)
- High latency (>500ms for 5 minutes)
- Service down (>1 minute)

## Backup and Recovery

### Model Backup
Models are stored in persistent volumes and backed up daily.

### Database Backup
Configuration and metrics are backed up to external storage.

## Support

For production issues, contact: support@terragonlabs.ai
