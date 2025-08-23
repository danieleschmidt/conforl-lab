#!/bin/bash
# ConfoRL Production Deployment Script

set -e

ENVIRONMENT=${1:-production}
VERSION=${2:-latest}

echo "🚀 Deploying ConfoRL $VERSION to $ENVIRONMENT"

# Build Docker image
echo "🐳 Building Docker image..."
docker build -f Dockerfile.production -t conforl:$VERSION .

# Tag for registry
if [ "$ENVIRONMENT" = "production" ]; then
    docker tag conforl:$VERSION your-registry.com/conforl:$VERSION
    docker push your-registry.com/conforl:$VERSION
fi

# Deploy based on environment
case $ENVIRONMENT in
    "local")
        echo "🏠 Deploying locally with Docker Compose..."
        docker-compose -f docker-compose.production.yml up -d
        ;;
    "staging"|"production")
        echo "☸️ Deploying to Kubernetes ($ENVIRONMENT)..."
        kubectl apply -f kubernetes/namespace.yaml
        kubectl apply -f kubernetes/
        kubectl set image deployment/conforl-app conforl=your-registry.com/conforl:$VERSION -n conforl
        kubectl rollout status deployment/conforl-app -n conforl --timeout=300s
        ;;
    *)
        echo "❌ Unknown environment: $ENVIRONMENT"
        exit 1
        ;;
esac

# Health check
echo "🏥 Performing health check..."
sleep 30
./scripts/health-check.sh $ENVIRONMENT

echo "✅ Deployment completed successfully!"
