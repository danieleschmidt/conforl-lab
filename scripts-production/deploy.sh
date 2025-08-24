#!/bin/bash
set -e

# ConfoRL Production Deployment Script

echo "🚀 Starting ConfoRL Production Deployment"

# Configuration
ENVIRONMENT=${1:-production}
VERSION=${2:-latest}
NAMESPACE="conforl-production"

# Check requirements
echo "📋 Checking deployment requirements..."
command -v docker >/dev/null 2>&1 || { echo "❌ Docker is required"; exit 1; }
command -v kubectl >/dev/null 2>&1 || { echo "❌ kubectl is required"; exit 1; }

# Build and tag image
echo "🐳 Building production image..."
docker build -f Dockerfile.production.optimized -t conforl:${VERSION} .
docker tag conforl:${VERSION} conforl:latest

# Apply Kubernetes manifests
echo "☸️ Deploying to Kubernetes..."
kubectl apply -f kubernetes-production/namespace.yaml
kubectl apply -f kubernetes-production/deployment.yaml
kubectl apply -f kubernetes-production/service.yaml
kubectl apply -f kubernetes-production/hpa.yaml
kubectl apply -f kubernetes-production/ingress.yaml

# Wait for rollout
echo "⏳ Waiting for deployment rollout..."
kubectl rollout status deployment/conforl-app -n ${NAMESPACE} --timeout=300s

# Health check
echo "🏥 Running health checks..."
kubectl wait --for=condition=ready pod -l app=conforl -n ${NAMESPACE} --timeout=120s

# Start monitoring stack
echo "📊 Starting monitoring stack..."
docker-compose -f docker-compose.production.complete.yml up -d prometheus grafana jaeger

echo "✅ ConfoRL Production Deployment Complete!"
echo "📊 Grafana: http://localhost:3000 (admin/admin)"
echo "📈 Prometheus: http://localhost:9090"
echo "🔍 Jaeger: http://localhost:16686"
echo "🏥 Health: kubectl get pods -n ${NAMESPACE}"
