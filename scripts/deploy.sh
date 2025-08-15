#!/bin/bash
set -e

# ConfoRL Production Deployment Script

ENVIRONMENT=${1:-production}
VERSION=${2:-latest}
NAMESPACE="conforl"

echo "🚀 Deploying ConfoRL to $ENVIRONMENT environment..."
echo "Version: $VERSION"
echo "Namespace: $NAMESPACE"

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl not found. Please install kubectl."
    exit 1
fi

# Check if cluster is accessible
if ! kubectl cluster-info &> /dev/null; then
    echo "❌ Cannot connect to Kubernetes cluster."
    exit 1
fi

# Create namespace if it doesn't exist
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Apply security policies
echo "🔒 Applying security policies..."
kubectl apply -f security/

# Apply Kubernetes manifests
echo "☸️ Applying Kubernetes manifests..."
kubectl apply -f kubernetes/

# Update image version
kubectl set image deployment/conforl-app conforl=conforl:$VERSION -n $NAMESPACE

# Wait for rollout to complete
echo "⏳ Waiting for deployment to complete..."
kubectl rollout status deployment/conforl-app -n $NAMESPACE --timeout=300s

# Check deployment health
echo "🏥 Checking deployment health..."
kubectl get pods -n $NAMESPACE -l app=conforl

# Run health checks
HEALTH_CHECK_URL=$(kubectl get service conforl-service -n $NAMESPACE -o jsonpath='{.spec.clusterIP}')
if kubectl run health-check --rm -i --restart=Never --image=curlimages/curl -- curl -f http://$HEALTH_CHECK_URL/health; then
    echo "✅ ConfoRL deployment successful!"
    echo "🌐 Service is healthy and ready to serve requests."
else
    echo "❌ Health check failed!"
    exit 1
fi

echo "📊 Deployment summary:"
kubectl get all -n $NAMESPACE
