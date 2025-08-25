#!/bin/bash

# ConfoRL Health Check Script

NAMESPACE="conforl"
SERVICE_NAME="conforl-service"

echo "🏥 ConfoRL Health Check"
echo "======================="

# Check namespace
if ! kubectl get namespace $NAMESPACE &> /dev/null; then
    echo "❌ Namespace $NAMESPACE not found"
    exit 1
fi

# Check deployment
DEPLOYMENT_STATUS=$(kubectl get deployment conforl-app -n $NAMESPACE -o jsonpath='{.status.readyReplicas}/{.status.replicas}')
echo "📦 Deployment Status: $DEPLOYMENT_STATUS"

# Check service
SERVICE_IP=$(kubectl get service $SERVICE_NAME -n $NAMESPACE -o jsonpath='{.spec.clusterIP}')
echo "🌐 Service IP: $SERVICE_IP"

# Check pods
echo "🐳 Pod Status:"
kubectl get pods -n $NAMESPACE -l app=conforl

# Check HPA
echo "📈 Autoscaler Status:"
kubectl get hpa conforl-hpa -n $NAMESPACE

# Check metrics
echo "📊 Resource Usage:"
kubectl top pods -n $NAMESPACE --containers

echo "✅ Health check completed"
