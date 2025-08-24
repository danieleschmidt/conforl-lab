#!/bin/bash

# ConfoRL Production Health Check Script

echo "🏥 ConfoRL Production Health Check"
echo "=================================="

NAMESPACE="conforl-production"

# Kubernetes health
echo "☸️ Kubernetes Status:"
kubectl get pods -n ${NAMESPACE} -o wide
kubectl get svc -n ${NAMESPACE}
kubectl get hpa -n ${NAMESPACE}

# Application health
echo -e "\n🔍 Application Health:"
kubectl logs -l app=conforl -n ${NAMESPACE} --tail=10

# Monitoring health
echo -e "\n📊 Monitoring Status:"
docker ps | grep -E "(prometheus|grafana|jaeger)" || echo "⚠️ Monitoring stack not running"

# Performance metrics
echo -e "\n⚡ Performance Metrics:"
kubectl top pods -n ${NAMESPACE} 2>/dev/null || echo "⚠️ Metrics server not available"

echo -e "\n✅ Health check complete"
