#!/bin/bash
set -e

# ConfoRL Production Rollback Script

echo "🔄 Starting ConfoRL Production Rollback"

NAMESPACE="conforl-production"
REVISION=${1:-}

if [ -z "$REVISION" ]; then
    echo "📋 Available revisions:"
    kubectl rollout history deployment/conforl-app -n ${NAMESPACE}
    echo "Usage: $0 <revision-number>"
    exit 1
fi

echo "🔙 Rolling back to revision ${REVISION}..."
kubectl rollout undo deployment/conforl-app --to-revision=${REVISION} -n ${NAMESPACE}

echo "⏳ Waiting for rollback to complete..."
kubectl rollout status deployment/conforl-app -n ${NAMESPACE} --timeout=300s

echo "🏥 Running health checks..."
kubectl wait --for=condition=ready pod -l app=conforl -n ${NAMESPACE} --timeout=120s

echo "✅ ConfoRL Production Rollback Complete!"
