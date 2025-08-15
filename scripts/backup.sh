#!/bin/bash

# ConfoRL Backup Script

DATE=$(date +"%Y%m%d_%H%M%S")
BACKUP_DIR="backups/conforl_$DATE"
NAMESPACE="conforl"

echo "💾 Creating ConfoRL backup..."
mkdir -p $BACKUP_DIR

# Backup Kubernetes resources
echo "☸️ Backing up Kubernetes resources..."
kubectl get all -n $NAMESPACE -o yaml > $BACKUP_DIR/k8s_resources.yaml
kubectl get configmaps -n $NAMESPACE -o yaml > $BACKUP_DIR/configmaps.yaml
kubectl get secrets -n $NAMESPACE -o yaml > $BACKUP_DIR/secrets.yaml

# Backup application data
echo "📁 Backing up application data..."
kubectl exec -n $NAMESPACE deployment/conforl-app -- tar czf - /app/data > $BACKUP_DIR/app_data.tar.gz

# Create backup manifest
cat > $BACKUP_DIR/backup_manifest.json << EOF
{
  "timestamp": "$(date -Iseconds)",
  "version": "$(kubectl get deployment conforl-app -n $NAMESPACE -o jsonpath='{.spec.template.spec.containers[0].image}')",
  "namespace": "$NAMESPACE",
  "backup_type": "full"
}
EOF

echo "✅ Backup created: $BACKUP_DIR"
