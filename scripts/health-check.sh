#!/bin/bash
# ConfoRL Health Check Script

ENVIRONMENT=${1:-local}

case $ENVIRONMENT in
    "local")
        HEALTH_URL="http://localhost:8000/health"
        ;;
    "staging")
        HEALTH_URL="http://conforl-staging.yourcompany.com/health"
        ;;
    "production")
        HEALTH_URL="http://conforl.yourcompany.com/health"
        ;;
esac

echo "🏥 Checking health at $HEALTH_URL"

for i in {1..30}; do
    if curl -f $HEALTH_URL > /dev/null 2>&1; then
        echo "✅ Service is healthy!"
        exit 0
    else
        echo "⏳ Waiting for service... (attempt $i/30)"
        sleep 10
    fi
done

echo "❌ Service health check failed!"
exit 1
