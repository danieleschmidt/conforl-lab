#!/bin/bash

# ConfoRL Production Deployment Script
# This script handles deployment to various environments

set -euo pipefail

# Default values
ENVIRONMENT="production"
VERSION="latest"
NAMESPACE="conforl"
DRY_RUN=false
VERBOSE=false
FORCE=false

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Usage information
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Deploy ConfoRL to production environment

OPTIONS:
    -e, --environment ENV    Target environment (dev, staging, production) [default: production]
    -v, --version VERSION    Version to deploy [default: latest]
    -n, --namespace NS       Kubernetes namespace [default: conforl]
    -d, --dry-run           Perform dry run without actual deployment
    -f, --force             Force deployment without confirmation
    -V, --verbose           Enable verbose output
    -h, --help              Show this help message

EXAMPLES:
    $0 --environment staging --version v0.1.0
    $0 --dry-run --verbose
    $0 --force --environment production

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -v|--version)
                VERSION="$2"
                shift 2
                ;;
            -n|--namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            -d|--dry-run)
                DRY_RUN=true
                shift
                ;;
            -f|--force)
                FORCE=true
                shift
                ;;
            -V|--verbose)
                VERBOSE=true
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
}

# Validate environment
validate_environment() {
    case $ENVIRONMENT in
        dev|development)
            ENVIRONMENT="development"
            ;;
        staging|stage)
            ENVIRONMENT="staging"
            ;;
        prod|production)
            ENVIRONMENT="production"
            ;;
        *)
            log_error "Invalid environment: $ENVIRONMENT"
            log_error "Valid environments: dev, staging, production"
            exit 1
            ;;
    esac
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if required tools are installed
    local required_tools=("docker" "kubectl" "helm")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "$tool is not installed or not in PATH"
            exit 1
        fi
    done
    
    # Check if we can connect to Kubernetes cluster
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        log_error "Please check your kubectl configuration"
        exit 1
    fi
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Build Docker image
build_image() {
    log_info "Building Docker image..."
    
    local build_args=(
        --build-arg "BUILD_DATE=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        --build-arg "VERSION=$VERSION"
        --build-arg "VCS_REF=$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
        --tag "conforl:$VERSION"
    )
    
    if [[ "$ENVIRONMENT" == "production" ]]; then
        build_args+=(--build-arg "RUN_TESTS=true")
    else
        build_args+=(--build-arg "RUN_TESTS=false")
    fi
    
    if [[ "$VERBOSE" == "true" ]]; then
        build_args+=(--progress=plain)
    fi
    
    if [[ "$DRY_RUN" == "false" ]]; then
        docker build "${build_args[@]}" .
        log_success "Docker image built successfully"
    else
        log_info "DRY RUN: Would build Docker image with: ${build_args[*]}"
    fi
}

# Create namespace if it doesn't exist
create_namespace() {
    log_info "Ensuring namespace exists: $NAMESPACE"
    
    if [[ "$DRY_RUN" == "false" ]]; then
        kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
        log_success "Namespace $NAMESPACE ready"
    else
        log_info "DRY RUN: Would ensure namespace $NAMESPACE exists"
    fi
}

# Deploy secrets
deploy_secrets() {
    log_info "Deploying secrets..."
    
    local secrets_file="kubernetes/secrets-$ENVIRONMENT.yaml"
    
    if [[ -f "$secrets_file" ]]; then
        if [[ "$DRY_RUN" == "false" ]]; then
            kubectl apply -f "$secrets_file" -n "$NAMESPACE"
            log_success "Secrets deployed"
        else
            log_info "DRY RUN: Would deploy secrets from $secrets_file"
        fi
    else
        log_warning "Secrets file not found: $secrets_file"
        log_warning "Make sure to create secrets manually if needed"
    fi
}

# Deploy application
deploy_application() {
    log_info "Deploying ConfoRL application..."
    
    # Update image version in deployment
    local deployment_file="kubernetes/deployment.yaml"
    local temp_file=$(mktemp)
    
    # Replace image version
    sed "s|image: conforl:.*|image: conforl:$VERSION|g" "$deployment_file" > "$temp_file"
    
    if [[ "$DRY_RUN" == "false" ]]; then
        kubectl apply -f "$temp_file" -n "$NAMESPACE"
        rm "$temp_file"
        log_success "Application deployed"
    else
        log_info "DRY RUN: Would deploy application with version $VERSION"
        rm "$temp_file"
    fi
}

# Deploy monitoring
deploy_monitoring() {
    log_info "Deploying monitoring stack..."
    
    local monitoring_files=(
        "kubernetes/monitoring.yaml"
        "kubernetes/hpa.yaml"
    )
    
    for file in "${monitoring_files[@]}"; do
        if [[ -f "$file" ]]; then
            if [[ "$DRY_RUN" == "false" ]]; then
                kubectl apply -f "$file" -n "$NAMESPACE"
            else
                log_info "DRY RUN: Would deploy $file"
            fi
        else
            log_warning "Monitoring file not found: $file"
        fi
    done
    
    if [[ "$DRY_RUN" == "false" ]]; then
        log_success "Monitoring deployed"
    fi
}

# Wait for deployment to be ready
wait_for_deployment() {
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would wait for deployment to be ready"
        return 0
    fi
    
    log_info "Waiting for deployment to be ready..."
    
    local timeout=300  # 5 minutes
    local deployment_name="conforl-app"
    
    if kubectl rollout status deployment/"$deployment_name" -n "$NAMESPACE" --timeout="${timeout}s"; then
        log_success "Deployment is ready!"
    else
        log_error "Deployment failed to become ready within $timeout seconds"
        return 1
    fi
}

# Run health checks
run_health_checks() {
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would run health checks"
        return 0
    fi
    
    log_info "Running health checks..."
    
    # Get service endpoint
    local service_name="conforl-service"
    local service_ip
    
    # Wait for service to have endpoints
    for i in {1..30}; do
        service_ip=$(kubectl get service "$service_name" -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")
        if [[ -n "$service_ip" ]]; then
            break
        fi
        log_info "Waiting for service endpoint... ($i/30)"
        sleep 10
    done
    
    # Port forward for health check if no external IP
    if [[ -z "$service_ip" ]]; then
        log_info "Using port forwarding for health check..."
        kubectl port-forward "service/$service_name" 8080:80 -n "$NAMESPACE" &
        local port_forward_pid=$!
        sleep 5
        service_ip="localhost:8080"
    fi
    
    # Perform health check
    local health_url="http://$service_ip/health"
    local max_attempts=10
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        if curl -sf "$health_url" > /dev/null 2>&1; then
            log_success "Health check passed!"
            break
        else
            log_info "Health check attempt $attempt/$max_attempts failed, retrying..."
            sleep 10
            ((attempt++))
        fi
    done
    
    # Clean up port forward if used
    if [[ -n "${port_forward_pid:-}" ]]; then
        kill "$port_forward_pid" 2>/dev/null || true
    fi
    
    if [[ $attempt -gt $max_attempts ]]; then
        log_error "Health checks failed after $max_attempts attempts"
        return 1
    fi
}

# Cleanup old deployments
cleanup_old_deployments() {
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would cleanup old deployments"
        return 0
    fi
    
    log_info "Cleaning up old deployments..."
    
    # Keep last 3 ReplicaSets
    kubectl delete replicasets -n "$NAMESPACE" \
        --selector=app=conforl \
        --sort-by=.metadata.creationTimestamp \
        --field-selector=status.replicas=0 \
        $(kubectl get replicasets -n "$NAMESPACE" --selector=app=conforl --sort-by=.metadata.creationTimestamp -o name | head -n -3) \
        2>/dev/null || true
    
    log_success "Cleanup completed"
}

# Print deployment summary
print_summary() {
    cat << EOF

${GREEN}========================================${NC}
${GREEN}     ConfoRL Deployment Summary${NC}
${GREEN}========================================${NC}

Environment:    $ENVIRONMENT
Version:        $VERSION
Namespace:      $NAMESPACE
Dry Run:        $DRY_RUN

$(if [[ "$DRY_RUN" == "false" ]]; then
    cat << EOL
Status:         ${GREEN}SUCCESS${NC}

Access URLs:
- Application:  http://$(kubectl get service conforl-service -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "pending")
- Monitoring:   http://$(kubectl get service grafana -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "pending"):3000

Next Steps:
1. Verify application is working correctly
2. Monitor logs: kubectl logs -f deployment/conforl-app -n $NAMESPACE
3. Check metrics in Grafana dashboard
4. Run integration tests if available
EOL
else
    echo "Status:         ${YELLOW}DRY RUN COMPLETED${NC}"
fi)

${GREEN}========================================${NC}

EOF
}

# Confirmation prompt
confirm_deployment() {
    if [[ "$FORCE" == "true" || "$DRY_RUN" == "true" ]]; then
        return 0
    fi
    
    cat << EOF
${YELLOW}========================================${NC}
${YELLOW}     Deployment Confirmation${NC}
${YELLOW}========================================${NC}

Environment:    $ENVIRONMENT
Version:        $VERSION
Namespace:      $NAMESPACE

This will deploy ConfoRL to the $ENVIRONMENT environment.
Are you sure you want to continue?

EOF
    
    read -p "Type 'yes' to continue: " -r
    if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
        log_info "Deployment cancelled by user"
        exit 0
    fi
}

# Main deployment function
main() {
    log_info "Starting ConfoRL deployment..."
    
    parse_args "$@"
    validate_environment
    confirm_deployment
    
    check_prerequisites
    build_image
    create_namespace
    deploy_secrets
    deploy_application
    deploy_monitoring
    wait_for_deployment
    run_health_checks
    cleanup_old_deployments
    
    print_summary
    
    log_success "ConfoRL deployment completed successfully!"
}

# Error handling
trap 'log_error "Deployment failed on line $LINENO"' ERR

# Run main function
main "$@"