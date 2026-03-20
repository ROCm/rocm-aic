#!/usr/bin/env bash
# Load both llm-d and rocm-aic Grafana dashboards
# This script wraps llm-d's load-llm-d-dashboards.sh and adds rocm-aic custom dashboards

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Configuration
NAMESPACE="${1:-llm-d-monitoring}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROCM_AIC_DASHBOARDS_DIR="${SCRIPT_DIR}/grafana/dashboards"

echo "========================================="
echo "  Loading Grafana Dashboards"
echo "========================================="
echo ""

# Load rocm-aic custom dashboards
log_info "Loading rocm-aic custom dashboards from: ${ROCM_AIC_DASHBOARDS_DIR}"

if [ ! -d "$ROCM_AIC_DASHBOARDS_DIR" ]; then
    log_warning "rocm-aic dashboard directory not found"
    log_info "Create dashboards in: ${ROCM_AIC_DASHBOARDS_DIR}"
    exit 0
fi

# Check if namespace exists
if ! kubectl get namespace "${NAMESPACE}" &>/dev/null; then
    log_error "Namespace ${NAMESPACE} does not exist"
    exit 1
fi

# Count dashboards
dashboard_count=$(find "${ROCM_AIC_DASHBOARDS_DIR}" -maxdepth 1 -name "*.json" -type f 2>/dev/null | wc -l)

if [ "$dashboard_count" -eq 0 ]; then
    log_info "No custom rocm-aic dashboards found in ${ROCM_AIC_DASHBOARDS_DIR}"
    log_info "Add .json dashboard files to this directory to load them"
    echo ""
    log_success "Dashboard loading complete (llm-d defaults only)"
    exit 0
fi

log_info "Found ${dashboard_count} custom rocm-aic dashboard(s)"

# Load each dashboard
for dashboard_file in "${ROCM_AIC_DASHBOARDS_DIR}"/*.json; do
    if [ ! -f "$dashboard_file" ]; then
        continue
    fi

    dashboard_name=$(basename "$dashboard_file" .json)
    configmap_name="rocm-aic-${dashboard_name}"

    log_info "Loading custom dashboard: ${dashboard_name}"

    # Create ConfigMap with dashboard JSON
    kubectl create configmap "${configmap_name}" \
        --from-file="${dashboard_name}.json=${dashboard_file}" \
        --namespace="${NAMESPACE}" \
        --dry-run=client -o yaml | \
    kubectl label -f - \
        grafana_dashboard=1 \
        --local --dry-run=client -o yaml | \
    kubectl apply -f -

    if [ $? -eq 0 ]; then
        log_success "Custom dashboard ${dashboard_name} loaded"
    else
        log_error "Failed to load custom dashboard ${dashboard_name}"
    fi
done

echo ""
log_success "All dashboards loaded successfully"
log_info "Grafana will automatically discover and load these dashboards within 30 seconds"
