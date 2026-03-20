#!/usr/bin/env bash
# Common bash functions for deployment scripts

# Color codes for output
readonly COLOR_RED='\033[0;31m'
readonly COLOR_GREEN='\033[0;32m'
readonly COLOR_YELLOW='\033[1;33m'
readonly COLOR_BLUE='\033[0;34m'
readonly COLOR_RESET='\033[0m'

# Logging functions
log_info() {
    echo -e "${COLOR_BLUE}ℹ ${COLOR_RESET}$*"
}

log_success() {
    echo -e "${COLOR_GREEN}✅ ${COLOR_RESET}$*"
}

log_warning() {
    echo -e "${COLOR_YELLOW}⚠️  ${COLOR_RESET}$*"
}

log_error() {
    echo -e "${COLOR_RED}❌ ${COLOR_RESET}$*" >&2
}

# Check if command exists
command_exists() {
    command -v "$1" &> /dev/null
}

# Verify required commands are available
verify_commands() {
    local missing=()
    for cmd in "$@"; do
        if ! command_exists "$cmd"; then
            missing+=("$cmd")
        fi
    done

    if [ ${#missing[@]} -gt 0 ]; then
        log_error "Missing required commands: ${missing[*]}"
        return 1
    fi
    return 0
}

# Wait for a condition with timeout
wait_for_condition() {
    local condition_cmd="$1"
    local timeout="${2:-300}"
    local interval="${3:-5}"
    local elapsed=0

    while ! eval "$condition_cmd" &> /dev/null; do
        if [ $elapsed -ge $timeout ]; then
            log_error "Timeout waiting for condition: $condition_cmd"
            return 1
        fi
        sleep "$interval"
        elapsed=$((elapsed + interval))
    done
    return 0
}

# Check if namespace exists
namespace_exists() {
    kubectl get namespace "$1" &> /dev/null
}

# Create namespace if it doesn't exist
ensure_namespace() {
    local namespace="$1"
    if namespace_exists "$namespace"; then
        log_info "Namespace $namespace already exists"
    else
        log_info "Creating namespace $namespace..."
        kubectl create namespace "$namespace"
        log_success "Namespace $namespace created"
    fi
}

# Delete namespace with confirmation
delete_namespace_confirm() {
    local namespace="$1"
    local wait_time="${2:-5}"

    if ! namespace_exists "$namespace"; then
        log_warning "Namespace $namespace does not exist"
        return 0
    fi

    log_warning "This will delete namespace $namespace and all resources!"
    log_warning "Press Ctrl+C to cancel, or wait ${wait_time} seconds to continue..."
    sleep "$wait_time"

    kubectl delete namespace "$namespace"
    log_success "Namespace $namespace deleted"
}

# Export functions for use in other scripts
export -f log_info log_success log_warning log_error
export -f command_exists verify_commands wait_for_condition
export -f namespace_exists ensure_namespace delete_namespace_confirm
