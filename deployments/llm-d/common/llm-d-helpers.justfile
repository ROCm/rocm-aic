# LLM-D specific helper recipes
# Import this file in llm-d deployment justfiles with: import? "../common/llm-d-helpers.justfile"

# Path to llm-d submodule (relative from deployment directory)
LLM_D_PATH := "../../../submodules/llm-d"

# InferencePool chart configuration
INFERENCEPOOL_VERSION := "v1.4.0"
INFERENCEPOOL_CHART := "oci://registry.k8s.io/gateway-api-inference-extension/charts/inferencepool"

# HuggingFace token secret name
HF_TOKEN_NAME := env_var_or_default("HF_TOKEN_NAME", "llm-d-hf-token")

# Color output helpers
RED := env_var_or_default("RED", "\\033[0;31m")
GREEN := env_var_or_default("GREEN", "\\033[0;32m")
YELLOW := env_var_or_default("YELLOW", "\\033[1;33m")
NC := env_var_or_default("NC", "\\033[0m")

# Verify llm-d submodule is initialized
verify-llm-d-submodule:
    @if [ ! -e "{{LLM_D_PATH}}/.git" ]; then \
        echo "❌ llm-d submodule not initialized"; \
        echo "   Run 'just setup-submodules' from repository root"; \
        exit 1; \
    fi
    @echo "✅ llm-d submodule initialized"

# Show llm-d version from submodule
show-llm-d-version:
    @if [ -f "{{LLM_D_PATH}}/VERSION" ]; then \
        echo "llm-d version: $(cat {{LLM_D_PATH}}/VERSION)"; \
    else \
        cd {{LLM_D_PATH}} && git describe --tags 2>/dev/null || echo "unknown"; \
    fi

# Helper: Create namespace if it doesn't exist (call from deployment justfiles)
_create-namespace NAMESPACE:
    @echo "Creating namespace {{NAMESPACE}} if it doesn't exist..."
    @kubectl create namespace {{NAMESPACE}} 2>/dev/null || echo "Namespace {{NAMESPACE}} already exists"

# Helper: Delete namespace with confirmation (call from deployment justfiles)
_delete-namespace NAMESPACE:
    @echo "⚠️  WARNING: This will delete namespace {{NAMESPACE}} and all resources!"
    @echo "Press Ctrl+C to cancel, or wait 5 seconds to continue..."
    @sleep 5
    @kubectl delete namespace {{NAMESPACE}}
    @echo "✅ Namespace {{NAMESPACE}} deleted"

# Helper: Wait for a deployment to be ready
_wait-for-deployment NAMESPACE DEPLOYMENT_NAME COND TIMEOUT="300s":
    @echo "Waiting for deployment {{DEPLOYMENT_NAME}} in namespace {{NAMESPACE}}..."
    @kubectl wait --for=condition={{COND}} --timeout={{TIMEOUT}} deployment/{{DEPLOYMENT_NAME}} -n {{NAMESPACE}} || exit 1
    @echo "✅ Deployment {{DEPLOYMENT_NAME}} is ready"

# Helper: Get logs from pods with a specific label
_get-logs-by-label NAMESPACE LABEL_SELECTOR TAIL="50":
    @kubectl logs -n {{NAMESPACE}} -l {{LABEL_SELECTOR}} --tail={{TAIL}} --prefix=true

# Helper: Follow logs from pods with a specific label
_follow-logs-by-label NAMESPACE LABEL_SELECTOR:
    @kubectl logs -n {{NAMESPACE}} -l {{LABEL_SELECTOR}} -f --prefix=true

# Register HuggingFace token as Kubernetes secret in the specified namespace
register-hf-token NAMESPACE:
    #!/usr/bin/env bash
    set -euo pipefail

    echo -e "{{GREEN}}Registering HuggingFace token in namespace {{NAMESPACE}}...{{NC}}"

    # Check if HF_TOKEN is set
    if [ -z "${HF_TOKEN:-}" ]; then
        echo -e "{{RED}}Error: HF_TOKEN environment variable is not set{{NC}}" >&2
        echo ""
        echo "Please set HF_TOKEN before running this target:"
        echo "  export HF_TOKEN=your_token_here"
        echo "  just register-hf-token {{NAMESPACE}}"
        exit 1
    fi

    # Verify kubectl can connect
    if ! kubectl cluster-info &> /dev/null; then
        echo -e "{{RED}}Error: Cannot connect to Kubernetes cluster{{NC}}" >&2
        exit 1
    fi

    # Create namespace if it doesn't exist
    kubectl create namespace {{NAMESPACE}} 2>/dev/null || echo "Namespace {{NAMESPACE}} already exists"

    # Create or update secret
    echo "Creating secret {{HF_TOKEN_NAME}} in namespace {{NAMESPACE}}..."
    if ! kubectl create secret generic {{HF_TOKEN_NAME}} \
        --from-literal="HF_TOKEN=${HF_TOKEN}" \
        --namespace "{{NAMESPACE}}" \
        --dry-run=client -o yaml | kubectl apply -f -; then
        echo -e "{{RED}}Error: Failed to create HuggingFace token secret{{NC}}" >&2
        echo -e "{{YELLOW}}Completed: namespace creation{{NC}}"
        echo -e "{{YELLOW}}Remaining: secret creation{{NC}}"
        exit 1
    fi

    echo -e "{{GREEN}}✅ HuggingFace token registered as {{HF_TOKEN_NAME}} in namespace {{NAMESPACE}}{{NC}}"
