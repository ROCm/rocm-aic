# LLM-D specific helper recipes
# Import this file in llm-d deployment justfiles with: import? "../common/llm-d-helpers.justfile"

# Path to llm-d submodule (relative from deployment directory)
LLM_D_PATH := "../../../submodules/llm-d"

# InferencePool chart configuration
INFERENCEPOOL_VERSION := "v1.3.1"
INFERENCEPOOL_CHART := "oci://registry.k8s.io/gateway-api-inference-extension/charts/inferencepool"

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
_wait-for-deployment NAMESPACE DEPLOYMENT_NAME TIMEOUT="300s":
    @echo "Waiting for deployment {{DEPLOYMENT_NAME}} in namespace {{NAMESPACE}}..."
    @kubectl wait --for=condition=available --timeout={{TIMEOUT}} deployment/{{DEPLOYMENT_NAME}} -n {{NAMESPACE}} || exit 1
    @echo "✅ Deployment {{DEPLOYMENT_NAME}} is ready"

# Helper: Get logs from pods with a specific label
_get-logs-by-label NAMESPACE LABEL_SELECTOR TAIL="50":
    @kubectl logs -n {{NAMESPACE}} -l {{LABEL_SELECTOR}} --tail={{TAIL}} --prefix=true

# Helper: Follow logs from pods with a specific label
_follow-logs-by-label NAMESPACE LABEL_SELECTOR:
    @kubectl logs -n {{NAMESPACE}} -l {{LABEL_SELECTOR}} -f --prefix=true
