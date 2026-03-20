# ROCm-ICMS Deployment Infrastructure
# Root justfile for managing llm-d and custom deployments

# Default recipe - show available commands
default:
    @just --list

# Show all available deployments organized by type
list:
    @echo "Available Deployments:"
    @echo ""
    @echo "LLM-D Based Deployments (deployments/llm-d/):"
    @echo "  - tiered-prefix-cache    CPU/GPU tiered prefix caching"
    @echo "  - inference-scheduling   Intelligent load-aware scheduling"
    @echo ""
    @echo "Custom Deployments (deployments/custom/):"
    @echo "  (none yet - add your custom deployments here)"
    @echo ""
    @echo "To deploy, navigate to the deployment directory and run 'just deploy'"
    @echo "Example: cd deployments/llm-d/tiered-prefix-cache && just setup"

# Show detailed information about each deployment
info:
    @echo "Deployment Information"
    @echo "====================="
    @echo ""
    @echo "Tiered Prefix Cache (deployments/llm-d/tiered-prefix-cache/):"
    @echo "  Description: Offload KV cache from GPU HBM to CPU RAM or storage"
    @echo "  Variants: offloading-connector (default), lmcache-connector"
    @echo "  Target: AMD GPUs"
    @echo "  Method: Kustomize + Helm"
    @echo ""
    @echo "Inference Scheduling (deployments/llm-d/inference-scheduling/):"
    @echo "  Description: Intelligent prefix-cache aware request routing"
    @echo "  Target: AMD GPUs"
    @echo "  Method: Helmfile with 3 charts"
    @echo ""
    @echo "For more details, see README.md in each deployment directory"

# Verify all prerequisites (kubectl, helm, helmfile)
verify-prereqs:
    @echo "Verifying prerequisites..."
    @command -v kubectl >/dev/null 2>&1 || (echo "❌ kubectl not found" && exit 1)
    @command -v helm >/dev/null 2>&1 || (echo "❌ helm not found" && exit 1)
    @command -v helmfile >/dev/null 2>&1 || (echo "❌ helmfile not found" && exit 1)
    @command -v just >/dev/null 2>&1 || (echo "❌ just not found" && exit 1)
    @kubectl cluster-info >/dev/null 2>&1 || (echo "❌ Cannot connect to Kubernetes cluster" && exit 1)
    @echo "✅ All prerequisites satisfied"
    @echo "  ✓ kubectl: $(kubectl version --client --short 2>/dev/null || kubectl version --client)"
    @echo "  ✓ helm: $(helm version --short)"
    @echo "  ✓ helmfile: $(helmfile --version)"
    @echo "  ✓ just: $(just --version)"

# Initialize llm-d submodule
setup-submodules:
    @echo "Initializing submodules..."
    @./scripts/setup-submodules.sh

# Update llm-d submodule to latest
update-submodules:
    @echo "Updating submodules..."
    @git submodule update --remote --merge
    @echo "✅ Submodules updated"

# Show git submodule status
submodule-status:
    @echo "Submodule status:"
    @git submodule status

# Quick health check of the repository setup
health-check:
    @echo "Repository Health Check"
    @echo "======================="
    @echo ""
    @echo "Submodules:"
    @if [ -f "submodules/llm-d/.git" ]; then \
        echo "  ✓ llm-d submodule initialized"; \
    else \
        echo "  ✗ llm-d submodule not initialized (run 'just setup-submodules')"; \
    fi
    @echo ""
    @echo "Deployment directories:"
    @if [ -d "deployments/llm-d/tiered-prefix-cache" ]; then \
        echo "  ✓ tiered-prefix-cache"; \
    else \
        echo "  ✗ tiered-prefix-cache"; \
    fi
    @if [ -d "deployments/llm-d/inference-scheduling" ]; then \
        echo "  ✓ inference-scheduling"; \
    else \
        echo "  ✗ inference-scheduling"; \
    fi

# Check for configuration drift between rocm-icms and llm-d
check-drift:
    @echo "Checking for configuration drift with llm-d upstream..."
    @python3 scripts/check-llm-d-drift.py

# Check drift for specific deployment
check-drift-deployment DEPLOYMENT:
    @echo "Checking {{DEPLOYMENT}} drift with llm-d upstream..."
    @python3 scripts/check-llm-d-drift.py --deployment {{DEPLOYMENT}}

# Generate JSON drift report
check-drift-json OUTPUT:
    @echo "Generating JSON drift report..."
    @python3 scripts/check-llm-d-drift.py --json {{OUTPUT}}
    @echo "Report saved to: {{OUTPUT}}"
