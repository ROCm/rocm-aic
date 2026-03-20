# ROCm-ICMS Deployments

This directory contains deployment automation for various LLM inference guides targeting AMD GPUs.

## Directory Structure

```
deployments/
├── common/                      # Global shared utilities (all deployments)
│   ├── prerequisites.justfile   # Common prerequisite checks
│   ├── monitoring.justfile      # Shared monitoring commands
│   └── config.yaml              # Global configuration
│
├── llm-d/                       # LLM-D based deployments
│   ├── common/                  # LLM-D specific utilities
│   ├── tiered-prefix-cache/     # KV cache offloading deployment
│   └── inference-scheduling/    # Intelligent request routing deployment
│
└── custom/                      # Custom deployments (non-llm-d)
    └── (your custom deployments here)
```

## Organization

### LLM-D Deployments (`llm-d/`)

Deployments based on the [llm-d project](https://github.com/llm-d/llm-d), which provides well-tested "lit paths" for LLM inference on Kubernetes.

**Available deployments:**
- **tiered-prefix-cache**: Offload KV cache from GPU HBM to CPU RAM
- **inference-scheduling**: Load-aware and cache-aware request routing

See [`llm-d/README.md`](llm-d/README.md) for details.

### Custom Deployments (`custom/`)

For deployments not based on llm-d guides. Add your own deployment automation here.

Custom deployments can still leverage shared utilities from `common/`.

## Common Utilities

### Global Common (`common/`)

Utilities available to **all** deployments (llm-d and custom):

- **prerequisites.justfile**: Shared prerequisite verification (kubectl, helm, helmfile, cluster connectivity)
- **monitoring.justfile**: Common monitoring commands (port-forwards for Prometheus/Grafana, password retrieval)
- **config.yaml**: Global configuration (monitoring namespace, default resources)

**Usage in justfiles:**
```just
import? "../../common/prerequisites.justfile"
import? "../../common/monitoring.justfile"

# Then create wrapper recipes that use the helpers:
create-namespace:
    @just _create-namespace {{NAMESPACE}}
```

### LLM-D Common (`llm-d/common/`)

Utilities specific to llm-d deployments:

- **config.yaml**: LLM-D configuration (chart versions, image tags, namespace prefix)
- **llm-d-helpers.justfile**: LLM-D specific recipes (submodule verification, common deployment patterns)

**Usage in llm-d justfiles:**
```just
import? "../common/llm-d-helpers.justfile"

# Helper recipes (prefixed with _) are available:
# - _create-namespace NAMESPACE
# - _delete-namespace NAMESPACE
# - _wait-for-deployment NAMESPACE DEPLOYMENT_NAME TIMEOUT
# - _get-logs-by-label NAMESPACE LABEL_SELECTOR TAIL
# - _follow-logs-by-label NAMESPACE LABEL_SELECTOR
```

## Quick Start

### Prerequisites

1. Initialize llm-d submodule (from repository root):
   ```bash
   cd /path/to/rocm-icms
   just setup-submodules
   ```

2. Verify prerequisites:
   ```bash
   just verify-prereqs
   ```

### Deploy a Guide

Navigate to the deployment directory and run setup:

```bash
# Tiered Prefix Cache
cd deployments/llm-d/tiered-prefix-cache
just setup

# Inference Scheduling
cd deployments/llm-d/inference-scheduling
just setup
```

## Standard Justfile Recipes

All deployments follow consistent naming conventions:

### Deployment
- `just deploy` - Deploy full stack
- `just setup` - Complete setup (create namespace + deploy + wait)
- `just teardown` - Remove all resources
- `just cleanup` - Complete cleanup (teardown + stop port-forwards)

### Monitoring
- `just status` - Check deployment status
- `just logs` - View logs
- `just logs-follow` - Follow logs
- `just port-forward-start` - Start port forwards
- `just port-forward-stop` - Stop port forwards

### Verification
- `just verify-prereqs` - Verify prerequisites
- `just wait` - Wait for deployment readiness
- `just check` - Quick health check

### Namespace Management
- `just create-namespace` - Create namespace if needed
- `just delete-namespace` - Delete namespace (WARNING: removes all resources)

## Adding a New Deployment

### LLM-D Based

1. Create directory: `deployments/llm-d/your-deployment/`
2. Create `justfile` importing common utilities:
   ```just
   import? "../../common/prerequisites.justfile"
   import? "../../common/monitoring.justfile"
   import? "../common/llm-d-helpers.justfile"
   ```
3. Create deployment manifests or Helmfile
4. Create `README.md` documentation
5. Update `deployments/llm-d/README.md`

### Custom (Non-LLM-D)

1. Create directory: `deployments/custom/your-deployment/`
2. Create `justfile` importing common utilities:
   ```just
   import? "../../common/prerequisites.justfile"
   import? "../../common/monitoring.justfile"
   ```
3. Create your deployment automation
4. Create `README.md` documentation
5. Update root `justfile` to list your deployment

## Best Practices

1. **Reuse common utilities**: Don't duplicate prerequisite checks or monitoring commands
2. **Consistent naming**: Follow the standard recipe naming conventions
3. **Documentation**: Always include a README.md with prerequisites, quick start, and troubleshooting
4. **Configuration**: Use environment variables for namespace and other configurable values
5. **Error handling**: Use `|| exit 1` for critical commands, `-` prefix for cleanup commands
6. **User feedback**: Use emojis and clear messages for deployment progress

## Configuration

### Namespaces

Each deployment uses its own default namespace but can be overridden:

```bash
NAMESPACE=my-custom-namespace just deploy
```

### Monitoring Stack

All deployments assume a monitoring stack exists in `llm-d-monitoring` namespace:
- Prometheus
- Grafana

Override if needed:
```bash
MONITORING_NAMESPACE=my-monitoring just deploy
```

## References

- [llm-d Project](https://github.com/llm-d/llm-d)
- [Just Command Runner](https://github.com/casey/just)
- [Helmfile](https://helmfile.readthedocs.io/)
- [Kustomize](https://kustomize.io/)
