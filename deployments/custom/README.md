# Custom Deployments

This directory is for deployment automation that is **not** based on llm-d guides.

## Purpose

Use this directory for:
- Custom inference solutions
- Non-llm-d deployment patterns
- Experimental deployments
- Vendor-specific integrations
- Alternative inference frameworks

## Structure

Each custom deployment should have its own subdirectory:

```
custom/
├── README.md                    # This file
├── my-deployment-1/
│   ├── justfile                 # Deployment automation
│   ├── README.md                # Documentation
│   └── (manifests, configs, etc.)
└── my-deployment-2/
    ├── justfile
    ├── README.md
    └── ...
```

## Creating a Custom Deployment

### 1. Create Directory Structure

```bash
mkdir -p deployments/custom/my-deployment
cd deployments/custom/my-deployment
```

### 2. Create Justfile

Create a `justfile` with standard recipes, importing common utilities:

```just
# Import shared utilities
import? "../../common/prerequisites.justfile"
import? "../../common/monitoring.justfile"

# Default namespace
NAMESPACE := env_var_or_default('NAMESPACE', 'my-deployment')

# Default recipe
default:
    @just --list

# Deploy
deploy: verify-all create-namespace
    @echo "Deploying my-deployment..."
    # Your deployment commands here

# Teardown
teardown:
    @echo "Tearing down my-deployment..."
    # Your teardown commands here

# Status
status:
    @echo "Deployment status:"
    kubectl get all -n {{NAMESPACE}}

# Create namespace
create-namespace:
    @just create-namespace {{NAMESPACE}}

# Delete namespace
delete-namespace:
    @just delete-namespace {{NAMESPACE}}

# Setup (deploy + wait)
setup: deploy
    @echo "Setup complete!"

# Cleanup
cleanup: teardown
    @echo "Cleanup complete!"
```

### 3. Create README.md

Document your deployment:

```markdown
# My Custom Deployment

Brief description of what this deploys.

## Prerequisites

List requirements here.

## Quick Start

\`\`\`bash
just setup
\`\`\`

## Configuration

Document configuration options.

## Troubleshooting

Common issues and solutions.
```

### 4. Add Deployment Automation

Add your Kubernetes manifests, Helm charts, or other deployment files.

### 5. Update Root Documentation

Update the root `justfile` to include your deployment in the list.

## Available Common Utilities

Custom deployments can leverage shared utilities from `../common/`:

### Prerequisites (`common/prerequisites.justfile`)

```just
import? "../../common/prerequisites.justfile"

# Then use:
verify-kubectl
verify-helm
verify-helmfile
verify-kustomize
verify-all
```

### Monitoring (`common/monitoring.justfile`)

```just
import? "../../common/monitoring.justfile"

# Then use:
grafana-info
stop-monitoring-port-forwards
_port-forward-prometheus
_port-forward-grafana
```

## Best Practices

1. **Consistent naming**: Follow llm-d deployment naming conventions
2. **Reuse utilities**: Don't duplicate prerequisite checks
3. **Documentation**: Always include comprehensive README
4. **Error handling**: Use proper exit codes
5. **Namespace isolation**: Use unique namespaces
6. **Monitoring**: Integrate with existing monitoring stack when possible

## Examples

### Example 1: Custom vLLM Deployment

```
custom/vllm-custom/
├── justfile
├── README.md
├── manifests/
│   ├── deployment.yaml
│   ├── service.yaml
│   └── kustomization.yaml
└── values/
    └── custom-config.yaml
```

### Example 2: Alternative Inference Framework

```
custom/tensorrt-llm/
├── justfile
├── README.md
├── helm/
│   └── tensorrt-chart/
└── scripts/
    └── setup.sh
```

### Example 3: Multi-Framework Stack

```
custom/multi-framework/
├── justfile
├── README.md
├── vllm/
├── tensorrt/
└── triton/
```

## Integration with LLM-D Common

While custom deployments can't use `llm-d/common/llm-d-helpers.justfile`, they can still use global utilities from `../common/`.

## Testing

Before committing your custom deployment:

1. Verify it deploys successfully
2. Test teardown works cleanly
3. Ensure documentation is complete
4. Validate it doesn't conflict with other deployments

## Contributing

If your custom deployment becomes stable and useful:
- Consider contributing it back to llm-d project
- Share with the community
- Document lessons learned

## Support

For custom deployments:
- Check existing llm-d deployments for patterns
- Refer to Kubernetes best practices
- Leverage common utilities to avoid duplication
