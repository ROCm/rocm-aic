# LLM-D Deployment Design Guide

This document explains the design principles and patterns used for llm-d deployments in rocm-icms.

## Table of Contents

- [Kustomize Layering](#kustomize-layering)
- [Design Principles](#design-principles)
- [Customization Patterns](#customization-patterns)
- [File Organization](#file-organization)
- [Best Practices](#best-practices)

## Kustomize Layering

### How Kustomize Layering Works

The llm-d deployments use **layered Kustomize overlays** to enable customization without modifying the upstream llm-d submodule:

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 1: llm-d Base Recipe (recipes/vllm/amd)              │
│ - Defines base deployment structure                         │
│ - Sets AMD ROCm image                                       │
│ - Applies AMD-specific labels                               │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 2: llm-d Guide Manifest                               │
│         (tiered-prefix-cache/cpu/.../offloading-connector)  │
│ - References Layer 1 as base                                │
│ - Adds guide-specific vLLM arguments                        │
│ - Sets KV transfer config for offloading                    │
│ - Configures resource requests (GPU, memory)                │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 3: rocm-icms Overlay                                  │
│         (deployments/llm-d/tiered-prefix-cache/manifests)   │
│ - References Layer 2 as resource                            │
│ - OVERRIDES specific fields with custom values             │
│ - Adds deployment-specific customizations                   │
│ - Keeps customizations separate from submodule              │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
                   Final Deployment
```

### Layer Details

#### Layer 1: Base Recipe
**Location**: `submodules/llm-d/guides/recipes/vllm/amd/kustomization.yaml`

**Purpose**: Provides AMD-specific base configuration

**Contents**:
```yaml
resources:
  - ../base
images:
  - name: INFERENCE_SERVER_IMAGE
    newName: vllm/vllm-openai-rocm
    newTag: v0.17.1
patches:
  - # AMD labels
```

#### Layer 2: Guide Manifest
**Location**: `submodules/llm-d/guides/tiered-prefix-cache/cpu/manifests/vllm/offloading-connector-amd/kustomization.yaml`

**Purpose**: Guide-specific configuration (tiered prefix cache with CPU offloading)

**Contents**:
```yaml
resources:
  - ../../../../../recipes/vllm/amd  # References Layer 1
patches:
  - # vLLM serve command with KV transfer config
  - # Resource requests (2 GPUs, 400GB RAM)
```

#### Layer 3: rocm-icms Overlay
**Location**: `deployments/llm-d/tiered-prefix-cache/manifests/vllm/offloading-connector-amd/kustomization.yaml`

**Purpose**: Deployment-specific customizations

**Contents**:
```yaml
resources:
  - ../../../../../../submodules/llm-d/guides/.../offloading-connector-amd  # References Layer 2
patches:
  - # Custom vLLM arguments
  - # Deployment-specific overrides
```

### Why This Layering?

**Separation of Concerns**:
- **Layer 1**: Hardware vendor concerns (AMD vs NVIDIA)
- **Layer 2**: Use case concerns (tiered caching, inference scheduling)
- **Layer 3**: Deployment environment concerns (rocm-icms specific)

**Benefits**:
1. **No Submodule Modifications**: rocm-icms customizations don't require forking llm-d
2. **Easy Updates**: Update llm-d submodule without losing customizations
3. **Clear Ownership**: Each layer has a clear maintainer
4. **Reusability**: Lower layers can be reused across deployments

## Design Principles

### 1. Separation of Deployment Types

```
deployments/
├── llm-d/        # LLM-D based deployments
└── custom/       # Non-llm-d deployments
```

**Rationale**: Clear distinction between deployments that follow llm-d patterns and custom solutions.

### 2. Three Levels of Utilities

```
deployments/
├── common/                # Global utilities (all deployments)
├── llm-d/
│   └── common/           # LLM-D specific utilities
└── llm-d/deployment-x/   # Deployment-specific logic
```

**Rationale**: Avoid duplication while allowing deployment-specific customization.

### 3. Reference, Don't Copy

All llm-d deployments **reference** the submodule rather than copying files:

```yaml
# Good: Reference submodule
resources:
  - ../../../../../../submodules/llm-d/guides/...

# Bad: Copy files into rocm-icms
# (This would require manual updates)
```

**Rationale**: Stay in sync with upstream llm-d improvements and fixes.

### 4. Helper Recipes with Underscore Prefix

```just
# Public recipe (in deployment justfile)
create-namespace:
    @just _create-namespace {{NAMESPACE}}

# Helper recipe (in common/llm-d-helpers.justfile)
_create-namespace NAMESPACE:
    @kubectl create namespace {{NAMESPACE}} 2>/dev/null || echo "exists"
```

**Rationale**: Avoid recipe name collisions when importing justfiles.

## Customization Patterns

### Pattern 1: Override vLLM Arguments

**Use Case**: Change model, add arguments, tune performance

**Method**: JSON Patch in kustomization.yaml

```yaml
patches:
  - target:
      kind: Deployment
      name: llm-d-model-server
    patch: |-
      - op: replace
        path: /spec/template/spec/containers/0/args/0
        value: |-
          exec vllm serve \
            your-model/name \
            --your-custom-args
```

### Pattern 2: Override Resources

**Use Case**: Change GPU count, memory allocation

**Method**: JSON Patch for resource limits/requests

```yaml
patches:
  - target:
      kind: Deployment
      name: llm-d-model-server
    patch: |-
      - op: replace
        path: /spec/template/spec/containers/0/resources/requests/memory
        value: 500G
```

### Pattern 3: Add Environment Variables

**Use Case**: Configure runtime behavior

**Method**: JSON Patch to add env vars

```yaml
patches:
  - target:
      kind: Deployment
      name: llm-d-model-server
    patch: |-
      - op: add
        path: /spec/template/spec/containers/0/env
        value:
          - name: VLLM_LOGGING_LEVEL
            value: DEBUG
```

### Pattern 4: Override InferencePool Scorers

**Use Case**: Tune routing weights

**Method**: Helm values file

```yaml
# manifests/inferencepool/values.yaml
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: queue-scorer
    weight: 3.0  # Increased from 2.0
```

## File Organization

### Tiered Prefix Cache Structure

```
tiered-prefix-cache/
├── justfile                                    # Deployment automation
├── README.md                                   # User documentation
└── manifests/
    ├── vllm/
    │   ├── offloading-connector-amd/
    │   │   └── kustomization.yaml             # Layer 3: rocm-icms overlay
    │   └── lmcache-connector-amd/
    │       └── kustomization.yaml             # Layer 3: rocm-icms overlay
    └── inferencepool/
        └── values.yaml                         # Helm values for InferencePool
```

**Referenced Layers** (in submodule):
```
submodules/llm-d/guides/
├── recipes/vllm/amd/                          # Layer 1: AMD base
│   └── kustomization.yaml
└── tiered-prefix-cache/cpu/manifests/vllm/
    ├── offloading-connector-amd/              # Layer 2: Guide manifest
    │   └── kustomization.yaml
    └── lmcache-connector-amd/                 # Layer 2: Guide manifest
        └── kustomization.yaml
```

### Inference Scheduling Structure

```
inference-scheduling/
├── justfile                                    # Deployment automation
├── README.md                                   # User documentation
├── helmfile.yaml                               # Orchestrates 3 Helm charts
└── values/
    ├── amd-default.yaml                        # Production config
    ├── amd-small.yaml                          # Testing config
    └── amd-custom.yaml.example                 # Template
```

## Best Practices

### 1. Keep Customizations Minimal

**Good**: Override only what you need
```yaml
patches:
  - op: replace
    path: /spec/template/spec/containers/0/args/0
    value: |-
      exec vllm serve model --custom-arg value
```

**Bad**: Copy entire manifest and modify
```yaml
# Don't copy the whole deployment and change it
```

### 2. Document Customizations

Always add comments explaining why you override defaults:

```yaml
patches:
  # Override: Increase max sequences for higher throughput testing
  - op: replace
    path: /spec/template/spec/containers/0/args/0
    value: |-
      exec vllm serve model --max-num-seq 2048
```

### 3. Test Before Committing

```bash
# Validate kustomization builds correctly
kustomize build ./manifests/vllm/offloading-connector-amd

# Dry-run deployment
kubectl apply -k ./manifests/vllm/offloading-connector-amd --dry-run=client
```

### 4. Version Pin Important Overrides

When overriding critical configurations, document which llm-d version you're targeting:

```yaml
# Compatible with llm-d v0.5.1
# If updating submodule, verify this override is still needed
patches:
  - op: replace
    path: /spec/template/spec/containers/0/args/0
```

### 5. Use Strategic Merge for Complex Changes

For complex changes, consider strategic merge patch instead of JSON patch:

```yaml
patchesStrategicMerge:
  - |-
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: llm-d-model-server
    spec:
      template:
        spec:
          containers:
          - name: vllm
            env:
            - name: MY_VAR
              value: my-value
```

## Troubleshooting Layering Issues

### Issue: "Error: accumulating resources"

**Cause**: Incorrect path to base resource

**Solution**: Verify relative path to submodule
```yaml
resources:
  - ../../../../../../submodules/llm-d/guides/...
  # Count the ../ based on your file location
```

### Issue: Patch doesn't apply

**Cause**: Path doesn't exist in base

**Solution**: Build base first to see structure
```bash
kustomize build ../../../../../../submodules/llm-d/guides/... | less
# Find the correct path
```

### Issue: Image not overridden

**Cause**: Image override happens in wrong layer

**Solution**: Image overrides should be in Layer 1 (AMD base recipe)
- Edit `submodules/llm-d/guides/recipes/vllm/amd/kustomization.yaml`
- Not in Layer 3 overlay

## References

- [Kustomize Documentation](https://kustomize.io/)
- [JSON Patch RFC 6902](https://tools.ietf.org/html/rfc6902)
- [Strategic Merge Patch](https://github.com/kubernetes/community/blob/master/contributors/devel/sig-api-machinery/strategic-merge-patch.md)
- [llm-d Repository](https://github.com/llm-d/llm-d)
