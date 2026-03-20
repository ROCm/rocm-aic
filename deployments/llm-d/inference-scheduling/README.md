# Inference Scheduling Deployment for AMD GPUs

Intelligent load-aware and prefix-cache aware request scheduling for reduced tail latency and increased throughput on AMD GPUs.

## Overview

This deployment creates an intelligent inference serving stack using:
- **Gateway**: Istio gateway for traffic routing
- **InferencePool**: Endpoint Picker Pod (EPP) for smart request routing
- **Model Servers**: Multiple vLLM replicas serving the same model

The InferencePool routes requests based on:
- **Load awareness**: Queue lengths and request counts
- **Prefix cache awareness**: Cache hit probability per endpoint
- **Resource utilization**: KV cache and GPU memory usage

## Architecture

```
User Request
    ↓
Istio Gateway (port 8080)
    ↓
InferencePool Endpoint Picker
    ↓ (intelligent routing)
vLLM Replica 1  |  vLLM Replica 2  |  ... |  vLLM Replica N
```

## Deployment Method

This deployment uses the **llm-d guide's helmfile** to minimize code duplication:

**Helmfile**: `submodules/llm-d/guides/inference-scheduling/helmfile.yaml.gotmpl`
**Environment**: `amd`
**Values Layering**:
1. llm-d AMD defaults (`ms-inference-scheduling/values_amd.yaml`)
2. rocm-icms image override (via `--set decode.containers[0].image=...`)
3. Optional custom (`values/amd-default.yaml` or `values/amd-small.yaml`)

This approach:
- ✅ Eliminates helmfile duplication (zero maintenance)
- ✅ Auto-updates chart versions from llm-d
- ✅ Preserves rocm-icms customization flexibility
- ✅ Clear separation of concerns

## Prerequisites

- Kubernetes cluster with AMD GPU nodes
- `kubectl`, `helm`, `helmfile`, and `just` installed
- llm-d submodule initialized (run `just setup-submodules` from repository root)
- Monitoring stack deployed (Prometheus + Grafana in `llm-d-monitoring` namespace)
- HuggingFace token secret (if using private models)

### Create HuggingFace Token Secret

```bash
kubectl create secret generic llm-d-hf-token \
  --from-literal=token=YOUR_HF_TOKEN \
  -n llm-d-inference-scheduling-amd
```

## Quick Start

```bash
# From this directory
just setup
```

This will:
1. Create namespace `llm-d-inference-scheduling-amd`
2. Deploy using llm-d guide's helmfile with AMD environment
3. Apply rocm-icms base overrides (vLLM image)
4. Deploy llm-d infrastructure (Istio gateway)
5. Deploy InferencePool with intelligent routing
6. Deploy 8x vLLM model server replicas
7. Deploy monitoring (PodMonitor)
8. Wait for all components to be ready

**Note**: This deployment uses the llm-d guide's `helmfile.yaml.gotmpl` directly to minimize code duplication while allowing rocm-icms specific customizations through values files.

## Available Recipes

### Deployment

- `just deploy` - Deploy full stack using Helmfile
- `just setup` - Complete setup (create namespace + deploy + wait)
- `just sync` - Update existing deployment without recreating
- `just diff` - Show Helmfile diff before applying

### Monitoring

- `just status` - Check deployment status and Helm releases
- `just logs` - View model server logs
- `just logs-follow` - Follow model server logs
- `just port-forward-start` - Start port forwards (Gateway:8080, Prometheus:9090, Grafana:3000)
- `just port-forward-stop` - Stop all port forwards
- `just test-metrics` - Test metrics endpoint from a model server pod
- `just check` - Quick pod and release count check
- `just show-config` - Parse and display vLLM configuration from logs (deduplicated)
- `just show-config-all` - Show config for each pod separately
- `just show-config-pod POD_NAME` - Show config for specific pod

### Cleanup

- `just teardown` - Remove all resources using Helmfile (keeps namespace)
- `just cleanup` - Complete cleanup (teardown + stop port forwards)
- `just delete-namespace` - Delete namespace (WARNING: removes everything)

## Configuration

### Namespace

Default namespace: `llm-d-inference-scheduling-amd`

To use a different namespace:
```bash
NAMESPACE=my-custom-namespace just deploy
```

### Values File

Default values file: `values/amd-default.yaml`

To use a different values file:
```bash
VALUES_FILE=values/amd-small.yaml just deploy
```

Combine namespace and values file:
```bash
NAMESPACE=my-namespace VALUES_FILE=values/amd-small.yaml just deploy
```

### Value Files and Layering

rocm-icms uses a **layered values approach** to minimize duplication while allowing customization:

#### Layer 1: llm-d AMD Defaults (from submodule)
- Base configuration from `submodules/llm-d/guides/inference-scheduling/ms-inference-scheduling/values_amd.yaml`
- Provides standard AMD GPU configuration
- Automatically updated when submodule updates

#### Layer 2: rocm-icms Base (always applied)
- Overrides vLLM image: `vllm/vllm-openai-rocm:v0.17.0`
- Applied via `--set` flag to avoid schema validation issues

#### Layer 3: Optional Custom Overrides
Specify with `VALUES_FILE` environment variable:

**1. amd-default.yaml** (Production)
- **Model**: Qwen/Qwen3-32B (32B parameters)
- **Replicas**: 8 model servers
- **GPUs per replica**: 2 (tensor parallelism)
- **Memory per replica**: 100Gi
- **Use for**: Production deployments with high throughput requirements

**2. amd-small.yaml** (Testing/Development)
- **Model**: Qwen/Qwen2.5-7B-Instruct (7B parameters)
- **Replicas**: 2 model servers
- **GPUs per replica**: 1
- **Memory per replica**: 50Gi
- **Use for**: Development, testing, or resource-constrained environments

### Switching Configurations

Use the `VALUES_FILE` environment variable:

```bash
# Use llm-d defaults + rocm-icms base only
just deploy

# Use llm-d defaults + rocm-icms base + production overrides
VALUES_FILE=amd-default.yaml just deploy

# Use llm-d defaults + rocm-icms base + testing overrides
VALUES_FILE=amd-small.yaml just deploy
```

## Components

This deployment uses the **llm-d guide's helmfile** from `submodules/llm-d/guides/inference-scheduling/helmfile.yaml.gotmpl` with the AMD environment.

### 1. llm-d-infra Chart
Deploys the gateway infrastructure.

**Chart**: `llm-d-infra/llm-d-infra:v1.3.6` (from llm-d)

Components:
- Istio Gateway
- HTTPRoute configuration
- Destination rules

### 2. InferencePool Chart
Deploys the Endpoint Picker for intelligent routing.

**Chart**: `inferencepool:v1.3.1` (from llm-d)

Scorers (from llm-d guide):
- Load-aware scoring
- Prefix cache-aware scoring
- Queue length scoring
- KV cache utilization scoring

### 3. llm-d-modelservice Chart
Deploys vLLM model server replicas.

**Chart**: `llm-d-modelservice:v0.4.7` (from llm-d)

Configuration:
- Image: `vllm/vllm-openai-rocm:v0.17.0` (rocm-icms override)
- vLLM with AMD ROCm support
- Configurable replicas and parallelism
- PodMonitor for metrics

**Chart versions are automatically managed by llm-d** - update the submodule to get new versions.

## Monitoring

After starting port forwards (`just port-forward-start`):

**Istio Gateway**: http://localhost:8080
- Send inference requests here
- Example:
  ```bash
  curl -X POST http://localhost:8080/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "Qwen/Qwen3-32B", "prompt": "Hello", "max_tokens": 50}'
  ```

**Prometheus**: http://localhost:9090
- View vLLM metrics
- Query request latency, cache hit rates
- Monitor resource usage across replicas

**Grafana**: http://localhost:3000
- Username: admin
- Password: Run `just grafana-password` (from common monitoring utilities)

## Troubleshooting

### Pods not starting
```bash
# Check status
just status

# View logs from all model servers
just logs

# Check a specific pod
kubectl logs -n llm-d-inference-scheduling-amd deployment/ms-inference-scheduling-amd -c vllm
```

### Helmfile errors
```bash
# Show what would change
just diff

# Check Helm releases
helm list -n llm-d-inference-scheduling-amd

# Manual cleanup if needed
helm uninstall infra-inference-scheduling-amd -n llm-d-inference-scheduling-amd
helm uninstall gaie-inference-scheduling-amd -n llm-d-inference-scheduling-amd
helm uninstall ms-inference-scheduling-amd -n llm-d-inference-scheduling-amd
```

### Model download failures
```bash
# Verify HuggingFace token secret exists
kubectl get secret llm-d-hf-token -n llm-d-inference-scheduling-amd

# Check model server logs for download progress
just logs-follow
```

## Performance Tuning

### Adjusting vLLM Parameters
Edit the values file and add vLLM arguments:

```yaml
decode:
  containers:
    - name: "vllm"
      args:
        - "--gpu-memory-utilization=0.95"
        - "--max-model-len=4096"  # Adjust context length
        - "--max-num-seqs=256"     # Adjust batch size
```

Then redeploy:
```bash
just deploy
```

### InferencePool Scoring Weights
Edit llm-d guide values to adjust routing behavior:
- See: `../../../submodules/llm-d/guides/inference-scheduling/gaie-inference-scheduling/values.yaml`

## Advanced Usage

### Custom Models

To use a different model, create a custom values file:

1. Copy one of the existing values files:
   ```bash
   cp values/amd-default.yaml values/my-model.yaml
   ```

2. Edit `values/my-model.yaml` and update only what differs from llm-d defaults:
   - `modelArtifacts.uri` - Your model path
   - `modelArtifacts.name` - Model name
   - `decode.parallelism.tensor` - Tensor parallelism size
   - `decode.replicas` - Number of replicas
   - Container resources as needed

3. Deploy with your custom values:
   ```bash
   VALUES_FILE=my-model.yaml just deploy
   ```

**Note**: Values are layered (llm-d defaults → rocm-icms-base → your custom values), so you only need to specify what changes.

### Multi-Node Deployments

Enable multi-node support in values:
```yaml
multinode: true
```

### Custom Namespace

```bash
just deploy --namespace my-namespace
```

## References

- [llm-d Inference Scheduling Guide](../../../submodules/llm-d/guides/inference-scheduling/README.md)
- [vLLM Documentation](https://docs.vllm.ai/)
- [InferencePool Documentation](https://github.com/llm-d-incubation/inference-pool)
- [Helmfile Documentation](https://helmfile.readthedocs.io/)
