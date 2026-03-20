# Tiered Prefix Cache Deployment for AMD GPUs

This deployment enables KV cache offloading from GPU HBM to CPU RAM for improved cache hit rates and performance on AMD GPUs.

## Overview

Tiered prefix caching allows vLLM to offload KV cache from expensive GPU memory (HBM) to cheaper storage tiers:
- **Tier 1 (HBM)**: GPU memory for active requests
- **Tier 2 (CPU RAM)**: CPU memory for cached prefixes

This deployment supports two connector variants:
1. **offloading-connector** (default): vLLM's native CPU offloading
2. **lmcache-connector**: LMCache framework for CPU offloading

## Prerequisites

- Kubernetes cluster with AMD GPU nodes
- `kubectl`, `helm`, and `just` installed
- llm-d submodule initialized (run `just setup-submodules` from repository root)
- Monitoring stack deployed (Prometheus + Grafana in `llm-d-monitoring` namespace)

## Quick Start

### Using Offloading Connector (Default)

```bash
# From this directory
just setup
```

This will:
1. Create namespace `llm-d-tiered-prefix-cache-amd`
2. Deploy Istio gateway
3. Deploy vLLM with offloading connector
4. Deploy InferencePool with tiered cache scoring
5. Deploy monitoring (PodMonitor)
6. Wait for all components to be ready

### Using LMCache Connector

```bash
just setup-lmcache
```

## Available Recipes

### Deployment

- `just deploy` - Deploy full stack with offloading-connector (default)
- `just deploy-offloading` - Explicitly deploy with offloading-connector
- `just deploy-lmcache` - Deploy with LMCache connector
- `just setup` - Complete setup (create namespace + deploy + wait)
- `just setup-lmcache` - Complete setup with LMCache

### Monitoring

- `just status` - Check deployment status
- `just logs` - View model server logs
- `just logs-follow` - Follow model server logs
- `just port-forward-start` - Start port forwards (Gateway:8080, Prometheus:9090, Grafana:3000)
- `just port-forward-stop` - Stop all port forwards
- `just test-metrics` - Test metrics endpoint
- `just check` - Quick pod count check
- `just show-config` - Parse and display vLLM configuration from logs (deduplicated)
- `just show-config-all` - Show config for each pod separately
- `just show-config-pod POD_NAME` - Show config for specific pod

### Cleanup

- `just teardown` - Remove all resources (keep namespace)
- `just teardown-offloading` - Teardown offloading variant
- `just teardown-lmcache` - Teardown LMCache variant
- `just cleanup` - Complete cleanup (teardown + stop port forwards)
- `just delete-namespace` - Delete namespace (WARNING: removes everything)

## Configuration

### Namespace

Default namespace: `llm-d-tiered-prefix-cache-amd`

To use a different namespace:
```bash
NAMESPACE=my-custom-namespace just deploy
```

### Connector Variant

Choose between:
- **offloading-connector**: Best for simple CPU offloading, native vLLM support
- **lmcache-connector**: Advanced caching with LMCache framework

Run `just show-variant` to see current variant.

### CPU Cache Size

The CPU cache size is configured in the vLLM manifests. Default: 100GB

To modify:
1. Edit `manifests/vllm/{connector}/kustomization.yaml`
2. Update `cpu_bytes_to_use` in the vLLM configuration
3. Update `lruCapacityPerServer` in `manifests/inferencepool/values.yaml`

Formula: `lruCapacityPerServer ≈ cpu_bytes_to_use / 2560`

## Components

### 1. Istio Gateway
Routes traffic to the InferencePool endpoint picker.

### 2. vLLM Model Server
Serves the LLM with KV cache offloading to CPU.

Configuration:
- **Image**: `ghcr.io/llm-d/llm-d-rocm:v0.5.1`
- **GPUs**: 2x AMD GPUs per pod
- **Memory**: 400Gi RAM per pod
- **CPU Cache**: 100GB (configurable)

### 3. InferencePool
Intelligent request routing with prefix cache awareness.

Scorers:
- Queue Scorer (weight: 2.0) - Load balancing
- KV Cache Utilization Scorer (weight: 2.0) - Memory awareness
- GPU Prefix Cache Scorer (weight: 1.0) - HBM cache hits
- CPU Prefix Cache Scorer (weight: 1.0) - CPU tier cache hits

### 4. PodMonitor
Prometheus metrics scraping for vLLM pods.

## Architecture

```
User Request
    ↓
Istio Gateway (port 8080)
    ↓
InferencePool Endpoint Picker
    ↓ (intelligent routing based on cache + load)
vLLM Model Server(s)
    ├─ HBM Cache (GPU)
    └─ CPU Cache (RAM - 100GB)
```

## Monitoring

After starting port forwards (`just port-forward-start`):

**Prometheus**: http://localhost:9090
- View vLLM metrics
- Query cache hit rates
- Monitor resource usage

**Grafana**: http://localhost:3000
- Username: admin
- Password: Run `just grafana-password`

**Istio Gateway**: http://localhost:8080
- Send requests to the deployment

### Configuration Inspection

Extract vLLM configuration from running pods:

```bash
# Deduplicated configuration across all pods
just show-config

# Configuration for each pod separately
just show-config-all

# Specific pod
just show-config-pod llm-d-model-server-abc123
```

Example output (deduplicated):
```json
{
  "deployment": "multiple_pods",
  "pod_count": 2,
  "pod_names": ["llm-d-model-server-abc", "llm-d-model-server-xyz"],
  "vllm_version": "0.17.1",
  "non_default_args": {
    "model": "Qwen/Qwen3-32B",
    "tensor_parallel_size": 2,
    "max_num_seqs": 1024,
    "gpu_memory_utilization": 0.95
  },
  "model_config": {
    "max_model_len": 40960,
    "attention_backend": "Triton"
  },
  "parallelism": {
    "tensor_parallel_size": 2,
    "pipeline_parallel_size": 1,
    "data_parallel_size": 1
  },
  "kv_cache": {
    "total_gpu_cache_tokens": 491360,
    "total_available_memory_gib": 59.98,
    "max_concurrency_per_pod": 6.0
  },
  "connectors": {
    "OffloadingConnector": {
      "offloading_spec": "CPUOffloadingSpec",
      "cpu_bytes_to_use": 107374182400,
      "cpu_cache_size_gb": 100.0
    }
  },
  "performance": {
    "avg_model_loading_time_seconds": 30.5,
    "avg_torch_compile_time_seconds": 19.7
  }
}
```

## Troubleshooting

### Pods not starting
```bash
# Check pod status
just status

# View logs
just logs

# Check events
kubectl get events -n llm-d-tiered-prefix-cache-amd --sort-by='.lastTimestamp'
```

### Insufficient GPU resources
Verify AMD GPU availability:
```bash
kubectl get nodes -o custom-columns=NAME:.metadata.name,AMD_GPUS:.status.capacity.amd\.com/gpu
```

### Port forward issues
```bash
# Stop all port forwards
just port-forward-stop

# Restart
just port-forward-start
```

### Connector variant confusion
```bash
# Check which variant is deployed
kubectl get deployment llm-d-model-server -n llm-d-tiered-prefix-cache-amd -o yaml | grep -A 5 "kv-transfer-config"
```

## Performance Tuning

### Increase CPU Cache
1. Edit vLLM manifest to increase `cpu_bytes_to_use`
2. Increase pod memory allocation accordingly
3. Update InferencePool `lruCapacityPerServer`

### Adjust Scorer Weights
Edit `manifests/inferencepool/values.yaml` to tune routing behavior:
- Increase queue weight for better load balancing
- Increase cache weights for better cache hit rates

## References

- [llm-d Tiered Prefix Cache Guide](../../../submodules/llm-d/guides/tiered-prefix-cache/README.md)
- [vLLM KV Cache Offloading Documentation](https://docs.vllm.ai/)
- [InferencePool Documentation](https://github.com/llm-d-incubation/inference-pool)
