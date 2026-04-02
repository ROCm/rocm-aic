# LLM-D Deployments for AMD GPUs

Organization:
```
llm-d/
├── benchmarks/              # Benchmark sweep framework
├── common/                  # LLM-D specific utilities
├── inference-scheduling/    # Intelligent request routing deployment
├── monitoring/              # Monitoring & Dashboarding
├── setup/                   # setup helper scripts
└── tiered-prefix-cache/     # KV cache offloading deployment
```

## Setup

### LLM-D Submodule
Initialize from repository root:
```bash
cd /path/to/rocm-icms
just setup-submodules
```

### Pre-requisite tools

- `kubectl` - Kubernetes CLI; not installed automatically
- `helm` - Helm package manager
- `helmfile` - (inference-scheduling only)
- `just` - Command runner

Install the pre-requisites tools:
```bash
cd setup/

# Interactive pre-requisites installation (helm, helmfile, just, etc...)
./prereqs.sh

# install all pre-requisites without prompting
./prereqs.sh -y
```

### Minikube setup

For testing purpose, a minikube environment can be setup as follow:

```bash
cd setup/

# Install the minikube binary
just minikube-install

# Start a minikube instance with support for AMD GPUs
just minikube-start

# Setup minikube to use AMD GPUs; one-time install after minikube-start is invoked
just minikube-setup

# Stop the minikube instance
just minikube-stop
```


### Kubernetes cluster setup for LLM-D

This step assumes the following Kubernetes cluster setup:
- AMD GPU nodes with `amd.com/gpu` resource

Setup LLM-D deployments pre-requisites:
* Huggingface token secret in `llm-d-hf-token`.
* Prometheus and Grafana to `llm-d-monitoring` namespace.
* ISTIO gateway to `istio-system` namespace.

This step requires the `HF_TOKEN` environment variable to be set.

```bash
cd setup/

# Setup LLM-D pre-requisites on the cluster (Minikube or else)
just llm-d-setup
```

## Available Deployments

### 1. Tiered Prefix Cache

**Directory**: `tiered-prefix-cache/`

Offload KV cache from GPU HBM to CPU RAM for improved cache hit rates and performance.

**Features:**
- Two connector variants: offloading-connector (default) and lmcache-connector
- 100GB CPU cache tier (configurable)
- Intelligent prefix cache-aware routing
- AMD ROCm vLLM image

**Quick Start:**
```bash
cd tiered-prefix-cache
just setup
```

**Documentation**: [tiered-prefix-cache/README.md](tiered-prefix-cache/README.md)

### 2. Inference Scheduling

**Directory**: `inference-scheduling/`

Intelligent load-aware and prefix-cache aware request routing for reduced tail latency.

**Features:**
- Multiple vLLM replica deployment (8 replicas default)
- InferencePool with smart routing
- Load balancing based on queue length and cache hits
- AMD ROCm vLLM image
- Helmfile-based deployment

**Quick Start:**
```bash
cd inference-scheduling
just setup
```

**Documentation**: [inference-scheduling/README.md](inference-scheduling/README.md)

## Benchmarking

The `benchmarks` folder contains scripts to perform benchmarking sweeps targeting tiered prefix cache deployments.

Check `benchmarks/README.md`

## Common Characteristics

All llm-d deployments:
- Target AMD GPUs exclusively
- Use AMD ROCm-optimized container images (`ghcr.io/llm-d/llm-d-rocm:v0.5.1`)
- Include monitoring with Prometheus + Grafana
- Provide consistent justfile recipes
- Reference llm-d submodule for base manifests


## Deployment Methods

### Tiered Prefix Cache
Uses **Kustomize + Helm**:
- Gateway and vLLM: Kustomize overlays referencing llm-d submodule bases
- InferencePool: Helm chart

### Inference Scheduling
Uses **Helmfile**:
- 3 Helm charts: llm-d-infra, inferencepool, llm-d-modelservice
- Single Helmfile orchestrates all releases
- AMD-specific value files

## Configuration

### Namespace Convention

Default namespaces:
- Tiered Prefix Cache: `llm-d-tiered-prefix-cache-amd`
- Inference Scheduling: `llm-d-inference-scheduling-amd`

Override with environment variable:
```bash
NAMESPACE=my-namespace just deploy
```

### Common Configuration

Shared LLM-D settings in `common/config.yaml`:
- Chart versions
- Image tags
- AMD GPU defaults

### Deployment-Specific Configuration

Each deployment has its own manifests or value files:
- Tiered Prefix Cache: `manifests/` directory
- Inference Scheduling: `values/` directory

## Architecture

### Tiered Prefix Cache
```
User → Istio Gateway → InferencePool → vLLM (GPU HBM + CPU RAM caches)
```

### Inference Scheduling
```
User → Istio Gateway → InferencePool → Multiple vLLM Replicas
                            ↓
                  (Smart routing based on load + cache)
```

## Monitoring

All deployments integrate with:
- **Prometheus**: Metrics collection from vLLM pods
- **Grafana**: Visualization dashboards
- **PodMonitor**: Automatic scraping configuration

Access after port-forwarding:
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

## Troubleshooting

### Common Issues

**Submodule not initialized:**
```bash
# From rocm-icms root
just setup-submodules
```

**AMD GPUs not found:**
```bash
kubectl get nodes -o custom-columns=NAME:.metadata.name,AMD_GPUS:.status.capacity.amd\.com/gpu
```

**Insufficient resources:**
- Reduce replicas in inference-scheduling
- Reduce CPU cache size in tiered-prefix-cache
- Use smaller models

**Monitoring not available:**
```bash
# Check monitoring namespace
kubectl get pods -n llm-d-monitoring

# Deploy monitoring stack from llm-d guides
cd ../../../submodules/llm-d/guides/prereq/monitoring
kubectl apply -k .
```

## Performance Tuning

### Tiered Prefix Cache
- Adjust `cpu_bytes_to_use` for CPU cache size
- Tune InferencePool scorer weights
- Modify GPU memory utilization

### Inference Scheduling
- Adjust vLLM parameters in value files
- Tune InferencePool scorer weights

## Adding More Deployments

To add another llm-d guide:

1. Create directory: `deployments/llm-d/new-guide/`
2. Create `justfile` with standard recipes
3. Import common utilities:
   ```just
   import? "../../common/prerequisites.justfile"
   import? "../../common/monitoring.justfile"
   import? "../common/llm-d-helpers.justfile"
   ```
4. Create manifests or Helmfile referencing llm-d submodule
5. Document in `README.md`
6. Update this file

## References

- [llm-d GitHub Repository](https://github.com/llm-d/llm-d)
- [llm-d Guides](../../../submodules/llm-d/guides/)
- [vLLM Documentation](https://docs.vllm.ai/)
- [InferencePool Documentation](https://github.com/llm-d-incubation/inference-pool)
- [AMD ROCm](https://www.amd.com/en/products/software/rocm.html)
