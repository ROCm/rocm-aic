# rocm-icms

AMD internal exploration of storage infrastructure for ROCm-based GPU clusters,
inspired by NVIDIA's [Inference Context Memory Storage (ICMS)][icms] platform
(recently rebranded **CMX**). ICMS uses BlueField-4 DPUs and disaggregated NVMe
flash to create a shared KV-cache tier for large-scale AI inference; this repo
investigates analogous approaches on AMD hardware.

## LLM Deployment Infrastructure

This repository provides production-ready deployment automation for LLM inference on AMD GPUs, with a focus on:
- **Tiered KV-cache storage**: Offload from GPU HBM to CPU RAM and storage
- **Intelligent request routing**: Load-aware and cache-aware scheduling
- **Modular architecture**: Reusable components for custom deployments

### Quick Start

```bash
# Initialize submodules
just setup-submodules

# Verify prerequisites
just verify-prereqs

# List available deployments
just list

# Deploy tiered prefix cache
cd deployments/llm-d/tiered-prefix-cache
just setup

# Deploy inference scheduling
cd deployments/llm-d/inference-scheduling
just setup
```

### Available Deployments

#### LLM-D Based Deployments

**[Tiered Prefix Cache](deployments/llm-d/tiered-prefix-cache/)** - KV cache offloading
- Offload GPU HBM cache to 100GB CPU RAM
- Two connector variants: offloading-connector, lmcache-connector
- Intelligent prefix cache-aware routing
- Method: Kustomize + Helm

**[Inference Scheduling](deployments/llm-d/inference-scheduling/)** - Intelligent routing
- vLLM replicas with smart load balancing
- Prefix cache-aware request routing
- Reduced tail latency and increased throughput
- Method: Helmfile (3 charts)

**[Monitoring](deployments/llm-d/monitoring/)** - Grafana dashboard management
- Load llm-d default dashboards (6 dashboards)
- Load rocm-icms custom dashboards
- Automatic Grafana discovery

See [deployments/README.md](deployments/README.md) for details.

## Repository Structure

```
rocm-icms/
├── justfile                    # Root automation (setup, verification)
├── submodules/
│   └── llm-d/                  # LLM-D project (submodule)
├── deployments/
│   ├── common/                 # Shared utilities (all deployments)
│   ├── llm-d/                  # LLM-D based deployments
│   │   ├── tiered-prefix-cache/
│   │   └── inference-scheduling/
│   └── custom/                 # Custom deployments (add your own)
├── scripts/
│   ├── setup-submodules.sh     # Initialize llm-d submodule
│   ├── verify-cluster.sh       # Kubernetes cluster verification
│   └── common-functions.sh     # Bash utility functions
├── tools/                      # Build and development tools
└── vendors/                    # Vendor-specific configurations
```

## Prerequisites

### Tools
- [just](https://github.com/casey/just) - Command runner
- [kubectl](https://kubernetes.io/docs/tasks/tools/) - Kubernetes CLI
- [helm](https://helm.sh/docs/intro/install/) - Kubernetes package manager
- [helmfile](https://helmfile.readthedocs.io/) - Declarative Helm releases

### Kubernetes Cluster
- AMD GPU nodes with `amd.com/gpu` resource
- Sufficient CPU and memory (varies by deployment)
- StorageClass for persistent volumes

### Monitoring (Recommended)
- Prometheus + Grafana in `llm-d-monitoring` namespace
- See llm-d monitoring setup guide

## Documentation

- **[Deployments Overview](deployments/README.md)** - Architecture and organization
- **[LLM-D Deployments](deployments/llm-d/README.md)** - LLM-D guide deployments
- **[Custom Deployments](deployments/custom/README.md)** - Add your own deployments
- **[Tiered Prefix Cache Guide](deployments/llm-d/tiered-prefix-cache/README.md)** - KV cache offloading
- **[Inference Scheduling Guide](deployments/llm-d/inference-scheduling/README.md)** - Smart routing

## Common Operations

```bash
# Repository setup
just setup-submodules              # Initialize llm-d submodule
just update-submodules             # Update to latest llm-d
just verify-prereqs                # Verify tools and cluster

# Information
just list                          # Show available deployments
just info                          # Detailed deployment information
just health-check                  # Repository health status
just check-drift                   # Check for drift with llm-d upstream

# Drift checking
just check-drift                   # Check all deployments
just check-drift-deployment inference-scheduling  # Check specific deployment
just check-drift-json drift.json   # Generate JSON report

# Deployment (from deployment directory)
just deploy                        # Deploy infrastructure
just status                        # Check status
just logs                          # View logs
just port-forward-start            # Access services locally
just teardown                      # Remove resources
```

## Drift Detection

rocm-icms maintains custom configurations that may drift from llm-d upstream. Use drift checking to stay aligned:

```bash
# Check for drift with llm-d
just check-drift

# Check specific deployment
just check-drift-deployment inference-scheduling

# Generate machine-readable report
just check-drift-json drift-report.json
```

**What is checked**:
- Chart versions (llm-d-infra, inferencepool, llm-d-modelservice)
- Container images and tags
- Helm repository URLs
- Key configuration values

**Severity Levels**:
- 🔴 HIGH: May cause compatibility issues, review immediately
- 🟡 MEDIUM: Consider updating to stay aligned
- 🟢 LOW: Minor differences, monitor for future updates
- ℹ️ INFO: Informational only, likely intentional

## Adding New Deployments

### LLM-D Based
1. Create directory in `deployments/llm-d/`
2. Create justfile importing common utilities
3. Reference llm-d submodule for base manifests
4. Document in README.md

### Custom (Non-LLM-D)
1. Create directory in `deployments/custom/`
2. Create justfile importing common utilities
3. Add your deployment automation
4. Document in README.md

See [deployments/README.md](deployments/README.md) for details.

## References

- [NVIDIA ICMS technical blog][icms]
- [WEKA blog on BlueField-4 and ICMS][weka]
- [llm-d Project](https://github.com/llm-d-incubation/llm-d)
- [vLLM Documentation](https://docs.vllm.ai/)
- [AMD ROCm](https://www.amd.com/en/products/software/rocm.html)

[icms]: https://developer.nvidia.com/blog/introducing-nvidia-bluefield-4-powered-inference-context-memory-storage-platform-for-the-next-frontier-of-ai/
[weka]: https://www.weka.io/blog/ai-ml/demystifying-the-bluefield-4-inference-context-memory-storage-announcement/
