# ROCm-ICMS Deployments

This directory contains deployment automation for various LLM inference guides targeting AMD GPUs.

## Directory Structure

```
deployments/
└── llm-d/                       # LLM-D based deployments
    ├── common/                  # LLM-D specific utilities
    ├── tiered-prefix-cache/     # KV cache offloading deployment
    └── inference-scheduling/    # Intelligent request routing deployment
```

## Organization

### LLM-D Deployments (`llm-d/`)

Deployments based on the [llm-d project](https://github.com/llm-d/llm-d), which provides well-tested "lit paths" for LLM inference on Kubernetes.

**Available deployments:**
- **tiered-prefix-cache**: Offload KV cache from GPU HBM to CPU RAM
- **inference-scheduling**: Load-aware and cache-aware request routing

See [`llm-d/README.md`](llm-d/README.md) for details.
