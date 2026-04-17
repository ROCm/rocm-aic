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

### Available Deployments

#### LLM-D Based Deployments

**[Tiered Prefix Cache](deployments/llm-d/tiered-prefix-cache/)** - KV cache offloading
- Offload GPU HBM cache to CPU RAM
- Two connector variants: offloading-connector, lmcache-connector
- Intelligent prefix cache-aware routing
- Method: Kustomize + Helm

**[Inference Scheduling](deployments/llm-d/inference-scheduling/)** - Intelligent routing
- vLLM replicas with smart load balancing
- Prefix cache-aware request routing
- Reduced tail latency and increased throughput
- Method: Helmfile (3 charts)

**[Benchmarking](deployments/llm-d/benchmarks/)** - Benchmarking framework
- Declarative benchmark sweep configurations
- Result post-processing and plotting

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
│   │   ├── benchmarks/
│   │   ├── monitoring/
│   │   ├── tiered-prefix-cache/
│   │   └── inference-scheduling/
│   └── custom/                 # Custom deployments (add your own)
├── scripts/                    # Utilities
├── tools/                      # Build and development tools
└── vendors/                    # Vendor-specific configurations
```
## Documentation

- **[Deployments Overview](deployments/README.md)** - Architecture and organization
- **[LLM-D Deployments](deployments/llm-d/README.md)** - LLM-D guide deployments
- **[Tiered Prefix Cache Guide](deployments/llm-d/tiered-prefix-cache/README.md)** - KV cache offloading
- **[Inference Scheduling Guide](deployments/llm-d/inference-scheduling/README.md)** - Smart routing

## References

- [NVIDIA ICMS technical blog][icms]
- [WEKA blog on BlueField-4 and ICMS][weka]
- [llm-d Project](https://github.com/llm-d-incubation/llm-d)
- [vLLM Documentation](https://docs.vllm.ai/)
- [AMD ROCm](https://www.amd.com/en/products/software/rocm.html)

[icms]: https://developer.nvidia.com/blog/introducing-nvidia-bluefield-4-powered-inference-context-memory-storage-platform-for-the-next-frontier-of-ai/
[weka]: https://www.weka.io/blog/ai-ml/demystifying-the-bluefield-4-inference-context-memory-storage-announcement/
