<!--
Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
-->

# rocm-aic

AMD internal exploration of storage infrastructure for ROCm-based GPU
clusters, inspired by NVIDIA's [Inference Context Memory Storage
(ICMS)][icms] platform (recently rebranded **CMX**). ICMS uses
BlueField-4 DPUs and disaggregated NVMe flash to create a shared KV-cache
tier for large-scale AI inference; this repo investigates analogous
approaches on AMD hardware.

## Host Python dependencies

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Benchmark Docker images install slimmer, image-local
`requirements.txt` files under `benchmarks/ttft-*` (see each
benchmark README). Scripts such as `recipies/vllm-radeon/scripts/
test-aic.py` need only `openai` unless you use the full tool stack.

## Documentation map

| Area | Path | README |
|------|------|--------|
| TTFT benchmark (vLLM + LMCache) | [benchmarks/ttft-lmcache][b-lmc] | [README][r-lmc] |
| TTFT benchmark (llama.cpp) | [benchmarks/ttft-llamacpp][b-lcp] | [README][r-lcp] |
| vLLM + LMCache Radeon recipe | [recipies/vllm-radeon][r-vr] | [README][r-vr] |
| LMCache patch index | [recipies/vllm-radeon/patches][r-patches] | [README][r-patches] |
| ROCm inference stack image | [recipies/rocm-inference-stack][r-ris] | [README][r-ris] |
| LMCache IO simulator | [tools/lmcache-io-tester][t-lit] | [README][t-lit-readme] |
| LMCache IO detailed usage | [tools/lmcache-io-tester/docs/USAGE.md][t-lit-usage] | — |
| amdgpu-dkms repack tool | [tools/amdgpu-dkms][t-dkms] | [README][t-dkms-readme] |
| WEKA FS PoC | [vendors/weka][v-weka] | [README][v-weka-readme] |
| Dell vLLM + LMCache + hipFile | [vendors/dell/vllm-lmcache-hipfile][v-dell] | [README][v-dell-readme] |
| Ansible discovery / provision | [ansible][ansible-dir] | [site.yml][site-yml] |

## Benchmarks

The `benchmarks/` directory contains Dockerized TTFT
(Time-To-First-Token) benchmarks that measure the impact of KV-cache
offload on inference latency across different storage tiers (CPU RAM,
NVMe, hipFile/AIS, NFS).

| Benchmark | Engine | GPU support | README |
|-----------|--------|-------------|--------|
| [ttft-lmcache][b-lmc] | vLLM + LMCache | Instinct (CDNA) | [README][r-lmc] |
| [ttft-llamacpp][b-lcp] | llama.cpp | Instinct + Radeon | [README][r-lcp] |

The llama.cpp benchmark includes a `--cache-disk` patch for automatic
disk-tier prompt caching (see [patches/0001-cache-disk.patch][patch]).

## Host discovery and provisioning

The [`ansible/`][ansible-dir] directory contains **`site.yml`** at the
repo `ansible/` root plus [`ansible/playbooks/`][playbooks-dir] for
discovery. From `ansible/`, run `ansible-playbook site.yml` (or
`ansible-playbook playbooks/discover.yml` for discovery only).

- **[discover.yml][discover-yml]** — inventories each node and produces a
  per-host JSON report covering GPUs, NVMe drives, RDMA NICs, AIS status,
  Linux kernel version, ROCm version, and DKMS module status. A second play
  compares all hosts and flags differences (plus AIS and rocminfo fields)
  and writes `summary.json`.
- **[site.yml][site-yml]** — imports discovery from `playbooks/` and runs
  the local **`host_setup`** role (**user_setup**, **fave_packages**,
  **rdma_setup**, **rocm_setup** from [sbates130272.batesste][galaxy], plus
  **extra_packages** from [`group_vars/gpu_nodes.yml`][group-vars]). Use
  `--tags discover`, `--tags provision`, or tags such as `--tags rocm` or
  `--tags user_setup` to limit work. Run `ansible-playbook site.yml
  --list-tags` to list tags.

## CI

GitHub Actions under [`.github/workflows/`][gh-workflows] include spellcheck,
lint, benchmark-adjacent recipes (`vllm-radeon-*`, `lmcache-io-tester`), and
`test-amdgpu-dkms`.

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

[b-lmc]: benchmarks/ttft-lmcache/
[b-lcp]: benchmarks/ttft-llamacpp/
[r-lmc]: benchmarks/ttft-lmcache/README.md
[r-lcp]: benchmarks/ttft-llamacpp/README.md
[patch]: benchmarks/ttft-llamacpp/patches/0001-cache-disk.patch
[r-vr]: recipies/vllm-radeon/
[r-patches]: recipies/vllm-radeon/patches/README.md
[r-ris]: recipies/rocm-inference-stack/README.md
[t-lit]: tools/lmcache-io-tester/
[t-lit-readme]: tools/lmcache-io-tester/README.md
[t-lit-usage]: tools/lmcache-io-tester/docs/USAGE.md
[t-dkms]: tools/amdgpu-dkms/
[t-dkms-readme]: tools/amdgpu-dkms/README.md
[v-weka]: vendors/weka/
[v-weka-readme]: vendors/weka/README.md
[v-dell]: vendors/dell/vllm-lmcache-hipfile/
[v-dell-readme]: vendors/dell/vllm-lmcache-hipfile/README.md
[ansible-dir]: ansible/
[playbooks-dir]: ansible/playbooks/
[discover-yml]: ansible/playbooks/discover.yml
[site-yml]: ansible/site.yml
[galaxy]: https://galaxy.ansible.com/ui/repo/published/sbates130272/batesste/
[group-vars]: ansible/inventory/group_vars/gpu_nodes.yml
[gh-workflows]: .github/workflows/

- [NVIDIA ICMS technical blog][icms]
- [WEKA blog on BlueField-4 and ICMS][weka-blog]
- [llm-d Project](https://github.com/llm-d-incubation/llm-d)
- [vLLM Documentation](https://docs.vllm.ai/)
- [AMD ROCm](https://www.amd.com/en/products/software/rocm.html)

[icms]: https://developer.nvidia.com/blog/introducing-nvidia-bluefield-4-powered-inference-context-memory-storage-platform-for-the-next-frontier-of-ai/
[weka-blog]: https://www.weka.io/blog/ai-ml/demystifying-the-bluefield-4-inference-context-memory-storage-announcement/
