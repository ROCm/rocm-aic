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
benchmark README). Scripts such as `recipies/vllm-lmcache-hipfile/scripts/
test-aic.py` need only `openai` unless you use the full tool stack.

## Documentation map

| Area | Path | README |
|------|------|--------|
| TTFT benchmark (vLLM + LMCache) | [benchmarks/ttft-lmcache][b-lmc] | [README][r-lmc] |
| TTFT benchmark (llama.cpp) | [benchmarks/ttft-llamacpp][b-lcp] | [README][r-lcp] |
| vLLM + LMCache hipfile recipe | [recipies/vllm-lmcache-hipfile][r-vr] | [README][r-vr] |
| LMCache patch index | [recipies/vllm-lmcache-hipfile/patches][r-patches] | [README][r-patches] |
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
lint, benchmark-adjacent recipes (`vllm-lmcache-hipfile-*`, `lmcache-io-tester`), and
`test-amdgpu-dkms`.

## References

[b-lmc]: benchmarks/ttft-lmcache/
[b-lcp]: benchmarks/ttft-llamacpp/
[r-lmc]: benchmarks/ttft-lmcache/README.md
[r-lcp]: benchmarks/ttft-llamacpp/README.md
[patch]: benchmarks/ttft-llamacpp/patches/0001-cache-disk.patch
[r-vr]: recipies/vllm-lmcache-hipfile/
[r-patches]: recipies/vllm-lmcache-hipfile/patches/README.md
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

[icms]: https://developer.nvidia.com/blog/introducing-nvidia-bluefield-4-powered-inference-context-memory-storage-platform-for-the-next-frontier-of-ai/
[weka-blog]: https://www.weka.io/blog/ai-ml/demystifying-the-bluefield-4-inference-context-memory-storage-announcement/
