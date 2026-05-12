# rocm-icms

AMD internal exploration of storage infrastructure for ROCm-based GPU clusters,
inspired by NVIDIA's [Inference Context Memory Storage (ICMS)][icms] platform
(recently rebranded **CMX**). ICMS uses BlueField-4 DPUs and disaggregated NVMe
flash to create a shared KV-cache tier for large-scale AI inference; this repo
investigates analogous approaches on AMD hardware.

## Benchmarks

The `benchmarks/` directory contains Dockerized TTFT
(Time-To-First-Token) benchmarks that measure the impact
of KV-cache offload on inference latency across different
storage tiers (CPU RAM, NVMe, hipFile/AIS, NFS).

| Benchmark | Engine | GPU support | README |
|-----------|--------|-------------|--------|
| [ttft-lmcache][b-lmc] | vLLM + LMCache | Instinct (CDNA) | [README][r-lmc] |
| [ttft-llamacpp][b-lcp] | llama.cpp | Instinct + Radeon | [README][r-lcp] |

The llama.cpp benchmark includes a `--cache-disk` patch
for automatic disk-tier prompt caching (see
[patches/0001-cache-disk.patch][patch]).

## Recipes

The `recipies/` directory holds self-contained container recipes
(Dockerfile and Makefile assets, plus scripts) for experiments and integration
workflows, separate from the `benchmarks/` TTFT harnesses. Each
recipe directory has its own README.

| Recipe | Focus | README |
|--------|-------|--------|
| [rocm-inference-stack][d-rec-inf] | ROCm inference stack image (NIXL, LMCache, tooling) | [README][r-rec-inf] |
| [ROCm vLLM + LMCache + hipFile][d-vllm-kurt] | Kurt-derived vLLM image build and run scripts | [README][r-vllm-kurt] |

## Host Discovery and Provisioning

The [`ansible/`][ansible-dir] directory contains **`site.yml`** at the repo
`ansible/` root plus [`ansible/playbooks/`][playbooks-dir] for discovery. From
`ansible/`, run `ansible-playbook site.yml` (or `ansible-playbook
playbooks/discover.yml` for discovery only).

- **[discover.yml][discover-yml]** -- inventories each node and produces a
  per-host JSON report covering GPUs, NVMe drives, RDMA NICs, AIS status,
  Linux kernel version, ROCm version, and DKMS module status. A second play
  compares all hosts and flags differences (plus AIS and rocminfo fields) and
  writes `summary.json`.
- **[site.yml][site-yml]** -- imports discovery from `playbooks/` and runs the
  local **`host_setup`** role (**user_setup**, **fave_packages**,
  **rdma_setup**, **rocm_setup** from [sbates130272.batesste][galaxy], plus
  **extra_packages** from [`group_vars/gpu_nodes.yml`][group-vars]). Use
  `--tags discover`, `--tags provision`, or tags such as `--tags rocm` or
  `--tags user_setup` to limit work. Run `ansible-playbook site.yml
  --list-tags` to list tags.

## References

[b-lmc]: benchmarks/ttft-lmcache/
[b-lcp]: benchmarks/ttft-llamacpp/
[r-lmc]: benchmarks/ttft-lmcache/README.md
[r-lcp]: benchmarks/ttft-llamacpp/README.md
[patch]: benchmarks/ttft-llamacpp/patches/0001-cache-disk.patch
[d-rec-inf]: recipies/rocm-inference-stack/
[d-vllm-kurt]: recipies/vllm-from-kurt/
[r-rec-inf]: recipies/rocm-inference-stack/README.md
[r-vllm-kurt]: recipies/vllm-from-kurt/README.md
[ansible-dir]: ansible/
[playbooks-dir]: ansible/playbooks/
[discover-yml]: ansible/playbooks/discover.yml
[site-yml]: ansible/site.yml
[galaxy]: https://galaxy.ansible.com/ui/repo/published/sbates130272/batesste/
[group-vars]: ansible/inventory/group_vars/gpu_nodes.yml

- [NVIDIA ICMS technical blog][icms]
- [WEKA blog on BlueField-4 and ICMS][weka]

[icms]: https://developer.nvidia.com/blog/introducing-nvidia-bluefield-4-powered-inference-context-memory-storage-platform-for-the-next-frontier-of-ai/
[weka]: https://www.weka.io/blog/ai-ml/demystifying-the-bluefield-4-inference-context-memory-storage-announcement/
