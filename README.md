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

## Host Discovery and Provisioning

The [`ansible/`][ansible-dir] directory contains Ansible playbooks under
[`ansible/playbooks/`][playbooks-dir] for managing GPU cluster nodes. From
`ansible/`, run `ansible-playbook playbooks/<playbook>.yml`.

- **[discover.yml][discover-yml]** -- inventories each node and produces a
  per-host JSON report covering GPUs, NVMe drives, RDMA NICs, AIS status,
  Linux kernel version, ROCm version, and DKMS module status. A second play
  compares all hosts and flags differences (plus AIS and rocminfo fields) and
  writes `summary.json`.
- **[provision.yml][provision-yml]** -- applies the local `host_setup` role,
  which runs **user_setup**, **fave_packages**, **rdma_setup**, and
  **rocm_setup** from the [sbates130272.batesste][galaxy] collection, then
  installs **extra_packages** from [`group_vars/gpu_nodes.yml`][group-vars].
- **[site.yml][site-yml]** -- imports discover and provision; use
  `--tags discover`, `--tags provision`, or component tags such as `--tags
  rocm` or `--tags user_setup` to limit work. Run `ansible-playbook
  playbooks/site.yml --list-tags` to list tags.

## References

[b-lmc]: benchmarks/ttft-lmcache/
[b-lcp]: benchmarks/ttft-llamacpp/
[r-lmc]: benchmarks/ttft-lmcache/README.md
[r-lcp]: benchmarks/ttft-llamacpp/README.md
[patch]: benchmarks/ttft-llamacpp/patches/0001-cache-disk.patch
[ansible-dir]: ansible/
[playbooks-dir]: ansible/playbooks/
[discover-yml]: ansible/playbooks/discover.yml
[provision-yml]: ansible/playbooks/provision.yml
[site-yml]: ansible/playbooks/site.yml
[galaxy]: https://galaxy.ansible.com/ui/repo/published/sbates130272/batesste/
[group-vars]: ansible/inventory/group_vars/gpu_nodes.yml

- [NVIDIA ICMS technical blog][icms]
- [WEKA blog on BlueField-4 and ICMS][weka]

[icms]: https://developer.nvidia.com/blog/introducing-nvidia-bluefield-4-powered-inference-context-memory-storage-platform-for-the-next-frontier-of-ai/
[weka]: https://www.weka.io/blog/ai-ml/demystifying-the-bluefield-4-inference-context-memory-storage-announcement/
