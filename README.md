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

The [`ansible/`][ansible-dir] directory contains Ansible
playbooks for managing GPU cluster nodes.

- **[discover.yml][discover-yml]** -- inventories each
  node and produces a per-host JSON report covering GPUs,
  NVMe drives, RDMA NICs, AIS status, Linux kernel
  version, ROCm version, and DKMS module status. A second
  play compares all hosts and flags differences.
- **[provision.yml][provision-yml]** -- installs a base
  set of developer packages via the
  [sbates130272.batesste][galaxy] Galaxy collection's
  `fave_packages` role, then layers on project-specific
  packages defined in
  [`group_vars/gpu_nodes.yml`][group-vars].

## References

[b-lmc]: benchmarks/ttft-lmcache/
[b-lcp]: benchmarks/ttft-llamacpp/
[r-lmc]: benchmarks/ttft-lmcache/README.md
[r-lcp]: benchmarks/ttft-llamacpp/README.md
[patch]: benchmarks/ttft-llamacpp/patches/0001-cache-disk.patch
[ansible-dir]: ansible/
[discover-yml]: ansible/discover.yml
[provision-yml]: ansible/provision.yml
[galaxy]: https://galaxy.ansible.com/ui/repo/published/sbates130272/batesste/
[group-vars]: ansible/inventory/group_vars/gpu_nodes.yml

- [NVIDIA ICMS technical blog][icms]
- [WEKA blog on BlueField-4 and ICMS][weka]

[icms]: https://developer.nvidia.com/blog/introducing-nvidia-bluefield-4-powered-inference-context-memory-storage-platform-for-the-next-frontier-of-ai/
[weka]: https://www.weka.io/blog/ai-ml/demystifying-the-bluefield-4-inference-context-memory-storage-announcement/
