**ROCm AMD Infinity Context — Short Outline**

- **Abstract** — Introduces the KV cache memory-wall problem in large-scale LLM inference and
  previews AIC as AMD's tiered, shared cache solution integrating with LMCache, NIXL, vLLM, and
  SGLang.

- **1. Introduction & Motivation** — Growing model sizes and context lengths are pushing AMD
  Instinct HBM to its limits, and repeated long prompts in agentic/RAG workloads mean clusters
  re-compute the same KV tensors constantly. A shared, disaggregated KV cache tier is the missing
  primitive — analogous to NVIDIA's ICMS/CMX but built for the open ROCm ecosystem.

- **2. Background: The AMD Instinct Inference Stack** — Covers the MI300X memory subsystem, ROCm
  primitives (HIP, hipFile), vLLM's paged attention and KV connector API, SGLang's radix
  attention, and the fundamental problem of per-replica cache isolation in multi-GPU deployments.

- **3. AIC Architecture** — Defines the four-tier hierarchy (HBM → CPU DRAM → local NVMe →
  network storage) with expected latency and capacity at each level. Describes the LMCache chunk
  model: content-addressed KV blocks with sha256_cbor keying and blending for partial cache hits.

- **4. LMCache: Cache Management Layer** — LMCache sits between vLLM's attention engine and
  external storage, managing chunk lifecycle (store, retrieve, evict) across multiple backends:
  CPU RAM, POSIX disk, hipFile/GDS, NIXL, and AIS. Covers vLLM connector integration, runtime
  storage-mode switching without restart, and the planned SGLang connector.

- **5. NIXL: Cross-Node KV Transport** — NIXL is a pluggable tensor I/O library providing a
  unified API over POSIX, AIS object store, and eventually RDMA/UCX for cross-node KV migration.
  On ROCm, the AIS_MT thread-pool plugin bridges the hipFile batch API gap; the long-term payoff
  is zero-CPU KV transfer between prefill and decode nodes.

- **6. Cluster Orchestration with llm-d** — The llm-d InferencePool routes requests using
  composite scoring (queue length, HBM prefix cache hits, CPU cache hits) to maximize locality
  without cross-node fetches. Two deployment patterns: tiered prefix cache (HBM + CPU DRAM via
  Kustomize/Helm) and inference scheduling (multi-replica with cache-aware dispatch only).

- **7. Benchmarking & Results** — Five benchmark workloads covering controlled hit-rate sweeps,
  Gutenberg long-context prefill, Claude Code agent trace replay, and synthetic I/O
  characterization. Key finding: even modest real-world hit rates (20–40%) deliver substantial P99
  TTFT improvements; 100% hit rate reduces prefill latency from ~6 s to ~150 ms on representative
  long-context inputs.

- **8. Deployment Guide Summary** — Practical entry points for three deployment models:
  single-node Docker (hipfile and NIXL recipes), Kubernetes with llm-d (`just setup`), and Slurm
  with automatic NVMe discovery. Covers Ansible-based cluster inventory as the provisioning
  starting point.

- **9. Roadmap** — Near-term: hipFile batch API completion, SGLang LMCache connector, NIXL RDMA
  hardening. Longer-term: cluster-wide AIS shared cache, speculative KV prefetching, and KV
  quantization (INT8/FP8) to multiply effective cache capacity.

- **10. Conclusion** — Summarizes how the layered AIC architecture (LMCache + NIXL + llm-d +
  vLLM/SGLang) delivers a composable, open-source KV cache tier for AMD Instinct clusters and
  points to the rocm-aic repository for hands-on exploration.

- **Appendices** — Glossary, benchmark reproducibility notes, configuration variable reference
  (VLH/VLN/ADE env vars), hardware compatibility matrix (MI300X, MI250X, RX 9070 XT, Strix
  Halo), and full reference list.
