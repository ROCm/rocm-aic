# White Paper Outline: ROCm AMD Infinity Context (AIC)
## A Shared, Low-Latency KV Cache Tier for Large-Scale Distributed Inference on AMD GPUs

---

## Abstract

*~300 words. Summarizes the memory-wall problem in large-scale LLM inference, introduces AIC as AMD's answer to a disaggregated, shared KV cache tier, and previews how it integrates with LMCache, NIXL, vLLM, and SGLang to deliver measurable TTFT and throughput improvements across real workloads.*

---

## 1. Introduction and Motivation

### 1.1 The Inference Scaling Inflection Point
- Growth in model size (70B → 400B+ parameters), context length (8k → 1M tokens), and concurrent user demand are pushing inference infrastructure to its limits.
- Cost of ownership is increasingly dominated by GPU hours, not training; operational efficiency of inference clusters is now the key economic lever.
- Enterprise deployments (agentic pipelines, RAG, multi-turn chat) exhibit high prompt re-use — the same long system prompts and document contexts are processed repeatedly across thousands of requests per hour.

### 1.2 The KV Cache Bottleneck
- Transformer attention requires materializing KV tensors for every token in the context at decode time; at large context lengths this dominates both memory and compute.
- On AMD Instinct GPUs (MI300X: 192 GB HBM3), KV cache competes directly with model weights for high-bandwidth memory — a 70B model leaves only a fraction of HBM for cache.
- KV cache is *not* shared between replicas in a standard multi-GPU deployment: every worker re-computes or independently caches the same prompts, wasting both cycles and bandwidth.
- As sequence lengths grow, the prefill phase (where KV tensors are generated) dominates TTFT; recovering cached KV from a lower-cost tier can eliminate this cost entirely on cache hits.

### 1.3 The NVIDIA Precedent and the AMD Opportunity
- NVIDIA's Inference Context Memory Storage (ICMS / CMX) platform uses BlueField-4 DPUs and disaggregated NVMe to create a cluster-wide KV cache tier for H100/H200 deployments.
- AMD Instinct clusters require an analogous but AMD-native solution: one that integrates with ROCm, respects the AMD memory hierarchy (HBM3, CPU DRAM, local NVMe, network-attached storage), and connects to the open-source inference ecosystem rather than a proprietary stack.
- This paper presents *ROCm AMD Infinity Context (AIC)* — AMD's exploration and implementation of a tiered, shared KV cache for Instinct GPU clusters.

---

## 2. Background: The AMD Instinct Inference Stack

### 2.1 AMD Instinct GPU Architecture Relevant to Inference
- MI300X memory subsystem: 192 GB HBM3 at ~5.3 TB/s; HBM bandwidth as the principal resource constraint for long-context inference.
- CDNA3 compute: matrix-core throughput, attention kernel efficiency (Triton-based flash attention on ROCm).
- ROCm software stack: HIP, hipFile (GPU Direct Storage analogue), RDMA over InfiniBand/RoCE.

### 2.2 vLLM on ROCm
- vLLM as the dominant open-source LLM serving engine; ROCm port maturity, paged attention, continuous batching.
- Key vLLM primitives: prefix cache (GPU HBM), KV transfer config (`--kv-transfer-config`), connector abstraction (offloading-connector, lmcache-connector).

### 2.3 SGLang on ROCm
- SGLang as a complementary inference framework emphasizing structured generation and radix attention for prefix reuse.
- ROCm support status; relevance to KV cache sharing at the framework level.

### 2.4 The Multi-Replica Problem
- A typical production deployment runs N vLLM (or SGLang) replicas behind a load balancer; each replica has a fully independent in-HBM prefix cache.
- Cache-aware routing (e.g. llm-d InferencePool scoring: queue length, HBM prefix cache hits, CPU cache hits) can improve locality but cannot solve the fundamental isolation between replicas.
- A *shared, out-of-HBM* cache tier that all replicas can read from and write to is the missing primitive.

---

## 3. AMD Infinity Context: Architecture

### 3.1 Design Goals
- **Latency**: cache hit latency must be low enough that loading KV from the tier costs less than re-prefilling; targets: CPU DRAM O(10s ms), local NVMe O(100s ms) for typical long-context workloads.
- **Shareability**: KV chunks stored by replica A must be readable by replica B without data movement through the application layer.
- **Composability**: the tier should be pluggable into existing vLLM and SGLang deployments via standard connector APIs, not require engine modification.
- **AMD-native I/O path**: leverage hipFile / ROCm GPU Direct Storage where available to minimize CPU involvement in DMA transfers between GPU HBM and NVMe.

### 3.2 The Cache Tier Hierarchy
```
Tier 0 — GPU HBM (per-replica, private)
   ~10s GB available after weights; sub-millisecond access
Tier 1 — CPU DRAM (per-node, potentially shared across replicas on same host)
   100s GB; O(1–10 ms) transfer to/from HBM over PCIe
Tier 2 — Local NVMe (per-node or per-rack, shared via network filesystem)
   1–100 TB; O(10–200 ms) with O_DIRECT / hipFile paths
Tier 3 — Network-attached storage (cluster-wide, fully shared)
   Petabyte-scale; O(100 ms – 1 s); WEKA FS, AIS/MinIO, NFS
```
- Discussion of how each tier's latency and capacity interacts with prefill cost savings at different cache hit rates.
- Relationship to hardware: MI300X host topology (NUMA, PCIe gen5, InfiniBand HDR/NDR).

### 3.3 KV Chunk Model
- LMCache chunk abstraction: KV cache split into fixed-size chunks keyed by a hash of the token sequence prefix (sha256_cbor algorithm for reproducibility).
- Chunk blending: non-contiguous cached chunks can be merged with freshly computed KV, enabling partial cache hits rather than requiring full prefix matches.
- Eviction policy: LRU per server; `lruCapacityPerServer` tuned to CPU cache allocation.
- On-disk layout: `obj_*.bin` pool files (NIXL) or `.data` files (LMCache hipFile backend); pool pre-allocation for O_DIRECT alignment.

---

## 4. LMCache: The KV Cache Management Layer

### 4.1 What LMCache Does
- Open-source KV cache offload framework that sits between vLLM's attention engine and external storage.
- Implements the `kv-transfer-config` connector interface consumed by vLLM; analogous connector for SGLang in progress.
- Manages chunk lifecycle: store on prefill completion, retrieve on cache-hit prefill, evict on capacity pressure.

### 4.2 Backend Abstraction
- **CPU backend**: in-process DRAM store; fast but node-local and volatile.
- **POSIX/disk backend**: standard filesystem I/O to local NVMe or NFS; portable, no special drivers.
- **hipFile / GdsBackend**: AMD GPU Direct Storage path using ROCm `hipFile` API; eliminates CPU bounce buffer on DMA from HBM to NVMe.
- **NIXL NixlStorageBackend**: pluggable transport layer (see Section 5).
- **AIS backend**: AMD AIS (AI Storage) object store via hipFile; cluster-wide shared cache with GPU Direct path.

### 4.3 Integration with vLLM
- `--kv-transfer-config` YAML structure; offloading-connector vs. lmcache-connector trade-offs.
- Runtime storage mode switching via HTTP API (`POST /storage/mode`) without vLLM restart.
- Prefix cache interactions: GPU-tier prefix cache (`reset_prefix_cache` dev API) vs. LMCache off-HBM tier.
- Cache salt / `pre_caching_hash_algorithm` correctness requirement for cross-replica sharing.

### 4.4 Integration with SGLang
- SGLang radix attention and its natural alignment with prefix-keyed KV storage.
- Current integration status; planned connector API.

### 4.5 Observability
- `rocm-aic-exporter.py`: Prometheus textfile metrics — KV file inventory, filesystem utilization, chunk hit histogram, NFS I/O, ROCm version.
- Grafana dashboards: TTFT distribution, cache hit rates by tier, NVMe throughput, MFU.
- LMCache KV events in vLLM engine log; `LMCACHE_LOG_LEVEL` tuning.

---

## 5. NIXL: Low-Latency Cross-Node KV Transport

### 5.1 What NIXL Is
- NIXL (NVIDIA/AMD eXtensible I/O Library — community fork `andyluo7/nixl`, branch `amd-support`): a pluggable, high-performance I/O abstraction for transferring tensors between GPUs, CPUs, and storage across nodes.
- Designed for disaggregated inference (prefill/decode disaggregation, KV migration); relevant to any scenario where KV tensors need to move between pods or nodes.

### 5.2 NIXL on AMD: ROCm Build and Plugins
- Build configuration: `meson` with `-Dwheel_variant=rocm`, `-Drocm_path=/opt/rocm`, UCX path.
- AIS plugin overlay (`libplugin_AIS.so`, `libplugin_AIS_MT.so`): thread-pool synchronous I/O using `hipFileRead`/`hipFileWrite`; batch API not yet supported on ROCm 0.2.x hipFile.
- POSIX plugin: standard filesystem path; production-ready MVP.
- Staging buffer: GPU (VRAM) NIXL buffer for `hipFileBufRegister` registration — sizing trade-offs on 16 GB vs. 192 GB VRAM devices.

### 5.3 LMCache × NIXL Integration
- `enable_nixl_storage: true` in LMCache config; `NixlStorageBackend` routes chunk I/O through NIXL.
- Pool file pre-allocation (`nixl_pool_size`), FD limits, lazy `ftruncate`.
- Storage mode matrix: `nixl-posix` (NVMe), `ais` / `ais_mt` (AIS object store + hipFile), `ais_batch` (future).
- Cross-node transfer potential: NIXL over UCX/RDMA enabling KV migration between decode nodes without host CPU involvement.

### 5.4 Comparison: hipFile Direct vs. NIXL
| Dimension | vllm-lmcache-hipfile | vllm-lmcache-nixl |
|---|---|---|
| KV disk path | LMCache GdsBackend → hipFile | LMCache NixlStorageBackend → NIXL POSIX/AIS |
| Cross-node | No (single node) | Yes (UCX/RDMA capable) |
| Maturity | Production-tested | MVP / active development |
| hipFile batch | N/A | Stub on ROCm 0.2.x |

---

## 6. Cluster Orchestration: llm-d and Intelligent Request Routing

### 6.1 The llm-d Project
- Open-source Kubernetes-native LLM deployment framework; AMD ROCm image (`ghcr.io/llm-d/llm-d-rocm`).
- InferencePool: smart request router with pluggable scorers.

### 6.2 Cache-Aware Routing
- Scorer hierarchy: Queue Scorer (load), KV Cache Utilization Scorer (HBM pressure), GPU Prefix Cache Scorer (HBM hits), CPU Prefix Cache Scorer (DRAM tier hits).
- Why routing matters: directing a request to the replica that already holds the matching KV prefix in its local CPU cache eliminates a cross-node fetch.
- `lruCapacityPerServer` sizing formula: `cpu_bytes_to_use / 2560`; 100 GB CPU cache default.

### 6.3 Tiered Prefix Cache Deployment
- Two-tier Kustomize deployment: offloading-connector (vLLM native CPU offload) and lmcache-connector variants.
- Tensor parallelism configuration: TP=2 per pod, 2 AMD GPUs per replica.
- Istio gateway → InferencePool → vLLM replica(s) with HBM + CPU cache.

### 6.4 Inference Scheduling Deployment
- 8-replica deployment with InferencePool prefix-cache-aware routing; baseline without explicit tiering but with cache-aware dispatch.
- Reduced tail latency under skewed prefix distributions.

---

## 7. Benchmarking Methodology and Results

### 7.1 Benchmark Suite Overview
| Benchmark | Workload | Primary Metric |
|---|---|---|
| ttft-lmcache | Controlled hit-rate sweep (0–100%), single prompt, Gutenberg | TTFT vs. hit rate |
| llm-prefill-benchmark | Random Gutenberg long-context (10k-word chunks), parallel workers | TTFT, prefill tok/s |
| kv-cache-tester | 739-trace Claude Code agent replay (MI300X blog profile) | Throughput, latency percentiles |
| llm-agentx | 470 SemiAnalysis CC traces (≤256k proxy tokens), agentic ISL growth | TTFT, wall time, token/s |
| lmcache-io-tester | Synthetic I/O characterization per storage backend | Bandwidth, latency |

### 7.2 Key Results: TTFT vs. Cache Hit Rate
- At 0% hit rate: baseline prefill TTFT (O(6 s) for 15k-token context on representative model).
- At 50% hit rate with CPU DRAM tier: ~50% TTFT reduction.
- At 100% hit rate: near-zero prefill TTFT (O(150 ms) load + decode overhead).
- Implication: even modest hit rates (20–40%) in production agentic workloads deliver significant P99 improvements.

### 7.3 Storage Backend Comparison
- CPU RAM: lowest latency, node-local, volatile.
- NVMe (POSIX / hipFile GdsBackend): durable, 10–200 ms load per chunk at typical NVMe throughput; hipFile path eliminates CPU bounce.
- NIXL POSIX: comparable to NVMe POSIX; added flexibility for future cross-node transport.
- NIXL AIS / hipFile: object store path; suitable for cluster-wide shared cache with GPU Direct I/O.
- NFS: highest latency but enables fully shared persistent cache with no per-node storage requirement.

### 7.4 Agentic Workload Results (kv-cache-tester / llm-agentx)
- CC agent traces show high prefix reuse: main-agent system prompt is repeated across sub-agent calls.
- Cache-aware routing doubles effective cache hit rate vs. round-robin for prefix-heavy traces.
- ISL (input sequence length) growth profile across trace: illustrates compounding benefit of tiered cache.

### 7.5 Vendor Configurations
- **DriveNets (gfx950)**: HBM L1 + CPU DRAM L2 + NVMe L3 via `LMCACHE_LOCAL_*` env API; recipe at `recipies/aic-drivenets/`.
- **Dell + hipFile**: vLLM + LMCache + hipFile on Dell AMD systems; `vendors/dell/`.
- **WEKA FS**: network-attached parallel filesystem as Tier 3; PoC at `vendors/weka/`; suitable for fully shared persistent KV store across all cluster nodes.

---

## 8. Deployment Guide Summary

### 8.1 Prerequisites
- AMD Instinct GPU nodes with ROCm ≥ 6.x; `amd.com/gpu` Kubernetes resource; InfiniBand or RoCE for NIXL cross-node paths.
- Cluster inventory via Ansible `discover.yml`: GPUs, NVMe, RDMA NICs, ROCm version, DKMS status.

### 8.2 Single-Node Docker Deployment
- `recipies/vllm-lmcache-hipfile/`: `make build && make run` — vLLM + LMCache + hipFile on a single Instinct node.
- `recipies/vllm-lmcache-nixl/`: NIXL POSIX or AIS backend; `VLN_LMCACHE_IO` selector.
- `recipies/aic-drivenets/`: gfx950 dGPU (Strix Halo) variant with `LMCACHE_LOCAL_*` tiering.

### 8.3 Kubernetes / llm-d Deployment
- `recipies/llm-d/tiered-prefix-cache/`: `just setup` — full tiered stack with InferencePool on K8s.
- `recipies/llm-d/inference-scheduling/`: `just setup` — multi-replica with cache-aware routing, no explicit tiering.
- Monitoring: Prometheus + Grafana dashboards; `just port-forward-start`.

### 8.4 Slurm Deployment
- `run-slurm.sh` / `run-slurm-nixl.sh`: Slurm job wrappers with automatic NVMe discovery, HF weight caching, and Gutenberg benchmark.
- NVMe auto-discovery order: blank `nvme*n*` (formatted ext4) → mounted NVMe under `/mnt`/`/local` → scratch fallback.

---

## 9. Roadmap and Open Work

### 9.1 NIXL Maturity on ROCm
- hipFile batch API (`hipFileBatchIOGetStatus`) not yet implemented in ROCm 0.2.x; AIS_MT thread-pool workaround in production; batch path targeted for future hipFile release.
- UCX/RDMA-backed cross-node KV migration (P/D disaggregation): the primary motivation for NIXL; proof-of-concept paths validated, production hardening in progress.

### 9.2 SGLang Integration
- SGLang radix attention aligns well with prefix-keyed chunk storage; LMCache connector for SGLang is an active development target.
- Expected to follow the same `kv-transfer-config`-style API as vLLM.

### 9.3 Cluster-Wide Shared Cache
- AIS object store as a fully shared, persistent Tier 2/3 accessible by all replicas without per-node NVMe; requires NIXL AIS_MT maturity.
- WEKA FS as an alternative cluster-wide tier: higher latency but simpler operational model (no object store).
- Cross-rack KV migration with NIXL + UCX: enables prefill/decode disaggregation patterns (prefill on a high-compute node, decode on a high-memory node) with KV transferred over RDMA.

### 9.4 Cache-Aware Scheduler Enhancements
- Integrating AIC storage metrics (Prometheus `rocm_aic_*`) into InferencePool scoring for NVMe-tier-aware routing decisions.
- Speculative prefetching: pre-loading KV chunks from NVMe to CPU DRAM based on predicted prompt patterns.

### 9.5 Quantization and Compression
- KV quantization (INT8 / FP8) as a multiplier on effective cache capacity per tier; ROCm quantization kernel support.
- Chunk-level delta compression for highly similar prompt prefixes in RAG workloads.

---

## 10. Conclusion

*~200 words. Restate the memory-wall problem. Summarize how the AIC tiered cache hierarchy (HBM → CPU DRAM → NVMe → network storage) combined with LMCache chunk management, NIXL low-latency transport, cache-aware routing (llm-d InferencePool), and integration with vLLM and SGLang delivers a composable, open-source KV cache tier for AMD Instinct clusters. Call to action: open-source collaboration, reference to rocm-aic repository.*

---

## Appendices

### A. Glossary
- AIC, HBM, KV cache, TTFT, ISL, prefill, decode, chunk, blending, hipFile, NIXL, AIS, LMCache, llm-d, InferencePool, CDNA, GDS, UCX, RDMA, TP, PP.

### B. Benchmark Reproducibility Notes
- Seed control (`SEED`, `PYTHONHASHSEED`), deletion manifests in `results.jsonl`, `RUN_LONG_SEED` for Gutenberg runs.
- How to replay any published result exactly.

### C. Configuration Reference
- Key `VLH_*` / `VLN_*` / `ADE_*` environment variables across recipes.
- LMCache YAML schema summary (chunk size, eviction policy, backend config keys).
- NIXL pool sizing formula and FD limit rule of thumb.

### D. Hardware Compatibility Matrix
| GPU | ROCm arch | Recipes | hipFile | Notes |
|---|---|---|---|---|
| MI300X | gfx942 | hipfile, nixl, llm-d | Yes | Primary target |
| MI250X | gfx90a | hipfile | Yes | |
| RX 9070 XT | gfx1201 | hipfile | Partial | Desktop GPU, no InfiniBand |
| Strix Halo (dGPU) | gfx950 | aic-drivenets | via LMCACHE_LOCAL | DriveNets partner config |

### E. References
- NVIDIA ICMS / CMX technical blog
- WEKA blog on BlueField-4 and ICMS
- LMCache GitHub (`LMCache/LMCache`)
- NIXL AMD fork (`andyluo7/nixl`, branch `amd-support`)
- llm-d project (`llm-d-incubation/llm-d`)
- vLLM documentation (`docs.vllm.ai`)
- AMD ROCm (`amd.com/en/products/software/rocm.html`)
- Andy Luo MI300X + LMCache blog post (`andyluo7.github.io`)
- SemiAnalysis CC traces dataset (`semianalysisai/cc-traces-weka-with-subagents-052726-256k`)
- kv-cache-tester (`callanjfox/kv-cache-tester`)
