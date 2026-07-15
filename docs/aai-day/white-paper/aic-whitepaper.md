---
title: "AMD Infinity Context: Network-Attached Storage for Scalable KV Cache Tiering in Agentic Inference Systems"
bibliography: references.bib
csl: ieee.csl
---

# AMD Infinity Context: Network-Attached Storage for Scalable KV Cache Tiering in Agentic Inference Systems

## Abstract

Large-scale LLM inference is hitting a memory wall. As agentic workflows drive context lengths past hundreds of thousands of tokens and request volumes grow, the KV cache—the attention state that must be materialized for every token—increasingly dominates both GPU memory and prefill compute time. Re-computing KV tensors for repeated prompts wastes compute cycles; evicting them under memory pressure destroys latency.

We introduce AMD Infinity Context (AIC), a tiered, shared KV cache architecture for AMD Instinct GPU clusters. AIC extends the KV cache hierarchy beyond GPU HBM, CPU DRAM, and local NVMe to off-the-shelf network storage, enabling cache entries to persist at scale across requests, replicas, and node failures. The AIC reference implementation integrates with the open-source vLLM [@kwon2023pagedattention] inference stack—vLLM [@kwon2023pagedattention] via LMCache [@liu2025lmcache] (KV chunk management) and NIXL [@nvidia2024nixl] (low-latency cross-node transport) with plans to extend to SGLang [@zheng2024sglang], and the llm-d orchestrator [@llmd2024].

We present benchmark results on AMD Instinct MI300X clusters. By substituting data movement for redundant prefill computation, AIC achieves up to 11× improvement in throughput and 9× reduction in time-to-first-token (TTFT) on cache hits, enabling serving systems to support more concurrent agentic workloads without increasing compute capacity.

AIC is open-source and available as a technology preview for AMD ROCm deployments.

---

## 1. Introduction

### The Agentic Inflection Point

The economics of LLM inference are shifting. Interactive chat—a human typing, waiting, reading—tolerates latencies of seconds and generates modest request volumes. Agentic workflows are different. An autonomous coding agent may invoke an LLM dozens of times per minute: analyzing code, proposing fixes, reviewing its own output, delegating to sub-agents. Each invocation expects sub-second response; each adds to a growing context that can reach hundreds of thousands of tokens.

This shift from human-driven to machine-driven inference demand creates three compounding pressures on serving infrastructure: *volume*, as autonomous agents generate requests at machine speed; *latency*, as multi-turn interactions compound delays into perceptible slowdowns; and *context*, as agents accumulate state across turns, leading to ever-growing prompt lengths. The growth in model size (70B to 400B+ parameters), context length (8k to 1M tokens), and concurrent user demand are pushing inference infrastructure to its limits. Operational efficiency of inference clusters executing agentic workloads is now a key economic lever.

### Context Growth as the Limiting Factor

Context growth is both a compute and memory limiting factor in current serving infrastructure. The attention algorithm that computes relationships between tokens scales quadratically with sequence length, meaning a significant amount of time is spent in the prefill operation that generates the attention's KV data. On AMD Instinct MI300X accelerators with 192 GB HBM3, KV cache competes directly with model weights for high-bandwidth memory. Mixture-Of-Expert models now routinely require multi-GPU nodes to even be loaded into HBM leaving a fraction of HBM available for KV cache. Agentic workloads exemplify the KV cache footprint challenge. The agent starts with an ask and data to work toward a goal, performing reasoning and tool calling across multiple steps. Each step builds on the previous one by appending to the context, steadily growing the KV footprint. A single million-long prompt has a footprint in the range of 30 to 70 GBs depending on the attention mechanism and data type used. Scaling agentic demand to hundreds of clients and the overall KV footprint quickly reaches into the terabytes range exerting pressure on the HBM and DRAM memory tiers that are to compute.

### Trading Compute for Storage

The key insight for serving agentic workloads is that multi-turn agentic prompts feature more than 97.5% prefix reuse across turns [@zhu2026tracelab].

This means we can trade fully re-computing expensive prefill on long contexts for re-using previously cached KV data along with a small "prefill extend" operation to compute only the new turn's tokens attention.

To improve performance and TCO for agentic workloads, the solution must provide enough KV cache storage capacity to absorb a large resident set of agentic sessions, while delivering cached data to compute units faster than it would take to fully re-compute.

Network-attached storage is uniquely positioned to address the needs of serving systems for agentic workloads:
- **GPU HBM and CPU DRAM** are too expensive to scale out to terabyte capacity
- **Local NVMe** offers a good capacity-performance trade-off but *strands* the cached data on individual nodes, requiring increased software complexity to coordinate for cross-replica sharing
- **Cloud storage** has sufficient capacity but latency too high and bandwidth too low to serve as a fast cold-storage tier

Network-attached storage occupies the sweet spot: terabyte-scale capacity, hundreds of GB/s aggregate bandwidth, and latencies low enough that fetching cached KV is faster than recomputing it.

### AMD Infinity Context

This paper introduces AMD Infinity Context (AIC), a reference architecture that leverages network-attached storage to provide scalable, low-latency, high-bandwidth KV cache storage for AMD Instinct accelerators. An AIC-based serving system can offer petabytes of KV cache storage while delivering data to compute nodes at hundreds of GB/s in aggregate.

AIC integrates into existing open-source inference serving stacks through software libraries that implement established KV cache management interfaces. Our evaluation demonstrates that AIC achieves up to 13× improvement in throughput and 10× reduction in TTFT on cache hits, enabling serving systems to support more concurrent agentic workloads without increasing compute capacity.

The remainder of this paper is organized as follows. Section 2 reviews the current state of KV cache tiering and its limitations. Section 3 presents the AIC architecture, including software integration through LMCache and NIXL. Section 4 presents our evaluation methodology and results. Section 5 concludes with future directions.

---

## 2. Background: KV Cache Tiering in Inference Systems

### 2.1 Current Architecture

Modern inference serving systems rely on KV cache management to improve throughput and reduce latency. The fundamental insight is that when multiple requests share a common prefix—whether a system prompt, few-shot examples, or conversation history—the KV cache computed for that prefix can be reused rather than recomputed [@kwon2023pagedattention]. This optimization requires two critical infrastructure capabilities.

First, the serving system must maintain a view of the current state of the distributed KV cache to make informed routing and scheduling decisions. When a new request arrives, the scheduler must determine which compute node holds a relevant cached prefix and route accordingly.

Second, the serving system must be able to transfer KV cache data across memory tiers and across compute nodes. This capability underpins both prefill-decode disaggregation—where prefill and decode phases execute on different hardware [@zhong2024distserve; @patel2024splitwise]—and load balancing, where KVs migrate to less-loaded nodes.

In practice, KV caches are computed locally on a GPU, immediately held in that GPU's memory, and advertised to a KV management layer that keeps track of the distributed state of KV caching as well as make storage placement decisions (promoting, demoting, replicating, etc...). The interface for KV cache storage is conceptually simple: store tensors representing KV blocks into a designated tier, where each tier may use either an in-memory or file-based representation.

The standard architecture organizes storage into four tiers, ordered by proximity to the GPU (Table 1).

\begin{table*}[ht]
\centering
\small
\begin{tabular}{>{\bfseries}p{1.5cm} p{2.2cm} p{2.2cm} p{2.2cm} p{6.5cm}}
\hline
\textbf{Tier} & \textbf{Medium} & \textbf{Capacity} & \textbf{Access Latency} & \textbf{Characteristics} \\
\hline
Tier 1 & GPU HBM & \textasciitilde10s GB after weights & Sub-millisecond & Per-replica, private; MI300X provides 192 GB HBM3 at \textasciitilde5.3 TB/s \\[4pt]
Tier 2 & Host DRAM & 100s GB & 1--10 ms & Per-node, potentially shared across replicas on same host \\[4pt]
Tier 3 & Local NVMe & 1--100 TB & 10--200 ms & Per-node or per-rack; O\_DIRECT / hipFile paths available \\[4pt]
Tier 4 & Network Storage & Petabyte-scale & 100 ms -- 1 s & Cluster-wide, fully shared; NFS, WEKA FS, or object store \\
\hline
\end{tabular}
\caption*{Table 1: KV cache storage tiers ordered by proximity to the GPU.}
\end{table*}

This hierarchy reflects a classic memory trade-off: faster tiers offer lower latency but limited capacity, while slower tiers provide persistence and scale at the cost of access time. The key insight is that even at Tier 3 or Tier 4 latencies, recovering cached KV from storage costs less than re-prefilling when prefill itself takes seconds for long-context workloads.

### 2.2 Limitations for Agentic Workloads

Without a network-attached storage, the architecture forces a choice between performance and resilience that becomes untenable at scale.

**The performance-resilience trade-off.** On one end, Tiers 1 through 3 offer the performance characteristics needed for a responsive globally-partitioned KV cache. Nodes can pull KV caches from one another for load balancing over a fast RDMA backend network, and the combined capacity across local NVMe drives can be substantial. However, this data is stranded on individual nodes. If a node fails, its cached data is lost. Maintaining resilience requires explicit software management overhead to replicate and transfer data—work that competes with serving inference requests.

On the other end, a cloud storage tier provides natural resilience through centralized, replicated storage. All nodes can access a common last-level cache, expanding the effective window of active agentic sessions that remain cached. However, we argue cloud storage latency and bandwidth are inadequate for serving as a fast cold-storage tier. The round-trip time to cloud storage adds tens to hundreds of milliseconds of latency, tying up memory footprint that could otherwise be used to cache evicted KVs or compute a prefill extend.

**Coupled compute and storage scaling.** In the current architecture, fast peer-to-peer KV data sharing is inherently coupled to compute nodes. Each node's local NVMe contributes to the aggregate KV storage capacity, but adding storage capacity requires adding compute nodes—and vice versa. This coupling prevents independent scaling. Agentic workloads exhibit distinct traffic patterns: long idle periods punctuated by bursts of activity, varying context sizes across agents, and unpredictable session lifetimes. Serving these workloads efficiently requires the flexibility to scale storage (to retain more cached sessions) independently of compute (to handle burst throughput).

**Management complexity.** Implementing a globally partitioned KV cache over independent local NVMe drives requires coordination. Load-balancing decisions must identify which node holds the relevant KV cache, negotiate the transfer, and update the global index—all while those nodes continue serving other requests. This complexity grows with cluster size and becomes particularly acute when sessions span long time horizons, as agentic workflows tend to do.

---

## 3. AIC: Network-Attached KV Storage

Having established the limitations of current tiered architectures, we now present AIC—a reference architecture for AMD Instinct clusters that introduces network-attached storage as a new tier unifying performance, resilience, and operational flexibility.

### 3.1 Design Goals

AIC targets four design goals:

- **Latency**: Cache hit latency must be low enough that loading KV from the tier costs less than re-prefilling. Target: sub-200ms for network storage, competitive with local NVMe for typical long-context workloads.
- **Shareability**: KV chunks stored by replica A must be readable by replica B without application-layer coordination beyond publishing KV events.
- **Composability**: The tier should be pluggable into existing vLLM and SGLang deployments via standard connector APIs to KV management software requiring no engine modifications.
- **AMD-native I/O path**: Leverage hipFile (ROCm GPU Direct Storage) where available to minimize CPU involvement in transfers between GPU HBM and leverage existing file-based abstractions to represent KV blocks.

Several factors make network-attached storage viable for meeting these goals today. 400 Gb/s RDMA NICs have become commodity hardware in AI clusters—the same fabric used for gradient synchronization during training can serve KV cache traffic during inference. NFS-over-RDMA implementations have matured [@ietfrfc8166; @ietfrfc8267], delivering latencies in the hundreds of microseconds while preserving standard POSIX semantics.

### 3.2 Architecture Overview

Figure 1 illustrates the AIC target software stack, organized into nine layers from application to storage. The reference implementation presented in this paper targets vLLM+LMCache.

![AIC Software Stack](aic-software-stack.png)
*Figure 1: AIC software stack showing the layered architecture from serving layer through storage backends.*

The stack comprises:

- **Serving Layer**: Open-source inference engines (vLLM, SGLang) that execute model inference and manage request scheduling.
- **Orchestration**: Deployment and routing infrastructure—llm-d with Kubernetes for open-source deployments, or AMD Infera for managed environments.
- **KV Management**: Cache lifecycle management including chunk storage, retrieval, and eviction—LMCache [@liu2025lmcache] for open-source, vLLM offload, and HiCache [@sglanghicache2025] as an alternative.
- **KV Transfer Engine**: Low-latency data movement abstraction—NIXL [@nvidia2024nixl] for cross-node transport, Mooncake [@qin2024mooncake] as an alternative.
- **GPU ↔ Storage Fastpath**: Direct GPU-to-storage I/O bypassing CPU—ROCm hipFile and AMD Infinity Storage (AIS).
- **GPU**: AMD Instinct accelerators (MI300X/MI355X series, future MI4xx).
- **Network Fabric**: High-bandwidth interconnect—dual-mode front-side initiator Ethernet with MI455X scale-out capability.
- **NIC Support**: RDMA-capable network adapters—Pensando AINIC, Broadcom Thor2, Mellanox ConnectX.
- **Storage Backend**: Persistent storage targets—NFS over RDMA (reference implementation), object stores over RDMA, WekaFS, Lustre, IBM GPFS.

**AIC directly involves four layers**: KV Management (LMCache), KV Transfer Engine (NIXL), GPU-Storage Fastpath (hipFile/AIS), and Storage Backend (NFS over RDMA). The serving, orchestration, GPU, network fabric, and NIC layers are leveraged but not modified—AIC integrates through standard interfaces rather than requiring changes to the inference engine or hardware.

**Hardware configuration.** AIC builds on the open storage ecosystem using off-the-shelf components. The reference implementation uses NFS over RDMA, which presents a standard POSIX filesystem interface while leveraging RDMA transport for low-latency, high-bandwidth data movement. This simplifies integration: existing file-based KV cache code paths work without modification, while the RDMA transport delivers performance approaching that of local storage.

The reference deployment uses compute nodes each equipped with 8 AMD Instinct MI300X accelerators (192 GB HBM3 each, ~5.3 TB/s memory bandwidth), 8 Broadcom Thor-2 400 Gb/s RDMA NICs, and 2 TB of host DRAM. The network-attached storage consists of NVMe SSDs exposed as an NFS-over-RDMA mount point.

> **TODO:** Fill in NVMe SSD count and aggregate storage capacity for the NAS configuration.

**Software integration.** AIC integrates into existing serving infrastructure by adhering to established abstractions. Existing KV cache tiering systems already distinguish between in-memory representations (tensors in GPU/CPU memory) and file-based representations (KV data serialized to a filesystem). AIC presents itself as a filesystem target, allowing existing code paths for file-based KV storage to work with minimal modification. The inference software architecture does not fundamentally change—the serving engine, KV cache manager, scheduler, and routing layers retain their existing responsibilities.

### 3.3 Reference Implementation

AIC provides two integration paths via AIS and NIXL for existing serving stacks. The AIC reference implementation targets the vLLM serving engine and LMCache for KV management.

### 3.3.1 LMCache: The KV Cache Management Layer

LMCache is an open-source KV cache offload framework that sits between the inference engine and external memory tiers (CPU DRAM, NVMe, network storage). Inference engines such as vLLM expose a KV connector interface; LMCache
implements this interface to receive notifications of KV-relevant events—prefill completions, cache lookups, memory pressure—and manages the resulting chunk lifecycle: store, retrieve, and evict.

LMCache splits KV cache into fixed-size chunks keyed by a hash of the token sequence prefix. This chunking enables partial cache hits: non-contiguous cached chunks can be merged with freshly computed KV, so a request benefits from any
matching prefix rather than requiring a full match.

Chunk size is a tunable parameter with a trade-off. Larger chunks amortize read/write overhead more effectively, but increase the likelihood of a partial mismatch: when the input sequence length does not align with chunk boundaries,
the system must perform a prefill extend for the remainder. Smaller chunks reduce wasted computation at the cost of higher I/O overhead.

LMCache supports multiple storage backends (Table 2).

\begin{table*}[ht]
\centering
\small
\begin{tabular}{>{\bfseries}p{3cm} p{5.5cm} p{6cm}}
\hline
\textbf{Backend} & \textbf{Description} & \textbf{Characteristics} \\
\hline
CPU & In-process DRAM store & Fast, node-local, volatile \\[4pt]
POSIX/disk & Standard filesystem I/O & Portable, works with local NVMe or NFS \\[4pt]
hipFile / GdsBackend & AMD GPU Direct Storage via ROCm hipFile API & Eliminates CPU bounce buffer on DMA \\[4pt]
NIXL & Pluggable transport layer (see below) & Cross-node capable \\[4pt]
AIS & AMD AIS object store via hipFile & Cluster-wide shared cache with GPU Direct path \\
\hline
\end{tabular}
\caption*{Table 2: LMCache storage backends.}
\end{table*}

### 3.3.2 AMD Infinity Storage (AIS)

The AIS library implements the KV storage interface expected by serving systems. AIS is AMD's equivalent of NVIDIA GPU Direct Storage (GDS), providing efficient direct data paths between AMD Instinct GPU memory and storage devices, bypassing unnecessary CPU involvement. It is responsible for moving data between GPU memory and storage, whether that storage is local NVMe or a mounted remote filesystem.

The library handles the mechanics of data movement: staging through host memory where required, managing transfer buffers, and optimizing for the characteristics of the underlying storage medium. For GPU-to-storage transfers, the library implements efficient pipelines that overlap computation with data movement.

### 3.3.3 NIXL: Low-Latency Cross-Node KV Transport

NIXL (eXtensible I/O Library) is a pluggable, high-performance I/O abstraction for transferring tensors between GPUs, CPUs, and storage across nodes. Originally developed for disaggregated inference scenarios (prefill/decode disaggregation, KV migration), NIXL is relevant to any scenario where KV tensors need to move between pods or nodes.

On AMD platforms, NIXL builds with ROCm support and provides several plugins:

- **AIS plugin** (`libplugin_AIS.so`, `libplugin_AIS_MT.so`): Thread-pool synchronous I/O using hipFileRead/hipFileWrite. The batch API is not yet supported on ROCm 0.2.x hipFile.
- **POSIX plugin**: Standard filesystem path; production-ready.
- **UCX/RDMA plugin**: Enables cross-node KV migration without host CPU involvement (in development).

When LMCache is configured with `enable_nixl_storage: true`, the `NixlStorageBackend` routes chunk I/O through NIXL, enabling a unified interface across storage modes: `nixl-posix` (local NVMe), `ais` (AIS object store + hipFile), or future cross-node RDMA paths.

While we recognize NIXL as a critical KV movement interface for upper layers of the software stack to orchestrate KV cache data movement across on an AIC-based inference system, this path is still under active development.

### 3.3.5 When to Use Direct P2P Transfer in favor of AIC

AIC does not replace all KV data movement. For latency-critical operations like prefill-decode disaggregation—where a partially-computed KV cache must be transferred from a prefill node to a decode node within a single request's lifetime—direct peer-to-peer GPU transfer remains the recommended approach. Copies to AIC can still happen on cache eviction, eagerly, or triggered by a scheduler. AIC is optimized for the "warm or cold storage" use case: retaining KV caches across requests and sessions, and serving them to whichever node needs them. 

---

## 4. Evaluation

With the architecture and integration paths established, we now evaluate AIC's performance. We use a benchmark suite designed to characterize how memory tier characteristics affect serving throughput and time-to-first-token (TTFT) when KV cache retrieval occurs.

### 4.1 Methodology

**Hardware configuration.** Experiments were conducted on compute nodes each equipped with 8 AMD Instinct MI300X accelerators (192 GB HBM3, ~5.3 TB/s bandwidth), 8 Broadcom Thor-2 400 Gb/s RDMA NICs, and 2 TB of host DRAM. The network-attached storage tier uses NVMe SSDs exposed as an NFS-over-RDMA mount point.

> **TODO:** Fill in NVMe SSD count and aggregate storage capacity. Also specify number of compute nodes used.

**The throughput cliff chart.** We introduce the "throughput cliff chart," a roofline-style analysis of the KV cache tier hierarchy. It visualizes how each memory tier acting as a KV cache impacts the throughput of the prefill service.

The benchmark operates in two phases. First, we submit a "cold" round of inference requests to populate the cache. Second, the requests are replayed for a "warm" round with a 100% cache hit rate on the target KV storage tier (GPU HBM, CPU DRAM, or AIC). As the number of clients increases, so does memory pressure to store the requests' KV cache context, leading to enforcement of an eviction policy (here LRU). The "cliff" is the point where a memory tier's capacity is exhausted and the eviction policy causes the warm round to experience cache misses, forcing prefill computation to take place.

### 4.2 Results

**Throughput cliff chart results.** Figure 2 shows the throughput cliff chart for an open-source 120B parameter model running on a single GPU with tensor parallelism=1. The load consists of requests with a footprint of 60,000 tokens each. We set the output sequence length to 1, indicating a 100% cache hit scenario where no prefill extend is performed. This is done to characterize the memory-tier's best performance.

![KV Cache Throughput Cliff Chart - MI300X](kv-throughput-cliff-mi300x.png)
*Figure 2: KV cache tier cliff chart showing throughput vs. number of concurrent clients for GPU HBM, CPU DRAM, and AIC tiers; vLLM+LMCache running GPT OSS 120B with ISL=60000, OSL=1, on MI300X*

We observe three memory "regimes" corresponding to the maximum KV footprint each tier can accommodate, plus a compute regime where prefill is forced to occur:

1. **GPU HBM regime (0 to ~20 clients).** Best performance is achieved when all requests can be held in GPU HBM. KV cache retrieval is an immediate memory lookup with no data movement and single token prefill extend required (given the 100% cache hit rate and output sequence length of 1). Maximum throughput reaches approximately 230,000 tokens per second.

2. **HBM cliff → DRAM regime (~20 to ~40 clients).** When HBM capacity is exhausted, the LRU policy evicts older KV entries to DRAM. Throughput drops because requests now require loading KV from DRAM to HBM instead of an immediate HBM hit. This reduces throughput to approximately 160,000 tokens per second, but allows the system to support more concurrent clients at sustained high throughput.

3. **DRAM cliff → AIC regime (~40 to 250 clients).** The AIC backend accommodates the KV footprint of all client scenarios in this range. Throughput is lower than DRAM due to network transfers, but the system continues to serve from cache at 130,000–170,000 tokens per second.

4. **Compute regime (beyond tier capacity).** When all tiers are exhausted, requests incur KV cache misses resulting in full prefill recomputation. Throughput drops dramatically to approximately 15,000 tokens per second—the cost of the compute-bound prefill operation.

The KV cache tiering hierarchy shifts the throughput cliff—the point where throughput drops to prefill rates—rightward, supporting more concurrent clients. AIC's unique value proposition is that its large storage capacity and aggregate bandwidth shift this throughput cliff further right, enabling support for hundreds of concurrent clients where local tiers would have been exhausted.

Under these conditions, we observe a **11× throughput increase** between compute-based prefill (~15,000 tokens/s) and an AIC KV-cache hit (~130,000–170,000 tokens/s). This demonstrates that even the highest-latency cache tier provides an order-of-magnitude improvement over recomputation for long-context workloads.

**TTFT climb chart results.** Figure 3 shows the "TTFT climb chart" for the same experimental setup. As more clients issue inference requests and compute becomes saturated with prefill operations, TTFT increases due to queuing. AIC reduces TTFT by replacing slow prefill recomputation with faster KV cache retrieval, **improving request turnaround by a factor of respectively 8x and 9x for GPU HBM and DRAM at high concurrency**. Note the TTFT remains quite high at 45 seconds for 250 clients. AIC is not a substitute for adding compute capacity—it improves efficiency within a given compute budget, but meeting strict TTFT SLAs at high load still requires adequate compute resources.

![KV Cache TTFT Climb Chart - MI300X](kv-ttft-climb-mi300x.png)
*Figure 3: KV cache tier TTFT climb chart showing TTFT vs. number of concurrent clients for GPU HBM, CPU DRAM, and AIC tiers; vLLM+LMCache running GPT OSS 120B with ISL=60000, OSL=1, on MI300X*

The AIC tier scaling behavior is critical for agentic workloads, where the number of concurrent sessions may be large and variable. While a single agent retrieving its KV cache sees moderate latency, a cluster serving hundreds of agents benefits from AIC's aggregate bandwidth.

### 4.3 Limitations

We frame AIC as a technology preview demonstrating the viability of network-attached KV cache storage. Production hardening and broader workload evaluation are ongoing.

The current evaluation has several limitations that inform the scope of our claims:

- **Idealized cache hit rates.** The 100% cache hit rate in the cliff benchmark represents a best-case scenario. In production, partial cache hits and cache misses will reduce the improvement factor.

- **Synthetic workloads.** While the benchmark models agentic access patterns (high prefix reuse, large context), results validation on production agentic traces is still in progress. Real workloads may exhibit different prefix reuse distributions and cache hit patterns.

- **Hardware-specific dynamics.** The 11× improvement reflects the MI300X's compute-to-network bandwidth ratio. As compute and network capabilities evolve—particularly as prefill computation becomes faster—the relative advantage of caching over recomputation may shift.

---

## 5. Conclusion

The agentic era introduces new constraints on inference serving infrastructure. Machine-driven demand increases request volume; multi-turn interactions demand lower latency; and accumulating context strains memory capacity. The quadratic scaling of attention computation makes these pressures particularly acute—but the high prefix reuse inherent in agentic workloads offers an escape: if the system can cache and retrieve KV state efficiently, it can skip the redundant computation entirely.

This reframes the challenge as a storage problem. Current tiered architectures force a choice between performance (local storage, limited capacity, stranded data) and resilience (cloud storage, high latency, centralized access). AIC resolves this trade-off by introducing network-attached storage as a tier that combines the performance characteristics of local storage with the operational benefits of centralized storage.

AIC integrates into existing serving stacks through two paths: direct use of the AMD Infinity Storage (AIS) library for GPU-to-storage data movement, and integration via the NIXL unified transfer API.

Our evaluation demonstrates that AIC-enabled inference engines can accommodate more concurrent clients than those relying solely on local storage tiers, achieving a 11× throughput improvement amd 9x TTFT reduction over prefill recomputation. The key insight is that network-attached storage bandwidth is underutilized when serving few clients but efficiently shared across many—a characteristic that aligns well with the bursty, high-concurrency nature of agentic workloads.

**Future work.** Several directions remain for active development:

- **LLM-D integration.** Active development includes targeting an AIC setup by adding a provider to the existing LLM-D guide for KV-tiering. 
- **NIXL maturity on ROCm.** The hipFile batch API (`hipFileBatchIOGetStatus`) is not yet implemented; the AIS_MT thread-pool serves as a production workaround while the batch path is targeted for a future hipFile release.
- **Cross-node KV migration.** UCX/RDMA-backed cross-node KV migration support for prefill/decode disaggregation patterns is upstreamed to vLLM. Future work is to support both prefill/decode transfer and KV cache tiering.
- **Cache-aware scheduler enhancements.** Integrating AIC storage metrics into routing decisions for NVMe-tier-aware dispatch, and speculative prefetching of KV chunks based on predicted prompt patterns.

---

## References

<!-- Bibliography is managed in references.bib and rendered by pandoc --citeproc.
     To export: pandoc aic-whitepaper.md --citeproc -o aic-whitepaper.docx

     All entries resolved.
-->

---

## Appendix A: Terminology

\begin{table*}[ht]
\centering
\small
\begin{tabular}{>{\bfseries}p{4.5cm} p{11cm}}
\hline
\textbf{Term} & \textbf{Definition} \\
\hline
AIC (AMD Infinity Context) & Reference architecture for network-attached KV cache storage on AMD platforms \\[4pt]
AIS (AMD Infinity Storage) & AMD library for efficient GPU-to-storage data movement; AMD's equivalent of NVIDIA GDS \\[4pt]
NIXL (eXtensible I/O Library) & Pluggable I/O abstraction for transferring tensors between GPUs, CPUs, and storage across nodes \\[4pt]
LMCache & Open-source KV cache offload framework; manages chunk lifecycle between inference engine and storage \\[4pt]
hipFile & ROCm API for GPU Direct Storage; enables DMA between GPU HBM and storage without CPU bounce \\[4pt]
Tier 1--4 & KV cache tier hierarchy: Tier 1 (GPU HBM), Tier 2 (Host DRAM), Tier 3 (Local NVMe), Tier 4 (Network Storage) \\[4pt]
Prefill & Transformer phase that generates KV tensors for input tokens; scales quadratically with context length \\[4pt]
Decode & Transformer phase that generates output tokens autoregressively; uses cached KV from prefill \\[4pt]
Prefill-decode disaggregation & Architectural pattern where prefill and decode phases execute on separate hardware \\[4pt]
KV cache & Cached key-value tensors from transformer attention layers, enabling computation reuse \\[4pt]
TTFT & Time to first token; latency from request submission to first output token \\[4pt]
ISL & Input sequence length; number of tokens in the prompt \\[4pt]
OSL & Output sequence length; number of tokens in the model response \\[4pt]
Chunk & Fixed-size segment of KV cache, keyed by prefix hash; enables partial cache hits \\[4pt]
Chunk blending & Merging non-contiguous cached chunks with freshly computed KV \\[4pt]
HBM & High Bandwidth Memory; MI300X provides 192 GB HBM3 at \textasciitilde5.3 TB/s \\
\hline
\end{tabular}
\caption*{Table 3: Glossary of terms used in this paper.}
\end{table*}

## Appendix B: Hardware Compatibility

\begin{table*}[ht]
\centering
\small
\begin{tabular}{>{\bfseries}p{3cm} p{2.5cm} p{3cm} p{7cm}}
\hline
\textbf{GPU} & \textbf{ROCm arch} & \textbf{hipFile Support} & \textbf{Notes} \\
\hline
MI355X & gfx950 & Yes & Primary target; 288 GB HBM3E \\[4pt]
MI300X & gfx942 & Yes & Primary target; 192 GB HBM3 \\[4pt]
MI250X & gfx90a & Yes & \\[4pt]
RX 9070 XT & gfx1201 & Partial & Desktop GPU, no InfiniBand \\[4pt]
Strix Halo (dGPU) & gfx950 & via LMCACHE\_LOCAL & Partner configuration \\[4pt]
\hline
\end{tabular}
\caption*{Table 4: Hardware compatibility matrix.}
\end{table*}
