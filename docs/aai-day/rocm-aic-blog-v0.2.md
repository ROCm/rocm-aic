---
blogpost: true
blog_title: "Introducing ROCm AMD Infinity Context: A Purpose-Built KV Cache Tier for Distributed Inference"
date: 22 Jul 2026
author: "Juergen Frick, Stephen Bates"
thumbnail: ""
tags: inference, storage, distributed-computing, vLLM, LMCache
category: Software tools & optimizations
target_audience: DevOps; Platform engineers; AI infrastructure engineers
key_value_propositions: Lower TTFT, higher GPU concurrency, reduced inference TCO via open shared KV cache tier
language: English
myst:
  html_meta:
    "author": "Juergen Frick; Stephen Bates"
    "description lang=en": "ROCm AMD Infinity Context (AIC) is AMD's open, GPU-direct KV cache tier for distributed LLM inference."
    "keywords": "rocm, KV cache, distributed inference, hipFile, LMCache, NIXL, vLLM, AMD Instinct, NFS over RDMA"
    "property=og:title": "Introducing ROCm AMD Infinity Context: A Purpose-Built KV Cache Tier for Distributed Inference"
    "property=og:description": "ROCm AMD Infinity Context (AIC) is AMD's open, GPU-direct KV cache tier for distributed LLM inference."
    "property=og:type": "article"
    "property=og:site_name": "ROCm Blogs"
    "property=og:locale": "en_US"
    "amd_category": "Developer Resources"
    "amd_asset_type": "Blogs"
    "amd_blog_topic_categories": "AI & Intelligent Systems; Software & Ecosystem"
    "amd_technical_blog_type": "Tools, Features and Optimizations"
    "amd_blog_hardware_platforms": "Instinct GPU Accelerators; Pensando Network Infrastructure"
    "amd_blog_development_tools": "ROCm Software"
    "amd_blog_applications": "AI Inference; Deploying AI at Scale; Conversational AI"
    "amd_blog_authors": "Juergen Frick; Stephen Bates"
    "amd_blog_releasedate": "Tue Jul 22, 09:00:00 PST 2026"
---

# Introducing ROCm™ AMD Infinity Context: A Purpose-Built KV Cache Tier for Distributed Inference

As AI models grow larger and inference workloads become more complex, a new bottleneck is emerging at the heart of production AI systems: **KV cache management**. Long-context requests, agentic workflows, and multi-turn conversations are generating key-value (KV) caches that no longer fit comfortably in GPU High-Bandwidth Memory (HBM). The result is either expensive recomputation, node-bound local NVMe storage, or slow general-purpose network file systems — none of which are designed for the demands of modern inference.

Today, AMD introduces **ROCm™ AMD Infinity Context (ROCm AIC)** — a purpose-built, open, AI-native KV cache storage tier for distributed inference on AMD Instinct™ GPUs. In this blog, we walk you through the problem, the solution, the components that make up the stack, and how ROCm AIC is designed to work with your existing AMD GPU deployments starting with the Instinct MI300X Series and MI350 Series, and extending to the new MI450 Series / Helios platform.

---

## The Problem: KV Cache Is Becoming the Bottleneck

When an LLM processes a prompt, it generates a **KV (key-value) cache** that represents the internal state for every token in the input. This cache is needed repeatedly during decoding and can be reused across similar requests. For short prompts on a single node, keeping the KV cache in GPU HBM works fine. But modern inference is changing fast:

- **Context windows are growing 10× per generation.** Models like GPT-OSS-120B and Qwen3 (235B) now support 128k–1M token contexts. A single 1M-token request can generate over 600 GB of KV data — more than the HBM capacity of an entire MI455X node.
- **Agentic AI demands cross-session KV reuse.** AI agents replay long prompt histories on every turn. Without KV caching across turns, every step pays the full prefill cost — an O(N²) FLOPS operation in context length N. At 100k tokens, that is trillions of FLOPs per turn on a GPU consuming over 1000W.
- **Local NVMe is node-bound.** The common workaround of offloading KV cache to local NVMe via host CPU and RAM works on a single node, but KV blocks cannot be efficiently shared across GPU nodes. This limits the value of local NVMe storage as a caching tier for frameworks like LMCache when used for multi-node, shared-context workloads such as RAG pipelines, agentic loops, and multi-tenant chat services.

The scale of the problem translates directly to real costs. At the time of the writing of this blog, HBM3e costs approximately $50/GB at the system level. Wasting HBM on cold KV cache is one of the largest TCO inefficiencies in inference deployments today. Recomputing KV cache wastes GPU compute that could be serving new requests.

---

## ROCm AIC: What It Is

**ROCm™ AMD Infinity Context (ROCm AIC)** is a combination of software and hardware that leverages low-latency, RDMA-capable networks as an intelligent caching tier for KV cache prefill data in distributed inference workloads.

ROCm AIC provides a third storage tier that sits below GPU HBM and CPU DRAM in the KV cache hierarchy:

![KV cache tier hierarchy — G1 (GPU HBM, nanoseconds), G2 (System DRAM, 10–100 ns), G3 (Local SSD/Flash, microseconds), G3.5 (Ethernet-attached Flash, microseconds), G4 (Shared Object/File, milliseconds)](kv-tier-diagram.png)

The key innovation in ROCm AIC is making the network-attached tier fast enough for production inference. It does this by combining AMD hardware assets with open-source inference frameworks to create a GPU-direct path between inference engines and network-attached storage — bypassing the CPU memory bottleneck entirely.

ROCm AIC is applicable to:
- **AMD Instinct MI300X Series and MI350 Series GPUs** — via front-side Ethernet network attach
- **AMD MI450 Series / Helios** — via front-side Ethernet *or* high-bandwidth MI455X Scale-Out network

---

## The ROCm AIC Software Stack

ROCm AIC is designed with an **upstream-first, open-source philosophy**. There is no proprietary software layer. Every component is either an existing AMD asset, an open-source framework, or an open standard. This means ROCm AIC integrations are portable and compatible with the inference ecosystem you are already using.

The stack is organized into the following layers. *PoC indicates supported as part of a Proof of Concept announced at AMD Advancing AI Day 2026; additional frameworks will follow.*

### Orchestration: llm-d + Kubernetes (PoC)

**llm-d** handles inference orchestration — routing requests to the right model replica, deciding on prefill vs. decode worker placement, and managing KV cache locality so that requests are directed to replicas that already hold the relevant cached KV blocks. Kubernetes provides the deployment plane for containerized inference clusters.

### Serving Layer: vLLM (PoC) → SGLang

The inference engine where LLM serving happens. ROCm AIC starts with **vLLM** as the primary serving framework — the leading open-source inference engine with ~50k GitHub stars. SGLang and Triton support follow in subsequent phases.

### KV Management: LMCache (PoC) → Mooncake

**LMCache** is the KV block manager. It decides which KV blocks live in GPU HBM, which are evicted to CPU DRAM, and which are offloaded to the remote storage tier. LMCache tracks block identifiers, handles prefix-aware caching, and orchestrates reuse across requests and across nodes. AMD contributes the ROCm AIC integration directly upstream to LMCache — no proprietary fork required.

### KV Transfer Engine: NIXL (PoC)

**NIXL** (Notional Inference Transfer Layer) is a high-performance, RDMA-capable transfer engine for moving KV blocks between GPU memory and the storage tier. NIXL is open-source and AMD co-maintains it. It supports both POSIX-based NVMe paths and GPU-direct object store paths, and is designed to enable cross-node KV migration for prefill/decode disaggregation scenarios.

### GPU ↔ Storage Fast Path: ROCm hipFile + AMD Infinity Storage

**ROCm hipFile** is AMD's GPU-direct file I/O library. It enables data transfers directly between GPU HBM and storage — bypassing the host CPU memory bus. hipFile reached GA with ROCm 7.14 (June 2026) and is the critical AMD asset that makes the remote KV tier practical in production. **AMD Infinity Storage (AIS)** is the umbrella for hipFile and future libraries — including hipObject for object store support — to provide optimized, GPU-aware I/O management across local and networked storage backends.

### Network Fabric: Dual-Mode Attach

ROCm AIC uniquely supports two network attach modes:

- **Front-side Ethernet** — uses standard Ethernet infrastructure, available on all AMD Instinct MI300 Series, MI350 Series, and MI400 Series GPU platforms. A key benefit is compatibility with existing GPU cluster deployments.
- **AMD Instinct MI400 Series Scale-Out Network** — for MI455X / Helios deployments requiring maximum storage bandwidth. Connects directly to the GPU scale-out fabric or can be used to create a dedicated context storage network, enabled by 800GbE AMD Pensando AI-NICs.

### NIC Support: Pensando AI-NIC (PoC), Broadcom Thor2, Mellanox ConnectX

ROCm AIC supports multiple RDMA-capable NICs:
- **AMD Pensando AI-NIC** — primary NIC with RDMA acceleration for NFS over RDMA paths
- **Broadcom BCM57608 (Thor2)** — alternative NIC for broader deployment compatibility
- **Mellanox ConnectX** — planned support for broader ecosystem reach

There is no requirement to adopt a specific SmartNIC. ROCm AIC works with existing RDMA-capable NIC deployments.

### Storage Backend: NFS over RDMA (PoC) → Object over RDMA, WekaFS, Lustre, IBM GPFS

The initial storage backend is **NFS over RDMA** — an open standard supported by storage vendors including VAST, Dell PowerScale (formerly Isilon), and NetApp. This means you can use your existing NAS/NFS storage infrastructure as the KV cache tier, as long as it is RDMA-capable. Later phases expand to object over RDMA, WekaFS, Lustre, and IBM GPFS for broader storage ecosystem support.

---

## Architecture: How It Flows

Here is how a KV cache offload request flows through the ROCm AIC stack:

1. **vLLM** receives an inference request and begins prefill. LMCache intercepts KV block generation.
2. **LMCache** checks whether the required KV prefix is already in GPU HBM or CPU DRAM. If not, it initiates a fetch from the ROCm AIC tier.
3. **NIXL** receives the transfer request and initiates an RDMA operation via the configured NIC (Pensando AI-NIC or BCM57608).
4. **hipFile / AIS** manages the GPU-direct data path — data moves from networked storage directly into GPU HBM without passing through host CPU memory.
5. **LMCache** signals vLLM that the KV blocks are ready. Prefill resumes from the cached state, skipping the expensive recompute.

The net result: requests that would have required seconds of GPU compute to recompute KV context can instead be served from the storage tier in a fraction of the time, enabling dramatically lower Time to First Token (TTFT) and higher concurrency at the same GPU footprint.

---

## Why This Matters: Key Benefits

**Reduced Time to First Token (TTFT)**
By eliminating redundant KV recompute, ROCm AIC removes the dominant latency driver for long-context inference. For shared-context workloads with high cache hit rates, TTFT improvements of 10× or more are achievable.

**Higher Throughput and Concurrency**
When GPU compute is no longer consumed by KV prefill for cached content, the same GPUs can handle more concurrent requests. Idle or partially active conversations can have their KV state evicted to the remote tier, freeing HBM for active requests.

**Lower Infrastructure Cost**
A terabyte of NVMe-based networked storage costs a fraction of a terabyte of GPU HBM. By storing cold KV cache in the ROCm AIC tier, clusters can handle orders of magnitude more KV state without adding GPU nodes. If your GPU cluster BOM had planned for large local NVMe storage for KV offload, moving this to a dedicated networked storage tier with ROCm AIC can dramatically increase cache hit rates across distributed inference workloads.

**Longer Context Windows in Practice**
With an expanded KV tier, inference engines can serve prompts far longer than GPU HBM alone would permit — enabling practical deployment of 1M+ token context models.

**Open and Vendor-Neutral**
All ROCm AIC components are contributed upstream to open-source projects. There is no proprietary API surface. Storage vendors can integrate with ROCm AIC using standard NFS over RDMA, and in the future with object over RDMA and other network file systems.

---

## Use Cases Where ROCm AIC Makes the Biggest Difference

ROCm AIC delivers the most value in workloads where KV cache reuse across requests or users is high:

- **Agentic AI and multi-turn chatbots** — system instructions and conversation history can be cached once and reused across all subsequent turns
- **RAG (Retrieval-Augmented Generation)** — documents indexed into KV cache are shared across all queries against the same corpus
- **Code repository analysis** — long codebases analyzed once; repeated queries reuse cached context
- **Legal document review** — large documents read once; multiple questions answered from the same KV state
- **Video and media analysis** — lengthy content processed once and cached for repeated downstream queries
- **Long-running inference services** — warm caches maintained in the storage tier survive across session boundaries

---

## Platform Support

| Platform | Network Attach | Status |
|----------|---------------|--------|
| AMD Instinct MI300X Series | Front-side Ethernet | Available |
| AMD Instinct MI350 Series | Front-side Ethernet | Available |
| AMD MI450 Series / Helios | Front-side Ethernet + MI455X Scale-Out | Upcoming |

---

## Open-Source and Community Contributions

ROCm AIC is built on upstream-first principles. AMD's contributions include:

- **LMCache** — KV block management integration contributed upstream
- **NIXL** — co-maintained transfer engine; connects to hipFile (AMD Infinity Storage)
- **vLLM** — ROCm AIC support contributed to the main vLLM repository
- **ROCm hipFile** — open-source under ROCm, GA in ROCm 7.14
- **ROCm-AIC** — reference implementation, benchmarks, container recipes, and Grafana dashboards at [github.com/ROCm/rocm-aic](https://github.com/ROCm/rocm-aic)

Storage vendors and NIC vendors can integrate with ROCm AIC using standard interfaces. No proprietary co-design certification is required.

---

## What's Coming Next

ROCm AIC is being demonstrated at AMD Advancing AI Day, July 2026. The roadmap beyond the initial demo includes:

- **Q3 2026:** Customer pilot engagements with Helios deployments; multi-node shared KV tier across nodes
- **Q4 2026:** Object over RDMA storage backend; expanded storage partner support (VAST, Dell PowerScale, WEKA); broader NIC support
- **H1 2027:** Production GA hardening; NVMe KV Command Set support; GPU-initiated storage I/O; Lustre and IBM GPFS integration

---

## Summary

ROCm™ AMD Infinity Context (ROCm AIC) is AMD's answer to the KV cache scaling challenge in modern distributed inference. As AI models grow and context windows expand, keeping all KV state in GPU HBM becomes technically and economically unsustainable.

ROCm AIC introduces a high-performance, open-source, GPU-direct storage tier for KV cache that:
- Uses **standard storage protocols and network storage devices** — no proprietary storage APIs required
- Integrates **upstream into open frameworks** (vLLM, LMCache, NIXL) — no private API surface
- Supports **AMD Instinct MI300X Series through MI450 Series** — immediate value for existing AMD deployments
- Enables **dual network attach** — front-side Ethernet for broad compatibility, MI455X Scale-Out for highest bandwidth

The result is lower TTFT, higher concurrency, longer practical context windows, and better GPU utilization — at a fraction of the cost of scaling HBM.

To get started, explore the reference implementation at [github.com/ROCm/rocm-aic](https://github.com/ROCm/rocm-aic) and try the benchmark scripts included in the repository. For more on AMD Infinity Storage and hipFile, see the [ROCm hipFile documentation](https://rocm.docs.amd.com/).

---

*Authors: Juergen Frick, Director of Infrastructure Software Product Management, AMD Data Center GPU Business Unit; Stephen Bates, Fellow, AI Storage Architecture, AMD Data Center GPU Business Unit.*
