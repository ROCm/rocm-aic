.. Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
.. SPDX-License-Identifier: MIT

Distributed inference KV-cache architecture
===========================================

This document sketches how **llm-d** on Kubernetes, **vLLM**, **LMCache**,
**hipFile**, and **NVMe-oF over RDMA** fit together when many **AMD Instinct**
workers run **distributed inference** and share a **single NVMe-oF namespace**
for **KV-cache block** storage. It aligns with the exploration goals of this
repository: disaggregated flash for inference context memory, analogous to
platforms that pair GPU clusters with shared NVMe tiers.

Scope and assumptions
---------------------

* **llm-d** provides charts, policies, and placement on the cluster; the
  inference process remains **vLLM** (see the inference stack container notes).
* **LMCache** implements the KV-block offload / reuse tier in front of or
  beside vLLM; configuration may use disk, AIS, or other backends supported by
  your build.
* **hipFile** (with **rocfile** where applicable) supplies **GPU-direct** file
  I/O so LMCache or the runtime can move KV data without extra host copies when
  the deployment enables that path.
* **NVMe-oF** exports a **shared namespace** from a **storage node**; every
  worker’s **initiator** attaches over **RDMA** (RoCE or InfiniBand). All hosts
  that mount or map that namespace see the **same logical block address space**,
  which is what makes a **cluster-wide KV block cache** possible at the storage
  layer.

**Caution:** concurrent **uncoordinated** writes from multiple hosts to one raw
namespace corrupt data. Production designs need explicit coordination: cluster
file systems, reservations, read-only replicas, or a single writer with a
sidecar protocol. The lab **performance** notes call this out for raw **fio**
tests as well.

Logical stack (block diagram)
---------------------------

The diagram below is **Mermaid**. View it in an editor or site that renders
Mermaid (for example GitHub preview, GitLab, or Sphinx with a Mermaid
extension). If your toolchain only shows plain text, copy the block into a
Mermaid live editor.

.. code-block:: mermaid

   flowchart TB
     subgraph Clients["Clients and edge"]
       U[Users / apps / batch jobs]
       API[OpenAI-compatible API / ingress]
       U --> API
     end

     subgraph K8s["Kubernetes cluster"]
       subgraph LlmD["llm-d (control plane on cluster)"]
         SCH[Policies / autoscaling / charts]
         RT[Request routing\nprefill-decode splits\nreplica placement]
         SCH --- RT
       end

       subgraph W1["Instinct worker 1 (ROCm)"]
         V1["vLLM\nmodel weights + scheduler"]
         L1["LMCache\nKV block cache / blending"]
         HIP1["hipFile / rocfile\nGPU-direct file I/O"]
         VR1["GPU HBM\nhot KV pages\n(PagedAttention)"]
         I1["Host NVMe-oF initiator\nRDMA queue pairs"]
         V1 <--> L1
         V1 <--> VR1
         L1 <--> HIP1
         HIP1 --> I1
       end

       subgraph WN["Instinct worker N (same pattern)"]
         VN[vLLM]
         LN[LMCache]
         HIPN[hipFile]
         IN[NVMe-oF initiator RDMA]
         VN <--> LN
         LN <--> HIPN
         HIPN --> IN
       end
     end

     API --> RT

     subgraph Fabric["RDMA-capable data network"]
       RDMA["RoCE or InfiniBand\nNVMe-oF RDMA data path"]
     end

     I1 --> RDMA
     IN --> RDMA

     subgraph Storage["Storage node (NVMe-oF target)"]
       TGT["NVMe-oF target\nsubsystem / ports"]
       NS["One shared namespace\n(same NGUID for all initiators)\nKV blocks as objects / files"]
       MED["NVMe SSDs / flash pool"]
       TGT --> NS --> MED
     end

     RDMA --> TGT

     classDef note fill:#f9f9f9,stroke:#999,color:#333
     class NOTE note
     NOTE["All workers mount or access the same\nnamespace so LMCache + hipFile see\none distributed KV-block store."]

Single compute node (block diagram)
-----------------------------------

The diagram below zooms in to **one** Instinct worker. It shows **vLLM**,
**LMCache**, and **hipFile / rocfile**, how **HBM** holds hot
**PagedAttention** KV pages, and how LMCache’s backing path ties a **mount or
block device** to the **NVMe-oF initiator** and **RNIC** before traffic leaves
the node for the **remote target** and shared namespace.

.. code-block:: mermaid

   flowchart TB
     subgraph Node["One AMD Instinct compute node"]
       direction TB

       subgraph Proc["Inference stack (Pod or host)"]
         H["HTTP / OpenAI handler\n(optional local)"]
         V["vLLM\nmodel weights + scheduler"]
         C["LMCache\nKV block tier"]
         H --> V
         V <--> C
       end

       subgraph GPU["GPU"]
         M["HBM\nPagedAttention KV"]
       end

       V <--> M

       subgraph GDS["GPU-direct cache I/O"]
         F["hipFile / rocfile\n(user APIs)"]
       end

       C <--> F
       F <--> M

       subgraph Host["Linux host"]
         P["LMCache backing path\nmount or block dev"]
         I["NVMe-oF initiator\n/dev/nvme*n*"]
         N["RNIC\nRoCE or IB"]
       end

       F --> P
       P --> I
       I --> N
     end

     N -->|"NVMe-oF over RDMA"| T["Remote target:\nshared namespace"]

Narrative walk-through
----------------------

**Clients** send requests through an **ingress** or gateway that speaks an
OpenAI-style HTTP API. Inside the cluster, **llm-d** (Helm releases, resource
policies, autoscaling) decides where replicas run and how traffic splits between
**prefill** and **decode** roles when that pattern is enabled.

Each **Instinct worker** runs **vLLM**, which holds the **model** and schedules
tokens. Active KV pages live in **GPU HBM** under vLLM’s paging strategy.
**LMCache** stores or retrieves **KV blocks** (often as files or extents) so
repeated or overlapping prompts can reuse work across requests or nodes. When
the stack is built for **AIS** and **hipFile**, LMCache’s I/O path can use
**GPU-direct** reads and writes into buffers tied to the cache files, reducing
**memcpy** pressure on large block moves.

Underneath LMCache, the operating system presents a **file system or raw
device** backed by the **same NVMe-oF namespace** on every worker. The **NVMe-oF
initiator** uses **RDMA** for data and completion traffic to the **target** on
the **storage node**. The target maps the namespace to **local NVMe** (single
drive or pooled volumes). Because every worker’s initiator is attached to that
**one shared namespace**, the set of LMCache objects forms a **single logical
store** visible cluster-wide—subject to the coordination caveat above.

Operational implications
------------------------

**Fabric:** Plan **bandwidth**, **PFC** or **ECN**, and **MTU** consistently
between GPU nodes and the storage tier so NVMe-oF RDMA stays stable under burst
KV traffic.

**Security:** Restrict which nodes may discover the subsystem; use **TLS** for
management paths where your distribution supports it; treat the KV tier as
**sensitive** if prompts or cache metadata must not leak across tenants.

**Sizing:** KV blocks compete with checkpoints, logs, and other I/O. Size the
namespace and drive pool for **working set + headroom**, and monitor **latency**
from LMCache’s perspective (time to first token when cache hits or misses).

Related material in this repository
-----------------------------------

* **Performance notes** for shared-namespace NVMe-oF on the lab fabric are in
  ``performance.rst`` in this directory.
* **ROCm inference stack** container (vLLM, LMCache, hipFile checkout, llm-d
  source for charts): ``recipies/rocm-inference-stack/README.md``.
* **LMCache TTFT benchmark** and AIS-oriented configs: ``benchmarks/ttft-lmcache/``.
* **Ansible** playbooks for NVMe-oF target and initiator roles:
  ``ansible/playbooks/nvme-of.yml`` and inventory under ``ansible/inventory/``.

Upstream references
-------------------

* `llm-d <https://github.com/llm-d/llm-d>`__
* `vLLM <https://github.com/vllm-project/vllm>`__
* `LMCache <https://github.com/LMCache/LMCache>`__
