# KV-cache Cliff Run — Full Analysis (Qwen2.5-3B, MI300X)

Definitive **with vs without NVMe** analysis of the KV-cache cliff, plus GPU/IO,
kernel, SDMA, and AIS-IO findings.

- **Primary run:** job `67534106` (`ctr-cx66-mi300x-13`) — both arms `vram_only`
  + `kvd_v2 nvme`, so it gives the real baseline (not a proxy).
- **Kernel inventory** sourced from matched run `67534081` (identical config;
  hsa-snoop mis-targeted PIDs on `67534106`'s busy node — see caveats). See
  [gpu-kernel-analysis.md](gpu-kernel-analysis.md) for the full kernel breakdown.
- **Config:** `Qwen/Qwen2.5-3B-Instruct` · ISL 20 000 (18 000 shared prefix +
  2 000 unique) · `kv-cache-dtype fp8` · `TRITON_ATTN` · `gpu-memory-utilization
  0.12` · LMCache `LMCacheConnectorV1` + NIXL `AIS_MT` → NVMe, `nixl_buffer_device:
  cuda`, `use_direct_io: true`, `local_cpu: false` · tier on dedicated `nvme1n1` ·
  prefill-dominated (`max_tokens=1`) · `BENCH_CONCUR=128,250`, `BENCH_ITERS=2`.

---

## 1. WITH vs WITHOUT NVMe + the cliff point

| Concurrency | `vram_only` (no NVMe) | `kvd_v2` NVMe (89.6% hit) | **NVMe speedup** |
|---|---|---|---|
| c=128 | **15,532 tok/s** | **63,186 tok/s** | **4.07×** |
| c=250 | **15,495 tok/s** | **62,987 tok/s** | **4.07×** |

- Both curves are **flat across concurrency** because both points sit **deep past
  the cliff**. `vram_only` is flat because it **recomputes** the full 20k-token
  prefill every iteration (compute-bound); NVMe is flat because it **fetches KV
  from `nvme1n1`** at 89.6% hit.
- **VRAM KV capacity = 929,984 tokens ≈ 46 concurrent requests** of 20k tokens
  (vLLM-reported; ~17 GB HBM for KV at the 0.12 mem-util budget). fp8 KV for this
  model = **18.4 KB/token** (36 layers × 2 × 2 KV-heads × 128 head-dim × 1 B).
- **Cliff onset ≈ 46 concurrent requests** (or ~51 by the 18k reusable prefix).
  Below it: VRAM holds the working set, both arms identical, NVMe idle. Above it:
  `vram_only` recomputes (the cliff) while NVMe spills to storage.
- Test points are **2.5× (c=128)** and **4.8× (c=250)** past the VRAM cap
  (working sets ~2.3 M / ~4.5 M prefix tokens vs 0.93 M capacity).

**Two knobs move the cliff:** `gpu_memory_utilization` (0.12 induces the cliff at
~46 req; 0.90 would push it ~8× higher) and **context length** (longer prompts →
cliff at far lower concurrency — where offload pays off most).

---

## 2. GPU compute/memory + IO — vram arm vs nvme arm

*(GPU `gpu_` metrics were absent from this run's TSDB — the amd-metrics-exporter
scraped `up=0` on :5000/:5050 — so GPU figures are from the node-side `amd-smi`
queue/activity sampler + live reads.)*

| | vram arm | nvme arm |
|---|---|---|
| GPU GFX activity | **96%** avg (95% of samples ≥95%) | **77%** avg (50% ≥95%) |
| GPU UMC (HBM mem-ctrl) | 2.2% | 1.5% |
| GPU power / clock (live) | ~747 W / 1976 MHz | — |
| `nvme1n1` read / write | 0 / 0 | **426 / 183 MB/s** |
| `nvme1n1` %util | 0% | **13.3%** (~361 read IOPS) |
| root `nvme0n1` write | 0.3 MB/s (baseline) | 0.3 MB/s (baseline) |
| DRAM MemAvailable | ~1525 GB | ~1519 GB |
| DRAM page-cache Δ | ~0 GB | ~0 GB |

- **`vram_only` is GPU-compute-bound** — GFX pinned ~96% recomputing prefill; UMC
  ~2% → **not** HBM-bandwidth-bound.
- **NVMe arm frees GPU compute** — GFX drops to 77% (no recompute), trading GPU
  cycles for NVMe fetches, and still delivers 4× throughput.
- **NVMe tier is far from saturated** — 13.3% util / 426 MB/s time-averaged vs
  LMCache's **4.35 GB/s** retrieval bursts. Bursty, huge headroom → **the ceiling
  is GPU compute, not storage.** Read ≫ write (cache hits dominate).
- **Root disk untouched** (0.3 MB/s baseline both) → dedicated-spare move worked.
- **Host DRAM barely moves** (MemAvailable ±6 GB, page-cache Δ≈0) → the
  GPU-VRAM-staging ↔ NVMe path (O_DIRECT + cuda staging + `local_cpu:false`)
  **bypasses host DRAM** as designed. (`L1_hit≈0%` in the cliff output confirms no
  DRAM L1 tier engaged.)
- **LMCache logical throughput (nvme arm):** retrieval avg **4.35 GB/s** (max
  4.58), store avg **7.31 GB/s** (max 13.1) — per-request bursts.

---

## 3. Data-movement GPU kernels

All are **COMPUTE / AQL dispatches** (they transform/place KV; they are not the
DMA transfer). Counts from matched run `67534081` — full list in
[gpu-kernel-analysis.md](gpu-kernel-analysis.md).

| Kernel | Role | Tier |
|---|---|---|
| `reshape_and_cache_kernel_flash` (~17.8k) | KV write into paged cache | **VRAM (HBM)** |
| `lmc::load_and_reshape_multi_layer_kernel<…false…>` (~3.2k) | LMCache **load** reshape | **NVMe** (this config) |
| `lmc::load_and_reshape_multi_layer_kernel<…true…>` (~1.7k) | LMCache **store** reshape | **NVMe** (this config) |
| `__amd_rocclr_copyBuffer` (~1.2k) | ROCr blit copy around the cuda staging buffer | staging |
| `vectorized_gather` / `scatter_gather` | indexing / gather | — |

**DRAM vs NVMe nuance:** the `lmc::load_and_reshape` kernels are **tier-agnostic** —
the same kernel serves a DRAM-L1 hit or an NVMe-L2 hit; the tier is a host-side
LMCache decision. Here `local_cpu:false` → these = **NVMe**. Use
`ext_hit`/`L1_hit` + `nvme_`/`node_disk` to attribute DRAM vs NVMe, not the kernel
name.

---

## 4. SDMA usage — measured

From the KFD queue sampler (`sdma_q` = type-1 queues), **constant 2 in both arms**
(vram: 2 for all 179 samples; nvme: 2 for 131/132):

- The vLLM/ROCr process holds **2 SDMA queues regardless of arm** → ordinary
  `hipMemcpyAsync` (H2D input tokens, D2H outputs) uses SDMA in both arms.
- **The nvme arm adds NO SDMA queues** → the AIS NVMe↔VRAM bulk transfer is **not**
  routed through extra GPU-SDMA. It uses **true-GDS PCIe P2PDMA** (NVMe controller
  DMAs into VRAM) + the compute reshape/blit kernels for staging.

**Three data paths, confirmed:**
1. **Compute reshape/blit kernels** — CU/AQL, visible in hsa-snoop (§3).
2. **SDMA async copies** — 2 baseline queues, *not* kernel-visible, unchanged by NVMe.
3. **GDS NVMe↔VRAM** — PCIe peer-to-peer DMA, bypasses **both** GPU compute and SDMA.

*(amd-smi exposes no SDMA busy%; conclusion rests on the queue sampler + kernel
evidence + architecture.)*

---

## 5. AIS IO — from ais-snoop (`kfd_ioctl_ais`)

| | vram arm | nvme arm |
|---|---|---|
| `kfd_ioctl_ais` calls | **0** | **88,529** (~**128/s**) |

- **Textbook validation:** zero AIS activity in `vram_only`; ~88.5k calls in the
  NVMe arm — the AIS path lights up exactly when the NVMe tier is engaged.
- At ~426 MB/s that's ~3.3 MB/AIS-call, correlating with LMCache's 4.35 GB/s
  retrieval / 7.31 GB/s store bursts.
- **Gap:** per-call **latency didn't populate** (`ais_kfd_latency_seconds_count`
  stayed 0 — the kretprobe/return side isn't counting). We have AIS **call rate**
  but not latency this run — a target for the ais-snoop fix.

---

## 6. Bottom line

On this VRAM-starved config (0.12 util → ~46-request cliff), the NVMe tier
delivers a flat **4.07× throughput win** (63k vs 15.5k tok/s) by serving **89.6%**
of prefix KV from `nvme1n1` instead of recomputing — at only **13.3% NVMe util**,
**~0 host-DRAM footprint**, **no extra SDMA queues**, and modest extra compute
kernels. The workload is **GPU-compute-bound**; the AIS/NVMe offload path does its
job cheaply with large headroom.

---

## Telemetry caveats (this run)

1. **amd-metrics-exporter didn't scrape** (`up=0` on :5000/:5050) → no `gpu_`
   metrics in the TSDB; GPU compute came from the `amd-smi` sampler instead.
2. **hsa-snoop mis-targeted PIDs** on this busy shared node — its per-PID
   `kfd_ioctl_create_queue` discovery latched onto departed neighbor tenants and
   never tracked our vLLM EngineCore (0 `hsa_` series). Kernel data therefore came
   from matched run `67534081`. (Fix ideas: register all PIDs / add a target
   PID-comm filter / periodically re-scan `/sys/class/kfd/.../queues`.)
3. **ais-snoop latency** not populated (call count only) — kretprobe return path.

*Sources: `logs/67534106/{cliff.out, prometheus/, engine-sample.log,
container-aai-cliff-kvd-vllm.log}`; kernel counts from `67534081`. Analysis
Prometheus was a throwaway container against the retained TSDB (torn down).*
