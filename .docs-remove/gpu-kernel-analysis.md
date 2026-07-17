# GPU Kernel Analysis — KV-cache cliff run (Qwen2.5-3B, MI300X)

What GPU kernels vLLM + LMCache launch during a cliff run, and which ones are
part of the **KV-cache** path (VRAM, CPU DRAM, or NVMe).

- **Source:** hsa-snoop (`hsa_kernel_launches_total`, AQL dispatch kprobe) on the
  cliff `kvd_v2` **nvme** arm — job `67534081` (`ctr-rack31-mi300x-3`), which runs
  the identical workload/config to the in-flight `67534106`
  (`Qwen/Qwen2.5-3B-Instruct`, ISL 20 000 = 18 000 shared prefix + 2 000 unique,
  `kv-cache-dtype fp8`, `TRITON_ATTN`, `gpu-memory-utilization 0.12`, LMCache
  `LMCacheConnectorV1` + NIXL `AIS_MT` → NVMe with a **cuda (VRAM) staging
  buffer**, `local_cpu: false`).
- **Scope:** this is a **prefill-dominated** benchmark (`max_tokens=1`), so
  attention/GEMM/KV kernels dominate and decode/sampling kernels are minimal.
- Launch counts are over the timed run window; treat them as *relative* weights,
  not absolutes.

## Top kernels by launch count

| Launches | Kernel | Category | KV-cache? |
|---:|---|---|---|
| 36,465 | `add_rmsnorm_quant_kernel<bf16,bf16>` | Norm + fp8 quant | — |
| 18,233 | `kernel_unified_attention` | Attention | **reads** VRAM KV cache |
| 18,233 | `triton_poi_fused_mul_silu_slice_0` | MLP activation (SwiGLU) | — |
| 18,122 | `Cijk_…_MT256x128x64_…` (Tensile) | GEMM (matmul) | — |
| 18,048 | `Cijk_…_MT256x192x64_…` (Tensile) | GEMM (matmul) | — |
| 18,048 | `Cijk_…_MT192x160x64_…` (Tensile) | GEMM (matmul) | — |
| 18,046 | `triton_poi_fused_1` | Pointwise (Triton) | — |
| 17,990 | `Cijk_…_MT256x160x64_…` (Tensile) | GEMM (matmul) | — |
| 17,758 | **`reshape_and_cache_kernel_flash`** | **KV-cache write (VRAM)** | ✅ VRAM (HBM) |
| 3,172 | **`lmc::load_and_reshape_multi_layer_kernel<long, false, …>`** | **LMCache load** | ✅ NVMe (this run) |
| 1,722 | **`lmc::load_and_reshape_multi_layer_kernel<long, true, …>`** | **LMCache store** | ✅ NVMe (this run) |
| 1,194 | **`__amd_rocclr_copyBuffer`** | ROCr blit copy | ✅ staging-buffer copy |
| 666 | `_gumbel_sample_kernel` | Sampling | — |
| 666 | `reduce_kernel<…ArgMaxOps…>` | Sampling (argmax) | — |
| 666 | `_scatter_gather_elementwise_kernel` | Token scatter/gather | — |
| 492–666 | `index_elementwise_kernel<…>` | Indexing | — |
| 174 | `vectorized_gather_kernel` | Embedding gather (HBM) | — |
| 48 | `wvSplitK_hf_sml_<bf16,…>` | Skinny/split-K GEMM | — |
| 7 | `_prepare_pos_seq_lens_kernel` | Attention metadata | (KV bookkeeping) |

## KV-cache kernels (called out)

There are **two tiers** of KV cache in play, and a different kernel touches each.

### 1. VRAM (HBM) paged KV cache — vLLM's own, hot tier
- **`reshape_and_cache_kernel_flash`** — writes freshly-computed K/V into vLLM's
  **paged KV-cache blocks in HBM**, reshaping from the attention layout into the
  paged block layout. This is the primary in-GPU KV-cache management kernel and
  runs in **every** arm (vram_only *and* nvme). ~17.8k launches ≈ one per
  layer-step, on par with the attention/GEMM kernels.
- **`kernel_unified_attention`** — not a KV-*movement* kernel, but it is the
  consumer: it **reads** the paged VRAM KV cache during attention. Included here
  because it's the demand side of the cache.

### 2. LMCache offload tier — DRAM L1 and/or NVMe L2 (the "cliff" fix)
When the working set overflows the VRAM KV budget (~929,984 tokens here, ~46
requests of 20k tokens — see the cliff analysis), KV spills to the LMCache tier.
The GPU-side of that movement is:

- **`lmc::load_and_reshape_multi_layer_kernel<long, false, (EngineKVFormat)2>`**
  — **LOAD**: reshape KV chunks *fetched from the LMCache tier* into vLLM's paged
  layout on the GPU (the `false` template = load direction). This is the GPU side
  of a **cache hit** being pulled back in. ~3.2k launches.
- **`lmc::load_and_reshape_multi_layer_kernel<long, true, (EngineKVFormat)2>`**
  — **STORE**: gather/reshape KV *out of* the paged layout to offload to the tier
  (`true` = store direction). ~1.7k launches.
- **`__amd_rocclr_copyBuffer`** — ROCr's compute-shader **blit copy**, used for
  the buffer copies around the NIXL **cuda staging buffer** (`nixl_buffer_device:
  cuda`) that the AIS_MT path stages KV through. ~1.2k launches.

**DRAM vs NVMe — important nuance:** the `lmc::load_and_reshape` kernels are
**tier-agnostic**. They reshape KV once it is already in a GPU-visible staging
buffer; whether those bytes were served from **DRAM (L1)** or **NVMe (L2)** is a
**host-side** LMCache decision (the lookup subprocess), *not* a different GPU
kernel. In **this** run `local_cpu: false` (no DRAM L1) and the backend is NIXL
`AIS_MT → NVMe` via hipFile, so these kernels correspond to **NVMe** traffic. If
DRAM L1 were enabled, the *same* kernels would serve DRAM hits — you cannot tell
DRAM vs NVMe from the kernel name alone (use `ext_hit`/`L1_hit` in the cliff
output, LMCache logs, and `node_disk`/`nvme_` metrics for that).

## What does *not* show up as a kernel (and why)

The bulk byte movement is often **not** a GPU kernel at all — hsa-snoop only sees
**AQL compute dispatches**:

- **SDMA async copies** (`hipMemcpyAsync` H2D/D2H/D2D) run on the GPU's **SDMA
  queues** (KFD queue type 1 — confirmed: the vLLM process holds 2 SDMA queues),
  not the compute queues, so they are **invisible** to `hsa_kernel_launches_total`.
- **True GDS / AIS_MT NVMe↔VRAM** transfer is **PCIe peer-to-peer DMA** (the NVMe
  controller DMAs directly into VRAM), which bypasses **both** GPU compute *and*
  SDMA — also invisible here.

So the KV *reshape/placement* is compute (visible above), while the KV *transfer*
is SDMA and/or NVMe-DMA (measured separately via the queue sampler, LMCache
throughput logs, and `nvme1n1` diskstats).

## One-line takeaway

Of the launched kernels, **`reshape_and_cache_kernel_flash`** is the VRAM KV-cache
write, and the **`lmc::load_and_reshape_multi_layer_kernel`** (load/store) plus
**`__amd_rocclr_copyBuffer`** are the LMCache offload-tier path — NVMe in this
config (DRAM if L1 were enabled). Everything else is the model's own compute
(GEMMs, attention, RMSNorm/quant, MLP, sampling).

---
*Generated from hsa-snoop telemetry; counts from job 67534081 (matched config to
67534106). Regenerate against a specific run's `logs/<job-id>/prometheus` TSDB via
`hsa_kernel_launches_total`.*
