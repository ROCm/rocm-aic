# Cross-GFX cliff sweep — how far does the release image travel?

Short cliff runs fanned out **in parallel** across as many Markham GFX arches as
had a free GPU, to answer two questions: (1) which AMD GPU generations can run the
aai-day release image at all, and (2) does the LMCache POSIX-tiered arm recover
the prefix-cache cliff on each.

Uniform recipe (one job per arch, submitted in parallel): **Qwen2.5-3B-Instruct**
(full-attention — the LMCache connector needs it), ISL 8000 / shared prefix 7000
(fits Qwen's native 32k window, no YaRN), `per_client`, `BENCH_CONCUR=1,8,32`,
1 iter, arms **vram_only + kvd_v2 nvme (LMCache native `LocalDiskBackend`, POSIX)**,
DRAM L1 = 4 GB, `gpu_memory_utilization=0.85` (fits Qwen weights on the smallest
24 GB card). HF cache + the 11 GB multi-arch image staged on shared `$HOME`.

## TL;DR

- **The release image (fp8 KV cache + ROCm AITER) runs vLLM only on CDNA3/3.5** —
  gfx942 (MI300X/MI300A) and gfx950 (MI350X). On **MI210, and all RDNA3** the
  vram arm dies at vLLM engine init.
- **Root cause: `--kv-cache-dtype fp8` + `VLLM_ROCM_USE_AITER=1`.** Both are
  MI300/MI350-specific (MI210/MI100 have no native fp8; AITER kernels target
  CDNA3+). Switching to **`--kv-cache-dtype auto` (fp16) + `VLLM_ROCM_USE_AITER=0`**
  brings up **MI210 and all three RDNA3 variants** cleanly. (Exposed as two new
  backward-compatible sbatch knobs, `AIC_KV_CACHE_DTYPE` / `AIC_ROCM_USE_AITER`;
  defaults unchanged = fp8/aiter.)
- **MI100 (gfx908, CDNA1) and RDNA4 (gfx1201) do not come up even with fp16 +
  no-AITER** — vLLM engine startup exceeds the 1200 s wait. Deeper support gap
  (kernels / attention backend); needs more than a dtype/AITER toggle.
- **Where vLLM runs, the LMCache POSIX-tiered arm recovers the cliff on every
  arch — a consistent ~3.3–4.3× at c=32.** The tiering win is architecture-independent.

## Coverage

| GFX | product | gen | vLLM boots? | config needed | both arms swept |
|---|---|---|---|---|---|
| gfx942 | MI300A (128 GB) | CDNA3 | yes | release (fp8 + AITER) | ✅ |
| gfx950 | MI350X (288 GB) | CDNA3.5 | yes | release (fp8 + AITER) | ✅ (prior runs) |
| gfx90a | MI210 (64 GB) | CDNA2 | yes | **fp16 + no-AITER** | ✅ |
| gfx1100 | Radeon RX 7900 | RDNA3 | yes | **fp16 + no-AITER** | ✅ |
| gfx1100w | Radeon Pro W7900 | RDNA3 | yes | **fp16 + no-AITER** | ✅ |
| gfx1100p | RDNA3 (Pro) | RDNA3 | yes | **fp16 + no-AITER** | ✅ |
| gfx908 | MI100 (32 GB) | CDNA1 | **no** | fp16+no-AITER still times out | ❌ |
| gfx1201 | RDNA4 | RDNA4 | **no** | fp16+no-AITER still times out | ❌ |

Not attempted (drained/reserved/other): gfx1101v, gfx1102, gfx1151 (Strix Halo)
all drained; gfx906 (MI60) down; the one free "GPU" node was an NVIDIA
RTX PRO 4000 (CUDA, not ROCm).

## Throughput (tok/s) — vram_only vs kvd_v2 (POSIX L2)

| GFX (product) | arm | c=1 | c=8 | c=32 | recovery @ c=32 |
|---|---|---|---|---|---|
| gfx942 (MI300A) | vram | 21,154 | 18,407 | 21,363 | — |
| | **kvd** | 20,754 | 65,955 | **70,003** | **3.3×** |
| gfx90a (MI210) | vram | 34,763 | 11,099 | 12,515 | — |
| | **kvd** | 37,850 | 48,906 | **50,010** | **4.0×** |
| gfx1100 (RX 7900) | vram | 24,679 | 5,909 | 6,660 | — |
| | **kvd** | 23,757 | 26,247 | **26,786** | **4.0×** |
| gfx1100w (W7900) | vram | 22,824 | 5,406 | 6,027 | — |
| | **kvd** | 20,736 | 23,858 | **24,017** | **4.0×** |
| gfx1100p (RDNA3 Pro) | vram | 14,524 | 3,463 | 3,841 | — |
| | **kvd** | 13,784 | 16,141 | **16,531** | **4.3×** |

Peak sustained (kvd, c=32) ranks by silicon as expected: **MI300A 70k > MI210 50k
> RX 7900 27k ≈ W7900 24k > RDNA3-Pro 16.5k**. (MI350X/gfx950 from the separate
128k O_DIRECT study peaked ~118k at c=16 with page-cache-served L2 — see
`cliff-odirect-readpath-gfx942-vs-gfx950.md`.)

## The cliff, and why the tiered arm holds

For **every** arch the vram_only arm shows the same shape: fast at c=1, then a
sharp drop at c=8/c=32 as vLLM's VRAM prefix cache hit rate collapses (scraped
`l1_hit`: **87% → 11% → 22%**) and the fresh per-client prefixes are recomputed.
The kvd_v2 arm keeps its scraped prefix hit at **87% across all concurrencies**:
LMCache persists each concurrency's KV (per-concurrency warmup populates it) and
restores it instead of recomputing, so throughput *rises* with concurrency rather
than cliffing. `ext_hit` was 0% throughout — at ISL 8000 the working set fit the
4 GB DRAM L1, so nothing spilled to the POSIX disk L2 (unlike the 128k study,
where it did). The recovery here is the **DRAM-L1 tier**; the disk L2 only engages
at larger working sets.

Caveat: the vram arm gets a single global warmup while the kvd arm gets a
per-concurrency warmup, so part of the c=8/c=32 gap is priming, not just tiering.
The absolute kvd throughput and the flat 87% hit are still the representative
"cache-served" numbers.

## Operational findings

1. **Arch portability is a config problem, not (mostly) a kernel problem** — for
   CDNA2 + RDNA3 the only blockers were fp8 KV cache and AITER. Two env toggles
   unlock four more arches. The multi-arch image itself already carries the
   kernels (it loaded and ran on all of them).
2. **CDNA1 (MI100) and RDNA4 (gfx1201) are not one-toggle-away** — vLLM startup
   never completed. Likely `--async-scheduling` and/or the attention backend;
   worth a follow-up with `TORCH_SDPA` and async-scheduling off.
3. **NFS image staging is the throughput bottleneck for wide fan-out** — seven
   jobs each decompressing the same 11 GB image off shared `$HOME` serialized on
   NFS read bandwidth (~10 min to load). For big sweeps, pre-`docker load` the
   image on each node, or stage per-node.
4. **Single-GPU consumer/workstation nodes are scarce and often reserved** — most
   scheduling friction was finding a free GPU, not running the job. Pin to a node
   with a free `gres/gpu` in `AllocTRES` and confirm it's in `defq` (some are in
   `vm`).

## Repro

```
make cliff-submit AIC_CLIFF_GFX=<arch> AIC_CLIFF_NODE=<node> \
  HF_HOME=$HOME/aic-hf AIC_IMAGE_DIR=$HOME/aic-images \
  VLLM_MODEL=Qwen/Qwen2.5-3B-Instruct \
  AIC_CLIFF_ARMS=vram,nvme AIC_L2_BACKEND=local_disk \
  AIC_KV_CACHE_DTYPE=auto AIC_ROCM_USE_AITER=0 \          # <-- for non-CDNA3 arches
  AIC_LOCAL_CPU=true LMCACHE_MAX_LOCAL_CPU_SIZE=4 \
  VLM_GPU_MEMORY_UTILIZATION=0.85 \
  BENCH_ISL=8000 BENCH_SHARED_TOK=7000 BENCH_PREFIX_MODE=per_client \
  BENCH_CONCUR=1,8,32 BENCH_ITERS=1 AIC_CLIFF_TIME=01:00:00
```

Drop `AIC_KV_CACHE_DTYPE`/`AIC_ROCM_USE_AITER` (defaults fp8/aiter) on gfx942/gfx950.

Jobs: gfx942 67545621 · gfx90a 67545783 · gfx1100 67545784 · gfx1100p 67545785 ·
gfx1100w 67545786 · (failed: gfx908 67545782, gfx1201 67545787).
