# Overnight GFX-breadth cliff+tier sweep

Autonomous overnight run: the same cliff-vs-tier comparison (plain vLLM VRAM prefix
cache vs LMCache POSIX `local_disk` L2) across **as many GFX architectures as
possible**, ≤3 jobs at a time. Complements the weekend backend-depth sweep
(`cliff-weekend-autotests.md`, which stayed on gfx942/gfx950).

Fixed config (so arches are comparable): **Qwen2.5-3B** (full attention),
arms **vram + nvme (local_disk POSIX L2)**, ISL **16000** native (no YaRN, fits
the 32k window on every card), `per_client`, **util 0.6**, DRAM-L1 **16 GB**,
1 iter, ladder **1,8,32,64**. Non-CDNA3 arches use **fp16 + no-AITER**
(`AIC_KV_CACHE_DTYPE=auto AIC_ROCM_USE_AITER=0`) per [[cliff-image-arch-portability]];
staging on `$HOME` (`HF_HOME=$HOME/aic-hf AIC_IMAGE_DIR=$HOME/aic-images`).
The cliff falls at different concurrency per card (VRAM-dependent) — the tier
**recovery ratio** is the comparable metric.

## Submit template
```
env VLLM_MODEL=Qwen/Qwen2.5-3B-Instruct HF_HOME=$HOME/aic-hf AIC_IMAGE_DIR=$HOME/aic-images \
  AIC_CLIFF_ARMS=vram,nvme AIC_L2_BACKEND=local_disk AIC_LOCAL_CPU=true LMCACHE_MAX_LOCAL_CPU_SIZE=16 \
  AIC_KV_CACHE_DTYPE=auto AIC_ROCM_USE_AITER=0 \
  VLM_GPU_MEMORY_UTILIZATION=0.6 VLM_MAX_MODEL_LEN=32768 \
  BENCH_ISL=16000 BENCH_SHARED_TOK=14000 BENCH_PREFIX_MODE=per_client BENCH_CONCUR=1,8,32,64 BENCH_ITERS=1 \
  [AIC_ATTENTION_BACKEND=TORCH_SDPA]  # MI100/RDNA4 retry only \
  sbatch --parsable --constraint='MARKHAM&GFX<ARCH>' --nodelist=<node> --mem=64G --time=03:00:00 \
    --job-name=aic-gfx-<arch> .slurm/run-cliff.sbatch
```

## Matrix (check off as harvested)

| # | GFX | product | VRAM | kv dtype | extra | status |
|---|-----|---------|------|----------|-------|--------|
| A | gfx90a | MI210 | 64 GB | fp16 | — | done 67553853 (4.1×) |
| B | gfx1100 | RX 7900 | 24 GB | fp16 | — | done 67553854 (4.7×) |
| C | gfx1100w | W7900 | 48 GB | fp16 | — | done 67553855 (4.1×) |
| D | gfx1100p | RDNA3 Pro | ~48 GB | fp16 | — | done 67554020 (5.0×) |
| E | gfx1201 | RDNA4 | ~32 GB | fp16 | **TORCH_SDPA** (retry) | FAIL 67554021 (no GPU enumerated) |
| F | gfx908 | MI100 | 32 GB | fp16 | **TORCH_SDPA** (retry) | FAIL 67554022 (engine-core init) |

Big CDNA (gfx942/gfx950) intentionally omitted here — well covered by the weekend
sweep; 16k ISL wouldn't overflow their large VRAM anyway.

## Results

_(appended as jobs complete — newest first)_

| harvested | job | GFX | product | vram c=1 | vram cliff | nvme c=1 | nvme peak | recovery | notes |
|-----------|-----|-----|---------|----------|-----------|----------|-----------|----------|-------|
| 2026-07-19 | 67553853 | gfx90a | MI210 64G | 28,565 | 7,501 (c8) | 27,420 | 30,786 (c8) | **4.1×** | tier holds 23–31k, ext 84.6% @c64, 0 err; MI210 fastest of the three |
| 2026-07-19 | 67553854 | gfx1100 | RX7900 24G | 15,616 | 3,329 (c64) | 14,762 | 15,779 (c8) | **4.7×** | tier holds ~14k, ext 86% @c32/64, 0 err |
| 2026-07-19 | 67553855 | gfx1100w | W7900 48G | 14,788 | 3,564 (c8) | 13,712 | 14,736 (c32) | **4.1×** | tier holds ~12–15k, ext 86% @c64, 0 err |
| 2026-07-19 | 67554020 | gfx1100p | RDNA3 Pro 48G | 10,298 | 2,061 (c64) | 9,924 | 10,237 (c8) | **5.0×** | tier holds ~9–10k, ext 85–86% @c32/64, 0 err; slowest RDNA3 variant |

## Failures / observations

_(root causes, surprises, tuning notes)_
- **A/B/C (MI210 + RDNA3):** consistent **~4× tier recovery** at ISL 16k, and
  crucially the tier **holds at high concurrency with 0 timeouts** (ext 84–86% @c=64) —
  unlike the 128k weekend runs that stalled. Confirms cross-arch that short-ish
  prefixes (16k) let the POSIX L2 serve cleanly at scale. Peak throughput ranks
  MI210 (31k) > RX7900 (16k) ≈ W7900 (15k) — HBM datacenter part beats the
  RDNA3 workstation cards. All ran fp16+no-AITER on $HOME staging, no issues.
- **E gfx1201 (RDNA4) FAIL (67554021):** vLLM engine core dies with `RuntimeError:
  No CUDA GPUs are available` — the RDNA4 GPU isn't enumerable by torch/ROCm in
  this image (not an attention-backend issue; TORCH_SDPA didn't help). RDNA4 is
  not runnable with the current build.
- **F gfx908 (MI100, CDNA1) FAIL (67554022):** engine-core init failure again,
  even with fp16+no-AITER+TORCH_SDPA (root cause not captured in the tail). MI100
  remains unrunnable with this vLLM build — consistent with the earlier cross-GFX
  sweep. Both MI100 and RDNA4 need a different image/torch build, not a config knob.
- **Sweep complete.** Runnable arches (all ~4–5× tier recovery, tier holds at 16k
  with 0 timeouts): MI210 (gfx90a), RX7900 (gfx1100), W7900 (gfx1100w), RDNA3-Pro
  (gfx1100p) — plus the weekend's gfx942/gfx950. Unrunnable: MI100 (gfx908), RDNA4
  (gfx1201). Peak throughput ranking (nvme, this config): MI210 31k > RX7900 16k ≈
  W7900 15k > RDNA3-Pro 10k.

- **Loop stopped 2026-07-19:** matrix fully resolved (4 runnable arches done, MI100+RDNA4 failed); cron `1fdd6cdb` deleted to stop idle hourly wakeups. Re-create a loop or add matrix rows to resume.
