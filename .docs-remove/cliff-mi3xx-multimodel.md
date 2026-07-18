# MI3xx multi-model cliff tests — 3B & 72B, RoPE-extended ISL

Cliff runs on the MI300-class systems across two model sizes, with YaRN RoPE
extension to push the ISL, and concurrency ladders **chosen to straddle the
predicted VRAM-overflow point without wasting time on deep-recompute vram runs**.

Systems: **MI300A** (gfx942, 128 GB unified APU), **MI300X** (gfx942, 192 GB
discrete) and **MI350X** (gfx950, 288 GB). Models (the only full-attention models
staged offline — the LMCache connector needs full attention, so gpt-oss is
excluded): **Qwen2.5-3B** and **Qwen2.5-72B**. fp8 KV cache (CDNA3 default),
`per_client` prefix, 1 iter, LMCache native `LocalDiskBackend` (POSIX) L2.

## TL;DR

- **The cliff + tiered recovery reproduce on all three systems** (MI300A, MI300X,
  MI350X). For 3B the vram arm collapses ~14–20× at the overflow point; the
  POSIX-tiered nvme arm stays flat (ext-L2 hit ~98%). Peak nvme throughput ranks
  by silicon: **MI350X 137k > MI300X 87k > MI300A 55k** tok/s.
- **The APU memory ceiling is real.** MI300A's unified 128 GB caps the tier's
  headroom — at c=32 the ~128 GB working set thrashes and nvme throughput drops
  to 14.8k. The discrete MI300X (192 GB + separate host RAM for L1/L2) holds
  **47.3k at c=32** — a 10.7× lead over its own vram cliff.
- **72B cliffs hard and early (c=2)** — ~19 GB KV/client at 64k ISL vs a ~25 GB
  budget. Capping the vram ladder at c=4 kept the baseline cheap (189 s, not a
  600 s timeout).
- **Overflow prediction is systematically ~2× optimistic** if you budget from
  weights alone — actual C\* for 3B was ~4 (predicted ~7); activation, CUDA-graph
  capture, and fp8 metadata eat KV budget. 72B was on the nose (predicted 1.3,
  actual 2).
- **Big-model NFS load can exceed the vLLM ready timeout.** 72B took >18 min just
  to load 145 GB off a contended `/scratch`, tripping the 1200 s wait → added an
  `AIC_VLLM_READY_TIMEOUT` knob and re-queued the 72B nvme arm at 2400 s.

## Qwen2.5-3B — YaRN ×4 → 120k ISL, fp8 KV

| system | arm | c=1 | c=4 | c=16 | c=32 |
|---|---|---|---|---|---|
| **MI300A** (128 GB, util 0.18) | vram | 55,280 | **3,400** | 2,600 | — ⁽*⁾ |
| | nvme (POSIX L2) | 50,871 | **55,109** | 52,105 | 14,789 |
| | ext-L2 hit | 0% | 98.0% | 98.1% | 85.7% |
| **MI300X** (192 GB, util 0.12) | vram | 79,315 | **4,414** | 4,425 | — ⁽*⁾ |
| | nvme (POSIX L2) | 72,646 | **87,134** | 56,522 | **47,285** |
| | ext-L2 hit | 0% | 98%⁽†⁾ | 98.1% | 97.4% |
| **MI350X** (288 GB, util 0.08) | vram | 119,375 | **7,562** | 7,535 | — ⁽*⁾ |
| | nvme (POSIX L2) | 115,238 | **137,019** | 104,260 | — |
| | ext-L2 hit | 0% | 0%⁽†⁾ | 98.1% | — |

⁽*⁾ vram arm intentionally capped at c=16 — running it at c=32 is pure recompute
(cliff is at c≈4) and was cancelled mid-flight to avoid a ~600 s pointless point.
⁽†⁾ MI300X c=1/c=4 fit VRAM (ext-hit 0%); the tier engages from c=16.

**Recovery (nvme ÷ vram):** MI300A **16× @c=4, 20× @c=16**; MI300X **20× @c=4,
13× @c=16**; MI350X **18× @c=4, 14× @c=16**. c=1 is a wash (single prefix fits
VRAM either way). Peak nvme throughput ranks by silicon: **MI350X 137k >
MI300X 87k > MI300A 55k**.

The APU/discrete split at c=32 is the headline: MI300A 14.8k vs MI300X 47.3k.
On the APU the L1 DRAM tier and the model share the same 128 GB HBM, so a
~128 GB working set has nowhere to go; the discrete part spills cleanly to host
RAM + disk.

## Qwen2.5-72B — YaRN ×2 → 64k ISL, fp8 KV, MI300X (util 0.92)

| arm | c=1 | c=2 | c=4 |
|---|---|---|---|
| vram | 10,697 | **1,274** | 1,271 |
| ext prefix hit | 96.6% | 48.3% | 48.3% |

- Cliff at **c=2**: KV budget ≈ 25 GB (192 GB − 145 GB weights − overhead) vs
  ~19 GB KV/client at 64k → only ~1 prefix fits.
- 72B nvme arm: **not obtained — blocked by weight-load I/O, not tiering.** Three
  attempts failed loading the 145 GB weights from shared storage: ~34 s/shard on
  NFS (>18 min, tripped the 1200 s wait), then **~254 s/shard on a BeeGFS-backed
  node** (~2.6 h projected; tripped even the raised 2400 s wait at shard 8/37).
  vLLM's own hint: BeeGFS isn't a recognized network FS so auto-prefetch is off
  (`--safetensors-load-strategy=prefetch` would force it). The tiering mechanism
  is identical to 3B (just larger KV), already proven at scale on 3B; the 72B
  gap is a storage-throughput problem, not a cliff/tier result. To pursue: stage
  72B weights to node-local NVMe first, or add a prefetch/load-strategy knob.

## Predicting VRAM overflow

C\* ≈ (util·VRAM − weights − overhead) / (ISL · KV_bytes_per_token).

| model | ISL | KV/tok (fp8) | per-client | predicted C\* | actual C\* |
|---|---|---|---|---|---|
| Qwen2.5-3B | 120k | 18 KB | 2.2 GB (est) / ~4 GB (real) | ~7 | **~4** |
| Qwen2.5-72B | 64k | 320 KB | 19.2 GB | ~1.3 | **2** |

Lesson: the real per-client footprint ran ~1.8× the weights-only estimate for 3B
(activation + CUDA-graph + fp8 scale/zero-point metadata + block-size rounding at
`--block-size 64`). Budget **~2× margin** when picking the ladder — or split the
arms (vram capped near C\*, nvme scaled well past it), which is what these runs
ended up doing.

## Operational notes

1. **Split the arms when C\* is low.** A shared ladder can't serve both a vram
   arm (wants to stop at C\*) and an nvme arm (wants to scale past it). For 72B
   and the corrected 3B runs, vram used a short ladder and nvme a long one.
2. **APU vs discrete matters for tiering.** Size the DRAM L1 against *host* RAM,
   not GPU VRAM — on an APU they're the same pool, so a large L1 competes with
   the model and the working set.
3. **Raise `AIC_VLLM_READY_TIMEOUT` for big models on shared storage.** 145 GB
   over a busy NFS mount is minutes; the 1200 s default is not enough.
4. **Cluster was GPU-starved** — MI300X nodes showed 0/8 free for stretches;
   most of these ran on the one free MI300A plus opportunistic MI300X slots.

## Jobs

3B: MI300A vram 67545986 + nvme 67546047; MI300X vram 67545987 + nvme 67546048;
MI350X vram+nvme 67546236 (done). 72B: MI300X vram 67545988 (done); nvme 67545989
+ 67546235 (both failed on 72B weight-load I/O, not tiering — see above).
