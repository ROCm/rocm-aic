# Overnight Cliff Run — Live Report

*Living document — updated through the night by the tracking job. Final analysis
(with the power-vs-cliff findings) is appended at the bottom once the run
completes.*

## Run under test

- **Job:** `67534362` on `ctr-cx66-mi300x-13` (started ~2026-07-14 ~23:30 EDT)
- **Arms:** `vram_only` + `kvd_v2 nvme` (the two proven arms; gds dropped)
- **Model / workload:** `Qwen/Qwen2.5-3B-Instruct`, ISL 20 000 (18 000 shared
  prefix + 2 000 unique), fp8 KV, `TRITON_ATTN`, prefill-dominated (`max_tokens=1`)
- **Sweep:** full default `1,2,4,8,16,32,48,64,80,100,128,160,200,250`, `iters=3`
- **VRAM budget:** `gpu-memory-utilization 0.12` → VRAM KV cap ≈ 929,984 tokens
  (~46 concurrent requests) — cliff onset ~46
- **NVMe tier:** dedicated spare `nvme1n1` (auto-detect, reused `/mnt/aai-lmcache`),
  `LMCACHE_NVME_POOL=65536`
- **CPU DRAM tier:** `local_cpu=true`, `max_local_cpu_size=16` GB (DRAM L1 →
  NVMe L2 tiered cache)
- **Wall-time:** 08:00:00

## Instrumentation

- `logs/67534362/cliff.out` — per-concurrency BW / p50 / p95 / L1_hit / ext_hit
- `logs/67534362/prometheus/` — TSDB. **GPU power IS here** via node-exporter's
  hwmon/DRM collectors: **`node_hwmon_power_watt`** (per-GPU socket power — the
  vLLM GPU is the chip that hits ~749 W under load, e.g. `0000:a5:00…`; idle GPUs
  ~135–148 W), **`node_drm_gpu_busy_percent`**, `node_drm_memory_vram_used_bytes`.
  This is the **primary** power source (timestamped alongside `vllm:num_requests_running`
  → clean power-vs-concurrency join). (The `rocm/device-metrics-exporter` `gpu_*`
  names are NOT present this run; irrelevant — hwmon covers it.)
- `logs/67534362/power-sample.log` — node-side amd-smi sampler (`epoch power gfx
  umc nrun` /5 s) — kept as a cross-check / backup for the TSDB power.
- `logs/67534362/container-aai-cliff-kvd-vllm.log` — LMCache retrieval/store GB/s

## What we're looking for
- Full cliff chart: `vram_only` vs `nvme` tok/s across the sweep; where `vram_only`
  falls off (~46 req) and NVMe holds.
- DRAM-vs-NVMe hit split (`L1_hit` = DRAM L1, `ext_hit` = NVMe L2) now that the
  16 GB DRAM tier is enabled.
- **Power consumption at different points on the cliff chart** (below vs above the
  cliff; vram_only recompute vs nvme cache-serve at matched concurrency).

---

## Progress log

_(entries appended chronologically as the run proceeds)_

- **setup** — job `67534362` running (arms vram,nvme; DRAM L1 16 GB; NVMe pool
  65536; tier on `nvme1n1`). Power sampler launched (idle baseline ~137 W).
  Tracking cron scheduled (~every 40 min).
- **2026-07-15 00:12 EDT** — RUNNING, **Arm A (vram_only)** at c=100. The cliff is
  forming across the sweep: c=48 median **23.4k tok/s** (L1_hit 37.7%→6.4% across
  iters — VRAM prefix cache thrashing right at the ~46-req cap), then c=64 **15.4k**,
  c=80 **16.1k**, c=100 ~15.6k → collapsed to the recompute floor (~15–16k). Below
  c≈46 vram_only holds; above it falls off, as expected. GPU power (TSDB
  `node_hwmon_power_watt`): **peak 751 W** on the vLLM GPU (idle GPUs ~137 W).
  Still to come: c=128/160/200/250 (vram), then all of Arm B (nvme). *Note: the
  node-side backup sampler stalled after a few samples; power analysis will use the
  TSDB (primary), which is healthy.*
- **2026-07-15 00:49 EDT** — RUNNING, **Arm A (vram_only)** finishing (c=200 done,
  c=250 next, then Arm B nvme). **Full vram_only cliff curve captured** — textbook:
  peaks **87.7k tok/s @ c=32** (below the ~46-req VRAM cap, prefix cache serving
  from VRAM), then **cliffs at c=48 → 23.4k**, collapsing to the **~15.3k recompute
  floor** for c=64…200 (a **~5.7× drop** from peak). Notable: below the cliff,
  vram_only (~88k) actually *beats* the nvme arm's usual steady ~63k — i.e. the
  NVMe tier is pure overhead below the cliff and a big win above it (crossover
  ≈ the 46-req cap; nvme sweep will confirm). GPU power (TSDB): **peak 748 W**.
  vram_only medians so far: c1 70.6k · c2 76.9k · c4 83.3k · c8 86.7k · c16 87.3k ·
  c32 **87.7k** · c48 23.4k · c64 15.4k · c80 16.1k · c100 15.4k · c128 15.4k ·
  c160 15.4k · c200 15.3k tok/s.
- **2026-07-15 01:12 EDT** — RUNNING, **Arm B (kvd_v2 nvme)** at c=64. vram_only
  arm finished (c=250 = 15.3k, floor confirmed). **The DRAM-tiered nvme arm behaves
  very differently from prior flat-~63k runs** — because the 16 GB DRAM L1 is now
  enabled: it peaks **~82k tok/s @ c=32** (DRAM L1 + VRAM absorb it), then **declines
  gracefully** as the working set overflows DRAM→NVMe: c=48 **71.1k**, c=64 **50.8k**
  (vs a hard vram_only cliff to 15k). So instead of one sharp cliff, we get a
  *softer, tiered roll-off*. Matched-concurrency NVMe win so far: c=48 **3.0×**
  (71.1k vs 23.4k), c=64 **3.3×** (50.8k vs 15.4k). nvme medians: c1 66.8k · c2
  71.3k · c4 77.9k · c8 80.3k · c16 81.8k · c32 **82.0k** · c48 71.1k · c64 50.8k.
  (c80…250 still to run.) DRAM tier config confirmed live: `local_cpu: true,
  max_local_cpu_size: 16, nixl_pool_size: 65536`. Power snapshot caught an idle
  gap (194 W, nrun=0 — between points); per-point loaded power comes in the final
  analysis.
- **2026-07-15 01:49 EDT** — job `67534362` **COMPLETED** (vram_only ok; nvme ok
  but **errored at c ≥ 100**). Full analysis written below (**with/without NVMe +
  cliff, DRAM-vs-NVMe, and the POWER + energy-efficiency findings** — headline:
  above the cliff the NVMe/DRAM tier is ~3× faster *and* ~3× more tokens/joule at
  the same ~750 W). Root cause of the c ≥ 100 errors: the 16 GB DRAM L1 pool + the
  512 MiB NIXL staging buffer exhaust under load → partial KV retrieval →
  `failure_policy=fail` → HTTP 500. **Fixed re-run launched: job `67534497`** —
  same config but **`AAI_NIXL_BUFFER_SIZE=8 GiB`** (up from 512 MiB) to survive
  high-concurrency staging; DRAM tier kept at 16 GB. Tracking now follows `67534497`
  (node sampler will relaunch once it's running — pam_slurm_adopt blocks ssh
  between jobs; TSDB `node_hwmon_power_watt` remains the primary power source).
- **2026-07-15 02:12 EDT** — RUNNING, re-run `67534497`, **Arm A (vram_only)** at
  c=32 (early). Deterministic vs the first run (c=16 = 88.5k). Peak GPU power **743
  W** (TSDB). Backup node sampler relaunched (ssh works again now the job is on the
  node). The 8 GiB NIXL buffer will show in the nvme-arm config once Arm B starts;
  the key check is whether nvme now completes c=100…250 **without 500s**.
- **2026-07-15 02:50 EDT** — RUNNING, re-run `67534497`, **Arm A (vram_only)** at
  c=160 (recompute floor confirmed: c=160 = 15.5k, err=0). Arm A nearly done
  (c=200/250 left), then Arm B (nvme) — the 8 GiB-buffer fix test. Peak GPU power
  **751 W**. No errors so far (vram arm). Next fires will show whether nvme clears
  c≥100 cleanly.
- **2026-07-15 03:12 EDT** — RUNNING, re-run `67534497`, **Arm A (vram_only)** on
  its last point **c=250** (322s/iter, floor 15.5k, err=0). Arm B (nvme, the fix
  test) starts next. Peak GPU power **751 W**. vram cliff fully reproduced. The
  next fire should catch Arm B and confirm the 8 GiB buffer clears c≥100.
- **2026-07-15 03:52 EDT** — re-run `67534497` **COMPLETED**. Full analysis written
  below ("RE-RUN job 67534497"). Result: 8 GiB buffer **fixed the clean range**
  (NVMe now 78–79k through c=64, **5.0× over vram at c=64**, **~6× better
  tokens/joule**), but errors persist at c≥80 (DRAM-pool + `failure_policy=fail`);
  recommended follow-up = `kv_load_failure_policy=recompute`. **Tracking stopped**
  (both overnight runs analyzed). Node power sampler will expire on its own.

---

## Final analysis — job 67534362 (vram_only + DRAM-16GB-L1 + NVMe-L2)

**Status:** completed. `vram_only` clean + full sweep. `nvme` (DRAM L1 16 GB → NVMe
L2) clean for **c ≤ 80**, then **errored at c ≥ 100** (root cause + fix below). A
**fixed re-run was launched** (see Progress log) — bigger NIXL staging buffer.

### 1. With vs without NVMe — the full cliff chart (median tok/s)
| c | vram_only | nvme (DRAM+NVMe) | NVMe × |
|---:|---:|---:|---:|
| 1 | 70,614 | 66,844 | 0.95 |
| 8 | 86,702 | 80,289 | 0.93 |
| 16 | 87,317 | 81,758 | 0.94 |
| 32 | **87,672** | **81,982** | 0.94 |
| 48 | 23,393 | 71,118 | **3.04** |
| 64 | 15,402 | 50,762 | **3.30** |
| 80 | 16,056 | 40,393 | **2.52** |
| 100 | 15,370 | 21,975 † | (1.43) |
| 128 | 15,378 | 11,945 † | — |
| 160 | 15,363 | 11,217 † | — |
| 200 | 15,333 | 10,064 † | — |
| 250 | 15,314 | 10,403 † | — |
† nvme c ≥ 100 is **invalid** (heavy 500s — see §3); ignore those rows.

- **vram_only:** peaks **87.7k @ c=32** (prefix cache fits in VRAM, below the
  ~46-req / 929,984-token cap), then **cliffs at c=48 → collapses to ~15.3k**
  recompute floor. Textbook cliff, ~5.7× drop.
- **nvme with DRAM L1:** below the cliff (c ≤ 32) it tracks vram_only (~82k, a
  ~6% offload tax). Above the cliff it **rolls off gently** (71k→51k→40k as the
  16 GB DRAM L1 overflows to NVMe) instead of collapsing — a *tiered* decline.
  Net win vs recompute: **3.0–3.5× at c=48–64**.
- **Crossover:** below ~46 concurrency the offload is pure overhead (vram_only
  slightly faster); above it, NVMe/DRAM is a large win. The tier earns its keep
  exactly past the VRAM cliff.

### 2. DRAM-vs-NVMe split (nvme arm, where valid)
Below overflow the DRAM L1 serves most hits (very high `L1_hit`, near-instant);
as concurrency climbs the split shifts to NVMe (`ext_hit`) and then to failures.
IO confirms it (§5): with the DRAM L1, NVMe **reads dropped 167 GB → 13 GB** vs
the pure-NVMe run — DRAM absorbed the reads; LMCache retrieval hit **up to 63.9
GB/s** (DRAM speed, ~14× the ~4.5 GB/s NVMe path).

### 3. Why nvme errored at c ≥ 100 (root cause)
The 16 GB DRAM L1 pool **and** the 512 MiB NIXL cuda staging buffer **exhaust
under high concurrency**, cascading to failure:
```
LMCache: Failed to allocate memory block ... no memory is available
LMCache: Failed to allocate memory, consider increasing the `nixl_buffer_size`
Retrieved 9728 out of 17920 required tokens ... less than expected!
scheduler: Failing request(s) due to KV load failure (failure_policy=fail) -> HTTP 500
```
At c ≥ 100 too many concurrent transfers contend for the small staging buffer →
partial KV retrieval → vLLM's `failure_policy=fail` turns it into a 500.
**Fix (re-run):** raise `AAI_NIXL_BUFFER_SIZE` 512 MiB → **8 GiB** (VRAM is
plentiful at 0.12 util). Keeps the reduced 16 GB DRAM tier + NVMe stress.

### 4. POWER across the cliff chart (user's key interest)
GPU power from the TSDB (`node_hwmon_power_watt`, our vLLM GPU = chip
`0000:a5:00…`; a neighbor's GPU on this shared node was excluded):

| state | GPU socket power |
|---|---|
| idle GPU | ~137 W |
| between points / drain | ~250 W |
| **vram_only, computing (any c)** | **~747–752 W** (≈ board TDP) |
| **nvme, computing** | **~712–725 W** |

- **The GPU pins near ~750 W whenever it computes** — power is ~flat-high across
  the whole loaded range, *not* proportional to offered concurrency (the cliff
  caps how many requests actually *run* at once, since KV space is the limiter, so
  `num_requests_running` stays low even at c=250).
- **The nvme arm draws ~25–35 W LESS** than vram_only at load — it does less
  compute per token (serves KV from cache instead of recomputing; GFX ~77% vs
  ~96% in the matched prior run).
- **Energy efficiency — the headline (W per tok/s, lower = better):**
  | c | vram_only | nvme | nvme advantage |
  |---:|---:|---:|---:|
  | 32 (below cliff) | 0.0086 | 0.0087 | ~equal |
  | 48 | 0.032 | 0.010 | **3.2× fewer J/token** |
  | 64 | 0.049 | 0.014 | **3.5×** |
  | 80 | 0.047 | 0.018 | **2.6×** |
  Above the cliff, **at essentially the same ~750 W draw, vram_only burns those
  watts recomputing for ~15k tok/s while NVMe delivers 40–71k tok/s** → the
  offload is **~3× more energy-efficient per token** (and slightly lower absolute
  power). Below the cliff the two are equal (both cache-serving). **Takeaway: the
  NVMe/DRAM tier doesn't just recover throughput past the cliff — it recovers it
  at the same power, i.e. ~3× better tokens-per-joule.**

### 5. IO + DRAM (nvme arm)
- `nvme1n1`: read **13 GB (6 MB/s)**, write **94 GB (40 MB/s)**, util **1.2%** —
  vs the pure-NVMe run's 167 GB read: the DRAM L1 absorbed the reads, while
  evictions pushed *more* writes to NVMe. NVMe is essentially idle (1.2%).
- Host DRAM: MemAvailable ~1506 GB (of 1.6 TB) — the 16 GB LMCache DRAM pool is
  tiny at the node level (its *internal* exhaustion caused §3, not host OOM).
- ais-snoop was armed on `kfd_ioctl_ais` this run too; AIS IO tracks the nvme arm.

### 6. Bottom line
The full cliff is captured cleanly: `vram_only` peaks 87.7k @ c=32 then cliffs to
~15.3k; the DRAM+NVMe tier rolls off gently and is **3–3.5× faster AND ~3× more
energy-efficient (tokens/joule) above the cliff at the same ~750 W**. The one
defect — nvme 500s at c ≥ 100 from staging-buffer/DRAM-pool exhaustion — is
understood and fixed in the re-run (8 GiB NIXL buffer). See the re-run's analysis
(appended by the tracker) for the clean high-concurrency DRAM-tier numbers.

---

## Final analysis — RE-RUN job 67534497 (8 GiB NIXL buffer; DRAM 16 GB + NVMe)

**Status:** completed (vram_only ok, kvd_nvme ok). The 8 GiB staging buffer
**substantially improved** the NVMe arm and **extended the clean range to c ≤ 64**,
but **errors still appear at c ≥ 80** (now bound by the 16 GB DRAM L1 pool +
`failure_policy=fail`, not the staging buffer). This is the definitive run.

### Full cliff chart (median tok/s)
| c | vram_only | nvme (DRAM+NVMe) | NVMe × | nvme errors |
|---:|---:|---:|---:|---:|
| 16 | 88,453 | 82,818 | 0.94 | 0 |
| 32 | **89,060** | 82,563 | 0.93 | 0 |
| 48 | 23,641 | **79,303** | **3.4** | 0 |
| 64 | 15,539 | **78,240** | **5.0** | 0 |
| 80 | 15,528 | 61,933 | 4.0 | 51 |
| 100 | 15,523 | 48,264 † | 3.1 | 115 |
| 128 | 15,519 | 34,729 † | 2.2 | 205 |
| 160 | 15,508 | 28,024 † | 1.8 | 303 |
| 200 | 15,501 | 20,684 † | 1.3 | 438 |
| 250 | 15,491 | 15,220 † | 1.0 | 604 |
† c ≥ 80 nvme has growing per-request errors (KV-load failures → 500); the *successful*
requests still get **ext_hit ≈ 89%**, but the medians are over fewer requests — treat
c ≥ 80 as degraded, not representative.

- **Buffer fix worked in the clean range:** c=64 jumped **50.8k → 78.2k** vs the
  first run, and c=48 71→79k. The DRAM+NVMe tier now holds **~78–79k through c=64**
  (well above pure-NVMe's flat ~63k) — the 16 GB DRAM L1 + big staging buffer beat
  pure NVMe when the hot set fits.
- **Cliff crossover:** vram_only cliffs at c=48 (23.6k) → floor 15.5k; NVMe stays
  78k → **5.0× at c=64**. Below c≤32 they're equal (~6% offload tax).
- **Remaining defect:** errors from c≥80 grow 51→604. Root cause is the 16 GB
  DRAM L1 pool exhausting + `kv_load_failure_policy=fail` turning KV-load
  contention into 500s. **Recommended next fix (needs a code/knob change, left for
  review): set the connector's `kv_load_failure_policy=recompute`** so overflow
  gracefully recomputes instead of erroring (and/or enlarge the DRAM L1). Not
  attempted overnight — it's a vLLM connector-config change worth doing deliberately.

### Power (the key ask)
GPU = chip `0000:c5:00…` this run. Per-arm socket power (`node_hwmon_power_watt`):

| | avg | peak |
|---|---:|---:|
| **vram_only** (recompute) | **726 W** | 752 W |
| **nvme** (DRAM+NVMe serve) | **616 W** | 749 W |
| idle GPU | ~137 W | — |

- The GPU pins **~750 W whenever it computes**; the **nvme arm averages ~110 W
  lower** than vram_only — it does less compute per token (cache serve vs recompute).
- **Energy efficiency (W per tok/s, lower = better) — the headline:**
  | c | vram_only | nvme | nvme advantage |
  |---:|---:|---:|---:|
  | 32 (below cliff) | 0.0082 | 0.0075 | ~equal |
  | 48 | 0.031 | 0.0078 | **~3.9× fewer J/token** |
  | 64 | 0.047 | 0.0079 | **~5.9×** |
  Above the cliff the NVMe/DRAM tier delivers **5× the throughput at lower power**
  → **up to ~6× better tokens-per-joule** (c=64). This is the definitive
  power result: offload past the cliff isn't just faster, it's dramatically more
  energy-efficient.

### IO
`nvme1n1` (nvme arm): read **53 MB/s**, write **75 MB/s**, util **3.1%** — reads
low (16 GB DRAM L1 absorbs them), writes higher (eviction spill), device ~idle.
vram arm: 0 (tier off). Root NVMe untouched. Host DRAM footprint negligible.

### Definitive takeaways
1. **The cliff is real and sharp:** vram_only 89k → 15.5k past ~46 concurrent
   requests (the 929,984-token VRAM KV cap).
2. **DRAM+NVMe offload recovers it at ~3–5× throughput and ~4–6× better
   tokens-per-joule**, holding ~78k through c=64 (better than pure-NVMe's 63k when
   the hot set fits the DRAM L1).
3. **Below the cliff, offload is a ~6% tax** — only turn it on when you'll exceed
   VRAM KV.
4. **Open item:** high-concurrency (c≥80) needs `kv_load_failure_policy=recompute`
   (or a larger DRAM L1 pool) to avoid KV-load-failure 500s — recommended follow-up.

---

## Progress log — run 67535846 (recompute fix)

- **2026-07-15 13:15 EDT** — **NEW run `67535846`** queued with the open-item fix
  applied: **`kv_load_failure_policy=recompute`** (new sbatch knob
  `AAI_KV_LOAD_FAILURE_POLICY`), plus NVMe pool **65536**, NIXL buffer **8 GiB**,
  DRAM L1 16 GB. Validated first in a c=128 short check: **`err=0`** (vs 205) and
  vLLM accepted the `recompute` enum. Goal for the full run: a **clean cliff curve
  through c=250** — cache-served where the 65536 pool holds it, graceful recompute
  for overflow, **no 500s**. Tracking resumed (cron `63621659`, ~40 min, dynamic
  node). Full analysis will append under "Final analysis - job 67535846".
- **2026-07-15 13:51 EDT** — RUNNING on `ctr-cx66-mi300x-31`, **Arm A (vram_only)**
  at c=128 (recompute floor ~16k, err=0). Arm B (nvme, the recompute-fix test) not
  started yet. Peak GPU power **750 W**. The decisive check — c≥80 err=0 + non-zero
  ext_hit — comes when Arm B runs (next fires).
- **2026-07-15 14:15 EDT** — RUNNING on `ctr-cx66-mi300x-31`, **Arm A (vram_only)**
  at c=200 (recompute floor ~15.9k, err=0), nearly done (c=250 left, then Arm B).
  Peak GPU power **749 W**. vram cliff reproduced; Arm B recompute-fix validation
  still pending.
- **2026-07-15 14:51 EDT** — `67535846` **COMPLETED**. Verdict: `recompute` gave the
  **best-yet clean low-c curve (c≤64, err=0, up to 84k)** but **crashed the vLLM
  EngineCore at c=80** (`AssertionError` → `EngineDeadError`) → c≥80 all failed
  instantly (BW=0). So recompute is **not** a safe high-c fix — it's *worse* than
  `fail` (hard crash vs degraded 500s). Full analysis below. **Tracking stopped.**

## Final analysis — job 67535846 (recompute fix: helps low-c, crashes high-c)

**Config:** vram+nvme, DRAM L1 16 GB, NVMe pool 65536, NIXL buffer 8 GiB,
`kv_load_failure_policy=recompute`. Node `ctr-cx66-mi300x-31`.

### With vs without NVMe (median tok/s)
| c | vram_only | nvme (DRAM+NVMe, recompute) | NVMe × |
|---:|---:|---:|---:|
| 16 | ~88k | 83,916 | 0.95 |
| 32 | ~89k | 84,014 | 0.94 |
| 48 | 23.6k | **81,401** | **3.4** |
| 64 | 15.5k | **78,505** | **5.1** |
| 80 | 15.5k | **0 (engine crash)** | — |
| 100–250 | 15.3–15.5k | **0 (engine dead)** | — |

- **Best clean low-c curve of all three runs:** the DRAM+NVMe tier holds **~78–84k
  through c=64** with **0 errors** (vs pure-NVMe's flat ~63k) — 3.4× at c=48, **5.1×
  at c=64** over the vram_only cliff.
- **Then it dies:** at **c=80** the `recompute` path hit a fatal
  `AssertionError` in vLLM's EngineCore (`vllm.v1.engine.exceptions.EngineDeadError`
  at 18:42:58) — the engine crashed, so every request from c=80 on failed instantly
  (wall ~0.2 s, ok=0, BW=0). This is a **regression** vs the `fail` policy, which at
  least degraded to partial-500s and kept serving ~60 req.

### Power
GPU chip `0000:e4:00…`. VRAM arm **avg 726 W / peak 752 W**; NVMe arm **peak 748 W**
(avg 573 W, pulled down by the post-crash idle). In the clean range the GPU is
~equally power-pinned, so with nvme delivering **5× the throughput at c=64** the
**tokens-per-joule advantage is again ~5×** (W/(tok/s): vram ~0.047 vs nvme ~0.009).
(nvme1n1 diskstats read 0 — on this node the spare auto-detect likely used a
different NVMe device name, and the arm crashed early; power/throughput are the
reliable signals here.)

### Cross-run verdict (67534362 → 67534497 → 67535846)
All three give the **same strong, clean curve to c≤64 (~78–84k, beating pure-NVMe
63k)** and **all break at c≥80** — the **16 GB DRAM L1 + high concurrency is the
fundamental limiter**, independent of staging-buffer size or failure policy:
| run | high-c failure mode |
|---|---|
| 67534362 (512 MiB buf, `fail`) | partial-500s from c≥100, growing |
| 67534497 (8 GiB buf, `fail`) | partial-500s from c≥80, growing to 604 |
| 67535846 (8 GiB buf, `recompute`) | **EngineCore crash at c=80** (worse) |

**Recommendations (deliberate, not overnight guesses):**
1. **Revert `kv_load_failure_policy` to `fail`** (or unset) — `recompute` crashes the
   engine; it's not viable as-is (looks like a vLLM recompute-path bug worth filing).
2. For a **clean full high-c curve today**, use **pure NVMe (no DRAM L1)** — run
   `67534106` did c=1→250 with 0 errors at flat ~63k. The DRAM L1 is what breaks high-c.
3. If DRAM L1 is desired at high concurrency, it needs a **much larger pool** (hold
   the top-of-sweep working set, ~83 GB) or an LMCache-side fix for the 16 GB-pool
   exhaustion — not a vLLM failure-policy toggle.
4. **Headline stands:** past the ~46-req cliff the offload tier is **3–5× faster at
   ~equal power (≈5× better tokens/joule)** — robustly demonstrated up to c≤64 here
   and across the full sweep in the pure-NVMe run.

---

## Planned next run — larger DRAM + NVMe pools (avoid the failure entirely)

**Goal:** a **clean full DRAM+NVMe curve through c=250** by sizing the cache tiers
so KV loads never fail (which also sidesteps the vLLM recompute crash — see
`vllm-recompute-bug.md`). Root cause of all high-c failures so far: the **16 GB
DRAM L1 pool exhausts** at c≥80.

**Sizing:** KV per token (Qwen2.5-3B, fp8) = 18.4 KB. Reusable prefix = 18k
tokens = **332 MB/request**; at c=250 the working set is ~**83 GB** (~92 GB incl.
the 2k unique tails). So:
- **DRAM L1 = 128 GB** (`LMCACHE_MAX_LOCAL_CPU_SIZE=128`) — holds the full c=250
  working set with headroom (node has ~1.6 TB DRAM). Expect it to *not* exhaust →
  no KV-load failures → no 500s / no recompute crash.
- **NVMe pool = 131072** (`LMCACHE_NVME_POOL=131072`, 2× prior) — overflow
  insurance on the dedicated spare (~3.5 TB).
- Keep 8 GiB NIXL staging buffer. **Leave `AAI_KV_LOAD_FAILURE_POLICY` unset**
  (default `fail`) — recompute is buggy with async scheduling; here we avoid
  failures rather than handle them.

**Command:**
```bash
VLLM_MODEL=Qwen/Qwen2.5-3B-Instruct VLM_GPU_MEMORY_UTILIZATION=0.12 \
AAI_CLIFF_ARMS=vram,nvme \
AAI_LOCAL_CPU=true LMCACHE_MAX_LOCAL_CPU_SIZE=128 \
LMCACHE_NVME_POOL=131072 AAI_NIXL_BUFFER_SIZE=8589934592 \
BENCH_ITERS=3 AAI_CLIFF_TIME=08:00:00 \
make -C aai-day-release cliff-submit
```

**Caveat:** a 128 GB DRAM L1 will absorb most/all of the working set, so `ext_hit`
(NVMe) will be low and NVMe barely exercised — this yields a **clean DRAM-tier full
curve** (high throughput held across the sweep), not an NVMe-stress test. Stressing
NVMe cleanly at high-c needs the recompute fix (or a fail-tolerant load path); that
config is blocked on the vLLM bug.

### Result — job 67536084 (COMPLETED, 1h44m, err=0 everywhere) ✅

**The big-pool hypothesis is confirmed: the c≥80 crash is gone, and the offload
tier holds ~79k tok/s flat all the way to c=250.** Both arms ran the full
1→250 sweep (3 iters each) with **zero errors** — no `AssertionError` /
`EngineDeadError` / HTTP 500 anywhere. `gds` arm skipped this run.

**Full curve (median tok/s):**

| c | vram_only | kvd_v2 nvme | ext_hit | speed-up |
|---|---|---|---|---|
| 1   | 71.6k | 67.7k | 0%    | 0.9× |
| 4   | 84.4k | 79.5k | 0%    | 0.9× |
| 8   | 89.0k | 82.1k | 0%    | 0.9× |
| 16  | 90.4k | 83.9k | 0%    | 0.9× |
| 32  | 90.5k | 84.4k | 0%    | 0.9× |
| 48  | 24.2k | 82.3k | 87%   | **3.4×** |
| 64  | 15.9k | 79.4k | 89%   | **5.0×** |
| **80**  | 15.9k | **79.1k** | 90% | **5.0×** |
| 100 | 15.9k | 78.9k | 90%   | **5.0×** |
| 128 | 15.9k | 79.2k | 90%   | **5.0×** |
| 160 | 15.9k | 78.8k | 90%   | **5.0×** |
| 200 | 15.9k | 78.7k | 90%   | **5.0×** |
| 250 | 15.8k | 78.5k | 90%   | **5.0×** |

- **c=80 is exactly where job 67535846 crashed** (async placeholder underflow, see
  `vllm-recompute-bug.md`). With the 128 GB DRAM L1 the pool never exhausts, no KV
  load fails, so neither `fail` nor `recompute` is ever invoked — the trigger is
  gone. The arm holds **~78–79k tok/s from c=48 through c=250 with no decay**.
- Below the cliff (c≤32) the offload arm is ~7–10% *slower* than pure VRAM (the
  connector's save/lookup overhead when everything already fits in VRAM) — expected
  and irrelevant; that regime isn't the point.
- At c≥48 the VRAM prefix cache overflows, `ext_hit` jumps to ~89–90%, and the
  external tier (DRAM L1 + NVMe) carries the load. The earlier "NVMe barely
  exercised" caveat was too pessimistic.

**Power / tokens-per-joule (from the run's retained Prometheus TSDB).** The active
GPU **pins ~750 W under load in *both* arms** (peak 751 W kvd / 752 W vram — the
750 W cap). Power is identical, so tokens-per-joule tracks throughput 1:1:

| regime | tok/s | GPU W | **tokens/joule** |
|---|---|---|---|
| vram_only, past cliff (c≥64) | 15.9k | ~750 | **~21** |
| kvd_v2 nvme, past cliff (c≥64) | ~79k | ~750 | **~105** |

→ **~5× better tokens/joule at the same power**, now demonstrated across the *entire*
high-concurrency range (c=64→250), not just c≤64. This is the headline efficiency
result: past the KV-cache cliff, DRAM+NVMe offload delivers ~5× the useful work per
watt as pure-VRAM, with zero failures.

**Bottom line:** sizing the DRAM L1 to hold the working set (128 GB here vs the
16 GB that failed) turns the offload arm from "crashes at c≥80" into a clean, flat,
~5×-throughput / ~5×-efficiency curve across the full sweep. Recompute stays
disabled (buggy under async scheduling) — and with correct pool sizing it's never
needed.
