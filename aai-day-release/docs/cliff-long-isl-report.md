# Long-ISL cliff report — YaRN context extension, all 3 arms

Companion to `overnight-cliff-report.md` (the 20k-ISL runs). This tracks the
**long input-sequence-length** sweeps that stress the LMCache DRAM→NVMe tiering
with a *modest, portable* DRAM L1 (64 GB) in front of a big NVMe L2 — the
representative product config — using the RoPE/YaRN scaling wired into
`run-cliff.sbatch` (`VLM_YARN_FACTOR`). See `vllm-recompute-bug.md` for the
async-scheduler crash that big pools sidestepped in the 20k run.

## Run 2 — job 67536798 (`cliff-long-64k`, trimmed), COMPLETED ✅

Resubmitted after run 1 hit long-ISL pacing/data-quality limits (see below), then
resubmitted again with a **3 h wall** (`AAI_CLIFF_TIME=03:00:00`) — the 16 h default
made the job un-schedulable (a 16 h reservation on the narrow MARKHAM&GFX942&NVME
node pool kept backfilling behind shorter jobs). 3 h scheduled instantly.

Changes vs run 1: **trimmed ladder `BENCH_CONCUR=1,8,16,32,64`** (stays at/below the
ReadTimeout onset so every point is clean). Arms run in the sbatch's hardcoded
order **vram → nvme → gds** (`AAI_CLIFF_ARMS` only selects, doesn't reorder — so gds
runs last, but the fast trimmed ladder lets all three finish in ~2.5 h < 3 h wall).
Sizing unchanged (DRAM L1 64 GB, NVMe pool 262144, YaRN ×2). Node ctr-cx66-mi300x-13.

<!-- RUN2 -->

#### Arm A — vram_only (baseline), in progress, err=0
| c | tok/s | note |
|---|---|---|
| 1 | 44.9k | |
| 8 | 27.9k | peak plateau |
| 16 | 8.7k | **cliff** (HBM full ~14 reqs) — matches run 1 (8.9k) |
| 32 | 5.9k | floor |
| 64 | ~5.8k | err=10 **ReadTimeout** (client, p95 580s) — not 500/crash |

#### Arm B — kvd_v2 nvme (64 GB DRAM L1 + NVMe L2), in progress, err=0
| c | nvme tok/s | vram | speed-up | note |
|---|---|---|---|---|
| 1 | 44.8k | 44.9k | 1.0× | |
| 8 | 47.5k | 27.9k | 1.7× | warm-cache (`--warmup-at-each-c`) |
| 16 | **46.8k** | 8.7k | **5.4×** | **cliff** — offload holds, no 500/crash |
| 32 | **45.7k** | 5.9k | **7.7×** | ext_hit **93.4%** (offload serving), wall 45s vs 336s |
| 64 | 9.75k | 5.8k | 1.7× | **DRAM L1 saturates** (67 GB WS > 64 GB) → NVMe spill, ext_hit 42.6%, **graceful — err=0, no 500/crash** |

#### Arm C — kvd_v2 gds (hipFile GDS NVMe slab L1, the FIX), in progress
**Bring-up succeeded** (was the fragile part): `lmcache-server (gds L1 slab) on
:6555` started, vLLM connected on :8000, sweep running. err=0, no
crash/500.

| c | gds tok/s | nvme | vram | note |
|---|---|---|---|---|
| 1 | 45.2k | 44.8k | 44.9k | |
| 8 | 48.4k | 47.5k | 27.9k | |
| 16 | 10.97k | 46.8k | 8.7k | **cliff — fix holds, err=0** (GDS NVMe slab L1_hit 40%) |
| 32 | 5.74k | 45.7k | 5.9k | ≈ vram floor — slab **cache not serving** (L1/ext hit 0%), err=0 |

**gds fix verdict:** bring-up + operation are **clean (err=0, no crash/500) across the
whole sweep** — the fix works. But the *performance* benefit is limited to the cliff
region (c=16): the no-DRAM NVMe slab can serve when churn is modest, but at c≥32 the
KV save/load rate outpaces the slab and hits drop to ~0 → falls back to recompute
(≈ vram floor). Contrast the nvme arm's **DRAM L1**, which held ~45k to c=32. So GDS
validates the GPU-direct path functionally, but a DRAM L1 is what sustains
throughput under heavy long-ISL churn. _(c=64 running.)_

### ⚠️ Open issue — GDS arm is I/O-rate-bound above the cliff (to explore)

> **⚠️ CONFOUNDED (2026-07-16) — re-evaluate before trusting this section.** The
> "capacity is fine: slab = 320 GB" premise is likely **false**: the same
> propagation bug that shrank the NVMe pool also hit `LMCACHE_L1_SIZE_GB`, so the
> gds slab was probably **20 GB, not 320 GB** (see the "REVISED ROOT CAUSE"
> section). A 20 GB slab is *overflowed* by the ~34 GB c=32 working set, so this
> collapse may be **capacity overflow**, not (only) an I/O-rate wall. Needs a
> re-run with the slab verified at 320 GB (Makefile fix landed) before the
> I/O-rate diagnosis below can be trusted.

**Diagnosis (pre-fix, now suspect — see warning above):** the gds throughput
collapse at c≥32 *appeared* to be a **NVMe-slab I/O-throughput limit, NOT a
capacity shortfall.** Evidence as recorded at the time:
- **Capacity assumed fine:** slab presumed = **320 GB** (`LMCACHE_L1_SIZE_GB`) —
  **but see warning: likely only 20 GB** — working set ~34 GB @ c=32 / ~67 GB @
  c=64. If the slab was 20 GB this is an *overflow*, not headroom.
- **Rate is the wall:** at 64k ISL each prefix is ~1.1 GiB KV; c=32 = ~34 GB/wave to
  write to the slab and read back over NVMe. NVMe write BW (few GB/s) can't persist
  that inside the harness's `5 s` async-save-drain window → prefix not yet on disk
  when reused → miss → recompute (L1/ext hit → 0% exactly at c=32).
- **Possible I/O serialization:** `ais_kfd_inflight` maxed at **1** (queue depth ~1)
  — GDS ops may not be parallelizing. *Caveat: ais/hsa-snoop PID targeting is
  unreliable on this shared node; treat as a hint.*
- **Proof by contrast:** the nvme arm sustained ~45k @ c=32 on identical churn — its
  **DRAM L1** (RAM BW ~100s of GB/s) absorbs the rate the NVMe slab cannot.

**To explore (future work):**
1. **GDS I/O parallelism / queue depth** — is the AIS_MT/hipFile path serializing
   (inflight=1)? Raise concurrent GDS ops; confirm with reliable per-PID snoop.
2. **Async-save drain / backpressure** — a longer drain (new knob) or true
   backpressure so reuse waits for persistence instead of recomputing.
3. **Small DRAM staging in front of the GDS slab** — even a modest DRAM tier may
   absorb the write burst (hybrid of the nvme + gds designs).
4. **Measure raw slab BW** in isolation (`tools/lmcache-io-tester`) vs the KV
   save/load rate demanded at each concurrency, to quantify the gap.
5. **Chunk size / write coalescing** — larger writes to the slab to raise effective
   NVMe BW.

### Final analysis — job 67536798 COMPLETED (2h11m, all 3 arms, 0 crashes/500s) ✅

Full trimmed sweep (c=1,8,16,32,64), **zero HTTP 500s and zero EngineCore
crashes** anywhere — only client-side `ReadTimeout`s at c=64 (requests >600 s at
64k ISL). RoPE/YaRN ×2→65536 confirmed working end-to-end.

**Throughput (median tok/s):**

| c | vram | nvme | gds | notes |
|---|---|---|---|---|
| 1  | 44.9k | 44.8k | 45.2k | |
| 8  | 27.9k | 47.5k | 48.4k | offload warm-cache wins even below cliff |
| 16 | 8.7k | **46.8k** | **10.97k** | **cliff** (HBM full ~14 req) — both offloads hold vs vram |
| 32 | 5.9k | **45.7k** | 5.74k | nvme (DRAM L1) sustains; gds I/O-bound → recompute |
| 64 | 5.8k | 9.75k | 5.7k | nvme DRAM L1 saturates (67 GB WS > 64 GB); all client-timeout-limited |

**Verdicts:**
- **gds fix: works** — clean bring-up + operation, err=0, no crash/500 across the
  sweep. Helps at the cliff (c=16: 10.97k vs vram 8.7k) but is **I/O-rate-bound at
  c≥32** (see Open issue above) → falls to the vram floor.
- **nvme (NIXL AIS_MT + 64 GB DRAM L1): the winner** — holds **~46k tok/s through
  c=32 (5–8× vram)** at 64k ISL, no failures. The DRAM L1 is what carries it.
- **Portable small-DRAM + big-NVMe config validated** — 64 GB DRAM L1 (vs the old
  16 GB that crashed) degrades **gracefully** (no 500/crash) when it saturates.

**KV-cache occupancy per tier (measured NVMe / analytical HBM+DRAM):**

| tier | at c=16 (cliff) | at c=32 | notes |
|---|---|---|---|
| **HBM** | ~16,320 MiB (full, 928,128 tok) | full | saturates at the cliff; analytical (scrape times out) |
| **DRAM L1** (nvme arm) | ~18,000 MiB / ~1.0 M tok (16 req × 1,125 MiB) | ~36,000 MiB / ~2.0 M tok | under 64 GB cap until ~c=57 |
| **NVMe L2** (nvme arm) | writes accumulating | — | NIXL `agent_tx_bytes_total` = **172 GB written** over the arm (incl. eviction churn); **`agent_rx_bytes_total` ≈ 0** |

> **Notable:** measured **NVMe read-back ≈ 0** (`agent_rx`). So above-cliff serving
> came from the **64 GB DRAM L1**, and when DRAM saturated (c=64) the arm
> **recomputed** rather than reading KV back from NVMe L2. See the dedicated bug
> section below.

### 🐞 BUG — DRAM-L1 miss does not fall through to an NVMe-L2 read (nvme arm c=64 collapse)

**Refined diagnosis (reconciles with "this used to work"):** the NVMe read path is
**not** categorically dead — it works in **NVMe-only** mode (`local_cpu:false`, the
recipe + the original pure-NVMe cliff run 67534106, which held 3–5× throughput
across the *full* sweep past the VRAM cliff, i.e. serving KV from NVMe). The bug is
specific to **`local_cpu:true`** (a DRAM L1 in front of NVMe, which we added only for
the recent DRAM-tier runs): **on a DRAM-L1 miss/eviction the connector does not fall
through to an NVMe-L2 read — it recomputes.** It stayed hidden because in every prior
`local_cpu:true` run the DRAM L1 was ≥ the working set, so NVMe reads were never
required. c=64 here (67 GB WS > 64 GB DRAM L1) is the first time the fallthrough was
exercised.

NIXL agent telemetry over the whole nvme arm (`local_cpu:true`):

| counter | value | meaning |
|---|---|---|
| `agent_tx_requests_num_total` | **2,572** | write requests issued |
| `agent_tx_bytes_total` | **172 GB** | KV written to NVMe |
| `agent_rx_requests_num_total` | **0** | **read requests issued — ZERO** |
| `agent_rx_bytes_total` | **0** | bytes read back |
| `agent_errors_total` | **0** | no I/O errors |

**Reads are not failing — they are never issued** (in `local_cpu:true` mode). The
connector writes KV to the NIXL NVMe L2 (write-through, 172 GB incl. eviction churn)
but on a **DRAM-L1 miss it does not query NVMe** — it recomputes. So with a DRAM L1
in front, the "DRAM L1 + NVMe L2 tiered cache" is effectively **DRAM-L1-only for
serving**. (In `local_cpu:false` NVMe-only mode the reads DO happen — see 67534106.)

**Consequence / the c=64 collapse:**
- c≤32: working set (≤34 GB) fits the 64 GB DRAM L1 → ext_hit ~93% (all from DRAM),
  ~45k tok/s.
- c=64: working set ~67 GB **> 64 GB DRAM L1** → LRU eviction → re-requests miss in
  DRAM → **no NVMe read fallback → recompute** → ext_hit 42%→0, throughput 9.75k.

So the earlier "graceful spill to NVMe" was wrong: there is **no read spill** — it's
DRAM-L1-serve-or-recompute. The big NVMe pool never helps *reads*.

**To debug (high priority):** why does `local_cpu:true` (DRAM L1 + NVMe L2) not fall
through to an NVMe read on a DRAM-L1 miss, when `local_cpu:false` (NVMe-only) reads
fine?
1. **Confirm first (cheap):** run nvme arm **NVMe-only** past the cliff —
   `AAI_CLIFF_ARMS=nvme AAI_LOCAL_CPU=false BENCH_CONCUR=1,16,32 make cliff-long-64k`.
   Expect `agent_rx_requests>0`, ext_hit high, throughput sustained → proves reads
   work sans DRAM L1 and isolates the bug to the `local_cpu:true` tiering.
2. LMCache lookup order with `local_cpu:true` — does a DRAM-L1 miss query the NIXL
   L2, or short-circuit to "miss → recompute"? (the fallthrough is the suspect.)
3. Does DRAM-L1 **eviction demote to NVMe** and keep the NVMe entry **findable** on
   re-lookup, or does eviction just drop the DRAM entry while the NVMe index isn't
   consulted?
4. Chunk-hash/index parity between the DRAM and NIXL tiers
   (`pre_caching_hash_algorithm: sha256_cbor`) — is the NIXL lookup index queried at
   all when DRAM L1 is enabled?

Likely an **upstream LMCache** tiering issue (DRAM-L1 + NIXL-L2 read fallthrough).

#### ⚠️ REVISED ROOT CAUSE — the NVMe pool was 4096 slots (~18 GiB), not 262144

The NVMe-only repro (job 67537066, `local_cpu:false`) revealed a simpler, dominant
bug — **`nixl_pool_size` was 4096 (the sbatch default), not the 262144 the
`cliff-long-64k` target intends.** 4096 slots × 4.5 MiB ≈ **18 GiB pool.**

**This alone explains the collapse — it's a pool-capacity overflow, not a read-path
or I/O-rate defect:**

| c | working set (chunks) | vs 4096-slot pool | result | ext_hit | tok/s |
|---|---|---|---|---|---|
| 16 | 16×234 = 3,744 | **fits** | reads serve | **93%** | 32.6k |
| 32 | 32×234 = 7,488 | **overflows** | LRU thrash → misses | **0%** | 5.6k |

**NVMe reads themselves WORK** — at c=16 the live node counter showed
`agent_rx_requests=28`, `agent_rx_bytes=25 GiB` read back, ext_hit 93%, ~33k tok/s.
The c=32 collapse is the 18 GiB pool overflowing (LMCache logs every c=32 request as
`hit tokens: 0, need to load: 0` — the lookup finds nothing because it was evicted;
`agent_rx` stays frozen at 28 → no reads even issued because there's nothing to
read). Confirmed: **both** long runs (67536798, 67537066) emitted
`nixl_pool_size: 4096` despite the target setting `LMCACHE_NVME_POOL=262144`.

**Propagation gap — ROOT CAUSE FOUND (2026-07-16), FIXED:** it was never a
make→SLURM `--export` issue — the wrong value was baked in *before* `sbatch` was
reached. `LMCACHE_NVME_POOL` is **both** Makefile-defaulted (`?= 4096`) **and**
`export`ed by the Makefile, so it arrives in the recipe shell already **set** to
`4096`. The recipe's `$${LMCACHE_NVME_POOL:-262144}` therefore never fires (`:-`
only substitutes when unset/empty) → it silently kept `4096`. `AAI_NIXL_BUFFER_SIZE`
propagated correctly precisely because it is **not** a Makefile variable (no `?=`,
not exported) → genuinely unset → its `:-` default fired. Reproduced in isolation:
the env reaching `sbatch` carried `LMCACHE_NVME_POOL=4096` + `AAI_NIXL_BUFFER_SIZE=
8589934592`, matching the emitted config exactly.

> **⚠️ Same bug hit `LMCACHE_L1_SIZE_GB` too (the GDS slab).** It is likewise
> `?= 20` + exported, and the recipe used `$${LMCACHE_L1_SIZE_GB:-320}` → the gds
> slab was almost certainly **20 GB, not 320 GB**. This *confounds the gds "Open
> issue" diagnosis below* (which assumed 320 GB and concluded "capacity is fine,
> I/O-rate-bound"): a 20 GB slab is **overflowed** by the ~34 GB c=32 working set,
> so the gds collapse may be capacity overflow, not (only) I/O rate. Must be
> re-evaluated with the slab verified at 320 GB.

**Fix (committed):** the `cliff-long-64k`/`128k` recipes now size these two knobs
via `$(if $(filter file,$(origin VAR)),<long-ISL value>,$(VAR))` — a Makefile
default resolves to the long-ISL value (262144 / 320, and 524288 / 640 for 128k),
while a real user override (command line / environment) is kept. Verified with a
shimmed `sbatch`: defaults → 262144/320, override → preserved. The new
`log "tiers …"` line in `run-cliff.sbatch` echoes the effective values so any
future mis-propagation is visible in `cliff.out` at a glance.

**Impact on the earlier `local_cpu:true` conclusion:** the c=64 collapse of
67536798 is now **confounded** — its pool was also only 18 GiB, so it can't hold the
67 GB working set regardless of the read path. The "DRAM-L1 miss doesn't fall
through to NVMe" hypothesis (agent_rx=0 there) is **not yet isolated** — it needs a
re-run with the pool *verified* at 262144.

**Next steps:**
1. **Fix observability:** echo the effective `nixl_pool_size` in `cliff.out` (this
   would have caught it immediately) — done in `run-cliff.sbatch`.
2. **Fix propagation:** ensure `LMCACHE_NVME_POOL` reaches the job (test submit),
   or raise the sbatch default.
3. **Re-run with pool verified at 262144**, NVMe-only first (should now hold c=32
   and beyond), then `local_cpu:true` to isolate whether the DRAM-L1→NVMe-L2
   fallthrough is a *separate* real bug or was just the tiny pool all along.

Workaround meanwhile: NVMe-only holds only while the working set fits the (currently
mis-sized) pool — so **verify the pool size before trusting a run.**

**Power / tokens-per-joule:** not reliably captured this run (amd GPU exporter not
scraped; node hwmon power empty in the TSDB). Since all arms are compute-bound and
pin the GPU (prior run measured ~750 W), **tokens/joule ≈ throughput ratio at equal
power** → at the c=16 cliff nvme is ~**5.4×** and gds ~**1.3×** the vram efficiency;
at c=32 nvme is ~**7.7×**. (Estimate — flagged for a power-instrumented rerun.)

NVMe offload written so far (NIXL `agent_tx_bytes_total`): **~20.0 GB (18.6 GiB ≈
1.09 M tok-equiv)**; read-back (`agent_rx_bytes`) begins as VRAM saturates past the
cliff. Above c=16 the VRAM prefix cache overflows (L1_hit → ~3%) and the external
DRAM+NVMe tier serves ~93% — that's the offload doing the work. _(c=64 running.)_

---

## Run 1 — job 67536671 (`cliff-long-64k`), ABORTED (cancelled at c=64)

Cancelled deliberately: at 64k ISL the full 14-point ladder × 3 arms was too slow
for the 16 h wall (vram tail alone ~5–6 h, gds arm last), and c≥64 vram points were
becoming **client `ReadTimeout`-dominated** (>9 min/request) — unreliable data, not
server failures. The **crash/500 watch stayed clean throughout** (0 AssertionError,
0 HTTP 500), and the **RoPE/YaRN wiring was confirmed working**. Useful baseline
captured before cancel:

**Launched:** 2026-07-15 19:35, node `ctr-cx66-mi300x-31`, image reloaded from
the fresh tarball (contains the **gds fix**).

### Config
| knob | value | note |
|---|---|---|
| model | Qwen2.5-3B-Instruct | fp8 KV, 18.4 KB/token |
| ISL / shared prefix | 64000 / 60000 | `per_client` (real cliff) |
| RoPE | YaRN ×2 → **65536** ctx | `VLM_MAX_MODEL_LEN=65536` |
| gpu-mem-util | 0.12 | VRAM KV budget ~929,984 tok → cliff ≈ **c=14** |
| DRAM L1 (nvme arm) | **64 GB** | modest/portable; spills to NVMe above ~c=60 |
| NVMe pool | 262144 slots (~1.15 TiB cap) | lazily sized; real disk = working set |
| GDS slab (gds arm) | 320 GB | on-NVMe L1 for the gds arm |
| NIXL staging | 8 GiB (cuda) | vs the old 512 MiB |
| arms | vram, nvme, gds | gds = the newly fixed path |
| iters | 2 | |
| working set @ c=250 | ~263 GB | 250 × 60k × 18.4 KB |

### What we're validating
1. **gds arm actually runs** (lmcache-server + GDS slab bring-up) and holds across
   the cliff — the fix under real spill/eviction.
2. **DRAM→NVMe spill works** with only 64 GB DRAM (the path that failed at c≥80
   in the old 16 GB run) — now with the gds fix + 8 GiB staging. Failure policy is
   `fail` (a spill miss → HTTP 500, not a crash), so 500s at high-c would flag a
   spill-throughput limit rather than the async crash.
3. **Long-context throughput/power/tokens-per-joule** vs the pure-VRAM cliff, at
   4× the KV-per-request of the 20k baseline.

### Progress

**RoPE/YaRN wiring confirmed live** (from the job's config echo):
`rope_scaling={"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":32768}`,
`max_model_len=65536`, `arms=vram,nvme,gds`. The new `VLM_YARN_FACTOR` knob works.

<!-- RESULTS -->

### KV-cache occupancy per tier (HBM / CPU-DRAM / NVMe)

Tracked in **tokens and MiB** from the run's retained Prometheus TSDB.
Conversion: fp8 KV = **18,432 B/token** → `MiB = tokens × 0.017578` (1 MiB ≈ 56.9 tok).

| tier | source (what's actually in the TSDB) | capacity |
|---|---|---|
| **HBM** (VRAM KV) | **analytical** — `vllm:kv_cache_usage_perc` scrape times out under long-ISL load (reads 0); HBM = min(active_reqs × 1,125 MiB, budget) | 928,128 tok = **16,320 MiB** (14502×64 blocks @ util 0.12) |
| **CPU DRAM L1** | **analytical** — no direct LMCache gauge in this build | 64 GB ≈ **3,640,889 tok** |
| **NVMe L2** | **measured** — NIXL `agent_tx_bytes_total` (bytes written) / `agent_rx_bytes_total` (read back), :19090 | 262,144 slots (~1.15 TiB cap; lazily sized) |

> **Metric-availability note:** the expected `rocm_aic_nixl_pool_*` / LMCache
> `local_cpu` gauges are **not** in this image's telemetry. What *is* exported is
> NIXL **agent** telemetry (`agent_tx_bytes_total` = KV bytes written to NVMe,
> `agent_rx_bytes_total` = read back, `agent_memory_registered_last_bytes` = staging)
> plus `ais_kfd_*` (AIS_MT GDS op counts). So **NVMe occupancy is measured** (write
> volume ≈ resident set for an append-mostly cache); **HBM and DRAM L1 are reported
> analytically** from the architectural caps + concurrency. `kv_cache_usage_perc`
> reading 0 is a scrape-timeout artifact under extreme load, not zero usage.

Per-request footprint at this ISL: 64,000 tok = **1,125 MiB**; the 60k shared prefix
= 1,055 MiB. So HBM (16.3 GiB) holds ~14 prefixes → cliff ≈ c=14; the 64 GB DRAM L1
holds ~57 → spill to NVMe begins ~c=57.

_Occupancy table (per tier, at the cliff / mid / top-of-sweep) filled during the
nvme + gds arms — TSDB-derived for this run; exact per-point CSV columns proposed
for future runs (harness change, see “Follow-up”)._

#### Arm A — vram_only (baseline), in progress, err=0
| c | median tok/s | note |
|---|---|---|
| 1 | 44.6k | |
| 2 | 28.9k | |
| 4 | 30.0k | |
| 8 | 30.1k | peak plateau |
| 16 | 8.9k | **cliff** — HBM KV budget (16.3 GiB ≈ 14 reqs) full |
| 32 | 5.9k | floor |
| 48 | 5.9k | floor (wall 524s/iter, p95 ~508s) |
| 64 | ~5.8k | **err=10 ReadTimeout** (client-side, p95 570s > client timeout) — NOT 500/crash |

> **Note:** at c≥64 the vram_only baseline starts hitting client `ReadTimeout`s
> (requests take >9 min at 64k ISL). These are benchmark client-timeout artifacts,
> not server failures — but they make the high-c vram points unreliable and slow.
> The gds/async-crash watch (HTTP 500, EngineCore AssertionError) remains clean.

The **cliff at c=16** matches the occupancy math (16,320 MiB HBM ÷ 1,125 MiB/req ≈
14 requests), landing far earlier than the 20k-ISL cliff (~c=48). `err=0`.
_(c=64+ running.)_

## Follow-up — exact per-point occupancy (harness change)

For per-concurrency exactness (not TSDB-derived), add tier-occupancy scraping to
`benchmarks/run_cliff.py` at each concurrency point (it already scrapes vLLM
`/metrics` for hit rates via `_snap_cache`): read `vllm:kv_cache_usage_perc`
(HBM), LMCache DRAM `/metrics` (:8080), and NIXL `rocm_aic_nixl_pool_bytes_total`
(:19090), and emit new CSV columns `hbm_kv_tokens,hbm_kv_mib,dram_kv_tokens,
dram_kv_mib,nvme_kv_tokens,nvme_kv_mib`. Needs an image rebuild to take effect.
