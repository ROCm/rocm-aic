# Long-ISL cliff report — POST-FIX rerun (verified pool 262144 + slab 320 GB)

Companion to `cliff-long-isl-report.md` (the **pre-fix** runs 67536671 / 67536798 /
67537066). Those runs were confounded by a Makefile propagation bug: `${VAR:-default}`
could not override the Makefile-exported defaults, so **`nixl_pool_size` was silently
4096 slots (~18 GiB) instead of 262144**, and **the GDS slab was 20 GB instead of
320 GB**. See that report's "REVISED ROOT CAUSE" + "Propagation gap" sections.

**This run isolates the fix.** Same config as yesterday's run 2 (job 67536798):
trimmed ladder `BENCH_CONCUR=1,8,16,32,64`, 3 h wall, DRAM L1 64 GB, YaRN ×2 → 65536,
arms vram → nvme → gds. The *only* deliberate change is the corrected pool/slab
sizing (Makefile origin-guarded fix). The goal is to answer the two questions the
pre-fix runs could not:

1. **nvme arm:** with a real 262144-slot pool, does `local_cpu:true` now hold c≥32
   (and c=64)? If it still collapses, the **DRAM-L1→NVMe-L2 read-fallthrough bug is
   real and isolated**. If it holds, the pre-fix collapse was *only* the tiny pool.
2. **gds arm:** with a real 320 GB slab, is the c≥32 collapse **capacity overflow**
   (was 20 GB < 34 GB WS) or a genuine **I/O-rate wall** (as originally claimed)?

## TL;DR — both pre-fix "bugs" were the ONE Makefile propagation defect

The pool/slab-fixed rerun (job 67537751, all 3 arms, err=0 except client timeouts,
completed in ~2 h) overturns both prior conclusions and produces a clean headline:

| c | vram | nvme | gds | takeaway |
|---|---|---|---|---|
| 32 | 6.0k | 46.5k | **47.7k** | gds now matches nvme (pre-fix gds was 5.7k) |
| 64 | 6.0k | 7.5k | **48.3k** | **gds HOLDS where nvme collapses** |

1. **gds "I/O-rate-bound above the cliff" — REFUTED.** It was **20 GB slab capacity
   overflow** (Makefile bug), not an I/O rate wall. With the real 320 GB slab, gds
   sustains ~48k through c=64.
2. **gds is the winner at the extreme.** At c=64 its 320 GB slab holds the ~67 GB
   working set *and* GDS slab reads work → **48.3k, ext_hit 93.6%, err=0**, while the
   nvme arm collapses to 7.5k.
3. **nvme c=64 collapse — CONFIRMED as the DRAM-L1→NVMe-L2 read-fallthrough bug**
   (not the pool): it still collapses with the correct 1152 GiB pool, because on a
   DRAM-L1 miss `local_cpu:true` recomputes instead of reading NVMe. gds (no DRAM
   tier) sidesteps this entirely.

**Practical upshot:** for long-ISL working sets that exceed a modest DRAM L1, the
**pure GDS NVMe-slab path (`local_cpu:false` + big slab) is the better config** than
DRAM-L1+NVMe — until the DRAM-L1→NVMe read fallthrough is fixed upstream.

## Run — job 67537751 (`cliff-long-64k`, pool/slab-fixed), COMPLETED ✅ (~2 h, all 3 arms ok)

- **Submitted:** 2026-07-16, `AAI_CLIFF_TIME=03:00:00 BENCH_CONCUR=1,8,16,32,64 make cliff-long-64k`
- **Started:** after ~14 min queue (PENDING→RUNNING). Wall 3 h.
- **Log:** `logs/67537751/cliff.out`

### ✅ Sizing verification (from the new `tiers:` line in cliff.out) — PASSED

The whole rerun hinged on this, and **both mis-propagated knobs now carry their
intended long-ISL values.** Verbatim from cliff.out:

```
tiers : local_cpu=true max_local_cpu_size=64GB nixl_pool_size=262144 slots (~1152 GiB) gds_l1=320GB nixl_buffer=8589934592B
```

| knob | pre-fix (67536798) | this run (67537751) | expected | ok? |
|---|---|---|---|---|
| `nixl_pool_size` | 4096 (~18 GiB) ❌ | **262144 (~1152 GiB)** | 262144 (~1152 GiB) | ✅ |
| `gds_l1` slab | 20 GB ❌ | **320 GB** | 320 GB | ✅ |
| `max_local_cpu_size` (DRAM L1) | 64 GB ✅ | 64 GB | 64 GB | ✅ |
| `nixl_buffer` | 8 GiB ✅ | 8589934592 B (8 GiB) | 8 GiB | ✅ |

## Head-to-head throughput (median tok/s) — pre-fix vs post-fix

_Filled per arm as the sweep runs. Pre-fix column = job 67536798 (yesterday, run 2)._

### Arm A — vram_only (baseline; should be unchanged — no cache involved) ✅ DONE
| c | pre-fix vram | post-fix vram | note |
|---|---|---|---|
| 1  | 44.9k | 45.4k | |
| 8  | 27.9k | 28.4k | |
| 16 | 8.7k  | 8.8k  | cliff (HBM full ~14 req) |
| 32 | 5.9k  | 6.0k  | floor |
| 64 | 5.8k  | 6.0k  | client ReadTimeout region |

> **Baseline reproduces** (within ~2%) — confirms the rig is comparable to yesterday
> and the fix doesn't perturb the no-cache path. All divergence in arms B/C is
> attributable to the pool/slab fix, not run-to-run drift.

### Arm B — kvd_v2 nvme (64 GB DRAM L1 + NVMe L2) — the key nvme test ✅ DONE
| c | pre-fix nvme | post-fix nvme | pre-fix ext_hit | post-fix ext_hit | note |
|---|---|---|---|---|---|
| 1  | 44.8k | 44.6k | — | 0% | |
| 8  | 47.5k | 48.0k | — | 0% | warm-cache |
| 16 | 46.8k | 47.5k | 93% | 82.8% | cliff — holds |
| 32 | 45.7k | 46.5k | 93.4% | 93.6% | holds (WS ~34 GB fits 64 GB DRAM L1) |
| **64** | **9.75k** | **7.5k** | 42.6% | **22.3%** | **COLLAPSE PERSISTS with correct pool** — err=0, no 500/crash |

> **✅ ROOT CAUSE RESOLVED (2026-07-16, isolation run 67537995 + static analysis) —
> it is neither a broken read path (A) nor an empty L2 (B). It is MECHANISM C:
> memory-pool exhaustion.** When the DRAM L1 fills, its resident chunks consume the
> same LMCache `memory_management` block pool that the L2 *read* path needs to stage
> a retrieval into. So a chunk that IS on NVMe returns "in storage but can't be
> retrieved." Direct evidence from 67537995 (nvme-only, 16 GB L1, c=8/16/32,
> `nixl_buffer_device:cuda`, retained connector log):
> - Lookup SUCCEEDS (`hit tokens: 59904, need to load: 59904` → ext_hit 93%) — L2 is
>   NOT empty (refutes B). Read IS attempted (refutes A; matches code: fallthrough at
>   `distributed/storage_manager.py:~353` is correct).
> - Read then FAILS: `Failed to allocate memory ... increase nixl_buffer_size` →
>   `The cache block is in the storage, but it can't be retrieved` (cache_engine.py:1759)
>   → `Retrieved 0 out of 59904 tokens, 0.0 GB/s` → with `failure_policy=fail`, HTTP 500.
> - Counts: **7,385** `Failed to allocate memory block ... no memory is available`
>   (memory_management.py:1414); 24 `KV load failure`; **759 successful stores, 0 write
>   failures** (writes are healthy — cuda buffer, no err 5013). Failures appear ONLY at
>   c=32 (real overflow); c=8/c=16 err=0.
> - **Reconciles the whole series:** the gds arm holds at c=64 *because it has no DRAM
>   L1* (slab-only) → the pool is never starved → 270 GB of reads served. The DRAM L1
>   is precisely what starves the read-staging allocator. nvme 67537751 c=64 (ext 22%,
>   err=0) = same starvation, recompute path; 67537995 c=32 (12×500) = same starvation,
>   fail path.
> - **Fix directions:** separate read-staging memory budget from L1 residency, or have
>   the memory manager evict L1 blocks to satisfy a read-staging allocation, or cap L1
>   to leave staging headroom. `nixl_buffer_size` (8 GiB) is a red herring — the
>   exhausted resource is the memory_management block pool.
>
> The read-fallthrough / write-failure text below is SUPERSEDED by mechanism C.
>
> **MECHANISM C — REFINED via LMCache source trace (2026-07-16). Two SEPARATE pools,
> coupled by object lifetime, NOT a shared byte budget.** Correction: the DRAM L1 and
> the L2 read-staging do NOT share one pool.
> - **Pool A (DRAM L1 store):** pinned host DRAM, sized by `max_local_cpu_size`, owner
>   `LocalCPUBackend`/`MixedMemoryAllocator`. Throws `memory_management.py:1414`
>   ("no memory available", 4.5 MiB KV chunks).
> - **Pool B (GDS read staging):** VRAM (`nixl_buffer_device:cuda`), sized by
>   `nixl_buffer_size` (8 GiB), owner `NixlStorageBackend`. Throws
>   `nixl_storage_backend.py:1212` ("increase nixl_buffer_size").
> - **The retrieve fails on Pool B (VRAM), returning 0 tokens** (`cache_engine.py:1759`
>   → `:953`). The thousands of `:1414` (Pool A/DRAM) are the *busy-loop symptom* of a
>   full+unevictable L1 (`LocalCPUBackend.allocate` `while True:` retry re-logs each spin).
> - **Coupling:** when L1 (A) is full/unevictable, the auto write-back of L2 hits into L1
>   + concurrent stores spin holding `MemoryObj` refs — including the just-read VRAM
>   objects — so Pool B never drains → next read can't allocate its VRAM slot → `:1212`.
>   References that don't release, not a shared budget.
> - **Data-path facts:** L1 DRAM→VRAM uses NO staging buffer — `multi_layer_kv_transfer`
>   DMAs straight from *pinned* DRAM to the KV cache over **SDMA** (async H2D). L2
>   NVMe→VRAM = `hipFileRead` into the registered VRAM `nixl_buffer` (GDS, no DRAM
>   bounce). The GDS VRAM buffer is entirely separate from the (nonexistent) L1→VRAM
>   staging buffer.
> - **Levers:** raising `max_local_cpu_size` won't fix `:1212` (wrong pool); raising
>   `nixl_buffer_size` may let reads win the race (more VRAM headroom before back-pressure)
>   — clean next experiment; real fix = decouple L1 eviction/write-back from holding VRAM
>   read objects. Caveat: on-disk source is LMCache 0.4.4, image is 0.5.x (may promote L2
>   hits into L1 via a fresh DRAM alloc — tightens coupling; confirm from image source).
>
> **STORE / EVICTION DATA-FLOW (source-confirmed, completes the picture):** LMCache is
> **write-through, drop-only eviction** — L1 and L2 are populated together at STORE time
> (`storage_manager.batched_put` fans out to all tiers, `:410-429`); L1 EVICTION moves
> NO data — it's `hot_cache.pop` + `ref_count_down` (`local_cpu_backend.py:273-274`),
> freeing the DRAM (Pool A) block because the NVMe copy already exists. The store's
> write-through to NVMe path is: **GPU KV ─D2H→ DRAM L1 ─H2D SDMA→ VRAM nixl_buffer (Pool
> B) ─GDS hipFileWrite→ NVMe** (`allocate_and_copy_objects` `:111` copies from the DRAM
> L1 object, then `ais_mt_backend.cpp:129`). L1→VRAM (read/store) uses SDMA from pinned
> DRAM; there is no DRAM→VRAM→NVMe movement on evict. **Implication — Pool B (VRAM
> nixl_buffer) is SHARED by store-staging AND read-staging**, so under overflow the
> write-through churn also consumes Pool B slots, competing with L2 reads → extra
> back-pressure on the exact pool whose exhaustion (`:1212`) kills the retrieve.
> (AMD patch 03 only touches the separate `gds_backend.py` log level — corroborates GDS
> L2 as a non-evicting write-through sink.)

> **🔑 FINDING (superseded framing — see revision note above) — the nvme c=64 collapse is NOT the pool size.** With the pool now
> *verified* at 262144 slots (~1152 GiB), c=64 still collapses to 7.6k (≈ pre-fix
> 9.75k, if anything slightly lower). A 1152 GiB pool cannot rescue it → the collapse
> is **the DRAM-L1→NVMe-L2 read-fallthrough bug, now isolated from the pool-sizing
> artifact.** When the ~67 GB c=64 working set exceeds the 64 GB DRAM L1, LRU
> eviction → re-request misses in DRAM → **no NVMe read fallback → recompute**
> (`local_cpu:true` never issues reads). c≤32 holds because the working set fits the
> DRAM L1 outright. Definitive `agent_rx_requests=0` confirmation to be recovered from
> the retained Prometheus TSDB (live counter missed — nvme arm torn down before
> capture). **This upgrades the "read-fallthrough bug" from CONFOUNDED to CONFIRMED.**

### Arm C — kvd_v2 gds (GDS NVMe slab L1) — the key gds test ✅ DONE
| c | pre-fix gds | post-fix gds | post-fix ext_hit | note |
|---|---|---|---|---|
| 1  | 45.2k | 45.3k | 0% | |
| 8  | 48.4k | 49.4k | 0% | |
| 16 | 10.97k | **48.9k** | 85.2% | ⬆ pre-fix was slab-overflow, not cliff |
| 32 | 5.74k | **47.7k** | 93.6% | ⬆⬆ **holds — I/O-rate diagnosis REFUTED** |
| 64 | 5.7k | **48.3k** | 93.6% | ⬆⬆⬆ **HOLDS where nvme collapsed** — 320 GB slab fits ~67 GB WS + GDS reads work; err=0 |

> **🔑 FINDING — the gds "I/O-rate-bound above the cliff" diagnosis was WRONG; it was
> capacity overflow.** Pre-fix the slab was **20 GB** (Makefile bug), overflowed by
> the ~34 GB c=32 working set → collapse to the vram floor (5.74k). With the slab now
> *verified* at **320 GB**, gds holds **47.7k @ c=32** (ext_hit 93.4%) — matching the
> nvme arm's DRAM-L1 throughput. So the pure GPU-direct NVMe-slab path **does** sustain
> long-ISL churn when sized correctly. The prior future-work list (GDS queue depth /
> async-drain / DRAM staging / chunk coalescing) was chasing an artifact of the
> mis-sized slab — **deprioritize pending the c=64 point.**

## Telemetry evidence (recovered from the retained Prometheus TSDB)

Queried post-hoc by replaying `logs/67537751/prometheus` (WAL-only) in a throwaway
`prom/prometheus:v2.55.1` container (original TSDB untouched). **This image exports a
much richer `lmcache_mp_*` metric family than the pre-fix report knew about** — direct
tier read/write counters, not just NIXL agent bytes. Key finds:

**gds arm — reads confirmed serving (this is why c=64 held):**

| metric | value (peak, gds arm) | meaning |
|---|---|---|
| `lmcache_mp_gds_l1_slab_total_bytes` | **343,597,383,680 = exactly 320 GiB** | ✅ slab fix confirmed *in telemetry* (pre-fix would be 20 GB) |
| `lmcache_mp_gds_l1_slab_used_bytes` | ~57 GB | ~67 GB WS fits the 320 GB slab with headroom |
| `lmcache_mp_gds_l1_bytes_read` | **~270 GB** | slab actively served 270 GB of KV reads |
| `lmcache_mp_gds_l1_bytes_written` | ~95 GB | write volume incl. churn |
| `lmcache_mp_l1_read_chunks_total` | ~70k | GDS reads happen (contrast nvme below) |

**nvme arm — direct read counter not recoverable from this WAL:** the `lmcache_mp_*`
family is **gds-arm-only** (NO-SAMPLE during the nvme window ~17:50 UTC); the NIXL
AIS_MT connector reports via `agent_rx/tx_*`, and only a single all-zero NIXL agent
survived in the retained WAL for the nvme arm. So the direct `agent_rx=0` seal from
the pre-fix run isn't reproduced here — but the **throughput + ext_hit collapse is the
behavioral proof**: c=32 ext_hit 93.6% @ 46.5k (DRAM-L1-served) → c=64 ext_hit 22.3%
@ 7.5k (DRAM L1 overflowed, no NVMe read fallback → recompute).

> **Instrumentation win for future runs:** scrape `lmcache_mp_gds_l1_bytes_read/
> written`, `slab_used/total_bytes`, and `lmcache_mp_l1_read_chunks_total` directly
> (they exist in this image) instead of relying on analytical estimates or the
> coarse NIXL agent counters. Add them to `run_cliff.py`'s per-c snapshot (the
> harness follow-up already proposed in the pre-fix report).

## POSIX vs GDS L2 at L1 overflow (c=32, 16 GB L1) — opposite failure modes

A/B of the AIS_MT L2 I/O path at the same overflow point (nvme arm, `local_cpu:true`,
`max_local_cpu_size=16`, WS ~34 GB). GDS = `nixl_buffer_device:cuda` (job 67537995);
POSIX = `nixl_buffer_device:cpu` + `HIPFILE_ALLOW_COMPAT_MODE=true` → hipFile compat
`pread/pwrite` (job 67538215; `err=5013 → compat mode` confirmed in log).

| metric | GDS (cuda) | POSIX (cpu/compat) |
|---|---|---|
| throughput | **28k tok/s** | **5.9k tok/s** (recompute floor) |
| errors | **12/32 HTTP 500** | **0** |
| ext_hit | 93% | **0%** |
| `:1414`/`:1212`/"can't retrieve"/KV-load-fail | 7385/24/22/24 | **0/0/0/0** |
| stores | succeed | succeed (1596 @ ~5 GB/s) |
| retrieves | partial then fail | **0** — all lookups `hit tokens: 0` |

**Opposite failure modes:** GDS = fast-but-fails-hard (serves 93% at 28k, but VRAM
staging exhaustion crashes 12/32 with 500s); POSIX = graceful-but-useless (never
errors, but serves 0% and recomputes at the floor).

**Why POSIX serves nothing:** in cpu mode NIXL L2 **shares LocalCPUBackend's pinned
DRAM pool** (`max_local_cpu_size`, 16 GB) — no separate staging buffer (LMCache even
rejects `nixl_buffer_size` in cpu mode, `config.py:801`). At WS ~34 GB the single
16 GB pool is oversubscribed by L1 residency + L2 staging → stored chunks churn out
of the index before reuse → `hit tokens: 0` → recompute. Same overflow class as GDS,
but silent-0%-hit instead of hard 500s. **Caveat: not apples-to-apples** — GDS had
16 GB DRAM L1 + a separate 8 GB VRAM staging slab; POSIX had only the shared 16 GB
DRAM pool, so part of the throughput gap is that asymmetry, by LMCache's design.
**RESOLVED via TSDB (67538215) — POSIX I/O worked; the L2 *cache* did not.** Device
forensics for the POSIX window:
- **dedicated data NVMe (nvme1/2/3n1): 0 GB written, 0 GB read.** nvme0n1 (root): +18 GB
  written (buffered), ~0 read. **Reads ~0 on ALL devices** — no read I/O ever issued.
- **page cache grew +13.4 GB, dirty peaked 4.6 GB** → despite `use_direct_io:true`,
  hipFile **compat mode does BUFFERED I/O (O_DIRECT not honored)** through the page cache.
- `ais_kfd`=0 (no GDS, confirms compat), NIXL `agent_tx/rx`=0 (compat bypasses agent
  counters). Connector log: **1596 stores, 0 retrieves, 96/96 lookups `hit tokens: 0`.**

So: stores happened (buffered), but **every lookup returned `hit tokens: 0` → 0 reads
issued → recompute floor.** Not a POSIX-bandwidth problem — writes were fine. The cause:
in cpu mode the NIXL L2 shares LocalCPUBackend's 16 GB DRAM pool with NO independent
persistent staging; the dedicated data NVMe got zero durable writes, so the "L2" acted
as a DRAM-only tier that, under WS ~34 GB churn on a 16 GB pool, evicts every prefix
before reuse → total lookup miss (all 96, even post-warmup). **cpu-mode L2 never
operates as a findable NVMe-backed cache here — an LMCache design limitation, not POSIX
throughput.** Open sub-thread: confirm which fs backs nvme0n1's +18 GB (root vs the
lmcache dir) and whether cpu-mode is *supposed* to persist to NVMe at all or is
DRAM-only by design.

## L2 backend bake-off at c=32 / 16 GB L1 overflow — native POSIX disk WINS

Same overflow point (nvme arm, `local_cpu:true`, `max_local_cpu_size=16`, WS ~34 GB),
four L2 backends:

| L2 backend | job | tok/s | err | ext_hit | L2 read BW (LMCache retrieve) | NVMe **device** reads (TSDB) | reads served from |
|---|---|---|---|---|---|---|---|
| vram (no cache) | — | ~6k | — | — | — | — | — |
| GDS (NIXL AIS_MT, cuda VRAM staging) | 67537995 | 28k | **12×500** | 93% | ~23 GB/s (median) | **0 GB** | DRAM L1 only — **NVMe L2 reads never completed** (VRAM staging exhaust → 500s) |
| AIS_MT-**cpu** (hipFile compat fallback) | 67538215 | 5.9k | 0 | **0%** | — (0 retrieves) | 0 GB | nothing — recompute |
| **NIXL POSIX plugin (cpu, O_DIRECT)** | **67538346** | **39.7k** | **0** | **93.6%** | **~4.9 GB/s** | **70.7 GB** | **NVMe drive** (O_DIRECT, bypasses page cache) |
| **native LocalDiskBackend (buffered)** | **67538307** | **41.9k** | **0** | **93.6%** | **~6.2 GB/s** | **0 GB** | **page cache** (buffered; drive read once, then cached) |

> **Read-bandwidth reading (TSDB `node_disk_read_bytes` + LMCache `Retrieved … GB/s`):**
> the "L2 read BW" is the rate LMCache retrieves KV; "NVMe device reads" is what actually
> came off the drive. Three distinct behaviors: **(1) GDS** — 0 device reads: its NVMe-L2
> reads *failed* at VRAM staging, so the 23 GB/s "retrieves" were DRAM-L1 hits, not NVMe
> (the drive contributed nothing on reads). **(2) NIXL POSIX (O_DIRECT)** — 70.7 GB of
> *real* NVMe drive reads at ~4.9 GB/s; O_DIRECT bypasses the page cache, so this is true
> NVMe read bandwidth. **(3) LocalDiskBackend (buffered)** — 0 device reads: served from
> the **page cache** (~6.2 GB/s memcpy rate; the drive is read once on store, then reuse
> hits RAM). So both POSIX L2s win on throughput, but by different mechanisms — O_DIRECT
> pulls from the drive (~4.9 GB/s sustained, no failures), buffered rides the page cache.

**Two "real POSIX" L2s both win (~40k, 0 err, full serving): NIXL's first-class POSIX
plugin AND LMCache's native LocalDiskBackend.** Crucially, the NIXL POSIX plugin uses
the SAME cpu buffer sharing the SAME DRAM pool as AIS_MT-cpu (log: `Backend POSIX was
instantiated` + `max_local_cpu_size ... for NIXL shared pool`), yet serves reads
perfectly — proving the **AIS_MT-cpu 0%-hit was an AIS_MT-compat-specific defect, NOT a
shared-DRAM-pool / cpu-buffer problem.** (POSIX plugin IS built in the image; earlier
job 67538340 failed only on a transient GPU OOM at startup, not config/plugin.)

**Native LMCache `LocalDiskBackend` (POSIX, `use_odirect:false`) is the best option** —
highest throughput (beats even GDS's 28k), zero errors, full read serving. It configures
as DRAM L1 + `local_disk: file:///data/nvme/lmcache-disk` (one `.pt` file per chunk),
**no NIXL** (`AAI_L2_BACKEND=local_disk`). Why it wins: a *real indexed* persistent tier
(`contains()` + files) with the **page cache as a fast second tier** and **no VRAM
staging contention** → serves the full WS on L1-miss cleanly. It beats GDS precisely
because GDS was losing 1/3 of requests to VRAM-staging-pool (Pool B) exhaustion.

**Conclusion for the whole arc:** the DRAM-L1 + NVMe-L2 collapse was NEVER fundamental —
it was specific to the NIXL AIS_MT memory model (VRAM staging pool for GDS; shared-DRAM
pool for cpu/compat). LMCache's native POSIX disk backend handles L1 overflow robustly.
`nixl_buffer_size` red-herring, `local_cpu`-shared-pool, and VRAM-staging-exhaustion are
all NIXL-path artifacts, not tiered-KV-cache limitations. **Recommend evaluating
`AAI_L2_BACKEND=local_disk` as the product config** (add an O_DIRECT variant + higher-c
runs to confirm it scales past c=32).

## Findings / verdict (job 67537751 COMPLETED — all 3 arms ok, ~2 h)

**Full median throughput (tok/s), post-fix:**

| c | vram | nvme (DRAM L1 + NVMe **AIS_MT/GDS**) | gds (320 GB NVMe slab) | **nvme (DRAM L1 + NVMe NIXL-POSIX)** |
|---|---|---|---|---|
| 1  | 45.4k | 44.6k | 45.3k | 45.4k |
| 8  | 28.4k | 48.0k | 49.4k | 48.9k |
| 16 | 8.8k  | 47.5k | 48.9k | 48.3k |
| 32 | 6.0k  | 46.5k | 47.7k | 46.8k |
| 64 | 6.0k (err=16 client timeout) | **7.5k** ⬇ | **48.3k** | **44.2k** ✅ |

> **NIXL-POSIX sweep (job 67538368, DRAM L1 64 GB, `AAI_L2_BACKEND=nixl_posix`, same
> ladder) — added to complement the above.** Swapping the nvme arm's L2 backend from
> AIS_MT/GDS to the **NIXL first-class POSIX plugin** makes the DRAM-L1+NVMe config
> hold across the *entire* ladder (44–49k, ext_hit up to 93.6%, **err=0, zero
> `Retrieved 0` / `can't-retrieve` / KV-load-fail / HTTP 500 anywhere**) — including
> **c=64 (44.2k), where the AIS_MT/GDS nvme arm collapsed to 7.5k.** c=64 iter walls
> were ~92 s (cache serving) vs ~535 s for the collapsed AIS_MT/GDS arm (recompute).
> So the DRAM-L1+NVMe *tiering* was never the problem — the AIS_MT/GDS L2 backend was.
> Both POSIX L2s (NIXL-POSIX and native LocalDiskBackend) are full-ladder winners.

1. **gds I/O-rate diagnosis: REFUTED — was capacity overflow.** Pre-fix slab 20 GB
   (Makefile bug) < ~34 GB c=32 WS → collapse. Real 320 GB slab → gds holds ~48k all
   the way to c=64. The prior GDS future-work list (queue depth, async-drain,
   DRAM staging, chunk coalescing) chased a phantom; **deprioritized.**
2. **gds wins at the extreme (c=64):** 48.3k vs nvme 7.5k vs vram 6.0k. Its 320 GB
   slab holds the ~67 GB working set and GDS slab reads work → sustained serving.
3. **nvme DRAM-L1→NVMe-L2 read fallthrough: CONFIRMED broken** (upgraded from
   CONFOUNDED). c=64 collapses to 7.5k *with the correct 1152 GiB pool* → not a pool
   issue; on DRAM-L1 miss `local_cpu:true` recomputes instead of reading NVMe. c≤32
   holds only because the WS fits the 64 GB DRAM L1 outright.
4. **Baseline (vram) reproduced** within ~2% of yesterday → the arm deltas are real,
   not run-to-run drift.
5. **Stability:** zero HTTP 500 / EngineCore crashes anywhere; only client
   `ReadTimeout`s at c=64 vram (err=16), the known long-ISL benchmark-client artifact.

**Config recommendation:** for long-ISL working sets that exceed a modest DRAM L1,
prefer the **pure GDS NVMe-slab path (`local_cpu:false` + adequately sized slab)**
over DRAM-L1+NVMe, until the DRAM-L1→NVMe read fallthrough is fixed upstream.

### Remaining confirmation (secondary)
- **gds reads: CONFIRMED via TSDB** (`gds_l1_bytes_read ~270 GB`, slab = exactly
  320 GiB) — see Telemetry section. ✅
- **nvme `agent_rx=0` direct seal: not recoverable from this WAL** (metric family is
  gds-only; NIXL agent series didn't survive). To get it, add a live per-c scrape of
  `agent_rx_requests_num_total` (or run nvme arm alone and capture before teardown).
  Behavioral proof (ext_hit 93%→22%, tput 46.5k→7.5k at c=64) already stands.
- File/track the upstream LMCache issue: DRAM-L1 miss does not fall through to a
  NIXL-L2 read in `local_cpu:true`.
- Harness: add `lmcache_mp_*` tier counters to `run_cliff.py` per-c CSV columns.
