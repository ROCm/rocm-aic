# Weekend autonomous cliff test log

Running unattended over the weekend: a scheduled task harvests finished jobs into
the **Results** section below and launches the next untried variant from the
**Matrix**, capped at ≤2 outstanding jobs at a time. All Qwen2.5-3B (full
attention; only 3B is staged in `$HOME` for gfx950), `per_client` prefix, 1 iter,
bounded ladders, no pointless vram recompute. Nothing here edits shared code —
jobs are env-configured; the sbatch/Makefile knobs already exist.

## Systems & staging
- **gfx942** (MI300X 192 GB): default `/scratch` paths, dedicated NVMe auto-provisioned (real L2 device).
- **gfx950** (MI350X 288 GB): `HF_HOME=$HOME/aic-hf AIC_IMAGE_DIR=$HOME/aic-images`, `/tmp` L2 (page-cache-assisted unless O_DIRECT).

## Submit template (direct sbatch so `--mem` can grow for big DRAM L1)
```
cd /home/AMD/stebates/Projects/rocm-aic
env VLLM_MODEL=Qwen/Qwen2.5-3B-Instruct \
    AIC_CLIFF_ARMS=<arms> AIC_L2_BACKEND=<local_disk|nixl|nixl_posix> \
    AIC_L2_ODIRECT=<true|false> AIC_NIXL_BUFFER_DEVICE=<cuda|cpu> \
    AIC_LOCAL_CPU=<true|false> LMCACHE_MAX_LOCAL_CPU_SIZE=<GB> \
    AIC_LOCAL_DISK_SIZE_GB=800 AIC_KV_CACHE_DTYPE=<fp8|auto> \
    VLM_GPU_MEMORY_UTILIZATION=<u> VLM_YARN_FACTOR=<2|4> VLM_MAX_MODEL_LEN=<65536|131072> \
    BENCH_ISL=<64000|128000> BENCH_SHARED_TOK=<isl-2000> BENCH_PREFIX_MODE=per_client \
    BENCH_CONCUR=<ladder> BENCH_ITERS=1 \
    [HF_HOME=$HOME/aic-hf AIC_IMAGE_DIR=$HOME/aic-images]  # gfx950 only \
  sbatch --parsable --constraint='<MARKHAM&GFX942&NVME|MARKHAM&GFX950>' \
    --mem=<128G|256G> --time=06:00:00 --job-name=aic-wknd-<tag> .slurm/run-cliff.sbatch
```
Prometheus harvest (per-tier NVMe/GPU): copy `logs/<jid>/prometheus`, run
`prom/prometheus:v2.55.1 --user $(id -u):$(id -g)` over it, query, tear down.

## Variant matrix (check off as harvested)

| # | sys | arms | L2 backend | O_DIRECT | L1 (GB) | kv dtype | ISL | util | ladder | status |
|---|-----|------|-----------|----------|---------|----------|-----|------|--------|--------|
| 0a | gfx942 | nvme | local_disk | false | 128 | fp8 | 128k | 0.5 | 1,16,64,128 | done 67546609 |
| 0b | gfx950 | nvme | local_disk | false | 128 | fp8 | 128k | 0.5 | 1,16,64,128 | done 67546610 |
| 1 | gfx942 | nvme | local_disk | **true** | 64 | fp8 | 128k | 0.3 | 1,16,64,128 | done 67546945 |
| 2 | gfx942 | nvme | nixl (AIS_MT/GDS, cuda) | true | 8 | fp8 | 128k | 0.3 | 1,16,64,128 | done 67547456 (collapsed c≥64, ext 0%) |
| 3 | gfx942 | nvme | nixl (AIS_MT compat, cpu) | true | 8 | fp8 | 128k | 0.3 | 1,16,64 (capped) | done 67547919 (collapsed c=64, ext 0%) |
| 4 | gfx942 | nvme | nixl_posix | true | 8 | fp8 | 128k | 0.3 | 1,16,64 (capped) | done 67548365 (WORKS, ext 56%) |
| 5 | gfx942 | gds | (hipFile slab L1) | n/a | slab 320 | fp8 | 128k | 0.3 | 1,16,64,128 | done 67548800 (serves, ext 64-73%) |
| 6 | gfx942 | nvme | local_disk | false | 64 | **auto(fp16)** | 128k | 0.3 | 1,16,64 (capped) | done 67549259 (floods c=16, faster low-c) |
| 7 | gfx942 | nvme | local_disk | false | 0 (LOCAL_CPU=false) | fp8 | 128k | 0.3 | 1,16,64 (capped) | done 67549688 (same 8,956 ceiling, no DRAM L1) |
| 8 | gfx942 | nvme | local_disk | false | 64 | fp8 | **64k** | 0.3 | 1,16,64,128 | done 67550118 (★ holds 34-70k, 0 err) |
| 9 | gfx942 | nvme | local_disk | true | 64 | fp8 | 128k | **0.8** | 1,16,64,128 | done 67550545 (big VRAM doesn't help 128k) |
| 10 | gfx950 | nvme | local_disk | **true** | 64 | fp8 | 128k | 0.3 | 1,16,64 | done 67553591 (147k@16, floods c=64) |
| 11 | gfx950 | nvme | nixl (AIS_MT, cuda) | true | 8 | fp8 | 128k | 0.3 | 1,16,64,128 | todo (GDS on MI350X) |
| 12 | gfx942 | vram,nvme | local_disk | false | 64 | fp8 | 128k | 0.3 | 1,8,16 | done 67551021 (★ 22× cliff recovery) |

Notes: NIXL/GDS arms need a dedicated NVMe (`&NVME` on gfx942); on gfx950 GDS may
fall back to POSIX-compat. `gds` arm uses `LMCACHE_L1_SIZE_GB` (hipFile slab), not
`LMCACHE_MAX_LOCAL_CPU_SIZE`. Skip a row if it needs a resource the target lacks;
note why.

## Results

_(appended as jobs complete — newest first)_

| harvested | job | sys | arms | backend | opts | c=1 | c=16 | c=64 | c=128 | ext-L2% | notes |
|-----------|-----|-----|------|---------|------|-----|------|------|-------|---------|-------|
| 2026-07-17 | 67546609 | gfx942 | nvme | local_disk | L1=128G util0.5 fp8 128k buffered | 76,409 | 43,756 | 8,956 (22 err) | 11,586 (62 err) | c128: 66.1 | **floods L2 at c=128**; c=64/128 stall on DRAM-L1→VRAM staging of 128k prefixes (600–729 s walls, client timeouts) |
| 2026-07-17 | 67546945 | gfx942 | nvme | local_disk | L1=64G util0.3 fp8 128k **O_DIRECT** | 76,414 | 45,365 | 8,956 (22 err) | 12,571 (63 err) | c64: 66.4, c128: 69.1 | floods L2 at **c=64** (smaller L1 lowers the knee); c64/128 stall identical to 0a → bottleneck is prefix rehydration, not the I/O mode |
| 2026-07-18 | 67546610 | gfx950 | nvme | local_disk | L1=128G util0.5 fp8 128k buffered | 119,684 | **155,393** | 14,716 | 22,260 (18 err) | c128: 74.8 | MI350X fastest at low c (155k @c=16); floods L2 at c=128; c=64 DRAM-L1 stall like gfx942 |
| 2026-07-18 | 67547456 | gfx942 | nvme | **nixl AIS_MT/GDS** (cuda buf) | L1=8G util0.3 fp8 128k O_DIRECT pool=524288 | 77,484 | 54,903 | **2,985 (50 err)** | **1,522 (114 err)** | 0.0 (L2 never served) | **NIXL AIS_MT/GDS collapses at c≥64** — ext-hit stays 0% (L2 reads never fulfilled), 50/114 client timeouts, c=128 wall 1178 s. Native POSIX local_disk (row 1: 12.5k, ext 69%) is far better on this path |
| 2026-07-18 | 67547919 | gfx942 | nvme | **nixl AIS_MT compat** (cpu buf) | L1=8G util0.3 fp8 128k pool=524288 | 79,142 | 52,390 | **2,985 (50 err)** | — (capped) | 0.0 (L2 never served) | **identical collapse to row 2** at c=64 (2,985 tok/s, ext 0%, 50 timeouts) → NIXL AIS_MT read-path failure is buffer-device-agnostic (cpu = cuda), not GDS-specific |
| 2026-07-18 | 67548365 | gfx942 | nvme | **nixl_posix** (NIXL POSIX plugin) | L1=8G util0.3 fp8 128k pool=524288 | 77,686 | 45,341 | **8,956 (22 err)** | — (capped) | c64: 56.3 | **WORKS** — L2 served (ext 56.3%), 8,956 tok/s = native local_disk (row 1). Unlike AIS_MT (rows 2/3, ext 0%) → the failure is **AIS_MT/hipFile-specific, not NIXL-general** |
| 2026-07-18 | 67548800 | gfx942 | **gds** | hipFile GDS slab-L1 320G | util0.3 fp8 128k | 76,776 | 57,906 | 8,956 (22 err) | 13,817 (62 err) | c64: 64.0, c128: 72.8 | **gds slab-L1 SERVES** (ext 64–73%, ~9–14k) — 320 GB slab absorbs the 128k working set; behaves like the POSIX backends, NOT the AIS_MT-as-L2 collapse (rows 2/3). Same c=64/128 rehydration stall (600 s, 22/62 timeouts) common to all backends |
| 2026-07-18 | 67549259 | gfx942 | nvme | local_disk | L1=64G util0.3 **fp16(auto)** 128k buffered | 97,158 | 75,437 | 10,662 (14 err) | — (capped) | c16: 98.4, c64: 65.1 | **fp16 vs fp8:** 2× KV footprint → floods L2 at **c=16** (ext 98% vs fp8 row 1 ext 0% at c=16). But c=1/c=16 throughput HIGHER than fp8 (97k/75k vs 76k/45k); c=64 similar (~10k). fp8 buys capacity (later flood), fp16 buys low-c speed |
| 2026-07-18 | 67549688 | gfx942 | nvme | local_disk | **L1=0 (LOCAL_CPU=false)** util0.3 fp8 128k | 76,579 | 54,451 | 8,956 (22 err) | — (capped) | c64: 52.8 | no DRAM L1 → VRAM overflows straight to NVMe (ext 52.8% @c64). c=64 = **8,956 tok/s, IDENTICAL to L1=64 (row 1) & L1=128 (0a)** → the DRAM L1 neither helps nor hurts; the ceiling is the VRAM-rehydration step, not the source tier |
| 2026-07-18 | 67550118 | gfx942 | nvme | local_disk | L1=64G util0.3 fp8 **64k** buffered | 77,082 | 89,728 | **70,583 (0 err)** | **34,470 (0 err)** | c64: 96.8, c128: 88.6 | **★ tier serving HOLDS at 64k** — 70k@c64 / 34k@c128, ext ~90%, **zero timeouts**. vs 128k (row 1) c=64 = 8,956 w/ 22 timeouts. The c≥64 collapse is **prefix-length-specific**: 64k rehydrates fine, 128k hits the wall (~8× slower, timeouts) |
| 2026-07-18 | 67550545 | gfx942 | nvme | local_disk | L1=64G **util0.8** fp8 128k O_DIRECT | 74,557 | 45,333 | 8,956 (22 err) | 12,148 (67 err) | 0.0 (VRAM holds, l1 66%) | **big VRAM doesn't help 128k high-c**: still 8,956/12k w/ timeouts, and L2 barely engages (ext 0%, l1 66% — the 148 GB VRAM budget keeps KV resident). Confirms the 128k c≥64 collapse is vLLM scheduling/rehydration at 128k, independent of VRAM size AND tier config |
| 2026-07-19 | 67551021 | gfx942 | **vram_only** (paired) | — | util0.3 fp8 128k | 79,249 | 6,094 | — | — | n/a | **cliffs at c=8** (79,249 → 3,584; c=16 6,094) — plain VRAM prefix cache can't hold 8×128k prefixes at util 0.3 |
| 2026-07-19 | 67551021 | gfx942 | **nvme** (paired) | local_disk | L1=64G util0.3 fp8 128k | 76,124 | 92,848 | — | — | 0.0 (DRAM L1 holds) | **★ tier holds flat 91–93k @c=8/16** vs the vram cliff to 3.6k = **~22× @c=8, ~15× @c=16** recovery. ext 0% (working set fits VRAM+64 GB L1; L2 not needed ≤c=16). The definitive cliff-vs-tier side-by-side |
| 2026-07-19 | 67553591 | gfx950 | nvme | local_disk | L1=64G util0.3 fp8 128k O_DIRECT | 106,211 | 147,165 | 13,434 (1 err) | — | c64: 60.4 | MI350X: fastest low-c (147k @c=16 > gfx942 87k); floods L2 at c=64 (ext 60.4%); 128k c=64 rehydration wall = 13,434 but only **1 timeout** (vs gfx942's 22) — faster silicon rides the wall better |

## Failures / observations

_(root causes, surprises, tuning notes)_

- **gfx950 rows (10/11) deferred:** at this pass no gfx950 GPU was schedulable in
  `defq` — every gfx950 node had its single GPU allocated except `bg-1e715-b02-1`,
  which is in a *different partition* (sbatch: "Requested nodes not in this
  partition"). Left as todo; a later fire will pick them up when a defq gfx950
  GPU frees. (We already have one gfx950 data point: row 0b flood, 67546610.)

- **67546609 (gfx942 flood, row 0a):** the big-VRAM(0.5)+big-DRAM-L1(128 GB) config
  *does* cascade — ext-L2 stays 0% through c=64 (working set fits VRAM+128 GB L1),
  then at c=128 (~294 GB WS > ~212 GB tier1+2) it floods NVMe (ext 66.1%). BUT
  c=64/128 collapse to 9–12k tok/s with 22/62 client timeouts (600–729 s walls):
  staging 128k-token prefixes DRAM-L1→VRAM at high concurrency is the bottleneck,
  not the NVMe. Takeaway: a huge DRAM L1 in front doesn't help throughput here —
  the L1→VRAM rehydration of very long prefixes is what stalls. Worth comparing
  vs O_DIRECT (row 1) and a smaller L1 (row 8/64k) and the NIXL/GDS staging path
  (rows 2–4).
  - TSDB (nvme window 20:09–21:05): dedicated **nvme3n1 WRITE 181.5 GB, READ 0.0 GB**;
    node RAM 1.6 TB, Cached ~1.3 TB. So the L2 populate physically hit the NVMe on
    writes, but every c=128 read was page-cache-served (0 device reads) — the same
    buffered-I/O masking seen earlier. Real device reads need O_DIRECT (row 1) or a
    working set > node RAM. The "ext-L2 66%" is a *logical* LMCache hit, not NVMe traffic.
- **67546945 (row 1, O_DIRECT) vs 67546609 (0a, buffered) — the read-path A/B:**
  TSDB (nvme window 21:58–22:53) shows nvme3n1 **READ 232.0 GB, WRITE 180.1 GB**
  under O_DIRECT — i.e. O_DIRECT genuinely bypassed the page cache and served the
  c=64/128 flood from the physical device (0a read 0 GB). Yet throughput was
  essentially identical to buffered (~9k/12.5k at c=64/128). So at 128k-prefix
  scale the limiter is **prefix rehydration serialization, not NVMe read
  bandwidth** — the device sustained ~230 GB of reads without being the
  bottleneck. O_DIRECT is the right knob for *measuring* real device traffic;
  it doesn't change the throughput story here.
- **67547456 (row 2, NIXL AIS_MT/GDS) — the L2 backend A/B:** at the same 128k
  scale, NIXL AIS_MT with a cuda staging buffer **collapses** at c≥64: ext-hit
  stays **0%** (the KV is written but reads are never fulfilled → the model
  recomputes, hence 50/114 timeouts and 1.5–3k tok/s). Same node/model/ISL, the
  native POSIX `local_disk` backend (row 1) held ~9–12k with ext 66–69%. This
  reproduces the earlier "NIXL L2 read path dead under L1 overflow" finding
  ([[nixl-l2-nvme-read-path-dead]]) even with the big pool (524288 slots) — the
  8 GB DRAM L1 starves NIXL's staging pool so L2 reads can't get a block.
  **Conclusion so far: native POSIX LocalDiskBackend is the robust L2; NIXL
  AIS_MT is not, at least under DRAM-L1 overflow.**
- **67547919 (row 3, NIXL AIS_MT compat/cpu buffer) confirms it's NIXL-general:**
  cpu-buffer compat mode collapses at c=64 *identically* to the cuda/GDS buffer
  (row 2) — 2,985 tok/s, ext 0%, 50 timeouts. So the AIS_MT read-path failure is
  **independent of the staging buffer device**; it's the AIS_MT backend itself
  (not GDS vs POSIX-compat). Remaining NIXL variant to try: `nixl_posix` (row 4,
  NIXL's first-class POSIX plugin) — the last chance for a NIXL path to match
  native local_disk.
- **67548365 (row 4, nixl_posix) — the failure is narrower than "NIXL":** the NIXL
  *first-class POSIX plugin* WORKS (ext 56.3%, 8,956 tok/s at c=64 = native
  local_disk), while NIXL *AIS_MT* (rows 2/3) dies (ext 0%). So the broken path is
  specifically **AIS_MT (hipFile)**, not NIXL as a whole. Two independent working
  L2 backends now (native `local_disk`, `nixl_posix`); one broken family (`nixl`
  AIS_MT, both cuda/GDS and cpu/compat). **L2 verdict: use local_disk or
  nixl_posix; avoid AIS_MT for the serve/read path under DRAM-L1 overflow.**
- **67548800 (row 5, gds slab-L1) — AIS_MT-as-slab-L1 works, only AIS_MT-as-L2 is broken:**
  the `gds` arm uses the same AIS_MT/hipFile backend but as a **direct GDS NVMe
  slab L1** (not an L2 tier behind a DRAM L1). It serves fine — ext 64–73%,
  9–14k tok/s at c=64/128, matching the POSIX backends. So the AIS_MT failure
  (rows 2/3) is specific to the **L2-read-under-DRAM-L1-overflow** path, not the
  hipFile/GDS engine itself. A 320 GB slab absorbed the full 128k working set
  (≤294 GB). Same c=64/128 rehydration stall as every other backend.
- **Cross-backend takeaway (128k, gfx942):** every backend that actually serves
  the tier (local_disk, nixl_posix, gds slab) lands at ~9k@c64 / ~12–14k@c128
  with 20–60 client timeouts — i.e. the ceiling is the **shared VRAM-rehydration
  path**, identical regardless of L2 backend. The only differentiator is
  pass/fail: AIS_MT-as-L2 fails to serve at all.
- **The DRAM L1 is irrelevant to the ceiling (row 7 clincher):** c=64 = **8,956
  tok/s** across L1=0 (row 7), L1=64 (row 1), L1=128 (0a), and across local_disk /
  nixl_posix / gds. So the stall is neither "DRAM-L1→VRAM" nor "NVMe→VRAM"
  specifically — it's the **VRAM-side rehydration of 128k prefixes at c≥64**, a
  fixed ceiling the tier config can't move. Practical implication: don't spend
  host RAM on a giant DRAM L1 for long-prefix serving; it won't raise throughput.
  fp8 vs fp16 (rows 1/6) *does* move the flood point (capacity) and low-c speed —
  that's the real lever, not tier sizing.
- **★ Prefix length is THE variable (row 8, 64k):** at 64k ISL the tier serves
  cleanly at scale — 70,583 tok/s @c=64 and 34,470 @c=128, ext ~90%, **zero
  timeouts** — vs 128k's 8,956 @c=64 with 22 timeouts (same config otherwise).
  So the tiered POSIX L2 is highly effective through at least 64k prefixes at
  c≤128; the 128k collapse is a **VRAM-rehydration-latency wall for very long
  prefixes** that trips the client timeout, not a tier/backend defect. Operating
  envelope so far: POSIX-tiered KV offload is a clear win up to ~64k prefixes;
  ≥128k needs either shorter effective prefixes, a higher client timeout, or a
  faster rehydration path. This reframes the whole sweep: backend choice only
  matters for pass/fail; **prefix length sets the throughput regime.**

- **2026-07-19 pass (no-op):** queue empty (0 outstanding); all completed jobs
  (r1–r9, r12) already harvested — no new results. Only remaining todo rows are
  10/11 (both gfx950). At this fire every gfx950 node had its single GPU
  allocated (running `gpu:1` jobs on bg-1w300-{g1-3,h3-2a,h3-3,k2-3a},
  asrock-1w300-e2-3) and the 8-GPU `smci355-...-n15-21` was `resv` — no
  schedulable gfx950 GPU. No gfx942 todo remains. Launched nothing; rows 10/11
  stay deferred until a defq gfx950 GPU frees.
