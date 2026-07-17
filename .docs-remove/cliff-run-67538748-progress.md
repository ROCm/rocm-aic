# Cliff run 67538748 — COMPLETE ✅ (~1h47m, all 3 arms ok, 0 HTTP 500s)

**Job:** `aic-cliff-long64k` (67538748) · node ctr-rack31-mi300x-2 · 5 h wall
**Config:** arms = **vram, nvme (GDS L2), gds** · ISL 64k / prefix 60k · YaRN ×2 → 65536
· DRAM L1 = 64 GB · NVMe pool 262144 · nvme L2 = NIXL **AIS_MT + cuda buffer (GDS)** · gds slab 320 GB
**Ladder:** BENCH_CONCUR = 1, 16, 32, 64 · iters = 2
**This is Job 1** of the overnight pair (Job 2 = nvme POSIX L2, run separately).

_Auto-updated every ~5 min while the job runs._

## Median throughput (tok/s)

| c | vram | nvme (GDS L2) | gds (slab) |
|---|---|---|---|
| 1  | 45.5k | 44.6k | 44.8k |
| 16 | 6.3k (cliff) | 47.0k (ext~90%) | 48.4k (ext~87%) |
| 32 | 6.1k (floor) | 46.6k (ext 93.5%) | 48.0k (ext 93.5%) |
| 64 | 6.0k (err=16 ReadTimeout) | **9.9k ⬇** (ext ~44%, collapse) | **48.7k ✅** (ext 93.5%) |

## Verdict

- **vram** — cliffs hard at c=16 (45.5k → 6.3k), floor thereafter (HBM full ~c=14 at 64k ISL).
- **nvme (DRAM L1 + NVMe GDS L2)** — carries the cliff (~47k through c=32, 5–7× vram) but
  **collapses at c=64** (9.9k, ext 93.5%→44%) once WS ~67 GB > 64 GB DRAM L1. Graceful
  (err=0, 0 hard read-fails) — degrades to recompute, not 500s.
- **gds (320 GB NVMe slab L1)** — **holds the whole ladder incl. c=64 (48.7k, ext 93.5%,
  err=0)**; the slab fits the ~67 GB WS, so no overflow. Best arm at the extreme.
- Stability: zero HTTP 500 / EngineCore crashes; only client ReadTimeouts on vram c=64.

Reproduces the earlier finding at **production L1 (64 GB)**: the DRAM-L1+NVMe collapse at
c=64 is a DRAM-L1-capacity effect, and the GDS slab (or a POSIX L2 — see Job 2) avoids it.
**Job 2 (nvme POSIX L2) pending** to slot in as the fourth comparison line.

## Progress log

| time | elapsed | arm / point | note |
|---|---|---|---|
| (start) | 3:00 | Arm A vram, c=1→16 | c=1 median 45.5k; vLLM up, err=0 |
| 22:19 | 8:46 | Arm A vram, c=32 | vram cliff confirmed: c=16 → 6.3k (HBM full); err=0 |
| 22:25 | 14:21 | Arm A vram, c=32 | c=32 ≈ 6.2k (floor); vram c=64 next (slow, ReadTimeout region); err=0 |
| 22:41 | 30:43 | Arm A vram done | vram c=64 ≈ 6.0k, err=8 (client ReadTimeout, expected); Arm B nvme (GDS L2) next |
| 22:58 | 46:55 | Arm B nvme(GDS-L2), c=32 | holds cliff: c=1 44.6k, c=16 47k (ext~90%) vs vram 6.3k; err=0. c=64 is the key test (prior runs collapsed there) |
| 23:19 | 1:08:45 | Arm B nvme(GDS-L2), c=64 | COLLAPSE reproduced: c=64 ≈ 10.6k, ext_hit 93.5%→48% (WS 67G > 64G L1); err=0, graceful (0 hard read-fails). Arm C gds slab next |
| 23:47 | 1:47 (done) | COMPLETE | gds slab HOLDS c=64 = 48.7k (ext 93.5%, err=0); nvme(GDS-L2) c=64 median 9.9k. All arms ok, 0 500s. |
