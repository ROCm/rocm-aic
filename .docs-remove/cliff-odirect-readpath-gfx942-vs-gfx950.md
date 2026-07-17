# POSIX-L2 read path: page-cache vs O_DIRECT (gfx950 vs gfx942)

First cliff runs on **new GFX arches** (MI350X / gfx950) plus an O_DIRECT A/B that
isolates the **real NVMe read path** of the LMCache native `LocalDiskBackend`
(POSIX) L2 tier. Two jobs:

| job | node | GFX | L2 backend | I/O mode | NVMe |
|---|---|---|---|---|---|
| 67545293 | bg-1w300-h3-3 | gfx950 (MI350X, 288 GB) | `local_disk` (POSIX) | buffered (page cache) | none → `/tmp` on root |
| 67545466 | ctr-cx66-mi300x-13 | gfx942 (MI300X, 192 GB) | `local_disk` (POSIX) | **O_DIRECT** | dedicated `/mnt/aic-lmcache` (`nvme2n1`) |

Common recipe: `cliff-long-128k` base — Qwen2.5-3B-Instruct (full-attention; the
LMCache connector needs it — gpt-oss is hybrid and can't unify KV specs under a
connector on vLLM 0.25), YaRN ×4 → 131072, ISL 128000 / shared prefix 126000,
`per_client`, arms **vram + nvme(POSIX)**, DRAM L1 = 8 GB, `BENCH_CONCUR=1,16`.
VRAM limited via `VLM_GPU_MEMORY_UTILIZATION` (0.06 on gfx950 / 0.08 on gfx942)
so the c=16 working set (~34 GB) overflows VRAM and the tier is exercised.

## TL;DR

- The cliff and its recovery reproduce on **both** arches. On gfx950 the vram arm
  collapses **122,696 → 6,029 tok/s** at c=16 (20×); on gfx942 (slower MI300X)
  **78,824 → 3,199 tok/s** (one request even hit the 600 s client timeout).
- **The gfx950 "recovery" was inflated by the page cache.** With buffered I/O on a
  540 GB-RAM node, the ~38 GB working set stayed resident (Cached grew +37.8 GB),
  so the nvme arm's c=16 reads were **RAM hits, not disk** — physical NVMe reads
  were **0.00 GB / 49 ops**. That bought 117,913 tok/s.
- **O_DIRECT on gfx942 shows the true read path.** Forcing the reads to the device
  (bypassing page cache) yields **36.22 GB of real NVMe reads (31,837 ops)** and a
  c=16 throughput of **35,840 tok/s** — still an **11.2× recovery** over the 3,199
  tok/s recompute cliff, but ~3.3× below the page-cache number. That gap is the
  cost of actually going to disk.
- Both are legitimate operating points: **buffered** = fast when the working set
  fits host RAM; **O_DIRECT** = the guaranteed floor when it doesn't.

## Throughput (tok/s)

| arm | gfx950 c=1 | gfx950 c=16 | gfx942 c=1 | gfx942 c=16 |
|---|---|---|---|---|
| vram_only | 122,696 | **6,029** | 78,824 | **3,199** (1 timeout) |
| nvme-posix | 119,312 | **117,913** | 77,357 | **35,840** |
| recovery | — | 19.6× | — | 11.2× |

gfx950 (MI350X) is ~1.5× faster than gfx942 (MI300X) at c=1 (122k vs 79k tok/s),
as expected for the newer part.

## vLLM compute metrics (from the Prometheus TSDB, per-arm window)

| metric | gfx950 vram | gfx950 nvme | gfx942 vram | gfx942 nvme (O_DIRECT) |
|---|---|---|---|---|
| prefill KV tokens computed | 1.92 M | 2.07 M ⁽*⁾ | 1.92 M | 1.96 M ⁽*⁾ |
| prompt tokens cached (compute skipped) | 11.6% | 44.1% | 11.6% | 52.3% |
| VRAM prefix-cache hit | 11.6% | 6.1% | 11.6% | 6.1% |
| **external (LMCache L2) prefix hit** | 0% | **45.6%** (1.64 M tok) | 0% | **49.5%** (1.96 M tok) |
| TTFT avg | 141.6 s | 100.8 s | 255.7 s | 148.3 s |
| GPU gfx activity (avg / max) | 95.9% / 100% | 89.2% / 100% | n/a ⁽†⁾ | n/a ⁽†⁾ |
| GPU package power (avg) | 905 W | 847 W | n/a ⁽†⁾ | n/a ⁽†⁾ |

⁽*⁾ The nvme arm's *computed*-token count is slightly **higher** than vram's: its
window includes the one-time warmup that **populates** the L2 (KV must be computed
once before it can be written). The win is not fewer FLOPs on first touch — it is
that the timed c=16 pass then **serves ~half the tokens from L2 instead of
recomputing**, which is what recovers the cliff.

⁽†⁾ GPU activity was captured on gfx950 (containerized `rocm/device-metrics-exporter`,
`gpu_gfx_activity`) but **not** on gfx942: that node had a pre-existing host GPU
exporter on :5000 which the sbatch reused, and it does not export `gpu_gfx_activity`.

## NVMe read data — the headline

Per-device disk I/O during the nvme arm (from `node_exporter` `node_disk_*`):

**gfx950 (buffered / page cache), device `nvme0n1` = `/tmp` on root:**

| | READ | WRITE |
|---|---|---|
| bytes | **0.00 GB** | 37.8 GB |
| ops | 49 | 48,531 |

Node memory context: MemTotal 540 GB; `Cached` 116.8 → 154.6 GB (**+37.8 GB**,
matching the write volume). The entire L2 working set was absorbed into the page
cache, so every "L2 read" was free.

**gfx942 (O_DIRECT), dedicated device `nvme2n1` = `/mnt/aic-lmcache`:**

| device | READ | WRITE | role |
|---|---|---|---|
| **nvme2n1** | **36.22 GB** (31,837 ops) | 37.63 GB (37,049 ops) | dedicated L2 |
| nvme0n1 | 0 | 0.18 GB | root (logs) |
| nvme1n1 / nvme3n1 | 0 | 0 | idle spares |

- **Real physical reads: 36.22 GB / 31,837 ops** — the KV working set genuinely
  read back from the NVMe platter (≈ the 37.63 GB written).
- **Read bandwidth: 1.78 GB/s** during the 20.3 s of active read-time. (The
  50.7 MB/s window-average is diluted by long compute-bound stretches with no I/O.)

## Method

The per-job TSDB (`logs/<job>/prometheus`) is stopped before head compaction, so
all samples live in the WAL. To query: copy the dir, drop `lock`/`queries.active`,
and run `prom/prometheus:v2.55.1` **as the current uid** (`--user $(id -u):$(id -g)`;
the default `nobody` can't write the mounted storage path) over it with a minimal
config — it replays the WAL (~0.4 s) and the data is queryable via the HTTP API.
Arms are separated by the vLLM scrape target: `instance="localhost:8001"` = vram,
`:8000` = nvme. Counter deltas = value(end) − value(start); gauges via
`avg()`/`avg_over_time` (collapse the per-`kfd_process_id` series churn, else the
sum inflates ~20×). NVMe device I/O = `node_disk_{read,written}_bytes_total`.

## Infra note (non-MI300X nodes)

The gfx950 (and other non-MI300X) Markham nodes have a **node-local `/scratch`**,
not the shared `/scratch` the MI300X rack sees, so the default HF cache and image
paths are empty there. Stage to the shared `$HOME` (NFS, `/home_mkm`) and pass
`HF_HOME=$HOME/aic-hf AIC_IMAGE_DIR=$HOME/aic-images` on the make command line
(command-line values aren't stripped by `_CLIFF_STRIP`, so they reach the job).
The default `make dist-build` image is already multi-arch (gfx90a;gfx942;gfx950;
gfx1100;gfx1101;gfx1150;gfx1151;gfx1200;gfx1201), so no rebuild is needed to run
on gfx950.
