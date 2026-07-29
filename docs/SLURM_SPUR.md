# Slurm / SPUR Build & Cliff Sweep

For local usage see [QUICK_START.md](QUICK_START.md). On a Slurm cluster the
`dist-*` / `cliff-*` targets build/distribute the image and run the full
three-arm sweep end-to-end (they wrap `.slurm/run-build-distribute.sh` and
`sbatch .slurm/run-cliff.sbatch`):

```bash
# Build the image (+ fabric exporters) on a CPU build node and save tarballs to
# the shared image dir; chain push + smoke-test:
make dist-build dist-push smoke-test AIC_PUSH_REF=<registry>/rocm-aic:latest

# Submit the full sweep (vram_only + kvd_v2 nvme + kvd_v2 gds) on a GPU+NVMe node.
# Output lands in logs/<job-id>/. Pin a node / narrow arms / override the sweep via env:
make cliff-submit AIC_CLIFF_NODE=<node-name>
make cliff-submit AIC_CLIFF_ARMS=nvme BENCH_CONCUR=1,8,64
make cliff-short          # 1-point smoke test of the whole flow
```

`smoke-test` validates the image on a GPU+NVMe node (GPU/arch, vLLM + LMCache
imports, `ais-check`, `nvme list`, the NIXL AIS_MT plugin). After those
checks it
also stands up the full exporter fleet + Prometheus (the same
`monitoring/monitoring-lib.sh` the cliff uses), scrapes briefly, health-checks
each `/metrics` endpoint, and leaves a TSDB under `logs/<job-id>/prometheus` to
sanity-check — all **informational** (only the in-image checks affect the exit
code). Tune with `AIC_SMOKE_EXPORTERS=0` (skip) and `AIC_SMOKE_SCRAPE_S=<secs>`
(default 45).

## Two-node LMCache P2P over RDMA

`.slurm/run-p2p.sbatch` (via `make p2p-submit`) stands up the cross-node
topology on **two** nodes of the same InfiniBand fabric and measures what
peer-to-peer KV sharing actually buys:

```bash
make dist-build                       # once -- both nodes load this image
make p2p-submit AIC_P2P_EXCLUSIVE=1   # whole nodes; results in logs/<job-id>/
```

One container per node runs an `lmcache server` plus a vLLM using
`LMCacheMPConnector`; node A additionally runs the `lmcache coordinator` both
servers register with. When node B misses locally it asks the coordinator,
learns node A holds the chunks, and pulls them through LMCache's NIXL transfer
channel. That channel's agent is created with the UCX backend, so the bulk KV
moves over IB verbs while the coordinator lookup and the metadata handshake stay
on TCP.

Per prompt length the job records three numbers: a cold full prefill, a peer
fetch, and a local cache hit (the floor). Each timed repeat uses a **distinct**
prompt, because once node B has fetched a prompt it is cached on B and a re-run
would silently measure a local hit. Around each peer fetch the job also diffs
every per-port counter on both nodes, which is the only way to tell a real RDMA
transfer from a silent UCX fallback to Ethernet: if the counters stay flat, the
bandwidth column is meaningless. `logs/<job-id>/` gets `summary.csv` (per-length
medians), `results.csv` (every measurement), the role logs, and the coordinator
and cache status captured at each phase.

### Metrics across the allocation

The job also stands up a **single** Prometheus (on node A) that scrapes **every**
allocated node, driven by `monitoring/docker-compose.monitoring.yml`. The
checked-in `monitoring/prometheus/prometheus.yml` is single-node -- all targets
are localhost -- so the job generates its own config into
`logs/<job-id>/prometheus/prometheus.yml` with `<node-ip>:<port>` targets and
`node` / `role` labels, and mounts it through the new `AIC_PROM_CONFIG`
indirection. Adding a third node to the allocation extends the target list
automatically; nothing in the config is hard-coded to two.

The exporter fleet runs on every node. The one that matters here is the
**rdma-exporter** (`exporters-fabric` profile, `:9879`), whose
`rdma_port_rcv_data_total` / `rdma_port_xmit_data_total` series are the
time-resolved version of the counter deltas in `results.csv` -- and
node-exporter's infiniband collector gives an independent second reading of the
same transfer. Build the fabric exporter images once with `make
dist-build-exporters`; the job loads them from the shared image dir alongside the
main image. Exporter startup uses `--no-build` so a node missing the image fails
that one service fast (reported by the health check) instead of silently starting
a multi-minute image build.

The TSDB is kept at `logs/<job-id>/tsdb` after teardown, and `results.csv`
carries `t_start` / `t_end` per measurement so a single phase can be sliced out
of it. Replay it long after the allocation is gone:

```bash
docker run --rm -p 9090:9090 -v <repo>/logs/<job-id>/tsdb:/prometheus \
  prom/prometheus:v2.55.1 --storage.tsdb.path=/prometheus
```

Useful knobs:

```bash
AIC_P2P_PREFLIGHT_ONLY=1 make p2p-submit   # validate a node pair, launch nothing
make p2p-submit AIC_P2P_NODES=<nodeA>,<nodeB>
make p2p-submit AIC_P2P_ISL_LIST=4096,16384 AIC_P2P_REPEATS=5
make p2p-submit AIC_P2P_GPU_EXPORTERS=1    # add amdgpu-exporter + hsa-snoop
make p2p-submit AIC_MONITORING=0           # skip Prometheus entirely
```

Two placement notes. The default constraint targets the ConnectX-7 MI300X pool;
the `rccl` partition groups the same machines but rejects submissions without a
matching Slurm account, so the job stays on `defq` with a feature constraint.
Those nodes are shared by default, and a co-tenant will both perturb the timings
and pollute the node-wide counter deltas -- use `AIC_P2P_EXCLUSIVE=1` for any
number you intend to report.

## SPUR cluster

For SPUR-specific setup (controller address, partition name, storage paths)
see [CLAUDE.md](../CLAUDE.md).

Set `AIC_SPUR_CLUSTER=1` to activate the SPUR-aware submission path:

```bash
make cliff-submit AIC_SPUR_CLUSTER=1 AIC_CLIFF_NODE=<node-name>
```
